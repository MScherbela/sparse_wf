# %%
import os

from sparse_wf.model.sparse_fwd_lap import SparseMLP, Linear, NodeWithFwdLap, get_pair_indices, multiply_with_1el_fn, slogdet_with_fwd_lap
from sparse_wf.model.sparse_fwd_lap import get

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
os.environ["JAX_ENABLE_X64"] = "1"
import jax.numpy as jnp
import flax.linen as nn
from flax.struct import PyTreeNode
from sparse_wf.api import Parameters, Int
from sparse_wf.model.utils import cutoff_function
from sparse_wf.model.graph_utils import NO_NEIGHBOUR
import jax
import numpy as np
from typing import NamedTuple, Generic, TypeVar
from sparse_wf.jax_utils import fwd_lap
import functools
from folx.api import FwdLaplArray, FwdJacobian

T = TypeVar("T", bound=int | Int)


class StaticArgs(NamedTuple, Generic[T]):
    n_pairs_same: T
    n_pairs_diff: T
    n_triplets: T

    def to_static(self):
        return StaticArgs(
            int(jnp.max(self.n_pairs_same)), int(jnp.max(self.n_pairs_diff)), int(jnp.max(self.n_triplets))
        )


class ElecInit(nn.Module):
    feature_dim: int
    n_updates: int

    @nn.compact
    def __call__(self, r):
        n_out = 1 + 2 * self.n_updates
        h0 = nn.Dense(
            features=self.feature_dim * n_out,
        )(r)
        h0 = nn.silu(h0)
        h0 = jnp.split(h0, n_out, axis=-1)
        return h0[0], h0[1:1 + self.n_updates], h0[1 + self.n_updates:]


class PairWiseFunction(nn.Module):
    cutoff: float
    feature_dim: int
    n_out: int

    @nn.compact
    def __call__(self, r, r_nb):
        diff = r - r_nb
        dist = jnp.linalg.norm(diff, axis=-1)
        diff_features = jnp.concatenate([diff, dist[..., None]], axis=-1)

        beta = nn.silu(nn.Dense(16)(diff_features))
        beta = nn.silu(nn.Dense(8)(beta))
        beta = beta * cutoff_function(dist / self.cutoff)
        Gamma = nn.Dense(features=self.feature_dim * self.n_out, use_bias=False)(beta)

        edge_features = diff_features / dist[..., None] * jnp.log1p(dist[..., None])
        edge_features = nn.Dense(self.feature_dim * self.n_out)(edge_features)
        return jnp.split(Gamma, self.n_out, axis=-1), jnp.split(edge_features, self.n_out, axis=-1)


class EmbeddingParams(NamedTuple):
    elec_init: Parameters
    edge_same: Parameters
    edge_diff: Parameters
    updates: Parameters
    orbitals: Parameters


def contract(h_residual, h, Gamma, edges, idx_ctr, idx_nb):
    """
    Compute out[i] = h_residual[i] + sum_j Gamma[i,j] * silu(edges[i,j] + h[i] + h[j])
    """
    pair_msg = Gamma * nn.silu(edges + get(h, idx_ctr) + get(h, idx_nb))
    h_out = h_residual.at[idx_ctr].add(pair_msg)
    return h_out

def contract_with_fwd_lap(h_residual: NodeWithFwdLap, h: FwdLaplArray, Gamma: FwdLaplArray, edges: FwdLaplArray, idx_ctr, idx_nb, idx_jac, n_pairs_out):
    """
    Compute out[i] = h_residual[i] + sum_j Gamma[i,j] * silu(edges[i,j] + h[i] + h[j])
    """
    n_el, feature_dim = h.x.shape
    padding = jnp.zeros([n_el, 3, feature_dim], dtype=h.x.dtype)
    h_center = FwdLaplArray(h.x, FwdJacobian(jnp.concatenate([h.jacobian.data, padding], axis=1)), h.laplacian)
    h_neighb = FwdLaplArray(h.x, FwdJacobian(jnp.concatenate([padding, h.jacobian.data], axis=1)), h.laplacian)

    # Compute the message for each pair of nodes using folx
    def get_msg_pair(gamma, edge, i, j):
        return fwd_lap(lambda G, e, h_ct, h_nb: G * nn.silu(e + h_ct + h_nb))(gamma, edge, get(h_center, i), get(h_neighb, j))

    msg_pair = jax.vmap(get_msg_pair)(Gamma, edges, idx_ctr, idx_nb) # vmap over pairs

    # Aggregate the messages to each center node
    h_out = h_residual.x.at[idx_ctr].add(msg_pair.x)
    lap_out = h_residual.lap.at[idx_ctr].add(msg_pair.laplacian)
    J_out = h_residual.jac.at[idx_ctr].add(msg_pair.jacobian.data[:, :3]) # dmsg/dx_ctr
    J_out = J_out.at[idx_jac].add(msg_pair.jacobian.data[:, 3:]) # dmsg/dx_nb
    return NodeWithFwdLap(h_out, J_out, lap_out, h_residual.idx_ctr, h_residual.idx_dep)


def build_dense_jacobian(J, indices):
    n_el = np.max(indices) + 1
    feature_dim = J.shape[-1]
    J_dense = np.zeros((n_el, 3, n_el, feature_dim))  # [dep, xyz, i, feature_dim]
    i, j = indices
    J_dense[np.arange(n_el), :, np.arange(n_el), :] = J[:n_el, :, :]
    J_dense[j, :, i, :] = J[n_el:, :, :]
    J_dense = J_dense.reshape([n_el * 3, n_el, feature_dim])
    return J_dense


def envelope(r, n_orbitals):
    sigma = np.linspace(1.0, 2.0, n_orbitals)
    return jnp.exp(-jnp.linalg.norm(r) / sigma)


class Embedding(PyTreeNode):
    # Molecule
    n_el: int
    n_up: int

    # Hyperparams
    feature_dim: int
    n_layers: int
    cutoff: float

    # Submodules
    elec_init: ElecInit
    edge_same: PairWiseFunction
    edge_diff: PairWiseFunction
    updates: list[SparseMLP]
    orbitals: Linear

    @property
    def n_dn(self):
        return self.n_el - self.n_up

    @classmethod
    def create(cls, n_el, n_up, feature_dim, n_layers, cutoff):
        return cls(
            n_el,
            n_up,
            feature_dim,
            n_layers,
            cutoff,
            elec_init=ElecInit(feature_dim, n_layers),
            edge_same=PairWiseFunction(cutoff, feature_dim, n_layers),
            edge_diff=PairWiseFunction(cutoff, feature_dim, n_layers),
            updates=[SparseMLP([feature_dim]*2) for _ in range(n_layers)],
            orbitals=Linear(n_el),
        )

    def init(self, rng):
        rngs = jax.random.split(rng, 4 + self.n_layers)
        return EmbeddingParams(
            self.elec_init.init(rngs[0], np.zeros(3)),
            self.edge_same.init(rngs[1], np.zeros(3), np.zeros(3)),
            self.edge_diff.init(rngs[2], np.zeros(3), np.zeros(3)),
            [u.init(key, np.zeros(self.feature_dim)) for u, key in zip(self.updates, rngs[3:-1])],
            self.orbitals.init(rngs[-1], np.zeros(self.feature_dim)),
        )

    def get_static_args(self, r) -> StaticArgs[Int]:
        spin = np.concatenate([np.zeros(self.n_up, bool), np.ones(self.n_dn, bool)])
        dist = jnp.linalg.norm(r[:, None, :] - r[None, :, :], axis=-1)
        dist = dist.at[jnp.arange(n_el), jnp.arange(n_el)].set(jnp.inf)
        in_cutoff = dist < self.cutoff
        is_same_spin = spin[:, None] == spin[None, :]

        n_pairs_same = jnp.sum(in_cutoff & is_same_spin)
        n_pairs_diff = jnp.sum(in_cutoff & ~is_same_spin)
        n_triplets = jnp.sum(in_cutoff[:, None, :] & in_cutoff[None, :, :])
        return StaticArgs(n_pairs_same, n_pairs_diff, n_triplets)

    def apply(self, params, electrons, static: StaticArgs[int]):
        n_el = electrons.shape[-2]
        (idx_ct_same, idx_nb_same, _), (idx_ct_diff, idx_nb_diff, _), _ = get_pair_indices(
            electrons, self.n_up, self.cutoff, static.n_pairs_same, static.n_pairs_diff
        )

        h0, h_nb_same, h_nb_diff = jax.vmap(lambda r: self.elec_init.apply(params.elec_init, r))(electrons)
        Gamma_same, edge_same = jax.vmap(lambda i, j: self.edge_same.apply(params.edge_same, electrons[i], electrons[j]))(idx_ct_same, idx_nb_same)
        Gamma_diff, edge_diff = jax.vmap(lambda i, j: self.edge_diff.apply(params.edge_diff, electrons[i], electrons[j]))(idx_ct_diff, idx_nb_diff)

        h = h0
        for h_same, h_diff, g_same, g_diff, e_same, e_diff, update_module, update_params in zip(
            h_nb_same, h_nb_diff, Gamma_same, Gamma_diff, edge_same, edge_diff, self.updates, params.updates
        ):
            h = contract(h, h_same, g_same, e_same, idx_ct_same, idx_nb_same)
            h = contract(h, h_diff, g_diff, e_diff, idx_ct_diff, idx_nb_diff)
            h = update_module.apply(update_params, h)

        orbitals = self.orbitals.apply(params.orbitals, h)
        env = jax.vmap(envelope, in_axes=(0, None))(electrons, n_el)
        orbitals *= env
        sign, logdet = jnp.linalg.slogdet(orbitals)
        return logdet

    def apply_with_fwd_lap(self, params, electrons, static: StaticArgs[int]):
        n_el = r.shape[0]
        (
            (idx_ct_same, idx_nb_same, idx_jac_same),
            (idx_ct_diff, idx_nb_diff, idx_jac_diff),
            (idx_jac_ctr, idx_jac_dep),
        ) = get_pair_indices(electrons, self.n_up, self.cutoff, static.n_pairs_same, static.n_pairs_diff)
        n_pairs = static.n_pairs_same + static.n_pairs_diff

        h0_fn = functools.partial(self.elec_init.apply, params.elec_init)
        edge_fn_same = functools.partial(self.edge_same.apply, params.edge_same)
        edge_fn_diff = functools.partial(self.edge_diff.apply, params.edge_diff)

        h0, h_nb_same, h_nb_diff = jax.vmap(fwd_lap(h0_fn))(electrons)
        Gamma_same, edge_same = jax.vmap(fwd_lap(edge_fn_same))(electrons[idx_ct_same], electrons[idx_nb_same])
        Gamma_diff, edge_diff = jax.vmap(fwd_lap(edge_fn_diff))(electrons[idx_ct_diff], electrons[idx_nb_diff])
        jac_h0_padded = jnp.concatenate(
            [h0.jacobian.data.reshape([n_el, 3, self.feature_dim]), jnp.zeros([n_pairs, 3, self.feature_dim])]
        )
        h = NodeWithFwdLap(
            h0.x,
            jac_h0_padded,
            h0.laplacian,
            idx_jac_ctr,
            idx_jac_dep,
        )
        for h_same, h_diff, g_same, g_diff, e_same, e_diff, update_module, update_params in zip(
            h_nb_same, h_nb_diff, Gamma_same, Gamma_diff, edge_same, edge_diff, self.updates, params.updates
        ):
            h = contract_with_fwd_lap(h, h_same, g_same, e_same, idx_ct_same, idx_nb_same, idx_jac_same, n_pairs)
            h = contract_with_fwd_lap(h, h_diff, g_diff, e_diff, idx_ct_diff, idx_nb_diff, idx_jac_diff, n_pairs)
            h = update_module.apply_with_fwd_lap(update_params, h)

        orbitals = self.orbitals.apply_with_fwd_lap(params.orbitals, h)
        env = jax.vmap(fwd_lap(lambda r: envelope(r, n_el)))(electrons)
        orbitals = multiply_with_1el_fn(orbitals, env)
        triplet_indices = get_distinct_triplet_indices(electrons, self.cutoff, static.n_triplets)
        sign, logdet = slogdet_with_fwd_lap(orbitals, triplet_indices)
        return logdet


def is_close(a, b, msg=""):
    tol = 1e-6 if a.dtype == jnp.float32 else 1e-12
    try:
        np.testing.assert_allclose(a, b, err_msg=msg, rtol=tol, atol=tol)
    except AssertionError as e:
        print(e)


if __name__ == "__main__":
    n_el = 10
    n_up = n_el // 2
    rng_params, rng_r = jax.random.split(jax.random.PRNGKey(0))
    emb = Embedding.create(n_el, n_up, feature_dim=32, n_layers=1, cutoff=3.0)
    params = emb.init(rng_params)
    r = jax.random.normal(rng_r, [n_el, 3]) * 2.0
    static = emb.get_static_args(r).to_static()
    print(static)

    # h_folx = fwd_lap(emb.apply, argnums=1)(params, r, static)
    # h_sparse = emb.apply_with_fwd_lap(params, r, static)
    # is_close(h_folx.x, h_sparse.x, "h_folx.x != h_sparse.x")
    # is_close(h_folx.jacobian.data, h_sparse.dense_jac(), "h_folx.jac != h_sparse.jac")
    # is_close(h_folx.laplacian, h_sparse.lap, "h_folx.lap != h_sparse.lap")

    logdet_folx = fwd_lap(emb.apply, argnums=1)(params, r, static)
    logdet_sparse = emb.apply_with_fwd_lap(params, r, static)
    is_close(logdet_folx.x, logdet_sparse.x, "logdet_folx.x != logdet_sparse.x")
    is_close(logdet_folx.jacobian.data, logdet_sparse.jacobian.data, "logdet_folx.jac != logdet_sparse.jac")
    is_close(logdet_folx.laplacian, logdet_sparse.laplacian, "logdet_folx.lap != logdet_sparse.lap")

    # h_dense = emb.apply_dense(params, r, static)
    # h = emb.apply(params, r, static)
    # is_close(h_dense, h, "h_dense != h")
    # is_close(h_folx.x, h, "h_folx.x != h")

    # # h = emb.apply(params, r, static)
    print("Done")

