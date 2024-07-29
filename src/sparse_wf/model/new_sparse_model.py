# %%
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import jax.numpy as jnp
import flax.linen as nn
from flax.struct import PyTreeNode
from sparse_wf.api import Parameters, Int
from sparse_wf.model.utils import (
    cutoff_function,
    GatedLinearUnit,
    get_diff_features,
    PairwiseFilter,
    DynamicFilterParams,
)
from sparse_wf.model.graph_utils import NO_NEIGHBOUR
import jax
import numpy as np
from typing import NamedTuple, Generic, TypeVar, Callable
from sparse_wf.jax_utils import fwd_lap, nn_vmap
import functools
from folx.api import FwdLaplArray, FwdJacobian
import jax.tree_util as jtu
from jaxtyping import Float, Array

T = TypeVar("T", bound=int | Int)


class StaticArgs(NamedTuple, Generic[T]):
    n_pairs_same: T
    n_pairs_diff: T
    n_triplets: T

    def to_static(self):
        return StaticArgs(
            int(jnp.max(self.n_pairs_same)), int(jnp.max(self.n_pairs_diff)), int(jnp.max(self.n_triplets))
        )


class NodeWithFwdLap(NamedTuple):
    # The jacobian is layed out as follows:
    # The first n_el entries correspond to the jacobian of the node with itself, ie. idx_dep = idx_ctr
    # The next n_pairs entries correspond to the jacobian of the node with its neighbours, i.e. idx_dep != idx_ctr
    x: jax.Array  # [n_el x feature_dim]
    jac: jax.Array  # [n_el + n_pairs x 3 x feature_dim]
    lap: jax.Array  # [n_el x feature_dim]
    idx_ctr: jax.Array  # [n_el + n_pairs]
    idx_dep: jax.Array  # [n_el + n_pairs]

    def get_jac_sqr(self):
        jac_sqr = jnp.zeros_like(self.x)
        # .at[idx_ctr].add => sum over dependency index; jnp.sum => sum over xyz
        jac_sqr = jac_sqr.at[self.idx_ctr].add(jnp.sum(self.jac**2, axis=1))
        return jac_sqr

    def dense_jac(self):
        n_el = np.max(self.idx_ctr) + 1
        feature_dim = self.jac.shape[-1]
        J_dense = np.zeros((n_el, 3, n_el, feature_dim))  # [dep, xyz, i, feature_dim]
        J_dense[self.idx_dep, :, self.idx_ctr, :] = self.jac
        J_dense = J_dense.reshape([n_el * 3, n_el, feature_dim])
        return J_dense

    def __add__(self, other):
        return NodeWithFwdLap(self.x + other.x, self.jac + other.jac, self.lap + other.lap, self.idx_ctr, self.idx_dep)


class Linear(PyTreeNode):
    features: int
    use_bias: bool = True

    def apply(self, params, x):
        y = x @ params["kernel"]
        if self.use_bias:
            y += params["bias"]
        return y

    def apply_with_fwd_lap(self, params, x: NodeWithFwdLap):
        y = x.x @ params["kernel"]
        if self.use_bias:
            y += params["bias"]
        jac = x.jac @ params["kernel"]
        lap = x.lap @ params["kernel"]
        return NodeWithFwdLap(y, jac, lap, x.idx_ctr, x.idx_dep)

    def init(self, rng, x):
        return dict(
            kernel=jax.nn.initializers.lecun_normal(dtype=jnp.float32)(rng, (x.shape[-1], self.features)),
            bias=jnp.zeros((self.features,), dtype=jnp.float32),
        )


class ElementWise(PyTreeNode):
    fn: callable

    def apply(self, x):
        return self.fn(x)

    def apply_with_fwd_lap(self, x: NodeWithFwdLap):
        grad_fn = jax.grad(self.fn)
        hess_fn = jax.grad(grad_fn)
        y = [jnp.vectorize(f, signature="()->()")(x.x) for f in [self.fn, grad_fn, hess_fn]]
        return NodeWithFwdLap(
            y[0],
            x.jac * get(y[1], (x.idx_ctr, None)),
            x.lap * y[1] + x.get_jac_sqr() * y[2],
            x.idx_ctr,
            x.idx_dep,
        )


class MLP(PyTreeNode):
    width: int
    depth: int
    activate_final: bool = False
    activation: callable = nn.silu

    def apply(self, params, x):
        for i, p in enumerate(params):
            x = Linear(self.width).apply(p, x)
            if i < len(params) - 1 or self.activate_final:
                x = ElementWise(self.activation).apply(x)
        return x

    def apply_with_fwd_lap(self, params, x: NodeWithFwdLap):
        for i, p in enumerate(params):
            x = Linear(self.width).apply_with_fwd_lap(p, x)
            if i < len(params) - 1 or self.activate_final:
                x = ElementWise(self.activation).apply_with_fwd_lap(x)
        return x

    def init(self, rng, x):
        params = []
        for _ in range(self.depth):
            rng, key = jax.random.split(rng)
            params.append(Linear(self.width).init(key, x))
            x = np.zeros_like(x, shape=(self.width,))
        return params


def slogdet_with_fwd_lap(A: NodeWithFwdLap, triplet_indices):
    # Build all triplets of indices i,k,n where i and k are both neighbours of n.
    # This is required for the tr(JHJ.T) term in the laplacian of the logdet.
    # Triplets are aranged in the following order:
    # 1) n_el entries where i=k=n
    # 2) n_pair entries where k=n, i!=n
    # 3) n_pair entries where i=n, k!=n
    # 4) n_triplet entries where i,k,n are all distinct

    n_el = A.x.shape[-1]
    n_pairs = len(A.idx_ctr) - n_el

    pairidx_ct = A.idx_ctr[n_el:]
    pairidx_nb = A.idx_dep[n_el:]

    tripidx_i = jnp.concatenate([np.arange(n_el), pairidx_ct, pairidx_nb, triplet_indices[0]])
    tripidx_k = jnp.concatenate([np.arange(n_el), pairidx_nb, pairidx_ct, triplet_indices[1]])
    tripidx_n = jnp.concatenate([np.arange(n_el), pairidx_nb, pairidx_nb, triplet_indices[2]])
    tripidx_in, tripidx_kn = get_pair_indices_from_triplets((pairidx_ct, pairidx_nb), triplet_indices, n_el)
    tripidx_in = jnp.concatenate([np.arange(n_el), np.arange(n_pairs) + n_el, pairidx_nb, tripidx_in + n_el])
    tripidx_kn = jnp.concatenate([np.arange(n_el), pairidx_nb, np.arange(n_pairs) + n_el, tripidx_kn + n_el])

    A_inv = jnp.linalg.inv(A.x)
    M = A.jac @ A_inv

    sign, logdet = jnp.linalg.slogdet(A.x)
    jac_logdet = jnp.zeros([n_el, 3], dtype=A.x.dtype)
    jac_logdet = jac_logdet.at[A.idx_dep, :].add(get(M, (np.arange(n_pairs + n_el), slice(None), A.idx_ctr)))

    lap_logdet = jnp.einsum("ij,ji", A.lap, A_inv)
    lap_logdet -= jnp.sum(
        get(M, (tripidx_in, slice(None), tripidx_k)) * get(M, (tripidx_kn, slice(None), tripidx_i)),
    )
    return sign, FwdLaplArray(logdet, FwdJacobian(jac_logdet.reshape([n_el * 3])), lap_logdet)


def multiply_with_1el_fn(a: NodeWithFwdLap, b: FwdLaplArray):
    n_el = a.x.shape[0]
    idx_ctr = a.idx_ctr[n_el:]

    out = a.x * b.x
    jac_out = jnp.zeros_like(a.jac)
    jac_out = jac_out.at[:n_el, :, :].add(a.x[:, None, :] * b.jacobian.data + b.x[:, None, :] * a.jac[:n_el])
    jac_out = jac_out.at[n_el:, :, :].add(get(b.x, (idx_ctr, None)) * a.jac[n_el:])
    lap_out = a.lap * b.x + b.laplacian * a.x + 2 * jnp.sum(a.jac[:n_el] * b.jacobian.data, axis=1)
    return NodeWithFwdLap(out, jac_out, lap_out, a.idx_ctr, a.idx_dep)


class NucleusDependentParams(NamedTuple):
    filter: DynamicFilterParams
    nuc_embedding: jax.Array | None


class EdgeFeatures(nn.Module):
    cutoff: float
    filter_dims: tuple[int, int]
    feature_dim: int
    n_envelopes: int
    return_edge_embedding: bool = True

    @nn.compact
    def __call__(
        self,
        r_center: Float[Array, "dim=3"],
        r_neighbour: Float[Array, "dim=3"],
        dynamic_params: NucleusDependentParams,
    ):
        features = get_diff_features(r_center, r_neighbour)
        beta = PairwiseFilter(self.cutoff, self.filter_dims[1])(features, dynamic_params.filter)
        gamma = nn.Dense(self.feature_dim, use_bias=False)(beta)
        scaled_features = features / features[..., :1] * jnp.log1p(features[..., :1])
        edge_embedding = nn.Dense(self.feature_dim, use_bias=False)(scaled_features) + dynamic_params.nuc_embedding
        return gamma, edge_embedding


class ElecInit(nn.Module):
    cutoff: float
    filter_dims: tuple[int, int]
    feature_dim: int
    n_envelopes: int
    n_updates: int
    activation: Callable = nn.silu

    @nn.compact
    def __call__(
        self,
        r: Float[Array, " dim=3"],
        R_nb: Float[Array, "n_neighbours dim=3"],
        dynamic_params: NucleusDependentParams,
    ):
        n_out = self.n_updates * 2 + 1
        edge_feat = EdgeFeatures(self.cutoff, self.filter_dims, self.feature_dim, self.n_envelopes)
        features, Gamma = nn_vmap(edge_feat, in_axes=(None, 0, 0))(r, R_nb, dynamic_params)  # vmap over nuclei
        h_init = jnp.einsum("...Jd,...Jd->...d", features, Gamma)
        h_init = nn.LayerNorm()(h_init)
        h_init = GatedLinearUnit(self.feature_dim)(h_init)
        h_init = self.activation(nn.Dense(self.feature_dim)(h_init))
        # let's parallize this will by having all operations in one go
        h_out = jnp.split(nn.Dense(self.feature_dim * n_out)(h_init), n_out, -1)
        h_init, h_nb_same, h_nb_diff = h_out[0], h_out[1 : 1 + self.n_updates], h_out[1 + self.n_updates :]
        return h_init, h_nb_same, h_nb_diff


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


class ElectronUpdate(nn.Module):
    @nn.compact
    def __call__(self, h, msg):
        feature_dim = h.shape[-1]
        msg = nn.Dense(feature_dim)(msg)
        msg = nn.silu(msg)
        msg = nn.Dense(feature_dim)(msg)
        return nn.silu(msg + h) + h


class EmbeddingParams(NamedTuple):
    elec_init: Parameters
    edge_same: Parameters
    edge_diff: Parameters
    updates: Parameters
    orbitals: Parameters


def get_pair_indices(r, n_up, cutoff, n_pairs_same: int, n_pairs_diff: int):
    n_el = r.shape[0]
    n_dn = n_el - n_up
    spin = np.concatenate([np.zeros(n_up, bool), np.ones(n_dn, bool)])
    dist = jnp.linalg.norm(r[:, None, :] - r[None, :, :], axis=-1)
    dist = dist.at[jnp.arange(n_el), jnp.arange(n_el)].set(jnp.inf)
    in_cutoff = dist < cutoff
    is_same_spin = spin[:, None] == spin[None, :]

    pair_indices_same = jnp.where(in_cutoff & is_same_spin, size=n_pairs_same, fill_value=NO_NEIGHBOUR)
    pair_indices_diff = jnp.where(in_cutoff & ~is_same_spin, size=n_pairs_diff, fill_value=NO_NEIGHBOUR)
    idx_ct, idx_nb = jnp.where(in_cutoff, size=n_pairs_same + n_pairs_diff, fill_value=NO_NEIGHBOUR)
    jac_indices = (jnp.concatenate([np.arange(n_el), idx_ct]), jnp.concatenate([np.arange(n_el), idx_nb]))

    search_key = idx_ct * n_el + idx_nb
    jac_idx_same = jnp.searchsorted(search_key, pair_indices_same[0] * n_el + pair_indices_same[1]) + n_el
    jac_idx_diff = jnp.searchsorted(search_key, pair_indices_diff[0] * n_el + pair_indices_diff[1]) + n_el
    return (*pair_indices_same, jac_idx_same), (*pair_indices_diff, jac_idx_diff), jac_indices


def get_distinct_triplet_indices(r, cutoff, n_triplets_max: int):
    dist = jnp.linalg.norm(r[:, None, :] - r[None, :, :], axis=-1)
    dist = dist.at[jnp.arange(n_el), jnp.arange(n_el)].set(jnp.inf)
    in_cutoff = dist < cutoff
    is_triplet = in_cutoff[:, None, :] & in_cutoff[None, :, :]
    return jnp.where(is_triplet, size=n_triplets_max, fill_value=NO_NEIGHBOUR)


def get_pair_indices_from_triplets(pair_indices, triplet_indices, n_el):
    pair_search_keys = pair_indices[0] * n_el + pair_indices[1]
    pair_in = jnp.searchsorted(pair_search_keys, triplet_indices[0] * n_el + triplet_indices[2])
    pair_kn = jnp.searchsorted(pair_search_keys, triplet_indices[1] * n_el + triplet_indices[2])
    return pair_in, pair_kn


def contract(Gamma, h, idx_ctr, idx_nb):
    feature_dim = Gamma.shape[-1]
    n_el = h.shape[0]
    h_out = jnp.zeros([n_el, feature_dim], dtype=h.dtype)
    h_out = h_out.at[idx_ctr, :].add(Gamma * h[idx_nb, :])
    return h_out


def get(x, idx):
    return x.at[idx].get(mode="fill", fill_value=0.0)


def contract_with_fwd_lap(Gamma, h, idx_ctr, idx_nb, idx_jac, n_pairs_out):
    feature_dim = Gamma.shape[-1]
    n_el = h.shape[0]
    J_Gamma = Gamma.jacobian.data.reshape([-1, 2, 3, feature_dim])  # [n_pairs, self/neighbour, xyz, feature_dim]
    J_h = h.jacobian.data  # [n_el, 3, feature_dim]

    h_out = jnp.zeros([n_el, feature_dim], dtype=h.dtype)
    h_out = h_out.at[idx_ctr].add(Gamma.x * get(h.x, idx_nb))

    J_out = jnp.zeros([n_el + n_pairs_out, 3, feature_dim], dtype=h.dtype)
    J_out_self = J_Gamma[:, 0, :, :] * get(h.x, (idx_nb, None))
    J_out = J_out.at[idx_ctr, :, :].add(J_out_self)
    J_out_nb = J_Gamma[:, 1, :, :] * get(h.x, h.x, (idx_nb, None))
    J_out_nb += Gamma.x[:, None, :] * get(J_h, idx_nb)
    J_out = J_out.at[idx_jac, :, :].add(J_out_nb)

    lap_out = jnp.zeros_like(h_out)
    lap_out = lap_out.at[idx_ctr].add(Gamma.x * get(h.laplacian, idx_nb) + Gamma.laplacian * get(h.x, idx_nb))
    # .at[idx_ctr].add => sum over dependency index; jnp.sum => sum over xyz
    lap_out = lap_out.at[idx_ctr].add(2 * jnp.sum(J_Gamma[:, 1, :, :] * get(J_h, idx_nb), axis=1))

    idx_ctr = jnp.concatenate([jnp.arange(n_el), idx_ctr])
    idx_dep = jnp.concatenate([jnp.arange(n_el), idx_nb])

    return NodeWithFwdLap(h_out, J_out, lap_out, idx_ctr, idx_dep)


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
    updates: list[MLP]
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
            elec_init=ElecInit(feature_dim, n_layers + 1),
            edge_same=PairWiseFunction(cutoff, feature_dim, n_layers),
            edge_diff=PairWiseFunction(cutoff, feature_dim, n_layers),
            updates=[MLP(feature_dim, 2) for _ in range(n_layers)],
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

    def apply(self, params, r, static: StaticArgs[int]):
        n_el = r.shape[-2]
        (idx_ct_same, idx_nb_same, _), (idx_ct_diff, idx_nb_diff, _), _ = get_pair_indices(
            r, self.n_up, self.cutoff, static.n_pairs_same, static.n_pairs_diff
        )

        h0, h_nb_same, h_nb_diff = self.elec_init.apply(params.elec_init, r)
        Gamma_same = jax.vmap(lambda i, j: self.edge_same.apply(params.edge_same, r[i], r[j]))(idx_ct_same, idx_nb_same)
        Gamma_diff = jax.vmap(lambda i, j: self.edge_diff.apply(params.edge_diff, r[i], r[j]))(idx_ct_diff, idx_nb_diff)

        h = h0
        for h_same, h_diff, g_same, g_diff, e_same, e_diff, update_module, update_params in zip(
            h_nb_same, h_nb_diff, Gamma_same, Gamma_diff, self.updates, params.updates
        ):
            msg_same = contract(g_same, h_same, idx_ct_same, idx_nb_same)
            msg_diff = contract(g_diff, h_diff, idx_ct_diff, idx_nb_diff)
            h = update_module.apply(update_params, h + msg_same + msg_diff)

        orbitals = self.orbitals.apply(params.orbitals, h)
        env = jax.vmap(envelope, in_axes=(0, None))(r, n_el)
        orbitals *= env
        sign, logdet = jnp.linalg.slogdet(orbitals)
        return logdet

    def apply_with_fwd_lap(self, params, r, static: StaticArgs[int]):
        n_el = r.shape[0]
        (
            (idx_ct_same, idx_nb_same, idx_jac_same),
            (idx_ct_diff, idx_nb_diff, idx_jac_diff),
            (idx_jac_ctr, idx_jac_dep),
        ) = get_pair_indices(r, self.n_up, self.cutoff, static.n_pairs_same, static.n_pairs_diff)
        n_pairs = static.n_pairs_same + static.n_pairs_diff

        h0_fn = functools.partial(self.elec_init.apply, params.elec_init)
        edge_fn_same = functools.partial(self.edge_same.apply, params.edge_same)
        edge_fn_diff = functools.partial(self.edge_diff.apply, params.edge_diff)

        h0 = jax.vmap(fwd_lap(h0_fn))(r)
        Gamma_same = jax.vmap(fwd_lap(edge_fn_same))(r[idx_ct_same], r[idx_nb_same])
        Gamma_diff = jax.vmap(fwd_lap(edge_fn_diff))(r[idx_ct_diff], r[idx_nb_diff])
        jac_h0_padded = jnp.concatenate(
            [h0[0].jacobian.data.reshape([n_el, 3, self.feature_dim]), jnp.zeros([n_pairs, 3, self.feature_dim])]
        )
        h = NodeWithFwdLap(
            h0[0].x,
            jac_h0_padded,
            h0[0].laplacian,
            idx_jac_ctr,
            idx_jac_dep,
        )
        for l, (h_nb, g_same, g_diff) in enumerate(zip(h0[1:], Gamma_same, Gamma_diff)):
            msg_same = contract_with_fwd_lap(g_same, h_nb, idx_ct_same, idx_nb_same, idx_jac_same, n_pairs)
            msg_diff = contract_with_fwd_lap(g_diff, h_nb, idx_ct_diff, idx_nb_diff, idx_jac_diff, n_pairs)
            h = self.updates[l].apply_with_fwd_lap(params.updates[l], h + msg_same + msg_diff)

        orbitals = self.orbitals.apply_with_fwd_lap(params.orbitals, h)
        env = jax.vmap(fwd_lap(lambda r_: envelope(r_, n_el)))(r)
        orbitals = multiply_with_1el_fn(orbitals, env)
        triplet_indices = get_distinct_triplet_indices(r, self.cutoff, static.n_triplets)
        sign, logdet = slogdet_with_fwd_lap(orbitals, triplet_indices)
        return logdet


def is_close(a, b, msg=""):
    tol = 1e-6 if a.dtype == jnp.float32 else 1e-12
    try:
        np.testing.assert_allclose(a, b, err_msg=msg, rtol=tol, atol=tol)
    except AssertionError as e:
        print(e)


if __name__ == "__main__":
    n_el = 20
    n_up = n_el // 2
    rng_params, rng_r = jax.random.split(jax.random.PRNGKey(0))
    emb = Embedding.create(n_el, n_up, feature_dim=32, n_layers=3, cutoff=3.0)
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
