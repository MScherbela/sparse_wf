# %%
import os

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
import jax.tree_util as jtu

T = TypeVar("T", bound=int | Int)


class StaticArgs(NamedTuple, Generic[T]):
    n_pairs: T
    n_triplets: T

    def to_static(self):
        return StaticArgs(int(jnp.max(self.n_pairs)), int(jnp.max(self.n_triplets)))


class FwdLapTuple(NamedTuple):
    x: jax.Array
    jac: jax.Array
    lap: jax.Array
    idx_ctr: jax.Array
    idx_dep: jax.Array

    def get_jac_sqr(self):
        jac_sqr = jnp.zeros_like(self.x)
        # .at[idx_ctr].add => sum over dependency index; jnp.sum => sum over xyz
        jac_sqr = jac_sqr.at[self.idx_ctr].add(jnp.sum(self.jac**2, axis=1), mode="drop")
        return jac_sqr

    def dense_jac(self):
        n_el = np.max(self.idx_ctr) + 1
        feature_dim = self.jac.shape[-1]
        J_dense = np.zeros((n_el, 3, n_el, feature_dim))  # [dep, xyz, i, feature_dim]
        J_dense[self.idx_dep, :, self.idx_ctr, :] = self.jac
        J_dense = J_dense.reshape([n_el * 3, n_el, feature_dim])
        return J_dense

    def __add__(self, other):
        return FwdLapTuple(self.x + other.x, self.jac + other.jac, self.lap + other.lap, self.idx_ctr, self.idx_dep)


class Linear(PyTreeNode):
    features: int
    use_bias: bool = True

    def apply(self, params, x):
        y = x @ params["kernel"]
        if self.use_bias:
            y += params["bias"]
        return y

    def apply_with_fwd_lap(self, params, x: FwdLapTuple):
        y = x.x @ params["kernel"]
        if self.use_bias:
            y += params["bias"]
        jac = x.jac @ params["kernel"]
        lap = x.lap @ params["kernel"]
        return FwdLapTuple(y, jac, lap, x.idx_ctr, x.idx_dep)

    def init(self, rng, x):
        return dict(
            kernel=jax.nn.initializers.lecun_normal(dtype=jnp.float32)(rng, (x.shape[-1], self.features)),
            bias=jnp.zeros((self.features,), dtype=jnp.float32),
        )


class ElementWise(PyTreeNode):
    fn: callable

    def apply(self, x):
        return self.fn(x)

    def apply_with_fwd_lap(self, x: FwdLapTuple):
        grad_fn = jax.grad(self.fn)
        hess_fn = jax.grad(grad_fn)
        y = [jnp.vectorize(f, signature="()->()")(x.x) for f in [self.fn, grad_fn, hess_fn]]
        return FwdLapTuple(
            y[0],
            x.jac * y[1].at[x.idx_ctr, None].get(mode="fill", fill_value=0.0),
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

    def apply_with_fwd_lap(self, params, x: FwdLapTuple):
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


def slogdet_with_fwd_lap(A: FwdLapTuple, triplet_indices):
    n_el = A.x.shape[-1]
    n_pairs = len(A.idx_ctr) - n_el

    pairidx_ct = A.idx_ctr[n_el:]
    pairidx_nb = A.idx_dep[n_el:]

    tripidx_i = np.concatenate([np.arange(n_el), pairidx_ct, pairidx_nb, triplet_indices[0]])
    tripidx_k = np.concatenate([np.arange(n_el), pairidx_nb, pairidx_ct, triplet_indices[1]])
    tripidx_n = np.concatenate([np.arange(n_el), pairidx_nb, pairidx_nb, triplet_indices[2]])
    tripidx_in, tripidx_kn = get_pair_indices_from_triplets((pairidx_ct, pairidx_nb), triplet_indices, n_el)
    tripidx_in = np.concatenate([np.arange(n_el), np.arange(n_pairs) + n_el, pairidx_nb, tripidx_in + n_el])
    tripidx_kn = np.concatenate([np.arange(n_el), pairidx_nb, np.arange(n_pairs) + n_el, tripidx_kn + n_el])

    A_inv = jnp.linalg.inv(A.x)
    M = A.jac @ A_inv

    sign, logdet = jnp.linalg.slogdet(A.x)
    jac_logdet = jnp.zeros([n_el, 3], dtype=A.x.dtype)
    jac_logdet = jac_logdet.at[A.idx_dep, :].add(
        M.at[np.arange(n_pairs + n_el), :, A.idx_ctr].get(mode="fill", fill_value=0.0), mode="drop"
    )

    lap_logdet = jnp.einsum("ij,ji", A.lap, A_inv)
    lap_logdet -= jnp.sum(
        M.at[tripidx_in, :, tripidx_k].get(mode="fill", fill_value=0.0)
        * M.at[tripidx_kn, :, tripidx_i].get(mode="fill", fill_value=0.0)
    )
    return sign, FwdLaplArray(logdet, FwdJacobian(jac_logdet.reshape([n_el * 3])), lap_logdet)


def multiply_with_1el_fn(a: FwdLapTuple, b: FwdLaplArray):
    n_el = a.x.shape[0]
    idx_ctr = a.idx_ctr[n_el:]

    out = a.x * b.x
    jac_out = jnp.zeros_like(a.jac)
    jac_out = jac_out.at[:n_el, :, :].add(
        a.x[:, None, :] * b.jacobian.data + b.x[:, None, :] * a.jac[:n_el], mode="drop"
    )
    jac_out = jac_out.at[n_el:, :, :].add(
        b.x.at[idx_ctr, None, :].get(mode="fill", fill_value=0.0) * a.jac[n_el:], mode="drop"
    )
    lap_out = a.lap * b.x + b.laplacian * a.x + 2 * jnp.sum(a.jac[:n_el] * b.jacobian.data, axis=1)
    return FwdLapTuple(out, jac_out, lap_out, a.idx_ctr, a.idx_dep)


class ElecInit(nn.Module):
    feature_dim: int
    n_out: int

    @nn.compact
    def __call__(self, r):
        h0 = nn.Dense(
            features=self.feature_dim * self.n_out,
        )(r)
        h0 = nn.silu(h0)
        return jnp.split(h0, self.n_out, axis=-1)


class PairWiseFunction(nn.Module):
    cutoff: float
    feature_dim: int
    n_out: int

    @nn.compact
    def __call__(self, r, r_nb):
        diff = r - r_nb
        dist = jnp.linalg.norm(diff, axis=-1)
        diff_features = jnp.concatenate([diff, dist[..., None]], axis=-1)
        diff_features = nn.silu(nn.Dense(16)(diff_features))
        diff_features = nn.silu(nn.Dense(8)(diff_features))
        beta = diff_features * cutoff_function(dist / self.cutoff)
        Gamma = nn.Dense(features=self.feature_dim * self.n_out, use_bias=False)(beta)
        return jnp.split(Gamma, self.n_out, axis=-1)


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
    edge: Parameters
    updates: Parameters
    orbitals: Parameters


def get_pair_indices(r, cutoff, n_pairs_max: int):
    n_el = r.shape[0]
    dist = jnp.linalg.norm(r[:, None, :] - r[None, :, :], axis=-1)
    dist = dist.at[jnp.arange(n_el), jnp.arange(n_el)].set(jnp.inf)
    return jnp.where(dist < cutoff, size=n_pairs_max, fill_value=NO_NEIGHBOUR)


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
    h_out = h_out.at[idx_ctr, :].add(Gamma * h[idx_nb, :], mode="drop")
    return h_out


def get(x, idx):
    return x.at[idx].get(mode="fill", fill_value=0.0)


# TODO: all indexing with .at[].get(mode="fill", fill_value=0.0)
def contract_with_fwd_lap(Gamma, h, idx_ctr, idx_nb):
    n_pairs, feature_dim = Gamma.shape
    n_el = h.shape[0]
    J_Gamma = Gamma.jacobian.data.reshape([n_pairs, 2, 3, feature_dim])  # [n_pairs, self/neighbour, xyz, feature_dim]
    J_h = h.jacobian.data  # [n_el, 3, feature_dim]

    h_out = jnp.zeros([n_el, feature_dim], dtype=h.dtype)
    h_out = h_out.at[idx_ctr].add(Gamma.x * get(h.x, idx_nb), mode="drop")

    J_out = jnp.zeros([n_el + n_pairs, 3, feature_dim], dtype=h.dtype)
    J_out_self = J_Gamma[:, 0, :, :] * h.x.at[idx_nb, None, :].get(mode="fill", fill_value=0.0)
    J_out = J_out.at[idx_ctr, :, :].add(J_out_self, mode="drop")
    J_out_nb = J_Gamma[:, 1, :, :] * h.x.at[idx_nb, None, :].get(mode="fill", fill_value=0.0)
    J_out_nb += Gamma.x[:, None, :] * J_h.at[idx_nb, :, :].get(mode="fill", fill_value=0.0)
    J_out = J_out.at[n_el:, :, :].add(J_out_nb, mode="drop")

    lap_out = jnp.zeros_like(h_out)
    lap_out = lap_out.at[idx_ctr].add(
        Gamma.x * get(h.laplacian, idx_nb) + Gamma.laplacian * get(h.x, idx_nb), mode="drop"
    )
    # .at[idx_ctr].add => sum over dependency index; jnp.sum => sum over xyz
    lap_out = lap_out.at[idx_ctr].add(2 * jnp.sum(J_Gamma[:, 1, :, :] * get(J_h, idx_nb), axis=1), mode="drop")

    idx_ctr = jnp.concatenate([jnp.arange(n_el), idx_ctr])
    idx_dep = jnp.concatenate([jnp.arange(n_el), idx_nb])

    return FwdLapTuple(h_out, J_out, lap_out, idx_ctr, idx_dep)


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
    # Hyperparams
    feature_dim: int
    n_layers: int
    cutoff: float

    # Submodules
    elec_init: ElecInit
    edge: PairWiseFunction
    updates: list[MLP]
    orbitals: Linear

    @classmethod
    def create(cls, n_el, feature_dim, n_layers, cutoff):
        return cls(
            feature_dim,
            n_layers,
            cutoff,
            elec_init=ElecInit(feature_dim, n_layers + 1),
            edge=PairWiseFunction(cutoff, feature_dim, n_layers),
            updates=[MLP(feature_dim, 2) for _ in range(n_layers)],
            orbitals=Linear(n_el),
        )

    def init(self, rng):
        rngs = jax.random.split(rng, 3 + self.n_layers)
        return EmbeddingParams(
            self.elec_init.init(rngs[0], np.zeros(3)),
            self.edge.init(rngs[1], np.zeros(3), np.zeros(3)),
            [u.init(key, np.zeros(self.feature_dim)) for u, key in zip(self.updates, rngs[2:-1])],
            self.orbitals.init(rngs[-1], np.zeros(self.feature_dim)),
        )

    def get_static_args(self, r) -> StaticArgs[Int]:
        n_el = r.shape[0]
        dist = jnp.linalg.norm(r[:, None, :] - r[None, :, :], axis=-1)
        dist = dist.at[jnp.arange(n_el), jnp.arange(n_el)].set(jnp.inf)
        n_pairs = jnp.sum(dist < self.cutoff)
        in_cutoff = dist < self.cutoff
        n_triplets = jnp.sum(in_cutoff[:, None, :] & in_cutoff[None, :, :])
        return StaticArgs(n_pairs, n_triplets)

    def apply(self, params, r, static: StaticArgs[int]):
        n_el = r.shape[-2]
        idx_ct, idx_nb = get_pair_indices(r, self.cutoff, static.n_pairs)

        h0 = self.elec_init.apply(params.elec_init, r)
        Gamma = jax.vmap(lambda i, j: self.edge.apply(params.edge, r[i], r[j]))(idx_ct, idx_nb)

        h = h0[0]
        for l, (h_nb, g) in enumerate(zip(h0[1:], Gamma)):
            msg = contract(g, h_nb, idx_ct, idx_nb)
            h = self.updates[l].apply(params.updates[l], h + msg)

        orbitals = self.orbitals.apply(params.orbitals, h)
        env = jax.vmap(envelope, in_axes=(0, None))(r, n_el)
        orbitals *= env
        sign, logdet = jnp.linalg.slogdet(orbitals)
        return logdet

    def apply_with_fwd_lap(self, params, r, static: StaticArgs[int]):
        n_el = r.shape[0]
        idx_ct, idx_nb = get_pair_indices(r, self.cutoff, static.n_pairs)

        idx_jac_ctr = jnp.concatenate([jnp.arange(n_el), idx_ct])
        idx_jac_dep = jnp.concatenate([jnp.arange(n_el), idx_nb])

        h0_fn = functools.partial(self.elec_init.apply, params.elec_init)
        edge_fn = functools.partial(self.edge.apply, params.edge)

        h0 = jax.vmap(fwd_lap(h0_fn))(r)
        Gamma = jax.vmap(fwd_lap(edge_fn))(r[idx_ct], r[idx_nb])
        jac_h0_padded = jnp.concatenate(
            [h0[0].jacobian.data.reshape([n_el, 3, self.feature_dim]), jnp.zeros([static.n_pairs, 3, self.feature_dim])]
        )
        h = FwdLapTuple(
            h0[0].x,
            jac_h0_padded,
            h0[0].laplacian,
            idx_jac_ctr,
            idx_jac_dep,
        )
        for l, (h_nb, g) in enumerate(zip(h0[1:], Gamma)):
            msg = contract_with_fwd_lap(g, h_nb, idx_ct, idx_nb)
            h = self.updates[l].apply_with_fwd_lap(params.updates[l], h + msg)

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
    n_el = 10
    rng_params, rng_r = jax.random.split(jax.random.PRNGKey(0))
    emb = Embedding.create(n_el, feature_dim=32, n_layers=3, cutoff=3.0)
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
