# %%
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
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
from folx.api import FwdLaplArray
import jax.tree_util as jtu


class FwdLapTuple(NamedTuple):
    x: jax.Array
    jac: jax.Array
    lap: jax.Array
    idx_ctr: jax.Array
    idx_dep: jax.Array

    def get_jac_sqr(self):
        jac_sqr = jnp.zeros_like(self.x)
        jac_sqr = jac_sqr.at[self.idx_ctr].add(jnp.sum(self.jac**2, axis=1), mode="drop")
        return jac_sqr

    def dense_jac(self):
        n_el = np.max(self.idx_ctr) + 1
        feature_dim = self.jac.shape[-1]
        J_dense = np.zeros((n_el, 3, n_el, feature_dim))  # [dep, xyz, i, feature_dim]
        J_dense[self.idx_dep, :, self.idx_ctr, :] = self.jac
        J_dense = J_dense.reshape([n_el * 3, n_el, feature_dim])
        return J_dense


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


T = TypeVar("T", bound=int | Int)


class StaticArgs(NamedTuple, Generic[T]):
    n_pairs: T

    def to_static(self):
        return StaticArgs(int(jnp.max(self.n_pairs)))


def get_pair_indices(r, cutoff, n_pairs_max: int):
    n_el = r.shape[0]
    dist = jnp.linalg.norm(r[:, None, :] - r[None, :, :], axis=-1)
    dist = dist.at[jnp.arange(n_el), jnp.arange(n_el)].set(jnp.inf)
    return jnp.where(dist < cutoff, size=n_pairs_max, fill_value=NO_NEIGHBOUR)


def contract(Gamma, h, idx_ctr, idx_nb):
    feature_dim = Gamma.shape[-1]
    n_el = h.shape[0]
    h_out = jnp.zeros([n_el, feature_dim], dtype=h.dtype)
    h_out = h_out.at[idx_ctr, :].add(Gamma * h[idx_nb, :], mode="drop")
    return h_out


def get_with_fill(x, idx):
    return x.at[idx].get(mode="fill", fill_value=0.0)


# TODO: all indexing with .at[].get(mode="fill", fill_value=0.0)
def contract_with_fwd_lap(Gamma, h, idx_ctr, idx_nb):
    n_pairs, feature_dim = Gamma.shape
    n_el = h.shape[0]
    J_Gamma = Gamma.jacobian.data.reshape([n_pairs, 2, 3, feature_dim])  # [n_pairs, self/neighbour, xyz, feature_dim]
    J_h = h.jacobian.data  # [n_el, 3, feature_dim]

    h_out = jnp.zeros([n_el, feature_dim], dtype=h.dtype)
    h_out = h_out.at[idx_ctr].add(Gamma.x * h.x[idx_nb], mode="drop")

    J_out = jnp.zeros([n_el + n_pairs, 3, feature_dim], dtype=h.dtype)
    J_out_self = J_Gamma[:, 0, :, :] * h.x.at[idx_nb, None, :].get(mode="fill", fill_value=0.0)
    J_out = J_out.at[idx_ctr, :, :].add(J_out_self, mode="drop")
    J_out_nb = J_Gamma[:, 1, :, :] * h.x.at[idx_nb, None, :].get(mode="fill", fill_value=0.0)
    J_out_nb += Gamma.x[:, None, :] * J_h.at[idx_nb, :, :].get(mode="fill", fill_value=0.0)
    J_out = J_out.at[n_el:, :, :].add(J_out_nb, mode="drop")

    return (h_out, J_out)


def build_dense_jacobian(J, indices):
    n_el = np.max(indices) + 1
    feature_dim = J.shape[-1]
    J_dense = np.zeros((n_el, 3, n_el, feature_dim))  # [dep, xyz, i, feature_dim]
    i, j = indices
    J_dense[np.arange(n_el), :, np.arange(n_el), :] = J[:n_el, :, :]
    J_dense[j, :, i, :] = J[n_el:, :, :]
    J_dense = J_dense.reshape([n_el * 3, n_el, feature_dim])
    return J_dense


class Embedding(PyTreeNode):
    # Hyperparams
    feature_dim: int
    n_layers: int
    cutoff: float

    # Submodules
    elec_init: ElecInit
    edge: PairWiseFunction
    updates: list[MLP]

    @classmethod
    def create(cls, feature_dim, n_layers, cutoff):
        return cls(
            feature_dim,
            n_layers,
            cutoff,
            elec_init=ElecInit(feature_dim, n_layers + 1),
            edge=PairWiseFunction(cutoff, feature_dim, n_layers),
            updates=[MLP(feature_dim, 2) for _ in range(n_layers)],
        )

    def init(self, rng):
        rngs = jax.random.split(rng, 2 + self.n_layers)
        return EmbeddingParams(
            self.elec_init.init(rngs[0], np.zeros(3)),
            self.edge.init(rngs[1], np.zeros(3), np.zeros(3)),
            [u.init(key, np.zeros(self.feature_dim)) for u, key in zip(self.updates, rngs[2:])],
        )

    def get_static_args(self, r) -> StaticArgs[Int]:
        n_el = r.shape[0]
        dist = jnp.linalg.norm(r[:, None, :] - r[None, :, :], axis=-1)
        n_pairs = jnp.sum(dist < self.cutoff) - n_el
        return StaticArgs(n_pairs)

    # def apply_dense(self, params, r, static: StaticArgs[int]):
    #     n_el = r.shape[0]
    #     dist = jnp.linalg.norm(r[:, None, :] - r[None, :, :], axis=-1)
    #     mask = (dist < self.cutoff) & (~np.eye(n_el, dtype=bool))
    #     mask = mask.astype(jnp.float32)

    #     h0_fn = functools.partial(self.elec_init.apply, params.elec_init)
    #     edge_fn = functools.partial(self.edge.apply, params.edge)

    #     h0 = jax.vmap(h0_fn)(r)
    #     Gamma = jax.vmap(jax.vmap(edge_fn, in_axes=(None, 0)), in_axes=(0, None))(r, r)
    #     Gamma = jtu.tree_map(lambda g: g * mask[:, :, None], Gamma)

    #     msg = jnp.einsum("jf,ijf->if", h0[1], Gamma[0])
    #     return Gamma[0]

    def apply(self, params, r, static: StaticArgs[int]):
        n_el = r.shape[0]
        idx_ct, idx_nb = get_pair_indices(r, self.cutoff, static.n_pairs)

        h0 = self.elec_init.apply(params.elec_init, r)
        Gamma = jax.vmap(lambda i, j: self.edge.apply(params.edge, r[i], r[j]))(idx_ct, idx_nb)

        msg = contract(Gamma[0], h0[1], idx_ct, idx_nb)
        h_out = self.updates[0].apply(params.updates[0], msg)
        # h = h0[0]
        # for l, (h_nb, g) in enumerate(zip(h0[1:], Gamma)):
        #     msg = contract(g, h_nb, idx_ct, idx_nb)
        # h = self.updates[l].apply(params.updates[l], h, msg)

        return h_out

    def apply_with_fwd_lap(self, params, r, static: StaticArgs[int]):
        n_el = r.shape[0]
        idx_ct, idx_nb = get_pair_indices(r, self.cutoff, static.n_pairs)

        h0_fn = functools.partial(self.elec_init.apply, params.elec_init)
        edge_fn = functools.partial(self.edge.apply, params.edge)

        h0 = jax.vmap(fwd_lap(h0_fn))(r)
        Gamma = jax.vmap(fwd_lap(edge_fn))(r[idx_ct], r[idx_nb])
        msg, msg_jac = contract_with_fwd_lap(Gamma[0], h0[1], idx_ct, idx_nb)
        msg = FwdLapTuple(
            msg,
            msg_jac,
            jnp.zeros_like(msg),
            jnp.concatenate([jnp.arange(n_el), idx_ct]),
            jnp.concatenate([jnp.arange(n_el), idx_nb]),
        )
        h_out = self.updates[0].apply_with_fwd_lap(params.updates[0], msg)
        # h = h0[0]
        # for l, (h_nb, g) in enumerate(zip(h0[1:], Gamma)):
        #     msg = contract_with_fwd_lap(g, h_nb, idx_ct, idx_nb)

        return h_out


def is_close(a, b, msg=""):
    try:
        np.testing.assert_allclose(a, b, err_msg=msg, rtol=1e-6, atol=1e-6)
    except AssertionError as e:
        print(e)


if __name__ == "__main__":
    n_el = 10
    rng_params, rng_r = jax.random.split(jax.random.PRNGKey(0))
    emb = Embedding.create(feature_dim=32, n_layers=1, cutoff=3.0)
    params = emb.init(rng_params)
    r = jax.random.normal(rng_r, [n_el, 3]) * 3.0
    static = emb.get_static_args(r).to_static()
    print(static)

    h_folx = fwd_lap(emb.apply, argnums=1)(params, r, static)
    h_sparse = emb.apply_with_fwd_lap(params, r, static)

    # h_dense = emb.apply_dense(params, r, static)
    # h = emb.apply(params, r, static)
    # is_close(h_dense, h, "h_dense != h")
    # is_close(h_folx.x, h, "h_folx.x != h")

    is_close(h_folx.x, h_sparse.x, "h_folx.x != h_sparse.x")
    is_close(h_folx.jacobian.data, h_sparse.dense_jac(), "h_folx.jac != h_sparse.jac")
    # # h = emb.apply(params, r, static)
