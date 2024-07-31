import flax.linen as nn
from flax.struct import PyTreeNode
from folx.api import FwdJacobian, FwdLaplArray
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from typing import Callable, Optional
from sparse_wf.model.graph_utils import NO_NEIGHBOUR
from sparse_wf.model.utils import slog_and_inverse


class NodeWithFwdLap(PyTreeNode):
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
        jac_sqr = jac_sqr.at[self.idx_ctr].add(jnp.sum(self.jac**2, axis=1), mode="drop")
        return jac_sqr

    def dense_jac(self):
        n_el = np.max(self.idx_ctr) + 1
        feature_dims = self.x.shape[1:]
        J_dense = np.zeros((n_el, 3, n_el, *feature_dims))  # [dep, xyz, i, feature_dim]
        J_dense[self.idx_dep, :, self.idx_ctr, ...] = self.jac
        J_dense = J_dense.reshape([n_el * 3, n_el, *feature_dims])
        return J_dense

    def to_folx(self):
        return FwdLaplArray(self.x, FwdJacobian(self.dense_jac()), self.lap)

    def __add__(self, other):
        return NodeWithFwdLap(self.x + other.x, self.jac + other.jac, self.lap + other.lap, self.idx_ctr, self.idx_dep)

    def __mul__(self, scalar: float):
        return NodeWithFwdLap(self.x * scalar, self.jac * scalar, self.lap * scalar, self.idx_ctr, self.idx_dep)

    @property
    def dtype(self):
        return self.x.dtype

    @property
    def shape(self):
        return self.x.shape


def to_slater_matrices(orbitals: NodeWithFwdLap, n_el: int, n_up):
    n_dets = orbitals.x.shape[-1] // n_el

    def reshape(X):
        X = X.reshape([n_el, n_dets, n_el])  # [el x det x orbital]
        X = jnp.moveaxis(X, 1, 0)  # [det x el x orbital]
        # swap bottom spin blocks
        top_block = X[:, :n_up, :]
        bottom_block = jnp.concatenate([X[:, n_up:, n_up:], X[:, n_up:, :n_up]], axis=2)
        X = jnp.concatenate([top_block, bottom_block], axis=1)
        return X

    phi = reshape(orbitals.x)
    lap = reshape(orbitals.lap)
    jac = orbitals.jac.reshape([-1, 3, n_dets, n_el])  # [pair(el,dep) x xyz x det x orb]
    jac = jnp.moveaxis(jac, 2, 0)  # [det x pair(el,dep) x xyz x orb]
    jac = jnp.where(
        (orbitals.idx_ctr < n_up)[None, :, None, None],
        jac,
        jnp.concatenate([jac[:, :, :, n_up:], jac[:, :, :, :n_up]], axis=3),
    )
    return phi, jac, lap


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
    n_el = r.shape[0]
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


class Linear(PyTreeNode):
    features: int
    use_bias: bool = True
    bias_init: Optional[Callable] = None
    kernel_init: Optional[Callable] = None

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
        rngs = jax.random.split(rng, 2)
        bias_init = self.bias_init or (lambda k, s: jnp.zeros(s, dtype=jnp.float32))
        kernel_init = self.kernel_init or nn.initializers.lecun_normal(dtype=jnp.float32)
        return dict(
            kernel=kernel_init(rngs[0], (x.shape[-1], self.features)),
            bias=bias_init(rngs[1], (self.features,)),
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


def sparse_slogdet_with_fwd_lap(A: NodeWithFwdLap, triplet_indices):
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
    # tripidx_n = jnp.concatenate([np.arange(n_el), pairidx_nb, pairidx_nb, triplet_indices[2]])
    tripidx_in, tripidx_kn = get_pair_indices_from_triplets((pairidx_ct, pairidx_nb), triplet_indices, n_el)
    tripidx_in = jnp.concatenate([np.arange(n_el), np.arange(n_pairs) + n_el, pairidx_nb, tripidx_in + n_el])
    tripidx_kn = jnp.concatenate([np.arange(n_el), pairidx_nb, np.arange(n_pairs) + n_el, tripidx_kn + n_el])

    (sign, logdet), A_inv = slog_and_inverse(A.x)
    M = A.jac @ A_inv

    jac_logdet = jnp.zeros([n_el, 3], dtype=A.x.dtype)
    jac_logdet = jac_logdet.at[A.idx_dep, :].add(get(M, (np.arange(n_pairs + n_el), slice(None), A.idx_ctr)))

    lap_logdet = jnp.einsum("ij,ji", A.lap, A_inv)
    lap_logdet -= jnp.sum(get(M, (tripidx_in, slice(None), tripidx_k)) * get(M, (tripidx_kn, slice(None), tripidx_i)))
    return sign, FwdLaplArray(logdet, FwdJacobian(jac_logdet.reshape([n_el * 3])), lap_logdet)


def multiply_with_1el_fn(a: NodeWithFwdLap, b: FwdLaplArray) -> NodeWithFwdLap:
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
    return NodeWithFwdLap(out, jac_out, lap_out, a.idx_ctr, a.idx_dep)


def get(x, idx, fill_value=0.0):
    if isinstance(x, jax.Array):
        return x.at[idx].get(mode="fill", fill_value=fill_value)
    else:
        return jtu.tree_map(lambda y: y.at[idx].get(mode="fill", fill_value=fill_value), x)
