import flax.linen as nn
from flax.struct import PyTreeNode
from folx.api import FwdJacobian, FwdLaplArray
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from typing import Callable, Optional, Sequence
from sparse_wf.model.graph_utils import NO_NEIGHBOUR
from sparse_wf.model.utils import slog_and_inverse


class NodeWithFwdLap(PyTreeNode):
    # The jacobian is layed out as follows:
    # The first n_el entries correspond to the jacobian of the node with itself, ie. idx_dep = idx_ctr
    # The next n_pairs entries correspond to the jacobian of the node with its neighbours, i.e. idx_dep != idx_ctr
    x: jax.Array  # [n_el x feature_dim]
    jac: jax.Array  # [(n_el + n_pairs) x 3 x feature_dim]
    lap: jax.Array  # [n_el x feature_dim]
    idx_ctr: jax.Array  # [n_el + n_pairs]
    idx_dep: jax.Array  # [n_el + n_pairs]

    def get_jac_sqr(self):
        jac_sqr = jnp.zeros_like(self.x)
        # .at[idx_ctr].add => sum over dependency index; jnp.sum => sum over xyz
        jac_sqr = jac_sqr.at[self.idx_ctr].add(jnp.sum(self.jac**2, axis=1), mode="drop")
        return jac_sqr

    def dense_jac(self):
        n_el = self.x.shape[0]
        feature_dims = self.x.shape[1:]
        J_dense = jnp.zeros_like(self.jac, shape=(n_el, 3, n_el, *feature_dims))  # [dep, xyz, i, feature_dim]
        J_dense = J_dense.at[self.idx_dep, :, self.idx_ctr, ...].set(self.jac, mode="drop")
        J_dense = J_dense.reshape([n_el * 3, n_el, *feature_dims])
        return J_dense

    def sum_over_nodes(self) -> FwdLaplArray:
        n_el = self.x.shape[0]
        feature_dims = self.x.shape[1:]
        x = jnp.sum(self.x, axis=0)
        lap = jnp.sum(self.lap, axis=0)
        jac = jnp.zeros_like(self.jac, shape=[n_el, 3, *feature_dims])
        jac = jac.at[self.idx_dep, :, ...].add(self.jac, mode="drop")
        return FwdLaplArray(x, FwdJacobian(jac.reshape([n_el * 3, *feature_dims])), lap)

    def sum_from_to(self, lower, upper) -> FwdLaplArray:
        n_el = self.x.shape[0]
        feature_dims = self.x.shape[1:]
        x = jnp.sum(self.x[lower:upper], axis=0)
        lap = jnp.sum(self.lap[lower:upper], axis=0)
        mask = jnp.logical_and(self.idx_ctr >= lower, self.idx_ctr < upper)
        mask = mask[:, *[None] * (self.jac.ndim - 1)]
        jac = jnp.zeros_like(self.jac, shape=[n_el, 3, *feature_dims])
        jac = jac.at[self.idx_dep, ...].add(self.jac * mask, mode="drop")
        return FwdLaplArray(x, FwdJacobian(jac.reshape([n_el * 3, *feature_dims])), lap)

    def sum(self, axis):
        assert isinstance(axis, int)
        axis = axis % self.x.ndim
        assert axis != 0, "Cannot sum over the first axis, use sum_over_nodes instead"
        return NodeWithFwdLap(
            jnp.sum(self.x, axis),
            jnp.sum(self.jac, 1 + axis),
            jnp.sum(self.lap, axis),
            self.idx_ctr,
            self.idx_dep,
        )

    def to_folx(self):
        return FwdLaplArray(self.x, FwdJacobian(self.dense_jac()), self.lap)

    def __add__(self, other):
        if isinstance(other, NodeWithFwdLap):
            return NodeWithFwdLap(
                self.x + other.x,
                self.jac + other.jac,
                self.lap + other.lap,
                self.idx_ctr,
                self.idx_dep,
            )
        elif isinstance(other, jax.Array):
            return NodeWithFwdLap(self.x + other, self.jac, self.lap, self.idx_ctr, self.idx_dep)
        else:
            raise TypeError(f"Unsupported type for addition with NodeWithFwdLap: {type(other)}")

    def __mul__(self, scalar: float) -> "NodeWithFwdLap":
        if isinstance(scalar, NodeWithFwdLap):
            # TODO: This only works if both have the same dependencies otherwise this will result in a wrong laplacian
            y = self.x * scalar.x
            jac = self.jac * scalar.x.at[self.idx_ctr, None].get(mode="fill", fill_value=0.0) + scalar.jac * self.x.at[
                self.idx_ctr, None
            ].get(mode="fill", fill_value=0.0)
            lap = self.lap * scalar.x + self.x * scalar.lap
            lap = lap.at[self.idx_ctr, ...].add(2 * jnp.sum(self.jac * scalar.jac, axis=1), mode="drop")
            return NodeWithFwdLap(y, jac, lap, self.idx_ctr, self.idx_dep)
        return NodeWithFwdLap(self.x * scalar, self.jac * scalar, self.lap * scalar, self.idx_ctr, self.idx_dep)

    def __getitem__(self, idx):
        assert self.x[idx].shape[0] == self.x.shape[0], "Indexing must not change the number of elements"
        return NodeWithFwdLap(
            self.x.at[idx].get(mode="fill", fill_value=0.0),
            jax.vmap(lambda x: x.at[idx].get(mode="fill", fill_value=0.0), in_axes=1, out_axes=1)(self.jac),
            self.lap.at[idx].get(mode="fill", fill_value=0.0),
            self.idx_ctr,
            self.idx_dep,
        )

    def reshape(self, *shape):
        new_shape = self.x.reshape(shape).shape
        assert new_shape[0] == self.x.shape[0]
        return NodeWithFwdLap(
            self.x.reshape(new_shape),
            self.jac.reshape([*self.jac.shape[:2], *new_shape[1:]]),
            self.lap.reshape(new_shape),
            self.idx_ctr,
            self.idx_dep,
        )

    @property
    def dtype(self):
        return self.x.dtype

    @property
    def shape(self):
        return self.x.shape


def merge_up_down(orb_up: NodeWithFwdLap, orb_dn: NodeWithFwdLap, n_el: int, n_up: int):
    assert orb_up.x.shape == orb_dn.x.shape
    assert orb_up.lap.shape == orb_dn.lap.shape
    assert orb_up.jac.shape == orb_dn.jac.shape
    assert orb_up.x.shape[0] == n_el
    x = jnp.concatenate([orb_up.x[:n_up, ...], orb_dn.x[n_up:, ...]], axis=0)
    lap = jnp.concatenate([orb_up.lap[:n_up, ...], orb_dn.lap[n_up:, ...]], axis=0)
    jac = jnp.where((orb_up.idx_ctr < n_up)[:, None, None], orb_up.jac, orb_dn.jac)
    return NodeWithFwdLap(x, jac, lap, orb_up.idx_ctr, orb_up.idx_dep)


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


class Linear(nn.Module):
    features: int
    use_bias: bool = True
    bias_init: Optional[Callable] = nn.initializers.zeros_init()
    kernel_init: Optional[Callable] = nn.initializers.lecun_normal()

    @nn.compact
    def __call__(self, x, use_bias=True):
        if isinstance(x, NodeWithFwdLap):
            # Forward-Laplacian pass with jacobian and laplacian
            y = self(x.x)
            jac = self(x.jac, use_bias=False)
            lap = self(x.lap, use_bias=False)
            return NodeWithFwdLap(y, jac, lap, x.idx_ctr, x.idx_dep)
        else:
            # Regular forward pass
            kernel = self.param("kernel", self.kernel_init, (x.shape[-1], self.features), jnp.float32)
            y = x @ kernel
            if self.use_bias and use_bias:
                bias = self.param("bias", self.bias_init, (self.features,), jnp.float32)
                y += bias
        return y


class ElementWise(PyTreeNode):
    fn: callable

    def apply(self, x):
        return self.fn(x)

    def __call__(self, x):
        if isinstance(x, NodeWithFwdLap):
            return self.apply_with_fwd_lap(x)
        else:
            return self.apply(x)

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


class SparseMLP(nn.Module):
    widths: Sequence[int]
    activate_final: bool = False
    activation: Callable = jax.nn.silu
    output_bias: bool = True

    @nn.compact
    def __call__(self, x):
        for i, width in enumerate(self.widths):
            is_last_layer = i == len(self.widths) - 1
            use_bias = self.output_bias or ~is_last_layer
            use_activation = self.activate_final or ~is_last_layer

            x = Linear(width, use_bias)(x)
            if use_activation:
                x = ElementWise(self.activation)(x)
        return x


def _get_distinct_triplet_indices(distinct_pair_idx_i, distinct_pair_idx_n, n_el, n_triplets):
    n_distinct_pairs = len(distinct_pair_idx_i)

    # Build an overcomplete list of triplets of size n_pairs * n_el, by combining (i,n) x k
    full_trip_idx_i = jnp.repeat(distinct_pair_idx_i, n_el)
    full_trip_idx_n = jnp.repeat(distinct_pair_idx_n, n_el)
    full_trip_idx_k = jnp.tile(np.arange(n_el), n_distinct_pairs)

    # Search for the reverse pairs (k, n)
    pair_keys = distinct_pair_idx_i * n_el + distinct_pair_idx_n
    idx_pair_kn = jnp.searchsorted(pair_keys, full_trip_idx_k * n_el + full_trip_idx_n)
    is_triplet = (distinct_pair_idx_i[idx_pair_kn] == full_trip_idx_k) & (
        distinct_pair_idx_n[idx_pair_kn] == full_trip_idx_n
    )
    valid_trip_idx = jnp.where(is_triplet, size=n_triplets, fill_value=NO_NEIGHBOUR)[0]

    # Filter the triplet indices to only include those that are valid
    trip_idx_in = valid_trip_idx // n_el
    trip_idx_k = full_trip_idx_k[valid_trip_idx]
    trip_idx_kn = idx_pair_kn[valid_trip_idx]
    trip_idx_i = full_trip_idx_i[valid_trip_idx]
    return (trip_idx_in, trip_idx_k), (trip_idx_kn, trip_idx_i)


def _get_triplet_indices(pair_idx_i, pair_idx_n, n_el, n_distinct_triplets):
    n_distinct_pairs = len(pair_idx_i) - n_el
    # Allocate output buffers
    n_triplets_total = n_el + 2 * n_distinct_pairs + n_distinct_triplets
    trip_idx_in = jnp.zeros(n_triplets_total, dtype=jnp.int32)
    trip_idx_kn = jnp.zeros(n_triplets_total, dtype=jnp.int32)
    trip_idx_k = jnp.zeros(n_triplets_total, dtype=jnp.int32)
    trip_idx_i = jnp.zeros(n_triplets_total, dtype=jnp.int32)

    # i == k == n
    trip_idx_in = trip_idx_in.at[:n_el].set(jnp.arange(n_el))
    trip_idx_kn = trip_idx_kn.at[:n_el].set(jnp.arange(n_el))
    trip_idx_k = trip_idx_k.at[:n_el].set(jnp.arange(n_el))
    trip_idx_i = trip_idx_i.at[:n_el].set(jnp.arange(n_el))

    # (i != n), (k == n)
    s = slice(n_el, n_el + n_distinct_pairs)
    distinct_pair_idx_i, distinct_pair_idx_n = pair_idx_i[n_el:], pair_idx_n[n_el:]
    trip_idx_in = trip_idx_in.at[s].set(np.arange(n_distinct_pairs) + n_el)
    trip_idx_k = trip_idx_k.at[s].set(distinct_pair_idx_n)
    trip_idx_kn = trip_idx_kn.at[s].set(distinct_pair_idx_n)
    trip_idx_i = trip_idx_i.at[s].set(distinct_pair_idx_i)

    # (k != n), (i == n)
    s = slice(n_el + n_distinct_pairs, n_el + 2 * n_distinct_pairs)
    trip_idx_in = trip_idx_in.at[s].set(distinct_pair_idx_n)
    trip_idx_k = trip_idx_k.at[s].set(distinct_pair_idx_i)
    trip_idx_kn = trip_idx_kn.at[s].set(np.arange(n_distinct_pairs) + n_el)
    trip_idx_i = trip_idx_i.at[s].set(distinct_pair_idx_n)

    # (i != n), (k != n)
    s = slice(n_el + 2 * n_distinct_pairs, n_triplets_total)
    (idx_in, idx_k), (idx_kn, idx_i) = _get_distinct_triplet_indices(
        distinct_pair_idx_i, distinct_pair_idx_n, n_el, n_distinct_triplets
    )
    trip_idx_in = trip_idx_in.at[s].set(idx_in + n_el)
    trip_idx_k = trip_idx_k.at[s].set(idx_k)
    trip_idx_kn = trip_idx_kn.at[s].set(idx_kn + n_el)
    trip_idx_i = trip_idx_i.at[s].set(idx_i)
    return trip_idx_in, trip_idx_k, trip_idx_kn, trip_idx_i


def sparse_slogdet_with_fwd_lap(A: NodeWithFwdLap, n_triplets: int):
    (sign, logdet), A_inv = slog_and_inverse(A.x)
    M = A.jac @ A_inv

    # Build all triplets of indices i,k,n where i and k are both neighbours of n.
    n_el = A.x.shape[-1]
    n_pairs = len(A.idx_ctr) - n_el
    pair_idx_i, pair_idx_n = A.idx_ctr, A.idx_dep

    # This is simpler, but generates a matrix of size n_pairs**2
    # n_triplets_total = n_el + 2 * n_pairs + n_triplets
    # is_triplet = pair_idx_n[:, None] == pair_idx_n[None, :]
    # trip_idx_in, trip_idx_kn = jnp.where(is_triplet, size=n_triplets_total, fill_value=NO_NEIGHBOUR)
    # trip_idx_i, trip_idx_k = pair_idx_i[trip_idx_in], pair_idx_i[trip_idx_kn]

    # This is more complex, but only requires an intermediate matrix of size n_pairs * n_el
    trip_idx_in, trip_idx_k, trip_idx_kn, trip_idx_i = _get_triplet_indices(pair_idx_i, pair_idx_n, n_el, n_triplets)

    jac_logdet = jnp.zeros([n_el, 3], dtype=A.x.dtype)
    jac_logdet = jac_logdet.at[A.idx_dep, :].add(get(M, (np.arange(n_pairs + n_el), slice(None), A.idx_ctr)))

    lap_logdet = jnp.einsum("ij,ji", A.lap, A_inv)
    lap_logdet -= jnp.sum(
        get(M, (trip_idx_in, slice(None), trip_idx_k)) * get(M, (trip_idx_kn, slice(None), trip_idx_i))
    )
    return sign, FwdLaplArray(logdet, FwdJacobian(jac_logdet.reshape([n_el * 3])), lap_logdet)


def multiply_with_1el_fn(a: NodeWithFwdLap, b: FwdLaplArray) -> NodeWithFwdLap:
    n_el, n_features = a.x.shape
    jac_b = jnp.moveaxis(b.jacobian.data, 1, 0)  # [3 x n_el x n_features] -> [n_el x 3 x n_features]
    assert jac_b.shape == (n_el, 3, n_features)
    idx_ctr = a.idx_ctr[n_el:]

    out = a.x * b.x
    jac_out = jnp.zeros_like(a.jac)
    jac_out = jac_out.at[:n_el, :, :].add(a.x[:, None, :] * jac_b + b.x[:, None, :] * a.jac[:n_el], mode="drop")
    jac_out = jac_out.at[n_el:, :, :].add(
        b.x.at[idx_ctr, None, :].get(mode="fill", fill_value=0.0) * a.jac[n_el:], mode="drop"
    )
    lap_out = a.lap * b.x + b.laplacian * a.x + 2 * jnp.sum(a.jac[:n_el] * jac_b, axis=1)
    return NodeWithFwdLap(out, jac_out, lap_out, a.idx_ctr, a.idx_dep)


def get(x, idx, fill_value=0.0):
    if isinstance(x, jax.Array):
        return x.at[idx].get(mode="fill", fill_value=fill_value)
    else:
        return jtu.tree_map(lambda y: y.at[idx].get(mode="fill", fill_value=fill_value), x)
