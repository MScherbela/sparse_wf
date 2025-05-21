#%%
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp
import jax
from sparse_wf.model.sparse_fwd_lap import NO_NEIGHBOUR, get_pair_indices
import numpy as np
import jax.tree_util as jtu


def get_static_args(r, cutoff, n_up):
    n_el = len(r)
    spin = np.concatenate([np.zeros(n_up, bool), np.ones(n_el - n_up, bool)])
    dist = jnp.linalg.norm(r[:, None, :] - r[None, :, :], axis=-1)
    dist = dist.at[jnp.arange(n_el), jnp.arange(n_el)].set(jnp.inf)
    in_cutoff = dist < cutoff
    is_same_spin = spin[:, None] == spin[None, :]

    n_pairs_same = jnp.sum(in_cutoff & is_same_spin)
    n_pairs_diff = jnp.sum(in_cutoff & ~is_same_spin)
    n_triplets = jnp.sum(in_cutoff[:, None, :] & in_cutoff[None, :, :])
    return n_pairs_same, n_pairs_diff, n_triplets

def get_triplet_indices_old(pair_idx_i, pair_idx_n, n_el, n_triplets):
    n_pairs = len(pair_idx_i) - n_el
    n_triplets_total = n_el + 2 * n_pairs + n_triplets
    is_triplet = pair_idx_n[:, None] == pair_idx_n[None, :]
    trip_idx_in, trip_idx_kn = jnp.where(is_triplet, size=n_triplets_total, fill_value=NO_NEIGHBOUR)
    trip_idx_i, trip_idx_k = pair_idx_i[trip_idx_in], pair_idx_i[trip_idx_kn]
    return trip_idx_in, trip_idx_k, trip_idx_kn, trip_idx_i

def get_distinct_triplet_indices(distinct_pair_idx_i, distinct_pair_idx_n, n_el, n_triplets):
    n_distinct_pairs = len(distinct_pair_idx_i)

    # Build an overcomplete list of triplets of size n_pairs * n_el, by combining (i,n) x k
    full_trip_idx_i = jnp.repeat(distinct_pair_idx_i, n_el)
    full_trip_idx_n = jnp.repeat(distinct_pair_idx_n, n_el)
    full_trip_idx_k = jnp.tile(np.arange(n_el), n_distinct_pairs)

    # Search for the reverse pairs (k, n)
    pair_keys = distinct_pair_idx_i * n_el + distinct_pair_idx_n
    idx_pair_kn = jnp.searchsorted(pair_keys, full_trip_idx_k * n_el + full_trip_idx_n)
    is_triplet = (distinct_pair_idx_i[idx_pair_kn] == full_trip_idx_k) & (distinct_pair_idx_n[idx_pair_kn] == full_trip_idx_n)
    valid_trip_idx = jnp.where(is_triplet, size=n_triplets, fill_value=NO_NEIGHBOUR)[0]

    # Filter the triplet indices to only include those that are valid
    trip_idx_in = valid_trip_idx // n_el
    trip_idx_k = full_trip_idx_k[valid_trip_idx]
    trip_idx_kn = idx_pair_kn[valid_trip_idx]
    trip_idx_i = full_trip_idx_i[valid_trip_idx]
    return (trip_idx_in, trip_idx_k), (trip_idx_kn, trip_idx_i)

def get_triplet_indices(pair_idx_i, pair_idx_n, n_el, n_distinct_triplets):
    n_distinct_pairs = len(pair_idx_i) - n_el
    # Allocate output buffers
    n_triplets_total = n_el + 2 * n_distinct_pairs + n_distinct_triplets
    trip_idx_in = jnp.zeros(n_triplets_total, dtype=jnp.int32)
    trip_idx_kn = jnp.zeros(n_triplets_total, dtype=jnp.int32)
    trip_idx_k =  jnp.zeros(n_triplets_total, dtype=jnp.int32)
    trip_idx_i =  jnp.zeros(n_triplets_total, dtype=jnp.int32)

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
    (idx_in, idx_k), (idx_kn, idx_i) = get_distinct_triplet_indices(distinct_pair_idx_i, distinct_pair_idx_n, n_el, n_distinct_triplets)
    trip_idx_in = trip_idx_in.at[s].set(idx_in + n_el)
    trip_idx_k = trip_idx_k.at[s].set(idx_k)
    trip_idx_kn = trip_idx_kn.at[s].set(idx_kn + n_el)
    trip_idx_i = trip_idx_i.at[s].set(idx_i)
    return trip_idx_in, trip_idx_k, trip_idx_kn, trip_idx_i



# Generate random electrons, which have ~ 1mio triplets
n_el = 20
n_up = n_el // 2
cutoff = 2.5

key = jax.random.PRNGKey(0)
r = jax.random.normal(key, [n_el, 3])
r = r.at[:n_up, 0].add(np.arange(n_up) * 0.1)
r = r.at[n_up:, 0].add(np.arange(n_up) * 0.1)
n_pairs_same, n_pairs_diff, n_triplets = jtu.tree_map(int, get_static_args(r, cutoff, n_up))
print(f"{n_pairs_same=}, {n_pairs_diff=}, {n_triplets=}")
_, _, (pair_idx_i, pair_idx_n) = get_pair_indices(r, n_up, cutoff, n_pairs_same, n_pairs_diff)
#%%

indices_old = get_triplet_indices_old(pair_idx_i, pair_idx_n, n_el, n_triplets)
indices_new = get_triplet_indices(pair_idx_i, pair_idx_n, n_el, n_triplets)

# print("Pairs")
# for i, n in zip(pair_idx_i, pair_idx_n):
#     print(f"i: {int(i)}, n: {int(n)}")

# print("Triplets (new)")
# for p1, k, p2, i in zip(*indices_new):
#     p1, k, p2, i = jtu.tree_map(int, (p1, k, p2, i))
#     s = f"p1: {p1:3d} (i={int(pair_idx_i[p1])}, n={int(pair_idx_n[p1])}, k={int(k)}); "
#     s += f"p2: {p2:3d} (k={int(pair_idx_i[p2])}, n={int(pair_idx_n[p2])}, i={int(i)})"
#     print(s)


def to_index_set(indices):
    indices = [tuple([int(i) for i in idx]) for idx in zip(*indices)]
    return set(indices)

indices_old_set = to_index_set(indices_old)
indices_new_set = to_index_set(indices_new)
print("n_el               : ", n_el)
print("2 * n_pairs        : ", 2 * (n_pairs_same + n_pairs_diff))
print("Old triplet indices: ", len(indices_old_set))
print("New triplet indices: ", len(indices_new_set))
print("Overlap:             ", len(indices_old_set & indices_new_set))





