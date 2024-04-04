# %%
import numpy as np
import jax.numpy as jnp
import jax
import functools
from collections import namedtuple
from utils import multi_vmap, vmap_batch_dims

NO_NEIGHBOUR = 1_000_000

NeighbourIndicies = namedtuple("NeighbourIndices", ["ee", "en", "ne"])


@functools.partial(jax.jit, static_argnums=(3,))
def _get_cutoff_matrix(r1, r2, cutoff, include_self=False):
    is_self_interaction = r2 is None
    if is_self_interaction:
        r2 = r1
    dist = jnp.linalg.norm(r1[..., :, None, :] - r2[..., None, :, :], axis=-1)
    if is_self_interaction and (not include_self):
        dist += np.inf * jnp.eye(r1.shape[-2])
    in_cutoff = dist < cutoff
    max_n_neighbours = jnp.max(jnp.sum(in_cutoff, axis=-1))
    return in_cutoff, max_n_neighbours


# vmap over total number of electrons
@functools.partial(vmap_batch_dims, nr_non_batch_dims=(1, None), in_axes=(0, None))
def _get_ind_neighbours(in_cutoff, n_max):
    indices = jnp.nonzero(in_cutoff, size=n_max, fill_value=NO_NEIGHBOUR)[0]
    indices = jnp.unique(indices, size=n_max, fill_value=NO_NEIGHBOUR)
    return indices


def get_ind_neighbours(r1, r2, cutoff, include_self=False):
    in_cutoff, n_neighbours = _get_cutoff_matrix(r1, r2, cutoff, include_self)
    ind_neighbours = jax.jit(lambda co: _get_ind_neighbours(co, int(n_neighbours)))(in_cutoff)
    return ind_neighbours


def get_all_neighbour_indices(r, R, cutoff):
    return NeighbourIndicies(
        ee=get_ind_neighbours(r, None, cutoff), en=get_ind_neighbours(r, R, cutoff), ne=get_ind_neighbours(R, r, cutoff)
    )


def get_all_dependencies(idx_nb: NeighbourIndicies, n_deps_max_H_nuc, n_deps_max_h_out):
    """Get the indices of electrons on which each embedding will depend on.

    Args:
        idx_nb: NeighbourIndicies, named tuple containing the indices of the neighbours of each electron and nucleus.
        n_deps_max_H_nuc: int, maximum number of dependencies for the nuclear embeddings.
        n_deps_max_h_out: int, maximum number of dependencies for the output electron embeddings.

    Returns:
        deps: tuple of jnp.ndarray, dependencies for the electron embeddings at each step.
            deps_h0: [batch_size x n_el  x nr_of_deps_level_1]
            deps_H:  [batch_size x n_nuc x nr_of deps_level_2]
            deps_hout: [batch_size x n_el x nr_of_deps_level_3]
        dep_maps: tuple of jnp.ndarray, maps the dependencies between the levels:
            dep_map_h0_to_H: [batch_size x n_nuc x n_neighbouring_el x nr_of_deps_level_1]; values are in [0 ... deps_level_2]
            dep_map_H_to_hout: [batch_size x n_el x n_neighbouring_nuc x nr_of_deps_level_2]; values are in [0 ... deps_level_3]
    """
    n_el = idx_nb.ee.shape[-2]
    batch_dims = tuple(range(idx_nb.ee.ndim - 2))
    self_dependency = jnp.arange(n_el)[:, None]
    self_dependency = jnp.expand_dims(self_dependency, axis=batch_dims)

    @functools.partial(vmap_batch_dims, nr_non_batch_dims=(2, 2))  # vmap over all batch-dims
    def get_deps_nb(deps, idx_nb):
        return get_with_fill(deps, idx_nb)

    # Step 1: Initial electron embeddings depend on themselves and their neighbours
    deps_h0 = jnp.concatenate([self_dependency, idx_nb.ee], axis=-1)

    # Step 2: Nuclear embeddings depend on all dependencies of their neighbouring electrons
    deps_neighbours = get_deps_nb(deps_h0, idx_nb.ne)
    deps_H, dep_map_h0_to_H = merge_dependencies(deps_neighbours, idx_nb.ne, n_deps_max_H_nuc)

    # Step 3: Output electron embeddings depend on themselves, their neighbouring electrons and all dependencies of their neighbouring nuclei
    deps_neighbours = get_deps_nb(deps_H, idx_nb.en)
    deps_hout, dep_map_H_to_hout = merge_dependencies(deps_neighbours, deps_h0, n_deps_max_h_out)

    return (deps_h0, deps_H, deps_hout), (dep_map_h0_to_H, dep_map_H_to_hout)


def get_with_fill(arr, ind, fill=NO_NEIGHBOUR):
    return arr.at[ind].get(mode="fill", fill_value=fill)


def pad_n_neighbours(n, n_min=10, factor=1.2):
    power_padded = jnp.log(n) / jnp.log(factor)
    n_padded = jnp.maximum(n_min, factor ** jnp.ceil(power_padded))
    return n_padded.astype(jnp.int32)


def get_max_nr_of_dependencies(r, R, cutoff):
    # The nuclear embeddings are computed with 2 message passing steps and can therefore depend at most on electrons within 2 * cutoff
    dist_ne = jnp.linalg.norm(R[..., :, None, :] - r[..., None, :, :], axis=-1)
    n_deps_max_ne = jnp.max(jnp.sum(dist_ne < cutoff * 2, axis=-1))

    # The output electron embeddings are computed with 3 message passing step and can therefore depend at most on electrons within 3 * cutoff
    dist_ee = jnp.linalg.norm(r[..., :, None, :] - r[..., None, :, :], axis=-1)
    n_deps_max_h_out = jnp.max(jnp.sum(dist_ee < cutoff * 3, axis=-1))
    return n_deps_max_ne, n_deps_max_h_out


@functools.partial(jax.jit, static_argnums=(2,))
@functools.partial(vmap_batch_dims, nr_non_batch_dims=(2, 1, None), in_axes=(0, 0, None))
def merge_dependencies(deps, fixed_deps, n_deps_max):
    # Get maximum number of new dependencies after the merge
    n_new_deps_max = n_deps_max - len(fixed_deps)

    new_deps = jnp.where(jnp.isin(deps, fixed_deps), NO_NEIGHBOUR, deps)
    new_unique_deps = jnp.unique(new_deps, size=n_new_deps_max, fill_value=NO_NEIGHBOUR)
    deps_out = jnp.concatenate([fixed_deps, new_unique_deps], axis=-1)

    dep_map = multi_vmap(lambda d: jnp.argwhere(d == deps_out, size=1, fill_value=NO_NEIGHBOUR)[0][0], 2)(deps)
    # n_deps_max = jnp.max(jnp.sum(deps_out != NO_NEIGHBOUR, axis=-1))
    return deps_out, dep_map


if __name__ == "__main__":
    n_el = 50

    rng_r = jax.random.PRNGKey(0)
    R = jnp.arange(-n_el // 2, n_el // 2)[:, None] * jnp.array([1, 0, 0])
    r = jax.random.normal(rng_r, (n_el, 3)) + R

    n_dependencies = get_max_nr_of_dependencies(r, cutoff=3.0, n_steps_max=2)
    n_dependencies = [int(n) for n in n_dependencies]
    ind_neighbours = get_ind_neighbours(r, cutoff=4.0, include_self=False)
    ind_dep = jnp.concatenate([jnp.arange(n_el)[:, None], ind_neighbours], axis=-1)

    i = 1
    fixed_deps = ind_dep[i]
    deps = get_with_fill(ind_dep, ind_neighbours[i])
    deps_out, dep_map = merge_dependencies(deps, fixed_deps, n_dependencies[1])

    print("Fixed deps: ")
    print(fixed_deps)
    print("")
    print("Dependecies: ")
    print(deps)
    print("")
    print("Merged deps: ")
    print(deps_out)
    print("")
    print("Dep map: ")
    print(dep_map)
