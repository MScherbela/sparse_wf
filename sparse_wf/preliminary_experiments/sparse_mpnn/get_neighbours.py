# %%
import numpy as np
import jax.numpy as jnp
import jax
import functools
from typing import NamedTuple
from utils import multi_vmap, vmap_batch_dims
from folx.api import FwdLaplArray, FwdJacobian
from sparse_wf.api import NeighbourIndices, NrOfDependencies, Electrons, Nuclei, Dependencies, DependencyMap
from jaxtyping import Shaped, Integer, Array, Scalar
import einops

NO_NEIGHBOUR = 1_000_000

class NrOfDependenciesMoon(NrOfDependencies):
    h0: int
    Hnuc: int
    hout: int

    def pad(self, factor=1.2, n_min=8):
        return NrOfDependenciesMoon(*[int(round_to_next_step(n, factor, n_min)) for n in self])


def round_to_next_step(n: int, factor: float, n_min: int):
    power_padded = jnp.log(n) / jnp.log(factor)
    n_padded = jnp.maximum(n_min, factor ** jnp.ceil(power_padded))
    return n_padded.astype(jnp.int32)


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
@functools.partial(jax.jit, static_argnums=(1,))
@functools.partial(vmap_batch_dims, nr_non_batch_dims=(1, None), in_axes=(0, None))
def _get_ind_neighbours(in_cutoff, n_max):
    indices = jnp.nonzero(in_cutoff, size=n_max, fill_value=NO_NEIGHBOUR)[0]
    indices = jnp.unique(indices, size=n_max, fill_value=NO_NEIGHBOUR)
    return indices


def get_ind_neighbours(r1, r2, cutoff, include_self=False):
    in_cutoff, n_neighbours = _get_cutoff_matrix(r1, r2, cutoff, include_self) # jitted
    n_neighbours = int(n_neighbours) # non-jitted
    ind_neighbours = _get_ind_neighbours(in_cutoff, n_neighbours) # jitted
    return ind_neighbours


def get_all_neighbour_indices(r: Electrons, R:Nuclei, cutoff: float):
    return NeighbourIndices(
        ee=get_ind_neighbours(r, None, cutoff), 
        en=get_ind_neighbours(r, R, cutoff), 
        ne=get_ind_neighbours(R, r, cutoff)
    )


def get_all_dependencies(idx_nb: NeighbourIndices, n_deps_max: NrOfDependenciesMoon) -> tuple[tuple[Dependencies, ...], tuple[DependencyMap, ...]]:
    """Get the indices of electrons on which each embedding will depend on.

    Args:
        idx_nb: NeighbourIndices, named tuple containing the indices of the neighbours of each electron and nucleus.
        n_deps_max: maximum_nr_of electrons that each embedding can depend on.
            - n_deps_max[0]: maximum number of dependencies for the electron embeddings at the first step.
            - n_deps_max[1]: maximum number of dependencies for the nuclear embeddings.
            - n_deps_max[2]: maximum number of dependencies for the output electron embeddings.

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
    batch_dims = idx_nb.ee.shape[:-2]
    self_dependency = jnp.arange(n_el)[:, None]
    self_dependency = jnp.tile(self_dependency, batch_dims + (1, 1))

    @functools.partial(vmap_batch_dims, nr_non_batch_dims=(2, 2))  # vmap over all batch-dims
    def get_deps_nb(deps, idx_nb):
        return get_with_fill(deps, idx_nb)

    # Step 1: Initial electron embeddings depend on themselves and their neighbours
    deps_h0: Dependencies = jnp.concatenate([self_dependency, idx_nb.ee], axis=-1)

    # Step 2: Nuclear embeddings depend on all dependencies of their neighbouring electrons
    deps_neighbours = get_deps_nb(deps_h0, idx_nb.ne)
    deps_H, dep_map_h0_to_H = merge_dependencies(deps_neighbours, idx_nb.ne, n_deps_max[1])

    # Step 3: Output electron embeddings depend on themselves, their neighbouring electrons and all dependencies of their neighbouring nuclei
    deps_neighbours = get_deps_nb(deps_H, idx_nb.en)
    deps_hout, dep_map_H_to_hout = merge_dependencies(deps_neighbours, deps_h0, n_deps_max[2])

    return (deps_h0, deps_H, deps_hout), (dep_map_h0_to_H, dep_map_H_to_hout)


def get_with_fill(arr: Shaped[Array, "n_elements *feature_dims"], 
                  ind: Integer[Array, "*batch_dims n_neighbours"], 
                  fill: float|int=NO_NEIGHBOUR) -> Shaped[Array, "*batch_dims n_neighbours *feature_dims"]:
    return arr.at[ind].get(mode="fill", fill_value=fill)


def get_max_nr_of_dependencies(r: Electrons, R: Nuclei, cutoff: float):
    dist_ee = jnp.linalg.norm(r[..., :, None, :] - r[..., None, :, :], axis=-1)
    dist_ne = jnp.linalg.norm(R[..., :, None, :] - r[..., None, :, :], axis=-1)

    # Thest first electron message passing step can depend at most on electrons within 1 * cutoff
    n_deps_max_h0 = jnp.max(jnp.sum(dist_ee < cutoff, axis=-1))

    # The nuclear embeddings are computed with 2 message passing steps and can therefore depend at most on electrons within 2 * cutoff
    n_deps_max_H = jnp.max(jnp.sum(dist_ne < cutoff * 2, axis=-1))

    # The output electron embeddings are computed with 3 message passing step and can therefore depend at most on electrons within 3 * cutoff
    n_deps_max_h_out = jnp.max(jnp.sum(dist_ee < cutoff * 3, axis=-1))
    return NrOfDependenciesMoon(int(n_deps_max_h0), int(n_deps_max_H), int(n_deps_max_h_out))


@functools.partial(jax.jit, static_argnums=(2,))
@functools.partial(vmap_batch_dims, nr_non_batch_dims=(2, 1, None), in_axes=(0, 0, None))
def merge_dependencies(deps: Dependencies, fixed_deps: Dependencies, n_deps_max: int) -> tuple[Dependencies, DependencyMap]:
    # Get maximum number of new dependencies after the merge
    n_new_deps_max = n_deps_max - len(fixed_deps)

    new_deps = jnp.where(jnp.isin(deps, fixed_deps), NO_NEIGHBOUR, deps) # TODO (ng): How to avoid this type error
    new_unique_deps = jnp.unique(new_deps, size=n_new_deps_max, fill_value=NO_NEIGHBOUR)
    deps_out = jnp.concatenate([fixed_deps, new_unique_deps], axis=-1)

    dep_map = multi_vmap(lambda d: jnp.argwhere(d == deps_out, size=1, fill_value=NO_NEIGHBOUR)[0][0], 2)(deps)
    # n_deps_max = jnp.max(jnp.sum(deps_out != NO_NEIGHBOUR, axis=-1))
    return deps_out, dep_map


# vmap over center particle
@functools.partial(jax.vmap, in_axes=(None, 0, None, 0))
def get_neighbour_with_FwdLapArray(h: FwdLaplArray, ind_neighbour, n_deps_out, dep_map):
    # Get and assert shapes
    n_neighbour = ind_neighbour.shape[-1]
    feature_dims = h.x.shape[1:]

    # Get neighbour data by indexing into the input data and padding with 0 any out of bounds indices
    h_neighbour = get_with_fill(h.x, ind_neighbour, 0.0)
    jac_neighbour = get_with_fill(h.jacobian.data, ind_neighbour, 0.0)
    lap_h_neighbour = get_with_fill(h.laplacian, ind_neighbour, 0.0)

    # Remaining issue: The jacobians for each embedding can depend on different input coordinates
    # 1) Split jacobian input dim into electrons x xyz
    jac_neighbour = einops.rearrange(
        jac_neighbour,
        "n_neighbour (n_dep_in dim) D -> n_neighbour n_dep_in dim D",
        n_neighbour=n_neighbour,
        dim=3,
    )

    # 2) Combine the jacobians into a larger jacobian, that depends on the joint dependencies
    @functools.partial(jax.vmap, in_axes=(0, 0), out_axes=2)
    def _jac_for_neighbour(J, dep_map_):
        jac_out = jnp.zeros([n_deps_out, 3, *feature_dims])
        jac_out = jac_out.at[dep_map_].set(J, mode="drop")
        return jac_out

    # total_neighbors_t
    jac_neighbour = _jac_for_neighbour(jac_neighbour, dep_map)

    # 3) Merge electron and xyz dim back together to jacobian input dim
    jac_neighbour = einops.rearrange(
        jac_neighbour,
        "n_dep_out dim n_neighbour D -> (n_dep_out dim) n_neighbour D",
        n_dep_out=n_deps_out,
        dim=3,
        n_neighbour=n_neighbour,
    )
    return FwdLaplArray(h_neighbour, FwdJacobian(data=jac_neighbour), lap_h_neighbour)

