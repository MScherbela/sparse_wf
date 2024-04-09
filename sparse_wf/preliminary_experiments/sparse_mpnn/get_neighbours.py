# %%
import numpy as np
import jax.numpy as jnp
import jax
import functools
from typing import NamedTuple
from utils import multi_vmap
from folx.api import FwdLaplArray, FwdJacobian
from sparse_wf.api import NrOfNeighbours, NeighbourIndices, NrOfDependencies, Electrons, Nuclei, Charges, Dependencies, DependencyMap, InputConstructor, StaticInput, DynamicInput, DynamicInputWithDependencies
from jaxtyping import Shaped, Integer, Array, Int, Float
import einops
from sparse_wf.jax_utils import jit
from typing import cast

NO_NEIGHBOUR = 1_000_000

DistanceMatrix = Float[Array, "*batch_dims n1 n2"]


class DependenciesMoon(NamedTuple):
    h0: Dependencies
    Hnuc: Dependencies
    hout: Dependencies

class NrOfDependenciesMoon(NrOfDependencies):
    h0: int
    Hnuc: int
    hout: int

class DependencyMapsMoon(NamedTuple):
    h0_to_H: DependencyMap
    H_to_hout: DependencyMap


def get_with_fill(arr: Shaped[Array, "n_elements *feature_dims"], # noqa: F821
                  ind: Integer[Array, "*batch_dims n_neighbours"], 
                  fill: float|int) -> Shaped[Array, "*batch_dims n_neighbours *feature_dims"]:
    return arr.at[ind].get(mode="fill", fill_value=fill)


class InputConstructorMoon(InputConstructor):
    def __init__(self, R: Nuclei, Z: Charges, cutoff: float, padding_factor: float = 1.0, n_neighbours_min: int = 0):
        self.R = R
        self.Z = Z
        self.cutoff = cutoff
        self.padding_factor = padding_factor
        self.n_neighbours_min = n_neighbours_min

    # This function cannot be jitted, because it returns a static tuple of integers
    def get_static_input(self, electrons: Electrons) -> StaticInput:
        dist_ee, dist_ne = self._get_distance_matrices(electrons)
        n_ne, n_en, n_ee = self._get_max_n_neighbours(dist_ee, dist_ne)
        n_deps_h0, n_deps_H, n_deps_hout = self._get_max_nr_of_dependencies(dist_ee, dist_ne)

        n_neighbours = NrOfNeighbours(ee=self._round_to_next_step(n_ee), 
                                      en=self._round_to_next_step(n_en), 
                                      ne=self._round_to_next_step(n_ne))
        n_deps = NrOfDependenciesMoon(h0=self._round_to_next_step(n_deps_h0),
                                      Hnuc=self._round_to_next_step(n_deps_H),
                                      hout=self._round_to_next_step(n_deps_hout))
        return StaticInput(n_neighbours=n_neighbours, n_deps=n_deps)

    @jit
    def get_dynamic_input(self, electrons: Electrons, static: StaticInput) -> DynamicInput:
        dist_ee, dist_ne = self._get_distance_matrices(electrons)
        idx_nb = NeighbourIndices(
            ee=self._get_ind_neighbour(dist_ee, static.n_neighbours.ee, exclude_diagonal=True), 
            ne=self._get_ind_neighbour(dist_ne, static.n_neighbours.ne),
            en=self._get_ind_neighbour(dist_ne.T, static.n_neighbours.en), 
        )
        return DynamicInput(electrons=electrons, neighbours=idx_nb)

    @jit
    def get_dynamic_input_with_dependencies(self, electrons: Electrons, static: StaticInput) -> DynamicInputWithDependencies:
        # Indices of neighbours
        dist_ee, dist_ne = self._get_distance_matrices(electrons)
        idx_nb = NeighbourIndices(
            ee=self._get_ind_neighbour(dist_ee, static.n_neighbours.ee, exclude_diagonal=True), 
            ne=self._get_ind_neighbour(dist_ne, static.n_neighbours.ne),
            en=self._get_ind_neighbour(dist_ne.T, static.n_neighbours.en), 
        )

        # Dependencies of embedddings
        deps, dep_maps = self._get_all_dependencies(idx_nb, cast(NrOfDependenciesMoon, static.n_deps))
        return DynamicInputWithDependencies(electrons=electrons, neighbours=idx_nb, dependencies=deps, dep_maps=dep_maps)

    @jit
    def _get_distance_matrices(self, r: Electrons) -> tuple[DistanceMatrix, DistanceMatrix]:
        dist_ee = jnp.linalg.norm(r[..., :, None, :] - r[..., None, :, :], axis=-1)
        dist_ne = jnp.linalg.norm(self.R[..., :, None, :] - r[..., None, :, :], axis=-1)
        return dist_ee, dist_ne

    @jit
    def _get_max_n_neighbours(self, dist_ee: DistanceMatrix, dist_ne: DistanceMatrix):
        n_el = dist_ee.shape[-1]
        dist_ee += np.inf * jnp.eye(n_el)
        n_ee = jnp.max(jnp.sum(dist_ee < self.cutoff, axis=-1))
        n_ne = jnp.max(jnp.sum(dist_ne < self.cutoff, axis=-1))
        n_en = jnp.max(jnp.sum(dist_ne < self.cutoff, axis=-2))
        return n_ne, n_en, n_ee
    
    @jit
    def _get_max_nr_of_dependencies(self, dist_ee: DistanceMatrix, dist_ne: DistanceMatrix):
        # Thest first electron message passing step can depend at most on electrons within 1 * cutoff
        n_deps_max_h0 = jnp.max(jnp.sum(dist_ee < self.cutoff, axis=-1))

        # The nuclear embeddings are computed with 2 message passing steps and can therefore depend at most on electrons within 2 * cutoff
        n_deps_max_H = jnp.max(jnp.sum(dist_ne < self.cutoff * 2, axis=-1))

        # The output electron embeddings are computed with 3 message passing step and can therefore depend at most on electrons within 3 * cutoff
        n_deps_max_h_out = jnp.max(jnp.sum(dist_ee < self.cutoff * 3, axis=-1))
        return n_deps_max_h0, n_deps_max_H, n_deps_max_h_out
    

    def _get_ind_neighbour(self, dist, max_n_neighbours: int, exclude_diagonal=False):
        if exclude_diagonal:
            dist += np.inf * jnp.eye(dist.shape[-1])
        in_cutoff = dist < self.cutoff
        n_neighbours = jnp.max(jnp.sum(in_cutoff, axis=-1))
        # TODO: dynamically assert that n_neighbours <= max_n_neighbours

        @jax.vmap
        @jax.vmap
        def _get_ind(in_cutoff_):
            indices = jnp.nonzero(in_cutoff_, size=max_n_neighbours, fill_value=NO_NEIGHBOUR)[0]
            return jnp.unique(indices, size=max_n_neighbours, fill_value=NO_NEIGHBOUR)

        return _get_ind(in_cutoff)
    
    def _round_to_next_step(self, n: int|Int) -> int:
        power_padded = jnp.log(n) / jnp.log(self.padding_factor)
        n_padded = jnp.maximum(self.n_neighbours_min, self.padding_factor ** jnp.ceil(power_padded))
        return int(n_padded)


    def _get_all_dependencies(self, idx_nb: NeighbourIndices, n_deps_max: NrOfDependenciesMoon) -> tuple[tuple[Dependencies, ...], tuple[DependencyMap, ...]]:
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

        @functools.partial(jnp.vectorize, signature="(center1,deps),(center2,neigbours)->(center2,neigbours,deps)")
        def get_deps_nb(deps, idx_nb):
            return get_with_fill(deps, idx_nb, NO_NEIGHBOUR)

        # Step 1: Initial electron embeddings depend on themselves and their neighbours
        deps_h0: Dependencies = jnp.concatenate([self_dependency, idx_nb.ee], axis=-1)

        # Step 2: Nuclear embeddings depend on all dependencies of their neighbouring electrons
        deps_neighbours = get_deps_nb(deps_h0, idx_nb.ne)
        deps_H, dep_map_h0_to_H = merge_dependencies(deps_neighbours, idx_nb.ne, n_deps_max.Hnuc)

        # Step 3: Output electron embeddings depend on themselves, their neighbouring electrons and all dependencies of their neighbouring nuclei
        deps_neighbours = get_deps_nb(deps_H, idx_nb.en)
        deps_hout, dep_map_H_to_hout = merge_dependencies(deps_neighbours, deps_h0, n_deps_max.hout)

        return DependenciesMoon(deps_h0, deps_H, deps_hout), DependencyMapsMoon(dep_map_h0_to_H, dep_map_H_to_hout)



# @functools.partial(jit, static_argnums=(2,))
@functools.partial(jnp.vectorize, excluded=(2,), signature="(element,deps_new),(deps_old)->(deps_out),(element,deps_new)")
def merge_dependencies(deps: Dependencies, fixed_deps: Dependencies, n_deps_max: int) -> tuple[Dependencies, DependencyMap]:
    # Get maximum number of new dependencies after the merge
    n_new_deps_max = n_deps_max - len(fixed_deps)

    new_deps = jnp.where(jnp.isin(deps, fixed_deps), NO_NEIGHBOUR, deps) # TODO (ng): How to avoid this type error
    new_unique_deps = jnp.unique(new_deps, size=n_new_deps_max, fill_value=NO_NEIGHBOUR)
    deps_out = jnp.concatenate([fixed_deps, new_unique_deps], axis=-1)

    dep_map = multi_vmap(lambda d: jnp.argwhere(d == deps_out, size=1, fill_value=NO_NEIGHBOUR)[0][0], 2)(deps)
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

