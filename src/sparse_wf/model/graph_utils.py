from jaxtyping import Shaped, Integer, Array
from sparse_wf.api import (
    Dependencies,
    DependencyMap,
    InputConstructor,
    Nuclei,
    Charges,
    Electrons,
    StaticInput,
    DynamicInput,
    NeighbourIndices,
    DistanceMatrix,
    Int,
    NrOfNeighbours,
)
import functools
import jax.numpy as jnp
import jax
from folx.api import FwdLaplArray, FwdJacobian
import einops
from sparse_wf.jax_utils import jit
import numpy as np

NO_NEIGHBOUR = 1_000_000


class GenericInputConstructor(InputConstructor):
    def __init__(self, R: Nuclei, Z: Charges, cutoff: float, padding_factor: float = 1.0, n_neighbours_min: int = 0):
        self.R = R
        self.Z = Z
        self.cutoff = cutoff
        self.padding_factor = padding_factor
        self.n_neighbours_min = n_neighbours_min

    @jit(static_argnames=("self", "static"))
    def get_dynamic_input(self, electrons: Electrons, static: StaticInput) -> DynamicInput:
        dist_ee, dist_ne = self.get_full_distance_matrices(electrons)
        return DynamicInput(
            electrons=electrons, neighbours=self.get_neighbour_indices(dist_ee, dist_ne, static.n_neighbours)
        )

    @jit(static_argnames=("self", "n_neighbours"))
    def get_neighbour_indices(
        self, dist_ee: DistanceMatrix, dist_ne: DistanceMatrix, n_neighbours: NrOfNeighbours
    ) -> NeighbourIndices:
        def _get_ind_neighbour(dist, max_n_neighbours: int, exclude_diagonal=False):
            if exclude_diagonal:
                dist += jnp.diag(jnp.ones(dist.shape[-1])*jnp.inf)
            in_cutoff = dist < self.cutoff

            # TODO: dynamically assert that n_neighbours <= max_n_neighbours
            n_neighbours = jnp.max(jnp.sum(in_cutoff, axis=-1))  # noqa: F841

            @jax.vmap
            def _get_ind(in_cutoff_):
                indices = jnp.nonzero(in_cutoff_, size=max_n_neighbours, fill_value=NO_NEIGHBOUR)[0]
                return jnp.unique(indices, size=max_n_neighbours, fill_value=NO_NEIGHBOUR)

            return _get_ind(in_cutoff)

        return NeighbourIndices(
            ee=_get_ind_neighbour(dist_ee, n_neighbours.ee, exclude_diagonal=True),
            ne=_get_ind_neighbour(dist_ne, n_neighbours.ne),
            en=_get_ind_neighbour(dist_ne.T, n_neighbours.en),
        )

    def get_nr_of_neighbours(self, dist_ee: DistanceMatrix, dist_ne: DistanceMatrix) -> NrOfNeighbours:
        n_ne, n_en, n_ee = self._get_max_n_neighbours(dist_ee, dist_ne)
        return NrOfNeighbours(
            ee=self._round_to_next_step(n_ee), en=self._round_to_next_step(n_en), ne=self._round_to_next_step(n_ne)
        )

    @jit(static_argnames="self")
    def get_full_distance_matrices(self, r: Electrons) -> tuple[DistanceMatrix, DistanceMatrix]:
        dist_ee = jnp.linalg.norm(r[..., :, None, :] - r[..., None, :, :], axis=-1)
        dist_ne = jnp.linalg.norm(self.R[..., :, None, :] - r[..., None, :, :], axis=-1)
        return dist_ee, dist_ne

    @jit(static_argnames="self")
    def _get_max_n_neighbours(self, dist_ee: DistanceMatrix, dist_ne: DistanceMatrix):
        n_el = dist_ee.shape[-1]
        dist_ee += jnp.diag(jnp.ones(n_el)*jnp.inf)
        n_ee = jnp.max(jnp.sum(dist_ee < self.cutoff, axis=-1))
        n_ne = jnp.max(jnp.sum(dist_ne < self.cutoff, axis=-1))
        n_en = jnp.max(jnp.sum(dist_ne < self.cutoff, axis=-2))
        return n_ne, n_en, n_ee

    def _round_to_next_step(self, n: int | Int) -> int:
        if self.padding_factor == 1.0:
            return int(jnp.maximum(n, self.n_neighbours_min))
        else:
            power_padded = jnp.log(n) / jnp.log(self.padding_factor)
            n_padded = jnp.maximum(self.n_neighbours_min, self.padding_factor ** jnp.ceil(power_padded))
        return int(n_padded)


def get_with_fill(
    arr: Shaped[Array, "n_elements feature_dim"] | Shaped[np.ndarray, " n_elements feature_dim"],
    ind: Integer[Array, "*batch_dims n_neighbours"],
    fill: float | int,
) -> Shaped[Array, "*batch_dims n_neighbours feature_dim"]:
    return jnp.asarray(arr).at[ind].get(mode="fill", fill_value=fill)


# @jit(static_argnums=(2,))
@functools.partial(
    jnp.vectorize, excluded=(2,), signature="(element,deps_new),(deps_old)->(deps_out),(element,deps_new)"
)
def merge_dependencies(
    deps: Dependencies, fixed_deps: Dependencies, n_deps_max: int
) -> tuple[Dependencies, DependencyMap]:
    # Get maximum number of new dependencies after the merge
    n_new_deps_max = n_deps_max - len(fixed_deps)

    new_deps = jnp.where(jnp.isin(deps, fixed_deps), NO_NEIGHBOUR, deps)
    new_unique_deps = jnp.unique(new_deps, size=n_new_deps_max, fill_value=NO_NEIGHBOUR)
    deps_out = jnp.concatenate([fixed_deps, new_unique_deps], axis=-1)

    @jax.vmap
    @jax.vmap
    def get_dep_map(d):
        return jnp.argwhere(d == deps_out, size=1, fill_value=NO_NEIGHBOUR)[0][0]

    dep_map = get_dep_map(deps)
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


@functools.partial(jax.vmap, in_axes=(0, None))
def densify_jacobian_by_zero_padding(h: FwdLaplArray, n_deps_out):
    jac = h.jacobian.data
    n_deps_sparse = jac.shape[0]
    padding = jnp.zeros([n_deps_out - n_deps_sparse, *jac.shape[1:]])
    return FwdLaplArray(x=h.x, jacobian=FwdJacobian(jnp.concatenate([jac, padding], axis=0)), laplacian=h.laplacian)


@jax.vmap
def densify_jacobian_diagonally(h: FwdLaplArray):
    jac = h.jacobian.data
    assert jac.shape[0] == 3
    n_neighbours = jac.shape[1]

    idx_neighbour = jnp.arange(n_neighbours)
    jac_out = jnp.zeros([n_neighbours, 3, n_neighbours, *jac.shape[2:]])
    jac_out = jac_out.at[idx_neighbour, :, idx_neighbour, ...].set(jac.swapaxes(0, 1))
    jac_out = jax.lax.collapse(jac_out, 0, 2)  # merge (n_deps, 3) into (n_deps*3)
    return FwdLaplArray(x=h.x, jacobian=FwdJacobian(jac_out), laplacian=h.laplacian)
