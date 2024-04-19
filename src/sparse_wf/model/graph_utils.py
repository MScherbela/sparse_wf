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
import jax.tree_util as jtu

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
                n_particles = dist.shape[-1]
                dist += jnp.diag(jnp.inf * jnp.ones(n_particles))
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
        dist_ee += jnp.diag(jnp.inf * jnp.ones(n_el))
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
@functools.partial(jax.vmap, in_axes=(None, 0, None, 0), out_axes=-3)
def get_neighbour_with_FwdLapArray(h: FwdLaplArray, ind_neighbour, n_deps_out, dep_map):
    n_neighbour = ind_neighbour.shape[-1]
    n_features = h.shape[-1]

    # Get neighbour data by indexing into the input data and padding with 0 any out of bounds indices
    h_neighbour, jac_neighbour, lap_neighbour = jtu.tree_map(
        lambda x: x.at[..., ind_neighbour, :].get(mode="fill", fill_value=0), h
    )
    jac_neighbour = jac_neighbour.data

    # Remaining issue: The jacobians for each embedding can depend on different input coordinates
    # 1) Split jacobian input dim into electrons x xyz
    jac_neighbour = einops.rearrange(
        jac_neighbour,
        "(n_dep_in dim) n_neighbour features -> n_dep_in dim n_neighbour features",
        n_neighbour=n_neighbour,
        dim=3,
    )

    # 2) Combine the jacobians into a larger jacobian, that depends on the joint dependencies
    @functools.partial(jax.vmap, in_axes=(-2, -2), out_axes=-2)  # vmap over neighbours
    def _jac_for_neighbour(J, dep_map_):
        jac_out = jnp.zeros([n_deps_out, 3, n_features])
        jac_out = jac_out.at[dep_map_].set(J, mode="drop")
        return jac_out

    jac_neighbour = _jac_for_neighbour(jac_neighbour, dep_map)

    # 3) Merge electron and xyz dim back together to jacobian input dim
    jac_neighbour = einops.rearrange(
        jac_neighbour,
        "n_dep_out dim n_neighbour D -> (n_dep_out dim) n_neighbour D",
        n_dep_out=n_deps_out,
        dim=3,
        n_neighbour=n_neighbour,
    )
    return FwdLaplArray(h_neighbour, FwdJacobian(data=jac_neighbour), lap_neighbour)


def slogdet_with_sparse_fwd_lap(orbitals: FwdLaplArray, dependencies: Integer[Array, "el ndeps"]):
    n_el, n_orb = orbitals.x.shape[-2:]
    n_deps = dependencies.shape[-1]
    assert n_el == n_orb
    sign, logdet = jnp.linalg.slogdet(orbitals.x)  # IDEA: re-use LU decomposition of orbitals to accelerate this

    # solve to contract over orbital dim of jacobian; vmap over dependencies (incl. dim=3) and over electrons
    M = jax.vmap(jax.vmap(lambda J: jnp.linalg.solve(orbitals.x.T, J)))(orbitals.jacobian.data)
    M = M.reshape([n_deps, 3, n_el, n_el])  # split (deps * dim) into (deps, dim)

    # Get reverse dependency map D_tilde
    @jax.vmap  # vmap over centers
    @functools.partial(jax.vmap, in_axes=(None, 0))  # vmap over neighbours
    def get_reverse_dep(idx_center: Int, idx_nb: Int):
        return jnp.nonzero(dependencies[idx_nb, :] == idx_center, size=1, fill_value=NO_NEIGHBOUR)[0][0]

    reverse_deps = get_reverse_dep(jnp.arange(n_el), dependencies)

    M_hat = M.at[reverse_deps[:, :, None], :, dependencies[:, :, None], dependencies[:, None, :]].get(
        mode="fill", fill_value=0.0
    )
    assert M_hat.shape == (n_el, n_deps, n_deps, 3)

    jvp_lap = jnp.trace(jnp.linalg.solve(orbitals.x, orbitals.laplacian))
    jvp_jac = jnp.einsum("naad->nd", M_hat)
    tr_JHJ = -jnp.einsum("nabd,nbad", M_hat, M_hat)

    # TODO: properly pass on the sign
    return sign, FwdLaplArray(logdet, FwdJacobian(data=jvp_jac), jvp_lap + tr_JHJ)


def densify_jacobian_by_zero_padding(h: FwdLaplArray, n_deps_out):
    jac = h.jacobian.data
    n_deps_sparse = jac.shape[0]
    padding = jnp.zeros([n_deps_out - n_deps_sparse, *jac.shape[1:]])
    return FwdLaplArray(x=h.x, jacobian=FwdJacobian(jnp.concatenate([jac, padding], axis=0)), laplacian=h.laplacian)


def densify_jacobian_diagonally(h: FwdLaplArray):
    jac = h.jacobian.data
    dim, n_centers, n_neighbours, n_features = jac.shape
    assert dim == 3

    idx_neighbour = jnp.arange(n_neighbours)
    jac_out = jnp.zeros([n_neighbours, 3, n_centers, n_neighbours, n_features])
    jac_out = jac_out.at[idx_neighbour, :, :, idx_neighbour, :].set(
        jnp.moveaxis(jac, 2, 0)
    )  # move n_neighbours to the front
    assert jac_out.shape == (n_neighbours, 3, n_centers, n_neighbours, n_features)
    jac_out = jax.lax.collapse(jac_out, 0, 2)  # merge (n_deps, 3) into (n_deps*3)
    return FwdLaplArray(x=h.x, jacobian=FwdJacobian(jac_out), laplacian=h.laplacian)
