from jaxtyping import Shaped, Integer, Array, Float
from sparse_wf.api import (
    Dependencies,
    DependencyMap,
    Dependants,
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
from sparse_wf.jax_utils import jit
import numpy as np

NO_NEIGHBOUR = 1_000_000


class GenericInputConstructor(InputConstructor):
    def __init__(
        self, R: Nuclei, Z: Charges, n_el: int, cutoff: float, padding_factor: float = 1.0, n_neighbours_min: int = 0
    ):
        self.R = R
        self.Z = Z
        self.n_el = n_el
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
                dist += jnp.diag(jnp.inf * jnp.ones(n_particles, dist.dtype))
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
        dist_ee += jnp.diag(jnp.ones(n_el, dist_ee.dtype) * jnp.inf)
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
        return min(int(n_padded), self.n_el)


def get_with_fill(
    arr: Shaped[Array, "n_elements feature_dim"] | Shaped[np.ndarray, " n_elements feature_dim"],
    ind: Integer[Array, "*batch_dims n_neighbours"],
    fill: float | int,
) -> Shaped[Array, "*batch_dims n_neighbours feature_dim"]:
    return jnp.asarray(arr).at[ind].get(mode="fill", fill_value=fill)


# @functools.partial(jnp.vectorize, excluded=(3,), signature="(n_nb,deps_nb),(deps_center),(deps_frozen)->(deps_out)")
@functools.partial(jax.vmap, in_axes=(0, 0, 0, None), out_axes=0)
def merge_dependencies(deps_nb, deps_center, deps_frozen, n_deps_max: int) -> Dependencies:
    if deps_frozen is None:
        deps_out = jnp.unique(
            jnp.concatenate([deps_nb.flatten(), deps_center]), size=n_deps_max, fill_value=NO_NEIGHBOUR
        )
    else:
        new_deps = jnp.concatenate([deps_nb.flatten(), deps_center])
        new_deps = jnp.where(jnp.isin(new_deps, deps_frozen), NO_NEIGHBOUR, new_deps)
        new_deps = jnp.unique(new_deps, size=n_deps_max - len(deps_frozen), fill_value=NO_NEIGHBOUR)
        deps_out = jnp.concatenate([deps_frozen, new_deps])
    return deps_out


def get_dependency_map(deps_in: Dependencies, deps_out: Dependencies) -> DependencyMap:
    @jax.vmap
    def _get_dep_map(d_in):
        mapping = jnp.nonzero(d_in == deps_out, size=1, fill_value=NO_NEIGHBOUR)[0][0]
        mapping = jnp.where(d_in == NO_NEIGHBOUR, NO_NEIGHBOUR, mapping)
        return mapping

    return _get_dep_map(deps_in)


def get_dependants(dependencies: Dependencies, n_dependants_max: int) -> Dependants:
    n_el = dependencies.shape[0]
    idx_dependency = jnp.arange(n_el)
    dependants = jax.vmap(lambda n: jnp.where(dependencies == n, size=n_dependants_max, fill_value=NO_NEIGHBOUR)[0])(
        idx_dependency
    )
    return dependants


def _merge_xyz_dim(jac):
    assert jac.shape[1] == 3
    return jac.reshape([jac.shape[0] * 3, *jac.shape[2:]])


def _split_off_xyz_dim(jac):
    assert jac.shape[0] % 3 == 0
    return jac.reshape([jac.shape[0] // 3, 3, *jac.shape[1:]])


def pad_jacobian_to_output_deps(x: FwdLaplArray, dep_map: Integer[Array, " deps"], n_deps_out: int) -> FwdLaplArray:
    jac: Float[Array, "deps*3 features"] = x.jacobian.data
    n_features = jac.shape[-1]
    jac = _split_off_xyz_dim(jac)
    jac_out = jnp.zeros([n_deps_out, 3, n_features], jac.dtype)
    jac_out = jac_out.at[dep_map, :, :].set(jac, mode="drop")
    jac_out = _merge_xyz_dim(jac_out)
    return FwdLaplArray(x=x.x, jacobian=FwdJacobian(jac_out), laplacian=x.laplacian)


def get_inverse_from_lu(lu, permutation):
    n = lu.shape[0]
    b = jnp.eye(n, dtype=lu.dtype)[permutation]
    x = jax.lax.linalg.triangular_solve(lu, b, left_side=True, lower=True, unit_diagonal=True)
    x = jax.lax.linalg.triangular_solve(lu, x, left_side=True, lower=False)
    return x


def slogdet_from_lu(lu, pivot):
    assert (lu.ndim == 2) and (lu.shape[0] == lu.shape[1])
    n = lu.shape[0]
    diag = jnp.diag(lu)
    logdet = jnp.sum(jnp.log(jnp.abs(diag)))
    parity = jnp.count_nonzero(pivot != jnp.arange(n))  # sign flip for each permutation
    parity += jnp.count_nonzero(diag < 0)  # sign flip for each negative diagonal element
    sign = jnp.where(parity % 2 == 0, 1.0, -1.0)
    return sign, logdet


def slogdet_with_sparse_fwd_lap(orbitals: FwdLaplArray, dependencies: Integer[Array, "el ndeps"]):
    n_el, n_orb = orbitals.x.shape[-2:]
    n_deps = dependencies.shape[-1]
    assert n_el == n_orb

    orbitals_lu, orbitals_pivot, orbitals_permutation = jax.lax.linalg.lu(orbitals.x)
    orbitals_inv = get_inverse_from_lu(orbitals_lu, orbitals_permutation)
    sign, logdet = slogdet_from_lu(orbitals_lu, orbitals_pivot)

    M = orbitals.jacobian.data @ orbitals_inv
    M = M.reshape([n_deps, 3, n_el, n_el])  # split (deps * dim) into (deps, dim)

    # TODO: n_deps_max != n_dependants_max in general
    dependants = get_dependants(dependencies, n_dependants_max=n_deps)

    # Get reverse dependency map D_tilde
    @jax.vmap  # vmap over centers
    @functools.partial(jax.vmap, in_axes=(0, None))  # vmap over neighbours
    def get_reverse_dep(idx_dependant: Int, idx_dependency: Int):
        deps_of_dependant = dependencies.at[idx_dependant, :].get(mode="fill", fill_value=NO_NEIGHBOUR)
        return jnp.nonzero(deps_of_dependant == idx_dependency, size=1, fill_value=NO_NEIGHBOUR)[0][0]

    cycle_map = get_reverse_dep(dependants, jnp.arange(n_el))

    M_hat = M.at[cycle_map[:, :, None], :, dependants[:, :, None], dependants[:, None, :]].get(
        mode="fill", fill_value=0.0
    )
    assert M_hat.shape == (n_el, n_deps, n_deps, 3)

    jvp_lap = jnp.trace(orbitals.laplacian @ orbitals_inv)
    jvp_jac = jnp.einsum("naad->nd", M_hat).reshape([n_el * 3])
    tr_JHJ = -jnp.einsum("nabd,nbad", M_hat, M_hat)

    return sign, FwdLaplArray(logdet, FwdJacobian(data=jvp_jac), jvp_lap + tr_JHJ)


def densify_jacobian_by_zero_padding(h: FwdLaplArray, n_deps_out):
    jac = h.jacobian.data
    n_deps_sparse = jac.shape[0]
    padding = jnp.zeros([n_deps_out - n_deps_sparse, *jac.shape[1:]], jac.dtype)
    return FwdLaplArray(x=h.x, jacobian=FwdJacobian(jnp.concatenate([jac, padding], axis=0)), laplacian=h.laplacian)
