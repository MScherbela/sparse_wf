import functools
from typing import NamedTuple, TypeAlias, Optional

import jax
import jax.numpy as jnp
import numpy as np
from folx.api import FwdJacobian, FwdLaplArray
from jaxtyping import Array, Float, Integer, Shaped

from sparse_wf.api import Electrons, Int, Nuclei, Spins
from sparse_wf.jax_utils import jit, vectorize, pmax_if_pmap
import jax.tree_util as jtu

NO_NEIGHBOUR = 1_000_000


DistanceMatrix: TypeAlias = Float[Array, "n1 n2"]
ElectronElectronEdges = Integer[Array, "n_electrons n_nb_ee"]
ElectronNucleiEdges = Integer[Array, "n_electrons n_nb_en"]
NucleiElectronEdges = Integer[Array, "n_nuclei n_nb_ne"]

Dependant = Integer[Array, "n_dependants"]
Dependency = Integer[Array, "n_deps"]
DependencyMap = Integer[Array, "n_center n_neighbour n_deps"]


class NrOfNeighbours(NamedTuple):
    ee: int
    en: int
    ne: int


class NeighbourIndices(NamedTuple):
    ee: ElectronElectronEdges
    en: ElectronNucleiEdges
    ne: NucleiElectronEdges


@jit
@vectorize(signature="(n,d),(m,d)->(n,n),(m,n)")
def get_full_distance_matrices(r: Electrons, R: Nuclei) -> tuple[DistanceMatrix, DistanceMatrix]:
    dist_ee = jnp.linalg.norm(r[:, None, :] - r[None, :, :], axis=-1)
    dist_ne = jnp.linalg.norm(R[:, None, :] - r[None, :, :], axis=-1)
    return dist_ee, dist_ne


@jit
def round_to_next_step(
    n: int | Int,
    padding_factor: float,
    n_neighbours_min: int,
    n_neighbours_max: int,
) -> Int:
    # jittable version of the following if statement:
    # if padding_factor == 1.0:
    pad_1_result = jnp.maximum(n, n_neighbours_min)
    # else:
    power_padded = jnp.log(n) / jnp.log(padding_factor)
    pad_else_result = jnp.maximum(n_neighbours_min, padding_factor ** jnp.ceil(power_padded))
    result = jnp.where(padding_factor == 1.0, pad_1_result, pad_else_result)
    return jnp.minimum(result, n_neighbours_max)


def get_nr_of_neighbours(
    dist_ee: DistanceMatrix,
    dist_ne: DistanceMatrix,
    cutoff: float,
):
    n_el = dist_ee.shape[-1]
    dist_ee += jnp.diag(jnp.ones(n_el, dist_ee.dtype) * jnp.inf)
    n_ee = pmax_if_pmap(jnp.max(jnp.sum(dist_ee < cutoff, axis=-1)))
    n_ne = pmax_if_pmap(jnp.max(jnp.sum(dist_ne < cutoff, axis=-1)))
    n_en = pmax_if_pmap(jnp.max(jnp.sum(dist_ne < cutoff, axis=-2)))
    return n_ne, n_en, n_ee


@jit(static_argnames=("n_neighbours", "cutoff_en", "cutoff_ee"))
def get_neighbour_indices(
    r: Electrons, R: Nuclei, n_neighbours: NrOfNeighbours, cutoff_en: float, cutoff_ee: Optional[float] = None
) -> NeighbourIndices:
    if cutoff_ee is None:
        cutoff_ee = cutoff_en
    dist_ee, dist_ne = get_full_distance_matrices(r, R)

    def _get_ind_neighbour(dist, max_n_neighbours: int, cutoff, exclude_diagonal=False):
        if exclude_diagonal:
            n_particles = dist.shape[-1]
            dist += jnp.diag(jnp.inf * jnp.ones(n_particles, dist.dtype))
        in_cutoff = dist < cutoff

        # TODO: dynamically assert that n_neighbours <= max_n_neighbours
        n_neighbours = jnp.max(jnp.sum(in_cutoff, axis=-1))  # noqa: F841

        @jax.vmap
        def _get_ind(in_cutoff_):
            indices = jnp.nonzero(in_cutoff_, size=max_n_neighbours, fill_value=NO_NEIGHBOUR)[0]
            return jnp.unique(indices, size=max_n_neighbours, fill_value=NO_NEIGHBOUR)

        return _get_ind(in_cutoff)

    return NeighbourIndices(
        ee=_get_ind_neighbour(dist_ee, n_neighbours.ee, cutoff_ee, exclude_diagonal=True),
        ne=_get_ind_neighbour(dist_ne, n_neighbours.ne, cutoff_en),
        en=_get_ind_neighbour(dist_ne.T, n_neighbours.en, cutoff_en),
    )


def get_with_fill(
    arr: Shaped[Array, "n_elements feature_dim"] | Shaped[np.ndarray, " n_elements feature_dim"],
    ind: Integer[Array, "*batch_dims n_neighbours"],
    fill: float | int,
) -> Shaped[Array, "*batch_dims n_neighbours feature_dim"]:
    return jnp.asarray(arr).at[ind].get(mode="fill", fill_value=fill)


def get_neighbour_features(h: FwdLaplArray, ind_neighbour: Integer[Array, "n_center n_neighbour"]) -> FwdLaplArray:
    return jtu.tree_map(lambda x: x.at[..., ind_neighbour, :].get(mode="fill", fill_value=0.0), h)


# @functools.partial(jnp.vectorize, excluded=(3,), signature="(n_nb,deps_nb),(deps_center),(deps_frozen)->(deps_out)")
@functools.partial(jax.vmap, in_axes=(0, 0, 0, None), out_axes=0)
def merge_dependencies(deps_nb, deps_center, deps_frozen, n_deps_max: int) -> Dependency:
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


def get_dependency_map(deps_in: Dependency, deps_out: Dependency) -> DependencyMap:
    @jax.vmap
    def _get_dep_map(d_in):
        mapping = jnp.nonzero(d_in == deps_out, size=1, fill_value=NO_NEIGHBOUR)[0][0]
        mapping = jnp.where(d_in == NO_NEIGHBOUR, NO_NEIGHBOUR, mapping)
        return mapping

    return _get_dep_map(deps_in)


def get_dependants(dependencies: Dependency, n_dependants_max: int) -> Dependant:
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


def _pad_jacobian_to_output_deps(x: FwdLaplArray, dep_map: Integer[Array, " deps"], n_deps_out: int) -> FwdLaplArray:
    jac: Float[Array, "deps*3 features"] = x.jacobian.data
    n_features = jac.shape[-1]
    jac = _split_off_xyz_dim(jac)
    jac_out = jnp.zeros([n_deps_out, 3, n_features], jac.dtype)
    jac_out = jac_out.at[dep_map, :, :].set(jac, mode="drop")
    jac_out = _merge_xyz_dim(jac_out)
    return FwdLaplArray(x=x.x, jacobian=FwdJacobian(jac_out), laplacian=x.laplacian)


pad_jacobian = jax.vmap(_pad_jacobian_to_output_deps, in_axes=(-2, -2, None), out_axes=-2)
pad_pairwise_jacobian = jax.vmap(pad_jacobian, in_axes=(-3, -3, None), out_axes=-3)


def zeropad_jacobian(x: FwdLaplArray, n_deps_out: int) -> FwdLaplArray:
    padding_shape = (n_deps_out - x.jacobian.data.shape[0], *x.jacobian.data.shape[1:])
    jac_padded = jnp.concatenate([x.jacobian.data, jnp.zeros(padding_shape, x.jacobian.data.dtype)], axis=0)
    return FwdLaplArray(x=x.x, jacobian=FwdJacobian(jac_padded), laplacian=x.laplacian)


def pad_jacobian_to_dense(x: FwdLaplArray, dependencies, n_deps_out: int) -> FwdLaplArray:
    jac = _split_off_xyz_dim(x.jacobian.data)
    jac_out = jnp.zeros([n_deps_out, 3, *jac.shape[2:]])
    jac_out = jac_out.at[dependencies, ...].set(jac, mode="drop")
    jac_out = _merge_xyz_dim(jac_out)
    return FwdLaplArray(x=x.x, jacobian=FwdJacobian(jac_out), laplacian=x.laplacian)


def get_inverse_from_lu(lu, permutation):
    n = lu.shape[0]
    b = jnp.eye(n, dtype=lu.dtype)[permutation]
    x = jax.lax.linalg.triangular_solve(lu, b, left_side=True, lower=True, unit_diagonal=True)  # type: ignore (private usage?)
    x = jax.lax.linalg.triangular_solve(lu, x, left_side=True, lower=False)  # type: ignore (private usage?)
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

    orbitals_lu, orbitals_pivot, orbitals_permutation = jax.lax.linalg.lu(orbitals.x)  # type: ignore (private usage?)
    orbitals_inv = get_inverse_from_lu(orbitals_lu, orbitals_permutation)
    sign, logdet = slogdet_from_lu(orbitals_lu, orbitals_pivot)

    M = orbitals.jacobian.data @ orbitals_inv
    M = M.reshape([n_deps, 3, n_el, n_el])  # split (deps * dim) into (deps, dim)

    # TODO: n_deps_max != n_dependants_max in general
    # [n_el, n_deps]
    # In contrast to dependencies, this tensor contains the indices of the electrons that depend on a given electron
    # rather than the indices of the electrons that a given electron depends on.
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


def get_neighbour_coordinates(electrons: Electrons, R: Nuclei, idx_nb: NeighbourIndices, spins: Spins):
    # [n_el  x n_neighbouring_electrons] - spin of each adjacent electron for each electron
    spin_nb_ee = get_with_fill(spins, idx_nb.ee, 0.0)

    # [n_el  x n_neighbouring_electrons x 3] - position of each adjacent electron for each electron
    r_nb_ee = get_with_fill(electrons, idx_nb.ee, NO_NEIGHBOUR)

    # [n_nuc  x n_neighbouring_electrons] - spin of each adjacent electron for each nucleus
    spin_nb_ne = get_with_fill(spins, idx_nb.ne, 0.0)

    # [n_nuc x n_neighbouring_electrons x 3] - position of each adjacent electron for each nuclei
    r_nb_ne = get_with_fill(electrons, idx_nb.ne, NO_NEIGHBOUR)

    # [n_el  x n_neighbouring_nuclei    x 3] - position of each adjacent nuclei for each electron
    R_nb_en = get_with_fill(R, idx_nb.en, NO_NEIGHBOUR)
    return spin_nb_ee, r_nb_ee, spin_nb_ne, r_nb_ne, R_nb_en
