import functools
from typing import TypeAlias

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from folx.api import FwdJacobian, FwdLaplArray
from jaxtyping import Array, Float, Integer, Shaped

from sparse_wf.api import Electrons, Int, Nuclei
from sparse_wf.jax_utils import vectorize
from sparse_wf.model.utils import slog_and_inverse

NO_NEIGHBOUR = 1_000_000


DistanceMatrix: TypeAlias = Float[Array, "n1 n2"]
ElectronElectronEdges = Integer[Array, "n_electrons n_nb_ee"]
ElectronNucleiEdges = Integer[Array, "n_electrons n_nb_en"]
NucleiElectronEdges = Integer[Array, "n_nuclei n_nb_ne"]

Dependant = Integer[Array, "n_dependants"]
Dependency = Integer[Array, "n_deps"]
DependencyMap = Integer[Array, "n_center n_neighbour n_deps"]


@vectorize(signature="(n,d),(m,d)->(n,n),(m,n)")
def get_full_distance_matrices(r: Electrons, R: Nuclei) -> tuple[DistanceMatrix, DistanceMatrix]:
    dist_ee = jnp.linalg.norm(r[:, None, :] - r[None, :, :], axis=-1)
    dist_ne = jnp.linalg.norm(R[:, None, :] - r[None, :, :], axis=-1)
    return dist_ee, dist_ne


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
    assert (
        dep_map.shape[0] * 3 == x.jacobian.data.shape[0]
    ), f"Dependency map and jacobian have inconsistent shapes: {dep_map.shape[0]} * 3 != {x.jacobian.data.shape[0]}"
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
    jac_out = jnp.zeros([n_deps_out, 3, *jac.shape[2:]], x.dtype)
    jac_out = jac_out.at[dependencies, ...].set(jac, mode="drop")
    jac_out = _merge_xyz_dim(jac_out)
    return FwdLaplArray(x=x.x, jacobian=FwdJacobian(jac_out), laplacian=x.laplacian)


def slogdet_with_sparse_fwd_lap(orbitals: FwdLaplArray, dependencies: Integer[Array, "el ndeps"]):
    n_el, n_orb = orbitals.x.shape[-2:]
    n_deps = dependencies.shape[-1]
    assert n_el == n_orb

    (sign, logdet), orbitals_inv = slog_and_inverse(orbitals.x)

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
