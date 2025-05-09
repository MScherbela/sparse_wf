import functools
from typing import Sequence, TypeVar, cast

import jax
import jax.numpy as jnp
import numpy as np
import pyscf
from jaxtyping import ArrayLike, Float, Array, Bool, Integer
from folx import batched_vmap

from sparse_wf.api import (
    ParameterizedWaveFunction,
    Electrons,
    Nuclei,
    Charges,
    StaticInput,
)
from sparse_wf.tree_utils import tree_max
from sparse_wf.jax_utils import vmap_reduction
from sparse_wf.model.graph_utils import get_with_fill, NO_NEIGHBOUR

P = TypeVar("P")
MS = TypeVar("MS")


# The first integer is the number of core electrons
# the second integers in the tuples are the angular momentums
# An angular momentum of -1 is the local part
# The two floats in the end are linear and exponential coefficients
EcpData = tuple[int, list[tuple[int, list[list[tuple[float, float]]]]]]
EcpValues = Float[Array, "n_ecp n_l n_grid"]
EcpMask = Bool[np.ndarray, "n_atoms"]
AtomEcpValues = Float[Array, "n_l n_grid"]
EcpGrid = Float[Array, "n_grid"]  # Radial grid distances
EcpCutoffs = Float[Array, "n_ecp"]  # for each ECP atom, the cutoff distance


def _sph_to_cart(spherical_coords: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    """Convert spherical coordinates to Cartesian coordinates."""
    theta, phi = np.array(spherical_coords).T
    return np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], np.float32).T


def make_spherical_grid(n_points: int):
    match n_points:
        case 1:
            r = _sph_to_cart([[0, 0]])
        case 2:
            r = _sph_to_cart([[0, 0], [np.pi, 0]])
        case 4:
            r = _sph_to_cart(
                [
                    [0, 0],
                    [np.arccos(-1 / 3), 0],
                    [np.arccos(-1 / 3), 2 * np.pi / 3],
                    [np.arccos(-1 / 3), 4 * np.pi / 3],
                ]
            )
        case 6:
            r = _sph_to_cart(
                [
                    [0, 0],
                    [np.pi, 0],
                    [np.pi / 2, 0],
                    [np.pi / 2, np.pi / 2],
                    [np.pi / 2, np.pi],
                    [np.pi / 2, 3 * np.pi / 2],
                ]
            )
        case 12:
            # spherical_coords = [[0, 0], [np.pi, 0]]
            # spherical_coords += [[np.arctan(2), 2 * np.pi * i / 5] for i in range(1, 6)]
            # spherical_coords += [[np.pi - np.arctan(2), np.pi / 5 * (2 * i - 11)] for i in range(6, 11)]
            # return _sph_to_cart(spherical_coords)

            # 12-point scheme to order 5 from http://neilsloane.com/sphdesigns/dim3/
            x, y = 0.850650808352, 0.525731112119
            r = np.array(
                [
                    [x, 0, -y],
                    [y, -x, 0],
                    [0, -y, x],
                    [x, 0, y],
                    [-y, -x, 0],
                    [0, y, -x],
                    [-x, 0, -y],
                    [-y, x, 0],
                    [0, y, x],
                    [-x, 0, y],
                    [y, x, 0],
                    [0, -y, -x],
                ]
            )
        case 24:
            # 24-point scheme to order 7 from http://neilsloane.com/sphdesigns/dim3/
            a = 0.8662468181078205913835980
            b = 0.4225186537611115291185464
            c = 0.2666354015167047203315344
            r = np.array(
                [
                    [a, b, c],
                    [a, -b, -c],
                    [a, c, -b],
                    [a, -c, b],
                    [-a, b, -c],
                    [-a, -b, c],
                    [-a, c, b],
                    [-a, -c, -b],
                    [c, a, b],
                    [-c, a, -b],
                    [-b, a, c],
                    [b, a, -c],
                    [-c, -a, b],
                    [c, -a, -b],
                    [b, -a, c],
                    [-b, -a, -c],
                    [b, c, a],
                    [-b, -c, a],
                    [c, -b, a],
                    [-c, b, a],
                    [b, -c, -a],
                    [-b, c, -a],
                    [c, b, -a],
                    [-c, -b, -a],
                ]
            )
        case _:
            raise ValueError(f"Unsupported number of points: {n_points}")

    assert len(r) == n_points
    assert np.allclose(np.linalg.norm(r, axis=1), 1)
    weights = jnp.ones(n_points, np.float32) / n_points
    return jnp.array(r, jnp.float32), weights


def vmap_sum(f, *vmap_args, **vmap_kwargs):
    return vmap_reduction(f, functools.partial(jnp.sum, axis=0), *vmap_args, **vmap_kwargs)


def eval_ecp(radius: EcpGrid, coeffs: list[list[tuple[float, float]]]):
    r"""Evaluates r^2 U_l = \\sum A_{lk} r^{n_{lk}} e^{- B_{lk} r^2}."""
    val = jnp.zeros((), dtype=jnp.float32)
    for r_exponent, v in enumerate(coeffs):
        for exp_coeff, linear_coeff in v:
            val += linear_coeff * radius**r_exponent * jnp.exp(-exp_coeff * radius**2)
    return val


def eval_ecp_on_grid(
    ecp_all: dict[int, EcpData],
    r_grid: jnp.ndarray | None = None,
    log_r0: float = -5,
    log_rn: float = 10,
    n_grid: int = 10001,
) -> tuple[dict[int, int], dict[int, AtomEcpValues], EcpGrid, int]:
    """
    Evaluate effective core potentials (ECP) on a logarithmic grid.

    Parameters:
    -----------
    ecp_all : EcpData
        A dictionary where keys are atomic numbers and values are tuples. Each tuple contains
        the number of core electrons and a list of ECP coefficients for different angular momentum channels.
    r_grid : jnp.ndarray, optional
        A predefined grid of radial distances. If None, a logarithmic grid will be generated.
    log_r0 : float, optional
        The logarithm of the starting value for the radial grid. Default is -5.
    log_rn : float, optional
        The logarithm of the ending value for the radial grid. Default is 10.
    n_grid : int, optional
        The number of points in the radial grid. Default is 10001.

    Returns:
    --------
    n_cores : dict[int, int]
        A dictionary where keys are atomic numbers and values are the number of core electrons.
    v_grid_dict : dict[int, jnp.ndarray]
        A dictionary where keys are atomic numbers and values are arrays of ECP values evaluated on the radial grid.
    r_grid : jnp.ndarray
        The radial grid used for evaluation.
    max_channels : int
        The maximum number of angular momentum channels across all atoms in `ecp_all`.
    """
    if r_grid is None:
        r_grid = jnp.logspace(log_r0, log_rn, n_grid)
    else:
        n_grid = r_grid.size

    max_channels = max(len(val[1]) for val in ecp_all.values())
    n_cores = {z: val[0] for z, val in ecp_all.items()}
    v_grid_dict = {}

    for z, (_, ecp_val) in ecp_all.items():
        v_grid = jnp.zeros((max_channels, n_grid))

        for angular, coeffs in ecp_val:
            # the local part will have index -1 and, thus, will be the last element in the array
            v_grid = v_grid.at[angular].set(eval_ecp(r_grid, coeffs) / r_grid**2)

        v_grid_dict[z] = jnp.asarray(v_grid)

    return n_cores, v_grid_dict, r_grid, max_channels


def make_local_pseudopotential(ecp_vals: EcpValues, grid_radius: EcpGrid, ecp_mask: EcpMask):
    def pp_loc(r_ae: Float[Array, " n_elec"], grid_radius: EcpGrid, ecp_vals: AtomEcpValues):
        # Interpolate the potential on the grid
        return jnp.interp(r_ae, grid_radius, ecp_vals).sum()  # sum over electrons

    # vmap over atoms
    vmap_pp_loc = vmap_sum(pp_loc, in_axes=(1, None, 0))  # sum over nuclei

    # TODO: Update input
    def electron_atom_local_pseudopotential(electrons: Electrons, atoms: Nuclei) -> jnp.ndarray:
        # mask out non-pseudized atoms
        r_ae_ecp = jnp.linalg.norm(electrons[:, None] - atoms[ecp_mask], axis=-1)
        return vmap_pp_loc(r_ae_ecp, ecp_vals, grid_radius)

    return electron_atom_local_pseudopotential


def random_rot_mat(key: Array) -> Float[Array, "3 3"]:
    v1, v2 = jax.random.normal(key, shape=(2, 3), dtype=jnp.float32)
    v1 /= jnp.linalg.norm(v1)
    v2 -= v1 * jnp.dot(v1, v2)
    v2 /= jnp.linalg.norm(v2)
    v3 = jnp.cross(v1, v2)
    return jnp.array([v1, v2, v3]).T


def leg_l0(x):
    return jnp.ones_like(x)


def leg_l1(x):
    return x


def leg_l2(x):
    return 0.5 * (3 * x**2 - 1)


def leg_l3(x):
    return 0.5 * (5 * x**3 - 3 * x)


def eval_leg(x: Float[ArrayLike, "..."], angular: Integer[ArrayLike, ""]) -> Float[Array, "..."]:
    """
    Evaluate the Legendre polynomial of degree `l` at point `x`.
    """
    return jax.lax.switch(angular, [leg_l0, leg_l1, leg_l2, leg_l3], x)


def eval_leg0to3(x: Float[ArrayLike, "..."]) -> Float[Array, "..."]:
    """
    Evaluate the Legendre polynomial of degree `l` at point `x`.
    """
    return jnp.stack([leg_l0(x), leg_l1(x), leg_l2(x), leg_l3(x)], axis=-1)


def get_integration_points(key: jax.random.PRNGKey, r: Float[Array, "3"], R: Float[Array, "3"], n_points: int):
    """
    Get integration points for the ECP evaluation.
    """
    dist = jnp.linalg.norm(r - R)
    rot_mat = random_rot_mat(key)
    points, weights = make_spherical_grid(n_points)
    rotated_points = (points @ rot_mat.T) * dist + R
    return rotated_points, weights, jnp.ones(n_points, int) * n_points


def build_all_integration_points(
    key: jax.random.PRNGKey,
    electrons: Electrons,
    R_ecp: Float[Array, "n_ecp_atoms 3"],
    unique_n_quad_points: tuple[int],
    n_quad_points: Integer[np.ndarray, " n_ecp_atoms"],
    cutoffs: EcpCutoffs,
    static_n_points: int,
):
    """
    Get a list of all electron coordinates r' at which the non-local potential needs to be evaluated
    This requires up to n_el * max(n_quad_points) evaluations, but it can be less due to 2 factors:
       1. Not all electrons are within the cutoff of the closest ECP atom
       2. Not all ECP atoms have the same number of quadrature points
    We first build a list of all r' n_el * sum(n_quad_points) and then filter to the active ones using a static arg
    """

    n_el = electrons.shape[0]
    ae_ecp = electrons[:, None] - R_ecp
    r_ae_ecp = jnp.linalg.norm(ae_ecp, axis=-1)

    # drop all but closest ecp atom
    idx_closest_atom = jnp.argmin(r_ae_ecp, axis=1)  # [n_el]
    R_closest_atom = R_ecp[idx_closest_atom]  # [n_el, 3]
    dr_closest_atom = electrons - R_closest_atom  # [n_el, 3]
    dist_closest_atom = jnp.linalg.norm(dr_closest_atom, axis=-1)  # [n_el]
    direction_closest_atom = dr_closest_atom / dist_closest_atom[:, None]  # [n_el, 3]

    keys = jax.random.split(key, n_el)
    r_shift_list = []
    weights_list = []
    is_active_list = []
    cos_theta_list = []
    for n in unique_n_quad_points:
        grid, w = make_spherical_grid(n)  # [n, 3]
        rot_mat = jax.vmap(random_rot_mat)(keys)  # [el, 3, 3]
        grid_rotated = jnp.einsum("ixy,gx->igy", rot_mat, grid)  # [n_el, n, 3]
        r_int = R_closest_atom[:, None, :] + dist_closest_atom[:, None, None] * grid_rotated  # [n_el, n, 3]
        cos = jnp.einsum("inx,ix->in", grid_rotated, direction_closest_atom)  # [n_el, n]
        r_shift_list.append(r_int - electrons[:, None, :])  # [n_el, n, 3]
        weights_list.append(jnp.tile(w[None, :], (n_el, 1)))  # [n_el, n]
        cos_theta_list.append(cos)

        is_in_range = dist_closest_atom < cutoffs[idx_closest_atom]  # [n_el]
        is_correct_grid = n_quad_points[idx_closest_atom] == n  # [n_el]
        is_active_list.append(jnp.tile((is_in_range & is_correct_grid)[:, None], (1, n)))

    r_shift = jnp.concatenate(r_shift_list, axis=-2).reshape([-1, 3])  # [n_el, n_total, 3] => [n_el * n_total, 3]
    weights = jnp.concatenate(weights_list, axis=-1).reshape([-1])  # [n_el, n_total] => [n_el * n_total]
    is_active = jnp.concatenate(is_active_list, axis=-1).reshape([-1])  # [n_el, n_total] => [n_el * n_total]
    idx_el = jnp.repeat(jnp.arange(n_el), sum(unique_n_quad_points))
    cos_theta = jnp.concatenate(cos_theta_list, axis=-1).reshape([-1])

    actual_n_active = is_active.sum()
    idx_active = jnp.where(is_active, size=static_n_points, fill_value=NO_NEIGHBOUR)[0]
    idx_el = get_with_fill(idx_el, idx_active, 0)
    idx_closest_atom = idx_closest_atom[idx_el]
    r_shift = get_with_fill(r_shift, idx_active, 0.0)
    r_integration = electrons[idx_el] + r_shift
    dist_ae = jnp.linalg.norm(r_integration - R_closest_atom[idx_el], axis=-1)
    weights = get_with_fill(weights, idx_active, 0.0)
    cos_theta = get_with_fill(cos_theta, idx_active, 0.0)
    return idx_el, idx_closest_atom, r_integration, dist_ae, weights, cos_theta, actual_n_active


def make_nonlocal_pseudopotential(
    r_grid: EcpGrid,
    v_grid_nonloc: EcpValues,
    ecp_mask: EcpMask,
    cutoffs: EcpCutoffs,
    n_quad_points: Integer[np.ndarray, " n_ecp_atoms"],
):
    unique_n_quad_points = np.unique(n_quad_points)
    unique_n_quad_points = unique_n_quad_points[unique_n_quad_points > 0]
    n_quad_points = jnp.array(n_quad_points, jnp.int32)  # type: ignore
    n_l_values = v_grid_nonloc.shape[1]
    assert n_l_values <= 4, "Non-local pp only supports angular momemtum up to l=3"

    def pp_nonloc(
        key: Array,
        logpsi_fn: ParameterizedWaveFunction[P, MS],
        params: P,
        electrons: Electrons,
        static: StaticInput,
    ):
        R = jnp.array(logpsi_fn.R[ecp_mask], jnp.float32)
        idx_el, idx_ecp_atom, r_integration, dist_ae, weights, cos_theta, actual_n_active = (
            build_all_integration_points(
                key,
                electrons,
                R,
                unique_n_quad_points,
                n_quad_points,
                cutoffs,
                static.n_pp_elecs,  # type: ignore
            )
        )
        (sign_denom, logpsi_denom), state = logpsi_fn.log_psi_with_state(params, electrons, static)

        def get_psi_ratio(r_int, idx_el):
            electrons_new = electrons.at[idx_el].set(r_int)
            idx_changed = idx_el[None]
            (sign, logpsi), _ = logpsi_fn.log_psi_low_rank_update(params, electrons_new, idx_changed, static, state)
            f_ratio = sign * sign_denom * jnp.exp(logpsi - logpsi_denom)
            actual_static = logpsi_fn.get_static_input(electrons, electrons_new, idx_changed)
            # We don't need to compute the triplets here and can safely set them to zero to avoid unnecessary computation
            actual_static = actual_static.replace(n_triplets=jnp.zeros((), dtype=jnp.int32))
            return f_ratio, actual_static

        @jax.vmap  # vmap over integration points
        def get_radial_potential(dist_ae, idx_ecp_atom):
            # vmap over l-values
            return jax.vmap(lambda v: jnp.interp(dist_ae, xp=r_grid, fp=v))(v_grid_nonloc[idx_ecp_atom])

        psi_ratio, actual_static = batched_vmap(get_psi_ratio, max_batch_size=64)(
            r_integration, idx_el
        )  # [n_integrations]
        V_nonloc = get_radial_potential(dist_ae, idx_ecp_atom)  # [n_integrations * n_l_values]
        legendre_poly = eval_leg0to3(cos_theta)[:, :n_l_values]  # [n_integrations * n_l_values]
        angular_l = np.arange(n_l_values)
        V = jnp.einsum("kl,k,k,l,kl", V_nonloc, weights, psi_ratio, 2 * angular_l + 1, legendre_poly)
        actual_static = tree_max(actual_static)
        new_static = actual_static.replace(n_pp_elecs=actual_n_active)
        return V, new_static

    return pp_nonloc


def mock_nl_pp(
    key: Array,
    logpsi_fn: ParameterizedWaveFunction[P, MS],
    params: P,
    electrons: Electrons,
    static: StaticInput,
):
    return jnp.zeros(()), static


def make_pseudopotential(
    charges: Charges,
    symbols: Sequence[str],
    n_quad_points: dict[str, int],
    ecp: str = "ccecp",  # basis set
    cutoff_value: float = 1e-7,
):
    if len(symbols) == 0:
        return charges, lambda *_, **__: 0.0, mock_nl_pp

    ecp_data: dict[int, EcpData] = {
        pyscf.lib.parameters.ELEMENTS_PROTON[symbol.capitalize()]: pyscf.gto.basis.load_ecp(ecp, symbol)
        for symbol in symbols
    }
    n_cores, v_grid_dict, grid_radius, max_channels = eval_ecp_on_grid(ecp_data)
    # residual atomic charge
    charge_list = cast(list[int], charges.tolist())
    effective_charges = np.asarray([z - n_cores.get(z, 0) for z in charge_list])
    # mask to separate pseudo atoms from regular atoms
    ecp_mask: EcpMask = np.abs(np.asarray(charges) - effective_charges) > 2.0e-6
    # construct v_grids
    n_ecp = ecp_mask.sum()  # number of atoms with ECP
    # if there are no ECP atoms, return zero potentials
    if n_ecp == 0:
        return charges, lambda *_, **__: 0.0, mock_nl_pp

    n_grid = grid_radius.size  # number of radial grid points - 1D
    ecp_grid_values = jnp.zeros((n_ecp, max_channels, n_grid))  # ECP values on the radial grid
    n_integration_points = np.zeros(n_ecp, int)
    for i, z in enumerate(charges[ecp_mask].tolist()):
        ecp_grid_values = ecp_grid_values.at[i].set(v_grid_dict[z])
        n_integration_points[i] = n_quad_points.get(pyscf.lib.parameters.ELEMENTS[z], n_quad_points["default"])

    non_loc_grid_values, loc_grid_values = jnp.split(ecp_grid_values, (max_channels - 1,), axis=1)
    loc_grid_values = loc_grid_values.reshape(-1, n_grid)

    # Find cutoffs after which the PP can be neglected
    larger_than_cutoff = np.abs(non_loc_grid_values) > cutoff_value
    cutoff_idx = np.max(larger_than_cutoff * np.arange(grid_radius.shape[-1]), axis=(1, 2))
    cutoffs = grid_radius[cutoff_idx]

    pp_local = make_local_pseudopotential(grid_radius, loc_grid_values, ecp_mask)
    pp_nonlocal = make_nonlocal_pseudopotential(
        grid_radius, non_loc_grid_values, ecp_mask, cutoffs, n_integration_points
    )
    return effective_charges, pp_local, pp_nonlocal
