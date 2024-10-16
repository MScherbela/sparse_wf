import functools
from typing import Any, Callable, Sequence, TypeVar

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import pyscf
from jaxtyping import ArrayLike, Float, Array, Bool, Integer

from sparse_wf.api import ParameterizedWaveFunction, Electrons, Nuclei, Charges, StaticInput
from sparse_wf.tree_utils import tree_maximum
from sparse_wf.jax_utils import vmap_reduction

P = TypeVar("P")
S = TypeVar("S", bound=StaticInput)
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
        return jnp.interp(r_ae, grid_radius, ecp_vals).sum()

    # vmap over atoms
    vmap_pp_loc = vmap_sum(pp_loc, in_axes=(1, None, 0))

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


def project_legendre(
    direction: Float[Array, "3"],
    low_rank_f: Callable[[Float[Array, "3"], Integer[Array, ""]], tuple[Float[Array, ""], Float[Array, ""]]],
    static_f: Callable[[Float[Array, "3"], Integer[Array, ""]], StaticInput],
    denom_sign: Float[Array, ""],
    denom_logpsi: Float[Array, ""],
    electron_distance_i: Float[Array, ""],
    electron_vector_i: Float[Array, "3"],
    index: Integer[Array, ""],
    atom_vector: Float[Array, "3"],
    angular_momentum: Integer[Array, ""],
):
    """Projects the Legrende polynomials."""
    cos = direction @ electron_vector_i / electron_distance_i
    leg = eval_leg(cos, angular_momentum)
    e_prime = electron_distance_i * direction
    e = e_prime + atom_vector

    sign, logpsi = low_rank_f(e, index)
    f_ratio = sign / denom_sign * jnp.exp(logpsi - denom_logpsi)
    return jnp.squeeze(leg * f_ratio), static_f(e, index)


def make_spherical_integral(quad_degree: int):
    """Creates callable for evaluating an integral over a spherical quadrature."""

    if quad_degree != 4:
        raise RuntimeError("quad_degree = 4 is the only implemented quadrature")
    # This matches (up to rotation and permutation) quadpy.u3.get_good_scheme(4)
    # and is the quadrature used in the ByteDance pseudopotential paper etc.
    n_points = 12
    weights = jnp.ones(n_points, dtype=jnp.float32) / n_points
    spherical_points = [[0, 0], [np.pi, 0]]
    spherical_points += [[np.arctan(2), 2 * np.pi * i / 5] for i in range(1, 6)]
    spherical_points += [[np.pi - np.arctan(2), np.pi / 5 * (2 * i - 11)] for i in range(6, 11)]
    theta, phi = zip(*spherical_points)
    points = jnp.stack(
        [
            np.cos(phi) * np.sin(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(theta),
        ],
        axis=1,
    ).astype(jnp.float32)

    def spherical_integral(
        key: Array,
        logpsi_fn: ParameterizedWaveFunction[P, S, MS],
        params: P,
        electrons: Electrons,
        static: S,
        electron_atom_distance: Float[Array, ""],
        electron_atom_vector: Float[Array, "3"],
        electron_index: Integer[Array, ""],
        atom_position: Float[Array, "3"],
        angular_momentum: Integer[Array, ""],
    ):
        # randomly rotate the integration grid
        rot_mat = random_rot_mat(key)
        aligned_points = points @ rot_mat.T
        (sign, logpsi), state = logpsi_fn.log_psi_with_state(params, electrons, static)

        def low_rank_f(electron, i):
            return logpsi_fn.log_psi_low_rank_update(
                params,
                electrons.at[i].set(electron),
                jnp.array([i], dtype=jnp.int32),
                static,
                state,
            )[0]

        def static_f(electron, i):
            return logpsi_fn.get_static_input(electrons, electrons.at[i].set(electron), jnp.array([i], dtype=jnp.int32))

        def _body_fun(i, vals: tuple[Array, S]):
            val, static = vals
            legendre_val, new_static = project_legendre(
                aligned_points[i],
                low_rank_f,
                static_f,
                sign,
                logpsi,
                electron_atom_distance,
                electron_atom_vector,
                electron_index,
                atom_position,
                angular_momentum,
            )
            return val + weights[i] * legendre_val, tree_maximum(static, new_static)

        init_loop = jnp.zeros_like(logpsi)
        # TODO: I guess we could use vmap_sum or vmap_sum + folx.batched_vmap here
        new_static = jtu.tree_map(jnp.asarray, static)
        legendre, new_static = jax.lax.fori_loop(0, n_points, _body_fun, (init_loop, new_static))
        return (2 * angular_momentum + 1) * legendre, new_static

    return spherical_integral


def make_nonlocal_pseudopotential(
    r_grid: EcpGrid, v_grid_nonloc: EcpValues, ecp_mask: EcpMask, cutoffs: EcpCutoffs, quad_degree: int
):
    """Creates callable for evaluating the non-local pseudopotential."""

    def pp_nonloc(
        key: Array,
        logpsi_fn: ParameterizedWaveFunction[P, S, MS],
        params: P,
        electrons: Electrons,
        static: S,
        electron_atom_distance_i: Float[Array, ""],
        electron_atom_vector_i: Float[Array, "3"],
        electron_index: Integer[Array, ""],
        atom_position: Float[Array, "3"],
        angular_momentum: Integer[Array, ""],
        v_grid: Float[Array, "nchan n_grid"],
    ):
        integral, new_static = make_spherical_integral(quad_degree)(
            key,
            logpsi_fn,
            params,
            electrons,
            static,
            electron_atom_distance_i,
            electron_atom_vector_i,
            electron_index,
            atom_position,
            angular_momentum,
        )

        potential_radial_value = jnp.interp(electron_atom_distance_i, xp=r_grid, fp=v_grid)
        return potential_radial_value * integral, new_static

    # vmap over electrons
    vmap_pp_nonloc = vmap_reduction(
        pp_nonloc,
        (functools.partial(jnp.sum, axis=0), jnp.max),
        in_axes=(0, None, None, None, None, 0, 0, 0, 0, None, 0),
    )
    # vmap_pp_nonloc = vmap_sum(pp_nonloc, in_axes=(0, None, None, None, None, 0, 0, 0, 0, None, 0))

    # vmap over angular momentum
    vmap_pp_nonloc = vmap_reduction(
        vmap_pp_nonloc,
        (functools.partial(jnp.sum, axis=0), jnp.max),
        in_axes=(None, None, None, None, None, None, None, None, None, 0, 1),
    )
    # vmap_pp_nonloc = vmap_sum(
    #     vmap_pp_nonloc,
    #     in_axes=(None, None, None, None, None, None, None, None, None, 0, 1),
    # )

    def electron_atom_nonlocal_pseudopotential(
        key: Array,
        logpsi_fn: ParameterizedWaveFunction,
        params: P,
        electrons: Electrons,
        static: Any,
    ) -> tuple[Float[Array, ""], S]:
        """Evaluates electron-atom contribution to non-local pseudopotential."""
        n_elec = electrons.shape[0]
        n_nonloc = v_grid_nonloc.shape[1]

        atoms = jnp.asarray(logpsi_fn.R[ecp_mask])
        ae_ecp = electrons[:, None] - atoms
        r_ae_ecp = jnp.linalg.norm(ae_ecp, axis=-1)

        # drop all but closest ecp atom
        closest_atom = jnp.argmin(r_ae_ecp, axis=1)
        # Compute the number of electrons that are within the cutoff
        closest_atom_dist = r_ae_ecp[jnp.arange(n_elec), closest_atom]
        in_cutoff = closest_atom_dist < cutoffs[closest_atom]
        num_in_cutoff = in_cutoff.sum()

        # Select only the electrons that are within the cutoff
        electron_idx = jnp.argsort(closest_atom_dist)[: static.n_pp_elecs]
        # For all of them use:
        # electron_idx = jnp.arange(n_elec)
        closest_atom = closest_atom[electron_idx]

        r_ae_ecp_closest = r_ae_ecp[electron_idx, closest_atom]
        ae_ecp_closest = ae_ecp[electron_idx, closest_atom, :]

        atoms_ecp_closest = atoms[closest_atom]

        v_grid_nonloc_closest = v_grid_nonloc[closest_atom]

        keys = jax.random.split(key, n_elec).reshape(n_elec, *key.shape)

        pp, new_static = vmap_pp_nonloc(
            keys,
            logpsi_fn,
            params,
            electrons,
            static,
            r_ae_ecp_closest,
            ae_ecp_closest,
            electron_idx,
            atoms_ecp_closest,
            jnp.arange(n_nonloc),
            v_grid_nonloc_closest,
        )
        return pp, new_static.replace(n_pp_elecs=num_in_cutoff)

    return electron_atom_nonlocal_pseudopotential


def mock_nl_pp(
    key: Array,
    logpsi_fn: ParameterizedWaveFunction[P, S, MS],
    params: P,
    electrons: Electrons,
    static: S,
):
    return jnp.zeros(()), static


def make_pseudopotential(
    charges: Charges,
    symbols: Sequence[str],
    quad_degree: int = 4,  # quadrature degree
    ecp: str = "ccecp",  # basis set
    cutoff_value: float = 1e-7,
):
    ecp_data: dict[int, EcpData] = {
        pyscf.lib.parameters.ELEMENTS_PROTON[symbol.capitalize()]: pyscf.gto.basis.load_ecp(ecp, symbol)
        for symbol in symbols
    }
    n_cores, v_grid_dict, grid_radius, max_channels = eval_ecp_on_grid(ecp_data)
    # residual atomic charge
    effective_charges = np.asarray([z - n_cores.get(z, 0) for z in charges.tolist()])
    # mask to separate pseudo atoms from regular atoms
    ecp_mask: EcpMask = np.abs(np.asarray(charges) - effective_charges) > 2.0e-6
    # construct v_grids
    n_ecp = ecp_mask.sum()  # number of atoms with ECP
    # if there are no ECP atoms, return zero potentials
    if n_ecp == 0:
        return charges, lambda *_, **__: 0.0, mock_nl_pp

    n_grid = grid_radius.size  # number of radial grid points - 1D
    ecp_grid_values = jnp.zeros((n_ecp, max_channels, n_grid))  # ECP values on the radial grid
    for i, z in enumerate(charges[ecp_mask].tolist()):
        ecp_grid_values = ecp_grid_values.at[i].set(v_grid_dict[z])

    non_loc_grid_values, loc_grid_values = jnp.split(ecp_grid_values, (max_channels - 1,), axis=1)
    loc_grid_values = loc_grid_values.reshape(-1, n_grid)

    # Find cutoffs after which the PP can be neglected
    larger_than_cutoff = np.abs(non_loc_grid_values) > cutoff_value
    cutoff_idx = np.max(larger_than_cutoff * np.arange(grid_radius.shape[-1]), axis=(1, 2))
    cutoffs = grid_radius[cutoff_idx]

    pp_local = make_local_pseudopotential(grid_radius, loc_grid_values, ecp_mask)
    pp_nonlocal = make_nonlocal_pseudopotential(grid_radius, non_loc_grid_values, ecp_mask, cutoffs, quad_degree)
    return effective_charges, pp_local, pp_nonlocal
