from typing import TypeVar, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
import pyscf
from jax import lax
import jax.tree_util as jtu
import functools

from sparse_wf.api import (
    Charges,
    ClosedLogLikelihood,
    Electrons,
    Int,
    LogAmplitude,
    MCStep,
    Nuclei,
    ParameterizedWaveFunction,
    PMove,
    PRNGKeyArray,
    Width,
    WidthScheduler,
    WidthSchedulerState,
    MCMC_proposal_type,
)
from sparse_wf.jax_utils import jit, psum


def mh_update_all_electron(
    log_prob_fn: ClosedLogLikelihood,
    key: PRNGKeyArray,
    electrons: Electrons,
    log_prob: LogAmplitude,
    num_accepts: Int,
    width: Width,
) -> tuple[PRNGKeyArray, Electrons, LogAmplitude, Int]:
    key, subkey = jax.random.split(key)
    eps = jax.random.normal(subkey, electrons.shape, dtype=electrons.dtype) * width
    new_electrons = electrons + eps
    new_log_prob = log_prob_fn(new_electrons)
    log_ratio = new_log_prob - log_prob

    key, subkey = jax.random.split(key)
    u = jax.random.uniform(subkey, log_ratio.shape)
    accept = log_ratio > jnp.log(u)
    new_electrons = jnp.where(accept[..., None, None], new_electrons, electrons)
    new_log_prob = jnp.where(accept, new_log_prob, log_prob)
    num_accepts += jnp.sum(accept).astype(jnp.int32)
    return key, new_electrons, new_log_prob, num_accepts


def mh_update_single_electron(
    update_logpsi_fn: Callable,
    params,
    static,
    key: PRNGKeyArray,
    electrons: Electrons,
    log_prob: LogAmplitude,
    model_cache: dict,
    num_accepts: Int,
    width: Width,
) -> tuple[PRNGKeyArray, Electrons, LogAmplitude, dict, Int]:
    cluster_size = 1
    n_el = electrons.shape[-2]

    key, key_select, key_propose, key_accept = jax.random.split(key, 4)
    ind_move = jax.random.randint(key_select, [cluster_size], 0, n_el)
    delta_r = jax.random.normal(key_propose, [cluster_size, 3], dtype=electrons.dtype) * width
    new_electrons = electrons.at[ind_move, :].add(delta_r)
    new_logpsi, new_model_cache = update_logpsi_fn(params, new_electrons, ind_move, model_cache, static)
    new_log_prob = 2 * new_logpsi
    log_ratio = new_log_prob - log_prob

    accept = log_ratio > jnp.log(jax.random.uniform(key_accept, log_ratio.shape))
    new_electrons, new_log_prob, new_model_cache = jtu.tree_map(
        lambda new, old: jnp.where(accept, new, old),
        (new_electrons, new_log_prob, new_model_cache),
        (electrons, log_prob, model_cache),
    )
    num_accepts += accept.astype(jnp.int32)
    return key, new_electrons, new_log_prob, new_model_cache, num_accepts


P, S = TypeVar("P"), TypeVar("S")


def mcmc_steps_all_electron(
    logpsi_fn: ParameterizedWaveFunction[P, S],
    steps: int,
    key: PRNGKeyArray,
    params: P,
    electrons: Electrons,
    static: S,
    width: Width,
):
    def log_prob_fn(electrons: Electrons) -> LogAmplitude:
        return 2 * jax.vmap(logpsi_fn, in_axes=(None, 0, None))(params, electrons, static)

    def step_fn(_, x):
        return mh_update_all_electron(log_prob_fn, *x, width)  # type: ignore

    logprob = log_prob_fn(electrons)
    num_accepts = jnp.zeros((), dtype=jnp.int32)
    key, electrons, logprob, num_accepts = lax.fori_loop(0, steps, step_fn, (key, electrons, logprob, num_accepts))
    pmove = psum(num_accepts) / (steps * electrons.shape[0] * jax.device_count())
    return electrons, pmove


def mcmc_steps_single_electron(
    logpsi_fn: ParameterizedWaveFunction[P, S],
    update_logpsi_fn: Callable,
    steps: int,
    key: PRNGKeyArray,
    params: P,
    electrons: Electrons,
    static: S,
    width: Width,
):
    logpsi, model_cache = logpsi_fn(params, electrons, static, return_cache=True)
    logprob = 2 * logpsi

    @functools.partial(jax.vmap, in_axes=(None, 0))
    def step_fn(_, x):
        return mh_update_single_electron(update_logpsi_fn, params, static, *x, width)

    local_batch_size = electrons.shape[0]
    x0 = (
        jax.random.split(key, local_batch_size),
        electrons,
        logprob,
        model_cache,
        jnp.zeros(local_batch_size, dtype=jnp.int32),
    )
    _, electrons, logprob, model_cache, num_accepts = lax.fori_loop(0, steps, step_fn, x0)
    pmove = psum(jnp.sum(num_accepts)) / (steps * local_batch_size * jax.device_count())
    return electrons, pmove


def make_mcmc(
    logpsi_fn: ParameterizedWaveFunction[P, S],
    update_logpsi_fn: Optional[Callable],
    proposal: MCMC_proposal_type,
    init_width,
    steps: int,
) -> tuple[MCStep[P, S], jax.Array]:
    # batch_network = jax.vmap(logpsi_fn, in_axes=(None, 0, None))

    if proposal == "all-electron":
        mcmc_step = functools.partial(mcmc_steps_all_electron, logpsi_fn, steps)
    elif proposal == "single-electron":
        mcmc_step = functools.partial(mcmc_steps_single_electron, logpsi_fn, update_logpsi_fn, steps)

    return mcmc_step, jnp.array(init_width, dtype=jnp.float32)


def make_width_scheduler(
    window_size: int = 20,
    target_pmove: float = 0.525,
    error: float = 0.025,
    width_multiplier: float = 1.1,
) -> WidthScheduler:
    @jit
    def init(init_width: Width) -> WidthSchedulerState:
        return WidthSchedulerState(
            width=jnp.array(init_width, dtype=jnp.float32),
            pmoves=jnp.zeros((window_size, *init_width.shape), dtype=jnp.float32),
            i=jnp.zeros((), dtype=jnp.int32),
        )

    @jit
    def update(state: WidthSchedulerState, pmove: PMove) -> WidthSchedulerState:
        pmoves = state.pmoves.at[jnp.mod(state.i, window_size)].set(pmove)
        pm_mean = state.pmoves.mean()
        i = state.i + 1

        upd_width = jnp.where(pm_mean < target_pmove - error, state.width / width_multiplier, state.width)
        upd_width = jnp.where(pm_mean > target_pmove + error, upd_width * width_multiplier, upd_width)
        width = jnp.where(
            jnp.mod(state.i, window_size) == 0,
            upd_width,
            state.width,
        )
        return WidthSchedulerState(width=width, pmoves=pmoves, i=i)

    return WidthScheduler(init, update)


def assign_spins_to_atoms(R: Nuclei, Z: Charges):
    n_el = np.sum(Z)

    # Assign equal nr of up and down spins to all atoms.
    # If the nuclear charge is odd, we'll redistribute the reamining spins below
    n_up_per_atom = Z // 2
    n_el_remaining = n_el - 2 * np.sum(n_up_per_atom)

    if n_el_remaining > 0:
        # Get the indices of the atoms with "open shells"
        ind_open_shell = np.where(Z % 2)[0]
        R_open = R[ind_open_shell]
        dist = np.linalg.norm(R_open[:, None, :] - R_open[None, :, :], axis=-1)
        kernel = np.exp(-dist * 0.5)

        # Loop over all remaining electrons
        spins = np.zeros(n_el_remaining)
        n_dn_left = n_el_remaining // 2
        n_up_left = n_el_remaining - n_dn_left
        for _ in range(n_el_remaining):
            is_free = spins == 0
            spin_per_site = kernel[is_free, :] @ spins

            # Compute the loss loss_i = sum_j kernel_ij * spin_j
            # and add another spin such that the loss is minimal (ie. as much anti-parallel as possible)
            ind_atom = np.arange(n_el_remaining)[is_free]
            loss_up = spin_per_site
            loss_dn = -spin_per_site
            if (n_up_left > 0) and (np.min(loss_up) < np.min(loss_dn)):
                ind = ind_atom[np.argmin(loss_up)]
                spins[ind] = 1
                n_up_left -= 1
            else:
                ind = ind_atom[np.argmin(loss_dn)]
                spins[ind] = -1
                n_dn_left -= 1

        # Add spins to the atoms with open shells
        n_up_per_atom[ind_open_shell] += spins == 1

    n_dn_per_atom = Z - n_up_per_atom
    # Collect a list of atom indices: first all up spins, then all down spins
    ind_atom = []
    for i, n_up in enumerate(n_up_per_atom):
        ind_atom += [i] * n_up
    for i, n_dn in enumerate(n_dn_per_atom):
        ind_atom += [i] * n_dn
    return np.array(ind_atom)


def init_electrons(key: PRNGKeyArray, mol: pyscf.gto.Mole, batch_size: int) -> Electrons:
    batch_size = batch_size - (batch_size % jax.device_count())
    local_batch_size = (batch_size // jax.device_count()) * jax.local_device_count()
    electrons = jax.random.normal(key, (local_batch_size, mol.nelectron, 3), dtype=jnp.float32)

    R = np.array(mol.atom_coords(), dtype=jnp.float32)
    n_atoms = len(R)
    if n_atoms > 1:
        assert mol.charge == 0, "Only atoms or neutral molecules are supported"
        assert abs(mol.spin) < 2, "Only atoms or singlet and doublet molecules are supported"  # type: ignore
        ind_atom = assign_spins_to_atoms(R, mol.atom_charges())
        electrons += R[ind_atom]
    return electrons
