from typing import TypeVar, Callable

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
    MCMCArgs,
    ElectronIdx,
    MCMCStaticArgs,
    StaticInput,
)
from sparse_wf.jax_utils import jit, pmean_if_pmap, pmax_if_pmap, psum_if_pmap
from sparse_wf.model.graph_utils import NO_NEIGHBOUR


def mh_update_all_electron(
    log_prob_fn: ClosedLogLikelihood,
    key: PRNGKeyArray,
    electrons: Electrons,
    log_prob: LogAmplitude,
    stats: dict,
    width: Width,
) -> tuple[PRNGKeyArray, Electrons, LogAmplitude, dict]:
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
    stats["num_accepts"] += jnp.sum(accept).astype(jnp.int32)
    return key, new_electrons, new_log_prob, stats


def proposal_single_electron(
    idx_step: int, key: PRNGKeyArray, electrons: Electrons, width: Width, max_cluster_size: int = 1
) -> tuple[Electrons, ElectronIdx, jax.Array, Int]:
    # key_select, key_propose = jax.random.split(key)
    n_el = electrons.shape[-2]
    # idx_el_changed = jax.random.randint(key_select, [1], 0, n_el)
    idx_el_changed = jnp.array([idx_step % n_el]).astype(jnp.int32)
    delta_r = jax.random.normal(key, [1, 3], dtype=electrons.dtype) * width
    proposed_electrons = electrons.at[idx_el_changed, :].add(delta_r)
    proposal_log_ratio = jnp.zeros([], dtype=electrons.dtype)
    actual_cluster_size = jnp.ones([], dtype=jnp.int32)
    return proposed_electrons, idx_el_changed, proposal_log_ratio, actual_cluster_size


def proposal_cluster_update(
    R: Nuclei,
    cluster_radius: float,
    idx_step: int,
    key: PRNGKeyArray,
    electrons: Electrons,
    width: Width,
    max_cluster_size: int,
) -> tuple[Electrons, ElectronIdx, jax.Array, Int]:
    dtype = electrons.dtype
    n_el = electrons.shape[-2]
    idx_center = idx_step % len(R)
    R_center = R[idx_center]

    rng_select, rng_move = jax.random.split(key)
    dist_before_move = jnp.linalg.norm(electrons - R_center, axis=-1)
    log_p_select1 = -dist_before_move / cluster_radius

    do_move = log_p_select1 >= jnp.log(jax.random.uniform(rng_select, (n_el,), dtype))
    idx_el_changed = jnp.nonzero(do_move, fill_value=NO_NEIGHBOUR, size=max_cluster_size)[0]
    dr = jax.random.normal(rng_move, (max_cluster_size, 3), dtype) * width
    proposed_electrons = electrons.at[idx_el_changed].add(dr, mode="drop")

    dist_after_move = jnp.linalg.norm(proposed_electrons - R_center, axis=-1)
    log_p_select2 = -dist_after_move / cluster_radius
    proposal_log_ratio = jnp.sum(log_p_select2) - jnp.sum(log_p_select1)
    actual_cluster_size = jnp.sum(do_move).astype(jnp.int32)
    return proposed_electrons, idx_el_changed, proposal_log_ratio, actual_cluster_size


def mh_update_low_rank(
    update_logprob_fn: Callable,
    proposal_fn: Callable,
    max_cluster_size: int,
    idx_step: int,
    key: PRNGKeyArray,
    electrons: Electrons,
    log_prob: LogAmplitude,
    model_state: dict,
    stats: dict,
    width: Width,
) -> tuple[PRNGKeyArray, Electrons, LogAmplitude, dict, dict]:
    # Make proposal
    key, key_propose, key_accept = jax.random.split(key, 3)
    proposed_electrons, idx_el_changed, proposal_log_ratio, actual_cluster_size = proposal_fn(
        idx_step, key_propose, electrons, width, max_cluster_size
    )

    # Accept/reject
    proposed_logprob, proposed_model_state = update_logprob_fn(proposed_electrons, idx_el_changed, model_state)
    log_ratio = proposal_log_ratio + proposed_logprob - log_prob
    accept = log_ratio > jnp.log(jax.random.uniform(key_accept, log_ratio.shape))
    new_electrons, new_logprob, new_model_state = jtu.tree_map(
        lambda new, old: jnp.where(accept, new, old),
        (proposed_electrons, proposed_logprob, proposed_model_state),
        (electrons, log_prob, model_state),
    )
    stats["num_accepts"] += jnp.sum(accept).astype(jnp.int32)
    stats["max_cluster_size"] = jnp.maximum(stats["max_cluster_size"], actual_cluster_size)
    stats["mean_cluster_size"] = stats["mean_cluster_size"] + actual_cluster_size
    return key, new_electrons, new_logprob, new_model_state, stats


P, S, MS = TypeVar("P"), TypeVar("S"), TypeVar("MS")


def mcmc_steps_all_electron(
    logpsi_fn: ParameterizedWaveFunction[P, S, MS],
    steps: int,
    key: PRNGKeyArray,
    params: P,
    electrons: Electrons,
    static: StaticInput[S],
    width: Width,
):
    def log_prob_fn(electrons: Electrons) -> LogAmplitude:
        return 2 * jax.vmap(lambda r: logpsi_fn(params, r, static.model))(electrons)

    def step_fn(_, x):
        return mh_update_all_electron(log_prob_fn, *x, width)  # type: ignore

    local_batch_size = electrons.shape[0]
    logprob = log_prob_fn(electrons)
    stats = dict(num_accepts=jnp.zeros((), dtype=jnp.int32))
    key, electrons, logprob, stats = lax.fori_loop(0, steps, step_fn, (key, electrons, logprob, stats))
    summary_stats = {"mcmc/pmove": psum_if_pmap(stats["num_accepts"]) / (steps * local_batch_size * jax.device_count())}
    return electrons, summary_stats


def mcmc_steps_low_rank(
    logpsi_fn: ParameterizedWaveFunction[P, S, MS],
    proposal_fn: Callable,
    steps: int,
    key: PRNGKeyArray,
    params: P,
    electrons: Electrons,
    static: StaticInput[S],
    width: Width,
):
    def log_prob_fn(r: Electrons):
        (_, logpsi), model_state = logpsi_fn.log_psi_with_state(params, r, static.model)
        return 2 * logpsi, model_state

    def update_log_prob_fn(r: Electrons, idx_changed: ElectronIdx, model_state):
        (_, logpsi), model_state = logpsi_fn.log_psi_low_rank_update(params, r, idx_changed, static.model, model_state)
        return 2 * logpsi, model_state

    logprob, model_state = jax.vmap(log_prob_fn)(electrons)

    @functools.partial(jax.vmap, in_axes=(None, 0))
    def step_fn(i, x):
        return mh_update_low_rank(update_log_prob_fn, proposal_fn, static.mcmc.max_cluster_size, i, *x, width)

    local_batch_size = electrons.shape[0]
    stats = dict(
        num_accepts=jnp.zeros(local_batch_size, dtype=jnp.int32),
        max_cluster_size=jnp.zeros(local_batch_size, dtype=jnp.int32),
        mean_cluster_size=jnp.zeros(local_batch_size, dtype=jnp.int32),
    )
    x0 = (
        jax.random.split(key, local_batch_size),
        electrons,
        logprob,
        model_state,
        stats,
    )
    _, electrons, logprob, model_state, stats = lax.fori_loop(0, steps, step_fn, x0)
    summary_stats = {
        "mcmc/pmove": pmean_if_pmap(jnp.mean(stats["num_accepts"]) / steps),
        "mcmc/max_cluster_size": pmax_if_pmap(jnp.max(stats["max_cluster_size"])),
        "mcmc/mean_cluster_size": pmean_if_pmap(jnp.mean(stats["mean_cluster_size"]) / steps),
    }
    return electrons, summary_stats


def make_mcmc(
    logpsi_fn: ParameterizedWaveFunction[P, S, MS],
    R: Nuclei,
    n_el: int,
    mcmc_args: MCMCArgs,
) -> tuple[MCStep[P, S], jax.Array]:
    proposal = mcmc_args["proposal"]
    proposal_args = dict(**mcmc_args[f"{proposal.replace('-', '_')}_args"])  # type: ignore
    init_width = proposal_args["init_width"]
    match proposal.lower():
        case "all-electron":
            steps = proposal_args["steps"]
            mcmc_step = functools.partial(mcmc_steps_all_electron, logpsi_fn, steps)
        case "single-electron":
            steps = proposal_args["sweeps"] * n_el
            mcmc_step = functools.partial(mcmc_steps_low_rank, logpsi_fn, proposal_single_electron, steps)
        case "cluster-update":
            cluster_radius = proposal_args["cluster_radius"]
            proposal_fn = functools.partial(proposal_cluster_update, jnp.array(R, jnp.float32), cluster_radius)
            steps = proposal_args["sweeps"] * len(R)
            mcmc_step = functools.partial(mcmc_steps_low_rank, logpsi_fn, proposal_fn, steps)
        case _:
            raise ValueError(f"Invalid proposal: {proposal}")
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


def round_with_padding(n: int, max_val: int, factor: float = 1.1, min_val: int = 1) -> int:
    power = np.log(n) / np.log(factor)
    n_padded = factor ** np.ceil(power)
    return int(np.clip(n_padded, min_val, max_val))


class ClusterSizeScheduler:
    def __init__(self, n_el, cluster_size_buffer=2):
        self.n_el = n_el
        self.cluster_size_buffer = cluster_size_buffer

    def step(self, mcmc_stats: dict) -> MCMCStaticArgs:
        # Stats should already by averaged over batch and devices, but will have a local device dimension which must be reduced
        mcmc_stats = jax.tree_map(lambda x: x.mean(), mcmc_stats)
        max_cluster_size = int(mcmc_stats.get("mcmc/max_cluster_size", self.n_el))
        max_cluster_size = round_with_padding(max_cluster_size + self.cluster_size_buffer, self.n_el)
        return MCMCStaticArgs(max_cluster_size=max_cluster_size)


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
