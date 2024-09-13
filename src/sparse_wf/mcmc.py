from typing import TypeVar, Callable, Optional, Literal

import jax
import jax.numpy as jnp
import numpy as np
import pyscf
from jax import lax
import jax.tree_util as jtu
import functools

from sparse_wf.api import (
    Charges,
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
    StaticInput,
    MCMCStats,
)
from sparse_wf.jax_utils import jit, pmean_if_pmap, pmax_if_pmap
from sparse_wf.model.graph_utils import NO_NEIGHBOUR
from sparse_wf.tree_utils import tree_add, tree_maximum, tree_zeros_like


P, S, MS = TypeVar("P"), TypeVar("S"), TypeVar("MS")


def mcmc_steps_all_electron(
    logpsi_fn: ParameterizedWaveFunction[P, S, MS],
    get_static_fn: Callable,
    steps: int,
    key: PRNGKeyArray,
    params: P,
    electrons: Electrons,
    static: StaticInput,
    width: Width,
):
    n_el = electrons.shape[-2]

    def log_prob_fn(electrons: Electrons) -> LogAmplitude:
        return 2 * logpsi_fn(params, electrons, static)

    @functools.partial(jax.vmap, in_axes=(None, 0))
    def step_fn(_, carry):
        key, electrons, log_prob, static_mean, static_max, num_accept = carry
        key, key_propose, key_accept = jax.random.split(key, 3)

        # Make proposal
        eps = jax.random.normal(key_propose, electrons.shape, dtype=electrons.dtype) * width
        new_electrons = electrons + eps
        new_log_prob = log_prob_fn(new_electrons)

        # Track actual static neigbour counts
        actual_static = get_static_fn(electrons, new_electrons, np.arange(n_el))
        static_mean = tree_add(static_mean, actual_static)
        static_max = tree_maximum(static_max, actual_static)

        # Accept/reject
        log_ratio = new_log_prob - log_prob
        u = jax.random.uniform(key_accept, log_ratio.shape)
        accept = log_ratio > jnp.log(u)
        new_electrons = jnp.where(accept, new_electrons, electrons)
        new_log_prob = jnp.where(accept, new_log_prob, log_prob)
        num_accept += accept.astype(jnp.int32)
        return key, new_electrons, new_log_prob, static_mean, static_max, num_accept

    local_batch_size = electrons.shape[0]
    logprob = jax.vmap(log_prob_fn)(electrons)
    actual_static = jax.vmap(get_static_fn)(electrons)
    x0 = (
        jax.random.split(key, local_batch_size),
        electrons,
        logprob,
        tree_zeros_like(actual_static, jnp.int32, local_batch_size),
        tree_zeros_like(actual_static, jnp.int32, local_batch_size),
        jnp.zeros(local_batch_size, dtype=jnp.int32),
    )
    _, electrons, _, static_mean, static_max, num_accept = lax.fori_loop(0, steps, step_fn, x0)
    stats = MCMCStats(
        pmove=pmean_if_pmap(jnp.mean(num_accept) / steps),
        stepsize=width,
        static_mean=jtu.tree_map(lambda x: pmean_if_pmap(jnp.mean(x) / steps), static_mean),
        static_max=jtu.tree_map(lambda x: pmax_if_pmap(jnp.max(x)), static_max),
    )
    return electrons, stats


def proposal_single_electron(
    idx_step: int, key: PRNGKeyArray, electrons: Electrons, width: Width
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


def _get_closest_k(dist, dist_max, k):
    neg_d, idx = jax.lax.top_k(-dist, k=k)
    idx = jnp.where(neg_d >= -dist_max, idx, NO_NEIGHBOUR)
    return idx


def _is_same_set(idx_a, idx_b):
    return jnp.all(jnp.sort(idx_a) == jnp.sort(idx_b))


def proposal_cluster_update(
    max_cluster_size: int,
    max_cluster_radius: float,
    sweep_type: Literal["sequential", "random"],
    idx_step: int,
    key: PRNGKeyArray,
    electrons: Electrons,
    width: Width,
) -> tuple[Electrons, ElectronIdx, jax.Array, Int]:
    dtype = electrons.dtype
    n_el = electrons.shape[-2]

    if sweep_type == "random":
        key, key_select = jax.random.split(key)
        idx_center = jax.random.randint(key_select, [], 0, n_el, jnp.int32)
    elif sweep_type == "sequential":
        idx_center = jnp.array(idx_step, jnp.int32) % n_el
    else:
        raise ValueError(f"Invalid sweep_type: {sweep_type}")

    dist_before_move = jnp.linalg.norm(electrons - electrons[idx_center], axis=-1)
    idx_el_changed = _get_closest_k(dist_before_move, max_cluster_radius, max_cluster_size)
    actual_cluster_size = jnp.sum(idx_el_changed != NO_NEIGHBOUR).astype(jnp.int32)

    def _propose(rng):
        dr = jax.random.normal(rng, (max_cluster_size, 3), dtype) * width
        r_proposed = electrons.at[idx_el_changed].add(dr, mode="drop")
        dist_after_move = jnp.linalg.norm(r_proposed - r_proposed[idx_center], axis=-1)
        idx_el_changed_reverse = _get_closest_k(dist_after_move, max_cluster_radius, max_cluster_size)
        is_valid = _is_same_set(idx_el_changed, idx_el_changed_reverse)
        return r_proposed, is_valid.astype(jnp.float32)

    n_trials = 5
    proposed_electrons, is_valid = jax.vmap(_propose)(jax.random.split(key, n_trials))
    idx_trial = jnp.argmax(is_valid)
    proposed_electrons = proposed_electrons[idx_trial]
    proposal_log_ratio = jnp.log(is_valid[idx_trial])
    return proposed_electrons, idx_el_changed, proposal_log_ratio, actual_cluster_size


def mcmc_steps_low_rank(
    logpsi_fn: ParameterizedWaveFunction[P, S, MS],
    get_static_fn: Callable,
    proposal_fn: Callable,
    steps: int,
    key: PRNGKeyArray,
    params: P,
    electrons: Electrons,
    static: StaticInput,
    width: Width,
):
    def log_prob_fn(r: Electrons):
        (_, logpsi), model_state = logpsi_fn.log_psi_with_state(params, r, static)
        return 2 * logpsi, model_state

    def update_log_prob_fn(r: Electrons, idx_changed: ElectronIdx, model_state):
        (_, logpsi), model_state = logpsi_fn.log_psi_low_rank_update(params, r, idx_changed, static, model_state)
        return 2 * logpsi, model_state

    logprob, model_state = jax.vmap(log_prob_fn)(electrons)

    @functools.partial(jax.vmap, in_axes=(None, 0))
    def step_fn(i, carry):
        key, electrons, log_prob, model_state, static_mean, static_max, num_accept, mean_cluster_size = carry
        key, key_propose, key_accept = jax.random.split(key, 3)

        # Make proposal
        proposed_electrons, idx_el_changed, proposal_log_ratio, actual_cluster_size = proposal_fn(
            i, key_propose, electrons, width
        )

        # Track actual static neigbour counts
        actual_static = get_static_fn(electrons, proposed_electrons, idx_el_changed)
        static_mean = tree_add(static_mean, actual_static)
        static_max = tree_maximum(static_max, actual_static)

        # Accept/reject
        proposed_logprob, proposed_model_state = update_log_prob_fn(proposed_electrons, idx_el_changed, model_state)
        log_ratio = proposal_log_ratio + proposed_logprob - log_prob
        accept = log_ratio > jnp.log(jax.random.uniform(key_accept, log_ratio.shape))
        electrons, log_prob, model_state = jtu.tree_map(
            lambda new, old: jnp.where(accept, new, old),
            (proposed_electrons, proposed_logprob, proposed_model_state),
            (electrons, log_prob, model_state),
        )
        num_accept += accept.astype(jnp.int32)
        mean_cluster_size += actual_cluster_size
        return key, electrons, log_prob, model_state, static_mean, static_max, num_accept, mean_cluster_size

    local_batch_size = electrons.shape[0]
    actual_static = jax.vmap(get_static_fn)(electrons)
    x0 = (
        jax.random.split(key, local_batch_size),
        electrons,
        logprob,
        model_state,
        tree_zeros_like(actual_static, jnp.int32, local_batch_size),
        tree_zeros_like(actual_static, jnp.int32, local_batch_size),
        jnp.zeros(local_batch_size, dtype=jnp.int32),
        jnp.zeros(local_batch_size, dtype=jnp.int32),
    )
    _, electrons, _, _, static_mean, static_max, num_accept, mean_cluster_size = lax.fori_loop(0, steps, step_fn, x0)
    stats = MCMCStats(
        pmove=pmean_if_pmap(jnp.mean(num_accept) / steps),
        stepsize=width,
        static_mean=jtu.tree_map(lambda x: pmean_if_pmap(jnp.mean(x) / steps), static_mean),
        static_max=jtu.tree_map(lambda x: pmax_if_pmap(jnp.max(x)), static_max),
        mean_cluster_size=pmean_if_pmap(jnp.mean(mean_cluster_size) / steps),
    )
    return electrons, stats


def make_mcmc(
    logpsi_fn: ParameterizedWaveFunction[P, S, MS],
    R: Nuclei,
    n_el: int,
    mcmc_args: MCMCArgs,
    get_static_fn: Optional[Callable] = None,
) -> tuple[MCStep[P, S], jax.Array]:
    proposal = mcmc_args["proposal"]
    proposal_args = dict(**mcmc_args[f"{proposal.replace('-', '_')}_args"])  # type: ignore
    init_width = proposal_args["init_width"]

    get_static_fn = get_static_fn or logpsi_fn.get_static_input

    match proposal.lower():
        case "all-electron":
            steps = proposal_args["steps"]
            mcmc_step = functools.partial(mcmc_steps_all_electron, logpsi_fn, get_static_fn, steps)
        case "single-electron":
            steps = proposal_args["sweeps"] * n_el
            mcmc_step = functools.partial(
                mcmc_steps_low_rank, logpsi_fn, get_static_fn, proposal_single_electron, steps
            )
        case "cluster-update":
            proposal_fn = functools.partial(
                proposal_cluster_update,
                min(proposal_args["max_cluster_size"], n_el),
                proposal_args["max_cluster_radius"],
                proposal_args["sweep_type"],
            )
            steps = int(proposal_args["sweeps"] * n_el)
            if proposal_args["sweep_type"] == "sequential":
                assert steps % n_el == 0, "Number of sweeps must be integer for sequential sweep"
            assert steps > 0, "Number of steps (sweeps * n_el) must be greater than 0"
            mcmc_step = functools.partial(mcmc_steps_low_rank, logpsi_fn, get_static_fn, proposal_fn, steps)
        case _:
            raise ValueError(f"Invalid proposal: {proposal}")
    return mcmc_step, jnp.array(init_width, dtype=jnp.float32)


def make_width_scheduler(
    window_size: int = 20,
    target_pmove: float = 0.5,
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
    if jax.device_count() > 1:
        batch_size = batch_size - (batch_size % jax.device_count())
        local_batch_size = (batch_size // jax.device_count()) * jax.local_device_count()
    else:
        local_batch_size = batch_size
    electrons = jax.random.normal(key, (local_batch_size, mol.nelectron, 3), dtype=jnp.float32)

    R = np.array(mol.atom_coords(), dtype=jnp.float32)
    n_atoms = len(R)
    if n_atoms > 1:
        assert mol.charge == 0, "Only atoms or neutral molecules are supported"
        assert abs(mol.spin) < 2, "Only atoms or singlet and doublet molecules are supported"  # type: ignore
        ind_atom = assign_spins_to_atoms(R, mol.atom_charges())
        electrons += R[ind_atom]
    return electrons
