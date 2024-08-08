from enum import Enum
from typing import TypeVar

import jax
import jax.numpy as jnp
import optax

from sparse_wf.api import (
    ClippingArgs,
    Electrons,
    LocalEnergy,
    MCStep,
    OptState,
    ParameterizedWaveFunction,
    Preconditioner,
    PRNGKeyArray,
    SpinOperator,
    Trainer,
    TrainingState,
    Width,
    WidthScheduler,
    StaticInput,
)
from sparse_wf.jax_utils import pgather, pmap, pmean, replicate
from sparse_wf.tree_utils import tree_dot

from folx import batched_vmap


class ClipStatistic(Enum):
    MEDIAN = "median"
    MEAN = "mean"


def clip_local_energies(e_loc: LocalEnergy, clip_local_energy: float, stat: str | ClipStatistic) -> LocalEnergy:
    stat = ClipStatistic(stat)

    def stat_fn(x):
        if stat == ClipStatistic.MEAN:
            return pmean(jnp.mean(x, keepdims=True))
        elif stat == ClipStatistic.MEDIAN:
            return pmean(jnp.median(x, keepdims=True))
        else:
            raise ValueError(f"Unknown statistic: {stat}")

    if clip_local_energy > 0.0:
        full_e = pgather(e_loc, axis=0, tiled=True)
        clip_center = stat_fn(full_e)
        mad = pmean(jnp.mean(jnp.abs(full_e - clip_center), keepdims=True))
        max_dev = clip_local_energy * mad
        e_loc = jnp.clip(e_loc, clip_center - max_dev, clip_center + max_dev)
    return e_loc


def local_energy_diff(e_loc: LocalEnergy, clip_local_energy: float, stat: str | ClipStatistic) -> LocalEnergy:
    e_loc = clip_local_energies(e_loc, clip_local_energy, stat)
    center = pmean(jnp.mean(e_loc, keepdims=True))
    e_loc -= center
    return e_loc


P, S, MS, SS = TypeVar("P"), TypeVar("S"), TypeVar("MS"), TypeVar("SS")


def make_trainer(
    wave_function: ParameterizedWaveFunction[P, S, MS],
    mcmc_step: MCStep[P, S],
    width_scheduler: WidthScheduler,
    optimizer: optax.GradientTransformation,
    preconditioner: Preconditioner[P, S],
    clipping_args: ClippingArgs,
    max_batch_size: int,
    spin_operator: SpinOperator[P, S, SS],
    energy_operator: str,
):
    match energy_operator.lower():
        case "dense":
            energy_fn = wave_function.local_energy_dense
        case "sparse":
            energy_fn = wave_function.local_energy
        case _:
            raise ValueError(f"Unknown energy operator: {energy_operator}")

    def init(key: PRNGKeyArray, params: P, electrons: Electrons, init_width: Width):
        params = replicate(params)
        return TrainingState(
            key=key,
            params=params,
            opt_state=OptState(
                opt=pmap(optimizer.init)(params),
                natgrad=pmap(preconditioner.init)(params),
            ),
            electrons=electrons.reshape(jax.local_device_count(), -1, *electrons.shape[1:]),
            width_state=replicate(width_scheduler.init(init_width)),
            spin_state=replicate(spin_operator.init_state()),
        )

    @pmap(static_broadcasted_argnums=(1, 2))
    def sampling_step(state: TrainingState[P, SS], static: StaticInput, compute_energy: bool):
        key, subkey = jax.random.split(state.key)
        electrons, stats = mcmc_step(subkey, state.params, state.electrons, static, state.width_state.width)
        width_state = width_scheduler.update(state.width_state, stats.pmove)
        state = state.replace(key=key, electrons=electrons, width_state=width_state)
        aux_data = {}
        if compute_energy:
            E_loc = batched_vmap(energy_fn, in_axes=(None, 0, None), max_batch_size=max_batch_size)(
                state.params, electrons, static
            )
            E_mean = pmean(E_loc.mean())
            E_std = pmean(((E_loc - E_mean) ** 2).mean()) ** 0.5
            aux_data["eval/E"] = E_mean
            aux_data["eval/E_std"] = E_std
        return state, aux_data, stats

    @pmap(static_broadcasted_argnums=1)
    def step(state: TrainingState[P, SS], static: StaticInput):
        key, subkey = jax.random.split(state.key)
        electrons, stats = mcmc_step(subkey, state.params, state.electrons, static, state.width_state.width)
        width_state = width_scheduler.update(state.width_state, stats.pmove)
        energy = batched_vmap(energy_fn, in_axes=(None, 0, None), max_batch_size=max_batch_size)(
            state.params, electrons, static
        )
        energy_diff = local_energy_diff(energy, **clipping_args)

        E_mean = pmean(energy.mean())
        E_std = pmean(((energy - E_mean) ** 2).mean()) ** 0.5
        aux_data = {"opt/E": E_mean, "opt/E_std": E_std}

        spin_op_value, spin_grad, spin_state = spin_operator(state.params, electrons, static, state.spin_state)
        aux_data["opt/S"] = spin_op_value

        natgrad, precond_state, preconditioner_aux = preconditioner.precondition(
            state.params,
            electrons,
            static,
            energy_diff,
            spin_grad,
            state.opt_state.natgrad,
        )
        aux_data.update(preconditioner_aux)
        aux_data["opt/update_norm"] = tree_dot(natgrad, natgrad) ** 0.5
        aux_data["opt/elec_max_extend"] = jnp.abs(electrons).max()  # TODO: remove?

        updates, opt = optimizer.update(natgrad, state.opt_state.opt, state.params)
        params = optax.apply_updates(state.params, updates)

        return (
            state.replace(
                key=key,
                params=params,
                electrons=electrons,
                opt_state=OptState(opt, precond_state),
                width_state=width_state,
                spin_state=spin_state,
            ),
            energy,
            aux_data,
            stats,
        )

    return Trainer(
        init=init,
        step=step,
        sampling_step=sampling_step,
        wave_function=wave_function,
        mcmc=mcmc_step,
        width_scheduler=width_scheduler,
        optimizer=optimizer,
        preconditioner=preconditioner,
        spin_operator=spin_operator,
    )
