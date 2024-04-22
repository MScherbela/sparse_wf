from enum import Enum

import jax
import jax.numpy as jnp
import optax

from sparse_wf.api import (
    AuxData,
    ClippingArgs,
    Electrons,
    EnergyFn,
    LocalEnergy,
    MCStep,
    Preconditioner,
    OptState,
    Parameters,
    ParameterizedWaveFunction,
    PRNGKeyArray,
    StaticInput,
    Trainer,
    TrainingState,
    Width,
    WidthScheduler,
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


def make_trainer(
    wave_function: ParameterizedWaveFunction,
    energy_function: EnergyFn,
    mcmc_step: MCStep,
    width_scheduler: WidthScheduler,
    optimizer: optax.GradientTransformation,
    preconditioner: Preconditioner,
    clipping_args: ClippingArgs,
) -> Trainer:
    def init(
        key: PRNGKeyArray,
        params: Parameters,
        electrons: Electrons,
        init_width: Width,
    ) -> TrainingState:
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
        )

    @pmap(static_broadcasted_argnums=1)
    def step(
        state: TrainingState,
        static: StaticInput,
    ) -> tuple[TrainingState, LocalEnergy, AuxData]:
        key, subkey = jax.random.split(state.key)
        electrons, pmove = mcmc_step(subkey, state.params, state.electrons, static, state.width_state.width)
        width_state = width_scheduler.update(state.width_state, pmove)
        energy_dense = batched_vmap(lambda r: energy_function(state.params, r, static), max_batch_size=64)(electrons)
        energy = wave_function.local_energy_sparse(state.params, electrons, static)  # type: ignore
        energy_diff = local_energy_diff(energy, **clipping_args)

        E_mean = pmean(energy.mean())
        E_std = pmean(((energy - E_mean) ** 2).mean()) ** 0.5
        aux_data = {"opt/E": E_mean, "opt/E_std": E_std, "mcmc/pmove": pmove, "mcmc/stepsize": state.width_state.width}
        aux_data["opt/E_sparse"] = pmean(energy.mean())
        aux_data["opt/E_dense"] = pmean(energy_dense.mean())
        natgrad = state.opt_state.natgrad
        gradient, natgrad, preconditioner_aux = preconditioner.precondition(
            state.params,
            electrons,
            static,
            energy_diff,
            state.opt_state.natgrad,  # type: ignore
        )
        aux_data.update(preconditioner_aux)
        aux_data["update_norm"] = tree_dot(gradient, gradient) ** 0.5

        updates, opt = optimizer.update(gradient, state.opt_state.opt, state.params)
        params = optax.apply_updates(state.params, updates)

        return (
            TrainingState(
                key=key,
                params=params,
                electrons=electrons,
                opt_state=OptState(opt, natgrad),
                width_state=width_state,
            ),
            energy,
            aux_data,
        )

    return Trainer(
        init=init,
        step=step,
        wave_function=wave_function,
        mcmc=mcmc_step,
        width_scheduler=width_scheduler,
        energy_fn=energy_function,
        optimizer=optimizer,
        preconditioner=preconditioner,
    )
