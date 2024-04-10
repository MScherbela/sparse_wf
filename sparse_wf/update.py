from enum import Enum

import jax
import jax.numpy as jnp
import optax

from sparse_wf.api import (
    Array,
    AuxData,
    Electrons,
    EnergyFn,
    LocalEnergy,
    MCStep,
    NaturalGradient,
    NaturalGradientOptState,
    Parameters,
    ParameterizedWaveFunction,
    PRNGKeyArray,
    StaticInput,
    Trainer,
    TrainingState,
    Width,
    WidthScheduler,
)
from sparse_wf.jax_utils import jit, pgather
from sparse_wf.tree_utils import tree_dot


class ClipStatistic(Enum):
    MEDIAN = "median"
    MEAN = "mean"


def clip_local_energies(e_loc: LocalEnergy, clip_local_energy: float, stat: str | ClipStatistic) -> LocalEnergy:
    stat = ClipStatistic(stat)

    def stat_fn(x):
        match stat:
            case ClipStatistic.MEAN:
                return jnp.mean(x, keepdims=True)
            case ClipStatistic.MEDIAN:
                return jnp.median(x, keepdims=True)
            case _:
                raise ValueError(f"Unknown statistic: {stat}")

    if clip_local_energy > 0.0:
        full_e = pgather(e_loc, axis=0, tiled=True)
        clip_center = stat_fn(full_e)
        mad = jnp.mean(jnp.abs(full_e - clip_center), keepdims=True)
        max_dev = clip_local_energy * mad
        e_loc = jnp.clip(e_loc, clip_center - max_dev, clip_center + max_dev)
    return e_loc


def local_energy_diff(e_loc: LocalEnergy, clip_local_energy: float, stat: str | ClipStatistic) -> LocalEnergy:
    e_loc = clip_local_energies(e_loc, clip_local_energy, stat)
    center = jnp.mean(e_loc, keepdims=True)
    e_loc -= center
    return e_loc


def make_trainer(
    wave_function: ParameterizedWaveFunction,
    energy_function: EnergyFn,
    mcmc_step: MCStep,
    width_scheduler: WidthScheduler,
    optimizer: optax.GradientTransformation,
    natural_gradient: NaturalGradient | None = None,
) -> Trainer:
    batch_log_amplitude = jax.vmap(wave_function, in_axes=(None, 0, 0))

    def init(
        key: PRNGKeyArray,
        params: Parameters,
        electrons: Electrons,
        init_width: Width,
    ) -> TrainingState:
        return TrainingState(
            params=params,
            opt_state=NaturalGradientOptState(
                opt=optimizer.init(params),
                natgrad=natural_gradient.init(params) if natural_gradient is not None else None,
            ),
            electrons=electrons,
            width_state=width_scheduler.init(init_width),
        )

    @jit
    def step(
        key: PRNGKeyArray,
        state: TrainingState,
        static: StaticInput,
    ) -> tuple[TrainingState, LocalEnergy, AuxData]:
        key, subkey = jax.random.split(key)
        electrons, pmove = mcmc_step(subkey, state.params, state.electrons, static, state.width_state.width)
        width_state = width_scheduler.update(state.width_state, pmove)
        energy = energy_function(state.params, electrons, static)
        energy_diff = local_energy_diff(energy, 5.0, "median")

        # TODO (ms,ng): Does this function actually parallelize across gpus when using pmap? S
        # Seems like it only avarages across the local devices...

        def loss_fn(params):
            return jnp.vdot(batch_log_amplitude(params, electrons, static), energy_diff) / energy_diff.size

        gradient = jax.grad(loss_fn)(state.params)
        aux_data: dict[str, Array] = {
            "E": energy.mean(),
            "E_std": energy.std(),
            "pmove": pmove.mean(),
            "grad_norm": jnp.sqrt(tree_dot(gradient, gradient)),
        }
        natgrad = state.opt_state.natgrad
        if natural_gradient is not None:
            gradient, natgrad, nat_aux_data = natural_gradient.precondition(
                state.params,
                electrons,
                static,
                energy_diff,
                state.opt_state.natgrad,  # type: ignore
            )
            aux_data.update(nat_aux_data)

        updates, opt = optimizer.update(gradient, state.opt_state.opt, state.params)
        params = optax.apply_updates(state.params, updates)

        return (
            TrainingState(
                params=params,
                electrons=electrons,
                opt_state=NaturalGradientOptState(opt, natgrad),
                width_state=width_state,
            ),
            energy,
            aux_data,
        )

    return Trainer(
        init=init,
        update=step,
        wave_function=wave_function,
        mcmc=mcmc_step,
        width_scheduler=width_scheduler,
        energy_fn=energy_function,
        optimizer=optimizer,
        natgrad=natural_gradient,
    )
