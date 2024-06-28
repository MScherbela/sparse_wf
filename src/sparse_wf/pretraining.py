from typing import TypeVar

import jax
import optax

from sparse_wf.api import (
    AuxData,
    HFOrbitalFn,
    Pretrainer,
    PretrainState,
    TrainingState,
    ParameterizedWaveFunction,
    MCStep,
    WidthScheduler,
)
from sparse_wf.jax_utils import pmap, pmean

P, S, MS = TypeVar("P"), TypeVar("S"), TypeVar("MS")


def make_pretrainer(
    wave_function: ParameterizedWaveFunction[P, S, MS],
    mcmc_step: MCStep[P, S],
    width_scheduler: WidthScheduler,
    source_model: HFOrbitalFn,
    optimizer: optax.GradientTransformation,
) -> Pretrainer[P, S]:
    batch_orbitals = jax.vmap(wave_function.orbitals, in_axes=(None, 0, None))
    batch_src_orbitals = jax.vmap(source_model, in_axes=(0,))

    def init(training_state: TrainingState[P]):
        return PretrainState(
            training_state.key,
            training_state.params,
            training_state.electrons,
            training_state.opt_state,
            training_state.width_state,
            pre_opt_state=pmap(optimizer.init)(training_state.params),
        )

    @pmap(static_broadcasted_argnums=1)
    def step(state: PretrainState[P], static: S) -> tuple[PretrainState[P], AuxData]:
        targets = wave_function.hf_transformation(batch_src_orbitals(state.electrons))

        @jax.value_and_grad
        def loss_and_grad(params):
            predicted_orbitals = batch_orbitals(params, state.electrons, static)
            return sum(((o - p_o) ** 2).mean() for o, p_o in zip(targets, predicted_orbitals))

        # Update
        loss_val, grad = pmean(loss_and_grad(state.params))
        updates, opt_state = optimizer.update(grad, state.pre_opt_state, state.params)
        params = optax.apply_updates(state.params, updates)

        # MCMC
        key, subkey = jax.random.split(state.key)
        electrons, pmove = mcmc_step(subkey, params, state.electrons, static, state.width_state.width)
        width_state = width_scheduler.update(state.width_state, pmove)

        return state.replace(
            key=key,
            params=params,
            electrons=electrons,
            pre_opt_state=opt_state,
            width_state=width_state,
        ), {
            "pretrain/loss": loss_val,
            "mcmc/pmove": pmove,
            "mcmc/stepsize": state.width_state.width,
        }

    return Pretrainer(init, step)
