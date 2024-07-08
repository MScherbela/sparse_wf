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
    StaticInput,
)
from sparse_wf.jax_utils import pmap, pmean

P, S, MS, SS = TypeVar("P"), TypeVar("S"), TypeVar("MS"), TypeVar("SS")


def make_pretrainer(
    wave_function: ParameterizedWaveFunction[P, S, MS],
    mcmc_step: MCStep[P, S],
    width_scheduler: WidthScheduler,
    source_model: HFOrbitalFn,
    optimizer: optax.GradientTransformation,
) -> Pretrainer[P, S, SS]:
    batch_orbitals = jax.vmap(wave_function.orbitals, in_axes=(None, 0, None))
    batch_src_orbitals = jax.vmap(source_model, in_axes=(0,))

    def init(training_state: TrainingState[P, SS]):
        return PretrainState(
            training_state.key,
            training_state.params,
            training_state.electrons,
            training_state.opt_state,
            training_state.width_state,
            training_state.spin_state,
            pre_opt_state=pmap(optimizer.init)(training_state.params),
        )

    @pmap(static_broadcasted_argnums=1)
    def step(state: PretrainState[P, SS], static: StaticInput[S]) -> tuple[PretrainState[P, SS], AuxData]:
        targets = wave_function.hf_transformation(batch_src_orbitals(state.electrons))

        @jax.value_and_grad
        def loss_and_grad(params):
            predicted_orbitals = batch_orbitals(params, state.electrons, static.model)
            return sum(((o - p_o) ** 2).mean() for o, p_o in zip(targets, predicted_orbitals))

        # Update
        loss_val, grad = pmean(loss_and_grad(state.params))
        updates, opt_state = optimizer.update(grad, state.pre_opt_state, state.params)
        params = optax.apply_updates(state.params, updates)

        # MCMC
        key, subkey = jax.random.split(state.key)
        electrons, stats = mcmc_step(subkey, params, state.electrons, static, state.width_state.width)
        width_state = width_scheduler.update(state.width_state, stats["mcmc/pmove"])

        return state.replace(
            key=key,
            params=params,
            electrons=electrons,
            pre_opt_state=opt_state,
            width_state=width_state,
        ), {"pretrain/loss": loss_val, "mcmc/stepsize": state.width_state.width, **stats}

    return Pretrainer(init, step)
