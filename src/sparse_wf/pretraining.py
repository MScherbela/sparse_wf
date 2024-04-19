import jax
import optax

from sparse_wf.api import (
    HFOrbitalFn,
    AuxData,
    Parameters,
    Pretrainer,
    PretrainState,
    StaticInput,
    Trainer,
    TrainingState,
)
from sparse_wf.jax_utils import pmap, pmean


def make_pretrainer(trainer: Trainer, source_model: HFOrbitalFn, optimizer: optax.GradientTransformation) -> Pretrainer:
    batch_orbitals = jax.vmap(trainer.wave_function.orbitals, in_axes=(None, 0, None))
    batch_src_orbitals = jax.vmap(source_model, in_axes=(0,))

    def init(training_state: TrainingState):
        return PretrainState(
            training_state.key,
            training_state.params,
            training_state.electrons,
            training_state.opt_state,
            training_state.width_state,
            pre_opt_state=pmap(optimizer.init)(training_state.params),
        )

    @pmap(static_broadcasted_argnums=1)
    def step(state: PretrainState, static: StaticInput) -> tuple[PretrainState, AuxData]:
        targets = trainer.wave_function.hf_transformation(batch_src_orbitals(state.electrons))

        @jax.value_and_grad
        def loss_and_grad(params: Parameters):
            predicted_orbitals = batch_orbitals(params, state.electrons, static)
            return sum(((o - p_o) ** 2).mean() for o, p_o in zip(targets, predicted_orbitals))

        # Update
        loss_val, grad = pmean(loss_and_grad(state.params))
        updates, opt_state = optimizer.update(grad, state.pre_opt_state, state.params)
        params = optax.apply_updates(state.params, updates)

        # MCMC
        key, subkey = jax.random.split(state.key)
        electrons, pmove = trainer.mcmc(subkey, params, state.electrons, static, state.width_state.width)
        width_state = trainer.width_scheduler.update(state.width_state, pmove)

        return state.replace(
            key=key,
            params=params,
            electrons=electrons,
            pre_opt_state=opt_state,
            width_state=width_state,
        ), {
            "loss": loss_val,
            "pmove": pmove,
        }

    return Pretrainer(init, step)
