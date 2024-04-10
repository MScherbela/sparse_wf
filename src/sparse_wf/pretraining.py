import jax
import optax

from sparse_wf.api import (
    HFOrbitalFn,
    AuxData,
    Parameters,
    Pretrainer,
    PretrainState,
    PRNGKeyArray,
    StaticInput,
    Trainer,
    TrainingState,
)
from sparse_wf.jax_utils import pmap


def make_pretrainer(trainer: Trainer, source_model: HFOrbitalFn, optimizer: optax.GradientTransformation) -> Pretrainer:
    batch_orbitals = jax.vmap(trainer.wave_function.orbitals, in_axes=(None, 0, 0))
    batch_src_orbitals = jax.vmap(source_model, in_axes=(0,))

    def init(training_state: TrainingState):
        return PretrainState(
            training_state.params,
            training_state.electrons,
            training_state.opt_state,
            training_state.width_state,
            pre_opt_state=pmap(optimizer.init)(training_state.params),
        )

    @pmap(static_broadcasted_argnums=2)
    def step(key: PRNGKeyArray, state: PretrainState, static: StaticInput) -> tuple[PretrainState, AuxData]:
        targets = trainer.wave_function.hf_transformation(batch_src_orbitals(state.electrons))

        def loss(params: Parameters):
            predicted_orbitals = batch_orbitals(params, state.electrons, static)
            return sum(((o - p_o) ** 2).mean() for o, p_o in zip(targets, predicted_orbitals))

        # Update
        loss_val, grad = jax.value_and_grad(loss)(state.params)
        updates, opt_state = optimizer.update(grad, state.pre_opt_state, state.params)
        params = optax.apply_updates(state.params, updates)

        # MCMC
        electrons, pmove = trainer.mcmc(key, params, state.electrons, static, state.width_state.width)
        width_state = trainer.width_scheduler.update(state.width_state, pmove)

        return state.replace(
            params=params,
            electrons=electrons,
            pre_opt_state=opt_state,
            width_state=width_state,
        ), {
            "loss": loss_val,
            "pmove": pmove,
        }

    return Pretrainer(init, step)
