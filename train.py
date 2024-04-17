import logging
import chex # noqa: F401
# chex.fake_pmap_and_jit().start()
import pathlib
import jax
# jax.config.update("jax_disable_jit", True)
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import optax
import pyscf
import tqdm
from seml.experiment import Experiment
from sparse_wf.api import AuxData, Electrons, ModelArgs, PRNGKeyArray, LoggingArgs, OptimizationArgs
from sparse_wf.mcmc import make_mcmc, make_width_scheduler
from sparse_wf.loggers import MultiLogger

from sparse_wf.model.moon import SparseMoonWavefunction  # noqa: F401
from sparse_wf.model.dense_ferminet import DenseFermiNet  # noqa: F401
from sparse_wf.preconditioner import make_preconditioner
from sparse_wf.pretraining import make_pretrainer
from sparse_wf.systems.scf import make_hf_orbitals
from sparse_wf.update import make_trainer
from sparse_wf.jax_utils import broadcast, p_split


jax.config.update("jax_default_matmul_precision", "float32")
jax.config.update("jax_enable_x64", False)
ex = Experiment()


ex.add_config(str(pathlib.Path(__file__).parent / "config/default.yaml"))


def init_electrons(key: PRNGKeyArray, mol: pyscf.gto.Mole, batch_size: int) -> Electrons:
    # TODO: center around nuclei, choose reasonable initial spin assignment
    batch_size = batch_size - (batch_size % jax.device_count())
    electrons = jax.random.normal(key, (batch_size, mol.nelectron, 3))
    return electrons


def to_log_data(aux_data: AuxData) -> dict[str, float]:
    return jtu.tree_map(lambda x: np.asarray(x).mean().item(), aux_data)


def set_postfix(pbar: tqdm.tqdm, aux_data: dict[str, float]):
    pbar.set_postfix(jtu.tree_map(lambda x: f"{x:.4f}", aux_data))


@ex.automain
def main(
    molecule: str,
    spin: int,
    model_args: ModelArgs,
    optimization: OptimizationArgs,
    batch_size: int,
    pretrain_steps: int,
    mcmc_steps: int,
    init_width: float,
    basis: str,
    seed: int,
    logging_args: LoggingArgs,
):
    config = locals()
    # TODO : add entity and make project configurable
    loggers = MultiLogger(logging_args)
    loggers.log_config(config)
    key = jax.random.PRNGKey(seed)

    mol = pyscf.gto.M(atom=molecule, basis=basis, spin=spin, unit="bohr")
    mol.build()

    if model_args["model_name"] == "moon":
        logging.info("Use Moon model.")
        wf = SparseMoonWavefunction.create(mol, **model_args)
    elif model_args["model_name"] == "ferminet":
        logging.info("Use FermiNet model.")
        wf = DenseFermiNet.create(mol)
    else:
        raise ValueError(f"Model {model_args['model_name']} doesn't exist!")

    key, subkey = jax.random.split(key)
    params = wf.init(subkey)
    logging.info(f"Number of parameters: {sum(jnp.size(p) for p in jax.tree_leaves(params))}")
    key, subkey = jax.random.split(key)
    electrons = init_electrons(subkey, mol, batch_size)

    trainer = make_trainer(
        wf,
        wf.local_energy,
        make_mcmc(wf, mcmc_steps),
        make_width_scheduler(),
        optax.chain(
            optax.clip_by_global_norm(optimization["grad_norm_constraint"]), optax.scale(-optimization["learning_rate"])
        ),
        make_preconditioner(wf, optimization["preconditioner_args"]),
        optimization["clipping"],
    )
    state = trainer.init(key, params, electrons, jnp.array(init_width))

    pretrainer = make_pretrainer(trainer, make_hf_orbitals(mol, basis), optax.adam(1e-3))
    state = pretrainer.init(state)

    key, *subkeys = jax.random.split(key, jax.device_count() + 1)
    shared_key = broadcast(jnp.stack(subkeys))

    logging.info("Pretraining")
    with tqdm.trange(pretrain_steps) as pbar:
        for _ in pbar:
            shared_key, subkey = p_split(shared_key)
            static = wf.input_constructor.get_static_input(state.electrons)
            state, aux_data = pretrainer.step(subkey, state, static)
            aux_data = to_log_data(aux_data)
            loggers.log(aux_data)
            if np.isnan(aux_data["loss"]):
                raise ValueError("NaN in pretraining loss")
            set_postfix(pbar, aux_data)

    state = state.to_train_state()

    logging.info("Training")
    with tqdm.trange(optimization["steps"]) as pbar:
        for opt_step in pbar:
            shared_key, subkey = p_split(shared_key)
            static = wf.input_constructor.get_static_input(state.electrons)
            # assert static.n_neighbours == (3, 1, 4)  # TODO: remove
            state, _, aux_data = trainer.step(subkey, state, static)
            aux_data = to_log_data(aux_data)
            loggers.log(dict(opt_step=opt_step, **aux_data))
            if np.isnan(aux_data["opt/E"]):
                raise ValueError("NaN in energy")
            set_postfix(pbar, aux_data)
