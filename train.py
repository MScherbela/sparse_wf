import logging
import os

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import optax
import pyscf
import tqdm
from seml.experiment import Experiment
from sparse_wf.api import AuxData, Electrons, LoggingArgs, ModelArgs, OptimizationArgs, PRNGKeyArray
from sparse_wf.jax_utils import assert_identical_copies
from sparse_wf.loggers import MultiLogger
from sparse_wf.mcmc import make_mcmc, make_width_scheduler
from sparse_wf.model.dense_ferminet import DenseFermiNet  # noqa: F401
from sparse_wf.model.moon import SparseMoonWavefunction  # noqa: F401
from sparse_wf.preconditioner import make_preconditioner
from sparse_wf.pretraining import make_pretrainer
from sparse_wf.systems.scf import make_hf_orbitals
from sparse_wf.optim import make_optimizer
from sparse_wf.update import make_trainer

jax.config.update("jax_default_matmul_precision", "float32")
jax.config.update("jax_enable_x64", False)
ex = Experiment()


ex.add_config("config/default.yaml")


def init_electrons(key: PRNGKeyArray, mol: pyscf.gto.Mole, batch_size: int) -> Electrons:
    # TODO: center around nuclei, choose reasonable initial spin assignment
    batch_size = batch_size // jax.process_count()
    batch_size = batch_size - (batch_size % jax.local_device_count())
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
    # initialize distributed training
    if int(os.environ.get("SLURM_NTASKS", 1)) > 1:
        jax.distributed.initialize()
    logging.info(f"Using {jax.device_count()} devices across {jax.process_count()} processes.")

    mol = pyscf.gto.M(atom=molecule, basis=basis, spin=spin, unit="bohr")
    mol.build()

    # wf = SparseMoonWavefunction.create(mol, **model_args)
    wf = DenseFermiNet.create(mol)

    # Setup random keys
    # the main key will always be identitcal on all processes
    main_key = jax.random.PRNGKey(seed)
    # the proc_key will be unique per process.
    main_key, subkey = jax.random.split(main_key)
    proc_key = jax.random.split(subkey, jax.process_count())[jax.process_index()]
    # device_keys will be unique per device.
    proc_key, subkey = jax.random.split(proc_key)
    device_keys = jax.random.split(subkey, jax.local_device_count())

    # We want the parameters to be identical so we use the main_key here
    main_key, subkey = jax.random.split(main_key)
    params = wf.init(subkey)
    logging.info(f"Number of parameters: {sum(jnp.size(p) for p in jtu.tree_leaves(params))}")

    # We want to initialize differently per process so we use the proc_key here
    proc_key, subkey = jax.random.split(proc_key)
    electrons = init_electrons(subkey, mol, batch_size)

    trainer = make_trainer(
        wf,
        wf.local_energy,
        make_mcmc(wf, mcmc_steps),
        make_width_scheduler(),
        make_optimizer(**optimization["optimizer_args"]),
        make_preconditioner(wf, optimization["preconditioner_args"]),
        optimization["clipping"],
    )
    # The state will only be fed into pmapped functions, i.e., we need a per device key
    state = trainer.init(device_keys, params, electrons, jnp.array(init_width))
    assert_identical_copies(state.params)

    pretrainer = make_pretrainer(trainer, make_hf_orbitals(mol, basis), optax.adam(1e-3))
    state = pretrainer.init(state)

    logging.info("Pretraining")
    with tqdm.trange(pretrain_steps) as pbar:
        for _ in pbar:
            static = wf.input_constructor.get_static_input(state.electrons)
            state, aux_data = pretrainer.step(state, static)
            aux_data = to_log_data(aux_data)
            loggers.log(aux_data)
            if np.isnan(aux_data["loss"]):
                raise ValueError("NaN in pretraining loss")
            set_postfix(pbar, aux_data)

    state = state.to_train_state()
    assert_identical_copies(state.params)

    logging.info("Training")
    with tqdm.trange(optimization["steps"]) as pbar:
        for opt_step in pbar:
            static = wf.input_constructor.get_static_input(state.electrons)
            # assert static.n_neighbours == (3, 1, 4)  # TODO: remove
            state, _, aux_data = trainer.step(state, static)
            aux_data = to_log_data(aux_data)
            loggers.log(dict(opt_step=opt_step, **aux_data))
            if np.isnan(aux_data["opt/E"]):
                raise ValueError("NaN in energy")
            set_postfix(pbar, aux_data)
    assert_identical_copies(state.params)
