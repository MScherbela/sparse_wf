import logging
import os
from collections import Counter
from typing import Any, Sequence, cast

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import pyscf
import tqdm
import wonderwords
from seml.experiment import Experiment
from seml.utils import flatten, merge_dicts
from sparse_wf.api import AuxData, LoggingArgs, ModelArgs, OptimizationArgs, PretrainingArgs
from sparse_wf.jax_utils import assert_identical_copies, copy_from_main, replicate
from sparse_wf.loggers import MultiLogger
from sparse_wf.mcmc import make_mcmc, make_width_scheduler, init_electrons
from sparse_wf.model.dense_ferminet import DenseFermiNet  # noqa: F401
from sparse_wf.model.moon import SparseMoonWavefunction  # noqa: F401
from sparse_wf.preconditioner import make_preconditioner
from sparse_wf.pretraining import make_pretrainer
from sparse_wf.systems.scf import make_hf_orbitals
from sparse_wf.optim import make_optimizer
from sparse_wf.update import make_trainer

jax.config.update("jax_default_matmul_precision", "highest")
jax.config.update("jax_enable_x64", False)
ex = Experiment()


ex.add_config("config/default.yaml")


def to_log_data(aux_data: AuxData) -> dict[str, float]:
    return jtu.tree_map(lambda x: np.asarray(x).mean().item(), aux_data)


def set_postfix(pbar: tqdm.tqdm, aux_data: dict[str, float]):
    pbar.set_postfix(jtu.tree_map(lambda x: f"{x:.4f}", aux_data))


def get_run_name(mol: pyscf.gto.Mole, name_keys: Sequence[str] | None, config):
    atoms = Counter([mol.atom_symbol(i) for i in range(mol.natm)])
    mol_name = "".join([f"{k}{v}" for k, v in atoms.items()])

    if name_keys:
        flat_config = flatten(config)
        config_name = "-".join([str(flat_config[k]) for k in name_keys])
        key_string = f"-{config_name}"
    else:
        key_string = ""

    array_id = os.environ.get("SLURM_ARRAY_JOB_ID", None)
    if array_id is not None:
        # We are running in slurm - here we get unique IDs via SLURM and seml
        exp_id = ex.current_run._id
        return f"{mol_name}{key_string}-{exp_id}-{array_id}"

    # If we are not running slurm let's just draw a random adjective and word
    adjective = wonderwords.RandomWord().word(include_parts_of_speech=["adjectives"], word_max_length=8)
    noun = wonderwords.RandomWord().word(include_parts_of_speech=["noun"], word_max_length=8)

    result = f"{mol_name}{key_string}-{adjective}-{noun}"
    return result


def update_logging_configuration(
    mol: pyscf.gto.Mole, db_collection: str, logging_args: LoggingArgs, config
) -> LoggingArgs:
    folder_name = db_collection if db_collection else os.environ.get("USER", "default")
    updates: dict[str, Any] = {}
    if logging_args.get("collection", None) is None:
        updates["collection"] = folder_name
    if logging_args["wandb"].get("project", None) is None:
        updates["wandb"] = dict(project=folder_name)
    if logging_args.get("name", None) is None:
        updates["name"] = get_run_name(mol, logging_args["name_keys"], config)
    if logging_args.get("comment", None) is None:
        updates["comment"] = None
    return cast(LoggingArgs, merge_dicts(logging_args, updates))


@ex.automain
def main(
    molecule: str,
    spin: int,
    model_args: ModelArgs,
    optimization: OptimizationArgs,
    pretraining: PretrainingArgs,
    batch_size: int,
    mcmc_steps: int,
    init_width: float,
    basis: str,
    seed: int,
    db_collection: str,
    logging_args: LoggingArgs,
):
    config = locals()

    mol = pyscf.gto.M(atom=molecule, basis=basis, spin=spin, unit="bohr")
    mol.build()

    loggers = MultiLogger(update_logging_configuration(mol, db_collection, logging_args, config))
    loggers.log_config(config)
    # initialize distributed training
    if int(os.environ.get("SLURM_NTASKS", 1)) > 1:
        jax.distributed.initialize()
    logging.info(f'Run name: {loggers.args["name"]}')
    logging.info(f"Using {jax.device_count()} devices across {jax.process_count()} processes.")

    mol = pyscf.gto.M(atom=molecule, basis=basis, spin=spin, unit="bohr")
    mol.build()

    if model_args["model_name"] == "moon":
        logging.info("Using Moon wavefunction")
        wf = SparseMoonWavefunction.create(mol, **model_args)
    else:
        logging.info("Using FermiNet wavefunction")
        wf = DenseFermiNet.create(mol)

    # Setup random keys
    # the main key will always be identitcal on all processes
    main_key = jax.random.PRNGKey(seed)
    main_key = copy_from_main(replicate(main_key))[0]  # make sure that the main key is the same on all processes
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
        make_mcmc(wf, mcmc_steps),
        make_width_scheduler(),
        make_optimizer(**optimization["optimizer_args"]),
        make_preconditioner(wf, optimization["preconditioner_args"]),
        optimization["clipping"],
    )
    # The state will only be fed into pmapped functions, i.e., we need a per device key
    state = trainer.init(device_keys, params, electrons, jnp.array(init_width))
    assert_identical_copies(state.params)

    pretrainer = make_pretrainer(trainer, make_hf_orbitals(mol, basis), make_optimizer(**pretraining["optimizer_args"]))
    state = pretrainer.init(state)

    logging.info("Pretraining")
    with tqdm.trange(pretraining["steps"]) as pbar:
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
            state, _, aux_data = trainer.step(state, static)
            aux_data = to_log_data(aux_data)
            loggers.log(dict(opt_step=opt_step, **aux_data))
            if np.isnan(aux_data["opt/E"]):
                raise ValueError("NaN in energy")
            set_postfix(pbar, aux_data)
    assert_identical_copies(state.params)
    loggers.store_blob(state.serialize(), "chkpt_final.msgpk")
