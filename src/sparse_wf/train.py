import logging
import os
import time
from typing import Any, Optional

os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
# ruff: noqa: E402
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from sparse_wf.api import (
    LoggingArgs,
    ModelArgs,
    MoleculeArgs,
    OptimizationArgs,
    PretrainingArgs,
    EvaluationArgs,
    MCMCArgs,
)
from sparse_wf.static_args import StaticScheduler
from sparse_wf.jax_utils import assert_identical_copies, copy_from_main, replicate, pmap, pmax, get_from_main_process
from sparse_wf.loggers import MultiLogger, to_log_data, mcmc_to_log_data
from sparse_wf.mcmc import init_electrons, make_mcmc, make_width_scheduler
from sparse_wf.model.dense_ferminet import DenseFermiNet  # noqa: F401

# from sparse_wf.model.moon_old import SparseMoonWavefunction  # noqa: F401
from sparse_wf.model.wave_function import MoonLikeWaveFunction
from sparse_wf.optim import make_optimizer
from sparse_wf.preconditioner import make_preconditioner
from sparse_wf.pretraining import make_pretrainer
from sparse_wf.scf import HFWavefunction, CASWavefunction
from sparse_wf.spin_operator import make_spin_operator
from sparse_wf.system import get_molecule
from sparse_wf.update import make_trainer
import functools


jax.config.update("jax_default_matmul_precision", "float32")
jax.config.update("jax_enable_x64", True)


@pmap(static_broadcasted_argnums=(0, 3))
def get_gradients(logpsi_func, params, electrons, static):
    def get_grad(r):
        g = jax.grad(logpsi_func)(params, r, static)
        g = jtu.tree_flatten(g)[0]
        g = jnp.concatenate([x.flatten() for x in g])
        return g

    return jax.vmap(get_grad)(electrons)


def isnan(args):
    leaves = jtu.tree_leaves(args)
    return any(np.isnan(leave).any() for leave in leaves)


def main(
    molecule_args: MoleculeArgs,
    model: str,
    model_args: ModelArgs,
    optimization: OptimizationArgs,
    pretraining: PretrainingArgs,
    evaluation: EvaluationArgs,
    batch_size: int,
    mcmc_args: MCMCArgs,
    seed: int,
    logging_args: LoggingArgs,
    load_checkpoint: str,
    metadata: Optional[dict[str, Any]] = None,
):
    config = locals()

    mol = get_molecule(molecule_args)
    R = np.array(mol.atom_coords())
    Z = np.array(mol.atom_charges())
    n_up, n_dn = mol.nelec
    n_el = n_up + n_dn

    loggers = MultiLogger(logging_args)
    loggers.log_config(config | dict(molecule=dict(R=R.tolist(), Z=Z.tolist(), n_el=n_el, n_up=n_up)))
    # initialize distributed training
    if int(os.environ.get("SLURM_NTASKS", 1)) > 1:
        jax.distributed.initialize()
    logging.info(f'Run name: {loggers.args["name"]}')
    logging.info(f"Using {jax.device_count()} devices across {jax.process_count()} processes.")

    match model.lower().strip():
        case "moon":
            wf = MoonLikeWaveFunction.create(mol, **model_args)
        case "ferminet":
            wf = DenseFermiNet.create(mol)
        case _:
            raise ValueError(f"Invalid model: {model}")

    # Setup random keys
    # the main key will always be identitcal on all processes
    main_key = jax.random.PRNGKey(seed)
    main_key = copy_from_main(replicate(main_key))[0]
    # the proc_key will be unique per process.
    main_key, subkey = jax.random.split(main_key)
    proc_key = jax.random.split(subkey, jax.process_count())[jax.process_index()]
    # device_keys will be unique per device.
    proc_key, subkey = jax.random.split(proc_key)
    device_keys = jax.random.split(subkey, jax.local_device_count())

    # We want to initialize differently per process so we use the proc_key here
    proc_key, subkey = jax.random.split(proc_key)
    electrons = init_electrons(subkey, mol, batch_size)
    mcmc_step, mcmc_state = make_mcmc(wf, R, n_el, mcmc_args)
    mcmc_width_scheduler = make_width_scheduler()
    static_scheduler = StaticScheduler(n_el, n_up, len(R))

    # We want the parameters to be identical so we use the main_key here
    main_key, subkey = jax.random.split(main_key)
    params = wf.init(subkey, electrons[0])
    # params can still be different per process due to different sample, leading to different normalizations
    # Use the params from the main process across all devices
    params = get_from_main_process(params)
    n_params = sum(jnp.size(p) for p in jtu.tree_leaves(params))
    loggers.log_config(dict(n_params=n_params))

    trainer = make_trainer(
        wf,
        mcmc_step,
        mcmc_width_scheduler,
        make_optimizer(**optimization["optimizer_args"]),
        make_preconditioner(wf, optimization["preconditioner_args"]),
        optimization["clipping"],
        optimization["max_batch_size"],
        make_spin_operator(wf, optimization["spin_operator_args"]),
        optimization["energy_operator"],
    )
    # The state will only be fed into pmapped functions, i.e., we need a per device key
    state = trainer.init(device_keys, params, electrons, mcmc_state)
    if load_checkpoint:
        with open(load_checkpoint, "rb") as f:
            state = state.deserialize(f.read(), batch_size)

    assert_identical_copies(state.params)

    # Build pre-training wavefunction and sampling step
    match pretraining["reference"].lower():
        case "hf":
            hf_wf = HFWavefunction(mol)
        case "cas":
            hf_wf = CASWavefunction(mol, model_args["n_determinants"], **pretraining["cas"])
        case _:
            raise ValueError(f"Invalid pretraining reference: {pretraining['reference']}")

    match pretraining["sample_from"].lower():
        case "hf":
            pretrain_mcmc_step = make_mcmc(hf_wf, R, n_el, mcmc_args, wf.get_static_input)[0]  # type: ignore
        case "wf":
            pretrain_mcmc_step = mcmc_step
        case _:
            raise ValueError(f"Invalid pretraining sample_from: {pretraining['sample_from']}")

    pretrainer = make_pretrainer(
        wf,
        pretrain_mcmc_step,
        mcmc_width_scheduler,
        hf_wf.orbitals,
        make_optimizer(**pretraining["optimizer_args"]),
    )
    state = pretrainer.init(state)
    model_static = pmap(jax.vmap(lambda r: pmax(wf.get_static_input(r))))(state.electrons)
    static = static_scheduler(model_static)

    logging.info("Pretraining")
    for step in range(pretraining["steps"]):
        t0 = time.perf_counter()
        state, aux_data, mcmc_stats = pretrainer.step(state, static)
        static = static_scheduler(mcmc_stats.static_max)
        log_data = to_log_data(aux_data) | mcmc_to_log_data(mcmc_stats) | to_log_data(static, "static/padded/")
        t1 = time.perf_counter()
        log_data["pretrain/t_step"] = t1 - t0
        log_data["pretrain/step"] = step
        loggers.log(log_data)
        if np.isnan(log_data["pretrain/loss"]):
            raise ValueError("NaN in pretraining loss")

    state = state.to_train_state()
    assert_identical_copies(state.params)

    logging.info("MCMC Burn-in")
    for _ in range(optimization["burn_in"]):
        state, aux_data, mcmc_stats = trainer.sampling_step(state, static, False, None)
        static = static_scheduler(mcmc_stats.static_max)
        log_data = to_log_data(aux_data) | mcmc_to_log_data(mcmc_stats) | to_log_data(static, "static/padded/")
        loggers.log(log_data)

    logging.info("Training")
    for opt_step in range(optimization["steps"]):
        t0 = time.perf_counter()
        state, _, aux_data, mcmc_stats = trainer.step(state, static)
        static = static_scheduler(mcmc_stats.static_max)
        log_data = to_log_data(aux_data) | mcmc_to_log_data(mcmc_stats) | to_log_data(static, "static/padded/")
        t1 = time.perf_counter()
        log_data["opt/t_step"] = t1 - t0
        log_data["opt/step"] = opt_step
        loggers.log(log_data)
        loggers.store_checkpoint(opt_step, state, "opt")
        if isnan(log_data):
            raise ValueError("NaN")

    assert_identical_copies(state.params)
    loggers.store_blob(state.serialize(), "chkpt_final.msgpk")

    logging.info("Evaluation")
    overlap_fn = (
        functools.partial(hf_wf.excited_signed_logpsi, jnp.array(evaluation["overlap_states"]))
        if evaluation["overlap_states"]
        else None
    )
    for eval_step in range(evaluation["steps"]):
        t0 = time.perf_counter()
        state, aux_data, mcmc_stats = trainer.sampling_step(state, static, evaluation["compute_energy"], overlap_fn)
        static = static_scheduler(mcmc_stats.static_max)
        log_data = to_log_data(aux_data) | mcmc_to_log_data(mcmc_stats) | to_log_data(static, "static/padded/")
        t1 = time.perf_counter()
        log_data["eval/t_step"] = t1 - t0
        log_data["eval/step"] = eval_step
        loggers.log(log_data)
        loggers.store_checkpoint(eval_step, state, "eval")
