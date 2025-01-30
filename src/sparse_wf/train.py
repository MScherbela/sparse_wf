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
from sparse_wf.static_args import StaticSchedulers
from sparse_wf.jax_utils import (
    assert_identical_copies,
    copy_from_main,
    replicate,
    pmap,
    pmax,
    get_from_main_process,
)
from sparse_wf.loggers import MultiLogger, to_log_data, save_expanded_checkpoint
from sparse_wf.mcmc import init_electrons, make_mcmc, make_width_scheduler
from sparse_wf.model.dense_ferminet import DenseFermiNet  # noqa: F401
from sparse_wf.model.wave_function import MoonLikeWaveFunction
from sparse_wf.model.utils import inverse_sigmoid
from sparse_wf.optim import make_optimizer
from sparse_wf.preconditioner import make_preconditioner
from sparse_wf.pretraining import make_pretrainer
from sparse_wf.scf import HFWavefunction, CASWavefunction
from sparse_wf.spin_operator import make_spin_operator
from sparse_wf.system import get_molecule, get_atomic_numbers
from sparse_wf.update import make_trainer
from sparse_wf.auto_requeue import should_abort, requeue_job
import functools


jax.config.update("jax_default_matmul_precision", "float32")
jax.config.update("jax_enable_x64", True)


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
    extract_checkpoint: bool,
    checkpoint_cutoff: Optional[float] = None,
    auto_requeue: int = 0,
    metadata: Optional[dict[str, Any]] = None,
):
    config = locals()

    mol = get_molecule(molecule_args)
    R = np.array(mol.atom_coords())
    Z = get_atomic_numbers(mol)
    effective_charges = mol.atom_charges()
    n_up, n_dn = mol.nelec
    n_el = n_up + n_dn

    loggers = MultiLogger(logging_args)
    loggers.log_config(config | dict(molecule=dict(R=R.tolist(), Z=Z.tolist(), n_el=n_el, n_up=n_up)))
    # initialize distributed training
    slurm_n_tasks = int(os.environ.get("SLURM_NTASKS", 1))
    slurm_gpus_per_task = int(os.environ.get("SLURM_GPUS_PER_TASK", 1))
    if slurm_n_tasks > 1:
        if slurm_gpus_per_task > 1:
            # SBATCH settings: -N {nr_of_nodes} --ntasks-per-node=1 --gpus-per-task={gpus_per_node}; do not use gres
            jax.distributed.initialize(num_processes=slurm_n_tasks, local_device_ids=range(slurm_gpus_per_task))
        else:
            # SBATCH settings: -N {nr_of_nodes} --ntasks-per-node={gpus_per_node} --gres=gpu:{gpus_per_node}; do not use --gpus-per-task
            jax.distributed.initialize()
    logging.info(f'Run name: {loggers.args["name"]}')
    logging.info(f"Using {jax.device_count()} devices across {jax.process_count()} processes.")
    logging.info(f"Atomic numbers: {Z}; Effective charges: {effective_charges}; Spin configuration: ({n_up}, {n_dn})")

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
    mcmc_step, mcmc_state = make_mcmc(wf, R, Z, n_el, mcmc_args)
    mcmc_width_scheduler = make_width_scheduler(target_pmove=mcmc_args["acceptance_target"])
    static_schedulers = StaticSchedulers(n_el, n_up, len(R))

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
        mol._ecp.keys(),
        optimization["pp_grid_points"],
        optimization["cutoff_transition_steps"],
    )

    # The state will only be fed into pmapped functions, i.e., we need a per device key
    state = trainer.init(device_keys, params, electrons, mcmc_state)
    if load_checkpoint:
        logging.info(f"Loading checkpoint {load_checkpoint}")
        with open(load_checkpoint, "rb") as f:
            state = state.deserialize(f.read(), batch_size)
        if extract_checkpoint:
            output_dir = load_checkpoint.replace(".msgpk", "")
            logging.info(f"Saveing expanded checkpoint to {output_dir}")
            save_expanded_checkpoint(state, output_dir)
        if checkpoint_cutoff is not None:
            old_cutoff_param = state.params.embedding.cutoff_param
            old_cutoff = jax.nn.sigmoid(old_cutoff_param) * checkpoint_cutoff
            new_cutoff_param = inverse_sigmoid(old_cutoff / model_args["embedding"]["new"]["cutoff"])
            state = state.replace(
                params=state.params._replace(embedding=state.params.embedding._replace(cutoff_param=new_cutoff_param))
            )

    assert_identical_copies(state.params)
    model_static = pmap(jax.vmap(lambda r: pmax(wf.get_static_input(r))))(state.electrons)
    statics = static_schedulers({"mcmc": model_static, "mcmc_jump": model_static, "pp": model_static})

    # Build pre-training wavefunction and sampling step
    if (pretraining["steps"] > 0) or evaluation["overlap_states"]:
        logging.info(f"Computing reference wavefunction for pretraining, using: {pretraining['reference']}")
        match pretraining["reference"].lower():
            case "hf":
                hf_wf = HFWavefunction(mol, pretraining["hf"])
            case "cas":
                hf_wf = CASWavefunction(mol, pretraining["hf"], model_args["n_determinants"], **pretraining["cas"])
            case _:
                raise ValueError(f"Invalid pretraining reference: {pretraining['reference']}")

    # Pretraining
    if pretraining["steps"] > 0:
        match pretraining["sample_from"].lower():
            case "hf":
                pretrain_mcmc_step = make_mcmc(hf_wf, R, Z, n_el, mcmc_args, wf.get_static_input)[0]  # type: ignore
            case "wf":
                pretrain_mcmc_step = mcmc_step
            case _:
                raise ValueError(f"Invalid pretraining sample_from: {pretraining['sample_from']}")

        if molecule_args["pseudopotentials"] and ("ccecp" not in molecule_args["basis"]):
            raise ValueError("Pretraining with pseudopotentials requires 'ccecp' basis")
        elif (not molecule_args["pseudopotentials"]) and ("ccecp" in molecule_args["basis"]):
            raise ValueError("Pretraining without pseudopotentials requires a basis without 'ccecp'")

        pretrainer = make_pretrainer(
            wf,
            pretrain_mcmc_step,
            mcmc_width_scheduler,
            hf_wf.orbitals,
            make_optimizer(**pretraining["optimizer_args"]),
        )
        state = pretrainer.init(state)

        logging.info("Pretraining")
        for step in range(pretraining["steps"]):
            t0 = time.perf_counter()
            state, aux_data, mcmc_stats = pretrainer.step(state, statics)
            statics = static_schedulers(mcmc_stats.static_max, pretrainer.step._cache_size)  # type: ignore
            log_data = to_log_data(mcmc_stats, statics, aux_data)
            t1 = time.perf_counter()
            log_data["pretrain/t_step"] = t1 - t0
            log_data["pretrain/step"] = step
            loggers.log(log_data)
            if np.isnan(log_data["pretrain/loss"]):
                raise ValueError("NaN in pretraining loss")

        state = state.to_train_state()
        assert_identical_copies(state.params)

    # Variational optimization
    logging.info("MCMC Burn-in")
    for _ in range(optimization["burn_in"]):
        state, aux_data, mcmc_stats = trainer.sampling_step(state, statics, False, None)
        statics = static_schedulers(mcmc_stats.static_max, trainer.sampling_step._cache_size)
        log_data = to_log_data(mcmc_stats, statics, aux_data)
        loggers.log(log_data)

    logging.info("Taking 1 step to get correct statics")
    _, _, _, mcmc_stats = trainer.step(state, statics)
    statics = static_schedulers(mcmc_stats.static_max, trainer.step._cache_size)

    logging.info("Training")
    n_steps_prev = int(state.step[0])
    for opt_step in range(n_steps_prev, optimization["steps"] + 1):
        loggers.store_checkpoint(opt_step, state, "opt", force=(opt_step == optimization["steps"]))
        if should_abort():
            chkpt_fname = loggers.store_checkpoint(opt_step, state, "opt", force=True)
            logging.info(f"Requeueing with checkpoint: {chkpt_fname}")
            if auto_requeue:
                requeue_job(opt_step, chkpt_fname)
            raise SystemExit(0)

        t0 = time.perf_counter()
        state, _, aux_data, mcmc_stats = trainer.step(state, statics)
        statics = static_schedulers(mcmc_stats.static_max, trainer.step._cache_size)
        log_data = to_log_data(mcmc_stats, statics, aux_data)
        t1 = time.perf_counter()
        log_data["opt/t_step"] = t1 - t0
        log_data["opt/step"] = opt_step
        loggers.log(log_data)
        if isnan(log_data):
            raise ValueError("NaN")

    assert_identical_copies(state.params)
    loggers.store_blob(state.serialize(), "chkpt_final.msgpk")

    # Evaluation / Inference
    logging.info("Evaluation")
    overlap_fn = (
        functools.partial(hf_wf.excited_signed_logpsi, jnp.array(evaluation["overlap_states"]))
        if evaluation["overlap_states"]
        else None
    )
    for eval_step in range(evaluation["steps"]):
        t0 = time.perf_counter()
        state, aux_data, mcmc_stats = trainer.sampling_step(state, statics, evaluation["compute_energy"], overlap_fn)
        statics = static_schedulers(mcmc_stats.static_max, trainer.sampling_step._cache_size)
        log_data = to_log_data(mcmc_stats, statics, aux_data)
        t1 = time.perf_counter()
        log_data["eval/t_step"] = t1 - t0
        log_data["eval/step"] = eval_step
        loggers.log(log_data)
        loggers.store_checkpoint(eval_step, state, "eval")
