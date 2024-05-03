import logging
import os

os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

# ruff: noqa: E402
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import tqdm
from sparse_wf.api import AuxData, LoggingArgs, ModelArgs, MoleculeArgs, OptimizationArgs, PretrainingArgs
from sparse_wf.jax_utils import assert_identical_copies, copy_from_main, replicate, pmap
from sparse_wf.loggers import MultiLogger
from sparse_wf.mcmc import init_electrons, make_mcmc, make_width_scheduler

from sparse_wf.model.dense_ferminet import DenseFermiNet  # noqa: F401
from sparse_wf.model.moon import SparseMoonWavefunction  # noqa: F401
from sparse_wf.optim import make_optimizer
from sparse_wf.preconditioner import make_preconditioner
from sparse_wf.pretraining import make_pretrainer
from sparse_wf.scf import make_hf_orbitals
from sparse_wf.system import get_molecule
from sparse_wf.update import make_trainer


jax.config.update("jax_default_matmul_precision", "float32")
jax.config.update("jax_enable_x64", False)


def to_log_data(aux_data: AuxData) -> dict[str, float]:
    return jtu.tree_map(lambda x: np.asarray(x).mean().item(), aux_data)


def set_postfix(pbar: tqdm.tqdm, aux_data: dict[str, float]):
    pbar.set_postfix(jtu.tree_map(lambda x: f"{x:.4f}", aux_data))


@pmap(static_broadcasted_argnums=(0, 3))
def get_gradients(logpsi_func, params, electrons, static):
    def get_grad(r):
        g = jax.grad(logpsi_func)(params, r, static)
        g = jtu.tree_flatten(g)[0]
        g = jnp.concatenate([x.flatten() for x in g])
        return g

    return jax.vmap(get_grad)(electrons)


def main(
    molecule_args: MoleculeArgs,
    model: str,
    model_args: ModelArgs,
    optimization: OptimizationArgs,
    pretraining: PretrainingArgs,
    batch_size: int,
    mcmc_steps: int,
    init_width: float,
    seed: int,
    logging_args: LoggingArgs,
    experiment_name: str = "",
):
    config = locals()

    mol = get_molecule(**molecule_args)

    loggers = MultiLogger(logging_args)
    loggers.log_config(config)
    # initialize distributed training
    if int(os.environ.get("SLURM_NTASKS", 1)) > 1:
        jax.distributed.initialize()
    logging.info(f'Run name: {loggers.args["name"]}')
    logging.info(f"Using {jax.device_count()} devices across {jax.process_count()} processes.")

    if model == "moon":
        wf = SparseMoonWavefunction.create(mol, **model_args)
    elif model == "ferminet":
        wf = DenseFermiNet.create(mol)
    else:
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

    # We want the parameters to be identical so we use the main_key here
    main_key, subkey = jax.random.split(main_key)
    params = wf.init(subkey)
    logging.info(f"Number of parameters: {sum(jnp.size(p) for p in jtu.tree_leaves(params))}")

    # We want to initialize differently per process so we use the proc_key here
    proc_key, subkey = jax.random.split(proc_key)
    electrons = init_electrons(subkey, mol, batch_size)
    mcmc_step = make_mcmc(wf, mcmc_steps)
    mcmc_width_scheduler = make_width_scheduler()

    trainer = make_trainer(
        wf,
        mcmc_step,
        mcmc_width_scheduler,
        make_optimizer(**optimization["optimizer_args"]),
        make_preconditioner(wf, optimization["preconditioner_args"]),
        optimization["clipping"],
    )
    # The state will only be fed into pmapped functions, i.e., we need a per device key
    state = trainer.init(device_keys, params, electrons, jnp.array(init_width))
    assert_identical_copies(state.params)

    pretrainer = make_pretrainer(trainer, make_hf_orbitals(mol), make_optimizer(**pretraining["optimizer_args"]))
    state = pretrainer.init(state)

    logging.info("Pretraining")
    with tqdm.trange(pretraining["steps"]) as pbar:
        for _ in pbar:
            static = wf.get_static_input(state.electrons)
            state, aux_data = pretrainer.step(state, static)
            aux_data = to_log_data(aux_data)
            loggers.log(aux_data)
            if np.isnan(aux_data["loss"]):
                raise ValueError("NaN in pretraining loss")
            set_postfix(pbar, aux_data)

    state = state.to_train_state()
    assert_identical_copies(state.params)

    logging.info("MCMC Burn-in")
    for _ in tqdm.trange(optimization["burn_in"]):
        static = wf.get_static_input(state.electrons)
        state, aux_data = trainer.sampling_step(state, static)
        loggers.log(dict(**aux_data))

    logging.info("Training")
    with tqdm.trange(optimization["steps"]) as pbar:
        for opt_step in pbar:
            static = wf.get_static_input(state.electrons)
            state, _, aux_data = trainer.step(state, static)
            # TODO: remove
            # if opt_step in [0, 1, 2] + [100, 101, 102]:
            #     gradients = get_gradients(wf.__call__, state.params, state.electrons, static)
            #     singular_values = jnp.linalg.svd(gradients, compute_uv=False, full_matrices=False)
            #     np.save(f"singular_values_{opt_step}.npy", singular_values)
            aux_data = to_log_data(aux_data)
            loggers.log(dict(opt_step=opt_step, **aux_data))
            if np.isnan(aux_data["opt/E"]):
                raise ValueError("NaN in energy")
            set_postfix(pbar, aux_data)
    assert_identical_copies(state.params)
    loggers.store_blob(state.serialize(), "chkpt_final.msgpk")
