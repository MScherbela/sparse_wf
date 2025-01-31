import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"


import jax
import jax.numpy as jnp
import yaml
import numpy as np
from sparse_wf.system import get_molecule, get_atomic_numbers
from sparse_wf.model.wave_function import MoonLikeWaveFunction
from sparse_wf.model.new_sparse_model import EmbeddingParams
from sparse_wf.mcmc import make_mcmc, make_width_scheduler
import jax.tree_util as jtu
from sparse_wf.optim import make_optimizer
from sparse_wf.update import make_trainer
from sparse_wf.preconditioner import make_preconditioner
from sparse_wf.spin_operator import make_spin_operator
import pathlib
from typing import Any, NamedTuple

DEFAULT_CONFIG_PATH = pathlib.Path(__file__).parent / "../config/default.yaml"


class EmbeddingParamsWithoutCutoffParam(NamedTuple):
    dynamic_params_en: Any
    elec_init: Any
    edge_same: Any
    edge_diff: Any
    updates: Any
    scales: Any


def _save_checkpoint(state, fname):
    with open(fname, "wb") as f:
        f.write(state.serialize())


def convert_checkpoint(checkpoint_fname, config_fname):
    with open(config_fname, "r") as f:
        config = yaml.load(f, Loader=yaml.CLoader)

    with open(DEFAULT_CONFIG_PATH, "r") as f:
        default_config = yaml.load(f, Loader=yaml.CLoader)

    mol = get_molecule(config["molecule_args"])
    R = np.array(mol.atom_coords())
    Z = get_atomic_numbers(mol)
    n_up, n_dn = mol.nelec
    n_el = n_up + n_dn

    print("Initializing dummy model...")
    wf = MoonLikeWaveFunction.create(mol, **config["model_args"])
    electrons = jnp.zeros([config["batch_size"], mol.nelectron, 3], jnp.float32)
    mcmc_step, mcmc_state = make_mcmc(wf, R, Z, n_el, config["mcmc_args"])
    mcmc_width_scheduler = make_width_scheduler(target_pmove=config["mcmc_args"]["acceptance_target"])

    params = wf.init(jax.random.PRNGKey(0), electrons[0])
    n_params = sum(jnp.size(p) for p in jtu.tree_leaves(params))
    print("Number of parameters (incl. cutoff_param):", n_params)

    optimization = config["optimization"]
    cutoff_transition_steps = optimization.get(
        "cutoff_transition_steps", default_config["optimization"]["cutoff_transition_steps"]
    )
    if isinstance(optimization["pp_grid_points"], int):
        pp_grid_points = default_config["optimization"]["pp_grid_points"]

    print("Initializing dummy trainer...")
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
        pp_grid_points,
        cutoff_transition_steps,
    )

    print("Exchanging params...")
    # The state will only be fed into pmapped functions, i.e., we need a per device key
    dummy_state = trainer.init(jax.random.PRNGKey(0)[None], params, electrons, mcmc_state)

    # Remove cutoff_param which is not present in the checkpoint which is being loaded
    dummy_cutoff_param = dummy_state.params.embedding[-1]
    dummy_state = dummy_state.replace(
        params=params._replace(embedding=EmbeddingParamsWithoutCutoffParam(*dummy_state.params.embedding[:-1]))
    )
    with open(checkpoint_fname, "rb") as f:
        state = dummy_state.deserialize(f.read())

    # Re-add cutoff_param
    state = state.replace(
        params=state.params._replace(embedding=EmbeddingParams(*state.params.embedding, dummy_cutoff_param))
    )
    print("Saving converted checkpoint...")
    _save_checkpoint(state, checkpoint_fname.replace(".msgpk", "_converted.msgpk"))
    print("Converted!")


if __name__ == "__main__":
    # run_dir = "/storage/scherbelam20/runs/sparse_wf/benzene_dimer/cutoff5.0/5.0_benzene_dimer_T_4.95A_singleEl_jumps_leonardo_from087695"
    run_dir = "/storage/scherbelam20/runs/sparse_wf/benzene_dimer/cutoff5.0/3.0_benzene_dimer_T_4.95A_vsc_from_062500_from080341"
    # run_dir = "/storage/scherbelam20/runs/sparse_wf/benzene_dimer/cutoff5.0/3.0_benzene_dimer_T_10.00A_vsc_from062500"
    config_fname = f"{run_dir}/full_config.yaml"
    checkpoint_fname = f"{run_dir}/optchkpt100000.msgpk"
    convert_checkpoint(checkpoint_fname, config_fname)
