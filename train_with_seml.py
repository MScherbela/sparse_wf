import os

os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
os.environ["JAX_DEFAULT_DTYPE_BITS"] = "32"

# ruff: noqa: E402
from sparse_wf.api import LoggingArgs, ModelArgs, MoleculeArgs, OptimizationArgs, PretrainingArgs
import pyscf.gto
import wonderwords
from collections import Counter
from typing import Any, cast, Sequence
from seml import Experiment
from seml.utils import flatten, merge_dicts
from sparse_wf.train import main
from sparse_wf.system import get_molecule


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
    mol: pyscf.gto.Mole, db_collection: str | None, logging_args: LoggingArgs, config
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


ex = Experiment()
ex.add_config("config/default.yaml")


@ex.automain
def seml_main(
    molecule_args: MoleculeArgs,
    model: str,
    model_args: ModelArgs,
    optimization: OptimizationArgs,
    pretraining: PretrainingArgs,
    batch_size: int,
    mcmc_steps: int,
    init_width: float,
    seed: int,
    db_collection: str | None,
    logging_args: LoggingArgs,
):
    mol = get_molecule(molecule_args)
    logging_args = update_logging_configuration(mol, db_collection, logging_args, locals())
    main(
        molecule_args=molecule_args,
        model=model,
        model_args=model_args,
        optimization=optimization,
        pretraining=pretraining,
        batch_size=batch_size,
        mcmc_steps=mcmc_steps,
        init_width=init_width,
        seed=seed,
        logging_args=logging_args,
    )
