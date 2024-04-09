import jax
import pyscf
import wandb
from seml.experiment import Experiment

from sparse_wf.api import ModelArgs

ex = Experiment()


ex.add_config("config/default.yaml")


@ex.automain
def main(
    molecule: str,
    spin: int,
    model_args: ModelArgs,
    batch_size: int,
    steps: int,
    mcmc_steps: int,
    energy_clip_threshold: float,
    basis: str,
    seed: int,
):
    config = locals()
    wandb.init(
        project="sparse_wf",
        config=config,
    )

    mol = pyscf.gto.M(atom=molecule, basis=basis, spin=spin)
    mol.RHF().run()
    key = jax.random.PRNGKey(seed)

    ...
