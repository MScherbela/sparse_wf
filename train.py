import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
import pyscf
import tqdm
import wandb
from seml.experiment import Experiment
from sparse_wf.api import ClippingArgs, Electrons, ModelArgs, PreconditionerArgs, PRNGKeyArray
from sparse_wf.mcmc import make_mcmc, make_width_scheduler
from sparse_wf.model.moon import SparseMoonWavefunction
from sparse_wf.preconditioner import make_preconditioner
from sparse_wf.pretraining import make_pretrainer
from sparse_wf.systems.scf import make_hf_orbitals
from sparse_wf.update import make_trainer
from sparse_wf.jax_utils import broadcast, p_split

ex = Experiment()


ex.add_config("config/default.yaml")


def init_electrons(key: PRNGKeyArray, mol: pyscf.gto.Mole, batch_size: int) -> Electrons:
    batch_size = batch_size - (batch_size % jax.device_count())
    electrons = jax.random.normal(key, (batch_size, mol.nelectron, 3))
    return electrons


@ex.automain
def main(
    molecule: str,
    spin: int,
    model_args: ModelArgs,
    preconditioner_args: PreconditionerArgs,
    clipping_args: ClippingArgs,
    batch_size: int,
    steps: int,
    pretrain_steps: int,
    mcmc_steps: int,
    init_width: float,
    basis: str,
    seed: int,
):
    config = locals()
    wandb.init(
        project="sparse_wf",
        config=config,
    )

    mol = pyscf.gto.M(atom=molecule, basis=basis, spin=spin)
    key = jax.random.PRNGKey(seed)

    wf = SparseMoonWavefunction.create(mol.atom_coords(), mol.atom_charges(), mol.nelectron, mol.nelec[0], **model_args)
    key, subkey = jax.random.split(key)
    params = wf.init(subkey)
    key, subkey = jax.random.split(key)
    electrons = init_electrons(subkey, mol, batch_size)

    trainer = make_trainer(
        wf,
        wf.local_energy,
        make_mcmc(wf, mcmc_steps),
        make_width_scheduler(),
        optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.scale(-0.1),
        ),
        make_preconditioner(wf, preconditioner_args),
        clipping_args,
    )
    state = trainer.init(key, params, electrons, jnp.array(init_width))

    pretrainer = make_pretrainer(trainer, make_hf_orbitals(mol, basis), optax.adam(1e-3))
    key, subkey = jax.random.split(key)
    state = pretrainer.init(state)

    key, *subkeys = jax.random.split(key, jax.device_count() + 1)
    shared_key = broadcast(jnp.stack(subkeys))

    with tqdm.trange(pretrain_steps) as pbar:
        for _ in pbar:
            shared_key, subkey = p_split(shared_key)
            static = wf.input_constructor.get_static_input(state.electrons)
            state, aux_data = pretrainer.step(subkey, state, static)
            wandb.log(aux_data)
            pbar.set_postfix(jtu.tree_map(lambda x: x.mean(), aux_data))

    state = state.to_train_state()

    with tqdm.trange(steps) as pbar:
        for _ in pbar:
            shared_key, subkey = p_split(shared_key)
            static = wf.input_constructor.get_static_input(state.electrons)
            state, _, aux_data = trainer.step(subkey, state, static)
            wandb.log(aux_data)
            pbar.set_postfix(jtu.tree_map(lambda x: x.mean(), aux_data))
