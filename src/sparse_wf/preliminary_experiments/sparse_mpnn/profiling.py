import os
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
from sparse_wf.model.wave_function import MoonLikeWaveFunction
from sparse_wf.system import get_molecule, get_atomic_numbers
from sparse_wf.mcmc import init_electrons
from sparse_wf.mcmc import make_mcmc
from sparse_wf.pseudopotentials import make_pseudopotential
from sparse_wf.api import StaticInputs
import jax
import jax.numpy as jnp
from pyscf.gto import Mole
import numpy as np
from jax import config as jax_config
import jax.tree_util as jtu
import time
import pathlib
import yaml

def perturb_electrons(rng, r, stepsize):
    batch_size, n_el, _ = r.shape
    rng_select, rng_move = jax.random.split(rng)
    idx = jax.random.randint(rng_select, (batch_size,), 0, n_el)
    delta = jax.random.normal(rng_move, (batch_size, 3)) * stepsize
    r_new = r.at[np.arange(batch_size), idx].add(delta)
    return r_new, idx[:, None]


DEFAULT_CONFIG_PATH = pathlib.Path(__file__).parent / "../../../../config/default.yaml"

dtype = jnp.float32
jax_config.update("jax_enable_x64", dtype is jnp.float64)
jax_config.update("jax_default_matmul_precision", "highest")
print(jax.devices())


default_config = yaml.load(open(DEFAULT_CONFIG_PATH, "r"), Loader=yaml.CLoader)
model_args = default_config["model_args"]
model_args["embedding"]["new"]["cutoff"] = 7.0
molecule_args = default_config["molecule_args"]
molecule_args["database_args"]["comment"] = "corannulene_dimer"
# molecule_args["database_args"]["comment"] = "cumulene_C4H4_0deg_singlet"
mcmc_args = default_config["mcmc_args"]
mcmc_args["single_electron_args"]["sweeps"] = 0.1
mcmc_args["jump_steps"] = 0

batch_size = 512
stepsize = 0.5
rng_r, rng_model, rng_mcmc = jax.random.split(jax.random.PRNGKey(0), 3)

mol = get_molecule(molecule_args)
R = np.array(mol.atom_coords())
Z = get_atomic_numbers(mol)
effective_charges = mol.atom_charges()
n_up, n_dn = mol.nelec
n_el = n_up + n_dn
electrons = init_electrons(rng_r, mol, batch_size, stddev=1.0)
electrons_new, idx_changed = perturb_electrons(rng_r, electrons, stepsize)

wf = MoonLikeWaveFunction.create(mol, **model_args)
print("Initializing params")
params = wf.init(rng_model, electrons[0])
params = jtu.tree_map(lambda x: jnp.array(x, dtype), params)
static_args = wf.get_static_input(electrons, electrons_new, idx_changed)
static_args = jtu.tree_map(lambda x: int(jnp.max(x)), static_args)
static_args = static_args.round_with_padding(1.0, n_el, n_el//2, len(R))
static_args = StaticInputs(mcmc=static_args, mcmc_jump=static_args, pp=static_args)
print(static_args)

# get_E_loc = jax.jit(jax.vmap(wf.local_energy, in_axes=(None, 0, None)), static_argnums=2)
# get_logpsi = jax.jit(jax.vmap(wf, in_axes=(None, 0, None)), static_argnums=2)

mcmc_step, mcmc_state = make_mcmc(wf, R, Z, n_el, mcmc_args)
mcmc_step = jax.jit(mcmc_step, static_argnums=3)


def func_to_profile(params, electrons, static_args):
    # result = get_E_loc(params, electrons, static_args)
    result = mcmc_step(rng_mcmc, params, electrons, static_args, mcmc_state)[0]
    # get_E_kin = jax.jit(jax.vmap(wf.kinetic_energy, in_axes=(None, 0, None)), static_argnums=2)

    # result = get_E_kin(params, electrons, static_args.mcmc)
    return result.block_until_ready()


print("Running warmup passes for compilation")
for i in range(1):
    print(f"Warmup pass {i}")
    result = func_to_profile(params, electrons, static_args)

print("Running actual passes for profiling")
with jax.profiler.trace("/tmp/tensorboard"):
# with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True, create_perfetto_trace=True):
    timings = []
    for i in range(1):
        print(f"Pass {i}")
        t0 = time.perf_counter()
        result = func_to_profile(params, electrons, static_args)
        t1 = time.perf_counter()
        timings.append(t1 - t0)
    print(f"Duration: T={np.mean(timings):.3f} +- {np.std(timings) / np.sqrt(len(timings)):.3f} s")
