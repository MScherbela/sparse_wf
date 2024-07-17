import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
from sparse_wf.model.wave_function import MoonLikeWaveFunction
from sparse_wf.api import EmbeddingArgs, JastrowArgs, MoonEmbeddingArgs, EnvelopeArgs, NewEmbeddingArgs
from sparse_wf.mcmc import make_mcmc
import jax
import jax.numpy as jnp
from pyscf.gto import Mole
import numpy as np
from jax import config as jax_config
import jax.tree_util as jtu
import time

dtype = jnp.float32
jax_config.update("jax_enable_x64", dtype is jnp.float64)
jax_config.update("jax_default_matmul_precision", "highest")
print(jax.devices())


def build_atom_chain(rng, n_nuc, n_el_per_nuc, batch_size):
    R = np.arange(n_nuc)[:, None] * np.array([1.8, 0, 0])
    r = R[:, None, :] + jax.random.normal(rng, [batch_size, n_nuc, n_el_per_nuc, 3], dtype)
    r = jax.lax.collapse(r, 1, 3)
    Z = np.ones(n_nuc, dtype=int) * n_el_per_nuc
    mol = Mole(atom=[(int(Z_), R_) for R_, Z_ in zip(R, Z)]).build()
    return r, mol


batch_size = 8
n_determinants = 16
mcmc_intersteps = 50
rng_r, rng_model, rng_mcmc = jax.random.split(jax.random.PRNGKey(0), 3)
electrons, mol = build_atom_chain(rng_r, n_nuc=362, n_el_per_nuc=1, batch_size=batch_size)
n_el = electrons.shape[-2]

model = MoonLikeWaveFunction.create(
    mol,
    EmbeddingArgs(
        embedding="new",
        new=NewEmbeddingArgs(
            cutoff=2.0,
            cutoff_1el=20.0,
            feature_dim=128,
            nuc_mlp_depth=2,
            pair_mlp_widths=(16, 8),
            pair_n_envelopes=32,
            low_rank_buffer=2,
            n_updates=2,
        ),
        moon=MoonEmbeddingArgs(
            cutoff=2.0,
            cutoff_1el=20.0,
            feature_dim=128,
            nuc_mlp_depth=2,
            pair_mlp_widths=(16, 8),
            pair_n_envelopes=32,
            low_rank_buffer=2,
        ),
        ),
    JastrowArgs(e_e_cusps="psiformer", use_log_jastrow=True, use_mlp_jastrow=True, mlp_depth=2, mlp_width=64),
    EnvelopeArgs(envelope="isotropic", isotropic_args=dict(n_envelopes=8), glu_args=None),  # type: ignore
    n_determinants,
)
params = model.init(rng_model, electrons[0])
params = jtu.tree_map(lambda x: jnp.array(x, dtype), params)
static_args = model.get_static_input(electrons)
print(static_args)

get_E_loc = jax.jit(jax.vmap(model.local_energy, in_axes=(None, 0, None)), static_argnums=2)
get_logpsi = jax.jit(jax.vmap(model, in_axes=(None, 0, None)), static_argnums=2)
mcmc_step, stepsize = make_mcmc(model, "single-electron", 0.2, mcmc_intersteps)
mcmc_step = jax.jit(mcmc_step, static_argnums=3)


def func_to_profile(params, electrons, static_args):
    # result = get_E_loc(params, electrons, static_args)
    result = mcmc_step(rng_mcmc, params, electrons, static_args, stepsize)[0]
    return result.block_until_ready()


print("Running warmup passes for compilation")
for i in range(3):
    print(f"Warmup pass {i}")
    result = func_to_profile(params, electrons, static_args)

print("Running actual passes for profiling")
with jax.profiler.trace("/tmp/tensorboard"):
    timings = []
    for i in range(3):
        print(f"Pass {i}")
        t0 = time.perf_counter()
        result = func_to_profile(params, electrons, static_args)
        t1 = time.perf_counter()
        timings.append(t1 - t0)
    print(f"Duration: T={np.mean(timings):.3f} +- {np.std(timings) / np.sqrt(len(timings)):.3f} s")
