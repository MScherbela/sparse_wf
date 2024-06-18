import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
from sparse_wf.model.wave_function import MoonLikeWaveFunction
from sparse_wf.api import JastrowArgs, EmbeddingArgs
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

batch_size = 64
n_determinants = 16
rng_r, rng_model = jax.random.split(jax.random.PRNGKey(0))
electrons, mol = build_atom_chain(rng_r, n_nuc=180, n_el_per_nuc=1, batch_size=batch_size)
n_el = electrons.shape[-2]

model = MoonLikeWaveFunction.create(
    mol,
    EmbeddingArgs(
        cutoff=5.0, feature_dim=256, nuc_mlp_depth=4, pair_mlp_widths=(16, 8), pair_n_envelopes=16
    ),
    JastrowArgs(e_e_cusps="psiformer", use_log_jastrow=True, use_mlp_jastrow=True, mlp_depth=2, mlp_width=64),
    n_determinants,
    n_envelopes=8,
)
params = model.init(rng_model, electrons[0])
params = jtu.tree_map(lambda x: jnp.array(x, dtype), params)
static_args = model.get_static_input(electrons)
print(static_args)

# get_E_loc = jax.jit(jax.vmap(model.local_energy, in_axes=(None, 0, None)), static_argnums=2)
get_logpsi = jax.jit(jax.vmap(model, in_axes=(None, 0, None)), static_argnums=2)

print("Running warmup passes for compilation")
for i in range(5):
    print(f"Warmup pass {i}")
    Eloc = get_logpsi(params, electrons, static_args).block_until_ready()

print("Running actual passes for profiling")
with jax.profiler.trace("/tmp/tensorboard"):
    timings = []
    for i in range(5):
        print(f"Pass {i}")
        t0 = time.perf_counter()
        # Eloc = get_E_loc(params, electrons, static_args).block_until_ready()
        logpsi = get_logpsi(params, electrons, static_args).block_until_ready()
        t1 = time.perf_counter()
        timings.append(t1 - t0)
    print(f"Duration: T={np.mean(timings):.3f} +- {np.std(timings) / np.sqrt(len(timings)):.3f} s")