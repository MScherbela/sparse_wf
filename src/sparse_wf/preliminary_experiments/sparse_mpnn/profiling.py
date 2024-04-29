import os
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from sparse_wf.model import SparseMoonWavefunction
import jax
import jax.numpy as jnp
from pyscf.gto import Mole
import numpy as np
from jax import config as jax_config
import jax.tree_util as jtu
import time

dtype = jnp.float64
jax_config.update("jax_enable_x64", dtype is jnp.float64)
jax_config.update("jax_default_matmul_precision", "highest")
print(jax.devices())


def build_atom_chain(rng, n_nuc, n_el_per_nuc, batch_size):
    R = np.arange(n_nuc)[:, None] * np.array([1, 0, 0])
    r = R[:, None, :] + jax.random.normal(rng, [batch_size, n_nuc, n_el_per_nuc, 3], dtype)
    r = jax.lax.collapse(r, 1, 3)
    Z = np.ones(n_nuc, dtype=int) * n_el_per_nuc
    mol = Mole(atom=[(int(Z_), R_) for R_, Z_ in zip(R, Z)]).build()
    return r, mol

batch_size = 8
n_determinants = 4
rng_r, rng_model = jax.random.split(jax.random.PRNGKey(0))
electrons, mol = build_atom_chain(rng_r, n_nuc=512, n_el_per_nuc=1, batch_size=batch_size)
n_el = electrons.shape[-2]

model = SparseMoonWavefunction.create(
    mol,
    n_determinants=n_determinants,
    cutoff=5.0,
    feature_dim=256,
    nuc_mlp_depth=3,
    pair_mlp_widths=(16, 8),
    pair_n_envelopes=32,
)
params = model.init(rng_model)
params = jtu.tree_map(lambda x: jnp.array(x, dtype), params)
static_args = model.input_constructor.get_static_input(electrons)
print(static_args)

get_E_loc = jax.jit(model.local_energy, static_argnums=2)

print("Running warmup passes for compilation")
for i in range(5):
    print(f"Warmup pass {i}")
    Eloc = get_E_loc(params, electrons[0], static_args).block_until_ready()

print("Running actual passes for profiling")
with jax.profiler.trace("/tmp/tensorboard"):
    timings = []
    for i in range(5):
        print(f"Pass {i}")
        t0 = time.perf_counter()
        Eloc = get_E_loc(params, electrons[0], static_args).block_until_ready()
        t1 = time.perf_counter()
        timings.append(t1 - t0)
    print(f"Duration: T={np.mean(timings):.3f} +- {np.std(timings) / np.sqrt(len(timings)):.3f} s")