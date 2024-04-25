# %%
import os
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
from sparse_wf.model import SparseMoonWavefunction
import jax
import jax.numpy as jnp
from pyscf.gto import Mole
import numpy as np
from jax import config as jax_config
import jax.tree_util as jtu
import functools

print(jax.devices())
dtype = jnp.float32
jax_config.update("jax_enable_x64", dtype is jnp.float64)
jax_config.update("jax_default_matmul_precision", "highest")


def build_atom_chain(rng, n_nuc, n_el_per_nuc, batch_size):
    R = np.arange(n_nuc)[:, None] * np.array([1, 0, 0])
    r = R[:, None, :] + jax.random.normal(rng, [batch_size, n_nuc, n_el_per_nuc, 3], dtype)
    r = jax.lax.collapse(r, 1, 3)
    Z = np.ones(n_nuc, dtype=int) * n_el_per_nuc
    mol = Mole(atom=[(int(Z_), R_) for R_, Z_ in zip(R, Z)]).build()
    return r, mol


def vmap_batch_and_jit(f):
    f = jax.vmap(f, in_axes=(None, 0, None))
    f = jax.jit(f, static_argnums=(2,))
    return f

def print_delta(x1, x2, name):
    delta = jnp.linalg.norm(x1 - x2)
    delta_rel = delta / jnp.linalg.norm(x2)
    print(f"{name:<20}: Delta abs: {delta:4.1e}, Delta rel: {delta_rel:4.1e}")


batch_size = 1
n_determinants = 32
rng_r, rng_model = jax.random.split(jax.random.PRNGKey(0))
electrons, mol = build_atom_chain(rng_r, n_nuc=25, n_el_per_nuc=2, batch_size=batch_size)
n_el = electrons.shape[-2]

model = SparseMoonWavefunction.create(
    mol,
    n_determinants=n_determinants,
    cutoff=10.0,
    feature_dim=32,
    nuc_mlp_depth=2,
    pair_mlp_widths=(16, 8),
    pair_n_envelopes=16,
    model_name="moon"
)
params = model.init(rng_model)
params = jtu.tree_map(lambda x: jnp.array(x, dtype), params)
static_args = model.input_constructor.get_static_input(electrons)

Eloc_ext = model.local_energy_dense(params, electrons[0], static_args)
Eloc_int = model.local_energy(params, electrons[0], static_args)
print(f"Eloc with full lap  : {Eloc_ext: 10.4f}")
print(f"Eloc with sparse lap: {Eloc_int: 10.4f}")
print(f"Delta (abs)         : {Eloc_int - Eloc_ext: 10.4f}")
print(f"Delta (rel)         : {(Eloc_int - Eloc_ext) / Eloc_ext: .2e}")
print("Done")
