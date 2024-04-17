# %%
from sparse_wf.model import SparseMoonWavefunction
import jax
import jax.numpy as jnp
from pyscf.gto import Mole
from sparse_wf.jax_utils import fwd_lap
import numpy as np
from jax import config as jax_config

jax_config.update("jax_enable_x64", False)
jax_config.update("jax_default_matmul_precision", "highest")


def build_atom_chain(rng, n_nuc, n_el_per_nuc, batch_size):
    R = np.arange(n_nuc)[:, None] * np.array([1, 0, 0])
    r = R[:, None, :] + jax.random.normal(rng, [batch_size, n_nuc, n_el_per_nuc, 3])
    r = jax.lax.collapse(r, 1, 3)
    Z = np.ones(n_nuc, dtype=int) * n_el_per_nuc
    mol = Mole(atom=[(int(Z_), R_) for R_, Z_ in zip(R, Z)]).build()
    return r, mol


def vmap_batch_and_jit(f):
    f = jax.vmap(f, in_axes=(None, 0, None))
    f = jax.jit(f, static_argnums=(2,))
    return f


rng_r, rng_model = jax.random.split(jax.random.PRNGKey(0))
electrons, mol = build_atom_chain(rng_r, n_nuc=25, n_el_per_nuc=2, batch_size=1)

model = SparseMoonWavefunction.create(
    mol,
    n_determinants=1,
    cutoff=4.0,
    feature_dim=64,
    nuc_mlp_depth=3,
    pair_mlp_widths=(16, 8),
    pair_n_envelopes=16,
)
params = model.init(rng_model)
static_args = model.input_constructor.get_static_input(electrons)


apply_with_external_fwd_lap = vmap_batch_and_jit(fwd_lap(model.orbitals, argnums=1))
apply_with_internal_fwd_lap = vmap_batch_and_jit(model.orbitals_with_fwd_lap)

h_with_internal_lap = apply_with_internal_fwd_lap(params, electrons, static_args)
print("Internal lap jacobian.data.shape: ", h_with_internal_lap.jacobian.data.shape)

h_with_external_lap = apply_with_external_fwd_lap(params, electrons, static_args)
print("External lap jacobian.data.shape: ", h_with_external_lap.jacobian.data.shape)

delta_h = jnp.linalg.norm(h_with_internal_lap.x - h_with_external_lap.x)
delta_lap = jnp.linalg.norm(h_with_internal_lap.laplacian - h_with_external_lap.laplacian)
print(f"delta_h: {delta_h:.1e} (rel: {delta_h / jnp.linalg.norm(h_with_external_lap.x):.1e})")
print(f"delta_lap: {delta_lap:.1e} (rel: {delta_lap / jnp.linalg.norm(h_with_external_lap.laplacian):.1e})")
