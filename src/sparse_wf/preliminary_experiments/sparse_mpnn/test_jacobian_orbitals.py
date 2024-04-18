# %%
from sparse_wf.model import SparseMoonWavefunction
import jax
import jax.numpy as jnp
from pyscf.gto import Mole
from sparse_wf.jax_utils import fwd_lap
from sparse_wf.model.graph_utils import slogdet_with_sparse_fwd_lap
import numpy as np
from jax import config as jax_config
import jax.tree_util as jtu
import einops

jax_config.update("jax_enable_x64", False)
jax_config.update("jax_default_matmul_precision", "highest")


def build_atom_chain(rng, n_nuc, n_el_per_nuc):
    R = np.arange(n_nuc)[:, None] * np.array([1, 0, 0])
    r = R[:, None, :] + jax.random.normal(rng, [n_nuc, n_el_per_nuc, 3])
    r = einops.rearrange(r, "nuc el_per_nuc xyz -> (nuc el_per_nuc) xyz", nuc=n_nuc, el_per_nuc=n_el_per_nuc)
    Z = np.ones(n_nuc, dtype=int) * n_el_per_nuc
    mol = Mole(atom=[(int(Z_), R_) for R_, Z_ in zip(R, Z)]).build()
    return r, mol


def print_delta(x1, x2, name):
    delta = jnp.linalg.norm(x1 - x2)
    delta_rel = delta / jnp.linalg.norm(x2)
    print(f"{name:<20}: Delta abs: {delta:4.1e}, Delta rel: {delta_rel:4.1e}")


rng_r, rng_model = jax.random.split(jax.random.PRNGKey(0))
electrons, mol = build_atom_chain(rng_r, n_nuc=25, n_el_per_nuc=2)
n_el = electrons.shape[-2]

model = SparseMoonWavefunction.create(
    mol,
    n_determinants=1,
    cutoff=3.0,
    feature_dim=32,
    nuc_mlp_depth=2,
    pair_mlp_widths=(16, 8),
    pair_n_envelopes=16,
)
params = model.init(rng_model)
static_args = model.input_constructor.get_static_input(electrons)

# External Laplacian
orbitals_ext = fwd_lap(lambda r: model.orbitals(params, r, static_args)[0])(electrons)

# Internal laplacian + convert sparse to dense laplacian
orbitals_int, dependencies = model.orbitals_with_fwd_lap(params, electrons, static_args)
jac_orbitals_int_sparse = orbitals_int.jacobian.data.reshape([-1, 3, n_el, n_el])
jac_orbitals_int_dense = jnp.zeros([n_el, 3, n_el, n_el])
for i in range(n_el):
    jac_orbitals_int_dense = jac_orbitals_int_dense.at[dependencies[i], :, i, :].set(jac_orbitals_int_sparse[:, :, i, :])
jac_orbitals_int_dense = jac_orbitals_int_dense.reshape([n_el * 3, n_el, n_el])

print_delta(orbitals_int.x, orbitals_ext.x, "Val Orbitals")
print_delta(orbitals_int.laplacian, orbitals_ext.laplacian, "Lap Orbitals")
print_delta(jac_orbitals_int_dense, orbitals_ext.jacobian.data.squeeze(axis=-3), "Jac Orbitals") # squeeze to remove determinant axis
print("Done")
