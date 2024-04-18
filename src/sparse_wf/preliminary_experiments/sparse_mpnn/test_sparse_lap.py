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

def print_delta(x1, x2, name):
    delta = jnp.linalg.norm(x1 - x2)
    delta_rel = delta / jnp.linalg.norm(x2)
    print(f"{name:<20}: Delta abs: {delta:4.1e}, Delta rel: {delta_rel:4.1e}")


batch_size = 1
rng_r, rng_model = jax.random.split(jax.random.PRNGKey(0))
electrons, mol = build_atom_chain(rng_r, n_nuc=25, n_el_per_nuc=2, batch_size=batch_size)
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

# @vmap_batch_and_jit
def apply_with_internal_lap(params, electrons, static_args):
    orbitals, dependencies = model.orbitals_with_fwd_lap(params, electrons, static_args)
    orbitals = jtu.tree_map(lambda x: jnp.squeeze(x, axis=-3), orbitals) # select first determinant
    sign, logdet = slogdet_with_sparse_fwd_lap(orbitals, dependencies)
    return sign, logdet, orbitals

# @vmap_batch_and_jit
def apply_with_external_lap(params, electrons, static_args):
    sign, logdet = fwd_lap(model.signed, argnums=1)(params, electrons, static_args)
    return sign, logdet


sign_int, logdet_int, orbitals_int = apply_with_internal_lap(params, electrons[0], static_args)
sign_ext, logdet_ext = apply_with_external_lap(params, electrons[0], static_args)

print_delta(logdet_int.x, logdet_ext.x, "Val LogDet")
print_delta(logdet_int.jacobian.data, logdet_ext.jacobian.data.reshape([n_el, 3]), "Jac LogDet")
print_delta(logdet_int.laplacian, logdet_ext.laplacian, "Lap LogDet")
print("Done")
