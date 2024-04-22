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
from folx.api import FwdLaplArray

dtype = jnp.float64
jax_config.update("jax_enable_x64", dtype is jnp.float64)
jax_config.update("jax_default_matmul_precision", "highest")


def build_atom_chain(rng, n_nuc, n_el_per_nuc):
    n_el = n_nuc * n_el_per_nuc
    R = np.arange(n_nuc)[:, None] * np.array([1, 0, 0])
    r = R[:, None, :] + jax.random.normal(rng, [n_nuc, n_el_per_nuc, 3], dtype)
    r = r.reshape([n_el, 3])
    Z = np.ones(n_nuc, dtype=int) * n_el_per_nuc
    mol = Mole(atom=[(int(Z_), R_) for R_, Z_ in zip(R, Z)]).build()
    return r, mol


def _print_delta(x1, x2, name):
    delta = jnp.linalg.norm(x1 - x2)
    delta_rel = delta / jnp.linalg.norm(x2)
    print(f"{name:<20}: Delta abs: {delta:4.1e}, Delta rel: {delta_rel:4.1e}")

def print_delta(x1: FwdLaplArray, x2: FwdLaplArray, dependencies, name):
    _print_delta(x1.x, x2.x, name + " value")
    _print_delta(x1.laplacian, x2.laplacian, name + " laplacian")
    _print_delta(x1.jacobian.data,  to_zero_padded(x2.jacobian.data, dependencies), name + " jacobian")


def to_zero_padded(jac, dependencies):
    jac = jac.reshape([-1, 3, *jac.shape[1:]])
    jac_out = jnp.zeros([n_el, 3, *jac.shape[2:]])
    for i in range(n_el):
        jac_out = jac_out.at[dependencies[i], ..., i, :].set(jac[:, ..., i, :], mode="drop")
    jac_out = jac_out.reshape([n_el * 3, *jac.shape[2:]])
    return jac_out



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
params = jtu.tree_map(lambda x: jnp.array(x, dtype), params)
static_args = model.input_constructor.get_static_input(electrons)

print("Embeddings")
embedding_int, dependencies = model._embedding_with_fwd_lap(params, electrons, static_args)
embedding_ext = fwd_lap(lambda r: model._embedding(params, r, static_args))(electrons)
print_delta(embedding_ext, embedding_int, dependencies, "Embedding")
print("-"*60 + "\n")

print("Orbitals")
orbitals_int, dependencies = model._orbitals_with_fwd_lap(params, electrons, static_args)
orbitals_ext = fwd_lap(lambda r: model.orbitals(params, r, static_args)[0])(electrons)
print_delta(orbitals_ext, orbitals_int, dependencies, "Orbitals")
print("-"*60 + "\n")

