# %%
# ruff: noqa: E402 # Allow setting environment variables before importing jax
import functools
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from folx.api import FwdJacobian, FwdLaplArray
from jax import config as jax_config
from pyscf.gto import Mole
from sparse_wf.jax_utils import fwd_lap
from sparse_wf.mcmc import init_electrons
from sparse_wf.model import SparseMoonWavefunction

jax_config.update("jax_enable_x64", True)
jax_config.update("jax_default_matmul_precision", "highest")


def build_atom_chain(n_nuc, Z):
    R = np.arange(n_nuc)[:, None] * np.array([1, 0, 0])
    Z = np.ones(n_nuc, dtype=int) * Z
    mol = Mole(atom=[(int(Z_), R_) for R_, Z_ in zip(R, Z)]).build()
    return mol


def build_model(mol):
    return SparseMoonWavefunction.create(
        mol,
        n_determinants=2,
        cutoff=2.0,
        feature_dim=32,
        nuc_mlp_depth=2,
        pair_mlp_widths=(16, 8),
        pair_n_envelopes=16,
    )


@functools.lru_cache()
def setup_inputs():
    rng = jax.random.PRNGKey(0)
    rng_r, rng_params = jax.random.split(rng)
    mol = build_atom_chain(10, 2)
    model = build_model(mol)
    electrons = init_electrons(rng_r, mol, batch_size=1)[0]
    params = model.init(rng_params)
    params, electrons = jtu.tree_map(lambda x: jnp.array(x, jnp.float64), (params, electrons))
    static_args = model.input_constructor.get_static_input(electrons)
    return model, electrons, params, static_args


def to_zero_padded(x, dependencies):
    jac = x.jacobian.data
    n_el = x.shape[-2]
    n_centers = jac.shape[-2]
    jac = jac.reshape([-1, 3, *jac.shape[1:]])
    jac_out = jnp.zeros([n_el, 3, *jac.shape[2:]])
    for i in range(n_centers):
        jac_out = jac_out.at[dependencies[i], ..., i, :].set(jac[:, ..., i, :], mode="drop")
    jac_out = jac_out.reshape([n_el * 3, *jac.shape[2:]])
    return FwdLaplArray(x.x, FwdJacobian(data=jac_out), x.laplacian)


def assert_close(x, y, atol=1e-8):
    error_val = jnp.linalg.norm(x.x - y.x)
    error_lap = jnp.linalg.norm(x.laplacian - y.laplacian)
    error_jac = jnp.linalg.norm(x.jacobian.dense_array - y.jacobian.dense_array)
    assert all([e < atol for e in [error_val, error_lap, error_jac]]), f"Errors: {error_val}, {error_lap}, {error_jac}"


def test_embedding():
    model, electrons, params, static_args = setup_inputs()
    embedding_int, dependencies = model._embedding_with_fwd_lap(params, electrons, static_args)
    embedding_ext = fwd_lap(lambda r: model._embedding(params, r, static_args))(electrons)
    embedding_int = to_zero_padded(embedding_int, dependencies)
    assert_close(embedding_int, embedding_ext)


def test_orbitals():
    model, electrons, params, static_args = setup_inputs()
    orbitals_int, dependencies = model._orbitals_with_fwd_lap(params, electrons, static_args)
    orbitals_ext = fwd_lap(lambda r: model.orbitals(params, r, static_args)[0])(electrons)
    orbitals_int = to_zero_padded(orbitals_int, dependencies)
    assert_close(orbitals_int, orbitals_ext)


def test_energy():
    model, electrons, params, static_args = setup_inputs()
    energy_sparse = model.local_energy(params, electrons, static_args)
    energy_dense = model.local_energy_dense(params, electrons, static_args)
    energy_sparse, energy_dense = float(energy_sparse), float(energy_dense)
    assert (
        jnp.abs(energy_sparse - energy_dense) < 1e-8
    ), f"Energy sparse: {energy_sparse:.6f}, Energy dense: {energy_dense:.6f}"


if __name__ == "__main__":
    test_embedding()
    test_orbitals()
    test_energy()
