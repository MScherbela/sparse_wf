# %%
import functools
import itertools
import os

# ruff: noqa: E402 # Allow setting environment variables before importing jax
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import pytest
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
        feature_dim=256,
        nuc_mlp_depth=2,
        pair_mlp_widths=(16, 8),
        pair_n_envelopes=32,
    )


def change_float_dtype(x, dtype):
    if hasattr(x, "dtype") and x.dtype in [jnp.float16, jnp.float32, jnp.float64, np.float16, np.float32, np.float64]:
        return jnp.array(x, dtype)
    else:
        return x


@functools.lru_cache()
def setup_inputs(dtype):
    rng = jax.random.PRNGKey(0)
    rng_r, rng_params = jax.random.split(rng)
    mol = build_atom_chain(10, 2)
    model = build_model(mol)
    model = jtu.tree_map(lambda x: change_float_dtype(x, dtype), model)
    electrons = init_electrons(rng_r, mol, batch_size=1)[0]
    params = model.init(rng_params)
    model, params, electrons = jtu.tree_map(lambda x: change_float_dtype(x, dtype), (model, params, electrons))
    static_args = model.get_static_input(electrons)
    return model, electrons, params, static_args


def to_zero_padded(x, dependencies):
    jac = x.jacobian.data
    n_el = x.shape[-2]
    n_centers = jac.shape[-2]
    jac = jac.reshape([-1, 3, *jac.shape[1:]])
    jac_out = jnp.zeros([n_el, 3, *jac.shape[2:]], jac.dtype)
    for i in range(n_centers):
        jac_out = jac_out.at[dependencies[i], ..., i, :].set(jac[:, ..., i, :], mode="drop")
    jac_out = jac_out.reshape([n_el * 3, *jac.shape[2:]])
    return FwdLaplArray(x.x, FwdJacobian(data=jac_out), x.laplacian)


def get_relative_tolerance(dtype):
    return 1e-12 if (dtype == jnp.float64) else 1e-6


def assert_close(x, y, rtol=None):
    rtol = get_relative_tolerance(x.x.dtype) if rtol is None else rtol

    def rel_error(a, b):
        return jnp.linalg.norm(a - b) / jnp.linalg.norm(b)

    error_val = rel_error(x.x, y.x)
    error_lap = rel_error(x.laplacian, y.laplacian)
    error_jac = rel_error(x.jacobian.dense_array, y.jacobian.dense_array)
    assert all(
        [e < rtol for e in [error_val, error_lap, error_jac]]
    ), f"Rel. errors: {error_val}, {error_lap}, {error_jac}"


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_embedding(dtype):
    model, electrons, params, static_args = setup_inputs(dtype)
    embedding_int, dependencies = model._embedding_with_fwd_lap(params, electrons, static_args)
    embedding_ext = fwd_lap(lambda r: model._embedding(params, r, static_args))(electrons)
    embedding_int = to_zero_padded(embedding_int, dependencies)
    assert embedding_ext.dtype == dtype
    assert embedding_int.dtype == dtype
    assert_close(embedding_int, embedding_ext)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_orbitals(dtype):
    model, electrons, params, static_args = setup_inputs(dtype)
    orbitals_int, dependencies = model._orbitals_with_fwd_lap(params, electrons, static_args)
    orbitals_ext = fwd_lap(lambda r: model.orbitals(params, r, static_args)[0])(electrons)
    orbitals_int = to_zero_padded(orbitals_int, dependencies)
    assert orbitals_int.dtype == dtype
    assert orbitals_ext.dtype == dtype
    assert_close(orbitals_int, orbitals_ext)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_energy(dtype):
    # Use higher tolerance for energy due to possibly ill-conditioned orbital matrx
    # TODO: find a way to get samples/parameters which don't lead to ill-conditioned matrices and allow better testing
    rtol = get_relative_tolerance(dtype) * 1e3
    model, electrons, params, static_args = setup_inputs(dtype)

    energies = {}
    energies["sparse"] = model.local_energy(params, electrons, static_args)
    energies["dense"] = model.local_energy_dense(params, electrons, static_args)
    energies["loop"] = model.local_energy_dense_looped(params, electrons, static_args)
    for k, E in energies.items():
        assert E.dtype == dtype, f"{k}: {E.dtype}"
        assert np.isfinite(E), f"{k}: {E}"

    def rel_error(a, b):
        return jnp.abs(a - b) / jnp.abs(b)

    rel_errors = {f"{k1}-{k2}": rel_error(E1, E2) for (k1, E1), (k2, E2) in itertools.combinations(energies.items(), 2)}

    assert all([err < rtol for err in rel_errors.values()]), f"Energies: {energies}, Rel. errors: {rel_errors}"


if __name__ == "__main__":
    for dtype in [jnp.float32, jnp.float64]:
        # test_embedding(dtype)
        # test_orbitals(dtype)
        test_energy(dtype)
