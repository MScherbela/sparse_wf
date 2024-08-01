# %%
import functools
import os
import socket

from sparse_wf.model.utils import get_relative_tolerance
from sparse_wf.static_args import to_static
from utils import build_atom_chain, build_model, change_float_dtype

# ruff: noqa: E402 # Allow setting environment variables before importing jax
if socket.gethostname() == "gpu1-mat":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable GPUs on the HGX, because otherwise it will use all GPUs
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import pytest
from folx.api import FwdJacobian, FwdLaplArray
from jax import config as jax_config
from sparse_wf.jax_utils import fwd_lap
from sparse_wf.mcmc import init_electrons
import chex
from sparse_wf.model.sparse_fwd_lap import NodeWithFwdLap

chex.fake_pmap_and_jit().start()


jax_config.update("jax_enable_x64", True)
jax_config.update("jax_default_matmul_precision", "highest")


@functools.lru_cache()
def setup_inputs(dtype):
    rng = jax.random.PRNGKey(0)
    rng_r, rng_params = jax.random.split(rng)
    mol = build_atom_chain(10, 2)
    model = build_model(mol, "new_sparse")
    model = jtu.tree_map(lambda x: change_float_dtype(x, dtype), model)
    electrons = init_electrons(rng_r, mol, batch_size=1)[0]
    params = model.init(rng_params, electrons)
    model, params, electrons = jtu.tree_map(lambda x: change_float_dtype(x, dtype), (model, params, electrons))
    static = model.get_static_input(electrons)
    static = jtu.tree_map(lambda x: 1.2 * x, static)
    static_args = to_static(static)
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


def assert_close(x: FwdLaplArray, y: FwdLaplArray, rtol=None):
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
    embedding_int = model.embedding.apply_with_fwd_lap(params.embedding, electrons, static_args)
    embedding_ext = fwd_lap(lambda r: model.embedding.apply(params.embedding, r, static_args))(electrons)
    assert embedding_ext.dtype == dtype
    assert embedding_int.dtype == dtype
    assert_close(embedding_int.to_folx(), embedding_ext)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_orbitals(dtype):
    model, electrons, params, static_args = setup_inputs(dtype)
    embeddings = model.embedding.apply_with_fwd_lap(params.embedding, electrons, static_args)
    orbitals_int = model._orbitals_with_fwd_lap_sparse(params, electrons, embeddings)
    orbitals_ext = fwd_lap(lambda r: model.orbitals(params, r, static_args)[0])(electrons)
    orbitals_ext = fwd_lap(lambda x: jnp.moveaxis(x, 0, 1))(orbitals_ext)
    orbitals_int = NodeWithFwdLap(
        jnp.moveaxis(orbitals_int.x, 0, 1),
        jnp.moveaxis(orbitals_int.jac, 0, 2),
        jnp.moveaxis(orbitals_int.lap, 0, 1),
        orbitals_int.idx_ctr,
        orbitals_int.idx_dep,
    ).to_folx()
    assert orbitals_int.dtype == dtype
    assert orbitals_ext.dtype == dtype
    assert_close(orbitals_int, orbitals_ext)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_energy(dtype):
    # Use higher tolerance for energy due to possibly ill-conditioned orbital matrx
    # TODO: find a way to get samples/parameters which don't lead to ill-conditioned matrices and allow better testing
    rtol = get_relative_tolerance(dtype) * 1e3
    model, electrons, params, static_args = setup_inputs(dtype)

    E_dense = model.local_energy_dense(params, electrons, static_args)
    E_sparse = model.local_energy(params, electrons, static_args)
    for E, label in zip([E_sparse, E_dense], ["sparse", "dense"]):
        assert E.dtype == dtype, f"energy {label}: {E.dtype} != {dtype}"
        assert np.isfinite(E), f"energy {label}: {E} != {dtype}"

    rel_error = jnp.abs(E_sparse - E_dense) / jnp.abs(E_dense)

    assert rel_error < rtol, f"Rel. error |E_sparse - E_dense| / |E_dense|: {rel_error}"


if __name__ == "__main__":
    for dtype in [jnp.float32, jnp.float64]:
        test_embedding(dtype)
        test_orbitals(dtype)
        test_energy(dtype)
