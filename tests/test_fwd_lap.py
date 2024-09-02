# %%
import os
import socket

from sparse_wf.model.utils import get_relative_tolerance
from utils import setup_inputs

# ruff: noqa: E402 # Allow setting environment variables before importing jax
if socket.gethostname() == "gpu1-mat":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable GPUs on the HGX, because otherwise it will use all GPUs
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

import jax.numpy as jnp
import numpy as np
import pytest
from folx.api import FwdJacobian, FwdLaplArray
from jax import config as jax_config
from sparse_wf.jax_utils import fwd_lap

from sparse_wf.model.sparse_fwd_lap import NodeWithFwdLap

jax_config.update("jax_enable_x64", True)
jax_config.update("jax_default_matmul_precision", "highest")


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


@pytest.mark.parametrize("embedding", ["moon", "new", "new_sparse"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_embedding(dtype, embedding):
    model, electrons, params, static_args = setup_inputs(dtype, embedding)
    if embedding == "new_sparse":
        embedding_int = model.embedding.apply_with_fwd_lap(params.embedding, electrons, static_args)
        embedding_int = embedding_int.to_folx()
    else:
        embedding_int, dependencies = model.embedding.apply_with_fwd_lap(params.embedding, electrons, static_args)
        embedding_int = to_zero_padded(embedding_int, dependencies)
    embedding_ext = fwd_lap(lambda r: model.embedding.apply(params.embedding, r, static_args))(electrons)
    assert embedding_ext.dtype == dtype
    assert embedding_int.dtype == dtype
    assert_close(embedding_int, embedding_ext)


@pytest.mark.parametrize("embedding", ["moon", "new", "new_sparse"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_orbitals(dtype, embedding):
    model, electrons, params, static_args = setup_inputs(dtype, embedding)
    orbitals_ext = fwd_lap(lambda r: model.orbitals(params, r, static_args)[0])(electrons)  # [det x n_el x n_orb]

    if embedding == "new_sparse":
        # new_sparse orbitals have shape [dets, n_el, n_orb]
        # Reorder to [n_el, dets, n_orb] such that the first dimension is the electron index, to allow to_folx()
        embeddings = model.embedding.apply_with_fwd_lap(params.embedding, electrons, static_args)
        orbitals_int = model.to_orbitals.fwd_lap(params.to_orbitals, electrons, embeddings)
        orbitals_int = NodeWithFwdLap(
            jnp.moveaxis(orbitals_int.x, 0, 1),
            jnp.moveaxis(orbitals_int.jac, 0, 2),
            jnp.moveaxis(orbitals_int.lap, 0, 1),
            orbitals_int.idx_ctr,
            orbitals_int.idx_dep,
        ).to_folx()
        # Move the electron index back to the first dimension: [dets, n_el, n_orb]
        orbitals_int = fwd_lap(lambda x: jnp.moveaxis(x, 1, 0))(orbitals_int)
    else:
        embeddings, dependencies = model.embedding.apply_with_fwd_lap(params.embedding, electrons, static_args)
        orbitals_int = model.to_orbitals.fwd_lap(params.to_orbitals, electrons, embeddings)
        orbitals_int = to_zero_padded(orbitals_int, dependencies)
    assert orbitals_int.dtype == dtype
    assert orbitals_ext.dtype == dtype
    assert_close(orbitals_int, orbitals_ext)


@pytest.mark.parametrize("embedding", ["moon", "new", "new_sparse"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_energy(dtype, embedding):
    # Use higher tolerance for energy due to possibly ill-conditioned orbital matrx
    # TODO: find a way to get samples/parameters which don't lead to ill-conditioned matrices and allow better testing
    rtol = get_relative_tolerance(dtype) * 1e3
    model, electrons, params, static_args = setup_inputs(dtype, embedding)

    E_dense = model.local_energy_dense(params, electrons, static_args)
    E_sparse = model.local_energy(params, electrons, static_args)
    for E, label in zip([E_sparse, E_dense], ["sparse", "dense"]):
        assert E.dtype == dtype, f"energy {label}: {E.dtype} != {dtype}"
        assert np.isfinite(E), f"energy {label}: {E} != {dtype}"

    rel_error = jnp.abs(E_sparse - E_dense) / jnp.abs(E_dense)

    assert rel_error < rtol, f"Rel. error |E_sparse - E_dense| / |E_dense|: {rel_error}"


if __name__ == "__main__":
    test_orbitals(jnp.float32, "new_sparse")
    # for embedding in ["moon", "new", "new_sparse"]:
    #     for dtype in [jnp.float64, jnp.float32]:
    #         test_embedding(dtype, embedding)
    #         test_orbitals(dtype, embedding)
    #         test_energy(dtype, embedding)
