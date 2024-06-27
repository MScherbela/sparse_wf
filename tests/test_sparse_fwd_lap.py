# %%
import functools
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
from sparse_wf.model.wave_function import MoonLikeWaveFunction
from sparse_wf.api import EmbeddingArgs, JastrowArgs, EnvelopeArgs

jax_config.update("jax_enable_x64", True)
jax_config.update("jax_default_matmul_precision", "highest")


def build_atom_chain(n_nuc, Z):
    R = np.arange(n_nuc)[:, None] * np.array([1, 0, 0])
    Z = np.ones(n_nuc, dtype=int) * Z
    mol = Mole(atom=[(int(Z_), R_) for R_, Z_ in zip(R, Z)]).build()
    return mol


def build_model(mol):
    return MoonLikeWaveFunction.create(
        mol,
        n_determinants=2,
        embedding=EmbeddingArgs(
            cutoff=2.0, feature_dim=128, nuc_mlp_depth=2, pair_mlp_widths=(16, 8), pair_n_envelopes=32
        ),
        jastrow=JastrowArgs(
            e_e_cusps="psiformer",
            use_log_jastrow=True,
            use_mlp_jastrow=True,
            mlp_depth=2,
            mlp_width=64,
        ),
        envelopes=EnvelopeArgs(envelope="glu", glu_args=dict(width=32, depth=2, n_envelopes=32), isotropic_args=None),
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
    params = model.init(rng_params, electrons)
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
    embedding_int, dependencies = model.embedding.apply_with_fwd_lap(params.embedding, electrons, static_args)
    embedding_int = to_zero_padded(embedding_int, dependencies)
    embedding_ext = fwd_lap(lambda r: model.embedding.apply(params.embedding, r, static_args))(electrons)
    assert embedding_ext.dtype == dtype
    assert embedding_int.dtype == dtype
    assert_close(embedding_int, embedding_ext)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_orbitals(dtype):
    model, electrons, params, static_args = setup_inputs(dtype)
    orbitals_int, dependencies = model.orbitals_with_fwd_lap(params, electrons, static_args)
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
