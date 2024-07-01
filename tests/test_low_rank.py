# %%
import functools
import os

from sparse_wf.model.utils import get_relative_tolerance
from utils import build_atom_chain, build_model, change_float_dtype

# ruff: noqa: E402 # Allow setting environment variables before importing jax
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import pytest
from jax import config as jax_config
from sparse_wf.mcmc import init_electrons

jax_config.update("jax_enable_x64", True)
jax_config.update("jax_default_matmul_precision", "highest")


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


# TODO: add separate testcases for embedding, jastrow, determinant, total_logpsi
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_low_rank_update_logpsi(dtype):
    model, electrons, params, static_args = setup_inputs(dtype)
    logpsi_old, state = model(params, electrons, static_args, return_state=True)
    assert logpsi_old.dtype == dtype

    ind_move = np.array(len(electrons) // 2)
    idx_changed = ind_move[None]
    dr = np.array([2, 0, 0]).astype(dtype)
    electrons_new = electrons.at[ind_move].add(dr)

    logpsi_new = model(params, electrons_new, static_args)
    logpsi_new_update, state_new = model.update_logpsi(params, electrons_new, idx_changed, static_args, state)

    assert logpsi_new.dtype == dtype
    assert logpsi_new_update.dtype == dtype
    assert jnp.allclose(logpsi_new, logpsi_new_update, rtol=get_relative_tolerance(dtype))
