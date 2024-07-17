# %%
import functools
import os
import socket

# ruff: noqa: E402 # Allow setting environment variables before importing jax
if socket.gethostname() == "gpu1-mat":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"


import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import pytest
from jax import config as jax_config
from sparse_wf.mcmc import init_electrons
from sparse_wf.model.utils import get_relative_tolerance
from sparse_wf.static_args import to_static

from utils import build_atom_chain, build_model, change_float_dtype

jax_config.update("jax_enable_x64", True)
jax_config.update("jax_default_matmul_precision", "highest")


@functools.lru_cache()
def setup_inputs(dtype, embedding):
    rng = jax.random.PRNGKey(0)
    rng_r, rng_params = jax.random.split(rng)
    mol = build_atom_chain(10, 2)
    model = build_model(mol, embedding)
    model = jtu.tree_map(lambda x: change_float_dtype(x, dtype), model)
    electrons = init_electrons(rng_r, mol, batch_size=1)[0]
    n_el = electrons.shape[-2]
    params = model.init(rng_params, electrons)
    params, electrons = jtu.tree_map(lambda x: change_float_dtype(x, dtype), (params, electrons))
    static_args = to_static(model.get_static_input(electrons, electrons, np.arange(n_el)))
    return model, electrons, params, static_args


# TODO: add separate testcases for embedding, jastrow, determinant, total_logpsi
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("embedding", ["moon", "new"])
def test_low_rank_update_logpsi(dtype, embedding):
    model, electrons, params, static_args = setup_inputs(dtype, embedding)
    (_, logpsi_old), state_old = model.log_psi_with_state(params, electrons, static_args)
    assert logpsi_old.dtype == dtype

    tol_kwargs = dict(rtol=get_relative_tolerance(dtype), atol=get_relative_tolerance(dtype))

    for step in range(2):
        ind_move = np.array(len(electrons) // 2)
        idx_changed = ind_move[None]
        dr = np.array([2, 0, 0]).astype(dtype)
        electrons_new = electrons.at[ind_move].add(dr)
        static_args = to_static(model.get_static_input(electrons, electrons_new, idx_changed))

        (_, logpsi_new), state_new = model.log_psi_with_state(params, electrons_new, static_args)
        (_, logpsi_new_update), state_new_update = model.log_psi_low_rank_update(
            params, electrons_new, idx_changed, static_args, state_old
        )
        assert logpsi_new.dtype == dtype
        assert logpsi_new_update.dtype == dtype

        for (key, s_new), s_new_update in zip(jtu.tree_leaves_with_path(state_new), jtu.tree_leaves(state_new_update)):
            np.testing.assert_allclose(
                s_new,
                s_new_update,
                err_msg=f"Step {step}, {key}",
                **tol_kwargs,
            )
        np.testing.assert_allclose(logpsi_new, logpsi_new_update, **tol_kwargs)
        state_old = state_new_update
        logpsi_old = logpsi_new_update


if __name__ == "__main__":
    for embedding in ["moon", "new"]:
        for dtype in [jnp.float32, jnp.float64]:
            test_low_rank_update_logpsi(dtype, embedding)
