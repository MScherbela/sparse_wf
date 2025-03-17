# %%
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
from sparse_wf.model.utils import get_relative_tolerance

from utils import setup_inputs

jax_config.update("jax_enable_x64", True)
jax_config.update("jax_default_matmul_precision", "highest")

MODELS_TO_TEST = ["new_sparse"]


# TODO: add separate testcases for embedding, jastrow, determinant, total_logpsi
@pytest.mark.parametrize("dtype", [jnp.float64])
@pytest.mark.parametrize("embedding", MODELS_TO_TEST)
def test_low_rank_update_logpsi(dtype, embedding):
    model, electrons, params, static_args = setup_inputs(dtype, embedding)
    (_, logpsi_old), state_old = model.log_psi_with_state(params, electrons, static_args)
    assert logpsi_old.dtype == dtype

    tol_kwargs = dict(rtol=get_relative_tolerance(dtype), atol=get_relative_tolerance(dtype))

    for step in range(2):
        idx_changed = np.array([3])
        dr = np.array([2, 0, 0]).astype(dtype)
        electrons_new = electrons.at[idx_changed].add(dr)
        static_args = model.get_static_input(electrons, electrons_new, idx_changed).to_static()

        (_, logpsi_new), state_new = model.log_psi_with_state(params, electrons_new, static_args)
        (_, logpsi_new_update), state_new_update = model.log_psi_low_rank_update(
            params, electrons_new, idx_changed, static_args, state_old
        )
        assert logpsi_new.dtype == dtype
        assert logpsi_new_update.dtype == dtype

        for (key, s_new), s_new_update in zip(jtu.tree_leaves_with_path(state_new), jtu.tree_leaves(state_new_update)):
            name = jax.tree_util.keystr(key)
            if "determinant" in name or "inverses" in name:
                current_tol_kwargs = {k: v * 1000 for k, v in tol_kwargs.items()}
            else:
                current_tol_kwargs = tol_kwargs
            np.testing.assert_allclose(
                s_new,
                s_new_update,
                err_msg=f"Step {step}, {key}",
                **current_tol_kwargs,
            )
        np.testing.assert_allclose(logpsi_new, logpsi_new_update, **tol_kwargs)
        state_old = state_new_update


if __name__ == "__main__":
    for embedding in MODELS_TO_TEST:
        for dtype in [jnp.float64, jnp.float32]:
            test_low_rank_update_logpsi(dtype, embedding)
