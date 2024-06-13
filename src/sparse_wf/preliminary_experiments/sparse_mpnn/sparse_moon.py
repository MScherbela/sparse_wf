# %%
import functools
import itertools
import os
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
from sparse_wf.api import JastrowArgs, EmbeddingArgs, JastrowFactorArgs

jax_config.update("jax_enable_x64", False)
jax_config.update("jax_default_matmul_precision", "highest")


def build_atom_chain(n_nuc, Z):
    R = np.arange(n_nuc)[:, None] * np.array([1, 0, 0])
    Z = np.ones(n_nuc, dtype=int) * Z
    mol = Mole(atom=[(int(Z_), R_) for R_, Z_ in zip(R, Z)]).build()
    return mol


def build_model(mol):
    return MoonLikeWaveFunction.create(
        mol,
        embedding=EmbeddingArgs(
        cutoff=2.0,
        feature_dim=256,
        nuc_mlp_depth=2,
        pair_mlp_widths=(16, 8),
        pair_n_envelopes=32,
        ),
        n_envelopes=8,
        n_determinants=2,
        jastrow=JastrowArgs(
        use_e_e_cusp=True,
        mlp=JastrowFactorArgs(use=False, embedding_n_hidden=None, soe_n_hidden=None),
        log=JastrowFactorArgs(use=False, embedding_n_hidden=None, soe_n_hidden=None),
        use_yukawa_jastrow=False)
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


if __name__ == "__main__":
    rng = jax.random.PRNGKey(0)

    model, electrons, params, static_args = setup_inputs(jnp.float32)
    params = model.init(rng, electrons)
    h = model.embedding.apply(params.embedding, electrons, static_args)
    embedding_int, dependencies = model.embedding.apply_with_fwd_lap(params.embedding, electrons, static_args)
    # embedding_ext = fwd_lap(lambda r: model.embedding(params, r, static_args))(electrons)
