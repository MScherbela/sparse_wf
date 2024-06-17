# %%
import functools
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
        feature_dim=25,
        nuc_mlp_depth=2,
        pair_mlp_widths=(8, 4),
        pair_n_envelopes=7,
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

def print_diff(h_dense, h_sparse):
    def _diff_string(a, b):
        diff = jnp.linalg.norm(a - b)
        return f"abs={diff:.1e} rel={diff / jnp.linalg.norm(a):.1e}"
    print(f"Value:     {_diff_string(h_dense.x, h_sparse.x)}")
    print(f"Jacobian:  {_diff_string(h_dense.jacobian.data, h_sparse.jacobian.data)}")
    print(f"Laplacian: {_diff_string(h_dense.laplacian, h_sparse.laplacian)}")


if __name__ == "__main__":
    rng = jax.random.PRNGKey(0)

    model, electrons, params, static_args = setup_inputs(jnp.float32)
    params = model.init(rng, electrons)
    print("Computing sparse")
    h_sparse, dependencies = model.embedding.apply_with_fwd_lap(params.embedding, electrons, static_args)
    h_sparse = to_zero_padded(h_sparse, dependencies)
    # h_sparse = model.embedding.apply_with_fwd_lap(params.embedding, electrons, static_args)

    print("Computing dense")
    h_dense = fwd_lap(model.embedding.apply, argnums=1)(params.embedding, electrons, static_args)
    # h_dense = model.embedding.apply(params.embedding, electrons, static_args)

    print_diff(h_dense, h_sparse)

    # diff = jnp.linalg.norm(h_dense - h_sparse)
    # print(f"Diff: {diff:.1e}, rel = {diff / jnp.linalg.norm(h_dense):.1e}")

    # for key in h_dense:
    #     diff = jnp.linalg.norm(h_sparse[key] - h_dense[key])
    #     print(f"{key:<10}: {diff:.1e}, rel = {diff / jnp.linalg.norm(h_dense[key]):.1e}")
    # print("Done")