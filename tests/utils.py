import functools

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from pyscf.gto import Mole

from sparse_wf.mcmc import init_electrons
from sparse_wf.api import EmbeddingArgs, MoonEmbeddingArgs, EnvelopeArgs, JastrowArgs, NewEmbeddingArgs
from sparse_wf.model.wave_function import MoonLikeWaveFunction


def build_atom_chain(n_nuc, Z):
    R = np.arange(n_nuc)[:, None] * np.array([1, 0, 0])
    Z = np.ones(n_nuc, dtype=int) * Z
    mol = Mole(atom=[(int(Z_), R_) for R_, Z_ in zip(R, Z)]).build()
    return mol


def build_model(mol, embedding="moon"):
    return MoonLikeWaveFunction.create(
        mol,
        n_determinants=2,
        embedding=EmbeddingArgs(
            embedding=embedding,
            new=NewEmbeddingArgs(
                cutoff=3.0,
                cutoff_1el=20.0,
                feature_dim=128,
                nuc_mlp_depth=2,
                pair_mlp_widths=(16, 8),
                pair_n_envelopes=32,
                low_rank_buffer=2,
                n_updates=1,
            ),
            moon=MoonEmbeddingArgs(
                cutoff=3.0,
                cutoff_1el=20.0,
                feature_dim=128,
                nuc_mlp_depth=2,
                pair_mlp_widths=(16, 8),
                pair_n_envelopes=32,
                low_rank_buffer=2,
            ),
        ),
        jastrow=JastrowArgs(
            e_e_cusps="psiformer",
            use_e_e_mlp=True,
            use_log_jastrow=True,
            use_mlp_jastrow=True,
            mlp_depth=2,
            mlp_width=64,
        ),
        envelopes=EnvelopeArgs(
            envelope="isotropic", glu_args=dict(width=32, depth=2, n_envelopes=32), isotropic_args=dict(n_envelopes=8)
        ),
        spin_restricted=True,
    )


def change_float_dtype(x, dtype):
    if hasattr(x, "dtype") and x.dtype in [jnp.float16, jnp.float32, jnp.float64, np.float16, np.float32, np.float64]:
        return jnp.array(x, dtype)
    else:
        return x


def change_jastrow_scales(key, param):
    name = jtu.keystr(key)
    if "jastrow" in name and "scale" in name:
        return jnp.ones(param.shape)
    else:
        return param


@functools.lru_cache()
def setup_inputs(dtype, embedding):
    rng = jax.random.PRNGKey(0)
    rng_r, rng_params = jax.random.split(rng)
    mol = build_atom_chain(10, 2)
    model = build_model(mol, embedding)
    model = jtu.tree_map(lambda x: change_float_dtype(x, dtype), model)
    electrons = init_electrons(rng_r, mol, batch_size=1)[0]
    params = model.init(rng_params, electrons)
    params = jtu.tree_map_with_path(change_jastrow_scales, params)
    model, params, electrons = jtu.tree_map(lambda x: change_float_dtype(x, dtype), (model, params, electrons))
    static_args = model.get_static_input(electrons).to_static()
    return model, electrons, params, static_args
