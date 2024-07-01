import jax.numpy as jnp
import numpy as np
from pyscf.gto import Mole

from sparse_wf.api import EmbeddingArgs, EnvelopeArgs, JastrowArgs
from sparse_wf.model.wave_function import MoonLikeWaveFunction


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
            cutoff=2.0,
            cutoff_1el=20.0,
            feature_dim=128,
            nuc_mlp_depth=2,
            pair_mlp_widths=(16, 8),
            pair_n_envelopes=32,
            low_rank_buffer=2,
        ),
        jastrow=JastrowArgs(
            e_e_cusps="psiformer",
            use_log_jastrow=True,
            use_mlp_jastrow=True,
            mlp_depth=2,
            mlp_width=64,
        ),
        envelopes=EnvelopeArgs(
            envelope="isotropic", glu_args=dict(width=32, depth=2, n_envelopes=32), isotropic_args=dict(n_envelopes=8)
        ),
    )


def change_float_dtype(x, dtype):
    if hasattr(x, "dtype") and x.dtype in [jnp.float16, jnp.float32, jnp.float64, np.float16, np.float32, np.float64]:
        return jnp.array(x, dtype)
    else:
        return x
