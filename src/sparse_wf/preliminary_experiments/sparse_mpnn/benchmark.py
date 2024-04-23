# %%
from sparse_wf.model import SparseMoonWavefunction
from sparse_wf.model.moon import NrOfDependenciesMoon
import jax
from jax.lib import xla_bridge
import jax.numpy as jnp
import numpy as np
import time
from jax import config
from pyscf.gto import Mole
from typing import cast

dtype = jnp.float32
config.update("jax_enable_x64", dtype is jnp.float64)
config.update("jax_default_matmul_precision", "highest")


def build_atom_chain(rng, n_nuc, n_el_per_nuc, batch_size):
    R = np.arange(n_nuc)[:, None] * np.array([1, 0, 0])
    r = R[:, None, :] + jax.random.normal(rng, [batch_size, n_nuc, n_el_per_nuc, 3], dtype)
    r = jax.lax.collapse(r, 1, 3)
    Z = np.ones(n_nuc, dtype=int) * n_el_per_nuc
    mol = Mole(atom=[(int(Z_), R_) for R_, Z_ in zip(R, Z)]).build()
    return r, mol

print(f"Platform: {xla_bridge.get_backend().platform}, devices: {jax.devices()}", flush=True)

rng_r, rng_model = jax.random.split(jax.random.PRNGKey(0))
batch_size = 8
cutoff = 5.0
n_iterations = 5
n_el_per_nuc = 1
n_determinants = 4

n_el_values = [2*(int(np.round(n))//2) for n in np.geomspace(32, 512, 9)][::-1]
cutoff_values = np.array([3.0, 5.0, 7.0, 9.0])
for n_el in n_el_values:
    for cutoff in cutoff_values:
        n_el = int(n_el)
        assert n_el % n_el_per_nuc == 0, "n_el must be divisible by n_el_per_nuc"
        n_nuc = n_el // n_el_per_nuc
        r, mol = build_atom_chain(rng_r, n_nuc, n_el_per_nuc, batch_size)

        model = SparseMoonWavefunction.create(mol, n_determinants, cutoff, feature_dim=256, nuc_mlp_depth=3, pair_mlp_widths=(16, 8), pair_n_envelopes=16)
        params = model.init(rng_model)
        static = model.input_constructor.get_static_input(r)
        energy_func = jax.jit(model.local_energy, static_argnums=2)

        timings = np.zeros(n_iterations)
        for n in range(n_iterations):
            t0 = time.perf_counter()
            Eloc = energy_func(params, r, static)
            Eloc = jax.block_until_ready(Eloc)
            t1 = time.perf_counter()
            timings[n] = t1 - t0

            n_nb = static.n_neighbours.ee
            n_deps_max = cast(NrOfDependenciesMoon, static.n_deps).h_el_out
            print(f"n_el={n_el}, cutoff={cutoff:3.1f}, n_nb={n_nb:2d}, n_deps_max={n_deps_max:3d}, iteration={n}, t sparse energy ={timings[n]:5.3f} sec", flush=True)

        print("-" * 80)
        t_sparse = np.mean(timings[1:])
        print(f"Summary: batch_size={batch_size}, n_el={n_el}, cutoff={cutoff:3.1f}, n_nb={n_nb:2d}, n_deps_max={n_deps_max:3d}, sparse time={t_sparse:.3f}", flush=True)
        print("=" * 80)
