# %%
from sparse_wf.model.wave_function import MoonLikeWaveFunction
from sparse_wf.api import EmbeddingArgs, MoonEmbeddingArgs, JastrowArgs, EnvelopeArgs, IsotropicEnvelopeArgs, NewEmbeddingArgs
from sparse_wf.model.moon import NrOfDependencies
from sparse_wf.mcmc import make_mcmc
import jax
from jax.lib import xla_bridge
import jax.numpy as jnp
import numpy as np
import time
from jax import config
from pyscf.gto import Mole
from typing import cast
import jax.tree_util as jtu
import functools

dtype = jnp.float32
config.update("jax_enable_x64", dtype is jnp.float64)
config.update("jax_default_matmul_precision", "highest")


def build_atom_chain(rng, n_nuc, n_el_per_nuc, batch_size):
    R = np.arange(n_nuc)[:, None] * np.array([1.8, 0, 0])
    r = R[:, None, :] + jax.random.normal(rng, [batch_size, n_nuc, n_el_per_nuc, 3], dtype)
    r = jax.lax.collapse(r, 1, 3)
    Z = np.ones(n_nuc, dtype=int) * n_el_per_nuc
    mol = Mole(atom=[(int(Z_), R_) for R_, Z_ in zip(R, Z)]).build()
    return r, mol


print(f"Platform: {xla_bridge.get_backend().platform}, devices: {jax.devices()}", flush=True)

rng_r, rng_model, rng_mcmc = jax.random.split(jax.random.PRNGKey(0), 3)
batch_size = 8
cutoff = 5.0
cutoff_1el = 20.0
n_iterations = 10
n_el_per_nuc = 1
n_determinants = 16
n_sampling_steps = 50
mcmc_stepsize = 0.1

csv_file = open("benchmark.csv", "w", buffering=1)
csv_file.write(
    "n_el,batch_size,n_sampling_steps,n_gpus,dtype,cutoff,cutoff_1el,n_nb,n_deps_max,iteration,t_sampling_low_rank,t_sampling_full_rank,t_energy_embed,t_energy\n"
)

# n_el_values = [20, 40]
n_el_values = [2 * (int(np.round(n)) // 2) for n in np.geomspace(32, 362, 8)]
cutoff_values = [3.0, 4.0, 5.0]
for cutoff in cutoff_values:
    for n_el in n_el_values:
        n_el = int(n_el)
        assert n_el % n_el_per_nuc == 0, "n_el must be divisible by n_el_per_nuc"
        n_nuc = n_el // n_el_per_nuc
        r, mol = build_atom_chain(rng_r, n_nuc, n_el_per_nuc, batch_size)

        print(f"Creating model for n_el={n_el}, cutoff={cutoff:3.1f}", flush=True)
        model = MoonLikeWaveFunction.create(
            mol,
            EmbeddingArgs(
                embedding="new",
                new=NewEmbeddingArgs(
                    cutoff=2.0,
                    cutoff_1el=20.0,
                    feature_dim=128,
                    nuc_mlp_depth=2,
                    pair_mlp_widths=(16, 8),
                    pair_n_envelopes=32,
                    low_rank_buffer=2,
                    n_updates=2,
                ),
                moon=MoonEmbeddingArgs(
                    cutoff=2.0,
                    cutoff_1el=20.0,
                    feature_dim=128,
                    nuc_mlp_depth=2,
                    pair_mlp_widths=(16, 8),
                    pair_n_envelopes=32,
                    low_rank_buffer=2,
                ),
            ),
            JastrowArgs(e_e_cusps="psiformer", use_log_jastrow=True, use_mlp_jastrow=True, mlp_depth=2, mlp_width=64),
            EnvelopeArgs(envelope="isotropic", isotropic_args=IsotropicEnvelopeArgs(n_envelopes=8), glu_args=None),
            n_determinants,
        )
        params = model.init(rng_model, r[0])
        static = model.get_static_input(r)

        mcmc_step_low_rank, _ = make_mcmc(model, "single-electron", mcmc_stepsize, n_sampling_steps)
        mcmc_step_low_rank = jax.jit(mcmc_step_low_rank, static_argnums=3)
        mcmc_step_full_rank, _ = make_mcmc(model, "all-electron", mcmc_stepsize, n_sampling_steps)
        mcmc_step_full_rank = jax.jit(mcmc_step_full_rank, static_argnums=3)

        embedding_with_lap_func = jax.jit(
            jax.vmap(lambda p, r_, s: model.embedding.apply_with_fwd_lap(p.embedding, r_, s), in_axes=(None, 0, None)),
            static_argnums=2,
        )
        energy_func = jax.jit(jax.vmap(model.local_energy, in_axes=(None, 0, None)), static_argnums=2)

        timings = np.zeros([n_iterations, 4])
        t = np.zeros(5)
        for n in range(n_iterations):
            t[0] = time.perf_counter()
            r_new = mcmc_step_low_rank(rng_mcmc, params, r, static, mcmc_stepsize)[0].block_until_ready()
            t[1] = time.perf_counter()
            r_new = mcmc_step_full_rank(rng_mcmc, params, r, static, mcmc_stepsize)[0].block_until_ready()
            t[2] = time.perf_counter()
            h, deps = embedding_with_lap_func(params, r, static)
            h = jtu.tree_map(jax.block_until_ready, h)
            t[3] = time.perf_counter()
            Eloc = energy_func(params, r, static).block_until_ready()
            t[4] = time.perf_counter()
            timings[n] = np.diff(t)

            n_nb = static.n_neighbours.ee
            n_deps_max = cast(NrOfDependencies, static.n_deps).h_el_out

            data = dict(
                n_el=n_el,
                batch_size=batch_size,
                n_sampling_steps=n_sampling_steps,
                n_gpus=jax.device_count(),
                dtype="float32" if dtype is jnp.float32 else "float64",
                cutoff=cutoff,
                cutoff_1el=cutoff_1el,
                n_nb=n_nb,
                n_deps_max=n_deps_max,
                iteration=n,
                t_sampling_low_rank=timings[n, 0],
                t_sampling_full_rank=timings[n, 1],
                t_embed=timings[n, 2],
                t_energy=timings[n, 3],
            )
            csv_file.write(",".join([str(v) for v in data.values()]) + "\n")

            print(
                f"n_el={n_el}, cutoff={cutoff:3.1f}, n_nb={n_nb:2d}, n_deps_max={n_deps_max:3d}, iteration={n}",
                flush=True,
            )

csv_file.close()
