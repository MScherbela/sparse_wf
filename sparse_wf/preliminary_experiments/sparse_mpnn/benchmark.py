# %%
from model import SparseMoonWavefunction
from get_neighbours import get_all_neighbour_indices, get_max_nr_of_dependencies, get_all_dependencies
import jax
from jax.lib import xla_bridge
import jax.numpy as jnp
import folx.api
import functools
import numpy as np
import time
from jax import config

config.update("jax_enable_x64", False)
config.update("jax_default_matmul_precision", "highest")


def build_atom_chain(rng, n_nuc, n_el_per_nuc, batch_size):
    R = jnp.arange(n_nuc)[:, None] * jnp.array([1, 0, 0])
    r = R[:, None, :] + jax.random.normal(rng, [batch_size, n_nuc, n_el_per_nuc, 3])
    r = jax.lax.collapse(r, 1, 3)
    n_el = r.shape[-2]
    spin = jnp.tile(jnp.arange(n_el) % 2, (batch_size, 1))
    return r, spin, R


print("Platform: ", xla_bridge.get_backend().platform, flush=True)

rng_r, rng_model = jax.random.split(jax.random.PRNGKey(0))
batch_size = 8
cutoff = 5.0
n_iterations = 3
n_el_per_nuc = 1

for n_el in np.arange(50, 550, 50):
    n_el = int(n_el)
    assert n_el % n_el_per_nuc == 0, "n_el must be divisible by n_el_per_nuc"
    n_nuc = n_el // n_el_per_nuc
    r, spin, R = build_atom_chain(rng_r, n_nuc, n_el_per_nuc, batch_size)

    model = SparseMoonWavefunction(
        n_orbitals=n_el, cutoff=cutoff, feature_dim=256, pair_mlp_widths=(16, 8), pair_n_envelopes=16
    )
    params = model.init(rng_model)

    idx_nb = get_all_neighbour_indices(r, R, cutoff=cutoff)
    n_deps_max = get_max_nr_of_dependencies(r, R, cutoff)
    n_deps_max = n_deps_max.pad(factor=1.1)
    deps, dep_maps = get_all_dependencies(idx_nb, n_deps_max)

    @functools.partial(jax.jit, static_argnums=(5,))
    @functools.partial(jax.vmap, in_axes=(None, 0, 0, None, 0, None, 0))
    def apply_with_internal_fwd_lap(params, r, s, R, idx_nb, n_deps, dep_maps):
        return model.apply(params, r, s, R, idx_nb, n_deps, dep_maps)

    @jax.jit
    @functools.partial(jax.vmap, in_axes=(None, 0, 0, None, 0))
    def apply_with_external_fwd_lap(params, r, s, R, idx_nb):
        return folx.forward_laplacian(lambda r_: model.apply(params, r_, s, R, idx_nb))(r)

    times_naive = np.zeros(n_iterations)
    times_sparse = np.zeros(n_iterations)
    sparse_oom = False
    naive_oom = False
    for n in range(n_iterations):
        t0 = time.perf_counter()
        if not sparse_oom:
            try:
                h = apply_with_internal_fwd_lap(params, r, spin, R, idx_nb, n_deps_max, dep_maps)
                h = jax.block_until_ready(h)
            except Exception:
                print(f"n_el={n_el}; Sparse model out of memory")
                sparse_oom = True
        t1 = time.perf_counter()
        if not naive_oom:
            try:
                h = apply_with_external_fwd_lap(params, r, spin, R, idx_nb)
                h = jax.block_until_ready(h)
            except Exception:
                print(f"n_el={n_el}; Naive model out of memory")
                naive_oom = True
        t2 = time.perf_counter()
        times_sparse[n] = t1 - t0
        times_naive[n] = t2 - t1

        print(
            f"n_el={n_el}, iteration={n}, sparse time={times_sparse[n]:.3f}, naive time={times_naive[n]:.3f}",
            flush=True,
        )
    print("-" * 80)
    t_naive = np.mean(times_naive[1:])
    t_sparse = np.mean(times_sparse[1:])
    print(
        f"Summary: batch_size={batch_size}, n_el={n_el}, sparse time={t_sparse:.3f}, naive time={t_naive:.3f}, speedup={t_naive/t_sparse:.1f}",
        flush=True,
    )
    print("=" * 80)
