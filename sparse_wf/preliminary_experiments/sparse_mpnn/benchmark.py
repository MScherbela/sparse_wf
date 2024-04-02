# %%
from model import SparseWavefunctionWithFwdLap
from get_neighbours import get_ind_neighbours, get_max_nr_of_dependencies, pad_n_neighbours
import jax
import jax.numpy as jnp
import folx.api
import functools
import numpy as np
import time

from jax.lib import xla_bridge

print(xla_bridge.get_backend().platform, flush=True)

rng_r, rng_model = jax.random.split(jax.random.PRNGKey(0))
batch_size = 8
cutoff = 5.0
n_steps = 2
n_iterations = 3

for n_el in np.arange(50, 550, 50):
    n_el = int(n_el)
    R = jnp.arange(-n_el // 2, n_el // 2)[:, None] * jnp.array([1, 0, 0])
    r = jax.random.normal(rng_r, (batch_size, n_el, 3)) + R
    max_n_dependencies = get_max_nr_of_dependencies(r, cutoff, n_steps_max=2)
    max_n_dependencies = tuple([int(pad_n_neighbours(n_dep)) for n_dep in max_n_dependencies])
    ind_neighbour = get_ind_neighbours(r, cutoff, include_self=False)

    model = SparseWavefunctionWithFwdLap(R, cutoff, width=256)
    params = model.init(rng_model, r[0], ind_neighbour[0])

    @jax.jit
    @functools.partial(jax.vmap, in_axes=(None, 0, 0))
    def get_h_with_naive_lap(params, r, ind_neighbour):
        return folx.forward_laplacian(lambda r_: model.apply(params, r_, ind_neighbour))(r)

    @functools.partial(jax.jit, static_argnums=(3,))
    @functools.partial(jax.vmap, in_axes=(None, 0, 0, None))
    def get_h_with_sparse_lap(params, r, ind_neighbour, max_n_dependencies):
        return model.apply_with_fwd_lap(params, r, ind_neighbour, max_n_dependencies)

    times_naive = np.zeros(n_iterations)
    times_sparse = np.zeros(n_iterations)
    sparse_oom = False
    naive_oom = False
    for n in range(n_iterations):
        t0 = time.perf_counter()
        if not sparse_oom:
            try:
                h_with_sparse_lap = jax.block_until_ready(
                    get_h_with_sparse_lap(params, r, ind_neighbour, max_n_dependencies)
                )
            except Exception:
                print(f"n_el={n_el}; Sparse model out of memory")
                sparse_oom = True
        t1 = time.perf_counter()
        if not naive_oom:
            try:
                h_with_naive_lap = jax.block_until_ready(get_h_with_naive_lap(params, r, ind_neighbour))
            except Exception:
                print(f"n_el={n_el}; Naive model out of memory")
                naive_oom = True
        t2 = time.perf_counter()
        times_sparse[n] = t1 - t0
        times_naive[n] = t2 - t1
        if n == 0 and (not sparse_oom) and (not naive_oom):
            delta_h = jnp.linalg.norm(h_with_sparse_lap.x - h_with_naive_lap.x)
            delta_lap = jnp.linalg.norm(h_with_sparse_lap.laplacian - h_with_naive_lap.laplacian)
            print("Naive jacobian.data.shape: ", h_with_naive_lap.jacobian.data.shape)
            print("Sparse jacobian.data.shape: ", h_with_sparse_lap.jacobian.data.shape)
            print(f"delta_h: {delta_h:.1e} (rel: {delta_h / jnp.linalg.norm(h_with_naive_lap.x):.1e})")
            print(f"delta_lap: {delta_lap:.1e} (rel: {delta_lap / jnp.linalg.norm(h_with_naive_lap.laplacian):.1e})")
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
