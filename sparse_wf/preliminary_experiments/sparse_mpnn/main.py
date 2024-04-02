# %%
from sparse_wf.preliminary_experiments.sparse_mpnn.model import SparseWavefunctionWithFwdLap
from sparse_wf.preliminary_experiments.sparse_mpnn.get_neighbours import get_ind_neighbours, get_max_nr_of_dependencies, pad_n_neighbours
import jax
import jax.numpy as jnp
import folx.api
import functools

rng_r, rng_model = jax.random.split(jax.random.PRNGKey(0))
batch_size = 10
n_el = 50
cutoff = 5.0
n_steps = 2
R = jnp.arange(-n_el // 2, n_el // 2)[:, None] * jnp.array([1, 0, 0])
r = jax.random.normal(rng_r, (batch_size, n_el, 3)) + R
max_n_dependencies = get_max_nr_of_dependencies(r, cutoff, n_steps_max=2)
max_n_dependencies = tuple([int(pad_n_neighbours(n_dep)) for n_dep in max_n_dependencies])
ind_neighbour = get_ind_neighbours(r, cutoff, include_self=False)
#%%
model = SparseWavefunctionWithFwdLap(R, cutoff)
params = model.init(rng_model, r[0], ind_neighbour[0])

@jax.jit
@functools.partial(jax.vmap, in_axes=(None, 0, 0))
def get_h_with_naive_lap(params, r, ind_neighbour):
    return folx.forward_laplacian(lambda r_: model.apply(params, r_, ind_neighbour))(r)

@functools.partial(jax.jit, static_argnums=(3,))
@functools.partial(jax.vmap, in_axes=(None, 0, 0, None))
def get_h_with_sparse_lap(params, r, ind_neighbour, max_n_dependencies):
    return model.apply_with_fwd_lap(params, r, ind_neighbour, max_n_dependencies)

h_with_sparse_lap = get_h_with_sparse_lap(params, r, ind_neighbour, max_n_dependencies)
h_with_naive_lap = get_h_with_naive_lap(params, r, ind_neighbour)
delta_h = jnp.linalg.norm(h_with_sparse_lap.x - h_with_naive_lap.x)
delta_lap = jnp.linalg.norm(h_with_sparse_lap.laplacian - h_with_naive_lap.laplacian)
print("Naive jacobian.data.shape: ", h_with_naive_lap.jacobian.data.shape)
print("Sparse jacobian.data.shape: ", h_with_sparse_lap.jacobian.data.shape)
print(f"delta_h: {delta_h:.1e} (rel: {delta_h / jnp.linalg.norm(h_with_naive_lap.x):.1e})")
print(f"delta_lap: {delta_lap:.1e} (rel: {delta_lap / jnp.linalg.norm(h_with_naive_lap.laplacian):.1e})")
