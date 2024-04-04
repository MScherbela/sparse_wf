# %%
from sparse_wf.preliminary_experiments.sparse_mpnn.model import SparseMoonWavefunction
from sparse_wf.preliminary_experiments.sparse_mpnn.get_neighbours import (
    get_all_neighbour_indices,
    get_all_dependencies,
    get_max_nr_of_dependencies,
)
import jax
import jax.numpy as jnp


def build_chain(rng, n_nuc, n_el_per_nuc, batch_size):
    R = jnp.arange(n_nuc)[:, None] * jnp.array([1, 0, 0])
    r = R[:, None, :] + jax.random.normal(rng, [batch_size, n_nuc, n_el_per_nuc, 3])
    r = jax.lax.collapse(r, 1, 3)
    n_el = r.shape[-2]
    spin = jnp.tile(jnp.arange(n_el) % 2, (batch_size, 1))
    return r, spin, R


cutoff = 2.0
rng_r, rng_model = jax.random.split(jax.random.PRNGKey(0))
r, s, R = build_chain(rng_r, n_nuc=25, n_el_per_nuc=2, batch_size=1)
n_el = r.shape[-2]

model = SparseMoonWavefunction(
    n_orbitals=n_el, cutoff=cutoff, feature_dim=64, pair_mlp_widths=(16, 8), pair_n_envelopes=16
)
params = model.init(rng_model)


idx_nb = get_all_neighbour_indices(r, R, cutoff=cutoff)
n_deps_max = get_max_nr_of_dependencies(r, R, cutoff)
n_deps_max = tuple([int(n) for n in n_deps_max])
deps, dep_maps = get_all_dependencies(idx_nb, *n_deps_max)
# apply = jax.vmap(model.apply, in_axes=(None, 0, 0, None, 0)) # vmap over batch

# vmap over batch-dim for all args except params and R
apply_with_fwd_lap = jax.vmap(model.apply, in_axes=(None, 0, 0, None, 0, 0, 0))
h = apply_with_fwd_lap(params, r, s, R, idx_nb, deps, dep_maps)


# cutoff = 5.0
# n_steps = 2
# R = jnp.arange(-n_el // 2, n_el // 2)[:, None] * jnp.array([1, 0, 0])
# r = jax.random.normal(rng_r, (batch_size, n_el, 3)) + R
# max_n_dependencies = get_max_nr_of_dependencies(r, cutoff, n_steps_max=2)
# max_n_dependencies = tuple([int(pad_n_neighbours(n_dep)) for n_dep in max_n_dependencies])
# ind_neighbour = get_ind_neighbours(r, cutoff, include_self=False)
# %%
# model = SparseWavefunctionWithFwdLap(R, cutoff)
# params = model.init(rng_model, r[0], ind_neighbour[0])

# @jax.jit
# @functools.partial(jax.vmap, in_axes=(None, 0, 0))
# def get_h_with_naive_lap(params, r, ind_neighbour):
#     return folx.forward_laplacian(lambda r_: model.apply(params, r_, ind_neighbour))(r)

# @functools.partial(jax.jit, static_argnums=(3,))
# @functools.partial(jax.vmap, in_axes=(None, 0, 0, None))
# def get_h_with_sparse_lap(params, r, ind_neighbour, max_n_dependencies):
#     return model.apply_with_fwd_lap(params, r, ind_neighbour, max_n_dependencies)

# h_with_sparse_lap = get_h_with_sparse_lap(params, r, ind_neighbour, max_n_dependencies)
# h_with_naive_lap = get_h_with_naive_lap(params, r, ind_neighbour)
# delta_h = jnp.linalg.norm(h_with_sparse_lap.x - h_with_naive_lap.x)
# delta_lap = jnp.linalg.norm(h_with_sparse_lap.laplacian - h_with_naive_lap.laplacian)
# print("Naive jacobian.data.shape: ", h_with_naive_lap.jacobian.data.shape)
# print("Sparse jacobian.data.shape: ", h_with_sparse_lap.jacobian.data.shape)
# print(f"delta_h: {delta_h:.1e} (rel: {delta_h / jnp.linalg.norm(h_with_naive_lap.x):.1e})")
# print(f"delta_lap: {delta_lap:.1e} (rel: {delta_lap / jnp.linalg.norm(h_with_naive_lap.laplacian):.1e})")
