# %%
from sparse_wf.preliminary_experiments.sparse_mpnn.model import SparseMoonWavefunction
from sparse_wf.preliminary_experiments.sparse_mpnn.get_neighbours import (
    get_all_neighbour_indices,
    get_all_dependencies,
    get_max_nr_of_dependencies,
)
import jax
import jax.numpy as jnp
import folx
import functools

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


cutoff = 2.0
rng_r, rng_model = jax.random.split(jax.random.PRNGKey(0))
r, spin, R = build_atom_chain(rng_r, n_nuc=25, n_el_per_nuc=2, batch_size=1)
n_el = r.shape[-2]

model = SparseMoonWavefunction(
    n_orbitals=n_el, cutoff=cutoff, feature_dim=64, pair_mlp_widths=(16, 8), pair_n_envelopes=16
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


h_with_internal_lap = apply_with_internal_fwd_lap(params, r, spin, R, idx_nb, n_deps_max, dep_maps)
h_with_external_lap = apply_with_external_fwd_lap(params, r, spin, R, idx_nb)
delta_h = jnp.linalg.norm(h_with_internal_lap.x - h_with_external_lap.x)
delta_lap = jnp.linalg.norm(h_with_internal_lap.laplacian - h_with_external_lap.laplacian)
print("Internal lap jacobian.data.shape: ", h_with_internal_lap.jacobian.data.shape)
print("External lap jacobian.data.shape: ", h_with_external_lap.jacobian.data.shape)
print(f"delta_h: {delta_h:.1e} (rel: {delta_h / jnp.linalg.norm(h_with_external_lap.x):.1e})")
print(f"delta_lap: {delta_lap:.1e} (rel: {delta_lap / jnp.linalg.norm(h_with_external_lap.laplacian):.1e})")
