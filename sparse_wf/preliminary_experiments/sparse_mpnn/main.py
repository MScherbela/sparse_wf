# %%
from sparse_wf.api import Electrons, StaticInput
from sparse_wf.model import SparseMoonWavefunction
import jax
import jax.numpy as jnp
import folx
import functools

from jax import config as jax_config

jax_config.update("jax_enable_x64", False)
jax_config.update("jax_default_matmul_precision", "highest")


def build_atom_chain(rng, n_nuc, n_el_per_nuc, batch_size):
    R = jnp.arange(n_nuc)[:, None] * jnp.array([1, 0, 0])
    r = R[:, None, :] + jax.random.normal(rng, [batch_size, n_nuc, n_el_per_nuc, 3])
    r = jax.lax.collapse(r, 1, 3)
    n_el = r.shape[-2]
    spin = jnp.tile(jnp.arange(n_el) % 2, (batch_size, 1))
    Z = jnp.ones(n_nuc, dtype=int) * n_el_per_nuc
    return Electrons(r, spin), R, Z


cutoff = 4.0
rng_r, rng_model = jax.random.split(jax.random.PRNGKey(0))
electrons, R, Z = build_atom_chain(rng_r, n_nuc=25, n_el_per_nuc=2, batch_size=1)

model = SparseMoonWavefunction(
    n_orbitals=electrons.n_el,
    R=R,
    Z=Z,
    cutoff=cutoff,
    feature_dim=64,
    nuc_mlp_depth=3,
    pair_mlp_widths=(16, 8),
    pair_n_envelopes=16,
)
params = model.init(rng_model)
static_args = model.input_constructor.get_static_input(electrons)


@functools.partial(jax.jit, static_argnums=(2,))
@functools.partial(jax.vmap, in_axes=(None, 0, None))
def apply_with_external_fwd_lap(params, electrons: Electrons, static_args: StaticInput):
    return folx.forward_laplacian(lambda r: model.orbitals(params, Electrons(r, electrons.spins), static_args))(
        electrons.r
    )


@functools.partial(jax.jit, static_argnums=(2,))
@functools.partial(jax.vmap, in_axes=(None, 0, None))
def apply_with_internal_fwd_lap(params, electrons: Electrons, static_args: StaticInput):
    return model.orbitals_with_fwd_lap(params, electrons, static_args)


h_with_external_lap = apply_with_external_fwd_lap(params, electrons, static_args)
h_with_internal_lap = apply_with_internal_fwd_lap(params, electrons, static_args)
delta_h = jnp.linalg.norm(h_with_internal_lap.x - h_with_external_lap.x)
delta_lap = jnp.linalg.norm(h_with_internal_lap.laplacian - h_with_external_lap.laplacian)
print("Internal lap jacobian.data.shape: ", h_with_internal_lap.jacobian.data.shape)
print("External lap jacobian.data.shape: ", h_with_external_lap.jacobian.data.shape)
print(f"delta_h: {delta_h:.1e} (rel: {delta_h / jnp.linalg.norm(h_with_external_lap.x):.1e})")
print(f"delta_lap: {delta_lap:.1e} (rel: {delta_lap / jnp.linalg.norm(h_with_external_lap.laplacian):.1e})")
