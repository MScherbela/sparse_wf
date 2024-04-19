#%%
import os
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
import folx
import jax.numpy as jnp
import jax
from sparse_wf.jax_utils import fwd_lap
from jaxtyping import Float, Array

jax.config.update("jax_default_matmul_precision", "float32")
jax.config.update("jax_enable_x64", True)

def to_zero_padded(jac: Float[Array, "3 el orb"]) -> Float[Array, "3*el el orb"]:
    n_el = jac.shape[-2]
    jac_out = jnp.zeros([n_el, 3, n_el, n_el])
    i = jnp.arange(n_el)
    jac_out = jac_out.at[i, :, i, :].set(jnp.swapaxes(jac, 0, 1))
    jac_out = jac_out.reshape([3 * n_el, n_el, n_el])
    return jac_out

def to_sparse(jac: Float[Array, "3*el el orb"]) -> Float[Array, "3 el orb"]:
    n_el = jac.shape[-2]
    jac = jac.reshape([n_el, 3, n_el, n_el])
    ind_el = jnp.arange(n_el)
    jac_out = jac[ind_el, :, ind_el, :]
    jac_out = jnp.swapaxes(jac_out, 0, 1)
    return jac_out

def print_delta(x1, x2, name):
    assert x1.shape == x2.shape
    delta = jnp.linalg.norm(x1 - x2)
    delta_rel = delta / jnp.linalg.norm(x2)
    name += f" {x1.shape}"
    print(f"{name:<30}: Delta abs: {delta:4.1e}, Delta rel: {delta_rel:4.1e}")


dtype = jnp.float64
jax.config.update("jax_default_matmul_precision", "float32")
jax.config.update("jax_enable_x64", dtype is jnp.float64)


rng_params1, rng_params2, rng_el = jax.random.split(jax.random.PRNGKey(0), 3)
n_el = 10

def logdet(A):
    return jnp.linalg.slogdet(A).logabsdet

def orbitals(params, r):
    return jax.nn.silu(r @ params[0]) @ params[1]

def get_logpsi(params, r):
    return logdet(orbitals(params, r))

params = [jax.random.normal(rng_params1, (3, n_el), dtype),
          jax.random.normal(rng_params2, (n_el, n_el), dtype)]

electrons = jax.random.normal(rng_el, (n_el, 3), dtype)

# Compute and validate orbitals
orbitals_ext = fwd_lap(orbitals, sparsity_threshold=0, argnums=1)(params, electrons)
orbitals_int = jax.vmap(fwd_lap(lambda r: orbitals(params, r), sparsity_threshold=0), in_axes=0, out_axes=-2)(electrons)

jac_orbitals_ext_dense = orbitals_ext.jacobian.data
jac_orbitals_int_sparse = orbitals_int.jacobian.data
jac_orbitals_ext_sparse = to_sparse(jac_orbitals_ext_dense)
jac_orbitals_int_dense = to_zero_padded(orbitals_int.jacobian.data)

print_delta(jac_orbitals_int_dense, jac_orbitals_ext_dense, "Jac Orbitals (dense)")
print_delta(jac_orbitals_int_sparse, jac_orbitals_ext_sparse, "Jac Orbitals (sparse)")

# Compute full wavefunction
logpsi_ext = fwd_lap(get_logpsi, sparsity_threshold=0, argnums=1)(params, electrons)
logpsi_2step = fwd_lap(logdet, sparsity_threshold=0)(orbitals_ext)

A_inv = jnp.linalg.inv(orbitals_ext.x)
jac_logpsi_manual_dense = jnp.einsum("kij,ji", jac_orbitals_ext_dense, A_inv)
jac_logpsi_manual_sparse = jnp.einsum("dnj,jn->nd", jac_orbitals_int_sparse, A_inv).reshape([3*n_el])

print_delta(logpsi_ext.jacobian.data, logpsi_2step.jacobian.data, "Jac LogPsi (full vs 2-step)")
print_delta(logpsi_ext.jacobian.data, jac_logpsi_manual_dense, "Jac LogPsi (full vs manual dense)")
print_delta(logpsi_ext.jacobian.data, jac_logpsi_manual_sparse, "Jac LogPsi (full vs manual sparse)")

