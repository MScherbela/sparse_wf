#%%
from sparse_wf.preliminary_experiments.sparse_mpnn.model import SparseWavefunction
import jax
import jax.numpy as jnp
import functools
import numpy as np
import matplotlib.pyplot as plt

def pad_n_neighbours(n, n_min=10, factor=1.2):
    power_padded = jnp.log(n) / jnp.log(factor)
    n_padded = jnp.maximum(n_min, factor ** jnp.ceil(power_padded))
    return n_padded.astype(jnp.int32)


def get_indices_in_receptive_field(r, cutoff, n_steps):
    @jax.jit
    def get_mask(r):
        dist = jnp.linalg.norm(r[..., :, None, :] - r[..., None, :, :], axis=-1)
        included = dist < (n_steps * cutoff)
        n_neighbours_max = jnp.max(jnp.sum(included, axis=-1))
        return included, n_neighbours_max
    
    @functools.partial(jax.jit, static_argnums=(1,))
    @functools.partial(jax.vmap, in_axes=(0, None))
    def get_indices(included, n_neighbours_max):
        indices = jnp.nonzero(included, size=n_neighbours_max, fill_value=-1)[0]
        weight = jnp.where(indices == -1, 0, 1)
        return indices, weight
    
    included, n_neighbours_max = get_mask(r)
    n_neighbours_max = int(pad_n_neighbours(n_neighbours_max))
    return get_indices(included, n_neighbours_max)


rng_r, rng_model = jax.random.split(jax.random.PRNGKey(0))
n_el = 100
cutoff = 5.0
n_steps = 2
R = jnp.arange(-n_el // 2, n_el // 2)[:, None] * jnp.array([1, 0, 0])
r = jax.random.normal(rng_r, (n_el, 3)) + R
ind_neighbour, weight_neighbour = get_indices_in_receptive_field(r, cutoff, 1)

model = SparseWavefunction(R, cutoff)
params = model.init(rng_model, r)

#%%
phi_dense = jax.block_until_ready(model.apply(params, r))
phi_sparse = jax.block_until_ready(model.apply(params, r, ind_neighbour, weight_neighbour))

rel_deviation = jnp.linalg.norm(phi_dense - phi_sparse) / jnp.linalg.norm(phi_dense)
print(f"{np.allclose(phi_dense, phi_sparse)=}; {rel_deviation=:.1e}")

plt.close("all")
plt.imshow(phi_dense, clim=np.quantile(np.abs(phi_dense), 0.95) * np.array([-1, 1]), cmap="bwr")


