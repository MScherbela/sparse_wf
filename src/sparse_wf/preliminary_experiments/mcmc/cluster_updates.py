# %%
import jax.numpy as jnp
import functools
from sparse_wf.system import database
import jax
import os
import numpy as np
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = ""

NO_NEIGHBOUR = 1_000_000

rng = jax.random.PRNGKey(0)

batch_size = 2_000
eps = 2.0
mol = database("f096790bded54b5e7791aebf007cab0d")  # C6NH7
R = mol.atom_coords()
Z = mol.atom_charges()
n_el = int(Z.sum())
n_nuc = len(R)
r = jnp.concatenate([jnp.tile(R_, (Z_, 1)) for R_, Z_ in zip(R, Z)], axis=0)
r = r[None, :, :] + eps * jax.random.normal(rng, (batch_size, n_el, 3))


cluster_radius = 4
max_cluster_size = n_el
stepsize = 0.2

def log_p_include(dist):
    # return -(dist / cluster_radius) ** 2
    return -(dist / cluster_radius)
    # return jnp.where(dist < cluster_radius, 0.0, -100)

@jax.jit
@functools.partial(jax.vmap, in_axes=(0, 0, None, None))
def propose(rng, r, ind_center, stepsize):
    rng, rng_select, rng_move = jax.random.split(rng, 3)
    R_center = jnp.array(R)[ind_center]
    dist_before_move = jnp.linalg.norm(r - R_center, axis=-1)
    log_p_select1 = log_p_include(dist_before_move)

    do_move = log_p_select1 >= jnp.log(jax.random.uniform(rng_select, (n_el,)))
    cluster_size = jnp.sum(do_move)
    ind_select = jnp.nonzero(do_move, fill_value=NO_NEIGHBOUR, size=max_cluster_size)[0]
    dr = jax.random.normal(rng_move, (max_cluster_size, 3)) * stepsize
    r_proposed = r.at[ind_select].add(dr, mode="drop")

    dist_after_move = jnp.linalg.norm(r_proposed - R_center, axis=-1)
    log_p_select2 = log_p_include(dist_after_move)
    log_ratio = jnp.sum(log_p_select2 - log_p_select1)
    return rng, r_proposed, log_ratio, cluster_size

log_ratios, cluster_sizes = [], []
rngs = jax.random.split(rng, batch_size)
for i in range(n_nuc):
    rngs, _, log_ratio, cluster_size = propose(rngs, r, i, stepsize)
    log_ratios.append(log_ratio)
    cluster_sizes.append(cluster_size)

log_ratios = np.stack(log_ratios)
cluster_sizes = np.stack(cluster_sizes)


plt.close("all")
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle(f"C6NH7 ({n_el} electrons, {n_nuc} nuclei)\ncluster_radius={cluster_radius}, stepsize={stepsize}")
axes[0].hist(cluster_sizes.flatten(), bins=np.arange(0, max_cluster_size + 1) - 0.5)
axes[0].set_xlabel("Cluster size")
axes[1].hist(log_ratios.flatten(), bins=100)
axes[1].set_xlabel("Proposal log ratio")

# dist = np.linspace(0, 5, 100)
# plt.plot(dist, p_include(dist))

