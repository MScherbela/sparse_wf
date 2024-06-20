# %%
import jax.numpy as jnp
from sparse_wf.system import database
import jax
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

NO_NEIGHBOUR = 1_000_000

rng = jax.random.PRNGKey(0)

n_samples = 10_000
eps = 0.5
mol = database("f096790bded54b5e7791aebf007cab0d")  # C6NH7
R = mol.atom_coords()
Z = mol.atom_charges()
n_el = int(Z.sum())
r = jnp.concatenate([jnp.tile(R_, (Z_, 1)) for R_, Z_ in zip(R, Z)],   axis=0)
r = r[None, :, :] + eps * jax.random.normal(rng, (n_samples, n_el, 3))


cluster_radius = 1.0
max_cluster_size = 10


def p_include(dist):
    return jnp.exp(-((dist / cluster_radius) ** 2))


@jax.vmap
def propose(rng, r):
    n_el = r.shape[-2]
    key_center, key_neighbours, key_dr = jax.random.split(rng, 3)
    ind_center = jax.random.choice(key_center, n_el)
    distances = jnp.linalg.norm(r - r[ind_center], axis=-1)
    p_select = p_include(distances)
    u = jax.random.uniform(key_neighbours, (n_el,))
    ind_select = jnp.where(
        p_select >= u, fill_value=NO_NEIGHBOUR, size=max_cluster_size
    )[0]
    cluster_size = jnp.sum(ind_select != NO_NEIGHBOUR)
    delta_r = jax.random.normal(key_dr, [max_cluster_size, 3])
    r_proposed = r.at[ind_select].add(delta_r, mode="drop")

    distances_fwd = jnp.linalg.norm(r[ind_select] - r[None, :, :], axis=-1)
    distances_bwd = jnp.linalg.norm(
        r_proposed[ind_select] - r_proposed[None, :, :], axis=-1
    )
    q_fwd = p_include(distances_fwd)
    q_bwd = p_include(distances_bwd)

    return r, cluster_size


rngs = jax.random.split(rng, n_samples)
r_new, cluster_sizes = propose(rngs, r)
print(cluster_sizes.max())
