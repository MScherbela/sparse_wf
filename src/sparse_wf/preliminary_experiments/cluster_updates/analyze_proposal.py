#%%
import flax
import jax.numpy as jnp
import numpy as np
import jax
import matplotlib.pyplot as plt
import functools
import jax.tree_util as jtu
from sparse_wf.model.graph_utils import NO_NEIGHBOUR
import pandas as pd
import seaborn as sns


fname = "/home/mscherbela/tmp/optchkpt003000.msgpk"
with open(fname, "rb") as f:
    state = flax.serialization.msgpack_restore(f.read())
    electrons = state["electrons"]
batch_size, n_el, _ = electrons.shape

def get_closest_k(dist, dist_max, k):
    neg_d, idx = jax.lax.top_k(-dist, k=k)
    idx = jnp.where(neg_d >= -dist_max, idx, NO_NEIGHBOUR)
    return idx

def is_same_set(idx_a, idx_b):
    return jnp.all(jnp.sort(idx_a) == jnp.sort(idx_b))

def get_n_affected(r_old, r_new, idx_changed, cutoff):
    dist_old = jnp.linalg.norm(r_old.at[idx_changed, None].get(mode="fill", fill_value=1e6) - r_old[None, :], axis=-1)
    dist_new = jnp.linalg.norm(r_new.at[idx_changed, None].get(mode="fill", fill_value=1e6) - r_new[None, :], axis=-1)
    dist_closest = jnp.min(jnp.concatenate([dist_old, dist_new], axis=0), axis=0)
    return jnp.sum(dist_closest <= cutoff)



def propose(key, r, idx_center, cutoff, stepsize, max_cluster_radius, max_cluster_size):
    dist_old = jnp.linalg.norm(r - r[idx_center], axis=-1)
    idx_closest = get_closest_k(dist_old, max_cluster_radius, max_cluster_size)

    def _proposal(rng):
        dr = jax.random.normal(rng, (max_cluster_size, 3)) * stepsize
        r_new = r.at[idx_closest].add(dr)
        dist_new = jnp.linalg.norm(r_new - r_new[idx_center], axis=-1)
        idx_closest_new = get_closest_k(dist_new, max_cluster_radius, max_cluster_size)
        is_valid = is_same_set(idx_closest, idx_closest_new)
        return r_new, is_valid

    n_trials = 5
    r_new, is_valid = jax.vmap(_proposal)(jax.random.split(key, n_trials))
    idx_trial = jnp.argmax(is_valid)
    r_new = r_new[idx_trial]
    is_valid = is_valid[idx_trial]

    actual_cluster_size = jnp.sum(idx_closest != NO_NEIGHBOUR)
    n_affected = get_n_affected(r, r_new, idx_closest, cutoff)
    return is_valid, actual_cluster_size, n_affected

# max_cluster_radii = np.array([1.0, 1.5, 2.0, 5.0, 10.0])
max_cluster_radii = [1.0, 1.5, 2.0]
stepsizes = np.geomspace(0.02, 0.2, 6)
max_cluster_sizes = [1, 5, 10]
cutoff = 4.0

key = jax.random.PRNGKey(0)
keys = jax.random.split(key, batch_size * n_el).reshape((batch_size, n_el, 2))

all_data = []
for max_cluster_size in max_cluster_sizes:
    prop_fn = functools.partial(propose, max_cluster_size=max_cluster_size)
    prop_fn = jax.vmap(prop_fn, in_axes=(0, None, 0, None, None, None)) # vmap over idx_center
    prop_fn = jax.vmap(prop_fn, in_axes=(0, 0, None, None, None, None)) # vmap over batch_size
    prop_fn = jax.jit(prop_fn)
    for i, stepsize in enumerate(stepsizes):
        for max_cluster_radius in max_cluster_radii:
            print(f"max_cluster_size={max_cluster_size}, stepsize={stepsize:.3f}, max_cluster_radius={max_cluster_radius:.2f}")

            data = prop_fn(keys, electrons, jnp.arange(n_el), cutoff, stepsize, max_cluster_radius)
            is_valid_mean, cluster_size_mean, n_affected_mean = jtu.tree_map(jnp.mean, data)
            n_affected_max = jnp.max(data[2])

            all_data.append(dict(
                max_cluster_size=max_cluster_size,
                max_cluster_radius=max_cluster_radius,
                is_valid_mean=float(is_valid_mean),
                cluster_size_mean=float(cluster_size_mean),
                n_affected_mean=float(n_affected_mean),
                n_affected_max=float(n_affected_max),
                stepsize=stepsize,
            ))
df = pd.DataFrame(all_data)
#%%
plt.close("all")
sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
sns.lineplot(data=df[df.max_cluster_radius < 5], x="stepsize", y="is_valid_mean", style="max_cluster_radius", hue="max_cluster_size", ax=axes[0,0], palette="tab10", markers=["o", "^", "v", "s", "D"], dashes=False)
sns.lineplot(data=df, x="max_cluster_radius", y="cluster_size_mean", hue="max_cluster_size", ax=axes[0,1], palette="tab10", marker="o")
sns.lineplot(data=df, x="max_cluster_radius", y="n_affected_mean", hue="max_cluster_size", ax=axes[1,0], palette="tab10", marker="o")
sns.lineplot(data=df, x="max_cluster_radius", y="n_affected_max", hue="max_cluster_size", ax=axes[1,1], palette="tab10", marker="o")
for ax in axes.flat:
    ax.set_ylim([0, None])
fig.suptitle(f"WG_09, {n_el} electrons")
fig.tight_layout()
fig.savefig("/home/mscherbela/ucloud/results/cluster_proposals_WG09_5trials.png", dpi=200)

# axes[0].set_title("Ratio of valid proposals")
# axes[1].set_title("Mean cluster size")



