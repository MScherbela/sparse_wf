#%%
import flax
import jax.numpy as jnp
import numpy as np
import jax
import matplotlib.pyplot as plt
from sparse_wf.model.graph_utils import NO_NEIGHBOUR


fname = "/home/mscherbela/tmp/optchkpt003000.msgpk"
with open(fname, "rb") as f:
    state = flax.serialization.msgpack_restore(f.read())
    electrons = state["electrons"]
batch_size, n_el, _ = electrons.shape


all_distances = jax.vmap(lambda r: jnp.linalg.norm(r[:, None] - r[None, :], axis=-1))(electrons)
n_cluster_radii = 10
quantiles = np.array([0, 0.05, 0.25, 0.5, 0.75, 0.95, 1])
cluster_radii = np.linspace(0.0, 3.0, n_cluster_radii)
n_in_cutoff = jnp.array([jnp.sum(all_distances <= rc, axis=-1) for rc in cluster_radii])
n_in_cutoff = n_in_cutoff.reshape((n_cluster_radii, batch_size * n_el))
n_in_cutoff_mean = jnp.mean(n_in_cutoff, axis=-1)
n_in_cutoff_percentiles = jnp.quantile(n_in_cutoff, quantiles, axis=-1)

plt.close("all")
fig, axes = plt.subplots(1, 2, figsize=(9, 5))
for idx_q, q in enumerate(quantiles):
    color = plt.cm.viridis(q)
    label = "min" if q == 0 else "max" if q == 1 else f"{q:.0%}"
    lw=2 if label in ["min", "max"] else 1.5
    axes[0].plot(cluster_radii, n_in_cutoff_percentiles[idx_q], label=label, color=color, lw=lw)
axes[0].plot(cluster_radii, n_in_cutoff_mean, label="mean", color='k', lw=3)
axes[0].set_ylabel("nr of electrons in_cutoff")
axes[0].legend()
axes[1].plot(cluster_radii, n_in_cutoff_percentiles[-1] / n_in_cutoff_mean, color='k')
axes[1].set_ylabel("max / mean")

for ax in axes:
    ax.grid(alpha=0.3, zorder=-1)
    ax.set_xlabel("cutoff radius")
fig.suptitle(f"WG_09, {n_el} electrons")
fig.tight_layout()
fig.savefig("/home/mscherbela/ucloud/results/electron_distribution.png", dpi=200)

