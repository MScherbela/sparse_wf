#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("/home/scherbelam20/develop/sparse_wf/benchmark.csv")
df = df.groupby(["n_el", "cutoff", "batch_size"])[["t_logpsi", "t_embed", "t_energy", "n_deps_max"]].min().reset_index()
df["t_orb_jastrow_det"] = df["t_energy"] - df["t_embed"]
cutoff_values = sorted(df.cutoff.unique())

plt.close("all")

# Plot timings
fig, axes = plt.subplots(2, 2, figsize=(10, 6))
for timing_type, ax in zip(["t_logpsi", "t_embed", "t_orb_jastrow_det", "t_energy"], axes.flatten()):
    for ind_co, cutoff in enumerate(cutoff_values):
        df_c = df[df["cutoff"] == cutoff]
        ax.plot(df_c.n_el, df_c[timing_type], label=f"cutoff={cutoff:.1f}", color=f"C{ind_co}", marker='o')

# # (Scaling lines)
# powers = np.array([1, 2])
# offsets = [0.04,  0.03]
# xmin = [32, 126]
# for p, o, x0 in zip(powers, offsets, xmin):
#     x_plot = np.array([x0, 512])
#     ax_t.plot(x_plot, t_factor * o * (x_plot / 126)**p, label=f"$O(N^{{{p}}})$", linestyle="--", color="k", alpha=p/2 - 0.2)


# GPU_DAYS_PER_SEC = (50e3 * 2048 / batch_size_plot) / (3600 * 24)

for ax in axes.flatten():
    ax.set_xlabel("Nr of electrons")
    ax.set_ylabel("t / s")
    ax.set_yscale("log")
    ax.set_xscale("log")
    x_tick_values = df.n_el.unique()
    ax.set_xticks(x_tick_values, minor=False)
    ax.set_xticks([], minor=True)
    ax.xaxis.set_major_formatter("{x:.0f}")
    ax.legend()
    ax.grid(alpha=0.3)

# # Plot nr of neighbours
# df_max = df[df.n_el == df.n_el.max()]
# ax_neighbours.plot(df_max.cutoff, df_max.n_deps_max, label=f"Recep. field", color=f"k", marker='o', ls='-')
# ax_neighbours.plot(df_max.cutoff, df_max.n_nb, label=f"Direct neighbours", color=f"gray", marker='o', ls='--')
# ax_neighbours.set_xticks(df_max.cutoff, minor=False)
# ax_neighbours.set_xticks([], minor=True)
# ax_neighbours.set_xlabel("Cutoff")
# ax_neighbours.set_ylabel("Nr of neighbours")

fig.suptitle(f"Scaling for H-chains with spacing of 1.8\n16dets, 256 features")
fig.tight_layout()
# fig.savefig("/home/mscherbela/ucloud/results/moon_scaling.png", dpi=200)
fig.savefig("moon_scaling.png", dpi=200)




