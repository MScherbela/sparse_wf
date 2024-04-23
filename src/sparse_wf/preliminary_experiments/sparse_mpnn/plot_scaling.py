#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

fname = "timings_batchsize_8_new.txt"
data = []
with open(fname, "r") as f:
    for line in f:
        if not line.startswith("Summary:"):
            continue
        line = line.strip().replace("Summary: ", "").replace("sparse time", "t")
        data.append(eval("dict(" + line + ")"))
df = pd.DataFrame(data)
df = df.sort_values(["n_el", "cutoff"])#

cutoff_values = df["cutoff"].unique()

batch_size_experiment = 8
batch_size_plot = 128
t_factor = batch_size_plot / batch_size_experiment


plt.close("all")

# Plot timings
fig, axes = plt.subplots(1, 2, figsize=(10, 6), width_ratios=[2,1])
ax_t, ax_neighbours = axes
for ind_co, cutoff in enumerate(cutoff_values):
    df_c = df[df["cutoff"] == cutoff]
    ax_t.plot(df_c.n_el, t_factor * df_c.t, label=f"cutoff={cutoff:.1f}", color=f"C{ind_co}", marker='o')

# (Scaling lines)
powers = np.array([1, 2])
offsets = [0.04,  0.03]
xmin = [32, 126]
for p, o, x0 in zip(powers, offsets, xmin):
    x_plot = np.array([x0, 512])
    ax_t.plot(x_plot, t_factor * o * (x_plot / 126)**p, label=f"$O(N^{{{p}}})$", linestyle="--", color="k", alpha=p/2 - 0.2)


GPU_DAYS_PER_SEC = (50e3 * 2048 / batch_size_plot) / (3600 * 24)

ax_t.set_xlabel("Nr of electrons")
ax_t.set_ylabel("Time for local energy (s)")
ax_t.set_yscale("log")
ax_t.set_xscale("log")
secax = ax_t.secondary_yaxis("right", functions=(lambda x: x * GPU_DAYS_PER_SEC, lambda x: x / GPU_DAYS_PER_SEC))
secax.set_ylabel("GPU-days for 50k energy steps (batchsize 2048)")
x_tick_values = df.n_el.unique()
ax_t.set_xticks(x_tick_values, minor=False)
ax_t.set_xticks([], minor=True)
ax_t.xaxis.set_major_formatter("{x:.0f}")

# Plot nr of neighbours
df_max = df[df.n_el == df.n_el.max()]
ax_neighbours.plot(df_max.cutoff, df_max.n_deps_max, label=f"Recep. field", color=f"k", marker='o', ls='-')
ax_neighbours.plot(df_max.cutoff, df_max.n_nb, label=f"Direct neighbours", color=f"gray", marker='o', ls='--')
ax_neighbours.set_xticks(df_max.cutoff, minor=False)
ax_neighbours.set_xticks([], minor=True)
ax_neighbours.set_xlabel("Cutoff")
ax_neighbours.set_ylabel("Nr of neighbours")

for ax in axes:
    ax.legend()
    ax.grid(alpha=0.3)
fig.suptitle(f"Scaling for H-chains with spacing of 1.0\nBatch size {batch_size_plot}, 4dets, 256 features")
fig.tight_layout()
fig.savefig("/home/mscherbela/ucloud/results/sparse_wf_scaling.png", dpi=200)



