#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def fit_scaling(n_el, t):
    return np.polyfit(np.log(n_el), np.log(t), deg=1)

df = pd.read_csv("/home/mscherbela/tmp/benchmark.csv")
batch_size = df.batch_size.median()
df = df[df.batch_size == batch_size]
df = df.groupby(["n_el", "cutoff", "batch_size"])[["t_sampling", "t_embed", "t_energy", "n_deps_max"]].min().reset_index()
df["t_orb_jastrow_det"] = df["t_energy"] - df["t_embed"]
df["t_eval"] = df["t_energy"] + df["t_sampling"]

df["rel_t_sampling"] = df["t_sampling"] / df["t_eval"]
df["rel_t_embed"] = df["t_embed"] / df["t_eval"]
df["rel_t_orb_jastrow_det"] = df["t_orb_jastrow_det"] / df["t_eval"]
cutoff_values = sorted(df.cutoff.unique())
fitting_n_min = 40

plt.close("all")

fig = plt.figure(figsize=(12, 6))
gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 1.5])  # 2 rows, 3 columns
axes_timing = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
ax_relative = fig.add_subplot(gs[:, 2])

for timing_type, ax in zip(["t_sampling", "t_embed", "t_orb_jastrow_det", "t_energy"], axes_timing):
    for ind_co, cutoff in enumerate(cutoff_values):
        df_c = df[df["cutoff"] == cutoff]
        df_fit = df_c[df_c.n_el >= fitting_n_min]
        exponent, offset = fit_scaling(df_c.n_el, df_c[timing_type])

        n_el_fit = np.geomspace(fitting_n_min, df_c.n_el.max(), 100)
        t_fit = np.exp(offset) * n_el_fit**exponent
        ax.plot(df_c.n_el, df_c[timing_type], label=f"cutoff={cutoff:.1f}; O(n^{exponent:.1f})", color=f"C{ind_co}", marker='o')
        ax.plot(n_el_fit, t_fit, color=f"C{ind_co}", linestyle="--", alpha=0.5)
        ax.set_title(timing_type.replace("t_", ""))


ax_relative.stackplot(df.n_el, df["rel_t_sampling"], df["rel_t_embed"], df["rel_t_orb_jastrow_det"], labels=["sampling", "embed", "orb_jastrow_det"], colors=["C0", "C1", "C2"])


def format_x_axis(ax):
    ax.set_xlabel("Nr of electrons")
    ax.set_xscale("log")
    x_tick_values = df.n_el.unique()
    ax.set_xticks(x_tick_values, minor=False)
    ax.set_xticks([], minor=True)
    ax.xaxis.set_major_formatter("{x:.0f}")


for ax in axes_timing:
    format_x_axis(ax)
    ax.set_ylabel("t / s")
    ax.set_yscale("log")
    ax.set_yticks([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0], minor=False)
    ax.set_yticks([], minor=True)
    ax.yaxis.set_major_formatter("{x:.2f}")
    ax.legend()
    ax.grid(alpha=0.3)


ax_relative.set_xscale("log")
ax_relative.set_xlabel("Nr of electrons")
ax_relative.set_ylabel("Relative time")
ax_relative.legend()
ax_relative.set_title("Relative timings for evaluation")
format_x_axis(ax_relative)


fig.suptitle(f"Scaling for H-chains with spacing of 1.8\n{batch_size:.0f} batch size, 16 dets, 256 features")
fig.tight_layout()
fig.savefig("/home/mscherbela/ucloud/results/moon_scaling.png", dpi=200)




