# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sparse_wf.plot_utils import savefig, COLOR_PALETTE, COLOR_FIRE
import scienceplots
plt.style.use(["science", "grid"])

GROUND_TRUTH = "MRCI+Q"
ref_methods = ["FermiNet", "AFQMC", "VMC(AGP)"]
df = pd.read_csv("hchain_energies_aggregated.csv")
df_ref = pd.read_csv("hchain_references.csv")
E_ref = df_ref[df_ref.method == GROUND_TRUTH].groupby("dist").E.mean()
E_ref = E_ref.rename("E_ref")

pivot_ref = df_ref.pivot_table(index="method", columns="dist", values="E")
pivot_ref["deltaE"] = pivot_ref[2.8] - pivot_ref[1.8]
pivot_ref = (pivot_ref - pivot_ref.loc[GROUND_TRUTH]) * 1000

df = df.merge(E_ref, on="dist", how="left")
df["error"] = df["E_mean"] - df["E_ref"]
df_rel = df.pivot_table(index="cutoff", columns="dist", values=["error", "E_mean_sigma"], aggfunc="mean")
df_rel["rel_error"] = df_rel[("error", 2.8)] - df_rel[("error", 1.8)]
df_rel["rel_error_sigma"] = np.sqrt(df_rel[("E_mean_sigma", 2.8)] ** 2 + df_rel[("E_mean_sigma", 1.8)] ** 2)
df_rel = df_rel.reset_index()
#%%
fig, axes = plt.subplots(1, 3, figsize=(8, 3))
ax_abs, ax_rel, ax_std = axes.flatten()

# Absolute quantities
distances = df["dist"].unique()
markers = "os"
linestyles = ["-", "--"]
for dist, ls, marker in zip(distances, linestyles, markers):
    df_dist = df[df["dist"] == dist]
    label = f"FiRE $d={{{dist:.1f}}} a_0$"
    ax_abs.errorbar(
        df_dist.cutoff,
        df_dist.error * 1000,
        df_dist.E_mean_sigma * 1000,
        label=label,
        capsize=3,
        marker=marker,
        ls=ls,
        ms=4,
        color=COLOR_FIRE,
    )
    ax_std.plot(df_dist.cutoff, df_dist.E_std * 1000, label=None, ls=ls, marker=marker, ms=4, color=COLOR_FIRE)

# Relative quantities
ax_rel.errorbar(df_rel.cutoff, df_rel.rel_error * 1000, df_rel.rel_error_sigma * 1000, color=COLOR_FIRE, capsize=3, label=None)
ax_rel.axhline(0, color="k", linestyle="-")

for idx_method, method in enumerate(ref_methods):
    error18, error28, error_rel = pivot_ref.loc[method, [1.8, 2.8, "deltaE"]]
    color = COLOR_PALETTE[idx_method]
    ax_rel.axhline(error_rel , color=color, linestyle="-")
    ax_rel.text(0.3, error_rel + (-0.05 if "AFQMC" in method else 0.05), method, va="top" if "AFQMC" in method else "bottom", ha="left", color=color)
    ax_abs.axhline(error28, color=color, linestyle="--")
    ax_abs.axhline(error18, color=color, linestyle="-", label=method)

for ax, label in zip(axes.flatten(), "abc"):
    ax.set_xlabel("cutoff / $a_0$")
    # ax.legend()
    ax.text(0, 1.02, f"{label}", transform=ax.transAxes, va="bottom", ha="left", fontweight="bold", fontsize=12)
plt.figlegend(loc="upper center", ncol=5)

ax_abs.set_title("absolute error")
ax_rel.set_title("relative error")
ax_std.set_title("standard deviation")
ax_abs.set_ylabel(f"$E_\\mathrm{{FiRE}} - E_\\mathrm{{{GROUND_TRUTH}}}$ / mHa")
ax_rel.set_ylabel(f"$\Delta E_\\mathrm{{FiRE}} - \Delta E_\\mathrm{{{GROUND_TRUTH}}}$ / mHa")
ax_std.set_ylabel("$\sigma(E_\\mathrm{FiRE})$ / mHa")
fig.tight_layout()
fig.subplots_adjust(top=0.8)
savefig(fig, "hchain_energies")
