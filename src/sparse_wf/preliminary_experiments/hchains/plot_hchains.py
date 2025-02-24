#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("hchain_energies_aggregated.csv")
df_ref = pd.read_csv("hchain_references.csv")
E_ref = df_ref[df_ref.method == "AFQMC"].groupby("dist").E.mean()
E_ref = E_ref.rename("E_ref")

df = df.merge(E_ref, on="dist", how="left")
df["error"] = df["E_mean"] - df["E_ref"]


df_rel = df.pivot_table(index="cutoff", columns="dist", values=["error", "E_mean_sigma"], aggfunc="mean")
df_rel["rel_error"] = df_rel[("error", 2.8)] - df_rel[("error", 1.8)]
df_rel["rel_error_sigma"] = np.sqrt(df_rel[("E_mean_sigma", 2.8)] ** 2 + df_rel[("E_mean_sigma", 1.8)] ** 2)
df_rel = df_rel.reset_index()

fig, axes = plt.subplots(1, 3, figsize=(10, 3))
ax_abs, ax_rel, ax_std = axes.flatten()
# ax_abs, ax_std, ax_rel, ax_t = axes.flatten()

# Absolute quantities
distances = df["dist"].unique()
for dist in distances:
    df_dist = df[df["dist"] == dist]
    ax_abs.errorbar(df_dist.cutoff, df_dist.error * 1000, df_dist.E_mean_sigma * 1000, label=f"$d={{{dist:.1f}}} a_0$", capsize=3)
    ax_std.plot(df_dist.cutoff, df_dist.E_std * 1000, label=f"$d={{{dist:.1f}}} a_0$", marker="o")
    # ax_t.plot(df_dist.cutoff, df_dist.t, label=f"$d={{{dist:.1f}}} a_0$")

# Relative quantities
ax_rel.errorbar(df_rel.cutoff, df_rel.rel_error * 1000, df_rel.rel_error_sigma * 1000, color="k", capsize=3)
ax_rel.axhline(0, color="k", linestyle="--")

for ax, label in zip(axes.flatten(), "abc"):
    ax.set_xlabel("cutoff / $a_0$")
    if ax != ax_rel:
        ax.legend()
    ax.grid(alpha=0.5)
    ax.text(0, 1.02, f"{label}", transform=ax.transAxes, va="bottom", ha="left", fontweight="bold", fontsize=12)

ax_abs.set_title("absolute error")
ax_rel.set_title("relative error")
ax_std.set_title("standard deviation")
ax_abs.set_ylabel("$E_\\mathrm{FiRE} - E_\\mathrm{AFQMC}$ / mHa")
ax_rel.set_ylabel("$\Delta E_\\mathrm{FiRE} - \Delta E_\\mathrm{AFQMC}$ / mHa")
ax_std.set_ylabel("$\sigma(E_\\mathrm{FiRE})$ / mHa")
# ax_t.set_ylabel("$t$ / s")
fig.tight_layout()
fig.savefig("hchain_energies.pdf", bbox_inches="tight")



