# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sparse_wf.plot_utils import (
    get_outlier_mask,
    extrapolate_relative_energy,
    COLOR_PALETTE,
    scale_lightness,
    savefig,
)
import scienceplots

plt.style.use(["science", "grid"])


def draw_vertical_arrow(ax, x, E1, E2, color, text_right=True, text_padding=0):
    ax.annotate(
        "",
        xy=(x, E1),
        xytext=(x, E2),
        arrowprops=dict(arrowstyle="|-|", color=color, shrinkA=0, shrinkB=0, mutation_scale=5),
    )
    dE = np.abs(E1 - E2) * 1000
    text_y = min(E1, E2) + 0.6 * dE / 1000

    if not text_right:
        text_padding *= -1
    ax.text(
        x + text_padding,
        text_y,
        f"{dE:.1f}\nm$E_\\text{{h}}$",
        va="center",
        ha="left" if text_right else "right",
        color=color,
    )


smoothing = 500
eval_steps = 5000
df_all = pd.read_csv("interaction_energies.csv")
molecule = "11_Phenol_dimer"



df = df_all[(df_all["molecule"] == molecule) & (df_all["cutoff"] == 5)].copy().rename(columns={"opt/E": "E"})

df["var"] = df["opt/E_std"] ** 2
df = df.pivot_table(index="opt/step", columns="geom", values=["E", "var", "grad"], aggfunc="mean")
df["grad"] = df["grad"] ** 2
df = df.dropna()
is_outlier = df.apply(get_outlier_mask, axis=0).any(axis=1)
df = df[~is_outlier]
df_final = df.iloc[-eval_steps:].mean()
df = df.rolling(smoothing).mean().iloc[smoothing :: smoothing // 10]
df = df[df.index >= df.index.max() * 0.5]

geoms = ["dissociated", "equilibrium"]
color_final, color_ext = COLOR_PALETTE[:2][::-1]
fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True, width_ratios=[1, 1.15])
# sm = plt.cm.ScalarMappable(cmap=plt.get_cmap("viridis"), norm=plt.Normalize(0, df.index.max(), clip=True))
for ax, method in zip(axes.flat, ["var", "grad"]):
    x_values = [v for v in df[method].values.T]
    E_values = [v for v in df["E"].values.T]
    E_ext0_values, slopes = extrapolate_relative_energy(
        df.index, x_values, E_values, min_frac_step=0, return_slopes=True, method="same_slope"
    )
    x_range = np.array([df[method].values.min(), df[method].values.max()])
    E_final_values = df_final["E"].values
    E_ext_values = np.array(E_ext0_values) + np.array(x_range[0]) * np.array(slopes)

    label_final, label_ext = "final step", "extrapolated"
    for g, x, E, E_final, E_ext0, E_ext, slope in zip(
        geoms, x_values, E_values, E_final_values, E_ext0_values, E_ext_values, slopes
    ):
        x_final = df_final[(method, g)]
        line_scatter = ax.scatter(x, E, c=df.index / 1000, s=10)
        ax.plot(x_range, E_ext0 + slope * x_range, color="k")
        ax.plot([x_final], [E_final], ls="none", marker="s", color=color_final, label=label_final)
        ax.plot([x_range[0]], [E_ext], ls="none", marker="d", color=color_ext, label=label_ext)
        label_final, label_ext = None, None
    text_padding = 0.05 if (method == "grad") else 0.002
    draw_vertical_arrow(
        ax, df_final[method].max(), *E_final_values, scale_lightness(color_final, 0.5), True, text_padding
    )
    draw_vertical_arrow(ax, x_range[0], *E_ext_values, scale_lightness(color_ext, 0.5), True, text_padding)

    ax.yaxis.get_major_formatter().set_useOffset(False)
    ax.grid(False)
    ax.legend(loc="lower right")
axes[0].set_ylabel("energy [$E_\\text{h}$]")
axes[0].set_xlabel("energy variance [$E_\\text{h}^2$]")
axes[1].set_xlabel("$\\left|\\text{preconditioned gradient}\\right|^2$ / a.u.")

for ax, label in zip(axes, "ab"):
    ax.text(0.03, 0.93, f"\\textbf{{{label})}}", transform=ax.transAxes)

fig.colorbar(line_scatter, label="opt steps [k]")
fig.tight_layout()
savefig(fig, "energy_extrapolation")
