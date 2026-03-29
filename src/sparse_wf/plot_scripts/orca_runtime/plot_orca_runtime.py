# %%
import matplotlib.pyplot as plt
from sparse_wf.plot_utils import COLOR_PALETTE, COLOR_FIRE, savefig
import pandas as pd
import seaborn as sns
import numpy as np

sns.set_style("whitegrid")

N_EL_MIN_FOR_FIT = 40
BASIS_SET = "cc-pVTZ"
FIRE_BATCH_SIZE = 512  # 1 node with 8 GPUs for 4096 total batch size
NUM_STEPS_BASE_CASE = 100e3
NUM_ELEC_BASE_CASE = 68
NUM_STEPS_SCALING_EXPONENT = 2.3


def fit_and_plot(ax, x, y, color, ls="-", n_fit_min=N_EL_MIN_FOR_FIT):
    include_in_fit = (x >= n_fit_min) & (np.isfinite(y))
    x_fit, y_fit = x[include_in_fit], y[include_in_fit]
    fit_coeffs = np.polyfit(np.log(x_fit), np.log(y_fit), 1)
    exponent = fit_coeffs[0]
    x_fit = np.array([min(x_fit), 500])
    y_fitted = np.exp(np.polyval(fit_coeffs, np.log(x_fit)))
    if ax is not None:
        ax.plot(x_fit, y_fitted, color=color, ls=ls, lw=2)
    return exponent


def format_exponent(exp):
    return f"$O(n^{{{exp:.1f}}})$"


# CCSD(T)
df_ccsdt = pd.read_csv("cumulene_orca.csv")
df_ccsdt = df_ccsdt.sort_values(["num_atoms", "num_basis_functions"])
df_ccsdt["num_valence_electrons"] = (df_ccsdt.num_atoms - 4) * 4 + 4


# FiRE
df_fire = pd.read_csv("../scaling/data/full_run_data.csv")
df_fire = df_fire[df_fire.model == "FiRE"]
df_fire = df_fire[~df_fire["opt/t_step"].isnull()]
df_fire["t_step"] = df_fire["opt/t_step"] * FIRE_BATCH_SIZE / df_fire.batch_size
df_fire = df_fire.groupby(["n_carbon", "cutoff"])["t_step"].min().reset_index()
df_fire["num_valence_electrons"] = df_fire.n_carbon * 4 + 4
df_fire["num_steps"] = (
    NUM_STEPS_BASE_CASE * (df_fire.num_valence_electrons / NUM_ELEC_BASE_CASE) ** NUM_STEPS_SCALING_EXPONENT
)
df_fire["total_hours"] = df_fire.t_step * df_fire.num_steps / 3600


fig, ax = plt.subplots(1, 1, figsize=(6, 4))
df_basis = df_ccsdt[(df_ccsdt.basis_set == BASIS_SET) & df_ccsdt.success]

color = COLOR_PALETTE[0]
exp = fit_and_plot(ax, df_basis.num_valence_electrons, df_basis.total_hours, color=color)

label = f"CCSD(T) ({BASIS_SET}): {format_exponent(exp)}"
ax.loglog(df_basis.num_valence_electrons, df_basis.total_hours, color=color, marker="o", ls="none", label=label)


cutoff = 3.0
df_cutoff = df_fire[df_fire.cutoff == cutoff]
color = COLOR_FIRE
ax.scatter(df_cutoff.num_valence_electrons, df_cutoff.total_hours, color=color)
exp = fit_and_plot(ax, df_cutoff.num_valence_electrons, df_cutoff.total_hours, color=color)

label = f"FiRE (c={cutoff:.1f}$a_0$): {format_exponent(exp)}"
ax.loglog(
    df_cutoff.num_valence_electrons,
    df_cutoff.total_hours,
    color=color,
    marker="s",
    ls="none",
    label=label,
)

X_TICKS = [20, 30, 50, 100, 140, 200, 300, 500]
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xticks(X_TICKS)
ax.set_xticklabels([str(x) for x in X_TICKS])
ax.legend()
ax.set_xlabel("Valence electrons")
ax.set_ylabel("Total node hours")
savefig(fig, "ccsdt_vs_fire")


# %%
