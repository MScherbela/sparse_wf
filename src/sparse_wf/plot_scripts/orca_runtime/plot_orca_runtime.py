# %%
import matplotlib.pyplot as plt
from sparse_wf.plot_utils import COLOR_PALETTE
import pandas as pd
import seaborn as sns
import numpy as np

sns.set_style("whitegrid")

N_EL_MIN_FOR_FIT = 28
BASIS_SET = "cc-pVTZ"


def fit_and_plot(ax, x, y, color, ls="-", n_fit_min=N_EL_MIN_FOR_FIT):
    include_in_fit = (x >= n_fit_min) & (np.isfinite(y))
    x_fit, y_fit = x[include_in_fit], y[include_in_fit]
    fit_coeffs = np.polyfit(np.log(x_fit), np.log(y_fit), 1)
    exponent = fit_coeffs[0]
    x_fit = np.array([20, 500])
    y_fitted = np.exp(np.polyval(fit_coeffs, np.log(x_fit)))
    if ax is not None:
        ax.plot(x_fit, y_fitted, color=color, ls=ls, lw=2)
    return exponent


def format_exponent(exp):
    return f"$O(n^{{{exp:.1f}}})$"


df = pd.read_csv("cumulene_orca.csv")
df = df.sort_values(["num_atoms", "num_basis_functions"])
df["num_valence_electrons"] = (df.num_atoms - 4) * 4 + 4

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
df_basis = df[(df.basis_set == BASIS_SET) & df.success]

color = "C0"
ax.scatter(df_basis.num_valence_electrons, df_basis.total_hours, color=color)
exp = fit_and_plot(ax, df_basis.num_valence_electrons, df_basis.total_hours, color=color)

label = f"CCSD(T)/{BASIS_SET}: {format_exponent(exp)}"
ax.loglog(
    df_basis.num_valence_electrons,
    df_basis.total_hours,
    color=color,
    marker="o",
    ls="none",
    label=label,
)


df_fire = pd.read_csv("../scaling/data/timings_4k.csv")


X_TICKS = [20, 30, 50, 100, 140, 200, 300, 500]
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xticks(X_TICKS)
ax.set_xticklabels([str(x) for x in X_TICKS])
ax.legend()


# %%

fig, axes = plt.subplots(1, 2, figsize=(8, 5))
BASIS_SETS = ["cc-pVDZ", "cc-pVTZ"]  # , "cc-pVQZ"]
X_TICKS = [20, 30, 50, 100, 140, 200, 300, 500]

ax = axes[0]
for idx_basis, basis_set in enumerate(BASIS_SETS):
    color = COLOR_PALETTE[idx_basis]
    df_basis = df[df.success & (df.basis_set == basis_set)]
    exp = fit_and_plot(
        ax,
        df_basis.num_valence_electrons,
        df_basis.total_hours,
        color,
    )
    label = f"{basis_set}: {format_exponent(exp)}"
    ax.loglog(
        df_basis.num_valence_electrons,
        df_basis.total_hours,
        color=color,
        marker="o",
        ls="none",
        label=label,
    )

ax.set_xticks(X_TICKS)
ax.set_xticklabels([str(x) for x in X_TICKS])
ax.legend()

ax = axes[1]
sns.scatterplot(
    df,
    x="num_basis_functions",
    # y="ccsd_step_seconds",
    y="total_hours",
    hue="basis_set",
    ax=ax,
)
ax.set_xscale("log")
ax.set_yscale("log")


# %%
