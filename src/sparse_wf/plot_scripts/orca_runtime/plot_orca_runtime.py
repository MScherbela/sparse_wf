# %%
import matplotlib.pyplot as plt
from sparse_wf.plot_utils import COLOR_PALETTE
import pandas as pd
import seaborn as sns
import numpy as np

sns.set_style("whitegrid")

N_EL_MIN_FOR_FIT = 20


def fit_and_plot(ax, x, y, color, ls="-", n_fit_min=N_EL_MIN_FOR_FIT):
    include_in_fit = (x >= n_fit_min) & (np.isfinite(y))
    x_fit, y_fit = x[include_in_fit], y[include_in_fit]
    fit_coeffs = np.polyfit(np.log(x_fit), np.log(y_fit), 1)
    exponent = fit_coeffs[0]
    y_fitted = np.exp(np.polyval(fit_coeffs, np.log(x_fit)))

    if ax is not None:
        ax.plot(x_fit, y_fitted, color=color, ls=ls, lw=2)
    return exponent


def format_exponent(exp):
    return f"$O(n^{{{exp:.1f}}})$"


df = pd.read_csv("cumulene_orca.csv")
df = df.sort_values(["num_atoms", "num_basis_functions"])
df["num_valence_electrons"] = (df.num_atoms - 4) * 4 + 4

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
BASIS_SETS = ["cc-pVDZ", "cc-pVTZ", "cc-pVQZ"]
X_TICKS = [100, 140, 200, 300, 500]

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

# %%
