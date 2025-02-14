# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ast import literal_eval
from scipy.optimize import curve_fit
import matplotlib.ticker as ticker

N_SWEEPS = 1
REFERENCE_BATCH_SIZE = 32
N_EL_MIN_FOR_FIT = 100
N_EL_FOR_BREAKDOWN = 260


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


# Load data
def read_data(fname):
    with open(fname, "r") as f:
        return pd.DataFrame([literal_eval(l.strip()) for l in f.readlines()])


data_files = {
    "SWANN": ["data/timings_with_ecp.txt", "data/timings_with_ecp_dense.txt"],
    "lapnet": ["data/timings_lapnet.txt"],
    "psiformer": ["data/timings_psiformer.txt"],
    "ferminet": ["data/timings_ferminet.txt"],
}

df = pd.DataFrame()
for model, data_fnames in data_files.items():
    for fname in data_fnames:
        df_file = read_data(fname)
        df_file["model"] = model
        df = pd.concat([df, df_file], ignore_index=True)
df["cutoff"] = df["cutoff"].fillna(0)
df["model"] = df["model"].apply(lambda s: s[0].capitalize() + s[1:])
df = df[((df.model == "SWANN") & (df.cutoff == 3)) | (df.model != "SWANN")]


# Assemble data and compute total times
n_ecp_evals = df.groupby(["n_el"])["pp_static/n_pp_elecs"].mean()
pivot = (
    df.groupby(["model", "cutoff", "n_el", "batch_size"])[
        ["t_wf_full", "t_wf_upd_local", "t_wf_upd_swap", "t_E_kin", "t_E_kin_sparse", "t_E_pot"]
    ]
    .mean()
    .reset_index()
)
for column in list(pivot):
    if column.startswith("t_"):
        pivot[column] = pivot[column] * REFERENCE_BATCH_SIZE / pivot.batch_size
pivot["n_ECP_evals"] = pivot.n_el.map(n_ecp_evals)
is_dense = pivot["model"] != "SWANN"
pivot["t_E_pot"] = pivot.t_E_pot.where(~is_dense, pivot.t_wf_full * pivot.n_ECP_evals)
pivot["t_Spin"] = pivot.t_wf_full.where(is_dense, pivot.t_wf_upd_swap) * pivot.n_el / 2
pivot["t_E_kin_best"] = pivot.t_E_kin.where(is_dense, pivot.t_E_kin_sparse)
pivot["t_sampling"] = pivot.t_wf_full.where(is_dense, pivot.t_wf_upd_local) * pivot.n_el * N_SWEEPS
pivot["t_total"] = pivot.t_sampling + pivot.t_E_kin_best + pivot.t_E_pot + pivot.t_Spin


plt.close("all")
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
ax_upd, ax_Ekin, ax_tot, ax_speedup = axes.flatten()

models = ["SWANN", "Lapnet", "Psiformer", "Ferminet"]
model_colors = ["red", "C0", "C1", "C2"]
for model, color in zip(models, model_colors):
    kwargs_filled = dict(marker="o", ls="none", color=color)
    kwargs_empty = dict(marker="o", ls="none", color=color, fillstyle="none")
    df_model = pivot[pivot.model == model]
    full_batch = df_model.batch_size == REFERENCE_BATCH_SIZE

    # Plot update times
    exponent = fit_and_plot(ax_upd, df_model.n_el, df_model.t_wf_full, color)
    label = model + " (dense)" if model == "SWANN" else model
    label += ": " + format_exponent(exponent)
    # ax_upd.plot(df_model.n_el, df_model.t_wf_full, color=color, marker="none", lw=1)
    ax_upd.plot(df_model[full_batch].n_el, df_model[full_batch].t_wf_full, label=label, **kwargs_filled)
    ax_upd.plot(df_model[~full_batch].n_el, df_model[~full_batch].t_wf_full, **kwargs_empty)
    if model == "SWANN":
        exponent = fit_and_plot(ax_upd, df_model.n_el, df_model.t_wf_upd_local, color)
        label = f"{model}: {format_exponent(exponent)}"
        ax_upd.plot(df_model.n_el, df_model.t_wf_upd_local, label=label, color=color, marker="s", ls="none")
    ax_upd.set_title("Wavefunction update")

    # Plot kinetic energy times
    exponent = fit_and_plot(ax_Ekin, df_model.n_el, df_model.t_E_kin, color)
    label = model + " (dense)" if model == "SWANN" else model
    label += ": " + format_exponent(exponent)
    # ax_Ekin.plot(df_model.n_el, df_model.t_E_kin, color=color, marker="none", lw=1)
    ax_Ekin.plot(df_model[full_batch].n_el, df_model[full_batch].t_E_kin, label=label, **kwargs_filled)
    ax_Ekin.plot(df_model[~full_batch].n_el, df_model[~full_batch].t_E_kin, **kwargs_empty)
    if model == "SWANN":
        exponent = fit_and_plot(ax_Ekin, df_model.n_el, df_model.t_E_kin_sparse, color)
        label = f"{model}: {format_exponent(exponent)}"
        ax_Ekin.plot(df_model.n_el, df_model.t_E_kin_sparse, label=label, color=color, marker="s", ls="none")
    ax_Ekin.set_title("Kinetic energy")

    # Plot total time
    exponent = fit_and_plot(ax_tot, df_model.n_el, df_model.t_total, color)
    label = model + ": " + format_exponent(exponent)
    # ax_tot.plot(df_model.n_el, df_model.t_total, color=color, marker="none", lw=1)
    marker = "s" if model == "SWANN" else "o"
    ax_tot.plot(df_model[full_batch].n_el, df_model[full_batch].t_total, label=label, **kwargs_filled)
    ax_tot.plot(df_model[~full_batch].n_el, df_model[~full_batch].t_total, **(kwargs_empty | dict(marker=marker)))
    ax_tot.set_title("Total optimization step")

for ax in [ax_upd, ax_Ekin, ax_tot]:
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("Number of electrons")
    ax.set_ylabel("Time / s")
    ax.legend(loc="upper left", labelspacing=0.2, frameon=False)
    ax.set_xticks([70, 100, 140, 200, 300, 500])
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.minorticks_off()
    # ax.axvline(N_EL_FOR_BREAKDOWN, color="dimgray", ls="-", zorder=-1)
    # ax.grid(alpha=0.5)
ax_Ekin.set_ylim([None, 3e2])

for ax, label in zip(axes.flatten(), "abcd"):
    ax.text(0, 1.02, f"{label})", transform=ax.transAxes, va="bottom", ha="left", fontweight="bold", fontsize=12)

# Speedup bar-chart
df_speedup = pivot[pivot.n_el == N_EL_FOR_BREAKDOWN].set_index("model").reindex(models)
t_total_swann = df_speedup.loc["SWANN"].t_total
df_speedup /= t_total_swann
columns = {
    "t_sampling": "MCMC sampling",
    "t_E_kin_best": "Kinetic energy",
    "t_E_pot": "Effective core potential",
    "t_Spin": "$S^+$ spin operator",
}
df_speedup = df_speedup[columns.keys()]
df_speedup = df_speedup.rename(columns=columns)
df_speedup.plot(kind="bar", stacked=True, ax=ax_speedup, rot=0)
for i, model in enumerate(models):
    ax_speedup.text(i, df_speedup.loc[model].sum(), f"{df_speedup.loc[model].sum():.1f}x", ha="center", va="bottom")
ax_speedup.set_ylabel("Total time relative to SWANN")
ax_speedup.set_title(f"Runtime breakdown for {N_EL_FOR_BREAKDOWN} electrons")
ax_speedup.set_ylim((0, df_speedup.sum(axis=1).max() * 1.1))
ax_speedup.set_xlabel(None)
ax_speedup.legend(frameon=False)
fig.tight_layout()
fig.savefig("scaling.pdf", bbox_inches="tight")
fig.savefig("scaling.png", bbox_inches="tight", dpi=200)
