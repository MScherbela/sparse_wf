# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker
from sparse_wf.plot_utils import COLOR_PALETTE, COLOR_FIRE, savefig
import scienceplots
plt.style.use(["science", "grid"])

N_SWEEPS = 2
REFERENCE_BATCH_SIZE = 512
N_EL_MIN_FOR_FIT = 100
N_EL_FOR_BREAKDOWN = 200


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

df = pd.read_csv("data/timings_4k.csv")
df["t"] = df["t"] * REFERENCE_BATCH_SIZE / df.batch_size
df["n_el"] = df["system_size"] * 4 + 4
df_fire_dense = df[df.model == "FiRE"].copy()
df_fire_dense["model"] = "FiRE (dense)"
df_fire_dense = df_fire_dense[~df_fire_dense.operation.isin(["E_kin", "wf_lowrank"])]
df_fire_dense["operation"] = df_fire_dense["operation"].str.replace("E_kin_dense", "E_kin")
df = pd.concat([df, df_fire_dense], ignore_index=True)
df["model"] = df["model"].apply(lambda s: s[0].capitalize() + s[1:])

pivot = df.pivot_table(index=["model", "n_el"], columns="operation", values="t").reset_index()
pivot = pivot.rename(columns={"E_kin": "t_E_kin", "wf_full": "t_wf_full", "wf_lowrank": "t_wf_lowrank"})
pivot = pivot.drop(columns=["E_kin_dense"])

pivot["n_ECP_evals"] = pivot.n_el * 4
is_dense = pivot["model"] != "FiRE"
pivot["t_update"] = pivot.t_wf_full.where(is_dense, pivot.t_wf_lowrank)
pivot["t_E_pot"] = pivot.n_ECP_evals * pivot.t_update
pivot["t_Spin"] = pivot.t_update * pivot.n_el / 2
pivot["t_sampling"] = (pivot.n_el * N_SWEEPS - 1) * pivot.t_update + pivot.t_wf_full
pivot["t_total"] = pivot.t_sampling + pivot.t_E_kin + pivot.t_E_pot + pivot.t_Spin

plt.close("all")
fig, axes = plt.subplots(1, 4, figsize=(10, 4), width_ratios=[1,1,1,1])
ax_upd, ax_Ekin, ax_tot, ax_speedup = axes.flatten()

models = ["Ferminet",  "Psiformer", "Lapnet", "FiRE (dense)", "FiRE"]
model_colors = [COLOR_PALETTE[i] for i in [0, 2, 3, 1]] + [COLOR_FIRE]
markers = ["o", "s", "d", "^", "v"]
for model, color, marker in zip(models, model_colors, markers):
    kwargs_filled = dict(marker=marker, ls="none", color=color)
    df_model = pivot[pivot.model == model]
    model = model.replace(" (dense)", "\ndense")

    # Plot update times
    exponent = fit_and_plot(ax_upd, df_model.n_el, df_model.t_update, color)
    label = f"{model}: {format_exponent(exponent)}"
    ax_upd.plot(df_model.n_el, df_model.t_update, label=label, **kwargs_filled)
    ax_upd.set_title("wavefunction update")

    # Plot kinetic energy times
    exponent = fit_and_plot(ax_Ekin, df_model.n_el, df_model.t_E_kin, color)
    label = f"{model}: {format_exponent(exponent)}"
    ax_Ekin.plot(df_model.n_el, df_model.t_E_kin, label=label, **kwargs_filled)
    ax_Ekin.set_title("kinetic energy")

    # Plot total time
    exponent = fit_and_plot(ax_tot, df_model.n_el, df_model.t_total, color)
    label = f"{model}: {format_exponent(exponent)}"
    ax_tot.plot(df_model.n_el, df_model.t_total, label=label, **kwargs_filled)
    ax_tot.set_title("total optimization step")

for ax, ymin in [(ax_upd, 1e-3), (ax_Ekin, 3e-2), (ax_tot, 3e-1)]:
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("valence electrons", fontsize=12)
    if ax == ax_upd:
        ax.set_ylabel("runtime / s")

    lines_others = ax.get_lines()[1:6:2]
    lines_fire = ax.get_lines()[7::2]
    leg = ax.legend(lines_others, [l.get_label() for l in lines_others], loc="upper left", frameon=False, handletextpad=0.0, bbox_to_anchor=(-0.04, 1.02))
    ax.legend(lines_fire, [l.get_label() for l in lines_fire], loc="lower right", frameon=False, handletextpad=0.0)
    ax.add_artist(leg)
    ax.set_xlim([60, 600])
    ax.set_ylim([ymin, None])
    ax.set_xticks([70, 100, 140, 200, 300, 500])
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.minorticks_off()
    # ax.axvline(N_EL_FOR_BREAKDOWN, color="dimgray", ls="-", zorder=-1)
    ax.grid(alpha=0.2)
# ax_Ekin.set_ylim([None, 1e4])

for ax, label in zip(axes.flatten(), "abcd"):
    ax.text(-0.12, 1.02, f"\\textbf{{{label})}}", transform=ax.transAxes, va="bottom", ha="left", fontweight="bold", fontsize=12)

# Speedup bar-chart
df_speedup = pivot[pivot.n_el == N_EL_FOR_BREAKDOWN].set_index("model").reindex(models)
t_total_swann = df_speedup.loc["FiRE"].t_total
df_speedup /= t_total_swann
columns = {
    "t_sampling": "Sampling",
    "t_E_kin": "E$_\\textrm{kin}$",
    "t_E_pot": "ECP",
    "t_Spin": "$S^+$",
}
df_speedup = df_speedup[columns.keys()]
df_speedup = df_speedup.rename(columns=columns)
df_speedup.plot(kind="bar", stacked=True, ax=ax_speedup, rot=0, color=COLOR_PALETTE, width=0.75)
for i, model in enumerate(models):
    ax_speedup.text(i, df_speedup.loc[model].sum(), f"{df_speedup.loc[model].sum():.1f}x", ha="center", va="bottom")
# ax_speedup.set_ylabel("Total time relative to FiRE")
ax_speedup.set_title(f"$T / T_\\textrm{{FiRE}}$ for {N_EL_FOR_BREAKDOWN} electrons")
ax_speedup.set_ylim((0, df_speedup.sum(axis=1).max() * 1.15))
print(df_speedup.sum(axis=1))
ax_speedup.set_xlabel(None)
ax_speedup.legend(loc="upper right", labelspacing=0.2, frameon=False, ncol=2, columnspacing=0.8)
ax_speedup.set_ylim([0, 21])
ax_speedup.grid(False)
ax_speedup.xaxis.minorticks_off()
ax_speedup.set_xticklabels(["Fermi-\nnet", "Psi-\nformer", "Lap-\nNet", "FiRE\ndense", "FiRE"], rotation=0)
# plt.setp(ax_speedup.get_xticklabels(), rotation=25, ha='right')
fig.tight_layout()
fig.subplots_adjust(wspace=0.2)

savefig(fig, "scaling")
