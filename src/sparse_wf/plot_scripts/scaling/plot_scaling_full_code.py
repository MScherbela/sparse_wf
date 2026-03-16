# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker
from sparse_wf.plot_utils import COLOR_PALETTE, COLOR_FIRE
import scienceplots  # noqa: F401

plt.style.use(["science", "grid"])

N_SWEEPS = 2
REFERENCE_BATCH_SIZE = 512
N_EL_MIN_FOR_FIT = 100
N_EL_MIN_FOR_PLOT = 100
N_EL_FOR_BREAKDOWN = 200

NAIVE_FIRE_NAME = "Naive FiRE"


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


# Runtime data from separately running sampling and kinetic energy evaluations
df = pd.read_csv("data/timings_4k.csv")
df["t"] = df["t"] * REFERENCE_BATCH_SIZE / df.batch_size
df["n_el"] = df["system_size"] * 4 + 4
df_fire_dense = df[df.model == "FiRE"].copy()
df_fire_dense["model"] = NAIVE_FIRE_NAME
df_fire_dense = df_fire_dense[~df_fire_dense.operation.isin(["E_kin", "wf_lowrank"])]
df_fire_dense["operation"] = df_fire_dense["operation"].str.replace("E_kin_dense", "E_kin")
df = pd.concat([df, df_fire_dense], ignore_index=True)
pivot = df.pivot_table(index=["model", "n_el"], columns="operation", values="t").reset_index()
pivot = pivot.rename(columns={"E_kin": "t_E_kin", "wf_full": "t_wf_full", "wf_lowrank": "t_wf_lowrank"})
pivot = pivot.drop(columns=["E_kin_dense"])
pivot["n_ECP_evals"] = pivot.n_el * 4
is_dense = pivot["model"] != "FiRE"
pivot["t_update"] = pivot.t_wf_full.where(is_dense, pivot.t_wf_lowrank)
pivot["t_E_pot"] = pivot.n_ECP_evals * pivot.t_update
pivot["t_Spin"] = pivot.t_update * pivot.n_el / 2
pivot["t_sampling"] = (pivot.n_el * N_SWEEPS - 1) * pivot.t_update + pivot.t_wf_full

# pivot["t_total"] = pivot.t_sampling + pivot.t_E_kin + pivot.t_E_pot + pivot.t_Spin
# Runtime data from full code runs
key_columns = ["model", "cutoff", "n_carbon"]
df_full_code_fire = pd.read_csv("data/full_run_data.csv")
df_full_code_fire = df_full_code_fire[df_full_code_fire["cutoff"] == 3.0]
# df_full_code_fire["model"] = "FiRE (c=" + df_full_code_fire.cutoff.astype(str) + ")"
df_full_code_lapnet = pd.read_csv("data/full_run_data_lapnet.csv")
df_full_code_lapnet["cutoff"] = np.nan
df_full_code = pd.concat([df_full_code_fire, df_full_code_lapnet], ignore_index=True)
df_full_code = df_full_code[df_full_code["opt/t_step"].notna()]
df_full_code = df_full_code.sort_values(["batch_size", "opt_batch_size"], ascending=False)
df_full_code = df_full_code.drop_duplicates(subset=key_columns, keep="first")
df_full_code = df_full_code.sort_values(key_columns)
df_full_code["n_el"] = df_full_code.n_carbon * 4 + 4
df_full_code["t_total"] = df_full_code["opt/t_step"] * REFERENCE_BATCH_SIZE / df_full_code.batch_size

df_full_code["model"] = df_full_code["model"].apply(lambda s: s[0].capitalize() + s[1:])
pivot["model"] = pivot["model"].apply(lambda s: s[0].capitalize() + s[1:])

pivot = pivot.merge(
    df_full_code[["model", "n_el", "t_total"]],
    on=["model", "n_el"],
    how="left",
)
pivot = pivot[pivot.n_el >= N_EL_MIN_FOR_PLOT]


plt.close("all")
fig, axes = plt.subplots(1, 4, figsize=(10, 4), width_ratios=[1, 1, 1, 1])
ax_upd, ax_Ekin, ax_tot, ax_speedup = axes.flatten()

models = ["Ferminet", "Psiformer", "Lapnet", NAIVE_FIRE_NAME, "FiRE"]
model_colors = {
    "Ferminet": COLOR_PALETTE[0],
    "Psiformer": COLOR_PALETTE[2],
    "Lapnet": COLOR_PALETTE[3],
    NAIVE_FIRE_NAME: COLOR_PALETTE[1],
    "FiRE": COLOR_FIRE,
}
markers = ["o", "s", "d", "^", "v"]
for model, marker in zip(models, markers):
    color = model_colors[model]
    kwargs_filled = dict(marker=marker, ls="none", color=color)
    df_model = pivot[pivot.model == model]

    # Plot update times
    exponent = fit_and_plot(ax_upd, df_model.n_el, df_model.t_update, color)
    label = f"{model}: {format_exponent(exponent)}"
    ax_upd.plot(df_model.n_el, df_model.t_update, label=label, **kwargs_filled)

    # Plot kinetic energy times
    exponent = fit_and_plot(ax_Ekin, df_model.n_el, df_model.t_E_kin, color)
    label = f"{model}: {format_exponent(exponent)}"
    ax_Ekin.plot(df_model.n_el, df_model.t_E_kin, label=label, **kwargs_filled)

    # Plot total time
    if model == NAIVE_FIRE_NAME:
        continue
    exponent = fit_and_plot(ax_tot, df_model.n_el, df_model.t_total, color)
    label = f"{model}: {format_exponent(exponent)}"
    ax_tot.plot(df_model.n_el, df_model.t_total, label=label, **kwargs_filled)

ax_upd.set_title("wavefunction update")
ax_Ekin.set_title("kinetic energy")
ax_tot.set_title("total optimization step")
for ax, ymin in [(ax_upd, 1e-3), (ax_Ekin, 3e-2), (ax_tot, 3e-1)]:
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("valence electrons", fontsize=12)
    if ax == ax_upd:
        ax.set_ylabel("runtime [sec]")

    lines_others = ax.get_lines()[1:6:2]
    lines_fire = ax.get_lines()[7::2]
    leg = ax.legend(
        lines_others,
        [l.get_label() for l in lines_others],
        loc="upper left",
        frameon=False,
        handletextpad=0.0,
        bbox_to_anchor=(-0.04, 1.02),
    )
    ax.legend(
        lines_fire,
        [l.get_label() for l in lines_fire],
        loc="lower right",
        frameon=False,
        handletextpad=0.0,
        bbox_to_anchor=(1.02, -0.02),
    )
    ax.add_artist(leg)
    ax.set_xlim([90, 550])
    ax.set_ylim([ymin, None])
    ax.set_xticks([100, 140, 200, 300, 500])
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.minorticks_off()
    # ax.axvline(N_EL_FOR_BREAKDOWN, color="dimgray", ls="-", zorder=-1)
    ax.grid(alpha=0.2)
# ax_Ekin.set_ylim([None, 1e4])

for ax, label in zip(axes.flatten(), "abcd"):
    ax.text(
        -0.12,
        1.02,
        f"\\textbf{{{label})}}",
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontweight="bold",
        fontsize=12,
    )

# Speedup bar-chart
models_for_speedup = ["Ferminet", "Psiformer", "Lapnet", "FiRE"]
df_speedup = pivot[pivot.n_el == N_EL_FOR_BREAKDOWN].set_index("model").reindex(models_for_speedup)
df_speedup["speedup"] = df_speedup.t_total / df_speedup.t_total.loc["FiRE"]
df_speedup = df_speedup[df_speedup.index != NAIVE_FIRE_NAME]
xticks = np.arange(len(df_speedup))
ax_speedup.set_title(f"total for {N_EL_FOR_BREAKDOWN} val. electrons")
ax_speedup.set_ylim([0, 140])
ax_speedup.set_xlabel(None)
ax_speedup.set_xticks(xticks)
ax_speedup.set_xticklabels(["Fermi-\nnet", "Psi-\nformer", "Lap-\nNet", "FiRE"], rotation=0)
# plt.setp(ax_speedup.get_xticklabels(), rotation=25, ha="right")
for i, model in enumerate(models_for_speedup):
    ax_speedup.bar([xticks[i]], [df_speedup.loc[model].t_total], color=model_colors[model], width=0.75)
    ax_speedup.text(
        i,
        df_speedup.loc[model].t_total + 2,
        f"{df_speedup.loc[model].speedup:.0f}$\\times$",
        ha="center",
        va="bottom",
    )
fig.tight_layout()
fig.subplots_adjust(wspace=0.2)
# fig.savefig("figures/scaling_full_code.pdf", bbox_inches="tight")
fig.savefig("scaling.png", bbox_inches="tight", dpi=300)
fig.savefig("scaling.pdf", bbox_inches="tight")

# %%
