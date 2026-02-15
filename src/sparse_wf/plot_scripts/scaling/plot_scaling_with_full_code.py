# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker
from sparse_wf.plot_utils import COLOR_PALETTE, COLOR_FIRE

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


key_columns = ["model", "cutoff", "n_carbon"]

df_full_code_fire = pd.read_csv("data/full_run_data.csv")
df_full_code_fire["model"] = "FiRE (c=" + df_full_code_fire.cutoff.astype(str) + ")"
df_full_code_lapnet = pd.read_csv("data/full_run_data_lapnet.csv")
df_full_code_lapnet["cutoff"] = np.nan
df_full_code = pd.concat([df_full_code_fire, df_full_code_lapnet], ignore_index=True)
df_full_code = df_full_code[df_full_code["opt/t_step"].notna()]
df_full_code = df_full_code.sort_values(["batch_size", "opt_batch_size"], ascending=False)
df_full_code = df_full_code.drop_duplicates(subset=key_columns, keep="first")
df_full_code = df_full_code.sort_values(key_columns)
df_full_code["model"] = df_full_code["model"].apply(lambda s: s[0].capitalize() + s[1:])
df_full_code["n_el"] = df_full_code.n_carbon * 4 + 4
df_full_code["t_total"] = df_full_code["opt/t_step"] * REFERENCE_BATCH_SIZE / df_full_code.batch_size
df_full_code["source"] = "full code"


df = pd.read_csv("data/timings_4k.csv")
df["t"] = df["t"] * REFERENCE_BATCH_SIZE / df.batch_size
df["n_el"] = df["system_size"] * 4 + 4
df["model"] = df["model"].apply(lambda s: s[0].capitalize() + s[1:])
df = df[df.n_el >= N_EL_MIN_FOR_PLOT]

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
pivot["source"] = "estimate"
pivot["model"] = pivot["model"].replace({"FiRE": "FiRE (c=3.0)"})

df_all = pd.concat([pivot, df_full_code], ignore_index=True)

plt.close("all")
fig, axes = plt.subplots(1, 4, figsize=(10, 4), width_ratios=[1, 1, 1, 1], sharey=True)


models = ["Ferminet", "Psiformer", "Lapnet", "FiRE (c=3.0)"]
model_colors = [COLOR_PALETTE[i] for i in [0, 2, 3]] + [COLOR_FIRE]
for ax, model, color in zip(axes.flatten(), models, model_colors):
    df_model = df_all[df_all.model == model]
    if model == "FiRE":
        df_model = df_model[df_model.cutoff == 3.0]
    for source, ls, marker in zip(["estimate", "full code"], ["-", "--"], ["o", "x"]):
        df_plot = df_model[df_model.source == source]
        exponent = fit_and_plot(ax, df_plot.n_el, df_plot.t_total, color, ls=ls)
        label = f"{model}: {format_exponent(exponent)}"
        ax.plot(df_plot.n_el, df_plot.t_total, label=label, ls="none", color=color, marker=marker)
    ax.set_title(model)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("valence electrons", fontsize=12)
    ax.set_xlim([90, 550])
    ax.set_xticks([100, 140, 200, 300, 500])
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.minorticks_off()
