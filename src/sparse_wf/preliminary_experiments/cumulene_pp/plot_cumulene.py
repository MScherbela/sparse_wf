#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def get_outlier_mask(x):
    qlow = x.quantile(0.01)
    qhigh = x.quantile(0.99)
    med = x.median()
    included_range = 5 * (qhigh - qlow)
    is_outlier = (x < med - included_range) | (x > med + included_range)
    return is_outlier



df_agg = pd.read_csv("cumulene_pp_aggregated.csv")
df_ref = pd.read_csv("energies.csv", sep=",")
include = True
# include = df_ref.basis_set.str.contains("TZ") | df_ref.method.isin(["NEVPT2", "CCSD(T)", "DLPNO", "CASSCF"])
include &= ~df_ref.method.isin(["RHF", "UHF"])
include &= ~df_ref.comment.str.contains("quintuplet")
df_ref = df_ref[include]
# df_hf = df_ref[df_ref["method"] == "DLPNO-CCSD(T)"].copy()
# df_hf["E_final"] = df_hf["E_hf"]
# df_hf["method"] = "HF"
# df_ref = pd.concat([df_ref, df_hf])
df_ref["method"] = df_ref["method"] + " / " + df_ref["basis_set"]
df_ref["n_carbon"] = df_ref["comment"].apply(lambda s: int(s.split("_")[1].replace("C", "").replace("H4", "")))
df_ref["angle"] = df_ref["comment"].apply(lambda s: 0 if "_0deg" in s else (90 if "90deg_triplet" in s else "90_quint"))
pivot_ref = df_ref.pivot_table(index=["method", "n_carbon"], columns="angle", values="E_final", aggfunc="mean")
pivot_ref["deltaE"] = (pivot_ref[90] - pivot_ref[0]) * 1000
# pivot_ref["deltaE_quint"] = (pivot_ref["90_quint"] - pivot_ref[0]) * 1000
pivot_ref = pivot_ref.reset_index()


fig, ax = plt.subplots(1, 1, figsize=(10, 7))
colors = ["orange", "red"]
cutoffs = [3.0]
for cutoff, color in zip(cutoffs, colors):
    df = df_agg[df_agg["cutoff"] == cutoff]
    ax.errorbar(
        df["n_carbon"],
        df["deltaE_mean"] * 1000,
        yerr=df["deltaE_err"] * 1000,
        label=f"SWANN, c={cutoff}",
        color=color,
        marker="o",
    )

ref_colors = ["black", "dimgray", "darkgreen", "C0", "navy"]
for method, color in zip(pivot_ref["method"].unique(), ref_colors):
    df = pivot_ref[pivot_ref["method"] == method]
    ax.plot(df["n_carbon"], df["deltaE"], label=method, linestyle="-", marker="s", color=color)
    # ax.plot(df["n_carbon"], df["deltaE_quint"], linestyle="--", marker="s", color=color)
# ax.scatter([4, 12, 16], [62.7,23.8,16], label="S+", color="navy", marker="x", zorder=10, s=100)

ax.legend()
ax.grid(alpha=0.5)
ax.axhline(0, color="black", linestyle="-", zorder=-1)
ax.set_ylim([-10, 70])
# x_fit = np.linspace(2, 24, 100)
# y_fit = 200/(x_fit-1)
# ax.plot(x_fit, y_fit, label="1/n", color="black", linestyle="--")
fig.savefig("cumulene_pp.png", bbox_inches="tight", dpi=200)

#%%
window = 2000
df_full = pd.read_csv("cumulene_pp_energies.csv")
fig, axes = plt.subplots(2, 4, figsize=(10, 8))
for idx_n_carbon, n_carbon in enumerate(df_full["n_carbon"].unique()):
    ax = axes.flatten()[idx_n_carbon]
    twinx = ax.twinx()
    ax.set_title(f"C{n_carbon}H4")
    for cutoff, color in zip(cutoffs, colors):
        df = df_full[(df_full["cutoff"] == cutoff) & (df_full["n_carbon"] == n_carbon)]
        pivot = df.pivot_table(index="opt/step", columns="angle", values="opt/E", aggfunc="mean")
        if len(pivot.columns) < 2:
            print(f"Not enough data for, c={cutoff}, n={n_carbon}")
            continue
        pivot["deltaE"] = (pivot[90] - pivot[0]) * 1000
        pivot = pivot[pivot.deltaE.notna()]
        is_outlier = get_outlier_mask(pivot["deltaE"])
        pivot = pivot[~is_outlier]
        pivot = pivot.rolling(window).mean()
        ax.plot(pivot.index / 1000, pivot[90], label=f"90 deg, c={cutoff}", color=color)
        ax.plot(pivot.index / 1000, pivot[0], label=f"0 deg, c={cutoff}", color=color)
        ax.set_ylim([pivot[0].iloc[-1]-5e-3, pivot[90].iloc[-1]+20e-3])
        twinx.plot(pivot.index / 1000, pivot["deltaE"], label=f"deltaE, c={cutoff}", color='k', linestyle="--")
        twinx.set_ylim([pivot["deltaE"].iloc[-1]-10, pivot["deltaE"].iloc[-1]+10])
fig.tight_layout()
fig.savefig("cumulene_pp_all.png", bbox_inches="tight", dpi=200)


