# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sparse_wf.plot_utils import get_outlier_mask, COLOR_PALETTE, COLOR_FIRE, savefig, get_colors_from_cmap
import scienceplots
plt.style.use(["science", "grid"])

cutoffs = ["3.0", "Transfer5.0" ,"Transfer7.0"]
dists = [4.95, 10.0]

df_agg = pd.read_csv("benzene_aggregated.csv")
df_agg["method"] = "FiRE $r_c=$" + df_agg.cutoff.str.replace("Transfer", "")
df_agg["ref_type"] = "FiRE"

df_ref = pd.read_csv("benzene_references.csv")
df_agg = pd.concat([df_ref, df_agg], axis=0, ignore_index=True)

df_agg["method"] = df_agg["method"].str.replace(" (", "\n").str.replace("al)", "al")
df_agg["method"] = df_agg["method"].str.replace(" $", "\n$")
df_agg["method"] = df_agg["method"].str.replace(" / ", "\n")


plt.close("all")
fig, ax = plt.subplots(1,1, figsize=(7, 6))

colors = {"Experiment": "k",
"Conventional method": "dimgray",
"Neural network WF": COLOR_PALETTE[0],
"FiRE": COLOR_FIRE}

delta_E_exp = df_agg[df_agg["ref_type"] == "Experiment"]["deltaE_mean"].iloc[0]

ref_type_labeled = {r: False for r in df_agg["ref_type"].unique()}
for i, r in df_agg.iterrows():
    E, ref_type = r["deltaE_mean"] * 1000, r["ref_type"]
    label = ref_type if not ref_type_labeled[ref_type] else None
    ref_type_labeled[ref_type] = True
    ax.barh(i, E, height=0.8, color=colors[ref_type], label=label, zorder=3)
    ax.text(0.5 * np.sign(E), i, f"{E:.1f}", va="center", ha="left" if np.sign(E) > 0 else "right", color="white", zorder=4)
ax.set_yticks(range(len(df_agg)))
ax.set_yticklabels(df_agg["method"])
ax.invert_yaxis()
ax.grid(False, axis="y")
ax.yaxis.minorticks_off()
# ax.grid(alpha=0.5, axis="x")
ax.axvline(delta_E_exp * 1000, color="k", linestyle="--", zorder=0)
ax.axvline(0, color="k", linestyle="-", zorder=0)
ax.legend()

for y in [2.5, 6.5]:
    ax.axhline(y, color="k", ls="--", alpha=0.5, zorder=0, lw=0.5)
ax.set_xlabel("binding energy / mHa")
savefig(fig, "benzene_dimer_barchart")



#%%

window_kwargs = dict(window=5000, min_periods=100)

df_all = pd.read_csv("benzene_energies.csv")
# molecules = sorted(df_all["molecule"].unique())
pivot = df_all.pivot_table(index="opt/step", columns=["cutoff", "dist"], values="opt/E", aggfunc="mean")
pivot = pivot.ffill(limit=10)
cutoffs = ["3.0", "Transfer5.0", "Transfer7.0"]
for cutoff in cutoffs:
    pivot.loc[:, (cutoff, "delta")] = (pivot[(cutoff, dists[0])] - pivot[(cutoff, dists[1])]) * 1000
    is_outlier = get_outlier_mask(pivot[(cutoff, "delta")])
    for col in (4.95, 10.0, "delta"):
        pivot.loc[is_outlier, (cutoff, col)] = np.nan
        pivot.loc[:, (cutoff, col)] = pivot.loc[:, (cutoff, col)].ffill(limit=10)
        pivot.loc[:, (cutoff, f"{col}_smooth")] = pivot.loc[:, (cutoff, col)].rolling(**window_kwargs).mean()
        pivot.loc[:, (cutoff, f"{col}_stderr")] = pivot.loc[:, (cutoff, col)].rolling(**window_kwargs).std() / np.sqrt(window_kwargs["window"])

refs = {
    "Experiment": (-3.8, "k"),
    "PsiFormer": (5.0, COLOR_PALETTE[0]),
    "FermiNet VMC (Glehn et al)": (-4.6, COLOR_PALETTE[2]),
    "FermiNet DMC (Ren et al)": (-9.2, COLOR_PALETTE[3]),
}

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
ax_abs, ax_rel = axes
for ref, (E_ref, color) in refs.items():
    ax_rel.axhline(E_ref, color=color, linestyle="dashed")
    ax_rel.text(0.1, E_ref, ref, color=color, va="bottom", ha="left")

cmap = plt.get_cmap("YlOrRd")
colors = get_colors_from_cmap("YlOrRd",  np.linspace(0.4, 1.0, len(cutoffs)))
steps_max = [25, 50, 75]
for cutoff, max_opt_step, color in zip(cutoffs, steps_max, colors):
    df_cutoff = pivot[cutoff]
    df_cutoff = df_cutoff[df_cutoff.index < max_opt_step * 1000]
    # Subsample to reduce pdf plot file size
    df_cutoff = df_cutoff.iloc[::10]
    delta_E = df_cutoff["delta_smooth"]
    delta_Estd = df_cutoff["delta_stderr"]
    label = f"FiRE $r_c={cutoff.replace('Transfer', '')}$"
    delta_E_final = delta_E[delta_E.notna()].iloc[-1]
    ax_rel.plot(df_cutoff.index / 1000,  delta_E, label=label, color=color)
    # ax.axhline(delta_E_final, color=color, zorder=0, ls="--")
    ax_rel.fill_between(
        df_cutoff.index / 1000,
        delta_E - 2 * delta_Estd,
        delta_E + 2 * delta_Estd,
        color=color,
        alpha=0.2,
    )
    ax_abs.plot(df_cutoff.index / 1000, df_cutoff["4.95_smooth"], label=label, color=color)
    ax_abs.plot(df_cutoff.index / 1000, df_cutoff["10.0_smooth"], color=color, ls="--")
    print(f"{cutoff}: {delta_E_final:.1f} mHa")
ax_rel.legend(loc="upper right")
ax_rel.set_ylim([-10, 6])
ax_rel.set_xlim([0, None])
ax_rel.set_xlabel("Opt Step / k")
ax_rel.set_ylabel("$E_{4.95A} - E_{10.0A}$ / mHa")

ax_abs.set_title("Absolute energy")
ax_abs.set_ylim([-75.37, -75.30])
ax_abs.set_ylabel("energy / Ha")
ax_rel.set_title("Relative energy")

for ax in axes:
    ax.set_xlabel("optimization step / k")
    ax.legend(loc="upper right")
savefig(fig, "benzene_dimer_optcurves")
