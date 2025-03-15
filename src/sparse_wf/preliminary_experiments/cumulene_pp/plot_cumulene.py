#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sparse_wf.plot_utils import get_outlier_mask, COLOR_FIRE, COLOR_PALETTE, cbs_extrapolate, get_colors_from_cmap, abbreviate_basis_set, savefig
from PIL import Image
import scienceplots
plt.style.use(['science', 'grid'])
CUTOFF = 3.0

df_agg = pd.read_csv("cumulene_pp_aggregated.csv")
df_agg = df_agg[(df_agg["n_carbon"] <= 16) & (df_agg["cutoff"] == CUTOFF)]
df_ref = pd.read_csv("orca_energies.csv", sep=",")
df_ref["method"] = df_ref["method"].str.replace("PBE0+CCSD(T)", "PBE0 CCSD(T)")
df_ref = df_ref[~df_ref.comment.str.contains("quintuplet")]
df_ref = df_ref[df_ref.method.isin(["PBE0 D3BJ", "PBE0 CCSD(T)", "PBE0 DLPNO-CCSD(T)", "UHF"])]
df_ref_cbs = cbs_extrapolate(df_ref)
df_ref = pd.concat([df_ref, df_ref_cbs], axis=0, ignore_index=True)
df_ref["method"] = df_ref["method"] + " / " + df_ref["basis_set"]
df_ref["n_carbon"] = df_ref["comment"].apply(lambda s: int(s.split("_")[1].replace("C", "").replace("H4", "")))
df_ref["angle"] = df_ref["comment"].apply(lambda s: 0 if "_0deg" in s else (90 if "90deg_triplet" in s else "90_quint"))
df_ref = df_ref[df_ref.n_carbon <= 16]
pivot_ref = df_ref.pivot_table(index=["method", "n_carbon"], columns="angle", values="E_final", aggfunc="mean")
pivot_ref["deltaE"] = (pivot_ref[90] - pivot_ref[0])
pivot_ref = pivot_ref.reset_index()
print(pivot_ref.method.unique())

columns = {"n_carbon": "n_carbon", "E0": 0, "E90": 90, "deltaE_mean": "deltaE"}
pivot_fire = df_agg.rename(columns=columns)
pivot_fire = pivot_fire[columns.values()].copy()
pivot_fire["method"] = "FiRE"
pivot = pd.concat([pivot_fire, pivot_ref], axis=0, ignore_index=True)

delta_E_fire = pivot_fire.groupby("n_carbon")["deltaE"].mean().to_dict()
pivot["deltaE_groundtruth"] = pivot.n_carbon.map(pivot[pivot.method == "PBE0 CCSD(T) / CBS"].set_index("n_carbon")["deltaE"])
pivot["deviation"] = pivot["deltaE"] - pivot["deltaE_groundtruth"]
pivot = pivot[pivot.n_carbon.isin([4, 6, 8, 12, 16, 20, 24, 36])]


fig, ax_deltaE = plt.subplots(1, 1, figsize=(6, 3.5))
ax_deviation = ax_deltaE.inset_axes([0.4, 0.5, 0.6, 0.5])
ax_deviation.axhspan(-1.6, 1.6, color="gray", alpha=0.5)
axes = [ax_deltaE, ax_deviation]

# colors_dlpno = get_colors_from_cmap("Greens", np.linspace(0.4, 0.9, 3))
colors_cc = get_colors_from_cmap("Greys", np.linspace(0.6, 0.9, 2))

methods = [
    ("PBE0 CCSD(T) / CBS", "k"),
    ("PBE0 D3BJ / def2-TZVP", COLOR_PALETTE[0]),
    ("FiRE", COLOR_FIRE),
]

n_carbon_to_idx = {n: i for i,n in enumerate(pivot.n_carbon.unique())}
bar_width = 0.9 / len(methods)
for idx_method, (method, color) in enumerate(methods):
    label = abbreviate_basis_set(method)
    marker = "^" if "DZ" in method else ("v" if "TZ" in method else "s")
    df = pivot[pivot["method"] == method]
    ax_deltaE.plot(df["n_carbon"], df["deltaE"] * 1000, label=label, linestyle="-", marker=marker, color=color, alpha=1)
    x_bar = df["n_carbon"].map(n_carbon_to_idx) + idx_method * bar_width - 0.9/2
    # # ax_deviation.plot(df["n_carbon"], df["deviation"] * 1000, label=None,linestyle="-", marker=marker, color=color, alpha=1)
    ax_deviation.bar(x_bar, df["deviation"] * 1000, label=None, color=color, width=bar_width, zorder=3)

ax_deltaE.axhline(0, color="black", linestyle="-", zorder=-1)
ax_deltaE.set_ylabel(r"$E_\text{twisted} - E_\text{planar}$  [m$E_\text{h}$]")
ax_deltaE.set_xlabel("number of carbon atoms")
ax_deltaE.set_ylim([None, 85])

ax_deviation.set_ylabel(r"$\Delta - \Delta_{\text{CCSD(T)}}$ [mE$_\text{h}$]")
ax_deviation.axhline(0, color="k")
ax_deviation.set_xticks(np.arange(len(n_carbon_to_idx)))
ax_deviation.set_xticklabels([f"C{n}H4" for n in n_carbon_to_idx.keys()])
ax_deviation.grid(False, axis="x")
ax_deviation.set_xlim([-0.5, len(n_carbon_to_idx) - 0.5])
ax_deviation.xaxis.minorticks_off()
ax_deviation.set_ylim([-15, 2])

plt.figlegend(ncol=3, loc="upper center")
fig.tight_layout()
fig.subplots_adjust(top=0.90)

for fname, bbox in [
    # ("cumulene_C4H4_0deg_singlet.png", [0.15, 0.1, 0.25, 0.2]),
    ("cumulene_C4H4_90deg_triplet.png", [0.15, 0.15, 0.3, 0.2]),
    ]:
    img = Image.open(fname)
    img = img.crop(img.getbbox())
    image_ax = fig.add_axes(bbox)
    image_ax.imshow(img)
    image_ax.axis("off")

savefig(fig, "cumulene_energy")

pivot_tex = pivot[pivot.method.isin([m[0] for m in methods])]
pivot_tex = pivot_tex.pivot_table(index="n_carbon", columns="method", values="deltaE") * 1000
with open("cumulene.tex", "w") as f:
    f.write("{molecule} & {FiRE} & {CCSD(T)} & {PBE0}\\\\\n")
    f.write("\\midrule\n")
    for n_carbon, row in pivot_tex.iterrows():
        mol = "\ch{C_{" + str(n_carbon)+ "}H4}"
        f.write(f"{mol} & {row['FiRE']:.1f} & {row['PBE0 CCSD(T) / CBS']:.1f} & {row['PBE0 D3BJ / def2-TZVP']:.1f} \\\\\n")

#%%
window = 1000
df_full = pd.read_csv("cumulene_pp_energies.csv")
fig, axes = plt.subplots(2, 3, figsize=(7, 7))
n_carbon_values = [2, 4, 6, 8, 12, 16]
for idx_n_carbon, n_carbon in enumerate(n_carbon_values):
    ax = axes.flatten()[idx_n_carbon]
    twinx = ax.twinx()
    ax.set_title(f"C{n_carbon}H4")
    df = df_full[(df_full["cutoff"] == CUTOFF) & (df_full["n_carbon"] == n_carbon)]
    pivot = df.pivot_table(index="opt/step", columns="angle", values="opt/E", aggfunc="mean")
    if len(pivot.columns) < 2:
        print(f"Not enough data for n={n_carbon}")
        continue
    pivot["deltaE"] = (pivot[90] - pivot[0]) * 1000
    pivot = pivot[pivot.deltaE.notna()]
    is_outlier = get_outlier_mask(pivot["deltaE"])
    pivot = pivot[~is_outlier]
    pivot = pivot.rolling(window).mean()
    pivot = pivot.iloc[::10] # subsample
    ax.plot(pivot.index / 1000, pivot[90], label=f"90 deg", color="dimgray")
    ax.plot(pivot.index / 1000, pivot[0], label=f"0 deg", color="black")
    ax.set_ylim([pivot[0].iloc[-1]-5e-3, pivot[90].iloc[-1]+20e-3])
    ax.yaxis.get_major_formatter().set_useOffset(False)
    twinx.plot(pivot.index / 1000, pivot["deltaE"], label=f"deltaE", color=COLOR_FIRE)
    twinx.set_ylim([pivot["deltaE"].iloc[-1]-10, pivot["deltaE"].iloc[-1]+10])
fig.tight_layout()
savefig(fig, "cumulene_optcurve", pdf=False)


