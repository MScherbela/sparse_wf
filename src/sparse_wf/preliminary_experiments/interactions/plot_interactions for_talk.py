# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from sparse_wf.plot_utils import (
    COLOR_PALETTE,
    COLOR_FIRE,
    get_outlier_mask,
    savefig,
    format_value_with_error,
    MILLIHARTREE,
    scale_lightness,
)
import scienceplots
from matplotlib.lines import Line2D

plt.style.use(["science", "grid"])
import matplotlib as mpl

mpl.rcParams["savefig.bbox"] = None


def linebreak(s, lmax=16):
    tokens = s.split()
    lines = [""]
    for t in tokens:
        if (len(lines[-1]) == 0) or (len(lines[-1]) + len(t)) <= lmax:
            lines[-1] += " " + t
        else:
            lines.append(t)
    return "\n".join(lines)


df_ref = pd.read_csv("interaction_references.csv")
df_fire = pd.read_csv("interactions_aggregated.csv")
df_fire = df_fire[df_fire.cutoff.isin([5])]
# df_fire = df_fire.pivot_table(["deltaE_mean", "deltaE_mean_extrapolated", "deltaE_err"], index="molecule", columns="cutoff", aggfunc="mean")
# df_fire = df_fire.groupby("molecule")[["deltaE_mean", "deltaE_mean_extrapolated", "deltaE_err"]].mean()
df_fire = df_fire[df_fire.cutoff == 5][["molecule", "deltaE_mean", "deltaE_extrapolated", "deltaE_err"]]
df_fire.columns = ["molecule", "FiRE_5", "FiRE_extrapolated_5", "FiRE_5_err"]
df_fire["FiRE_extrapolated_5_err"] = df_fire["FiRE_5_err"]
# df_fire.columns = [f"FiRE_{c}_err" if "err" in name else f"FiRE_{c}" for name, c in df_fire.columns.values]
df = df_fire.merge(df_ref, on="molecule").set_index("molecule") * 1000

fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), width_ratios=[1, 1], dpi=200)
ax_s22, ax_benz = axes
# Add inset axis in the bottom right of ax_s22
# ax_scatter = ax_s22.inset_axes([0.68, 0.15, 0.3, 0.35])

ref_method = "CCSD(T)"

methods = [
    ("LapNet", "LapNet", COLOR_PALETTE[0], "s"),
    # ("FiRE_5", "FiRE, $c=5$", COLOR_FIRE, "o"),
    ("FiRE_extrapolated_5", "FiRE, $c=5a_0$", COLOR_FIRE, "o"),
]

bar_width = 0.8 / len(methods)
for idx_method, (method, label, color, marker) in enumerate(methods):
    delta_to_ref = df[method] - df[ref_method]
    pos_barchart = np.arange(len(df)) + bar_width * (idx_method - len(methods) / 2 + 0.5)
    ax_s22.barh(pos_barchart, delta_to_ref, color=color, height=bar_width, zorder=3, label=label)
    color_err = scale_lightness(color, 0.7)
    ax_s22.errorbar(
        delta_to_ref,
        pos_barchart,
        xerr=df[method + "_err"],
        color=color_err,
        marker="none",
        ls="none",
        capsize=3,
        zorder=4,
    )

for method in ["LapNet", "FiRE_extrapolated_5", "FiRE_5"]:
    mae = np.abs(df[method] - df["CCSD(T)"]).mean()
    print(f"MAE {method:<20}: {mae:.1f} mEh")


with open("s22_table.tex", "w") as f:
    f.write(
        r"{molecule} & {\makecell[l]{FiRE, $c=5a_0$\\raw}} & {\makecell[l]{FiRE, $c=5a_0$\\extrapolated}} & {LapNet} & {CCSD(T)}\\"
    )
    f.write("\n\\midrule\n")
    for mol, row in df.iterrows():
        s_fire_5_raw = format_value_with_error(row["FiRE_5"], row["FiRE_5_err"])
        s_fire_5_ext = format_value_with_error(row["FiRE_extrapolated_5"], row["FiRE_5_err"])
        s_lapnet = format_value_with_error(row["LapNet"], row["LapNet_err"])
        s_ccsdt = f"{row['CCSD(T)']:.2f}"
        mol = " ".join(mol.split("_")[1:])
        f.write(f"{mol} & {s_fire_5_raw} & {s_fire_5_ext} & {s_lapnet} & {s_ccsdt}\\\\\n")


for y in [3.5, 6.5]:
    ax_s22.axhline(y, color="dimgray", ls="--", lw=0.5, alpha=0.5)

mol_name_translation = {
    "01_Water_dimer": "2$\\times$ water",
    "02_Formic_acid_dimer": "2$\\times$ formic acid",
    "03_Formamide_dimer": "2$\\times$ formamid",
    "04_Uracil_dimer_h-bonded": "H-bonded uracil",
    "05_Methane_dimer": "2$\\times$ methane",
    "06_Ethene_dimer": "2$\\times$ ethene",
    "07_Uracil_dimer_stack": "Stacked uracil",
    "08_Ethene-ethyne_complex": "Ethene-ethyne",
    "09_Benzene-water_complex": "Benzene-water",
    "10_Benzene_dimer_T-shaped": "2$\\times$ benzene",
    "11_Phenol_dimer": "2$\\times$ phenol",
}
mol_names = [mol_name_translation[m] for m in df.index]

xlims = [-5, 5]
ax_s22.set_xlabel("$E_\\textrm{NN-VMC}$ - $E_\\textrm{CCSD(T)}$ " + MILLIHARTREE)
ax_s22.grid(False, axis="y")
ax_s22.yaxis.minorticks_off()
ax_s22.legend(loc="upper right")
ax_s22.set_yticks(np.arange(len(df)))
ax_s22.set_yticklabels(mol_names)
ax_s22.set_xlim(xlims)
ax_s22.axvline(0, color="black", ls="-")
ax_s22.axvspan(-1.6, 1.6, color="dimgray", alpha=0.3)
ax_s22.invert_yaxis()

idx_start = 0
for interaction, n_mol in [("H-bonds", 4), ("Dispersion", 3), ("Mixed interactions", 4)]:
    x_pos = xlims[0] - 4
    line = Line2D([x_pos, x_pos], [idx_start - 0.4, idx_start + n_mol - 1 + 0.4], color="gray", lw=1)
    line.set_clip_on(False)
    ax_s22.add_line(line)
    ax_s22.text(
        x_pos - 0.1, idx_start + n_mol / 2 - 0.5, interaction, ha="right", va="center", rotation=90, color="gray"
    )
    idx_start += n_mol

# draw a rectangle with dashed lines on ax_s22
ax_s22.add_patch(plt.Rectangle((-3, 8.5), 6, 1, fill=False, linestyle="--", color="gray", lw=1))

fig.tight_layout()


# Benzene dimer
dists = [4.95, 10.0]

df_agg = pd.read_csv("interactions_aggregated.csv")
df_agg = df_agg[df_agg["molecule"] == "10_Benzene_dimer_T-shaped"]
# df_agg = df_agg[df_agg["cutoff"].isin([cutoffs])]
df_agg["method"] = "FiRE $c=" + df_agg.cutoff.astype(str) + "a_0$"
df_agg["ref_type"] = "FiRE"
df_agg = df_agg.sort_values("cutoff")
df_agg["deltaE_raw"] = df_agg["deltaE_mean"]
df_agg["deltaE_mean"] = df_agg["deltaE_extrapolated"]

df_ref = pd.read_csv("benzene_references.csv")
df_ref = df_ref[~df_ref.method.str.contains("UHF")]
df_agg = pd.concat([df_ref, df_agg], axis=0, ignore_index=True)

df_agg["method"] = df_agg["method"].str.replace(" (", "\n")
df_agg["method"] = df_agg["method"].apply(lambda s: s[:-1] if s.endswith(")") else s)
df_agg["method"] = df_agg["method"].str.replace(" $", "\n$")
# df_agg["method"] = df_agg["method"].str.replace(" / ", "\n")


colors = {
    "Experiment\nZPVE corrected": "dimgray",
    "CCSD(T)": "k",
    "Ferminet": COLOR_PALETTE[1],
    "Psiformer": COLOR_PALETTE[2],
    "LapNet": COLOR_PALETTE[0],
    "FiRE": COLOR_FIRE,
}
color_exp = "k"
color_CC = "dimgray"
color_ferminet, color_psiformer, color_lapnet = COLOR_PALETTE[1:4]

exp_uncertainty = 0.8  # mHa; original value is 2 - 2.8 kcal/mol
df_experiment = df_agg[df_agg["ref_type"] == "Experiment"]
delta_E_range = (
    (df_experiment["deltaE_mean"] - df_experiment["deltaE_err"]).min(),
    (df_experiment["deltaE_mean"] + df_experiment["deltaE_err"]).max(),
)

with open("benzene_table.tex", "w") as f:
    f.write("{method} & {interaction energy}\\\\\n")
    f.write("\\midrule\n")
    for _, row in df_agg.iterrows():
        method = row["method"].replace("\n", ", ")
        E, E_err = row["deltaE_mean"] * 1000, row["deltaE_err"] * 1000
        if "FiRE" in method:
            s = format_value_with_error(row["deltaE_raw"] * 1000, E_err)
            f.write(f"{method}, raw & {s}\\\\\n")
            s = format_value_with_error(E, E_err)
            f.write(f"{method}, extrapolated & {s}\\\\\n")
        elif np.isfinite(E_err):
            s = format_value_with_error(E, E_err)
            f.write(f"{method} & {s}\\\\\n")
        else:
            s = f"{E:.1f}"
            f.write(f"{method} & {s}\\\\\n")


ref_type_labeled = {r: False for r in df_agg["ref_type"].unique()}
bar_width = 0.8
for i, r in df_agg.iterrows():
    E, E_err, method, ref_type = r["deltaE_mean"] * 1000, r["deltaE_err"] * 1000, r["method"], r["ref_type"]
    if "Experiment" in method:
        color = colors["Experiment\nZPVE corrected"]
    elif "CCSD" in method:
        color = colors["CCSD(T)"]
    elif "FermiNet" in method:
        color = colors["Ferminet"]
    elif "Psiformer" in method:
        color = colors["Psiformer"]
    elif "LapNet" in method:
        color = colors["LapNet"]
    else:
        color = COLOR_FIRE
    color_err = scale_lightness(color, 0.7)
    # label = ref_type if not ref_type_labeled[ref_type] else None
    ref_type_labeled[ref_type] = True
    ax_benz.barh(i, E, height=bar_width, color=color, label=None, zorder=3)
    ax_benz.errorbar(E, i, xerr=E_err, color=color_err, marker="none", ls="none", capsize=3, zorder=4)
    x_pos_text, color = (0.3, "white") if np.abs(E) >= 2 else (E + 0.6, "k")
    ax_benz.text(x_pos_text, i, f"{E:.1f}", va="center", ha="left", color=color, zorder=4, fontsize=9)

for label, color in colors.items():
    ax_benz.barh([np.nan], [np.nan], color=color, label=label)
ax_benz.set_yticks(range(len(df_agg)))
ax_benz.set_yticklabels(df_agg["method"], fontsize=9)
ax_benz.grid(False, axis="y")
ax_benz.yaxis.minorticks_off()
# ax.grid(alpha=0.5, axis="x")
# ax_benz.axvline(delta_E_exp * 1000, color="k", linestyle="--", zorder=0)
ax_benz.axvline(0, color="k", linestyle="-", zorder=0)
ax_benz.legend(loc="lower right", ncol=1)
ax_benz.set_ylim(
    [
        len(df_agg) - 1 + bar_width / 2 + 0.1,
        -bar_width / 2 - 0.1,
    ]
)

ax_benz.axvspan(delta_E_range[0] * 1000, delta_E_range[1] * 1000, color="k", alpha=0.1, zorder=0)
ax_benz.set_xlabel("binding energy $E$ " + MILLIHARTREE)

# Insert a picture in the lower left corner
img = Image.open("../benzene_dimer/benzene_dimer_T_4.png")
img = img.rotate(90, expand=True)
img = img.crop(img.getbbox())
image_ax = fig.add_axes((0.83, 0.6, 0.12, 0.5))
image_ax.imshow(img)
image_ax.axis("off")
fig.tight_layout(rect=[0.0, 0, 1, 1])
fig.subplots_adjust(wspace=0.4)
ax_s22.text(0.04, 0.97, "\\textbf{a)}", transform=fig.transFigure, va="top", ha="left")
ax_s22.text(0.535, 0.97, "\\textbf{b)}", transform=fig.transFigure, va="top", ha="left")
savefig(fig, "interactions", bbox_inches=None)
