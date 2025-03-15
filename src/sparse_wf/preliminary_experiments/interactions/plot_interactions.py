#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from sparse_wf.plot_utils import COLOR_PALETTE, COLOR_FIRE, get_outlier_mask, savefig, format_value_with_error, MILLIHARTREE
import scienceplots
from matplotlib.lines import Line2D
plt.style.use(['science', 'grid'])
import matplotlib as mpl
mpl.rcParams["savefig.bbox"] = None


def linebreak(s, lmax=12):
    tokens = s.split()
    lines = [""]
    for t in tokens:
        if (len(lines[-1]) == 0) or (len(lines[-1]) + len(t)) <= lmax:
            lines[-1] += " " + t
        else:
            lines.append(t)
    return "\n".join(lines)



df_fire = pd.read_csv("interactions_aggregated.csv")
df_fire = df_fire[df_fire.cutoff == 3]
df_ref = pd.read_csv("interaction_references.csv")

df = df_fire[["molecule", "deltaE_mean", "deltaE_err"]].rename(columns={"deltaE_mean": "FiRE", "deltaE_err": "FiRE_err"})
df = df.merge(df_ref, on="molecule").set_index("molecule") * 1000

fig, axes = plt.subplots(1, 2, figsize=(8, 4.5), width_ratios=[1, 0.75])
ax_s22, ax_benz = axes

# Add inset axis in the bottom right of ax_s22
ax_scatter = ax_s22.inset_axes([0.68, 0.15, 0.3, 0.35])

ref_method = "CCSD(T)"

methods = [
    ("LapNet", COLOR_PALETTE[0], "s"),
    ("FiRE", COLOR_FIRE, "o")
]

for idx_method, (method, color, marker) in enumerate(methods):
    ax_scatter.errorbar(df[ref_method], df[method], yerr=df[method+"_err"], color=color, marker=marker, ls="none", capsize=3, label=method, ms=3)
    delta_to_ref = df[method] - df[ref_method]
    pos_barchart = np.arange(len(df)) - 0.2 + 0.4 * idx_method
    mae = np.mean(np.abs(delta_to_ref))
    print(f"MAE {method:<10}: {mae:.1f} mEh")

    # label = f"{method}, {mae:.1f} mE$_h$ MAE"
    label=method
    ax_s22.barh(pos_barchart, delta_to_ref, color=color, height=0.4, zorder=3, label=label)
    ax_s22.errorbar(delta_to_ref, pos_barchart, xerr=df[method+"_err"], color="k", marker="none", ls="none", capsize=3, zorder=4)

with open("s22_table.tex", "w") as f:
    f.write("{molecule} & {FiRE} & {LapNet} & {CCSD(T)}\\\\\n")
    f.write("\\midrule\n")
    for mol, row in df.iterrows():
        s_fire = format_value_with_error(row["FiRE"], row["FiRE_err"])
        s_lapnet = format_value_with_error(row["LapNet"], row["LapNet_err"])
        s_ccsdt = f"{row['CCSD(T)']:.2f}"
        mol = " ".join(mol.split("_")[1:])
        f.write(f"{mol} & {s_fire} & {s_lapnet} & {s_ccsdt}\\\\\n")



# for pos, n_steps in enumerate(df_fire.opt_step_end):
#     ax_s22.text(0.5, pos, f"{n_steps/1000:.0f}k steps", ha="left", va="center", color="black")

for y in [3.5, 6.5]:
    ax_s22.axhline(y, color="dimgray", ls="--", lw=0.5, alpha=0.5)

mol_names = [" ".join(m.split("_")[1:]) for m in df.index]
for i, m in enumerate(mol_names):
    mol_names[i] = m.replace(" T-shaped", "").replace("h-bonded", "H-bonded")
mol_names = [linebreak(m) for m in mol_names]


ax_s22.set_xlabel("$E_\\textrm{NN-VMC}$ - $E_\\textrm{CCSD(T)}$ " + MILLIHARTREE)
ax_s22.grid(False, axis="y")
ax_s22.yaxis.minorticks_off()
ax_s22.legend(loc="upper right")
ax_s22.set_yticks(np.arange(len(df)))
ax_s22.set_yticklabels(mol_names)
ax_s22.set_xlim([-6,5])
ax_s22.axvline(0, color="black", ls="-")
ax_s22.axvspan(-1.6, 1.6, color="dimgray", alpha=0.3)
ax_s22.invert_yaxis()

idx_start = 0
for interaction, n_mol in [("H-bonds", 4), ("Dispersion", 3), ("Mixed interactions", 4)]:
    x_pos = -9.8
    line = Line2D([x_pos, x_pos], [idx_start-0.4, idx_start+n_mol - 1 + 0.4], color="gray", lw=1)
    line.set_clip_on(False)
    ax_s22.add_line(line)
    ax_s22.text(x_pos-0.1, idx_start + n_mol/2 - 0.5, interaction, ha="right", va="center", rotation=90, color="gray")
    idx_start += n_mol

# draw a rectangle with dashed lines on ax_s22
ax_s22.add_patch(plt.Rectangle((-4.5, 8.5), 4.7, 1, fill=False, linestyle="--", color="gray", lw=1))

ax_scatter.set_xlabel("$E_\\textrm{CCSD(T)}$ " + MILLIHARTREE, labelpad=-1)
ax_scatter.set_ylabel("$E_\\textrm{NN-VMC}$ " + MILLIHARTREE, labelpad=0.5)
scatter_range = np.array([0, 40])
ax_scatter.plot(scatter_range, scatter_range, color="black", ls=":")
ax_scatter.fill_between(scatter_range, scatter_range - 1.6, scatter_range + 1.6, color="gray", alpha=0.3)
ax_scatter.set_xlim(scatter_range)
ax_scatter.set_ylim(scatter_range)
fig.tight_layout()



# Benzene dimer
# cutoffs = ["3.0", "Transfer7.0"]
cutoffs = ["3.0", "7.0"]
dists = [4.95, 10.0]

df_agg = pd.read_csv("../benzene_dimer/benzene_aggregated.csv")
df_agg = df_agg[df_agg["cutoff"].isin(cutoffs)]
df_agg["method"] = "FiRE $c=" + df_agg.cutoff.str.replace("Transfer", "") + "$"
df_agg["ref_type"] = "FiRE"

df_ref = pd.read_csv("../benzene_dimer/benzene_references.csv")
df_ref = df_ref[~df_ref.method.str.contains("UHF")]
df_agg = pd.concat([df_ref, df_agg], axis=0, ignore_index=True)

df_agg["method"] = df_agg["method"].str.replace(" (", "\n").str.replace("al)", "al")
df_agg["method"] = df_agg["method"].str.replace(" $", "\n$")
df_agg["method"] = df_agg["method"].str.replace(" / ", "\n")


colors = {
    "Experiment": "dimgray",
    "CCSD(T)": "k",
    "Ferminet": COLOR_PALETTE[1],
    "Psiformer": COLOR_PALETTE[2],
    "LapNet": COLOR_PALETTE[0],
    "FiRE": COLOR_FIRE,
}
color_exp = "k"
color_CC = "dimgray"
color_ferminet, color_psiformer, color_lapnet = COLOR_PALETTE[1:4]

exp_uncertainty = 0.8 # mHa; original value is 2 - 2.8 kcal/mol
df_agg["deltaE_mean"] *= -1
delta_E_exp = df_agg[df_agg["ref_type"] == "Experiment"]["deltaE_mean"].iloc[0]

with open("benzene_table.tex", "w") as f:
    f.write("{method} & {interaction energy}\\\\\n")
    f.write("\\midrule\n")
    for _, row in df_agg.iterrows():
        method = row["method"].replace("\n", ",")
        E, E_err = row["deltaE_mean"] * 1000, row["deltaE_err"] * 1000
        if np.isfinite(E_err):
            s = format_value_with_error(E, E_err)
        else:
            s = f"{E:.1f}"
        f.write(f"{method} & {s}\\\\\n")

ref_type_labeled = {r: False for r in df_agg["ref_type"].unique()}
for i, r in df_agg.iterrows():
    E, method, ref_type = r["deltaE_mean"] * 1000, r["method"], r["ref_type"]
    if method == "Experiment":
        color = colors["Experiment"]
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
    # label = ref_type if not ref_type_labeled[ref_type] else None
    ref_type_labeled[ref_type] = True
    ax_benz.barh(i, E, height=0.8, color=color, label=None, zorder=3)
    ax_benz.text(0.2 * np.sign(E), i, f"{E:.1f}", va="center", ha="left" if np.sign(E) > 0 else "right", color="white", zorder=4)

for label, color in colors.items():
    ax_benz.barh([np.nan], [np.nan], color=color, label=label)
ax_benz.set_yticks(range(len(df_agg)))
ax_benz.set_yticklabels(df_agg["method"])
ax_benz.invert_yaxis()
ax_benz.grid(False, axis="y")
ax_benz.yaxis.minorticks_off()
# ax.grid(alpha=0.5, axis="x")
ax_benz.axvline(delta_E_exp * 1000, color="k", linestyle="--", zorder=0)
ax_benz.axvline(0, color="k", linestyle="-", zorder=0)
ax_benz.legend(loc="lower right", ncol=1)

# for y in [2.5, 7.5]:
#     ax.axhline(y, color="k", ls="--", alpha=0.5, zorder=0, lw=0.5)
ax_benz.axvspan(delta_E_exp * 1000 - exp_uncertainty, delta_E_exp * 1000 + exp_uncertainty, color="k", alpha=0.1, zorder=0)
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
ax_s22.text(0.57, 0.97, "\\textbf{b)}", transform=fig.transFigure, va="top", ha="left")
savefig(fig, "interactions", bbox_inches=None)






