# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sparse_wf.plot_utils import get_outlier_mask, savefig, COLOR_FIRE, COLOR_PALETTE, cbs_extrapolate, format_value_with_error, MILLIHARTREE
from PIL import Image
import scienceplots
plt.style.use(['science', 'grid'])
geom_names = ["FerroceneCl_red_geom", "FerroceneCl_red_geom_charged"]

df_orca = pd.read_csv("orca_energies.csv")
df_cbs23 = cbs_extrapolate(df_orca, (2, 3), (3,4), "UHF", "CBS34", "CBS23")
df_cbs2 = cbs_extrapolate(df_orca, 2, (3,4), "UHF", "CBS34", "CBS2")
df_orca = pd.concat([df_orca, df_cbs23, df_cbs2], axis=0, ignore_index=True)
df_orca["method"] = df_orca["method"] + "/" + df_orca["basis_set"]
df_orca["charge"] = df_orca["comment"].apply(lambda c: "charged" if "charged" in c else "neutral")
pivot = df_orca.pivot_table(index="method", columns="charge", values="E_final")
deltaE_orca = (pivot["charged"] - pivot["neutral"]) * 1000


# Experiment
E_fpa = deltaE_orca["DLPNO-CCSD(T)/CBS23"] + (deltaE_orca["CCSD(T)/CBS2"] - deltaE_orca["DLPNO-CCSD(T)/CBS2"])

references = [
    ("Experiment", 258 - 0.2, "k"), # 0.2 mHa ZPE correction at PBE0/D3BJ TZ level
    ("B3LYP / TZ", 237.4, COLOR_PALETTE[0]),
    ("PBE0 / CBS", deltaE_orca["PBE0 D3BJ/CBS23"], COLOR_PALETTE[1]),
    ("DLPNO-CCSD(T) / CBS", deltaE_orca["DLPNO-CCSD(T)/CBS23"], COLOR_PALETTE[3]),
    ("CCSD(T) / DZ", deltaE_orca["CCSD(T)/CBS2"], COLOR_PALETTE[2]),
    ("CCSD(T) / FPA", E_fpa, COLOR_PALETTE[4]),
]

window = 10_000
CUTOFF = 3.0
df_fire = pd.read_csv("ferrocene_energies.csv")
df_fire = df_fire[df_fire["cutoff"] == CUTOFF]

df = df_fire.pivot_table(index="opt/step", columns="charge", values="opt/E")
df = df.ffill(limit=10)
df["deltaE"] = (df["charged"] - df["neutral"]) * 1000
is_outlier = get_outlier_mask(df["deltaE"])
df.loc[is_outlier, "deltaE"] = np.nan
df["deltaE"] = df["deltaE"].ffill(limit=10)
df["deltaE_smooth"] = df["deltaE"].rolling(window=window, min_periods=window//10).mean()

plt.close("all")
fig, ax = plt.subplots(1, 1, figsize=(2, 3))

exp_uncertainty = 1.65 # mHa; original resolution of photoemission is given as 0.04-0.05 eV
# 0.045 eV = 1.65 mHa
for ref, energy, color in references:
    ax.axhline(energy, color=color, linestyle="-", lw=1, label=None)
    if ref == "Experiment":
        ax.axhspan(energy - 1.6, energy + 1.6, color=color, alpha=0.2)
    legend_below = ("B3LYP" in ref) or ("FPA" in ref)
    va, offset = ("top", -0.7) if legend_below else ("bottom", 0.1)
    ax.text(100, energy + offset, ref, va=va, ha="right", color=color, fontsize=8)

# Subsample for smaller pdf size
df_sub = df[::10]
ax.plot(df_sub.index / 1000, df_sub["deltaE_smooth"], label="FiRE", color=COLOR_FIRE)
# ax.plot(df.index / 1000, df["deltaE_smooth"], label="FiRE", color=COLOR_FIRE)
E_final = df["deltaE_smooth"][df["deltaE_smooth"].last_valid_index()]
print(f"FiRE (cutoff={CUTOFF}): {E_final:.1f} mHa")
# ax.axhline(E_final, color=color, linestyle="--")

ax.set_ylabel("ionization potential " + MILLIHARTREE)
ax.set_xlabel("optimization step / k", labelpad=-0.2)
ax.set_ylim([230, 265])
ax.legend()


img = Image.open("FerroceneCl_red_geom_charged.png")
img = img.crop(img.getbbox())
image_ax = fig.add_axes([0.15, 0.1, 0.3, 0.3])
image_ax.imshow(img)
image_ax.axis("off")
ax.text(-0.28, 1.025, "\\textbf{e)}", transform=ax.transAxes, va="top", ha="left")

savefig(fig, "ferrocene")


with open("ferrocene.tex", "w") as f:
    f.write("{method} & {ionization potential}\\\\\n")
    f.write("\\midrule\n")
    for method, E, _ in references:
        f.write(f"{method} & {E:.1f}\\\\\n")
    E_final = df["deltaE"].iloc[-window:].values
    s_fire = format_value_with_error(np.mean(E_final), np.std(E_final) / np.sqrt(window))
    f.write(f"FiRE & {s_fire}\\\\\n")

