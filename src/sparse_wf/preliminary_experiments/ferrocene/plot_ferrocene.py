# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sparse_wf.plot_utils import get_outlier_mask, savefig, COLOR_FIRE, COLOR_PALETTE, cbs_extrapolate
from PIL import Image
import scienceplots
plt.style.use(['science', 'grid'])
geom_names = ["FerroceneCl_red_geom", "FerroceneCl_red_geom_charged"]

df_orca = pd.read_csv("orca_energies.csv")
cbs_energies = []
for method, extrapolation in [("DLPNO-CCSD(T)", (2,3)), ("CCSD(T)", 2), ("PBE0 D3BJ", (3,4))]:
    df_cbs = cbs_extrapolate(df_orca[df_orca["method"].isin([method, "UHF"])], extrapolate=extrapolation)
    pivot = df_cbs.pivot(index="method", columns="comment", values="E_final")
    E = (pivot[geom_names[1]] - pivot[geom_names[0]])[method]
    cbs_energies.append(E * 1000)

#%%

# Experiment
references = [
    ("Experiment", 258, "k"),
    ("B3LYP (Toma et al)", 238, COLOR_PALETTE[0]),
    ("PBE0", cbs_energies[2], COLOR_PALETTE[1]),
    ("CCSD(T) / DZ", cbs_energies[1], COLOR_PALETTE[2]),
    ("DLPNO-CCSD(T) / CBS", cbs_energies[0], COLOR_PALETTE[3]),
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
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
for ref, energy, color in references:
    ax.axhline(energy, color=color, linestyle="-", lw=1.5, label=None)
    if ref == "Experiment":
        ax.axhspan(energy - 1.6, energy + 1.6, color=color, alpha=0.2)
    va, offset = ("top", -0.5) if "B3LYP" in ref else ("bottom", 0.1)
    ax.text(100, energy + offset, ref, va=va, ha="right", color=color)

# Subsample for smaller pdf size
df_sub = df[::10]
ax.plot(df_sub.index / 1000, df_sub["deltaE_smooth"], label="FiRE", color=COLOR_FIRE)
# ax.plot(df.index / 1000, df["deltaE_smooth"], label="FiRE", color=COLOR_FIRE)
E_final = df["deltaE_smooth"][df["deltaE_smooth"].last_valid_index()]
print(f"FiRE (cutoff={CUTOFF}): {E_final:.1f} mHa")
# ax.axhline(E_final, color=color, linestyle="--")

ax.set_ylabel("ionization potential / mHa")
ax.set_xlabel("optimization step / k")
ax.set_ylim([230, 265])
ax.legend()


img = Image.open("FerroceneCl_red_geom_charged.png")
img = img.crop(img.getbbox())
image_ax = fig.add_axes([0.15, 0.1, 0.3, 0.3])
image_ax.imshow(img)
image_ax.axis("off")

savefig(fig, "ferrocene")

