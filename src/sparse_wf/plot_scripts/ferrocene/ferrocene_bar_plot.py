# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sparse_wf.plot_utils import (
    COLOR_FIRE,
    COLOR_PALETTE,
    cbs_extrapolate,
    MILLIHARTREE,
)

plt.style.use(["science", "grid"])
geom_names = ["FerroceneCl_red_geom", "FerroceneCl_red_geom_charged"]

df_orca = pd.read_csv("orca_energies.csv")
df_cbs23 = cbs_extrapolate(df_orca, (2, 3), (3, 4), "UHF", "CBS34", "CBS23")
df_cbs2 = cbs_extrapolate(df_orca, 2, (3, 4), "UHF", "CBS34", "CBS2")
df_orca = pd.concat([df_orca, df_cbs23, df_cbs2], axis=0, ignore_index=True)
df_orca["method"] = df_orca["method"] + "/" + df_orca["basis_set"]
df_orca["charge"] = df_orca["comment"].apply(lambda c: "charged" if "charged" in c else "neutral")
pivot = df_orca.pivot_table(index="method", columns="charge", values="E_final")
deltaE_orca = (pivot["charged"] - pivot["neutral"]) * 1000

# FiRE data
n_steps_eval = 10_000
df_fire = pd.read_csv("ferrocene_energies.csv")
df_fire = df_fire[df_fire["cutoff"] == 3.0].pivot_table("opt/E", "opt/step", "charge")
df_fire = df_fire.dropna()
deltaE = (df_fire["charged"] - df_fire["neutral"]).iloc[-n_steps_eval:] * 1000

E_fire = deltaE.mean()
E_fire_std = deltaE.std() / np.sqrt(n_steps_eval)

# Experiment
exp_uncertainty = 1.65  # mHa; original resolution of photoemission is given as 0.04-0.05 eV; 0.045 eV = 1.65 mHa
E_fpa = deltaE_orca["DLPNO-CCSD(T)/CBS23"] + (deltaE_orca["CCSD(T)/CBS2"] - deltaE_orca["DLPNO-CCSD(T)/CBS2"])


energies = [
    ("Experiment", 258 - 0.2, "k"),  # 0.2 mHa ZPE correction at PBE0/D3BJ TZ level
    ("B3LYP\nTZ", 237.4, COLOR_PALETTE[0]),
    ("PBE0\nCBS", deltaE_orca["PBE0 D3BJ/CBS23"], COLOR_PALETTE[0]),
    ("CCSD(T)\nDZ", deltaE_orca["CCSD(T)/CBS2"], COLOR_PALETTE[1]),
    ("DLPNO\nCCSD(T)\nCBS", deltaE_orca["DLPNO-CCSD(T)/CBS23"], COLOR_PALETTE[1]),
    ("CCSD(T)\nFPA", E_fpa, COLOR_PALETTE[1]),
    ("FiRE", E_fire, COLOR_FIRE),
]
df = pd.DataFrame(energies, columns=["method", "E", "color"]).set_index("method")
df["error"] = df["E"] - df.loc["Experiment"]["E"]
df = df[df.index != "Experiment"]
x = np.arange(len(df))

fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=300)
for i, (method, row) in enumerate(df.iterrows()):
    energy_error, color = abs(row["error"]), row["color"]
    ax.barh([i], [energy_error], color=color)
    ax.text(0.3, i, f"{energy_error:.1f}", color="white", ha="left", va="center")
ax.barh([0], [0], color=COLOR_PALETTE[0], label="Hybrid DFT")
ax.barh([0], [0], color=COLOR_PALETTE[1], label="CCSD(T)")
ax.barh([0], [0], color=COLOR_FIRE, label="Ours")
ax.axvspan(0, exp_uncertainty, color="gray", zorder=-1, alpha=0.5, label="Exp. uncertainty")
ax.legend()
ax.axvline(0, color="k", lw=2)
ax.set_xlabel("error in ionization potential " + MILLIHARTREE, size=14)


ax.set_yticks(x, df.index)
ax.invert_yaxis()
ax.grid(False, axis="y")
