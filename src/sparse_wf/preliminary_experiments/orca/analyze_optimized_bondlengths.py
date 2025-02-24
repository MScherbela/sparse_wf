#%%
import numpy as np
from sparse_wf.geometry import Geometry
import matplotlib.pyplot as plt
import os
import pandas as pd

plt.close("all")
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
base_path = "/storage/scherbelam20/runs/orca/cumulene_bondlength_opt_noise/manual/cartesian"

for n_carbon, ax in zip([6, 8, 10, 12, 16, 20, 24, 30], axes.flatten()):
    geom_files = [
        (f"C{n_carbon}H4_0deg_singlet", f"{base_path}/cumulene_C{n_carbon}H4_0deg_singlet/orca.xyz"),
        (f"C{n_carbon}H4_90deg_singlet", f"{base_path}/cumulene_C{n_carbon}H4_90deg_singlet/orca.xyz"),
        (f"C{n_carbon}H4_90deg_triplet", f"{base_path}/cumulene_C{n_carbon}H4_90deg_triplet/orca.xyz"),
    ]

    for idx, (name, fname) in enumerate(geom_files):
        color = f"C{idx}"
        if not os.path.exists(fname):
            print(f"Skipping {name}")
            continue
        orca_output = open(fname.replace(".xyz", ".out")).read()
        is_converged = "THE OPTIMIZATION HAS CONVERGED" in orca_output
        ls = ":" if not is_converged else "-"

        geom = Geometry.from_xyz(fname, comment=name)
        dR = np.diff(geom.R[:-4], axis=0)
        bond_lengths = np.linalg.norm(dR, axis=1)
        ax.plot(bond_lengths, label=name, color=color, ls=ls)
        ax.legend()

    for d, label in zip([2.90, 2.53, 2.273], ["Single", "Double", "Triple"]):
        ax.axhline(d, color="k", linestyle="--")
        ax.text(0, d, label, ha="left", va="bottom")
    ax.set_xlabel("bond index")
    ax.set_ylabel("bond length / Å")
    ax.set_title(f"C{n_carbon}H4")
fig.tight_layout()

#%%
df = pd.read_csv("energies.csv")
df["geom"] = df["angle"].astype(str) + "_" + df["spin_state"]
df = df[df.n_carbon.isin([2, 4, 6, 8, 12, 24])]
pivot = df.pivot(index="n_carbon", columns="geom", values="E_final")
# pivot = pivot - pivot["0_singlet"]

fig, ax = plt.subplots()
for geom in ["90_singlet", "90_triplet"]:
    delta_E = (pivot[geom] - pivot["0_singlet"]) * 1000
    ax.plot(delta_E.index, delta_E, label=geom, marker="o")
ax.legend()
ax.axhline(0, color="k")
ax.set_xlabel("n_carbon")
ax.set_ylabel("ΔE / mHa")
ax.set_title("Relative energy of relaxed geometries")

