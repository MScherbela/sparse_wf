#%%
import numpy as np
import matplotlib.pyplot as plt
import wandb
import re
import pandas as pd
import json


geom_hashes = {"13b1974852bbdb41a3edc82b988422bf": 0, "5c2f1fbbfe629463e3a4eba88c136f28": 90}
DATA_DIR = "../../../../data"

with open(f"{DATA_DIR}/geometries.json") as f:
    all_geoms = json.load(f)
geom = all_geoms[list(geom_hashes.keys())[0]]
R = np.array(geom["R"])
dist_CC = np.linalg.norm(R[0] - R[1])
dist_HH = np.linalg.norm(R[-4] - R[-2])

energies_ref = pd.read_csv(f"{DATA_DIR}/energies.csv")
energies_ref = energies_ref[energies_ref.geom_hash.isin(geom_hashes.keys())]
energies_ref["angle"] = energies_ref.geom_hash.map(geom_hashes)
energies_ref = pd.pivot_table(energies_ref, values="E", index="model", columns="angle", aggfunc="mean")
energies_ref["E_rel"] = energies_ref[90] - energies_ref[0]
energies_ref = energies_ref.reset_index()

#%%


api = wandb.Api()
runs = api.runs("tum_daml_nicholas/cumulene")
# runs = [r for r in runs if re.match(r"C4H4_\d\ddeg_\d\.0", r.name)]

def get_metadata(run_name):
    n_carbon = int(re.findall(r"C(\d+)H4", run_name)[0])
    angle = int(re.findall(r".*_(\d*)deg", run_name)[0])
    cutoff = float(run_name.split("_")[-1])
    model_version = "v1" if run_name.startswith("C4H4") else "v2"
    return dict(n_carbon=n_carbon, angle=angle, cutoff=cutoff, model_version=model_version, name=run_name)

data = []
n_steps = 15_000
for r in runs:
    metadata = get_metadata(r.name)
    for h in r.scan_history(keys=["opt/step", "opt/E_smooth", "opt/E_std"], min_step=n_steps, page_size=10_000):
        if h["opt/step"] >= n_steps and h["opt/step"] <= (n_steps + 10):
            data.append(dict(
                E=h["opt/E_smooth"],
                E_std=h["opt/E_std"],
                E_sigma = h["opt/E_std"] / np.sqrt(n_steps * 0.1 * 2048) * 3,
                **metadata
                ))
            break
    else:
        print("Warning: Did not find final step")
        print(h)
df = pd.DataFrame(data)
df = df.sort_values(["model_version", "angle", "cutoff"])

#%%
pivot = pd.pivot_table(df, values=["E", "E_sigma"], index=["model_version", "cutoff"], columns="angle", aggfunc="mean")
pivot["E_rel"] = pivot[("E", 90)] - pivot[("E", 0)]
pivot["E_rel_sigma"] = np.sqrt(pivot[("E_sigma", 90)]**2 + pivot[("E_sigma", 0)]**2)
pivot = pivot.reset_index()

E_hf = {0: -153.251964583315, 90: -153.107759385032}
E_rel_hf = E_hf[90] - E_hf[0]

#%%
plt.close("all")
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
cutoffs = sorted(pivot.cutoff.unique())
for ind_model, model_version in enumerate(["v1", "v2"]):
    ls = "-" if model_version == "v2" else "--"
    df_model = pivot[pivot.model_version == model_version]
    for ind_phi, phi in enumerate([0, 90]):
        color = f"C{ind_phi}"
        E = df_model[("E", phi)]
        E_sigma = df_model[("E_sigma", phi)]
        axes[0].errorbar(df_model.cutoff, E, E_sigma, color=color, ls=ls, label=f"Sparse WF {model_version}, phi={phi}", capsize=3)

    axes[1].errorbar(df_model.cutoff, df_model.E_rel * 1000, df_model.E_rel_sigma*1000, color="black", ls=ls, marker='o', capsize=3, label=f"Sparse WF {model_version}")

for ind_ref, ref_data in energies_ref.iterrows():
    color = ["brown", "gray"][ind_ref]
    axes[0].axhline(ref_data[0], color=color, linestyle="-", label=ref_data["model"], alpha=0.7, zorder=-1)
    axes[0].axhline(ref_data[90], color=color, linestyle="-", label=None, alpha=0.7, zorder=-1)
    axes[1].axhline(ref_data["E_rel"] * 1000, color=color, linestyle="-", label=ref_data["model"], alpha=0.7, zorder=-1)


for ax in axes:
    ax.set_xlabel("Cutoff / bohr")
    ax.legend()
    ax.grid(alpha=0.2, ls='--')

axes[0].set_ylabel("Energy / Ha")
axes[0].set_title("Absolute energy")
axes[1].set_title("Relative energy")
axes[1].set_ylabel("Energy difference / mHa")
axes[1].set_ylim([110, 145])
fig.suptitle(f"Cumulene, C4H4, {n_steps//1000}k steps\nCC-bond: {dist_CC:.1f} $a_0$, HH-distance: {dist_HH:.1f} $a_0$")
fig.tight_layout()
fig.savefig("/home/mscherbela/ucloud/results/cumulene_C4H4.png", bbox_inches="tight")
