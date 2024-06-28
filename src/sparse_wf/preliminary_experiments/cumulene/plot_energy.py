#%%
import numpy as np
import matplotlib.pyplot as plt
import wandb
import re
import pandas as pd
import json

with open("../../../../data/geometries.json") as f:
    all_geoms = json.load(f)
geom = all_geoms["13b1974852bbdb41a3edc82b988422bf"]
R = np.array(geom["R"])
dist_CC = np.linalg.norm(R[0] - R[1])
dist_HH = np.linalg.norm(R[-4] - R[-2])

api = wandb.Api()
runs = api.runs("tum_daml_nicholas/cumulene")
runs = [r for r in runs if re.match(r"C4H4_\d\ddeg_\d\.0", r.name)]    

data = []
n_steps = 20_000
for r in runs:
    print(r.name)
    angle = 90 if "90" in r.name else 0
    cutoff = float(r.name.split("_")[-1])
    for h in r.scan_history(keys=["opt/step", "opt/E_smooth", "opt/E_std"], min_step=n_steps, page_size=10_000):
        if h["opt/step"] >= n_steps and h["opt/step"] <= (n_steps + 10):
            data.append(dict(
                E=h["opt/E_smooth"],
                E_std=h["opt/E_std"],
                E_sigma = h["opt/E_std"] / np.sqrt(n_steps * 0.1 * 2048) * 3,
                name=r.name,
                angle=angle,
                cutoff=cutoff,
                ))
            break
    else:
        print("Warning: Did not find final step")
        print(h)
df = pd.DataFrame(data)
df = df.sort_values(["angle", "cutoff"])

df_rel = pd.pivot_table(df, values=["E", "E_sigma"], index="cutoff", columns="angle")
df_rel["E_rel"] = df_rel[("E", 90)] - df_rel[("E", 0)]
df_rel["E_rel_sigma"] = np.sqrt(df_rel[("E_sigma", 90)]**2 + df_rel[("E_sigma", 0)]**2)
df_rel.reset_index(inplace=True)

E_hf = {0: -153.251964583315, 90: -153.107759385032}
E_rel_hf = E_hf[90] - E_hf[0]

#%%
plt.close("all")
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
cutoffs = sorted(df.cutoff.unique())
for i, phi in enumerate([0, 90]):
    color = f"C{i}"
    df_phi = df[df.angle == phi]
    axes[0].errorbar(df_phi.cutoff, df_phi.E, yerr=df_phi.E_sigma, color=color, label=f"phi={phi}", capsize=3)
axes[1].errorbar(df_rel.cutoff, df_rel.E_rel * 1000, yerr=df_rel.E_rel_sigma*1000, color="black", marker='o', capsize=3, label="Sparse WF")
axes[1].axhline(E_rel_hf * 1000, color="gray", linestyle="--", label="HF STO-6G")

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
