# %%
import wandb
import pandas as pd
import matplotlib.pyplot as plt
import re

from collections import defaultdict
import numpy as np
import jax


plt.style.use(["science", "grid"])

energies_kcal_per_mol = {
    "naphthalene": [61.0, -3.4, 64.4, 68.0, 62.6, 65.8, 66.2, 70.6, 64.7, 62.2, 67.1],
}
methods = [
    "exp",
    "ZPE",
    "ZPE-corr'd exp",
    "AFQMC",
    "UB3LYP",
    "CCSD(T)/FPA",
    "B3LYP/pp-RPA",
    "GAS-pDFT (FP-1)",
    "GAS-pDFT (WFP-3)",
    "ACI-DSRG-MRPT2",
    "DMRG-pDFT",
]
reference = pd.DataFrame(energies_kcal_per_mol, index=methods) * 1.6
# %%
api = wandb.Api()


def get_runs(project, regex):
    pattern = re.compile(regex)
    all_runs = list(api.runs(project))
    all_runs = [r for r in all_runs if pattern.search(r.name) and "opt/E" in r.summary]
    all_runs = sorted(all_runs, key=lambda r: r.summary["_timestamp"])
    runs = defaultdict(list)
    for r in all_runs:
        runs[r.name.split("_")[2]].append(r)
    return runs


runs = {
    "baseline": get_runs("tum_daml_nicholas/acene", ".*naphthalene.*_splus_new(?:_from\\d{6})?$"),
    "1det": get_runs("tum_daml_nicholas/ablation", ".*naphthalene.*_1_ablation(?:_from\\d{6})?$"),
    "16det": get_runs("tum_daml_nicholas/ablation", ".*naphthalene.*_16_ablation(?:_from\\d{6})?$"),
    "nojastrow": get_runs("tum_daml_nicholas/ablation", ".*naphthalene.*_False_ablation(?:_from\\d{6})?$"),
}
# %%

energies = jax.tree.map(
    lambda r: pd.DataFrame(r.scan_history(keys=["opt/E", "opt/step"])).set_index("opt/step").sort_index(),
    runs,
    is_leaf=lambda x: isinstance(x, wandb.apis.public.Run),
)
# %%
energies = {k: {s: pd.concat(d) for s, d in v.items()} for k, v in energies.items()}
# %%
full_df = pd.concat(
    [
        d.rename(columns={"opt/E": (k, s)})[~d.index.duplicated(keep="first")]
        for k, v in energies.items()
        for s, d in v.items()
    ],
    axis=1,
)
full_df = full_df.sort_index()
tuples = full_df.transpose().index
new_columns = pd.MultiIndex.from_tuples(tuples, names=["Molecule", "State"])
full_df.columns = new_columns
full_df.to_csv("ablation.csv")
# %%
full_df = pd.read_csv("ablation.csv", header=[0, 1], index_col=0)
full_df.columns = full_df.columns.set_levels(
    ["16 determinants", "1 determinant", "default", "no Attention Jastrow"], level=0
)
# %%
fig, axes = plt.subplots(1, 3, figsize=(10, 2), sharex=True)
axes = np.array([axes]).reshape(-1)
window = 5000

keys = ["default", "1 determinant", "16 determinants", "no Attention Jastrow"]
ls = ["-", "--", "-.", ":"]
take_every = 100

colors = ["e15759", "4e79a7", "f28e2b", "59a14f", "9c755f", "b07aa1", "76b7b2", "ff9da7", "edc948", "bab0ac"]
colors = [f"#{c}" for c in colors]


def smooth(x):
    x = x.dropna()
    mean = x.rolling(window=window, min_periods=1).mean()
    mad = np.abs(x - mean).mean()
    x = x[np.abs(x - mean) < 10 * mad]
    return x.dropna().rolling(window=window, min_periods=1).mean().dropna().iloc[::take_every]


# Plot absolute singlet
ax = axes[0]
c_iter = iter(colors)
for m, l in zip(keys, ls):
    if "singlet" not in full_df[m].columns:
        continue
    diffs = full_df[m]["singlet"]
    diffs = smooth(diffs)
    ax.plot(diffs, l, label=m, color=next(c_iter), lw=1.2)
ax.set_ylim(-61.595, -61.56)
ax.set_title("Singlet")
ax.set_ylabel("Energy [Ha]")
ax.set_xlabel("Optimization step")

# Plot absolute triplet
ax = axes[1]
c_iter = iter(colors)
for m, l in zip(keys, ls):
    if "triplet" not in full_df[m].columns:
        continue
    diffs = full_df[m]["triplet"]
    diffs = smooth(diffs)
    ax.plot(diffs, l, label=m, color=next(c_iter), lw=1.2)
ax.set_ylim(-61.495, -61.46)
ax.set_title("Triplet")
ax.set_xlabel("Optimization step")

# Plot relative difference
ax = axes[2]
c_iter = iter(colors)
for m, l in zip(keys, ls):
    if "triplet" not in full_df[m].columns:
        continue
    diffs = full_df[m]["triplet"] - full_df[m]["singlet"]
    diffs = smooth(diffs)
    ax.plot(diffs, l, label=m, color=next(c_iter), lw=1.2)
ax.axhline(64.4 * 1.6 / 1000, color="black", linestyle="--", label="exp")
ax.set_ylim(0.1, 0.108)
ax.set_title("Triplet - Singlet")
ax.set_xlabel("Optimization step")
handles, labels = ax.get_legend_handles_labels()
legend_dict = dict(zip(labels, handles))
fig.subplots_adjust(wspace=0.2)
fig.legend(legend_dict.values(), legend_dict.keys(), loc="lower center", bbox_to_anchor=(0.5, 1), ncol=6)
plt.savefig("ablation.pdf", bbox_inches="tight")
# %%
# %%
full_df

fig, axes = plt.subplots(2, 2, figsize=(8, 3), sharex=True)
axes = axes.ravel()
colors = ["4e79a7", "f28e2b", "59a14f", "9c755f", "e15759", "b07aa1", "76b7b2", "ff9da7", "edc948", "bab0ac"]
colors = [f"#{c}" for c in colors]
axes[0].plot(smooth(full_df["baseline"]["singlet"]), "-", label="SWANN", color=colors[0])
axes[0].plot(smooth(full_df["baseline"]["triplet"]), "--", color=colors[0])
axes[0].plot(smooth(full_df["1det"]["singlet"]), "-", color=colors[1])
axes[0].plot(smooth(full_df["1det"]["triplet"]), "--", color=colors[1])
axes[0].plot(smooth(full_df["16det"]["singlet"]), "-", color=colors[2])
axes[0].plot(smooth(full_df["16det"]["triplet"]), "--", color=colors[2])
axes[0].set_ylim(-61.6, -61.46)

axes[2].plot(smooth(full_df["baseline"]["triplet"] - full_df["baseline"]["singlet"]), "-", color=colors[0])
axes[2].plot(smooth(full_df["1det"]["triplet"] - full_df["1det"]["singlet"]), "-", color=colors[1])
axes[2].plot(smooth(full_df["16det"]["triplet"] - full_df["16det"]["singlet"]), "-", color=colors[2])
axes[2].set_ylim(0.1, 0.11)


axes[1].plot(smooth(full_df["baseline"]["singlet"]), "-", color=colors[0])
axes[1].plot(smooth(full_df["baseline"]["triplet"]), "--", color=colors[0])
axes[1].plot(smooth(full_df["nojastrow"]["singlet"]), "-", color=colors[4])
axes[1].plot(smooth(full_df["nojastrow"]["triplet"]), "--", color=colors[4])
axes[1].set_ylim(-61.6, -61.46)


axes[3].plot(smooth(full_df["baseline"]["triplet"] - full_df["baseline"]["singlet"]), "-", color=colors[0])
axes[3].plot(smooth(full_df["nojastrow"]["triplet"] - full_df["nojastrow"]["singlet"]), "-", color=colors[4])
axes[3].set_ylim(0.1, 0.11)


# %%
runs["baseline"]["singlet"][0].history()["opt/t_step"].min()
jax.tree.map(lambda x: x.history()["opt/t_step"].min(), runs)

# %%


# %%
full_df
# %%
window = 5000
for m in full_df.keys().levels[0]:
    if "triplet" not in full_df[m].columns:
        continue
    diffs = full_df[m]["triplet"] - full_df[m]["singlet"]
    diffs = diffs.dropna()
    diffs = diffs.rolling(window).mean()
    diffs = diffs.dropna() * 1000
    plt.plot(diffs - reference["naphthalene"]["ZPE-corr'd exp"], label=m)
plt.ylim(-5, 5)
plt.legend()
plt.ylabel(r"$(E_\text{triplet} - E_\text{singlet}) - \Delta_\text{exp}$ [mHa]")
plt.xlabel("Optimization step")
plt.savefig("ablation_convergence.pdf", bbox_inches="tight")
plt.show()
# %%
window = 1000
for m in full_df.keys().levels[0]:
    if "singlet" not in full_df[m].columns:
        continue
    diffs = full_df[m]["singlet"]
    diffs = diffs.dropna()
    diffs = diffs.rolling(window).mean()
    plt.plot(diffs, label=m)
plt.ylim(-61.6, -61.56)
plt.legend()
# %%
window = 1000
for m in full_df.keys().levels[0]:
    if "triplet" not in full_df[m].columns:
        continue
    diffs = full_df[m]["triplet"]
    diffs = diffs.dropna()
    diffs = diffs.rolling(window).mean()
    plt.plot(diffs, label=m)
plt.ylim(-61.5, -61.46)
plt.legend()
# %%
