#%%
import wandb
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
import re

from collections import defaultdict
import numpy as np
import jax


plt.style.use(["science", "grid"])

energies_kcal_per_mol = {
    "naphthalene": [61.0, -3.4, 64.4, 68.0, 62.6, 65.8, 66.2, 70.6, 64.7, 62.2, 67.1],
    "anthracene": [43.1, -2.3, 45.4, 46.2, 41.8, 48.2, 45.7, 45.5, 43.1, 43.2, 46.1],
    "tetracene": [29.4, -1.8, 31.2, 34.0, 27.7, 33.5, 32.1, 33.6, 28.8, 28.3, 31.6],
    "pentacene": [19.8, -1.5, 21.3, 25.2, 17.9, 25.3, 22.6, 25.4, 20.5, 18.0, 22.6],
    "hexacene": [12.4, -1.3, 13.7, np.nan, 10.9, 17.7, np.nan, 19.7, 15.0, 11.4, 16.8],
    # "heptacene": [np.nan, np.nan, np.nan, np.nan, 5.6, 13.4, np.nan, 16.5, 10.0, 7.7, 14.3]
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
#%%

api = wandb.Api()
runs = ["tum_daml_nicholas/acene/kiq6t01c", "tum_daml_nicholas/acene/5lsw3pnf"]
# ending = "_ccecpccpvdz"
ending = "_splus_new"
pattern = re.compile(f"{ending}(?:_from\\d{{6}})?$")
# ending = "_b3lyp"

runs = list(api.runs("tum_daml_nicholas/acene"))
runs = [r for r in runs if pattern.search(r.name) and 'opt/E' in r.summary]
runs = sorted(runs, key=lambda r: r.summary['_timestamp'])
_runs = defaultdict(lambda: defaultdict(list))
for r in runs:
    _runs[r.name.split("_")[1]][r.name.split("_")[2]].append(r)
runs = {k: dict(v) for k, v in _runs.items()}
#%%
energies = jax.tree.map(
    lambda r: pd.DataFrame(r.scan_history(keys=["opt/E", "opt/step"])).set_index('opt/step').sort_index(), runs, is_leaf=lambda x: isinstance(x, wandb.apis.public.Run)
)
#%%
energies = {
    k: {
        s: pd.concat(d)
        for s, d in v.items()
    }
    for k, v in energies.items()
}

#%%
window = 2000
fig, axes = plt.subplots(1, len(energies_kcal_per_mol), figsize=(10, 3))
axes = np.array([axes]).reshape(-1)
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
ref_data = reference.loc[["AFQMC", "CCSD(T)/FPA", "ACI-DSRG-MRPT2"]]
for ax, k in zip(axes, energies_kcal_per_mol.keys()):
    states = energies[k]
    print(k)
    if "triplet" not in states:
        continue
    diff = (states["triplet"] - states["singlet"])["opt/E"] * 1000
    diff = diff.dropna()
    avg = diff.rolling(window, min_periods=1).mean()
    mad = np.abs(diff - avg).mean()
    diff = diff[np.abs(diff - avg) < 10 * mad]
    diff = diff.rolling(window, min_periods=1).mean()
    diff = diff.dropna()
    x = diff.index
    c_iter = iter(colors)
    x_max = x.max() * 1.05
    ax.plot(x, diff, label="SWANN (c=3)", color=next(c_iter))
    exp = reference[k]["ZPE-corr'd exp"]
    ax.axhline(exp, label="exp", color="black")
    ax.fill_between(
        [x.min(), x_max], exp - 1.6, exp + 1.6, color="black", alpha=0.1, label="exp $\pm$ chem. acc "
    )
    for ref, name in ref_data[k].items():
        ax.axhline(name, label=ref, color=next(c_iter))
    ax.set_title(k)
    ax.set_xlabel("Step")
    ax.set_xlim(x.min(), x_max)
    ax.set_ylim(ref_data[k].min() - 5, ref_data[k].max() + 5)
    handles, labels = ax.get_legend_handles_labels()
    legend_dict = dict(zip(labels, handles))
axes[0].set_ylabel("Energy difference [mHa]")
fig.legend(legend_dict, loc="upper center", bbox_to_anchor=(0.5, 0), ncol=6)
plt.savefig("acene.pdf", bbox_inches="tight")

# %%
