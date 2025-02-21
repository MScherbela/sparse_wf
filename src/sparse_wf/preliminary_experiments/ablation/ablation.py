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
    all_runs = [r for r in all_runs if pattern.search(r.name) and 'opt/E' in r.summary]
    all_runs = sorted(all_runs, key=lambda r: r.summary['_timestamp'])
    runs = defaultdict(list)
    for r in all_runs:
        runs[r.name.split("_")[2]].append(r)
    return runs

runs = {
    'baseline': get_runs("tum_daml_nicholas/acene", ".*naphthalene.*_splus_new(?:_from\\d{6})?$"),
    '1det': get_runs("tum_daml_nicholas/ablation", ".*naphthalene.*_1_ablation(?:_from\\d{6})?$"),
    '16det': get_runs("tum_daml_nicholas/ablation", ".*naphthalene.*_16_ablation(?:_from\\d{6})?$"),
    'nojastrow': get_runs("tum_daml_nicholas/ablation", ".*naphthalene.*_False_ablation(?:_from\\d{6})?$"),
}
# %%

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
# %%
full_df = pd.concat([
    d.rename(columns={'opt/E': (k, s)})[~d.index.duplicated(keep='first')]
    for k, v in energies.items()
    for s, d in v.items()
], axis=1)
full_df = full_df.sort_index()
tuples = full_df.transpose().index
new_columns = pd.MultiIndex.from_tuples(tuples, names=['Molecule', 'State'])
full_df.columns = new_columns
full_df.to_csv('ablation.csv')
# %%
full_df = pd.read_csv('ablation.csv', header=[0, 1], index_col=0)
full_df
# %%
window = 1000
for m in full_df.keys().levels[0]:
    if 'triplet' not in full_df[m].columns:
        continue
    diffs = full_df[m]['triplet'] - full_df[m]['singlet']
    diffs = diffs.dropna()
    diffs = diffs.rolling(window).mean()
    diffs = diffs.dropna() * 1000
    plt.plot(diffs - reference['naphthalene']['ZPE-corr\'d exp'], label=m)
plt.ylim(-5, 5)
plt.legend()
plt.ylabel(r"$(E_\text{triplet} - E_\text{singlet}) - \Delta_\text{exp}$ [mHa]")
plt.xlabel("Optimization step")
plt.savefig('ablation_convergence.pdf', bbox_inches='tight')
plt.show()
# %%
