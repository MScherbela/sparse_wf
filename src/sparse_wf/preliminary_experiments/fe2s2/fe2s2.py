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

energies = {
    "HS": [51.6, 77.14],
    "HFe": [87.8, 116],
    "HFe2": [76.1, 143],
}
methods = ["CCSD(T) CBS", "UHF"]
reference = pd.DataFrame(energies, index=methods)
#%%
dft_data = pd.read_csv('dft_data.csv', index_col=0) * 0.38088
dft_data = dft_data.loc[["PBE0", "B3LYP"]]
reference = pd.concat([dft_data, reference], axis=0)
#%%

api = wandb.Api()
ending = "_higher_lr"
pattern = re.compile(f"{ending}(?:_from\\d{{6}})?$")

runs = list(api.runs("tum_daml_nicholas/Fe2S2"))
runs = [r for r in runs if pattern.search(r.name) and 'opt/E' in r.summary]
runs = sorted(runs, key=lambda r: r.summary['_timestamp'])
_runs = defaultdict(list)
for r in runs:
    state = r.name.split("_df2-svp")[0].split("2023_")[1]
    _runs[state].append(r)
runs = dict(_runs)
#%%

our = jax.tree.map(
    lambda r: pd.DataFrame(r.scan_history(keys=["opt/E", "opt/step"])).set_index('opt/step').sort_index(), runs, is_leaf=lambda x: isinstance(x, wandb.apis.public.Run)
)
#%%
our = {
    k: pd.concat(v)
    for k, v in our.items()
}
#%%
window = 2000
fig, axes = plt.subplots(1, len(our) - 1, figsize=(10, 3))
axes = np.array([axes]).reshape(-1)
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
for ax, k in zip(axes, energies.keys()):
    diff = (our[k] - our["HC"])["opt/E"] * 1000
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
    for ref, name in reference[k].items():
        ax.axhline(name, label=ref, color=next(c_iter))
    ax.set_title(k)
    ax.set_xlabel("Step")
    ax.set_xlim(x.min(), x_max)
    ax.set_ylim(reference[k].min() - 2, reference[k].max() + 2)
    handles, labels = ax.get_legend_handles_labels()
    legend_dict = dict(zip(labels, handles))
axes[0].set_ylabel("Energy difference [mHa]")
fig.legend(legend_dict, loc="upper center", bbox_to_anchor=(0.5, 0), ncol=6)
plt.savefig("fe2s2.pdf", bbox_inches="tight")

# %%
min_length = min([len(v["opt/E"]) for v in our.values()])
total_E = np.array([v["opt/E"][:min_length].rolling(500).mean().iloc[-1] for k, v in our.items()])
our_diffs = pd.DataFrame((total_E[:, None] - total_E) * 1000, columns=our.keys(), index=our.keys())
our_diffs[reference.columns].loc[reference.columns]
# %%
ccsd = reference.T["CCSD(T) CBS"].to_numpy()
ccsd[0] = 0
pd.DataFrame(ccsd[:, None] - ccsd, columns=reference.columns, index=reference.columns)
# %%
