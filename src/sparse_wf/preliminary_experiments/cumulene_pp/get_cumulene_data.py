#%%
import wandb
import pandas as pd
import re

api = wandb.Api()
runs = [r for r in api.runs("tum_daml_nicholas/cumulene_pp") if r.name.startswith("2024-11-18")]

all_data = []
for r in runs:
    print(r.name)
    # 2024-11-18_cumulene_C4H4_90deg_triplet_from019859
    match = re.search(r"2024-11-18_cumulene_C(\d+)H4_(\d+)deg", r.name)
    meta_data = dict(
        name=r.name,
        n_carbon=int(match[1]),
        angle=int(match[2]),
        cutoff=3.0,
    )
    full_history = [
        h for h in r.scan_history(keys=["opt/step", "opt/E", "opt/E_std"], page_size=10_000)
    ]
    full_history = pd.DataFrame(full_history)
    full_history = full_history.rename(columns={"opt/E": "E", "opt/E_std": "E_std", "opt/t_step": "t", "opt/step": "step"})
    for k,v in meta_data.items():
        full_history[k] = v
    all_data.append(full_history)
df = pd.concat(all_data, axis=0, ignore_index=True)
df.to_csv("2024-11-18_cumulene_pp_energies.csv", index=False)
