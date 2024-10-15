#%%
import wandb
import re
import pandas as pd
import numpy as np

api = wandb.Api()
runs = [r for r in api.runs("tum_daml_nicholas/cutoff_extrapolation") if r.name.startswith("H10_HChain10_")]

full_data = []
final_data = []
for r in runs:
    name = r.name
    print(name)
    # Parse names like this: H10_HChain10_1.50_1.50_9
    match = re.search(r"HChain(\d+)_(\d+\.\d+)_(\d+\.\d+)_(\d+)", name)
    metadata = dict(
        n_atoms = int(match.group(1)),
        d_short = float(match.group(2)),
        d_long = float(match.group(3)),
        cutoff = float(match.group(4)),
    )
    full_history = [
        h for h in r.scan_history(keys=["opt/step", "opt/E", "opt/E_std", "opt/t_step"], page_size=10_000)
    ]
    full_history = pd.DataFrame(full_history)
    full_history = full_history.rename(columns={"opt/E": "E", "opt/E_std": "E_std", "opt/t_step": "t", "opt/step": "steps"})
    for k,v in metadata.items():
        full_history[k] = v
    full_data.append(full_history)

    final_history = full_history[(full_history.steps >= 15_000) & (full_history.steps <= 25_000)]
    df_final = dict(
        E = final_history["E"].mean(),
        E_std = final_history["E"].std() / np.sqrt(len(final_history)),
        **metadata
    )
    final_data.append(df_final)

df = pd.concat(full_data, axis=0, ignore_index=True)
df.to_csv("plot_data/full_history.csv", index=False)

df_final = pd.DataFrame(final_data)
df_final.to_csv("plot_data/final_energy.csv", index=False)

