# %%
import numpy as np
import matplotlib.pyplot as plt
import wandb
import pandas as pd
import re

api = wandb.Api()
runs = [r for r in api.runs("tum_daml_nicholas/cumulene") if r.name.startswith("09-26")]

def get_geometry(name):
    # find string of the form C16H4_90deg within name and extract the values
    match = re.search(r"C(\d+)H4_(\d+)deg", name)
    n_carbon = int(match.group(1))
    angle = int(match.group(2))
    return n_carbon, angle

data = []
for r in runs:
    name = r.name
    print(name)
    tokens = name.split("_")
    n_carbon, angle = get_geometry(name)
    meta_data = dict(
        jastrow="attention",
        n_carbon=n_carbon,
        angle=angle,
        # cutoff=float(name.split("_")[-1]),
        cutoff=3.0,
        batch_size=4096,
    )
    full_history = [
        h for h in r.scan_history(keys=["opt/step", "opt/E", "opt/E_std", "opt/t_step"], page_size=10_000)
    ]
    full_history = pd.DataFrame(full_history)
    full_history = full_history.rename(columns={"opt/E": "E", "opt/E_std": "E_std", "opt/t_step": "t", "opt/step": "steps"})
    for k,v in meta_data.items():
        full_history[k] = v
    data.append(full_history)

df_all = pd.concat(data, axis=0, ignore_index=True)
df_all.sort_values(["n_carbon", "angle", "cutoff", "steps"], inplace=True)
df_all.to_csv("cumulene_09-26.csv", index=False)
