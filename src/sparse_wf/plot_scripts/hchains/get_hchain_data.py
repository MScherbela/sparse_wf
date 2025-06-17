# %%
import wandb
import pandas as pd
import numpy as np


api = wandb.Api()
runs = [r for r in api.runs("tum_daml_nicholas/HChains")]

all_data = []
for r in runs:
    print(r.name)
    geom = r.config["molecule_args"]["database_args"]["comment"]
    dist = float(geom.split("_")[-1])
    cutoff = r.config["model_args"]["embedding"]["new"]["cutoff"]
    meta_data = dict(
        name=r.name,
        dist=dist,
        cutoff=cutoff,
    )
    full_history = [h for h in r.scan_history(keys=["opt/step", "opt/E", "opt/E_std", "opt/t_step"], page_size=10_000)]
    full_history = pd.DataFrame(full_history)
    for k, v in meta_data.items():
        full_history[k] = v
    full_history = full_history.sort_values("opt/step")
    full_history = full_history[full_history["opt/step"] <= 10_000]
    all_data.append(full_history)
df = pd.concat(all_data, axis=0, ignore_index=True)
df = df.sort_values(["cutoff", "dist", "opt/step"])
df.to_csv("hchain_energies.csv", index=False)
# %%
n_eval_steps = 2000
df = df[df["opt/step"] >= 10_000 - n_eval_steps]
pivot = df.groupby(["dist", "cutoff"]).agg(
    E_mean=("opt/E", "mean"),
    E_mean_sigma=("opt/E", lambda x: np.std(x) / np.sqrt(n_eval_steps)),
    E_std=("opt/E", "std"),
    t=("opt/t_step", "median"),
)
pivot = pivot.reset_index()
pivot.to_csv("hchain_energies_aggregated.csv", index=False)
