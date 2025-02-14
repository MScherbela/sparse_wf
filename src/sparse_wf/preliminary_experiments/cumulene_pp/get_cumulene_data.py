#%%
import wandb
import pandas as pd
import itertools
import numpy as np

api = wandb.Api()
runs = [r for r in api.runs("tum_daml_nicholas/cumulene_pp") if r.name.startswith("HLR")]

all_data = []
for r in runs:
    print(r.name)
    geom = r.config["molecule_args"]["database_args"]["comment"]
    angle = 90 if "90" in geom else 0
    n_carbon = int(geom.split("_")[1].replace("C", "").replace("H4", ""))
    meta_data = dict(
        name=r.name,
        n_carbon=n_carbon,
        angle=angle,
        cutoff=r.config["model_args"]["embedding"]["new"]["cutoff"],
    )
    full_history = [
        h for h in r.scan_history(keys=["opt/step", "opt/E", "opt/E_std"], page_size=10_000)
    ]
    full_history = pd.DataFrame(full_history)
    for k,v in meta_data.items():
        full_history[k] = v
    full_history = full_history.sort_values("opt/step")
    full_history = full_history.iloc[1:]  # Drop first step
    all_data.append(full_history)
df = pd.concat(all_data, axis=0, ignore_index=True)
df = df.sort_values(["n_carbon", "angle", "opt/step"])
df.to_csv("cumulene_pp_energies.csv", index=False)

#%%
df = pd.read_csv("cumulene_pp_energies.csv")
final_data = []

n_eval_steps = 2000
for cutoff, n_carbon in itertools.product(df["cutoff"].unique(), df["n_carbon"].unique()):
    pivot = df[(df["cutoff"] == cutoff) & (df["n_carbon"] == n_carbon)]
    pivot = pivot.pivot_table(index="opt/step", columns="angle", values="opt/E", aggfunc="mean")
    if len(pivot.columns) < 2:
        print(f"Not enough data for, c={cutoff}, n={n_carbon}")
        continue
    pivot["deltaE"] = pivot[90] - pivot[0]
    pivot = pivot[pivot.deltaE.notna()]
    spread = pivot["deltaE"].quantile(0.9) - pivot["deltaE"].quantile(0.1)
    mask_min = pivot["deltaE"].median() - 3 * spread
    mask_max = pivot["deltaE"].median() + 3 * spread
    outlier_mask = pivot["deltaE"].between(mask_min, mask_max)
    print(f"c={cutoff}, n={n_carbon}, Outliers removed:", np.sum(~outlier_mask))
    pivot = pivot[outlier_mask]
    pivot = pivot.iloc[-n_eval_steps:]
    E_mean = pivot.mean()
    E_err = pivot.std() / np.sqrt(len(pivot))

    delta_E = pivot[90] - pivot[0]
    delta_E = delta_E.dropna()
    delta_E = delta_E.iloc[-n_eval_steps:]
    final_data.append(
        dict(
            cutoff=cutoff,
            n_carbon=n_carbon,
            E0=E_mean[0],
            E0_err=E_err[0],
            E90=E_mean[90],
            E90_err=E_err[90],
            deltaE_mean=E_mean["deltaE"],
            deltaE_err=E_err["deltaE"],
            opt_step_begin=pivot.index[0],
            opt_step_end=pivot.index[-1],
            n_steps_averaging=len(pivot),
        )
    )
df_final = pd.DataFrame(final_data)
df_final.to_csv("cumulene_pp_aggregated.csv", index=False)