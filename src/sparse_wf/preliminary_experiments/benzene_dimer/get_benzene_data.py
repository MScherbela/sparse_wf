# %%
import wandb
import pandas as pd
import re
import numpy as np
from sparse_wf.plot_utils import get_outlier_mask

api = wandb.Api()
all_runs = api.runs("tum_daml_nicholas/benzene")
runs = [r for r in all_runs if re.match("HLR.*", r.name)]
runs = [r for r in runs if "eval" not in r.name]

data = []
for r in runs:
    print(r.name)
    dist = float(re.search(r"(\d+\.\d+)A", r.name).group(1))
    cutoff = r.name.split("_")[1]
    if r.name.startswith("HLRTransfer"):
        cutoff = "Transfer" + cutoff
    metadata = dict(
        dist=dist,
        cutoff=str(cutoff),
        run_name=r.name,
    )

    history = []
    for h in r.scan_history(["opt/step", "opt/E"], page_size=10_000):
        history.append(h | metadata)
    df = pd.DataFrame(history)
    if len(df):
        df = df.sort_values("opt/step").iloc[1:] # drop first step
        data.append(df)
    else:
        print(f"No data for {r.name}")

df_all = pd.concat(data)
df_all.to_csv("benzene_energies.csv", index=False)

#%%
df = pd.read_csv("benzene_energies.csv")
df["cutoff"] = df["cutoff"].astype(str)
final_data = []

n_eval_steps = 5000
for cutoff in df["cutoff"].unique():
    pivot = df[df["cutoff"] == cutoff]
    pivot = pivot.pivot_table(index="opt/step", columns="dist", values="opt/E", aggfunc="mean")
    if len(pivot.columns) < 2:
        print(f"Not enough data for, c={cutoff}")
        continue
    pivot["deltaE"] = pivot[4.95] - pivot[10.0]
    pivot = pivot[pivot.deltaE.notna()]
    is_outlier = get_outlier_mask(pivot["deltaE"])
    print(f"c={cutoff}, Outliers removed:", np.sum(is_outlier))
    pivot = pivot[~is_outlier]
    pivot = pivot.iloc[-n_eval_steps:]
    E_mean = pivot.mean()
    E_err = pivot.std() / np.sqrt(len(pivot))

    final_data.append(
        dict(
            cutoff=cutoff,
            E495=E_mean[4.95],
            E495_err=E_err[4.95],
            E10=E_mean[10.0],
            E10_err=E_err[10.0],
            deltaE_mean=E_mean["deltaE"],
            deltaE_err=E_err["deltaE"],
            opt_step_begin=pivot.index[0],
            opt_step_end=pivot.index[-1],
            n_steps_averaging=len(pivot),
        )
    )
df_final = pd.DataFrame(final_data)
df_final.to_csv("benzene_aggregated.csv", index=False)