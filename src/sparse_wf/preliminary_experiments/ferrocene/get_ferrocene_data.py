# %%
import numpy as np
import pandas as pd
import wandb
import re


def get_data(run, **metadata):
    data = []
    for h in run.scan_history(["opt/step", "opt/E", "opt/E_std", "opt/update_norm", "opt/spring/last_grad_not_in_J_norm"], page_size=10_000):
        data.append(h)
    df = pd.DataFrame(data)
    df = df.sort_values("opt/step")
    df = df.iloc[1:]  # Drop first step
    for k, v in metadata.items():
        df[k] = v
    return df

def get_outlier_mask(x):
    qlow = x.quantile(0.01)
    qhigh = x.quantile(0.99)
    med = x.median()
    included_range = 5 * (qhigh - qlow)
    is_outlier = (x < med - included_range) | (x > med + included_range)
    return is_outlier


name_template = f"HLR.*"
all_runs = wandb.Api().runs("tum_daml_nicholas/ferrocene")
runs = [r for r in all_runs if re.match(name_template, r.name)]

fire_data = []
for r in runs:
    print(r.name)
    charge = "charged" if "charged" in r.name else "neutral"
    df = get_data(
        r, cutoff=float(r.name.split("_")[1]), charge=charge
    )
    fire_data.append(df)
df_fire = pd.concat(fire_data)
df_fire = df_fire.sort_values(["charge", "cutoff", "opt/step"])
df_fire.to_csv("ferrocene_energies.csv", index=False)