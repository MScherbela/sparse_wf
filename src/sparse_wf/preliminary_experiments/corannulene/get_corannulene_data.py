# %%
import wandb
import pandas as pd
import numpy as np
from sparse_wf.plot_utils import get_outlier_mask, extrapolate_relative_energy

api = wandb.Api()

runs = list(api.runs("tum_daml_nicholas/corannulene"))
runs = [r for r in runs if r.name.startswith("c5_") and "opt/E" in r.summary]

all_data = []
for r in runs:
    print(r.name)
    metadata = dict(geom="diss" if "dissociated" in r.config["molecule_args"]["database_args"]["comment"] else "equ")
    for h in r.scan_history(
        ["opt/step", "opt/E", "opt/E_std", "opt/spring/natgrad_norm"], page_size=10_000
    ):
        all_data.append(metadata | h)
df = pd.DataFrame(all_data)
df = df.rename(columns={"opt/spring/natgrad_norm": "grad"})
df = df.sort_values(["geom", "opt/step"])
df.to_csv("corannulene_energies.csv", index=False)
