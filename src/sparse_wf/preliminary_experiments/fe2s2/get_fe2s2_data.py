# %%
import wandb
import pandas as pd
import numpy as np
from sparse_wf.plot_utils import get_outlier_mask, extrapolate_relative_energy

api = wandb.Api()

runs = list(api.runs("tum_daml_nicholas/Fe2S2"))
runs = [r for r in runs if r.name.startswith("c5_") and "opt/E" in r.summary]

all_data = []
for r in runs:
    print(r.name)
    metadata = dict(geom=r.config["molecule_args"]["database_args"]["name"].split("_")[4])
    for h in r.scan_history(
        ["opt/step", "opt/E", "opt/E_std", "opt/update_norm", "opt/spring/last_grad_not_in_J_norm"], page_size=10_000
    ):
        all_data.append(metadata | h)
df = pd.DataFrame(all_data)
df["grad"] = np.sqrt(df["opt/update_norm"] ** 2 - df["opt/spring/last_grad_not_in_J_norm"] ** 2)
df = df.drop(columns=["opt/update_norm", "opt/spring/last_grad_not_in_J_norm"])
df.to_csv("fe2s2_energies.csv", index=False)

# %%
df = pd.read_csv("fe2s2_energies.csv")
n_eval_steps = 5000
pivot = df.pivot_table(index="opt/step", columns="geom", values=["opt/E", "grad"], aggfunc="mean").dropna()
is_outlier = pivot.apply(get_outlier_mask, axis=0).any(axis=1)
pivot = pivot[~is_outlier]
pivot_last = pivot["opt/E"].iloc[-n_eval_steps:]
pivot_smooth = pivot.rolling(500).mean()
E_ext = extrapolate_relative_energy(pivot_smooth.index, pivot_smooth["grad"].values.T, pivot_smooth["opt/E"].values.T)


df_agg = pivot_last.agg(["mean", lambda x: np.std(x) / np.sqrt(len(x))]).T
df_agg.columns = ["E", "E_err"]
df_agg["E_ext"] = E_ext
df_agg.to_csv("fe2s2_energies_aggregated.csv")
