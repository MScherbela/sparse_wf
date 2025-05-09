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
    run_data = []
    for h in r.scan_history(
        ["opt/step", "opt/E", "opt/E_std", "opt/update_norm", "opt/spring/last_grad_not_in_J_norm"], page_size=10_000
    ):
        run_data.append(metadata | h)
    run_data = pd.DataFrame(run_data)
    if r.name == "c5_zhai_et_al_2023_HFe_df2-svp_from065444":
        # crashed run
        run_data = run_data[run_data["opt/step"] <= 70_000]
    all_data.append(run_data)
df = pd.concat(all_data, axis=0, ignore_index=True)
df["grad"] = np.sqrt(df["opt/update_norm"] ** 2 - df["opt/spring/last_grad_not_in_J_norm"] ** 2)
df = df.drop(columns=["opt/update_norm", "opt/spring/last_grad_not_in_J_norm"])
df.to_csv("fe2s2_energies.csv", index=False)


# %%
df = pd.read_csv("fe2s2_energies.csv")
n_eval_steps = 5000
smoothing_ext = 500
pivot = df.pivot_table(index="opt/step", columns="geom", values=["opt/E", "grad"], aggfunc="mean").dropna()
pivot["grad"] = pivot["grad"] ** 2
is_outlier = pivot.apply(get_outlier_mask, axis=0)
pivot = pivot.mask(is_outlier, np.nan).ffill(limit=5)
pivot_last = pivot["opt/E"].iloc[-n_eval_steps:]
pivot_smooth = (
    pivot.rolling(smoothing_ext).mean().iloc[smoothing_ext :: smoothing_ext // 10]
)
E_ext = extrapolate_relative_energy(pivot_smooth.index,
pivot_smooth["grad"].values.T,
pivot_smooth["opt/E"].values.T
    , method="same_slope", min_frac_step=0.6
)


df_agg = pivot_last.agg(["mean", lambda x: np.std(x) / np.sqrt(len(x))]).T
df_agg.columns = ["E", "E_err"]
df_agg["E_ext"] = E_ext
df_agg.to_csv("fe2s2_energies_aggregated.csv")