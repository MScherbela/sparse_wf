# %%
import wandb
import pandas as pd
import numpy as np
from sparse_wf.plot_utils import get_outlier_mask
import re

api = wandb.Api()
runs = api.runs("tum_daml_nicholas/interactions")
runs = [r for r in runs if re.match("(\d\d_|c5_\d\d_).*", r.name)]

benzene_dimer_runs = api.runs("tum_daml_nicholas/benzene")
benzene_dimer_runs = [r for r in benzene_dimer_runs if r.name.startswith("HLR_")]
runs = list(runs) + benzene_dimer_runs

gpu_hours_total = 0
data = []
for r in runs:
    print(r.name)
    mol_comment = r.config["molecule_args"]["database_args"]["comment"]
    mol_comment = mol_comment.replace("benzene_dimer_T_4.95A", "10_Benzene_dimer_T-shaped")
    mol_comment = mol_comment.replace("benzene_dimer_T_10.00A", "10_Benzene_dimer_T-shaped_Dissociated")

    metadata = dict(
        molecule=mol_comment.replace("_Dissociated", ""),
        geom="dissociated" if mol_comment.endswith("_Dissociated") else "equilibrium",
        cutoff=r.config["model_args"]["embedding"]["new"]["cutoff"],
    )
    gpu_hours_total += 4 * r.summary.get("_runtime", 0) / 3600

    history = []
    for h in r.scan_history(["opt/step", "opt/E", "opt/E_std", "opt/update_norm"], page_size=10_000):
        history.append(h | metadata)
    df = pd.DataFrame(history)
    if len(df):
        df = df.sort_values("opt/step").iloc[1:]
        data.append(df)
    else:
        print(f"No data for {r.name}")

print("Total GPU hours:", gpu_hours_total)

df_all = pd.concat(data)
df_all.to_csv("interaction_energies.csv", index=False)

#%%
df = pd.read_csv("interaction_energies.csv")

n_eval_steps = 5000
final_data = []


for mol in sorted(df.molecule.unique()):
    df_mol = df[df["molecule"] == mol]
    for cutoff in df_mol.cutoff.unique():
        pivot = df_mol[df_mol["cutoff"] == cutoff].pivot_table(index="opt/step", columns="geom", values="opt/E", aggfunc="mean")
        if len(pivot.columns) != 2:
            print("Not enough data for", mol)
            continue
        pivot["deltaE"] = pivot["dissociated"] - pivot["equilibrium"]
        pivot = pivot[pivot.deltaE.notna()]
        is_outlier = get_outlier_mask(pivot["deltaE"])
        pivot = pivot[~is_outlier]
        print(f"molecule={mol}, Outliers removed:", np.sum(is_outlier))
        pivot = pivot.iloc[-n_eval_steps:]
        final_data.append(
            dict(
                molecule=mol,
                cutoff=cutoff,
                E_equ=pivot["equilibrium"].mean(),
                E_diss=pivot["dissociated"].mean(),
                deltaE_mean=pivot["deltaE"].mean(),
                E_equ_err=pivot["equilibrium"].std() / np.sqrt(len(pivot)),
                E_diss_err=pivot["dissociated"].std() / np.sqrt(len(pivot)),
                deltaE_err=pivot["deltaE"].std() / np.sqrt(len(pivot)),
                opt_step_begin=pivot.index[0],
                opt_step_end=pivot.index[-1],
                n_steps_averaging=len(pivot),
            )
        )
df_final = pd.DataFrame(final_data)
df_final.to_csv("interactions_aggregated.csv", index=False)