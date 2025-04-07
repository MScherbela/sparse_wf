# %%
import wandb
import pandas as pd
import numpy as np
from sparse_wf.plot_utils import get_outlier_mask, extrapolate_relative_energy
import re
#%%

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
    for h in r.scan_history(
        ["opt/step", "opt/E", "opt/E_std", "opt/update_norm", "opt/spring/last_grad_not_in_J_norm"]
    ):
        history.append(metadata | h)
    df = pd.DataFrame(history)
    if len(df):
        df = df.sort_values("opt/step").iloc[1:]
        data.append(df)
    else:
        print(f"No data for {r.name}")

print("Total GPU hours:", gpu_hours_total)
df_all = pd.concat(data, ignore_index=True)
df_all["grad"] = np.sqrt(df_all["opt/update_norm"]**2 - df_all["opt/spring/last_grad_not_in_J_norm"]**2)
df = df_all
df = df_all.drop(columns=["opt/update_norm", "opt/spring/last_grad_not_in_J_norm"])
df = df.sort_values(["cutoff", "molecule", "geom", "opt/step"])
df.to_csv("interaction_energies.csv", index=False)


# %%
df = pd.read_csv("interaction_energies.csv")
df["var"] = df["opt/E_std"]**2

n_eval_steps = 5000
smoothing = 500
final_data = []

for mol in sorted(df.molecule.unique()):
    df_mol = df[df["molecule"] == mol]
    for cutoff in df_mol.cutoff.unique():
        pivot = (
            df_mol[df_mol["cutoff"] == cutoff]
            .pivot_table(index="opt/step", columns="geom", values=["opt/E", "grad", "var"], aggfunc="mean")
            .dropna()
        )
        if (len(pivot.columns) != 6) or (len(pivot) < n_eval_steps):
            print("Not enough data for", mol)
            continue
        is_outlier = pivot.apply(get_outlier_mask, axis=0).any(axis=1)
        pivot = pivot[~is_outlier]
        print(f"molecule={mol}, Outliers removed:", np.sum(is_outlier))
        pivot_eval = pivot["opt/E"].iloc[-n_eval_steps:]
        pivot_eval["deltaE"] = pivot_eval["dissociated"] - pivot_eval["equilibrium"]
        if smoothing > 1:
            pivot_smooth = pivot.rolling(smoothing).mean().iloc[smoothing::smoothing//10]
        else:
            pivot_smooth = pivot

        E_ext_diss, E_ext_equ = extrapolate_relative_energy(
            pivot_smooth.index,
            pivot_smooth["grad"].values.T,
            pivot_smooth["opt/E"].values.T,
            min_frac_step=0.5,
            method="same_slope",
        )
        final_data.append(
            dict(
                molecule=mol,
                cutoff=cutoff,
                E_equ=pivot_eval["equilibrium"].mean(),
                E_diss=pivot_eval["dissociated"].mean(),
                deltaE_mean=pivot_eval["deltaE"].mean(),
                E_equ_err=pivot_eval["equilibrium"].std() / np.sqrt(n_eval_steps),
                E_diss_err=pivot_eval["dissociated"].std() / np.sqrt(n_eval_steps),
                deltaE_err=pivot_eval["deltaE"].std() / np.sqrt(n_eval_steps),
                opt_step_begin=pivot_eval.index[0],
                opt_step_end=pivot_eval.index[-1],
                E_diss_extrapolated=E_ext_diss,
                E_equ_extrapolated=E_ext_equ,
                deltaE_extrapolated=E_ext_diss - E_ext_equ,
            )
        )
df_final = pd.DataFrame(final_data)
df_final.to_csv("interactions_aggregated.csv", index=False)
print(df_final)
