#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sparse_wf.plot_utils import get_outlier_mask
from scipy.optimize import curve_fit

def fit_with_joint_slope(x1, E1, x2, E2):
    y = np.concat([E1, E2])
    X = np.stack([
        np.concat((np.ones_like(x1), np.zeros_like(x2))),
        np.concat((np.zeros_like(x1), np.ones_like(x2))),
        np.concat([x1, x2]),
        ], axis=1)
    return np.linalg.lstsq(X, y)[0]


def get_E_extrapolated(energies, variances):
    def model(v, E_inf, k):
        return E_inf + k * v
    popt, _ = curve_fit(model, variances, energies, [min(energies), 1])
    return popt


df_all = pd.read_csv("interaction_energies.csv")
df_all["run"] = df_all["molecule"] + df_all["geom"] + df_all["cutoff"].astype(str)
molecules = sorted(df_all.molecule.unique())
# molecules = [m for m in molecules if m[:2] in ["02"]]
# molecules = [m for m in molecules if m[:2] in ["02", "03", "04", "11"]]

df_all = df_all[(df_all["cutoff"] == 3) & df_all["molecule"].isin(molecules)]
geoms = ["equilibrium", "dissociated"]

#%%
plt.close("all")
fig, axes = plt.subplots(4, 3, figsize=(12, 14))
smoothing = 500
steps_min = 10000
eval_steps = 5000

data_final = []
for ax, mol in zip(axes.flat, molecules):
    print(mol)
    df_mol = df_all[df_all["molecule"] == mol].copy()
    df_mol["E"] = df_mol["opt/E"]
    df_mol["var"] = df_mol["opt/E_std"]**2
    df_mol["grad"] = df_mol["opt/update_norm"]
    pivot = df_mol.pivot_table(index="opt/step", columns="geom", values=["E", "var", "grad"], aggfunc="mean")
    pivot = pivot[~pivot["E"].isnull().any(axis=1)]
    pivot = pivot[pivot.index >= steps_min - smoothing]

    def get_outlier(x):
        return get_outlier_mask(x, window_size=100, quantile=0.1)

    is_outlier_std = pivot["var"].apply(get_outlier)
    is_outlier_E = pivot["E"].apply(get_outlier)
    is_outlier = (is_outlier_E | is_outlier_std).any(axis=1)
    pivot = pivot[~is_outlier]
    E_final = pivot["E"].iloc[-eval_steps:].mean()
    for g in geoms:
        data_final.append(dict(molecule=mol, geom=g, E=E_final[g], method="last"))

    pivot = pivot.rolling(window=500).mean()
    pivot = pivot.iloc[smoothing::smoothing//10]

    for method in ["var", "grad"]:
        E1, E2 = [pivot[("E", g)] for g in geoms]
        x1, x2 = [pivot[(method, g)] for g in geoms]
        energies = fit_with_joint_slope(x1, E1, x2, E2)[:2]
        for g, E in zip(geoms, energies):
            data_final.append(dict(molecule=mol, geom=g, E=E, method=method))

    var_range = np.array([pivot["grad"].min(), pivot["grad"].max()])
    for g in geoms:
        ax.scatter(pivot[("grad", g)], pivot[("E", g)], label=g)
    ax.set_title(mol)

#%%
df_final = pd.DataFrame(data_final)
# df_final["do_trust"] = (df_final.r_value > 0.9) & (df_final.var_ratio > 1.2) & (df_final.E_inf < df_final.E_last)
# df_final.loc[~df_final.do_trust, "E_inf"] = np.nan
pivot = df_final.pivot_table(index=["molecule", "method"], columns="geom", values="E").reset_index()
pivot["FiRE"] = (pivot["dissociated"] - pivot["equilibrium"])


# pivot = pivot.swaplevel(axis=1)
# interaction = (pivot["dissociated"] - pivot["equilibrium"]) * 1000
# # interaction["shift"] = interaction["E_inf"] - interaction["E_last"]
# interaction["extrapolation"] = (pivot[("equilibrium", "E_last")] - pivot[("equilibrium", "E_inf")] ) * 1000

df_ref = pd.read_csv("interaction_references.csv")[["molecule", "CCSD(T)"]]
pivot = pivot.merge(df_ref, "left", "molecule")
pivot["deviation"] = (pivot["FiRE"] - pivot["CCSD(T)"]) * 1000

pivot = pivot.pivot_table(index="molecule", columns="method", values="deviation")
df_mae = pivot.abs().mean(axis=0)
print(pivot)
print(df_mae)

fig, ax = plt.subplots(1, 1)
pivot.plot.barh(ax=ax)
ax.invert_yaxis()
ax.axvline(color="k")
ax.set_xlabel("Error vs CCSD(T) / mHa")

#%%

interaction = interaction.join(df_ref[["LapNet", "CCSD(T)"]])
# interaction_error = interaction - interaction["CCSD(T)"]
# Subtract CCSD(T) column from all others
interaction_error = interaction.sub(interaction["CCSD(T)"], axis=0)
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', 100)
print(interaction)
print(interaction_error)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for ax, cutoff in zip(axes, [3, 5]):
    df_cut = interaction_error.reset_index()
    df_cut = df_cut[df_cut.cutoff == cutoff]
    molecules = sorted(df_cut.molecule.unique())
    methods = ["LapNet", "E_last", "E_inf"]
    labels = ["LapNet", "FiRE: last 5k", "FiRE: var ext."]
    for idx_method, (method, label) in enumerate(zip(methods, labels)):
        ax.barh(np.arange(len(molecules))-0.3 + 0.3*idx_method, df_cut[method], label=label, height=0.3)
    ax.set_yticks(np.arange(len(molecules)))
    ax.set_yticklabels(molecules)
    ax.invert_yaxis()
    ax.set_title(f"cutoff = {cutoff}")
    ax.axvline(0, color="black", label="CCSD(T)")
    ax.legend()
# axes[0].set_xlim([-20, 20])
fig.tight_layout()

interaction_error_all = interaction_error[interaction_error.isna().sum(axis=1) == 0]
mae = interaction_error_all.reset_index().groupby(["cutoff"])[["E_inf", "E_last", "LapNet"]].apply(lambda x: np.abs(x).mean())
print(mae)








