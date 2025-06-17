# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sparse_wf.plot_utils import get_outlier_mask


def fit_with_joint_slope(x1, E1, x2, E2):
    y = np.concat([E1, E2])
    X = np.stack(
        [
            np.concat((np.ones_like(x1), np.zeros_like(x2))),
            np.concat((np.zeros_like(x1), np.ones_like(x2))),
            np.concat([x1, x2]),
        ],
        axis=1,
    )
    return np.linalg.lstsq(X, y)[0]


df_all = pd.read_csv("interaction_energies.csv")
df_all["run"] = df_all["molecule"] + df_all["geom"] + df_all["cutoff"].astype(str)
molecules = sorted(df_all.molecule.unique())
# molecules = [m for m in molecules if m[:2] in ["02"]]
# molecules = [m for m in molecules if m[:2] in ["02", "03", "04", "11"]]

df_all = df_all[(df_all["cutoff"] == 5) & df_all["molecule"].isin(molecules)]
geoms = ["equilibrium", "dissociated"]

# %%
plt.close("all")
fig, axes = plt.subplots(4, 3, figsize=(12, 14))
smoothing = 2000
steps_min = 20_000
eval_steps = 5000
methods = ["last", "var", "grad", "grad_mom"]

data_final = []
for ax, mol in zip(axes.flat, molecules):
    print(mol)
    df_mol = df_all[df_all["molecule"] == mol].copy()
    df_mol["E"] = df_mol["opt/E"]
    df_mol["var"] = df_mol["opt/E_std"] ** 2
    df_mol["grad_mom"] = df_mol["opt/update_norm"]
    df_mol["grad"] = np.sqrt(df_mol["opt/update_norm"] ** 2 - df_mol["opt/spring/last_grad_not_in_J_norm"] ** 2)
    pivot = df_mol.pivot_table(
        index="opt/step", columns="geom", values=["E", "var", "grad", "grad_mom"], aggfunc="mean"
    )
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

    pivot = pivot.rolling(window=smoothing).mean()
    pivot = pivot.iloc[smoothing :: smoothing // 10]

    for method in methods[1:]:
        E1, E2 = [pivot[("E", g)] for g in geoms]
        x1, x2 = [pivot[(method, g)] for g in geoms]
        fit_coeffs = fit_with_joint_slope(x1, E1, x2, E2)
        energies, slope = fit_coeffs[:2], fit_coeffs[2]
        for g, E in zip(geoms, energies):
            data_final.append(dict(molecule=mol, geom=g, E=E, method=method))

        if method != "grad":
            continue

        x_range = np.array([pivot[method].min().min(), pivot[method].max().max()])
        for g, E_inf in zip(geoms, energies):
            ax.scatter(pivot[(method, g)], pivot[("E", g)], label=g)
            ax.plot(x_range, E_inf + slope * x_range, color="k")

    ax.set_title(mol)

# %%
df_final = pd.DataFrame(data_final)
pivot = df_final.pivot_table(index=["molecule", "method"], columns="geom", values="E").reset_index()
pivot["FiRE"] = pivot["dissociated"] - pivot["equilibrium"]

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
