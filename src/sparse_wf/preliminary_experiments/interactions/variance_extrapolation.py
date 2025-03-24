#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sparse_wf.plot_utils import get_outlier_mask
from scipy.optimize import curve_fit
import scipy.stats
import itertools

def get_E_extrapolated(energies, variances):
    def model(v, E_inf, k):
        return E_inf + k * v
    popt, _ = curve_fit(model, variances, energies, [min(energies), 1])
    return popt

window = 5000
steps_min = 5000
plt.close("all")
fig, axes = plt.subplots(7,4, figsize=(12, 14))

df_all = pd.read_csv("interaction_energies.csv")
df_all["run"] = df_all["molecule"] + df_all["geom"] + df_all["cutoff"].astype(str)
molecules = sorted(df_all.molecule.unique())
# cutoffs = [3, 5]
cutoffs = [5]
geoms = ["equilibrium", "dissociated"]


final_data = []
idx_plot = 0
for mol, cutoff in itertools.product(molecules, cutoffs):
    df_mol = df_all[(df_all["molecule"] == mol) & (df_all["cutoff"] == cutoff)]
    df_group = df_mol.groupby("geom")["opt/step"].max()
    if len(df_group) != 2:
        print("Not enough runs for", mol, cutoff)
        continue
    steps_max = int(df_group.min())
    if steps_max < 15_000:
        print("Not enough data for", mol, cutoff)
        continue
    for geom in geoms:
        ax = axes.flatten()[idx_plot]
        ax.set_title(f"{mol}\n{geom},{cutoff}")
        idx_plot += 1
        df = df_mol[(df_mol["geom"] == geom) & (df_mol["opt/step"] >= steps_min) & (df_mol["opt/step"] <= steps_max)]
        is_outlier_std = get_outlier_mask(df["opt/E_std"], window_size=100, quantile=0.1)
        is_outlier_E = get_outlier_mask(df["opt/E"], window_size=100, quantile=0.1)
        is_outlier = is_outlier_E | is_outlier_std
        df = df[~is_outlier]

        df["E_smooth"] = df["opt/E"].rolling(window=2000).mean()
        df["var_smooth"] = (df["opt/E_std"]**2).rolling(window=2000).mean()
        df = df[~df.E_smooth.isna()]
        df_sub = df.iloc[::50]

        res = scipy.stats.linregress(df_sub.var_smooth, df_sub.E_smooth)
        E_inf, slope, E_inf_err, r_value = res.intercept, res.slope, res.intercept_stderr, res.rvalue


        var_range = np.array([np.min(df_sub.var_smooth), np.max(df_sub.var_smooth)])
        var_ratio = var_range[1] / var_range[0]

        scatter = ax.scatter(df_sub.var_smooth, df_sub.E_smooth, c=df_sub["opt/step"])
        ax.plot(var_range, var_range * slope + E_inf, color="k")
        # Add colorbar
        cbar = plt.colorbar(scatter)

        E_last = df["opt/E"].iloc[-window:].mean()
        # extrapolation = (E_last - E_inf) * 1000
        # print(f"{extrapolation} mHa")
        final_data.append(dict(molecule=df.molecule.iloc[0],
        geom=df.geom.iloc[0],
        cutoff=df.cutoff.iloc[0],
        E_last=E_last,
        E_inf=E_inf,
        E_inf_err=E_inf_err,
        var_ratio=var_ratio,
        r_value=r_value))
    fig.tight_layout()
#%%
df_final = pd.DataFrame(final_data)
df_final["do_trust"] = (df_final.r_value > 0.9) & (df_final.var_ratio > 1.2) & (df_final.E_inf < df_final.E_last)
df_final.loc[~df_final.do_trust, "E_inf"] = np.nan
pivot = df_final.pivot_table(index=["molecule", "cutoff"], columns="geom", values=["E_last", "E_inf"])
pivot = pivot.swaplevel(axis=1)
interaction = (pivot["dissociated"] - pivot["equilibrium"]) * 1000
# interaction["shift"] = interaction["E_inf"] - interaction["E_last"]
interaction["extrapolation"] = (pivot[("equilibrium", "E_last")] - pivot[("equilibrium", "E_inf")] ) * 1000

df_ref = pd.read_csv("interaction_references.csv").set_index("molecule") * 1000
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








