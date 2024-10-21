#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

df = pd.read_csv("plot_data/final_energy.csv")
df.sort_values(["n_atoms", "d_short", "d_long", "cutoff"], inplace=True)
d_short_values = np.sort(df.d_short.unique())

MAX_CUTOFF_FOR_FIT = 6

def exponential(cutoff, E_inf, Ecorr, alpha):
    return E_inf + Ecorr * np.exp(-alpha * cutoff**3)

def fit_model(func, n_params, cutoff, E, E_std):
    popt, _ = curve_fit(func, cutoff, E, sigma=E_std, p0=[np.min(E), 0.1, 0.1])
    return popt

df_extrapolated = []
for d_short in d_short_values:
    for d_long in df.d_long.unique():
        for n_atoms in df.n_atoms.unique():
            df_fit = df[(df.n_atoms == n_atoms) & (df.d_short == d_short) & (df.d_long == d_long) & (df.cutoff <= MAX_CUTOFF_FOR_FIT)]
            if len(df_fit) == 0:
                continue
            E_inf = fit_model(exponential, 3, df_fit.cutoff, df_fit.E, df_fit.E_std)[0]
            df_extrapolated.append(dict(d_short=d_short, d_long=d_long, n_atoms=n_atoms, E=E_inf, cutoff=1000))
df_extrapolated = pd.DataFrame(df_extrapolated)
df = pd.concat([df, df_extrapolated], axis=0, ignore_index=True)

df_d_min = df[df.d_short == 1.3].copy().rename(columns={"E": "E_min"})
df = df.merge(df_d_min[["n_atoms", "cutoff", "E_min"]], on=["n_atoms", "cutoff"], how="left")
df["E_rel"] = 1000 * (df["E"] - df["E_min"])



plt.close("all")
fig, axes = plt.subplots(2, len(d_short_values), figsize=(14, 8))

ax_relative = axes[-1]

E_extrapolated = []
for ind_dist, d_short in enumerate(d_short_values):
    ax_abs = axes[0, ind_dist]
    ax_rel = axes[1, ind_dist]
    df_exp = df[df.d_short == d_short]

    df_fit = df_exp[df_exp.cutoff < 7]
    df_test = df_exp[(df_exp.cutoff >= 7) & (df_exp.cutoff < 1000)]

    ax_abs.errorbar(df_fit.cutoff, df_fit.E, yerr=df_fit.E_std, fmt="o", color="navy", label="Fit data")
    ax_abs.errorbar(df_test.cutoff, df_test.E, yerr=df_test.E_std, fmt="o", color="C1", label="'Test' data")

    fit_params = fit_model(exponential, 3, df_fit.cutoff, df_fit.E, df_fit.E_std)
    cutoff_fit = np.linspace(2.8, 10, 100)
    E_fit = exponential(cutoff_fit, *fit_params)
    E_inf = fit_params[0]
    E_extrapolated.append(E_inf)
    ax_abs.plot(cutoff_fit, E_fit, ls="-", color="C0", label="Exponential fit")
    ax_abs.axhline(E_inf, color="C0", ls="--")

    ax_abs.set_title(f"d_short={d_short:.1f}")
    ax_abs.set_xlabel("Cutoff / bohr")
    ax_abs.set_ylabel("Energy / Ha")
    ax_abs.legend(loc="upper right")

    ax_rel.plot(df_exp[df_exp.cutoff < 1000].cutoff, df_exp[df_exp.cutoff < 1000].E_rel, marker="o", color="dimgray", label="Calculations")
    ax_rel.axhline(df_exp[df_exp.cutoff == 1000].E_rel.values[0], color="C0", ls="--", label="rel energy from\nextrapolated abs energy")
    ax_rel.set_xlabel("Cutoff / bohr")
    ax_rel.set_ylabel("$\\Delta E$ to d=1.3 / mHa")
    ax_rel.legend(loc="lower right")


fig.suptitle("Cutoff extrapolation for H10, Total chain length = 15 bohr")
fig.tight_layout()
fig.savefig("/home/mscherbela/ucloud/results/cutoff_extrapolation_H10.png", dpi=300, bbox_inches="tight")