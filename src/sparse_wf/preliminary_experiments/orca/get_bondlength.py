#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from scipy.optimize import curve_fit

def get_metadata(geom_name):
    m = re.match("cumulene_C(\d+)H4_(\d+)deg_.*_(\d+\.\d+)", geom_name)
    n_carbon = int(m.group(1))
    angle = int(m.group(2))
    bond_length = float(m.group(3))
    return n_carbon, angle, bond_length

def morse_potential(r, E0, r0, D, a):
    return (E0 + D) + D * (np.exp(-2 * a * (r - r0)) - 2 * np.exp(-a * (r - r0)))

df_all = pd.read_csv("energies.csv")
df_all["n_carbon"], df_all["angle"], df_all["bond_length"] = zip(*df_all.comment.apply(get_metadata))

plt.close("all")
calculated_bondlength = 2.53222
for method in ["PBE0", "UHF", "B3LYP"]:
    df = df_all[df_all.method == method]
    fit_results = []
    for n_carbon in df.n_carbon.unique():
        for angle in df.angle.unique():
            df_sub = df[(df.n_carbon == n_carbon) & (df.angle == angle)]
            popt, pcov = curve_fit(morse_potential, df_sub.bond_length, df_sub.E_final, p0=[-100, 2.5, 100, 1])
            E_calculated = morse_potential(calculated_bondlength, *popt)
            fit_results.append(dict(n_carbon=n_carbon, angle=angle, E_min_fit=popt[0], r_min_fit=popt[1], E_calculated=E_calculated, geom_correction=(popt[0]-E_calculated) * 1000))
    fit_results = pd.DataFrame(fit_results)
    df = df.merge(fit_results, on=["n_carbon", "angle"])

    df["deltaE"] = df["E_final"] - df["E_min_fit"]
    pivot = df.pivot_table(index="n_carbon", columns="angle", values=["geom_correction", "E_min_fit", "E_calculated"])
    pivot.columns = pivot.columns.swaplevel(0, 1)
    pivot = pivot[90] - pivot[0]

    pivot = pivot.reset_index()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    axes[0].axhline(calculated_bondlength, color="k", linestyle="--", label="calculated")
    sns.lineplot(ax=axes[0], data=df, x="n_carbon", y="r_min_fit", hue="angle", markers=True, marker="o", dashes=False)
    axes[1].plot(pivot.n_carbon, pivot.E_calculated * 1000, label=f"d={calculated_bondlength:.2f} a0")
    axes[1].plot(pivot.n_carbon, pivot.E_min_fit * 1000, label="d=opt")
    axes[1].axhline(0, color="k", linestyle="-")
    axes[1].legend()
    sns.lineplot(ax=axes[2], data=df, x="n_carbon", y="geom_correction", hue="angle", markers=True, marker="o", dashes=False)
    sns.lineplot(ax=axes[3], data=pivot, x="n_carbon", y="geom_correction", markers=True, marker="o", dashes=False)
    axes[3].set_ylim([0, None])

    titles = [
        "Optimal bond length",
        "Relative energy",
        "Abs. energy change\nthrough geom relaxation",
        "Rel. energy change\nthrough geom relaxation",
    ]
    for ax, title in zip(axes, titles):
        ax.set_xlim([0, None])
        ax.set_title(title)
        ax.set_xlabel("n_carbon")
        ax.grid(alpha=0.5)
    fig.suptitle(method)
    fig.tight_layout()
# sns.lineplot(data=df, x="bond_length", y="deltaE", hue="n_carbon", style="angle", markers=True, dashes=True)