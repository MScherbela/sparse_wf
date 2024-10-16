# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df_all = pd.concat([pd.read_csv(f) for f in ["cumulene_09-10.csv", "cumulene_09-26.csv"]], ignore_index=True)
fig_fname = "/home/mscherbela/ucloud/results/09-26_cumulene"
df_all = pd.pivot_table(
    df_all, values=("E", "t"), index=["n_carbon", "cutoff", "jastrow", "steps"], columns="angle"
).reset_index()
df_all.columns = ["".join([str(c) for c in col]).strip() for col in df_all.columns.values]
#%%
def smoothen_timeseries(df, columns, window=5000, outlier_threshold=2.0):
    # Remove large outliers from restart
    include = np.ones(len(df), dtype=bool)
    for column in columns:
        baseline = df[column].rolling(window=100).mean()
        include_column = baseline.isnull() | ((baseline - df[column]).abs() < outlier_threshold)
        include = include & include_column
    df = df[include].copy()

    # Running average
    for column in columns:
        df[column] = df[column].rolling(window=window, min_periods=100).mean()
        df[f"{column}_var"] = df[column].rolling(window=window, min_periods=100).var() / window
    df = df[df.steps % 100 == 0]
    return df

df = []
for n_carbon in df_all.n_carbon.unique():
    for cutoff in df_all.cutoff.unique():
        for jastrow in df_all.jastrow.unique():
            df_exp = df_all[(df_all.n_carbon == n_carbon) & (df_all.cutoff == cutoff) & (df_all.jastrow == jastrow)]
            if len(df_exp) == 0:
                continue
            window = 5000 if n_carbon >= 16 else 1000
            df.append(smoothen_timeseries(df_exp, ["E0", "E90"], window, outlier_threshold=2.0))
df = pd.concat(df, ignore_index=True)

#%%
df["E_rel"] = 1000 * (df["E90"] - df["E0"])
df["E_rel_std"] = 1000 * np.sqrt(df["E0_var"] + df["E90_var"])
df["model"] = df.apply(lambda x: f"cutoff={x.cutoff:.1f} ({x.jastrow} jas)", axis=1)

df_final = df[~df.E_rel.isnull()]
idx = df_final.groupby(["n_carbon", "cutoff"])["steps"].transform("max") == df_final["steps"]
df_final = df_final[idx]

plt.close("all")
n_carbon = sorted(df_final.n_carbon.unique())
n_panels = len(n_carbon)
fig, axes = plt.subplot_mosaic(
    [[f"{n}" for n in n_carbon],
    [f"{n}rel" for n in n_carbon],
    ["final"]*n_panels],
    figsize=(17, 10),
)
for ind_n_carbon, n_carbon in enumerate(n_carbon):
    df_exp = df[df.n_carbon == n_carbon]
    ax_abs = axes[f"{n_carbon}"]
    ax_rel = axes[f"{n_carbon}rel"]
    for ind_model, model in enumerate(sorted(df_exp.model.unique())):
        df_plot = df_exp[df_exp.model == model]
        color = f"C{ind_model}"
        ax_abs.plot(df_plot.steps / 1000, df_plot.E0, label=None, color=color, ls="--")
        ax_abs.plot(df_plot.steps / 1000, df_plot.E90, label=model, color=color)
        ax_rel.plot(df_plot.steps / 1000, df_plot.E_rel, label=model, color=color)
    Emin = min(df_exp.E0.min(), df_exp.E90.min())
    ax_abs.set_ylim([Emin - 0.02, Emin + 0.2])
    ax_rel.set_ylim([-10, 140])
    ax_rel.axhline(0, color="dimgray", ls="-")
    ax_abs.set_title(f"C{n_carbon}")
    ax_abs.set_ylabel("Energy / Ha")
    ax_rel.set_ylabel("Energy difference / mHa")
    ax_rel.set_xlabel("Steps / k")
    ax_abs.legend(loc="upper right")
    ax_abs.grid(alpha=0.5)
    ax_rel.grid(alpha=0.5)

ax_final = axes["final"]
for ind_cutoff, cutoff in enumerate(sorted(df_final.cutoff.unique())):
    df_plot = df_final[df_final.cutoff == cutoff]
    ax_final.errorbar(df_plot.n_carbon, df_plot.E_rel, yerr=df_plot.E_rel_std, label=f"cutoff={cutoff:.1f}", marker="o", capsize=5)

n_fit = np.linspace(2, 24, 100)
E_fit = 180 / (n_fit-1)
ax_final.plot(n_fit, E_fit, label="1/n", color="black", ls="--")

ax_final.legend()
ax_final.set_xlabel("Number of carbon atoms")
ax_final.set_ylabel("Energy difference / mHa")
ax_final.set_ylim([-5, 130])
ax_final.axhline(0, color="dimgray", ls="-")
ax_final.grid(alpha=0.5)

fig.tight_layout()
fig.savefig(fig_fname + ".png", dpi=300, bbox_inches="tight")

