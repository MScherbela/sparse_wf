# %%
import numpy as np
import matplotlib.pyplot as plt
import wandb
import pandas as pd

#%%
api = wandb.Api()
runs = [r for r in api.runs("tum_daml_nicholas/cumulene") if r.name.startswith("09-10")]

data = []
for r in runs:
    print(r.name)
    # 09-10_cumulene_C8H4_90deg_5.0
    tokens = r.name.split("_")
    meta_data = dict(
        n_carbon=int(tokens[2].replace("C", "").replace("H4", "")),
        angle=int(tokens[3].replace("deg", "")),
        cutoff=float(tokens[4]),
    )
    full_history = [h for h in r.scan_history(keys=["opt/step", "opt/E_smooth", "opt/E_std"], page_size=10_000)]
    full_history = pd.DataFrame(full_history)
    steps = np.arange(0, full_history["opt/step"].max() + 1, 100)

    sampled_history = []
    for step in steps:
        mask = np.abs(full_history["opt/step"] - step) < 10
        if np.any(mask):
            step_data = full_history[mask].mean()
            step_data = dict(steps=step, E=step_data["opt/E_smooth"], E_std=step_data["opt/E_std"])
            data.append(meta_data | step_data)
df_all = pd.DataFrame(data)
df_all.to_csv("cumulene_09-10.csv", index=False)

#%%
df_all = pd.read_csv("cumulene_09-10.csv")
fig_fname = "/home/mscherbela/ucloud/results/09-10_cumulene"
df = pd.pivot_table(df_all, values="E", index=["n_carbon", "cutoff", "steps"], columns="angle").reset_index()
df = df.rename(columns={0: "E_0", 90: "E_90"})
df["E_rel"] = 1000 * (df["E_90"] - df["E_0"])

df_final = df[~df.E_rel.isnull()]
idx = df_final.groupby(["n_carbon", "cutoff"])["steps"].transform("max") == df_final["steps"]
df_final = df_final[idx]

#%%
plt.close("all")
fig, axes = plt.subplot_mosaic([["2", "4", "6", "8"], ["2rel", "4rel", "6rel", "8rel"], ["final"]*4], figsize=(13, 9))
for ind_n_carbon, n_carbon in enumerate([2, 4, 6, 8]):
    df_exp = df[df.n_carbon == n_carbon]
    ax_abs = axes[f"{n_carbon}"]
    ax_rel = axes[f"{n_carbon}rel"]
    for ind_cutoff, cutoff in enumerate(sorted(df_exp.cutoff.unique())):
        df_plot = df_exp[df_exp.cutoff == cutoff]
        color = f"C{ind_cutoff}"
        ax_abs.plot(df_plot.steps / 1000, df_plot.E_0, label=None, color=color, ls='--')
        ax_abs.plot(df_plot.steps / 1000, df_plot.E_90, label=f"cutoff={cutoff}", color=color)
        ax_rel.plot(df_plot.steps / 1000, df_plot.E_rel, label=f"cutoff={cutoff}", color=color)
    ax_abs.set_ylim([df_plot.E_0.min() - 0.01, df_plot.E_0.min() + 0.2])
    ax_rel.set_ylim([0, 130])
    ax_abs.set_title(f"C{n_carbon}")
    ax_abs.set_ylabel("Energy / Ha")
    ax_rel.set_ylabel("Energy difference / mHa")
    ax_rel.set_xlabel("Steps / k")
    ax_abs.legend(loc="upper right")
    ax_abs.grid(alpha=0.5)
    ax_rel.grid(alpha=0.5)

ax_final = axes["final"]
for ind_cutoff, cutoff in enumerate(sorted(df_exp.cutoff.unique())):
    df_plot = df_final[df_final.cutoff == cutoff]
    ax_final.plot(df_plot.n_carbon, df_plot.E_rel, label=f"cutoff={cutoff}", marker='o')
ax_final.legend()
ax_final.set_xlabel("Number of carbon atoms")
ax_final.set_ylabel("Energy difference / mHa")
ax_final.set_ylim([0, 125])
ax_final.grid(alpha=0.5)

fig.suptitle("Cumulene, new_sparse model, no jastrow\nCASCI init")
fig.tight_layout()
fig.savefig(fig_fname + "_cutoff.png", dpi=200, bbox_inches="tight")








