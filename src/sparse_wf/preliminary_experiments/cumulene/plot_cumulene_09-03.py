# %%
import numpy as np
import matplotlib.pyplot as plt
import wandb
import re
import pandas as pd
import json

api = wandb.Api()
runs = [r for r in api.runs("tum_daml_nicholas/cumulene") if r.name.startswith("09-03")]


data = []
for r in runs:
    print(r.name)
    if "Cutoff" in r.name:
        # 09-03_Cutoff_cumulene_C8H4_90deg_5.0
        tokens = r.name.split("_")
        meta_data = dict(
            n_carbon=int(tokens[3].replace("C", "").replace("H4", "")),
            angle=int(tokens[4].replace("deg", "")),
            cutoff=float(tokens[5]),
            model="new_sparse",
            pair_jastrow=False,
            experiment="cutoff",
        )
    elif "PairJas" in r.name:
        # 09-03_PairJas_True_cumulene_C4H4_90deg_3.0
        tokens = r.name.split("_")
        meta_data = dict(
            n_carbon=int(tokens[4].replace("C", "").replace("H4", "")),
            angle=int(tokens[5].replace("deg", "")),
            cutoff=float(tokens[6]),
            model="new_sparse",
            pair_jastrow=tokens[2] == "True",
            experiment="pair_jastrow",
        )
    elif "Model" in r.name:
        # 09-03_Model_new_sparse_cumulene_C4H4_0deg_3.0
        tokens = r.name.replace("new_sparse", "new#sparse").split("_")
        meta_data = dict(
            n_carbon=int(tokens[4].replace("C", "").replace("H4", "")),
            angle=int(tokens[5].replace("deg", "")),
            cutoff=float(tokens[6]),
            model=tokens[2].replace("new#sparse", "new_sparse"),
            pair_jastrow=False,
            experiment="model",
        )
    else:
        print(f"Skipping {r.name}")
        continue

    history = []
    for h in r.scan_history(keys=["opt/step", "opt/E_smooth", "opt/E_std"], page_size=10_000):
        if h["opt/step"] % 1000 == 0:
            history.append(dict(E=h["opt/E_smooth"], E_std=h["opt/E_std"], steps=h["opt/step"]))

    for h in history:
        data.append(meta_data | h)
df_all = pd.DataFrame(data)

#%%
df = pd.pivot_table(df_all, values="E", index=["experiment", "model","pair_jastrow", "n_carbon", "cutoff", "steps"], columns="angle").reset_index()
df = df.rename(columns={0: "E_0", 90: "E_90"})
df["E_rel"] = 1000 * (df["E_90"] - df["E_0"])

df_final = df[~df.E_rel.isnull()]
idx = df_final.groupby(["experiment", "model", "pair_jastrow", "n_carbon", "cutoff"])["steps"].transform("max") == df_final["steps"]
df_final = df_final[idx]

#%%
plt.close("all")
fig, axes = plt.subplots(1, 2, figsize=(8, 5))
df_exp = df[df.experiment == "model"]
for model in ["new", "new_sparse"]:
    df_plot = df_exp[df_exp.model == model]
    color = "C0" if model == "new" else "C1"
    axes[0].plot(df_plot.steps, df_plot.E_0, label=f"0 deg: {model}", ls="--", color=color)
    axes[0].plot(df_plot.steps, df_plot.E_90, label=f"90 deg: {model}", ls="-", color=color)
    axes[1].plot(df_plot.steps, df_plot.E_rel, label=model, ls="-", color=color)
for ax in axes:
    ax.legend()
    ax.set_xlabel("Steps")
    ax.grid(alpha=0.5)
axes[0].set_ylabel("Energy / Ha")
axes[0].set_ylim([-154.7, -154.5])
axes[1].set_ylim([-5, 100])
axes[1].set_ylabel("Energy difference / mHa")
fig.suptitle("Embedding model\nC4H4, cutoff 3.0")
fig.tight_layout()

#%%
plt.close("all")
fig, axes = plt.subplots(1, 2, figsize=(8, 5))
df_exp = df[df.experiment == "pair_jastrow"]
for jastrow in [False, True]:
    df_plot = df_exp[df_exp.pair_jastrow == jastrow]
    color = "C0" if not jastrow else "C1"
    axes[0].plot(df_plot.steps, df_plot.E_0, label=f"0 deg: PairJastrow={jastrow}", ls="--", color=color)
    axes[0].plot(df_plot.steps, df_plot.E_90, label=f"90 deg: PairJastrow={jastrow}", ls="-", color=color)
    axes[1].plot(df_plot.steps, df_plot.E_rel, label=f"PairJastrow={jastrow}", ls="-", color=color)
for ax in axes:
    ax.legend()
    ax.set_xlabel("Steps")
    ax.grid(alpha=0.5)
axes[0].set_ylabel("Energy / Ha")
axes[0].set_ylim([-154.7, -154.5])
axes[1].set_ylim([-5, 100])
axes[1].set_ylabel("Energy difference / mHa")
fig.suptitle("Pairwise MLP jastrow\nC4H4, new_sparse model, cutoff=3.0")
fig.tight_layout()

#%%
plt.close("all")
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
df_exp = df_final[(df_final.experiment == "cutoff") & (df_final.steps >= 7_000)]
for ind_cutoff, cutoff in enumerate(sorted(df_exp.cutoff.unique())):
    df_plot = df_exp[df_exp.cutoff == cutoff]
    ax.plot(df_plot.n_carbon, df_plot.E_rel, label=f"cutoff={cutoff}", marker='o')
ax.legend()
ax.set_xlabel("Number of carbon atoms")
ax.set_ylabel("Energy difference / mHa")
ax.grid(alpha=0.5)









