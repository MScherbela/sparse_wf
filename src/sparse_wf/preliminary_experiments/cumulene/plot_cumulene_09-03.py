# %%
import numpy as np
import matplotlib.pyplot as plt
import wandb
import pandas as pd

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
            jastrow="none",
            experiment="cutoff",
        )
    elif ("PairJas" in r.name) or ("Attention" in r.name):
        # 09-03_PairJas_True_cumulene_C4H4_90deg_3.0
        if "Attention" in r.name:
            jastrow = "attention"
        elif "PairJas_True" in r.name:
            jastrow = "pairwise_MLP"
        else:
            jastrow = "none"

        tokens = r.name.split("_")
        meta_data = dict(
            n_carbon=int(tokens[4].replace("C", "").replace("H4", "")),
            angle=int(tokens[5].replace("deg", "")),
            cutoff=float(tokens[6]),
            model="new_sparse",
            jastrow=jastrow,
            experiment="jastrow",
        )
    elif "Model" in r.name:
        # 09-03_Model_new_sparse_cumulene_C4H4_0deg_3.0
        tokens = r.name.replace("new_sparse", "new#sparse").split("_")
        meta_data = dict(
            n_carbon=int(tokens[4].replace("C", "").replace("H4", "")),
            angle=int(tokens[5].replace("deg", "")),
            cutoff=float(tokens[6]),
            model=tokens[2].replace("new#sparse", "new_sparse"),
            jastrow="none",
            experiment="model",
        )
    else:
        print(f"Skipping {r.name}")
        continue

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
df_all.to_csv("cumulene_09-03.csv", index=False)

#%%
df_all = pd.read_csv("cumulene_09-03.csv")
fig_fname = "/home/mscherbela/ucloud/results/09-03_cumulene"
df = pd.pivot_table(df_all, values="E", index=["experiment", "model","jastrow", "n_carbon", "cutoff", "steps"], columns="angle").reset_index()
df = df.rename(columns={0: "E_0", 90: "E_90"})
df["E_rel"] = 1000 * (df["E_90"] - df["E_0"])

df_final = df[~df.E_rel.isnull()]
idx = df_final.groupby(["experiment", "model", "jastrow", "n_carbon", "cutoff"])["steps"].transform("max") == df_final["steps"]
df_final = df_final[idx]

#%%
plt.close("all")
fig, axes = plt.subplots(1, 2, figsize=(8, 5))
df_exp = df[df.experiment == "model"]
for model in ["new", "new_sparse"]:
    df_plot = df_exp[df_exp.model == model]
    color = "C0" if model == "new" else "C1"
    axes[0].plot(df_plot.steps / 1000, df_plot.E_0, label=f"0 deg: {model}", ls="--", color=color)
    axes[0].plot(df_plot.steps / 1000, df_plot.E_90, label=f"90 deg: {model}", ls="-", color=color)
    axes[1].plot(df_plot.steps / 1000, df_plot.E_rel, label=model, ls="-", color=color)
for ax in axes:
    ax.legend()
    ax.set_xlabel("Steps / k")
    ax.grid(alpha=0.5)
axes[0].set_ylabel("Energy / Ha")
axes[0].set_ylim([-154.71, -154.55])
axes[1].set_ylim([-5, 100])
axes[1].set_ylabel("Energy difference / mHa")
fig.suptitle("Embedding model\nC4H4, cutoff 3.0")
fig.tight_layout()
fig.savefig(fig_fname + "_model.png", dpi=200, bbox_inches="tight")

#%%
plt.close("all")
fig, axes = plt.subplots(1, 2, figsize=(8, 5))
df_exp = df[df.experiment == "jastrow"]
for ind_jastrow, jastrow in enumerate(["none", "pairwise_MLP", "attention"]):
    df_plot = df_exp[df_exp.jastrow == jastrow]
    color = f"C{ind_jastrow}"
    axes[0].plot(df_plot.steps / 1000, df_plot.E_0, label=f"0 deg: jastrow={jastrow}", ls="--", color=color)
    axes[0].plot(df_plot.steps / 1000, df_plot.E_90, label=f"90 deg: jastrow={jastrow}", ls="-", color=color)
    axes[1].plot(df_plot.steps / 1000, df_plot.E_rel, label=f"jastrow={jastrow}", ls="-", color=color)
for ax in axes:
    ax.legend()
    ax.set_xlabel("Steps / k")
    ax.grid(alpha=0.5)
axes[0].set_ylabel("Energy / Ha")
axes[0].set_ylim([-154.71, -154.55])
axes[1].set_ylim([-5, 100])
axes[1].set_ylabel("Energy difference / mHa")
fig.suptitle("Pairwise MLP jastrow\nC4H4, new_sparse model, cutoff=3.0")
fig.tight_layout()
fig.savefig(fig_fname + "_jastrow.png", dpi=200, bbox_inches="tight")


#%%
plt.close("all")
# fig, axes = plt.subplots(3, 3, figsize=(11, 9))
fig, axes = plt.subplot_mosaic([["4", "6", "8"], ["4rel", "6rel", "8rel"], ["final", "final", "final"]], figsize=(11, 9))
for ind_n_carbon, n_carbon in enumerate([4, 6, 8]):
    df_exp = df[(df.experiment == "cutoff") & (df.n_carbon == n_carbon)]
    ax_abs = axes[f"{n_carbon}"]
    ax_rel = axes[f"{n_carbon}rel"]
    for ind_cutoff, cutoff in enumerate(sorted(df_exp.cutoff.unique())):
        df_plot = df_exp[df_exp.cutoff == cutoff]
        color = f"C{ind_cutoff}"
        ax_abs.plot(df_plot.steps / 1000, df_plot.E_0, label=None, color=color, ls='--')
        ax_abs.plot(df_plot.steps / 1000, df_plot.E_90, label=f"cutoff={cutoff}", color=color)
        ax_rel.plot(df_plot.steps / 1000, df_plot.E_rel, label=f"cutoff={cutoff}", color=color)
    ax_abs.set_ylim([df_plot.E_0.min() - 0.01, df_plot.E_0.min() + 0.2])
    ax_rel.set_ylim([0, 100])
    ax_abs.set_title(f"C{n_carbon}")
    ax_abs.set_ylabel("Energy / Ha")
    ax_rel.set_ylabel("Energy difference / mHa")
    ax_rel.set_xlabel("Steps / k")
    ax_abs.legend()
    ax_abs.grid(alpha=0.5)
    ax_rel.grid(alpha=0.5)

ax_final = axes["final"]
df_exp = df_final[(df_final.experiment == "cutoff") & (df_final.steps >= 30_000)]
for ind_cutoff, cutoff in enumerate(sorted(df_exp.cutoff.unique())):
    df_plot = df_exp[df_exp.cutoff == cutoff]
    ax_final.plot(df_plot.n_carbon, df_plot.E_rel, label=f"cutoff={cutoff}", marker='o')
ax_final.legend()
ax_final.set_xlabel("Number of carbon atoms")
ax_final.set_ylabel("Energy difference / mHa")
ax_final.set_ylim([0, 100])
ax_final.grid(alpha=0.5)

fig.suptitle("Cumulene, new_sparse model, no jastrow")
fig.tight_layout()
fig.savefig(fig_fname + "_cutoff.png", dpi=200, bbox_inches="tight")








