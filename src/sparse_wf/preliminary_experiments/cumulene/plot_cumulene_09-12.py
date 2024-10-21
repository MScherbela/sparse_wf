# %%
import numpy as np
import matplotlib.pyplot as plt
import wandb
import pandas as pd

api = wandb.Api()
runs = [r for r in api.runs("tum_daml_nicholas/cumulene") if r.name.startswith("09-12")]

data = []
for r in runs:
    name = r.name
    print(name)
    #09-12_cumulene_C16H4_0deg_8
    tokens = name.split("_")
    meta_data = dict(
        jastrow="attention",
        n_carbon=int(tokens[2].replace("C", "").replace("H4", "")),
        angle=int(tokens[3].replace("deg", "")),
        n_gpus=int(tokens[4]),
        cutoff=3.0,
    )
    full_history = [
        h for h in r.scan_history(keys=["opt/step", "opt/E", "opt/E_std", "opt/t_step"], page_size=10_000)
    ]
    full_history = pd.DataFrame(full_history)
    full_history = full_history.rename(columns={"opt/E": "E", "opt/E_std": "E_std", "opt/t_step": "t", "opt/step": "steps"})
    for k,v in meta_data.items():
        full_history[k] = v
    data.append(full_history)

df_all = pd.concat(data, axis=0, ignore_index=True)
df_all.to_csv("cumulene_09-12.csv", index=False)

# %%
df_all = pd.read_csv("cumulene_09-12.csv")
fig_fname = "/home/mscherbela/ucloud/results/09-12_cumulene"
df = pd.pivot_table(
    df_all, values=("E", "t"), index=["n_carbon", "cutoff", "jastrow", "steps"], columns="angle"
).reset_index()
df.columns = ["".join([str(c) for c in col]).strip() for col in df.columns.values]

# Remove large outliers from restart
outlier_energy = 2.0 # Ha
df["E0_smooth"] = df["E0"].rolling(window=100).mean()
df["E90_smooth"] = df["E90"].rolling(window=100).mean()
include_E0 = df["E0_smooth"].isnull() | ((df["E0_smooth"] - df["E0"]).abs() < outlier_energy)
include_E90 = df["E90_smooth"].isnull() | ((df["E90_smooth"] - df["E90"]).abs() < outlier_energy)
df = df[include_E0 & include_E90]

# Running average
smoothing_window = 5000
df["E0"] = df["E0"].rolling(window=smoothing_window).mean()
df["E90"] = df["E90"].rolling(window=smoothing_window).mean()


df["E_rel"] = 1000 * (df["E90"] - df["E0"])
df["t"] = (df["t0"].rolling(smoothing_window).median() + df["t90"].rolling(smoothing_window).median()) / 2
df["model"] = df.apply(lambda x: f"cutoff={x.cutoff:.1f} ({x.jastrow} jas)", axis=1)
df = df[df.steps % 200 == 0]

df_final = df[~df.E_rel.isnull()]
idx = df_final.groupby(["n_carbon", "model", "cutoff"])["steps"].transform("max") == df_final["steps"]
df_final = df_final[idx]

plt.close("all")
n_carbon = sorted(df_final.n_carbon.unique())
n_panels = len(n_carbon)
fig, axes = plt.subplot_mosaic(
    [[f"{n}" for n in n_carbon],
    [f"{n}rel" for n in n_carbon],
    ["final"]*n_panels],
    figsize=(15, 10),
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
    ax_rel.set_ylim([-100, 100])
    ax_rel.axhline(0, color="dimgray", ls="-")
    ax_abs.set_title(f"C{n_carbon}")
    ax_abs.set_ylabel("Energy / Ha")
    ax_rel.set_ylabel("Energy difference / mHa")
    ax_rel.set_xlabel("Steps / k")
    ax_abs.legend(loc="upper right")
    ax_abs.grid(alpha=0.5)
    ax_rel.grid(alpha=0.5)

ax_final = axes["final"]
for ind_model, model in enumerate(sorted(df_final.model.unique())):
    df_plot = df_final[df_final.model == model]
    ax_final.plot(df_plot.n_carbon, df_plot.E_rel, label=model, marker="o")
ax_final.legend()
ax_final.set_xlabel("Number of carbon atoms")
ax_final.set_ylabel("Energy difference / mHa")
ax_final.set_ylim([-70, 50])
ax_final.axhline(0, color="dimgray", ls="-")
ax_final.grid(alpha=0.5)


# ax_time, ax_time_per_el = axes["time"], axes["time_per_el"]
# for ind_model, model in enumerate(sorted(df_final.model.unique())):
#     df_plot = df_final[df_final.model == model]
#     n_el = 6 * df_plot.n_carbon + 4
#     ax_time.plot(df_plot.n_carbon, df_plot.t, label=model, marker="o")
#     ax_time_per_el.plot(df_plot.n_carbon, df_plot.t / n_el, label=model, marker="o")

# for ax in [ax_time, ax_time_per_el]:
#     ax.legend()
#     ax.set_xlabel("Number of carbon atoms")
#     ax.set_ylim([0, None])
#     ax.grid(alpha=0.5)
# ax_time.set_ylabel("Time per opt step / s")
# ax_time_per_el.set_ylabel("Time per electron / s")


fig.suptitle("Cumulene, new_sparse model; pre: CASCI, 5k LAMB; opt: lr=0.1, delay=1000, damp=1e-4; mcmc: 2048 batch, 1 sweep")
fig.tight_layout()
fig.savefig(fig_fname + "_cutoff.png", dpi=200, bbox_inches="tight")
