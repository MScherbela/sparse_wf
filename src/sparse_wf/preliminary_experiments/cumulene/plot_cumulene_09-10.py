# %%
import numpy as np
import matplotlib.pyplot as plt
import wandb
import pandas as pd

# %%
api = wandb.Api()
runs = [r for r in api.runs("tum_daml_nicholas/cumulene") if r.name.startswith("09-10")]

data = []
for r in runs:
    name = r.name
    print(name)
    # 09-10_cumulene_C8H4_90deg_5.0
    jastrow = "no"
    if "AttJas_True" in name:
        jastrow = "attention"
        name = name.replace("_AttJas_True", "")

    tokens = name.split("_")
    meta_data = dict(
        jastrow=jastrow,
        n_carbon=int(tokens[2].replace("C", "").replace("H4", "")),
        angle=int(tokens[3].replace("deg", "")),
        cutoff=float(tokens[4]),
    )
    full_history = [
        h for h in r.scan_history(keys=["opt/step", "opt/E_smooth", "opt/E_std", "opt/t_step"], page_size=10_000)
    ]
    full_history = pd.DataFrame(full_history)
    steps = np.arange(0, full_history["opt/step"].max() + 1, 200)

    sampled_history = []
    for step in steps:
        mask = np.abs(full_history["opt/step"] - step) < 10
        if np.any(mask):
            step_data = full_history[mask]
            step_data = dict(
                steps=step,
                E=step_data.loc[mask, "opt/E_smooth"].mean(),
                E_std=step_data.loc[mask, "opt/E_std"].mean(),
                t=step_data.loc[mask, "opt/t_step"].median(),
            )
            data.append(meta_data | step_data)
df_all = pd.DataFrame(data)
df_all.to_csv("cumulene_09-10.csv", index=False)

# %%
df_all = pd.read_csv("cumulene_09-10.csv")
fig_fname = "/home/mscherbela/ucloud/results/09-10_cumulene"
df = pd.pivot_table(
    df_all, values=("E", "t"), index=["n_carbon", "cutoff", "jastrow", "steps"], columns="angle"
).reset_index()
df.columns = ["".join([str(c) for c in col]).strip() for col in df.columns.values]
df["E_rel"] = 1000 * (df["E90"] - df["E0"])
df["t"] = (df["t0"] + df["t90"]) / 2
df["model"] = df.apply(lambda x: f"cutoff={x.cutoff:.1f} ({x.jastrow} jas)", axis=1)

df_final = df[~df.E_rel.isnull()]
idx = df_final.groupby(["n_carbon", "model", "cutoff"])["steps"].transform("max") == df_final["steps"]
df_final = df_final[idx]

# %%
plt.close("all")
fig, axes = plt.subplot_mosaic(
    [["2", "4", "6", "8"], ["2rel", "4rel", "6rel", "8rel"], ["final"] * 4, ["time"] * 2 + ["time_per_el"] * 2],
    figsize=(15, 10),
)
for ind_n_carbon, n_carbon in enumerate([2, 4, 6, 8]):
    df_exp = df[df.n_carbon == n_carbon]
    ax_abs = axes[f"{n_carbon}"]
    ax_rel = axes[f"{n_carbon}rel"]
    for ind_model, model in enumerate(sorted(df_exp.model.unique())):
        df_plot = df_exp[df_exp.model == model]
        color = f"C{ind_model}"
        ax_abs.plot(df_plot.steps / 1000, df_plot.E0, label=None, color=color, ls="--")
        ax_abs.plot(df_plot.steps / 1000, df_plot.E90, label=model, color=color)
        ax_rel.plot(df_plot.steps / 1000, df_plot.E_rel, label=model, color=color)
    ax_abs.set_ylim([df_plot.E0.min() - 0.01, df_plot.E0.min() + 0.2])
    ax_rel.set_ylim([0, 130])
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
ax_final.set_ylim([0, 125])
ax_final.grid(alpha=0.5)

ax_time, ax_time_per_el = axes["time"], axes["time_per_el"]
for ind_model, model in enumerate(sorted(df_final.model.unique())):
    df_plot = df_final[df_final.model == model]
    n_el = 6 * df_plot.n_carbon + 4
    ax_time.plot(df_plot.n_carbon, df_plot.t, label=model, marker="o")
    ax_time_per_el.plot(df_plot.n_carbon, df_plot.t / n_el, label=model, marker="o")

for ax in [ax_time, ax_time_per_el]:
    ax.legend()
    ax.set_xlabel("Number of carbon atoms")
    ax.set_ylim([0, None])
    ax.grid(alpha=0.5)
ax_time.set_ylabel("Time per opt step / s")
ax_time_per_el.set_ylabel("Time per electron / s")


fig.suptitle("Cumulene, new_sparse model; pre: CASCI, 2k LAMB; opt: lr=0.1, delay=2000, damp=1e-3")
fig.tight_layout()
fig.savefig(fig_fname + "_cutoff.png", dpi=200, bbox_inches="tight")
