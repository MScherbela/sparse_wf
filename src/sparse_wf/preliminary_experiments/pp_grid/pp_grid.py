#%%
import wandb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%%
api = wandb.Api()
runs = [r for r in api.runs("tum_daml_nicholas/pp_grid") if r.name.startswith("pp_dimers")]

data = []
for r in runs:
    print(r.name)
    metadata = dict(
        element=r.config["molecule_args"]["chain_args"]["element"],
        n_el=r.config["molecule"]["n_el"],
        n_grid=r.config["optimization"]["pp_grid_points"],
    )

    df_hist = r.history(keys=["opt/step", "opt/E_std"], samples=2000, pandas=True)
    df_hist = df_hist[df_hist["opt/step"] > 4000]
    if len(df_hist) > 0:
        data.append(dict(E_std=df_hist["opt/E_std"].median(), **metadata))
df = pd.DataFrame(data)

#%%

pivot = df.groupby(["n_el", "element", "n_grid"])["E_std"].mean().reset_index()
pivot["E_var"] = pivot["E_std"]**2
pivot["var_times_grid"] = pivot["E_var"] * pivot["n_grid"]
fig, axes = plt.subplots(2, 2, figsize=(10, 6))

for element, ax in zip(pivot.element.unique(), axes.flatten()):
    df_plot = pivot[pivot["element"] == element]
    ax.plot(df_plot["n_grid"], df_plot["E_var"], marker="o", label="Variance")
    twin = ax.twinx()
    twin.plot(df_plot["n_grid"], df_plot["var_times_grid"], marker="o", color="red", label="Variance * n_grid")
    ax.set_title(element)
    ax.set_yscale("log")
    ax.set_xscale("log")
    x_ticks = [2, 4, 6, 12, 24]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks)
    ax.set_xticks([], minor=True)
    if element == "Fe":
        ax.set_ylim([1, 2000])
        twin.set_ylim([10, 20000])
    else:
        ax.set_ylim([0.01, 4])
        twin.set_ylim([0.1, 40])

    ax.legend(loc="upper left")
    twin.legend(loc="upper right")
    ax.set_xlabel("n_pp_grid")
    ax.set_ylabel("Variance / Ha^2")
    twin.set_ylabel("Variance * n_grid / Ha^2")
    twin.set_yscale("log")


fig.tight_layout()
fig.savefig("pp_grid.png", bbox_inches="tight")
fig.suptitle("Energy variance for dimers after 5k opt steps")
