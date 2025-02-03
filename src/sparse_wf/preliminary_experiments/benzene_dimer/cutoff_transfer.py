# %%
import numpy as np
import pandas as pd
import wandb
import re
import matplotlib.pyplot as plt
import matplotlib


def mask_without_outliers(x, outlier_threshold=10):
    med = np.nanmedian(x)
    spread = np.nanquantile(x, 0.75) - np.nanquantile(x, 0.25)
    return np.abs(x - med) < outlier_threshold * spread


def robustmean(x):
    x = x[mask_without_outliers(x)]
    return np.mean(x)


def robuststderr(x):
    x = x[mask_without_outliers(x)]
    return np.std(x) / np.sqrt(len(x))


def get_data(run, **metadata):
    data = []
    for h in run.scan_history(["opt/step", "opt/E"], page_size=10_000):
        data.append(h)
    df = pd.DataFrame(data)
    for k, v in metadata.items():
        df[k] = v
    return df


geom_names = ["benzene_dimer_T_4.95A", "benzene_dimer_T_10.00A"]

## SWANN
reload_data = False
if reload_data:
    name_template = f"cutoff_transfer_3.0to5.0.*lroffset98"
    all_runs = wandb.Api().runs("tum_daml_nicholas/benzene")
    runs = [r for r in all_runs if re.match(name_template, r.name)]

    swann_data = []
    for r in runs:
        print(r.name)
        df = get_data(r, geom=geom_names[0] if "T_4.95A" in r.name else geom_names[1], cutoff="transfer")
        swann_data.append(df)
    df = pd.concat(swann_data)
    df.to_csv("cutoff_transfer.csv", index=False)
else:
    df = pd.read_csv("cutoff_transfer.csv")

df = pd.concat([df, pd.read_csv("swann_benzene_dimer.csv")])

smoothing_window = 5000
smoothing_min = 20

plt.close("all")
fig, axes = plt.subplots(1, 2, figsize=(10, 6))

colors = matplotlib.colormaps["plasma"](np.linspace(0.1, 0.9, 4))
colors = colors.tolist() + ["black"]
for cutoff, color in zip([3.0, 5.0, 7.0, 9.0, "transfer"], colors):
    for idx_geom, geom in enumerate(geom_names):
        data = df[(df["geom"] == geom) & (df["cutoff"] == cutoff)]
        if len(data) == 0:
            print("Not enough data for", geom, cutoff)
            continue
        data = data.sort_values("opt/step")
        mask = mask_without_outliers(data["opt/E"].values)
        data = data[mask]
        # rolling mean
        data["E_smooth"] = data["opt/E"].rolling(smoothing_window, min_periods=smoothing_min).mean()
        if idx_geom==0:
            label = None
        elif cutoff=="transfer":
            label = "3.0 -> 5.0"
        else:
            label = f"cutoff={cutoff}"
        axes[0].plot(data["opt/step"] / 1000, data["E_smooth"], label=label, color=color)
    pivot = df[df.cutoff == cutoff].pivot_table(index="opt/step", columns="geom", values="opt/E")
    pivot = pivot.fillna(method="ffill", limit=10)
    pivot["diff"] = pivot[geom_names[0]] - pivot[geom_names[1]]
    pivot.loc[~mask_without_outliers(pivot["diff"].values), "diff"] = np.nan
    pivot["diff"] = pivot["diff"].fillna(method="ffill", limit=10)
    pivot = pivot.rolling(smoothing_window, min_periods=smoothing_min).mean()
    axes[1].plot(pivot.index / 1000, pivot["diff"] * 1000, label=label, color=color)
axes[1].set_xlabel("Optimization step / k")
axes[1].set_ylabel("Energy difference / mHa")
axes[1].legend()
axes[1].grid(alpha=0.5)
axes[1].set_ylim([-10, 5])

axes[0].set_ylim([None, -75.275])
axes[0].grid(alpha=0.5)
axes[0].set_xlabel("Optimization step / k")
axes[0].legend()
fig.tight_layout()
fig.savefig("cutoff_transfer.png", dpi=200)
