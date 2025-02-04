# %%
import numpy as np
import pandas as pd
import wandb
import re
import matplotlib.pyplot as plt


def get_without_outliers(x, outlier_threshold=10):
    med = np.median(x)
    spread = np.quantile(x, 0.75) - np.quantile(x, 0.25)
    return x[np.abs(x - med) < outlier_threshold * spread]


def robustmean(x):
    x = get_without_outliers(x)
    return np.mean(x)


def robuststderr(x):
    x = get_without_outliers(x)
    return np.std(x) / np.sqrt(len(x))


def get_data(run, **metadata):
    data = []
    for h in run.scan_history(["opt/step", "opt/E"], page_size=10_000):
        data.append(h)
    df = pd.DataFrame(data)
    for k, v in metadata.items():
        df[k] = v
    return df


geom_names = ["FerroceneCl_red_geom", "FerroceneCl_red_geom_charged"]

# Experiment
rel_energies = {
    "Experiment": 258,
    "B3LYP (Toma et al)": 238,
    "HF (cc-pVTZ)": 194,
}

## SWANN
reload_data = True
if reload_data:
    name_template = f"3.0_({'|'.join(geom_names)})"
    all_runs = wandb.Api().runs("tum_daml_nicholas/ferrocene")
    runs = [r for r in all_runs if re.match(name_template, r.name)]

    swann_data = []
    for r in runs:
        print(r.name)
        charge = "charged" if "charged" in r.name else "neutral"
        df = get_data(
            r, cutoff=float(r.name.split("_")[0]), charge=charge
        )
        swann_data.append(df)
    df_swann = pd.concat(swann_data)
    df_swann.to_csv("swann_ferrocene.csv", index=False)
else:
    df_swann = pd.read_csv("swann_ferrocene.csv")


df = df_swann.pivot_table(index="opt/step", columns=["charge"], values="opt/E")
df = df.fillna(method="ffill", limit=10)
df["deltaE"] = (df["charged"] - df["neutral"]) * 1000
df["deltaE_rolling"] = df["deltaE"].rolling(1000).mean()

plt.close("all")
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ref_colors = ["black", "C0", "C2"]
for color, (ref, energy) in zip(ref_colors, rel_energies.items()):
    ax.axhline(energy, label=ref, color=color, linestyle="--")

ax.plot(df.index / 1000, df["deltaE_rolling"], label="SWANN (cutoff=3.0)", color="red")
ax.set_ylabel("Ionization potential / mHa")
ax.set_xlabel("Optimization step / k")
ax.set_title("Chloro-ferrocene ionization potential")
# Set minor ticks on y-axis every 5
ax.yaxis.set_minor_locator(plt.MultipleLocator(5))
ax.grid(True, "major", ls="-", alpha=0.5)
ax.grid(True, "minor", ls=":", alpha=0.5)
ax.legend()
fig.tight_layout()
fig.savefig("ferrocene.png")





