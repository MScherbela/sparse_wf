# %%
import numpy as np
import pandas as pd
import wandb
import re
import matplotlib.pyplot as plt


def get_data(run, **metadata):
    data = []
    for h in run.scan_history(["opt/step", "opt/E"], page_size=10_000):
        data.append(h)
    df = pd.DataFrame(data)
    df = df.sort_values("opt/step")
    df = df.iloc[1:]  # Drop first step
    for k, v in metadata.items():
        df[k] = v
    return df

def get_outlier_mask(x):
    qlow = x.quantile(0.01)
    qhigh = x.quantile(0.99)
    med = x.median()
    included_range = 5 * (qhigh - qlow)
    is_outlier = (x < med - included_range) | (x > med + included_range)
    return is_outlier


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
    name_template = f"HLR.*"
    all_runs = wandb.Api().runs("tum_daml_nicholas/ferrocene")
    runs = [r for r in all_runs if re.match(name_template, r.name)]

    swann_data = []
    for r in runs:
        print(r.name)
        charge = "charged" if "charged" in r.name else "neutral"
        df = get_data(
            r, cutoff=float(r.name.split("_")[1]), charge=charge
        )
        swann_data.append(df)
    df_swann = pd.concat(swann_data)
    df_swann.to_csv("swann_ferrocene.csv", index=False)
else:
    df_swann = pd.read_csv("swann_ferrocene.csv")

window = 2000
cutoffs = [3.0, 5.0]

df = df_swann.pivot_table(index="opt/step", columns=["cutoff", "charge"], values="opt/E")
df = df.ffill(limit=10)
for cutoff in cutoffs:
    df.loc[:, (cutoff, "deltaE")] = (df[(cutoff, "charged")] - df[(cutoff, "neutral")]) * 1000
    mask = get_outlier_mask(df[(cutoff, "deltaE")])
    df.loc[mask, (cutoff, "deltaE")] = np.nan
    df.loc[:, (cutoff, "deltaE")] = df.loc[:, (cutoff, "deltaE")].ffill(limit=10)
    df.loc[:, (cutoff, "deltaE_rolling")] = df[(cutoff, "deltaE")].rolling(window).mean()

plt.close("all")
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ref_colors = ["black", "C0", "C2"]
for color, (ref, energy) in zip(ref_colors, rel_energies.items()):
    ax.axhline(energy, label=ref, color=color, linestyle="--")
    if ref == "Experiment":
        ax.axhspan(energy - 1.6, energy + 1.6, color=color, alpha=0.2)

swann_colors = ["orange", "red"]
for color, cutoff in zip(swann_colors, cutoffs):
    df_cut = df[cutoff]
    ax.plot(df_cut.index / 1000, df_cut["deltaE_rolling"], label=f"SWANN (cutoff={cutoff:.1f}])", color=color)
    E_final = df_cut["deltaE_rolling"][df_cut["deltaE_rolling"].last_valid_index()]
    print(f"SWANN (cutoff={cutoff:.1f}): {E_final:.1f} mHa")
    ax.axhline(E_final, color=color, linestyle="--")

ax.set_ylabel("Ionization potential / mHa")
ax.set_xlabel("Optimization step / k")
ax.set_title("Chloro-ferrocene ionization potential")
# Set minor ticks on y-axis every 5
ax.yaxis.set_minor_locator(plt.MultipleLocator(5))
ax.grid(True, "major", ls="-", alpha=0.5)
ax.grid(True, "minor", ls=":", alpha=0.5)
ax.set_ylim([190, 265])
ax.legend()
fig.tight_layout()
fig.savefig("ferrocene.png")





