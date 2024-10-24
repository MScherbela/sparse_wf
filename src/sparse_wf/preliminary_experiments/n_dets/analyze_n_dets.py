#%%
import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

data_fname = "data_cumulene_ndets.csv"

#%% Load data
api = wandb.Api()
runs = api.runs("tum_daml_nicholas/n_dets")
data_final = []
data_full = []
for run in runs:
    print(run.name)
    # Parse names like this: dets_4_cumulene_C8H4_90deg
    match = re.match(r"dets_(\d+)_cumulene_C(\d+)H4_(\d+)deg", run.name)
    metadata = dict(
        n_dets=int(match.group(1)),
        n_carbon=int(match.group(2)),
        angle=int(match.group(3)),
    )
    full_history = [h for h in run.scan_history(keys=["opt/step", "opt/E"], page_size=10_000)]
    h = pd.DataFrame(full_history)
    h["E_smooth"] = h["opt/E"].rolling(2000).mean()
    for k, v in metadata.items():
        h[k] = v
    data_full.append(h)

    if h["opt/step"].max() < 25_000:
        continue
    h_final = h[(h["opt/step"] >= 20_000) & (h["opt/step"] <= 25_000)]
    data_final.append(dict(
        **metadata,
        E=h_final["opt/E"].mean(),
        E_sigma=h_final["opt/E"].std() / np.sqrt(len(h_final)),
    ))
df_final = pd.DataFrame(data_final)
df_full = pd.concat(data_full, axis=0, ignore_index=True)
df_final.to_csv(data_fname, index=False)
df_full.to_csv(data_fname.replace(".csv", "_full.csv"), index=False)

#%%
df = pd.read_csv(data_fname)
df_full = pd.read_csv(data_fname.replace(".csv", "_full.csv"))


full_pivot = df_full.pivot(index=["n_carbon", "n_dets", "opt/step"], columns="angle", values="E_smooth")
full_pivot.columns = ["E0", "E90"]
full_pivot = full_pivot.reset_index()
full_pivot["E_rel"] = 1000 * (full_pivot["E90"] - full_pivot["E0"])

pivot = df.pivot(index=["n_carbon", "n_dets"], columns="angle", values="E")
pivot.columns = ["E0", "E90"]
pivot = pivot.reset_index()
pivot["E_rel"] = 1000 * (pivot["E90"] - pivot["E0"])

plt.close("all")
fig, axes = plt.subplots(3, 2, figsize=(12, 8))
for n_carbon, ax in zip([4, 8], axes.T):
    ax_abs, ax_rel, ax_final = ax

    df_plot = pivot[pivot["n_carbon"] == n_carbon]
    ax_final.plot(df_plot["n_dets"], df_plot["E_rel"], marker="o", color='gray')
    ax_final.set_ylabel("Relative Energy / mHa")
    ax_final.set_xlabel("Number of determinants")
    ax_final.set_title(f"C{n_carbon}H4")

    for idx_dets, n_dets in enumerate(sorted(df_final.n_dets.unique())):
        df_plot_full = full_pivot[(full_pivot["n_carbon"] == n_carbon) & (full_pivot["n_dets"] == n_dets)]

        ax_abs.plot(df_plot_full["opt/step"] / 1000, df_plot_full["E0"], label=None, ls='--', color=f"C{idx_dets}")
        ax_abs.plot(df_plot_full["opt/step"] / 1000, df_plot_full["E90"], label=f"{n_dets} dets", ls='-', color=f"C{idx_dets}")
        ax_rel.set_xlabel("Steps / k")
        ax_abs.set_ylabel("Energy / Ha")

        ax_rel.plot(df_plot_full["opt/step"] / 1000, df_plot_full["E_rel"], ls="-", color=f"C{idx_dets}")
        ax_rel.set_xlabel("Steps / k")
        ax_rel.set_ylabel("Relative Energy / mHa")
        ax_rel.set_ylim([df_plot.E_rel.min()-10, df_plot.E_rel.max()+10])

        E_final = df_plot[df_plot["n_dets"] == n_dets].E_rel.mean()
        ax_final.plot([n_dets], [E_final], marker="o", color=f"C{idx_dets}")

    ax_abs.legend()
    ax_abs.set_ylim([df_plot.E0.min()-0.01, df_plot.E90.min()+0.1])
    # ax_final.set_ylim([0, df_plot.E_rel.max()+5])

    for a in ax:
        a.grid(alpha=0.5)
fig.tight_layout()




