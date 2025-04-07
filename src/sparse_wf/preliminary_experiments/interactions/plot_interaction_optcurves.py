# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sparse_wf.plot_utils import get_outlier_mask, COLOR_FIRE, COLOR_PALETTE

df_all = pd.read_csv("interaction_energies.csv")
df_ref = pd.read_csv("interaction_references.csv").set_index("molecule") * 1000

colors = {
    3: COLOR_FIRE,
    5: "orange",
    7: "red",
}

window = 2_000

molecules = sorted(df_all.molecule.unique())
fig, axes = plt.subplots(4, 3, figsize=(10, 12))
for mol, ax in zip(molecules, axes.flatten()):
    df_mol = df_all[df_all["molecule"] == mol]
    cutoffs = sorted(df_mol.cutoff.unique())
    fire_energies = []
    for cutoff in df_mol.cutoff.unique():
        # for cutoff in [3]:
        pivot = df_mol[df_mol["cutoff"] == cutoff].pivot_table(
            index="opt/step", columns="geom", values="opt/E", aggfunc="mean"
        )
        if (len(pivot.columns) != 2) or (len(pivot) < window):
            print("Not enough data for", mol, cutoff)
            continue
        pivot["deltaE"] = pivot["dissociated"] - pivot["equilibrium"]
        pivot = pivot[pivot.deltaE.notna()]
        is_outlier = get_outlier_mask(pivot["deltaE"])
        outlier_steps = pivot.index[is_outlier]
        for x in outlier_steps:
            # ax.axvline(x / 1000, color=colors[int(cutoff)], ls="--")
            print(f"Outlier at {x:5d} for {mol} with cutoff {cutoff}")
        pivot[is_outlier] = np.nan
        pivot = pivot.ffill(limit=5)
        pivot = pivot.rolling(window=window).mean()
        pivot = pivot.iloc[::10]
        label = f"c={cutoff:.0f}" if len(cutoffs) > 1 else None
        ax.plot(pivot.index / 1000, pivot["deltaE"] * 1000, color=colors[int(cutoff)], label=label)
        ax.set_title(mol)
        # twin = ax.twinx()
        # twin.plot(pivot.index / 1000, pivot["equilibrium"], color="black", ls="-")
        # twin.plot(pivot.index / 1000, pivot["dissociated"], color="dimgray", ls="-")
        # twin.set_ylim([pivot["equilibrium"].min() - 2e-3, pivot["dissociated"].min() +30e-3])

        fire_energies.append(pivot["deltaE"].iloc[-1] * 1000)
    for method, color in [("CCSD(T)", "black"), ("LapNet", COLOR_PALETTE[0])]:
        ax.axhline(df_ref.loc[mol, method], color=color, ls="-")
    if len(cutoffs) > 1:
        ax.legend()
    energies = fire_energies + [df_ref.loc[mol, "LapNet"], df_ref.loc[mol, "CCSD(T)"]]
    ax.set_ylim([min(energies) - 5, max(energies) + 5])
fig.tight_layout()
fig.savefig("interaction_energies.png")
