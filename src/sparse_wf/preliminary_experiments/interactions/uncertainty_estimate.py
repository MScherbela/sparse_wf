# %%
import pandas as pd
import numpy as np
from sparse_wf.plot_utils import get_outlier_mask
import matplotlib.pyplot as plt

df = pd.read_csv("interaction_energies.csv")

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
for molecule in df.molecule.unique():
    df_mol = df[df.molecule == molecule]
    pivot = df_mol.pivot_table(index="opt/step", columns="geom", values=["opt/E", "opt/E_std"], aggfunc="mean")
    pivot["deltaE"] = pivot["opt/E"]["dissociated"] - pivot["opt/E"]["equilibrium"]
    pivot = pivot[pivot.deltaE.notna()]
    is_outlier = get_outlier_mask(pivot["deltaE"])
    pivot = pivot[~is_outlier]
    # pivot = pivot.rolling(window=10).mean() # turn on to introduce artificial autocorrelation as a sanity check
    pivot = pivot.iloc[-5000:]

    batch_size = 4096
    n_steps = len(pivot)
    n_samples_total = n_steps * batch_size
    E_std_diss = np.sqrt(np.mean(pivot["opt/E_std"]["dissociated"] ** 2))
    E_std_eq = np.sqrt(np.mean(pivot["opt/E_std"]["equilibrium"] ** 2))

    std_err_batch = 1000 * np.sqrt(E_std_diss**2 + E_std_eq**2) / np.sqrt(n_samples_total)
    std_err_time = 1000 * np.std(pivot["deltaE"]) / np.sqrt(n_steps)

    print(
        f"{molecule:<25}: Stderr across batch: {std_err_batch:.3f} mHa, across time: {std_err_time:.3f} mHa, ratio: {std_err_batch / std_err_time:.3f}"
    )

    dE = pivot["deltaE"]
    dE_normalized = (dE - np.mean(dE)) / np.std(dE)
    ac = np.correlate(dE_normalized, dE_normalized, mode="same")
    ac = ac[len(ac) // 2 :]
    ac /= n_steps - np.arange(len(ac))
    ax.plot(ac[:20], label=molecule)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
