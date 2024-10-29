#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ast

def plot_and_fit(ax, x, y, label="", color="C0", n_fit_min=0):
    include_in_fit = x >= n_fit_min
    x_fit, y_fit = x[include_in_fit], y[include_in_fit]
    fit_coeffs = np.polyfit(np.log(x_fit), np.log(y_fit), 1)
    exponent = fit_coeffs[0]
    y_fitted = np.exp(np.polyval(fit_coeffs, np.log(x_fit)))
    ax.plot(x_fit, y_fitted, color=color, linestyle="--", alpha=0.5)
    ax.plot(x, y, color=color, label=" ".join([label, f"$O(n^{{{exponent:.2f}}})$"]), marker="o")

BATCH_SIZE_PLOT = 32
data_fname = "timings_with_ecp.txt"
# data_fname = "timings_no_ecp.txt"
data = []
with open(data_fname, "r") as f:
    for line in f:
        data.append(ast.literal_eval(line))
df = pd.DataFrame(data)
df.t_wf_full = df.t_wf_full * BATCH_SIZE_PLOT / df.batch_size
df.t_wf_lr = df.t_wf_lr * BATCH_SIZE_PLOT / df.batch_size
df.t_E_kin = df.t_E_kin * BATCH_SIZE_PLOT / df.batch_size
df.t_E_pot = df.t_E_pot * BATCH_SIZE_PLOT / df.batch_size
df["t_sampling"] = df.t_wf_full + df.n_el * df.t_wf_lr
df["t_total"] = df.t_sampling + df.t_E_kin + df.t_E_pot


fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharey=True)

ax_tot = axes[0, 0]
ax_psi = axes[0, 1]
ax_Ekin = axes[1, 0]
ax_Epot = axes[1, 1]

plot_and_fit(ax_tot, df.n_el, df.t_total, color="C0", n_fit_min=50)
plot_and_fit(ax_psi, df.n_el, df.t_wf_full, label="Full", color="C0", n_fit_min=50)
plot_and_fit(ax_psi, df.n_el, df.t_wf_lr, label="Low-rank", color="C1", n_fit_min=50)
plot_and_fit(ax_Ekin, df.n_el, df.t_E_kin, color="C0", n_fit_min=50)
plot_and_fit(ax_Epot, df.n_el, df.t_E_pot, color="C0", n_fit_min=50)

ax_tot.set_title("Total time")
ax_psi.set_title("Wavefunction evaluation")
ax_Ekin.set_title("Kinetic energy")
ax_Epot.set_title("Potential energy")

for ax in axes.flatten():
    ax.set_xlabel("Number of electrons")
    ax.set_ylabel("Time (s)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    xticks = np.round(np.geomspace(16, 362, 10)).astype(int)
    ax.set_xticks(xticks)
    ax.set_xticks([], minor=True)
    ax.set_xticklabels([str(x) for x in xticks])
    ax.legend()

fig.tight_layout()
fig.savefig(data_fname.replace(".txt", ".png"))



