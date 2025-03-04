# %%
import pandas as pd
import numpy as np
from sparse_wf.plot_utils import get_outlier_mask, get_colors_from_cmap, savefig
import itertools
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scienceplots
plt.style.use(['science', 'grid'])


def fractional_smoothing(x, indices, frac):
    y = np.zeros_like(indices, dtype=float)
    for i, idx in enumerate(indices):
        y[i] = np.mean(x[int(idx * (1 - frac)) : idx])
    return y

window_outlier = 100
n_steps_min = 500
n_samples = 500
frac_smoothing = 0.1

df_all = pd.read_csv("cumulene_pp_energies.csv")
df_all = df_all[df_all["cutoff"] == 3.0]
smooth_data = []
for n_carbon, angle in itertools.product(df_all.n_carbon.unique(), df_all.angle.unique()):
    df = df_all[(df_all["n_carbon"] == n_carbon) & (df_all["angle"] == angle)].copy()
    mask = get_outlier_mask(df["opt/E"], window_size=window_outlier)
    mask = mask | get_outlier_mask(df["opt/E_std"], window_size=window_outlier)
    print(f"Found {mask.sum()} outliers for n={n_carbon}, angle={angle}")
    df = df[~mask]
    indices = np.geomspace(n_steps_min, len(df), n_samples).astype(int)
    indices = np.unique(indices)

    smooth_data.append(
        pd.DataFrame(
            {
                "step": indices,
                "E": fractional_smoothing(df["opt/E"], indices, frac_smoothing),
                "E_std": fractional_smoothing(df["opt/E_std"], indices, frac_smoothing),
                "n": n_carbon,
                "angle": angle,
            }
        )
    )
df_smooth = pd.concat(smooth_data)

# %%
angle = 0
df_fit = df_smooth[df_smooth["angle"] == angle]

def powerlaw(t, E_inf, tau, k):
    return E_inf + (t/tau)**-k

def powerlaw_combined(t, n, alpha, beta, const):
    return t**(-alpha) * n**beta * np.exp(const)

steps_min = 2000
n_samples = 500
eps_sigma = 1e-3

fig, axes = plt.subplots(3, 3, figsize=(7, 7))
fit_values = []
n_values = [2, 4, 6, 8, 12, 16, 20, 24]

for ax, n in zip(axes.flatten(), n_values):
    df = df_fit[df_fit["n"] == n]

    E_min, E_max = df.E.min(), df.E.max()
    p0 = [E_min, 10, 1]
    popt, pcov = curve_fit(powerlaw, df.step, df.E, sigma=df.E - E_min + eps_sigma, p0=p0, bounds=([-np.inf, 0, 0], [E_min, np.inf, np.inf]))
    E_inf, tau, k = popt
    E_fit = powerlaw(df.step, *popt)
    ax.plot(df.step, df.E - E_inf, label="Data")
    ax.plot(df.step, E_fit - E_inf, label="Fit")

    E_final = df.E.min() - 1e-3
    ax.set_xscale("log")
    ax.set_yscale("log")
    fit_values.append(dict(n=n, E_inf=E_inf, tau=tau, k=k))
df_fitparams = pd.DataFrame(fit_values)
df_fit = df_fit.merge(df_fitparams[["n", "E_inf"]], on="n")

X_combined = np.stack([-np.log(df_fit.step), np.log(df_fit.n), np.ones_like(df_fit.n)], axis=1)
y_combined = np.log(df_fit.E - df_fit.E_inf)
popt = np.linalg.lstsq(X_combined, y_combined, rcond=None)[0]

fig, ax_combined = plt.subplots(1, 1, figsize=(6, 4))
colors = get_colors_from_cmap("viridis", np.linspace(0, 0.8, len(n_values)))
for color, n in zip(colors, n_values):
    df = df_fit[df_fit["n"] == n]
    ax_combined.plot(df.step, df.E - df.E_inf,color=color, alpha=0.5)
    ax_combined.plot(df.step, powerlaw_combined(df.step, n, *popt), color=color, label=f"C{n}H4")
y_range = powerlaw_combined(np.array([df_fit.step.max(), df_fit.step.min()]), np.array([df_fit.n.min(), df_fit.n.max()]), *popt)
ax_combined.set_ylim(y_range * np.array([0.9, 1.1]))
ax_combined.set_xscale("log")
ax_combined.set_yscale("log")
ax_combined.legend(ncol=2, loc="lower left")
ax_combined.set_title(f"$E - E_\infty \propto t^{{-{popt[0]:.1f}}}\\; n^{{{popt[1]:.1f}}}$")
ax_combined.set_xlabel("optimization step")
ax_combined.set_ylabel("$E - E_\infty$ / m$E_h$")
savefig(fig, "scaling_law")


# combined = np.polyfit(np.log(df_fit.step), np.log(df_fit.E - df_fit.E_inf), 1)
# popt, pcov = curve_fit(powerlaw, df_fit.n, df_fit.E - df_fit.E_in, sigma=df_fit.E_inf, p0=[0, 1, 1])

# fig, axes = plt.subplots(1, 2, figsize=(6, 4))
# axes[0].plot(df_fitparams.n, df_fitparams.k, marker="o")
# axes[1].plot(df_fitparams.n, df_fitparams.tau, marker="o")
# axes[1].set_xscale("log")
# axes[1].set_yscale("log")
