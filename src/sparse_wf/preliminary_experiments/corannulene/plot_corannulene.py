#%%
import pandas as pd
import numpy as np
from sparse_wf.plot_utils import get_outlier_mask, extrapolate_relative_energy, COLOR_PALETTE, MILLIHARTREE, COLOR_FIRE
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(["science", "grid"])

def get_time_resolved_extrapolation(df, frac_min=0.5, n_steps=100):
    df = df.dropna()
    frac_max = np.linspace(frac_min, 1.0, n_steps)
    frac_min = 0.7 * frac_max
    E_fit = []
    steps = np.max(df.index.values) * frac_max
    for fmin, fmax in zip(frac_min, frac_max):
        E_fit.append(
            extrapolate_relative_energy(
                df.index,
                df["grad"].values.T,
                df["E"].values.T,
                min_frac_step=fmin,
                max_frac_step=fmax,
            )
        )
    return steps, np.array(E_fit)

smoothing = 500
steps_min = 30_000


df = pd.read_csv("corannulene_energies.csv")
df = df.rename(columns={"opt/E": "E"})
pivot = df.pivot_table(index="opt/step", columns="geom", values=["E", "grad"], aggfunc="mean")
pivot["grad"] = pivot["grad"]**2
is_outlier = pivot.apply(get_outlier_mask, axis=0)
pivot = pivot.mask(is_outlier, np.nan).ffill(limit=5)

pivot_smooth = pivot.rolling(smoothing).mean().iloc[smoothing::smoothing//10]
pivot_smooth = pivot_smooth[pivot_smooth.index >= steps_min]

E_fit, slopes = extrapolate_relative_energy(
    pivot_smooth.index,
    pivot_smooth["grad"].values.T,
    pivot_smooth["E"].values.T,
    method="same_slope",
    min_frac_step=0,
    return_slopes=True,
)

geoms = pivot["grad"].columns
fig, axes = plt.subplots(1, 2, figsize=(6, 4))
ax_scatter, ax_t = axes
x_range = np.array([np.nanmin(pivot_smooth["grad"].values), np.nanmax(pivot_smooth["grad"].values)])

for idx_g, g in enumerate(geoms):
    ax_scatter.scatter(pivot_smooth["grad"][g], pivot_smooth["E"][g], label=g)
    ax_scatter.plot(x_range, E_fit[idx_g] + slopes[idx_g] * x_range, color="k")

dE = (E_fit[0] - E_fit[1]) * 1000

print(f"{dE:.1f} mEh")
ax_scatter.legend()

references = {
    "CCSD": 13.115,
    "CCSD(T)": 22.661,
}

steps, E_fit = get_time_resolved_extrapolation(pivot)
dE = (E_fit[:, 0] - E_fit[:, 1]) * 1000
ax_t.plot(steps / 1000, dE, color=COLOR_FIRE, label="FiRE, $c=5a_0$")
ax_t.set_xlabel("opt steps / k")
ax_t.set_ylabel("interaction energy " + MILLIHARTREE)
ax_t.set_ylim([0, 25])
for color, (k, E) in zip(COLOR_PALETTE, references.items()):
    ax_t.axhline(E, linestyle="--", label=k, color=color)
ax_t.legend()

