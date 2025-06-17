# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sparse_wf.plot_utils import (
    get_outlier_mask,
    extrapolate_relative_energy,
    MILLIHARTREE,
    COLOR_PALETTE,
    scale_lightness,
    COLOR_FIRE,
)

plt.style.use(["science", "grid"])

df = pd.read_csv("fe2s2_energies.csv")

n_steps_min = 65_000
smoothing = 500
n_steps_eval = 5000
pivot = df.pivot_table(index="opt/step", columns="geom", values=["opt/E", "grad"]).rename(columns={"opt/E": "E"})
pivot["grad"] = pivot["grad"] ** 2
# pivot = pivot.dropna()
is_outlier = pivot.apply(get_outlier_mask, axis=0)
pivot = pivot.mask(is_outlier, np.nan).ffill(limit=5)
df_last = pivot.dropna().iloc[-n_steps_eval:].mean()
if smoothing > 1:
    pivot_smooth = pivot.rolling(smoothing).mean().iloc[smoothing :: max(smoothing // 5, 1)]
else:
    pivot_smooth = pivot
pivot_smooth = pivot_smooth[pivot_smooth.index >= n_steps_min]
geoms = list(pivot["E"])

fig, ax_ext = plt.subplots(1, 1, figsize=(6, 4))

fig, axes = plt.subplots(1, 2, figsize=(9, 5))
ax_bar, ax_mae = axes
# E_fit_values, slope = fit_with_joint_slope(pivot_smooth["grad"].values.T, pivot_smooth["E"].values.T)
E_fit_values, slopes = extrapolate_relative_energy(
    pivot_smooth.index,
    pivot_smooth["grad"].values.T,
    pivot_smooth["E"].values.T,
    method="same_slope",
    return_slopes=True,
    min_frac_step=0,
)
x_range = np.array([pivot_smooth["grad"].min().min(), pivot_smooth["grad"].max().max()])

df_agg = []
for g, E_fit, slope, color in zip(geoms, E_fit_values, slopes, COLOR_PALETTE[:4]):
    ax_ext.scatter(
        pivot_smooth[("grad", g)],
        pivot_smooth[("E", g)],
        c=pivot_smooth.index,
        clim=(pivot_smooth.index.min(), pivot_smooth.index.max()),
    )
    ax_ext.plot(x_range, E_fit + slope * x_range, color=color, label=g)
    ax_ext.plot([df_last["grad"][g]], [df_last["E"][g]], color=scale_lightness(color, 0.7), marker="d")
    df_agg.append(dict(geom=g, method="ext", E=E_fit))
    df_agg.append(dict(geom=g, method="final", E=df_last["E"][g]))
ax_ext.legend()


df_agg = pd.DataFrame(df_agg)
df_agg = df_agg.pivot_table(index="method", columns="geom", values="E") * 1000
df_ref = pd.read_csv("ref_energies.csv", index_col=0) * 0.38088  # kJ/mol in mHa
df_agg = pd.concat([df_ref, df_agg], axis=0)
df_agg = df_agg.sub(df_agg.mean(axis=1), axis=0)


ground_truth_method = "composite"

methods = [
    ("composite", "composite", "k"),
    ("UHF/CCSD(T)/CBS_23", "CCSD(T)/CBS", "dimgray"),
    # ("UHF", "UHF", "dimgray"),
    # ("r2SCAN", "r$^2$SCAN", "teal"),
    ("PBE0", "PBE0", "C0"),
    ("B3LYP", "B3LYP", "navy"),
    ("final", "FiRE raw", COLOR_FIRE),
    ("ext", "FiRE ext", "red"),
]

geom_labels = df_agg.index
for idx_method, (method, label, color) in enumerate(methods):
    x = np.arange(4) + 0.8 * idx_method / len(methods) - 0.4
    ax_bar.bar(x, df_agg.loc[method], label=label, width=0.8 / len(methods), color=color, align="edge")
    mae = np.mean(np.abs(df_agg.loc[method] - df_agg.loc[ground_truth_method]))
    ax_mae.barh([idx_method], [mae], color=color, label=label, height=0.8)

uncertainty = np.sqrt(8**2 + 10**2) * 0.38088  # 8 kJ/mol basis, 10 kJ/mol multireference
for i, E in enumerate(df_agg.loc[ground_truth_method]):
    x_vals = np.array([i - 0.4, i + 0.4])
    ax_bar.plot(x_vals, [E, E], color="k", ls="--")
    ax_bar.fill_between(x_vals, E - uncertainty, E + uncertainty, color="k", alpha=0.1)
ax_mae.axvline(uncertainty, color="k", ls="--")

ax_mae.set_yticks(np.arange(len(methods)))
ax_mae.set_yticklabels([m[1] for m in methods])
ax_mae.set_ylim([0.5, None])
ax_mae.invert_yaxis()
ax_mae.set_xlabel(f"MAE vs. {ground_truth_method} " + MILLIHARTREE)
ax_bar.set_ylabel("($E$ - $E_\\text{HC})$ " + MILLIHARTREE)
ax_bar.legend(ncol=1)
ax_bar.set_xticks(np.arange(4))
ax_bar.set_xticklabels(df_agg.columns)
# ax_bar.set_ylim([0, 120])

fig.tight_layout()


# %%
def get_time_resolved_extrapolation(df, frac_min=0.3, n_steps=100):
    df = df.dropna()
    frac_max = np.linspace(frac_min, 1.0, n_steps)
    frac_min = 0.8 * frac_max
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


smoothing_fit = 5000
smoothing_last = 10_000

df_smooth = pivot.dropna().ffill(limit=10)
if smoothing_fit > 1:
    df_smooth = df_smooth.rolling(smoothing_fit).mean().iloc[smoothing_fit :: max(smoothing_fit // 10, 1)]
df_last = (
    pivot["E"].dropna().ffill(limit=10).rolling(smoothing_last).mean().iloc[smoothing_last :: smoothing_last // 10]
)


steps_fit, E_fit = get_time_resolved_extrapolation(df_smooth)
df_fit = pd.DataFrame(data=E_fit, index=steps_fit, columns=pivot["E"].columns)

df_last = df_last.sub(df_last.mean(axis=1), axis=0) * 1000
df_fit = df_fit.sub(df_fit.mean(axis=1), axis=0) * 1000
ref_dict = df_agg.loc["composite"]
ref_dict -= ref_dict.mean()

mae_last = np.abs(df_last.iloc[-1] - ref_dict).mean()
mae_fit = np.abs(df_fit.iloc[-1] - ref_dict).mean()
print(f"MAE last: {mae_last:.1f} mEh")
print(f"MAE fit : {mae_fit:.1f} mEh")

fig, axes = plt.subplots(1, 5, figsize=(8, 4))
for idx_g, (ax, geom) in enumerate(zip(axes, ["HC", "HS", "HFe", "HFe2"])):
    ax.plot(df_last.index / 1000, df_last[geom])
    ax.plot(df_fit.index / 1000, df_fit[geom])
    ax.set_title(geom)
    # ax.axhline(ref_dict[geom] - ref_dict[geom_ref], color="k")
    ax.axhline(ref_dict[geom], color="k")
    ax.set_ylim(ref_dict[geom] + np.array([-7, 7]))
ax_mae = axes[-1]
mae_last = np.abs(df_last - ref_dict).mean(axis=1)
mae_fit = np.abs(df_fit - ref_dict).mean(axis=1)
ax_mae.plot(mae_last.index / 1000, mae_last, label="last 5k steps")
ax_mae.plot(mae_fit.index / 1000, mae_fit, label="extrapolated")
ax_mae.set_ylim([0, 5])
ax_mae.set_title("MAE")
fig.legend(loc="upper right", ncol=2)
fig.tight_layout()
fig.subplots_adjust(top=0.85)
