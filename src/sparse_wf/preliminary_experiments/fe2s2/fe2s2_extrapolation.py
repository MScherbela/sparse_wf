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
    savefig,
    fit_with_joint_slope,
    COLOR_FIRE,
)
import scienceplots

plt.style.use(["science", "grid"])

df = pd.read_csv("fe2s2_energies.csv")

n_steps_min = 35_000
smoothing = 500
n_steps_eval = 2000
pivot = df.pivot_table(index="opt/step", columns="geom", values=["opt/E", "grad"]).rename(columns={"opt/E": "E"})
pivot = pivot.dropna()
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
    ax_ext.scatter(pivot_smooth[("grad", g)], pivot_smooth[("E", g)], label=g, color=color)
    ax_ext.plot(x_range, E_fit + slope * x_range, color=scale_lightness(color, 0.7))
    ax_ext.plot([df_last["grad"][g]], [df_last["E"][g]], color=scale_lightness(color, 0.7), marker="d")
    df_agg.append(dict(geom=g, method="ext", E=E_fit))
    df_agg.append(dict(geom=g, method="final", E=df_last["E"][g]))
ax_ext.legend()


df_agg = pd.DataFrame(df_agg)
df_HC = df_agg[df_agg["geom"] == "HC"].rename(columns={"E": "E_HC"})
df_agg = df_agg.merge(df_HC[["method", "E_HC"]], on=["method"])
df_agg["delta"] = (df_agg["E"] - df_agg["E_HC"]) * 1000
df_agg = df_agg[["geom", "method", "delta"]]
df_agg = df_agg[df_agg["geom"] != "HC"]

# energies_ref = {
#     "HS": [51.6, 77.14, 51.609],
#     "HFe": [87.8, 116, 87.831],
#     "HFe2": [76.1, 143, 76.138],
# }
# methods = ["CCSD(T)", "UHF", "composite"]
# df_ref = pd.DataFrame(energies_ref, index=methods)
df_ref = pd.read_csv("ref_energies.csv", index_col=0) * 0.38088  # kJ/mol in mHa
# df_ref = df_ref.loc[["PBE0", "B3LYP"]]
# df_ref = pd.concat([df_ref, df_ref], axis=0)

df_ref = df_ref.reset_index(names="method").melt(id_vars="method", var_name="geom", value_name="delta")
df_agg = pd.concat([df_agg, df_ref])
df_agg = df_agg.pivot_table(index="geom", columns="method", values="delta")
df_agg = df_agg[df_agg.index != "HC"]
print(df_agg)


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
    x = np.arange(3) + 0.8 * idx_method / len(methods) - 0.4
    ax_bar.bar(x, df_agg[method], label=label, width=0.8 / len(methods), color=color, align="edge")
    mae = np.mean(np.abs(df_agg[method] - df_agg[ground_truth_method]))
    ax_mae.barh([idx_method], [mae], color=color, label=label, height=0.8)

uncertainty = np.sqrt(8**2 + 10**2) * 0.38088  # 8 kJ/mol basis, 10 kJ/mol multireference
for i, E in enumerate(df_agg[ground_truth_method]):
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
ax_bar.legend(ncol=2)
ax_bar.set_xticks(np.arange(3))
ax_bar.set_xticklabels(df_agg.index)
ax_bar.set_ylim([0, 120])

fig.tight_layout()
# %%
# geom_ref = "HS"
dE_smooth = pivot["E"]
# dE_smooth = dE_smooth.sub(dE_smooth[geom_ref], axis=0).ffill(limit=100)
dE_smooth = dE_smooth.sub(dE_smooth.mean(axis=1), axis=0).ffill(limit=10)
dE_smooth = dE_smooth.rolling(500).mean()
ref_dict = pd.concat([df_agg["composite"], pd.Series({"HC": 0})])

fig, axes = plt.subplots(1, 4, figsize=(8, 4))
for ax, geom in zip(axes, ["HC", "HFe", "HFe2", "HS"]):
    ax.plot(dE_smooth.index / 1000, dE_smooth[geom] * 1000)
    ax.set_title(geom)
    # ax.axhline(ref_dict[geom] - ref_dict[geom_ref], color="k")
    ax.axhline(ref_dict[geom] - ref_dict.mean(), color="k")
    ax.set_ylim(dE_smooth[geom].dropna().iloc[-1] * 1000 + np.array([-15, 15]))
