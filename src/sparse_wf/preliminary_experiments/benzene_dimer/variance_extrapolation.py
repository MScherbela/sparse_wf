#%%
import bokeh.palettes
import pandas as pd
import numpy as np
import bokeh.plotting as bpl
import bokeh.layouts as bl
import bokeh
from scipy.optimize import curve_fit

window = 4000

df = pd.read_csv("benzene_energies.csv")
df = df.sort_values(["dist", "opt/step"])

smoothed_data = []
for dist in df.dist.unique():
    df_dist = df[df.dist == dist]
    E_smooth = df_dist["opt/E"].rolling(window=window).mean()
    var_baseline = df_dist["opt/E_std"].rolling(window=200).mean()
    include = df_dist["opt/E_std"] < 2 * var_baseline
    variance = df_dist["opt/E_std"]**2
    variance[~include] = np.nan
    var_smooth = variance.fillna(method="ffill").rolling(window=window).mean()
    smoothed_data.append(pd.DataFrame(dict(step=df_dist["opt/step"], E_smooth=E_smooth, Var_smooth=var_smooth, dist=dist)))
df_smooth = pd.concat(smoothed_data)


tools = "box_zoom,hover,save"
p_energy = bpl.figure(width=400, title="Energy", x_axis_label="Optimization Step", y_axis_label="E / Ha", tools=tools)
p_variance = bpl.figure(width=400, title="Variance", x_axis_label="Optimization Step", y_axis_label="Var / Ha^2", tools=tools)
p_extrapolation = bpl.figure(width=400, title="Extrapolation", x_axis_label="Energy / Ha", y_axis_label="Var / Ha^2", tools=tools)


var_max = 1.0
colors = bokeh.palettes.Category10[10]
extrapolated_energies = []
for dist, color in zip(df.dist.unique(), colors):
    filt = df.dist == dist
    p_energy.line(df_smooth.loc[filt, "step"], df_smooth.loc[filt, "E_smooth"], legend_label=f"dist={dist}", line_width=2, color=color)
    p_variance.line(df_smooth.loc[filt, "step"], df_smooth.loc[filt, "Var_smooth"], legend_label=f"dist={dist}", line_width=2, color=color)
    p_extrapolation.line(df_smooth.loc[filt, "E_smooth"], df_smooth.loc[filt, "Var_smooth"], legend_label=f"dist={dist}", line_width=2, color=color)
    df_fit = df_smooth.loc[filt & (df_smooth["Var_smooth"] < var_max), ["E_smooth", "Var_smooth"]]
    (slope, E0) = np.polyfit(df_fit["Var_smooth"], df_fit["E_smooth"], 1)
    print(f"dist={dist:5.2f}: E0={E0:.4f}")
    E_fitted = np.array([E0, -75.28])
    var_fitted = (E_fitted - E0) / slope
    p_extrapolation.line(E_fitted, var_fitted, color=color, line_dash="dashed")
    extrapolated_energies.append(E0)
print("Delta E extrapolated: ", (extrapolated_energies[0] - extrapolated_energies[1]) * 1000, "mHa")




grid = bl.gridplot([p_energy, p_variance, p_extrapolation], ncols=3)
bpl.output_notebook()
bpl.show(grid)