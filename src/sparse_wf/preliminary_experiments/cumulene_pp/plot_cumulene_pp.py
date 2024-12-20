#%%
import pandas as pd
import wandb
import numpy as np
import bokeh.plotting as bpl
import bokeh
import scipy.optimize

df_ref = pd.read_csv("../../../../data/energies.csv")
df_ref = df_ref[df_ref.geom_comment.str.startswith("cumulene_") & (df_ref.model == "CCSD(T)") & (df_ref.model_comment == "Extrapolate(4,cc-core)")]
df_ref = df_ref.groupby("geom_comment")[["E"]].mean()

window = 10_000
df = pd.read_csv("2024-11-18_cumulene_pp_energies.csv")
df["geom"] = df.n_carbon.astype(str) + "_" + df.angle.astype(str) +"_" + df.cutoff.astype(str)
for geom in df.geom.unique():
    filt = df.geom == geom
    df.loc[filt, "E_smooth"] = df.loc[filt, "E"].rolling(window=1000).mean()

pivot = df.pivot_table(index="step", columns=["angle", "n_carbon"], values="E_smooth", aggfunc="mean")
df_delta_E = (pivot[90] - pivot[0]) * 1000

df_final = []
for n_carbon in df.n_carbon.unique():
    idx_final = df_delta_E[n_carbon].last_valid_index()
    df_final.append(dict(n_carbon=n_carbon, delta_E=df_delta_E.loc[idx_final, n_carbon]))
df_final = pd.DataFrame(df_final)

#%%
figure_width = 500
p_opt = bpl.figure(width=figure_width, title="Cumulene PP", x_axis_label="Optimization Step", y_axis_label="E_90 - E_0 / mHa", tools="box_zoom,hover,save", tooltips=[("step", "$x"), ("delta", "$y")])
n_carbon_values = df.n_carbon.unique()
colors = bokeh.palettes.Category10[10][:len(n_carbon_values)]
for n_carbon, color in zip(n_carbon_values, colors):
    p_opt.line(pivot.index, df_delta_E[n_carbon], legend_label=f"C{n_carbon}H4", line_width=2, color=color)
    # try:
    #     E_90 = df_ref.loc[f"cumulene_C{n_carbon}H4_90deg", "E"]
    #     E_0 = df_ref.loc[f"cumulene_C{n_carbon}H4_0deg", "E"]
    #     delta_E_ref = (E_90 - E_0) * 1000
    #     p_opt.add_layout(bokeh.models.Span(location=delta_E_ref, dimension="width", line_color=color, line_dash="dashed", line_width=2))
    # except KeyError:
    #     print(f"No reference for {n_carbon=}")


p_n_carbon = bpl.figure(width=figure_width, title="Cumulene PP", x_axis_label="Number of Carbons", y_axis_label="Delta E / mHa", tools="box_zoom,hover,save", tooltips=[("n_carbon", "$x"), ("delta", "$y")])
p_n_carbon.line(df_final.n_carbon, df_final.delta_E, line_width=2, color="black")
for n_carbon, delta_E, color in zip(df_final.n_carbon, df_final.delta_E, colors):
    p_n_carbon.scatter([n_carbon], [delta_E], size=10, color=color)

def fit_model(n_carbon, scale):
    return scale / n_carbon

popt = scipy.optimize.curve_fit(fit_model, df_final[df_final.n_carbon > 2].n_carbon, df_final[df_final.n_carbon > 2].delta_E, p0=[200])[0]

n_carbon_fitted = np.linspace(2, 16, 200)
delta_E_fitted = fit_model(n_carbon_fitted, *popt)
p_n_carbon.line(n_carbon_fitted, delta_E_fitted, color="gray", line_dash="dashed")

for p in [p_opt, p_n_carbon]:
    p.y_range.start = 0
    p.y_range.end = 125

plot_grid = bpl.gridplot([p_opt, p_n_carbon], ncols=2)
bpl.output_notebook()
bpl.show(plot_grid)

