#%%
import bokeh.layouts
import wandb
import pandas as pd

api = wandb.Api()
runs = api.runs("tum_daml_nicholas/mpconf")
runs = [r for r in runs if r.name.startswith("mpconf_WG") or r.name.startswith("mpconf_hf_v2_WG") or r.name.startswith("mpconf_damp1e-3_WG")]

data = []
for r in runs:
    print(r.name)
    if "hf_v2" in r.name:
        model = "hf_like"
    elif "mpconf_damp1e-3" in r.name:
        model = "swann_high_damp"
    else:
        model = "swann_low_damp"
    metadata = dict(
        molecule=r.config["molecule_args"]["database_args"]["comment"],
        run_name=r.name,
        model=model)

    for h in r.scan_history(["opt/step", "opt/E", "opt/E_std"], page_size=10_000):
        data.append(h | metadata)

df_all = pd.DataFrame(data)
df_all.to_csv("mpconf_energies.csv", index=False)

#%%
import bokeh.plotting as bpl
import bokeh

window_length = 5_000
models = ["hf_like", "swann_low_damp", "swann_high_damp"]

df_all = pd.read_csv("mpconf_energies.csv")
molecules = sorted(df_all["molecule"].unique())
pivot_all = df_all.pivot_table(index="opt/step", columns=["model", "molecule"], values="opt/E", aggfunc="mean")
for model in models:
    E_smooth = 0.0
    molecules = list(pivot_all[model])
    for m in molecules:
        smoothed = pivot_all[(model, m)].rolling(window=window_length).mean()
        pivot_all.loc[:, (model, f"E_smooth_{m}")] = smoothed
        E_smooth += smoothed
    E_mean_smooth = E_smooth / len(molecules)
    for m in molecules:
        pivot_all.loc[:, (model, f"delta_{m}")] = (pivot_all.loc[:, (model, f"E_smooth_{m}")] - E_mean_smooth) * 1000


plots = []
for model, title in zip(models, ["HF-like model (high damping)", "Full SWANN (low damping)", "Full SWANN (high damping)"]):
    pivot = pivot_all[model]
    molecules = [m for m in list(pivot) if m.startswith("WG_")]

    df_ref = pd.read_csv("../../../../data/energies.csv")
    df_ref = df_ref[(df_ref.model == "CCSD(T)") & (df_ref.model_comment == "CBS") & (df_ref.geom_comment.isin(molecules))]
    df_ref = df_ref.groupby("geom_comment")[["E"]].mean()
    df_ref["delta"] = (df_ref - df_ref.mean(axis=0)) * 1000

    p = bpl.figure(width=400, height=600, title=f"Relative Energies vs CCSD(T): {title}", x_axis_label="Optimization Step / k", y_axis_label="E - E_mean / mHa", tools="box_zoom,hover,save", tooltips=[("step", "$x"), ("delta", "$y")])
    for m, color in zip(molecules, bokeh.palettes.HighContrast3):
        p.line(pivot.index / 1000, pivot[f"delta_{m}"], legend_label=m, color=color, line_width=2)
        p.add_layout(bokeh.models.Span(location=df_ref.loc[m, "delta"], dimension="width", line_color=color, line_dash="dashed", line_width=2))
    p.y_range.start = -10
    p.y_range.end = 10
    plots.append(p)

grid = bokeh.layouts.gridplot(plots, ncols=3)
bpl.output_notebook()
bpl.show(grid)
bokeh.io.export_png(grid, filename="mpconf.png")





