#%%
import bokeh.layouts
import wandb
import pandas as pd

api = wandb.Api()
runs = api.runs("tum_daml_nicholas/mpconf")
runs = [r for r in runs if r.name.startswith("mpconf_")]

data = []
for r in runs:
    print(r.name)
    metadata = dict(molecule=r.config["molecule_args"]["database_args"]["comment"],
    run_name=r.name,)

    for h in r.scan_history(["opt/step", "opt/E", "opt/E_std"], page_size=10_000):
        data.append(h | metadata)

df_all = pd.DataFrame(data)
df_all.to_csv("mpconf_energies.csv", index=False)

#%%
import bokeh.plotting as bpl
import bokeh

window_length = 10_000



df_all = pd.read_csv("mpconf_energies.csv")
molecules = sorted(df_all["molecule"].unique())
pivot = df_all.pivot_table(index="opt/step", columns="molecule", values="opt/E", aggfunc="mean")
E_smooth = 0.0
for m in molecules:
    smoothed = pivot[m].rolling(window=window_length).mean()
    pivot[f"E_smooth_{m}"] = smoothed
    E_smooth += smoothed
pivot["E_mean_smooth"] = E_smooth / len(molecules)
for m in molecules:
    pivot[f"delta_{m}"] = (pivot[f"E_smooth_{m}"] - pivot["E_mean_smooth"]) * 1000

df_ref = pd.read_csv("../../../../data/energies.csv")
df_ref = df_ref[(df_ref.model == "CCSD(T)") & (df_ref.model_comment == "CBS") & (df_ref.geom_comment.isin(molecules))]
df_ref = df_ref.groupby("geom_comment")[["E"]].mean()
df_ref["delta"] = (df_ref - df_ref.mean(axis=0)) * 1000

p = bpl.figure(width=900, title="Relative MPConf Energies vs CCSD(T)", x_axis_label="Optimization Step", y_axis_label="E - E_mean / mHa", tools="box_zoom,hover,save", tooltips=[("step", "$x"), ("delta", "$y")])
for m, color in zip(molecules, bokeh.palettes.HighContrast3):
    p.line(pivot.index, pivot[f"delta_{m}"], legend_label=m, color=color, line_width=2)
    p.add_layout(bokeh.models.Span(location=df_ref.loc[m, "delta"], dimension="width", line_color=color, line_dash="dashed", line_width=2))
p.y_range.start = -10
p.y_range.end = 10
bpl.output_notebook()
bpl.show(p)

bokeh.io.export_png(p, filename="mpconf.png")





