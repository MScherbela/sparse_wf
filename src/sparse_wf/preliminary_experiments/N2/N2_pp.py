# %%
import bokeh.layouts
import bokeh.model
import bokeh.models
import bokeh.palettes
import wandb
import pandas as pd
import bokeh.plotting as bpl
import bokeh

# %%

api = wandb.Api()
runs = api.runs("tum_daml_nicholas/N2")

all_data = []
for r in runs:
    df = pd.DataFrame(r.scan_history(["opt/step", "opt/E", "opt/E_std"], page_size=10_000))
    df["cutoff"] = 3.0
    df["run_name"] = r.name
    df["distance"] = r.config["molecule_args"]["chain_args"]["distance"]
    all_data.append(df)
df_all = pd.concat(all_data, ignore_index=True)
df_all.to_csv("N2_energies.csv", index=False)

# %%
dist_ref = 2.07
df_all = pd.read_csv("N2_energies.csv")
df_ref = pd.read_csv("N2_mrci.csv")[["distance", "E"]].rename(columns={"E": "E_mrci"})
E0_ref = df_ref[df_ref["distance"] == dist_ref]["E_mrci"].values[0]
df_ref["delta_E"] = (df_ref["E_mrci"] - E0_ref) * 1000
df_ref.set_index("distance", inplace=True)

df_dist_ref = df_all[df_all["distance"] == dist_ref]
df_dist_ref = df_dist_ref.rename(columns={"opt/E": "opt/E_ref"})
df_dist_ref = df_dist_ref[["opt/step", "opt/E_ref"]]
df_all = df_all.merge(df_dist_ref, how="left", on="opt/step")

df_all["delta_E"] = (df_all["opt/E"] - df_all["opt/E_ref"]) * 1000

# %%
distances = sorted(df_all.distance.unique())
smoothing_window=4000

fig_kwargs = dict(width=400, height=400, tools="box_zoom,hover,save", tooltips=[("x", "$snap_x"), ("y", "$snap_y")])
p_opt = bpl.figure(title="Optimization Energies", x_axis_label="Step", y_axis_label="(E - E_2.07) / mHa", **fig_kwargs)
p_final = bpl.figure(title="Final Energies", x_axis_label="Distance", y_axis_label="(E - E_2.07) / mHa", **fig_kwargs)
p_final_error = bpl.figure(title="Final Error", x_axis_label="Distance", y_axis_label="(E_E_2.07) - (E_ref - E_ref_2.07) / mHa", **fig_kwargs)

data_final = []
colors = bokeh.palettes.Category10[10][: len(distances)]
for color, d in zip(colors, distances):
    df_dist = df_all[df_all["distance"] == d].copy()
    df_dist["delta_E_smooth"] = df_dist["delta_E"].rolling(window=smoothing_window).mean()
    p_opt.line(df_dist["opt/step"], df_dist["delta_E_smooth"], legend_label=f"{d:.2f}", color=color)
    p_opt.add_layout(
        bokeh.models.Span(
            location=df_ref.loc[d, "delta_E"], dimension="width", line_color=color, line_dash="dashed", line_width=2
        )
    )
    data_final.append(
        dict(distance=d, delta_E=df_dist["delta_E_smooth"].iloc[-1], delta_E_ref=df_ref.loc[d, "delta_E"])
    )
p_opt.y_range.start = -2
p_opt.y_range.end = 400

df_final = pd.DataFrame(data_final)
p_final.line(df_final["distance"], df_final["delta_E"], legend_label="SWANN", color="red")
p_final.line(df_final["distance"], df_final["delta_E_ref"], legend_label="MRCI", color="black")

p_final_error.line(df_final["distance"], df_final["delta_E"] - df_final["delta_E_ref"], color="red")


grid = bokeh.layouts.gridplot([[p_opt, p_final, p_final_error]])
bpl.output_notebook()
bpl.show(grid)
