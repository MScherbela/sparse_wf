#%%
import bokeh.io
import bokeh.layouts
import bokeh.palettes
import bokeh.resources
import numpy as np
import pandas as pd
import ast
import bokeh.plotting as bpl
import bokeh

def plot_and_fit(p, x, y, title, color, dash="solid", n_fit_min=128, label=""):
    include_in_fit = x >= n_fit_min
    x_fit, y_fit = x[include_in_fit], y[include_in_fit]
    fit_coeffs = np.polyfit(np.log(x_fit), np.log(y_fit), 1)
    exponent = fit_coeffs[0]
    y_fitted = np.exp(np.polyval(fit_coeffs, np.log(x_fit)))

    if p is None:
        p = bpl.figure(width=500, height=400, x_axis_type="log", y_axis_type="log", tools="box_zoom,hover,save", tooltips=[("n_el", "$snap_x"), ("t", "$snap_y")])
    p.line(x_fit, y_fitted, color=color, line_dash=dash, line_width=2)

    label_kwargs = dict(legend_label=f"{label} O(n^{exponent:.1f})" ) if label else {}
    p.scatter(x, y, color=color, marker="circle" if dash == "solid" else "square", **label_kwargs)
    p.line(x, y, color=color, line_alpha=0.5, line_dash=dash)

    p.legend.location = "top_left"
    p.hover.renderers = p.renderers[1::2] # only show hover for data points, not lines
    p.title.text = title
    return p

def load_data(data_fname):
    data = []
    with open(data_fname, "r") as f:
        for line in f:
            data.append(ast.literal_eval(line))
    return pd.DataFrame(data)



BATCH_SIZE_PLOT = 32

data_fnames = [
    "/home/scherbelam20/develop/sparse_wf/src/sparse_wf/preliminary_experiments/scaling/timings_no_ecp.txt",
    "/home/scherbelam20/develop/sparse_wf/src/sparse_wf/preliminary_experiments/scaling/timings_with_ecp.txt",
    "/storage/scherbelam20/runs/lapnet_scaling/timings_ferminet.txt",
    "/storage/scherbelam20/runs/lapnet_scaling/timings_lapnet.txt",
    "/storage/scherbelam20/runs/lapnet_scaling/timings_psiformer.txt",

]
models = ["SWANN", "lapnet", "psiformer", "ferminet"]
n_el_for_breakdown = 190
n_grid_points_pp = 4

df = pd.concat([load_data(data_fname) for data_fname in data_fnames], ignore_index=True)
df["model"] = df["model"].fillna("SWANN")
df["model_idx"] = df.model.apply(models.index)
df["t_wf_lr"] = df.t_wf_lr.fillna(df.t_wf_full)
df["t_E_pot_estimate"] = df.t_wf_full + df.t_wf_lr * df.n_el * n_grid_points_pp
df.loc[df.model != "SWANN", "t_E_pot"] = df.t_E_pot_estimate[df.model != "SWANN"]

df.t_wf_full = df.t_wf_full * BATCH_SIZE_PLOT / df.batch_size
df.t_wf_lr = df.t_wf_lr * BATCH_SIZE_PLOT / df.batch_size
df.t_E_kin = df.t_E_kin * BATCH_SIZE_PLOT / df.batch_size
df.t_E_pot = df.t_E_pot * BATCH_SIZE_PLOT / df.batch_size
df["t_sampling"] = df.t_wf_full + df.n_el * df.t_wf_lr
df["t_total"] = df.t_sampling + df.t_E_kin + df.t_E_pot

df = df[((df.model == "SWANN") & df.use_ecp) | (df.model != "SWANN")]
df = df.sort_values(["model_idx", "system", "system_size"])


colors = bokeh.palettes.Category10[10]
p_tot, p_psi, p_Ekin, p_Epot = None, None, None, None

# Scaling plots t(n_el)
for model, color in zip(models, colors):
    df_model = df[df.model == model]
    if len(df_model) == 0:
        continue
    p_tot = plot_and_fit(p_tot, df_model.n_el, df_model.t_total, "Total", color=color, label=model)

    if model == "SWANN":
        p_psi = plot_and_fit(p_psi, df_model.n_el, df_model.t_wf_full, "Wavefunction update", color=color, dash="4 2", label=model + " (full)")
        p_psi = plot_and_fit(p_psi, df_model.n_el, df_model.t_wf_lr, "Wavefunction update", color=color, label=model + " (low-rank)")
    else:
        p_psi = plot_and_fit(p_psi, df_model.n_el, df_model.t_wf_full, "Wavefunction update", color=color, label=model)
    p_Ekin = plot_and_fit(p_Ekin, df_model.n_el, df_model.t_E_kin, "Kinetic energy", color=color, label=model)
    # p_Epot = plot_and_fit(p_Epot, df.n_el, df.t_E_pot, "Potential energy", color=colors[0], label=model)
plots = [p_tot, p_psi, p_Ekin]
for p in plots:
    p.xaxis.axis_label = "Number of electrons"
    p.xaxis.ticker = np.round(np.geomspace(16, 362, 10)).astype(int)
    p.yaxis.axis_label = "Time (s)"
    p.legend.click_policy = "hide"
    p.output_backend = "svg"
    p.add_layout(bokeh.models.Span(location=n_el_for_breakdown, dimension="height", line_color="gray", line_alpha=0.6, line_width=1))

# Breakdown of total step time
timings = ["t_sampling", "t_E_kin", "t_E_pot"]
# df_large = df[df.n_el == n_el_for_breakdown][["model", "model_idx"] + timings]
df_large = df[np.abs(df.n_el - n_el_for_breakdown) <= 6]
# df_large = pd.wide_to_long(df_large, "t", sep="_", i=["model_idx", "model"], j="timing",  suffix=r".*").reset_index()

#%%
# source = {k: df_large[k].values for k in timings}
# source["model"] = [str(x) for x in df_large.model.values]

df_large["speedup"] = (df_large.t_total / df_large.t_total.min()).apply(lambda x: f"{x:.1f}x")
ds = bokeh.models.ColumnDataSource(df_large)
p_ratio = bpl.figure(width=500, height=400, x_range=models, y_range=(0, df_large.t_total.max()+5), title=f"Total time for {n_el_for_breakdown} electrons", tooltips=[("model", "@model"), ("Potential energy", "@t_E_pot"), ("Kinetic energy", "@t_E_kin"), ("Sampling", "@t_sampling"), ])
p_ratio.vbar_stack(timings, x="model", source=ds, color=bokeh.palettes.HighContrast3, legend_label=timings, width=0.9)
labels = bokeh.models.LabelSet(x="model", y="t_total", text="speedup", x_offset=0, y_offset=5, source=ds, text_align="center")
p_ratio.add_layout(labels)
p_ratio.legend.location = "top_left"

# labels = LabelSet(x='x', y='y', text='y', level='glyph',
#                   text_align='center', y_offset=5, source=source)
p_ratio.legend[0].items.reverse()


grid = bokeh.layouts.gridplot([[p_psi, p_Ekin], [p_tot, p_ratio]])
bpl.output_notebook()
bpl.show(grid)

# with open("scaling.html", "w") as f:
#     f.write(bokeh.embed.file_html(grid, bokeh.resources.CDN, "scaling"))

# bokeh.io.export_svg(grid, filename="scaling.svg")
# bokeh.io.export_png(grid, filename="scaling.png", width=2000)

#%%
df_swann = df[df.model == "SWANN"]

p0 = plot_and_fit(None, df_swann["n_el"], df_swann["t_E_kin"], "E_kin (n_el)", "blue", label="SWANN")
p0.yaxis.axis_label = "Time (s)"
p0.xaxis.axis_label = "Number of electrons"

p1 = plot_and_fit(None, df_swann["pp_static/n_triplets"], df_swann["t_E_kin"], "E_kin (triplets)", "blue", label="SWANN", n_fit_min=8000)
p1.yaxis.axis_label = "Time (s)"
p1.xaxis.axis_label = "Number of triplets"

p2 = plot_and_fit(None, df_swann["n_el"], df_swann["pp_static/n_triplets"], "triplets (n_el)", "blue", label="SWANN", n_fit_min=100)
p2.yaxis.axis_label = "Number of triplets"
p2.xaxis.axis_label = "Number of electrons"

grid = bokeh.layouts.row([p0, p1, p2])
bpl.show(grid)




