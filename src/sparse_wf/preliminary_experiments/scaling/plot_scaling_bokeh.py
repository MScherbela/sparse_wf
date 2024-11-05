#%%
import bokeh.io
import bokeh.layouts
import bokeh.resources
import numpy as np
import pandas as pd
import ast
import bokeh.plotting as bpl
import bokeh

def plot_and_fit(p, x, y, title, color, n_fit_min=None, annotation_below=False, label=""):
    include_in_fit = x >= n_fit_min
    x_fit, y_fit = x[include_in_fit], y[include_in_fit]
    fit_coeffs = np.polyfit(np.log(x_fit), np.log(y_fit), 1)
    exponent = fit_coeffs[0]
    y_fitted = np.exp(np.polyval(fit_coeffs, np.log(x_fit)))

    if p is None:
        p = bpl.figure(width=500, height=400, x_axis_type="log", y_axis_type="log", tools="box_zoom,hover,save", tooltips=[("n_el", "$snap_x"), ("t", "$snap_y")])
    p.line(x_fit, y_fitted, color=color)
    scaling_text = f"$$O(n^{{{exponent:.2f}}})$$"

    label_kwargs = dict(legend_label=label) if label else {}
    p.scatter(x, y, color=color, marker="circle", **label_kwargs)

    # Add annotation
    if annotation_below:
        label_kwargs = dict(x_offset=10, y_offset=-10, text_align="left")
    else:
        label_kwargs = dict(x_offset=-10, y_offset=0, text_align="right")

    x_annotation = np.sqrt(np.min(x_fit) * np.max(x_fit))
    y_annotation = np.exp(np.polyval(fit_coeffs, np.log(x_annotation)))
    p.add_layout(bokeh.models.Label(x=x_annotation, y=y_annotation, text=scaling_text, text_font_size="12pt", text_color=color, **label_kwargs))
    p.legend.location = "top_left"
    p.hover.renderers = p.renderers[1::2] # only show hover for data points, not lines
    p.title.text = title
    return p


BATCH_SIZE_PLOT = 32
data_fname = "timings_with_ecp.txt"
# data_fname = "timings_no_ecp.txt"
data = []
with open(data_fname, "r") as f:
    for line in f:
        data.append(ast.literal_eval(line))
df = pd.DataFrame(data)
df.t_wf_full = df.t_wf_full * BATCH_SIZE_PLOT / df.batch_size
df.t_wf_lr = df.t_wf_lr * BATCH_SIZE_PLOT / df.batch_size
df.t_E_kin = df.t_E_kin * BATCH_SIZE_PLOT / df.batch_size
df.t_E_pot = df.t_E_pot * BATCH_SIZE_PLOT / df.batch_size
df["t_sampling"] = df.t_wf_full + df.n_el * df.t_wf_lr
df["t_total"] = df.t_sampling + df.t_E_kin + df.t_E_pot

colors = bokeh.palettes.Category10[10]


p_tot = plot_and_fit(None, df.n_el, df.t_total, "Total", color=colors[0], n_fit_min=50)
p_psi = plot_and_fit(None, df.n_el, df.t_wf_full, "Wavefunction", color=colors[0], n_fit_min=50, label="Full wavefunction")
p_psi = plot_and_fit(p_psi, df.n_el, df.t_wf_lr, "Wavefunction", color=colors[1], n_fit_min=50, annotation_below=True, label="Low-rank update")
p_Ekin = plot_and_fit(None, df.n_el, df.t_E_kin, "Kinetic energy", color=colors[0], n_fit_min=50)
p_Epot = plot_and_fit(None, df.n_el, df.t_E_pot, "Potential energy", color=colors[0], n_fit_min=50)
plots = [p_tot, p_psi, p_Ekin, p_Epot]

for p in plots:
    p.xaxis.axis_label = "Number of electrons"
    p.xaxis.ticker = np.round(np.geomspace(16, 362, 10)).astype(int)
    p.yaxis.axis_label = "Time (s)"
    p.legend.click_policy = "hide"
    p.output_backend = "svg"

grid = bokeh.layouts.gridplot([plots[:2], plots[2:]])

# bpl.output_file(data_fname.replace(".txt", ".html"))
# bpl.output_file(None)
# bpl.save(grid)
bpl.output_notebook()
bpl.show(grid)

with open("scaling.html", "w") as f:
    f.write(bokeh.embed.file_html(grid, bokeh.resources.CDN, "scaling"))

bokeh.io.export_svg(grid, filename="scaling.svg")
bokeh.io.export_png(grid, filename="scaling.png", width=2000)
# bpl.save(grid)
# fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharey=True)

# ax_tot = axes[0, 0]
# ax_psi = axes[0, 1]
# ax_Ekin = axes[1, 0]
# ax_Epot = axes[1, 1]

# plot_and_fit(ax_tot, df.n_el, df.t_total, color=colors[0], n_fit_min=50)
# plot_and_fit(ax_psi, df.n_el, df.t_wf_full, label="Full", color=colors[0], n_fit_min=50)
# plot_and_fit(ax_psi, df.n_el, df.t_wf_lr, label="Low-rank", color="C1", n_fit_min=50)
# plot_and_fit(ax_Ekin, df.n_el, df.t_E_kin, color=colors[0], n_fit_min=50)
# plot_and_fit(ax_Epot, df.n_el, df.t_E_pot, color=colors[0], n_fit_min=50)

# ax_tot.set_title("Total time")
# ax_psi.set_title("Wavefunction evaluation")
# ax_Ekin.set_title("Kinetic energy")
# ax_Epot.set_title("Potential energy")

# for ax in axes.flatten():
#     ax.set_xlabel("Number of electrons")
#     ax.set_ylabel("Time (s)")
#     ax.set_xscale("log")
#     ax.set_yscale("log")
#     xticks = np.round(np.geomspace(16, 362, 10)).astype(int)
#     ax.set_xticks(xticks)
#     ax.set_xticks([], minor=True)
#     ax.set_xticklabels([str(x) for x in xticks])
#     ax.legend()

# fig.tight_layout()
# fig.savefig(data_fname.replace(".txt", ".png"))



