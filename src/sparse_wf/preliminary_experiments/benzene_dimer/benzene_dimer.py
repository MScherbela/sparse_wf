# %%
import wandb
import pandas as pd
import re

api = wandb.Api()
all_runs = api.runs("tum_daml_nicholas/benzene")
runs = [r for r in all_runs if re.match("HLR_.*", r.name)]

data = []
for r in runs:
    print(r.name)
    dist = float(re.search(r"(\d+\.\d+)A", r.name).group(1))
    metadata = dict(
        dist=dist,
        cutoff=5.0 if "5.0" in r.name else 3.0,
        run_name=r.name,
    )

    for h in r.scan_history(["opt/step", "opt/E"], page_size=10_000):
        data.append(h | metadata)

df_all = pd.DataFrame(data)
df_all.to_csv("benzene_energies.csv", index=False)

# %%
import matplotlib.pyplot as plt
import numpy as np

window_length = 1_000
# cutoffs = [3.0, 5.0]
cutoffs = [3.0]
dists = [4.95, 10.0]

df_all = pd.read_csv("benzene_energies.csv")
# molecules = sorted(df_all["molecule"].unique())
pivot = df_all.pivot_table(index="opt/step", columns=["cutoff", "dist"], values="opt/E", aggfunc="mean")
for cutoff in cutoffs:
    for dist in dists:
        smoothed = pivot[(cutoff, dist)].fillna(method="ffill", limit=10).rolling(window=window_length).mean()
        pivot.loc[:, (cutoff, f"E{dist}_smooth")] = smoothed
    # pivot.loc[:, (cutoff, "delta_smooth")] = (pivot[cutoff][f"E{dists[0]}_smooth"] - pivot[cutoff][f"E{dists[1]}_smooth"]) * 1000
    pivot.loc[:, (cutoff, "delta")] = (pivot[(cutoff, dists[0])] - pivot[(cutoff, dists[1])]) * 1000
    pivot.loc[:, (cutoff, "delta_smooth")] = pivot.loc[:, (cutoff, "delta")].rolling(window=window_length).mean()
    pivot.loc[:, (cutoff, "delta_stderr")] = pivot.loc[:, (cutoff, "delta")].rolling(window=window_length).std() / np.sqrt(window_length)


refs = {
    "Experiment": (-3.8, "k"),
    "PsiFormer": (5.0, "forestgreen"),
    "FermiNet VMC (Glehn et al)": (-4.6, "C0"),
    "FermiNet DMC (Ren et al)": (-9.2, "navy"),
}

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
for ref, (E_ref, color) in refs.items():
    ax.axhline(E_ref, color=color, linestyle="dashed")
    if ref == "Experiment":
        ax.axhspan(E_ref - 0.6, E_ref + 0.6, color=color, alpha=0.2, zorder=-1)
    ax.text(0.1, E_ref, ref, color=color, va="bottom", ha="left")

for cutoff in cutoffs:
    ax.plot(pivot.index / 1000, pivot[(cutoff, "delta_smooth")], label=f"SWANN cutoff={cutoff:.1f}", color="red")
    ax.fill_between(
        pivot.index / 1000,
        pivot[(cutoff, "delta_smooth")] - 2 * pivot[(cutoff, "delta_stderr")],
        pivot[(cutoff, "delta_smooth")] + 2 * pivot[(cutoff, "delta_stderr")],
        color="red",
        alpha=0.2,
    )
ax.legend(loc="upper right")
ax.set_ylim([-10, 6])
ax.set_xlim([0, None])
ax.set_xlabel("Opt Step / k")
ax.set_ylabel("E_4.95 - E_10.0 / mHa")
ax.set_title("Benzene dimer binding energy")
fig.savefig("benzene_dimer.png")

# # df_ref = pd.read_csv("../../../../data/energies.csv")
# # df_ref = df_ref[(df_ref.model == "CCSD(T)") & (df_ref.model_comment == "CBS") & (df_ref.geom_comment.isin(molecules))]
# # df_ref = df_ref.groupby("geom_comment")[["E"]].mean()
# # df_ref["delta"] = (df_ref - df_ref.mean(axis=0)) * 1000

# p = bpl.figure(width=900, title="Benzene dimer binding energy", x_axis_label="Opt Step / k", y_axis_label="E_4.95 - E_10.0 / mHa", tools="box_zoom,hover,save", tooltips=[("step", "$x"), ("delta", "$y")])
# for cutoff, color in zip(cutoffs, bokeh.palettes.HighContrast3):
#     p.line(pivot.index / 1000, pivot[(cutoff, "delta_smooth")], legend_label=f"SWANN cutoff={cutoff:.1f}", color=color, line_width=2)
#     # p.add_layout(bokeh.models.Span(location=df_ref.loc[m, "delta"], dimension="width", line_color=color, line_dash="dashed", line_width=2))

# for (ref, E_ref), color in zip(refs.items(), ("black",) + bokeh.palettes.Category10[10]):
#     line_dash = "solid" if ref == "Experiment" else "dashed"
#     p.line([0, max(pivot.index / 1000)], [E_ref, E_ref], legend_label=ref, color=color, line_dash=line_dash, line_width=2)


# p.y_range.start = -10
# p.y_range.end = 10
# bpl.output_notebook()
# bpl.show(p)

# bokeh.io.export_png(p, filename="benzene_dimer.png")
