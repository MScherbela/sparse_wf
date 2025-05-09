# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sparse_wf.plot_utils import (
    MILLIHARTREE,
    COLOR_PALETTE,
    savefig,
    format_value_with_error,
    COLOR_FIRE,
)
import scienceplots
from PIL import Image

plt.style.use(["science", "grid"])

BAR_WIDTH = 0.8


KJ_PER_MOL_IN_mHA = 0.3808795808835581

df_ref = pd.read_csv("ref_energies.csv", index_col=0) * KJ_PER_MOL_IN_mHA
df_fire = pd.read_csv("fe2s2_energies_aggregated.csv", index_col=0) * 1000
uncertainty_fire = df_fire["E_err"]
df_fire = df_fire.drop(columns="E_err").T
df_fire.index = df_fire.index.map({"E": "FiRE", "E_ext": "FiRE_ext"})
df = pd.concat([df_ref, df_fire])
# df = df.sub(df["HC"], axis=0).drop(columns="HC")
df = df.sub(df.mean(axis=1), axis=0)
df_uncorrected = pd.DataFrame(
    {
        "no_relativistic": df.loc["composite/nonrel"],
        "no_triplets": df.loc["UHF/CCSD/TZ"] - df.loc["UHF/CCSD(T)/TZ"] + df.loc["composite"],
        "no_CBS": df.loc["UHF/CCSD(T)/TZ"] - df.loc["UHF/CCSD(T)/CBS_23"] + df.loc["composite"],
        "no_DMRG": df.loc["UHF/CCSD(T)/CBS_23"],
    }
)
df = pd.concat([df, df_uncorrected.T], axis=0)

mae_vs_fire = np.abs(df - df.loc["FiRE"]).mean(axis=1)
mae_vs_comp = np.abs(df - df.loc["composite"]).mean(axis=1)

methods = [
    # ("no_relativistic", "no relativistic\ncorrection", COLOR_PALETTE[0]),
    ("no_triplets", "No (T):\nCCSD/CBS\n+DMRG", COLOR_PALETTE[0]),
    ("no_CBS", "No CBS:\nCCSD(T)/TZ\n+DMRG", COLOR_PALETTE[1]),
    ("no_DMRG", "No DMRG:\nCCSD(T)/CBS", COLOR_PALETTE[2]),
    # ("composite/nonrel", "No relativistic", COLOR_PALETTE[3]),
    ("FiRE_ext", "FiRE, $c=5a_0$", COLOR_FIRE),
]

fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))
for idx_method, (method, label, color) in enumerate(methods):
    E = mae_vs_comp[method]
    ax.barh([idx_method], [E], color=color)
    if E > 1:
        ax.text(0.2, idx_method, f"{E:.1f}", va="center", ha="left", color="white")
    else:
        ax.text(E + 0.2, idx_method, f"{E:.1f}", va="center", ha="left", color="k")

ax.set_ylim([-BAR_WIDTH / 2 -0.1, len(methods) - 1 + BAR_WIDTH / 2 + 0.1])
ax.invert_yaxis()
ax.set_yticks(np.arange(len(methods)))
ax.set_yticklabels([m[1] for m in methods])
ax.set_xlabel("MAE vs. conventional best estimate " + MILLIHARTREE)
ax.grid(False, axis="y")

x_inset, y_inset, h_inset = 0.41, 0.07, 0.5
ax_inset = ax.inset_axes((x_inset, y_inset, 1 - x_inset, h_inset))
image_ax = ax.inset_axes((x_inset, h_inset + y_inset + 0.01, 1 - x_inset, 0.2))

methods_inset = methods + [("composite", "Conventional\nbest est.", "k")]
width_per_bar = BAR_WIDTH / len(methods_inset)
for idx_m, (m, label, color) in enumerate(methods_inset):
    # label=label.split(":\n")[0].replace("\n", " ")
    label = "conventional\nbest estimate" if m == "composite" else None
    ax_inset.bar(
        np.arange(4) - 0.5 * BAR_WIDTH + idx_m * width_per_bar,
        df.loc[m],
        label=label,
        color=color,
        width=width_per_bar,
        align="edge",
    )
ax_inset.set_xticks(np.arange(4))
ax_inset.set_xticklabels(df.columns)
# ax_inset.set_ylabel("$E - E_\\text{HC}$ " + MILLIHARTREE, labelpad=2)
ax_inset.set_ylabel("$E - E_\\text{mean}$ " + MILLIHARTREE, labelpad=2)
ax_inset.grid(False, axis="x")
ax_inset.legend(
    loc="upper left", handlelength=0.4, handleheight=2.3, handletextpad=0.3, frameon=False, bbox_to_anchor=(-0.02, 0.99)
)

handles, labels = ax_inset.get_legend_handles_labels()
# handles = handles[::3] + handles[1::3] + handles[2::3]
# labels = labels[::3] + labels[1::3] + labels[2::3]
# Create a custom handler map to align items properly

# fig.legend(handles, labels, loc="lower center", ncol=3, handlelength=1.2, handletextpad=0.5, columnspacing=1.5)
fig.tight_layout()
# fig.subplots_adjust(bottom=0.25)

img = Image.open("fe2s2_render.png")
img = img.crop(img.getbbox())
image_ax.imshow(img)
image_ax.axis("off")

fig.text(0.04, 0.94, "\\textbf{b)}", fontsize=12)

savefig(fig, "Fe2S2_barchart")

methods_tex = (
    [methods_inset[-1]]
    + [("FiRE", "FiRE, $c=5a_0$, raw", None), ("FiRE_ext", "FiRE, $c=5a_0$, extrapolated", None)]
    + methods_inset[:-2]
)
df_tex = df.copy()
# df_tex.insert(0, "HC", 0)

uncertainty_mean = np.sqrt(np.var(uncertainty_fire) / len(uncertainty_fire))
uncertainty_fire_rel = np.sqrt(uncertainty_fire**2 + uncertainty_mean ** 2)
uncertainty_MAE = np.sum(uncertainty_fire_rel**2) / np.sqrt(len(uncertainty_fire_rel))
with open("Fe2S2_energies.tex", "w") as f:
    geom_header = "{method} & " + " & ".join([f"{{{g}}}" for g in list(df_tex)]) + " & {MAE}\\\\\n"
    f.write(geom_header)
    f.write("\midrule\n")
    for method, name, _ in methods_tex:
        # name = name.split(":")[0]
        name = name.replace("\n", " ").replace(" +", "+")
        f.write(f"{name} & ")
        for g, E in df_tex.loc[method].items():
            if "FiRE" in method:
                s = format_value_with_error(E, uncertainty_fire_rel[g])
            else:
                s = f"{E:.1f}"
            f.write(s + " &")
        if "FiRE" in method:
            s_mae = format_value_with_error(mae_vs_comp[method], uncertainty_MAE)
        else:
            s_mae = f"{mae_vs_comp[method]:.1f}"
        f.write(s_mae + "\\\\\n")
