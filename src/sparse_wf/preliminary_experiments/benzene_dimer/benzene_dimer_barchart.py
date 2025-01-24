# %%
import numpy as np
import pandas as pd
import wandb
import re


def get_without_outliers(x, outlier_threshold=10):
    med = np.median(x)
    spread = np.quantile(x, 0.75) - np.quantile(x, 0.25)
    return x[np.abs(x - med) < outlier_threshold * spread]


def robustmean(x):
    x = get_without_outliers(x)
    return np.mean(x)

def robuststderr(x):
    x = get_without_outliers(x)
    return np.std(x) / np.sqrt(len(x))


def get_data(run, **metadata):
    data = []
    for h in run.scan_history(["opt/step", "opt/E"], page_size=10_000):
        data.append(h)
    df = pd.DataFrame(data)
    for k, v in metadata.items():
        df[k] = v
    return df


geom_names = ["benzene_dimer_T_4.95A", "benzene_dimer_T_10.00A"]

# Experiment
rel_energies = {"Experiment": -3.8}
rel_stderrors = dict()

# Conventional results
df_orca = pd.read_csv("orca.csv", sep=";")
orca_methods = [
    ("UHF", "UHF", "cc-pVQZ"),
    ("CCSD(T)", "CCSD(T)", "cc-pVDZ"),
]
for name, method, basis_set in orca_methods:
    df = df_orca[(df_orca.method == method) & (df_orca.basis_set == basis_set) & (df_orca.comment.isin(geom_names))]
    assert len(df) == 2, f"Expected 2 entries for {name}, got {len(df)}"
    df = df.groupby("comment").E_final.mean()
    rel_energies[name] = (df[geom_names[0]] - df[geom_names[1]]) * 1000

# Other DL-VMC methods
rel_energies |= {
    "FermiNet VMC\n(Ren et al)": -18.2,
    "FermiNet DMC\n(Ren et al)": -9.3,
    "FermiNet VMC\n(Glehn et al)": -4.6,
    "PsiFormer\n(Glehn et al)": 5.0,
}

## SWANN
reload_data = False
if reload_data:
    name_template_cutoff3 = f"3.0_({'|'.join(geom_names)})_vsc"
    name_template_cutoff5 = f"5.0_({'|'.join(geom_names)})_singleEl_jumps_leonardo"
    name_template_cutoff79 = f"(7|9).0_({'|'.join(geom_names)})_singleEl_jumps"
    all_runs = wandb.Api().runs("tum_daml_nicholas/benzene")
    runs = [
        r
        for r in all_runs
        if re.match(f"({name_template_cutoff3}|{name_template_cutoff5}|{name_template_cutoff79})", r.name)
    ]

    swann_data = []
    for r in runs:
        print(r.name)
        df = get_data(
            r, cutoff=float(r.name.split("_")[0]), geom=geom_names[0] if geom_names[0] in r.name else geom_names[1]
        )
        swann_data.append(df)
    df_swann = pd.concat(swann_data)
    df_swann.to_csv("swann_benzene_dimer.csv", index=False)
else:
    df_swann = pd.read_csv("swann_benzene_dimer.csv")

n_eval_steps = 10_000
for cutoff in [3.0, 5.0, 7.0, 9.0]:
    energies = []
    stderrors = []
    n_opt_steps = df_swann[df_swann.cutoff == cutoff].groupby("geom")["opt/step"].max().min()
    print(f"SWANN opt steps, cutoff {cutoff}: {n_opt_steps}")
    for geom in geom_names:
        df = df_swann[(df_swann.cutoff == cutoff) & (df_swann.geom == geom)]
        df = df[(df["opt/step"] >= n_opt_steps - n_eval_steps) & (df["opt/step"] <= n_opt_steps)]
        if len(df) < (n_eval_steps * 0.8):
            print("Not enough data for cutoff", cutoff, "and geom", geom)
            continue
        energies.append(robustmean(df["opt/E"]))
        stderrors.append(robuststderr(df["opt/E"]))
    if len(energies) == 2:
        name = f"SWANN\n(cutoff {cutoff})"
        rel_energies[name] = (energies[0] - energies[1]) * 1000
        rel_stderrors[name] = np.sqrt(stderrors[0] ** 2 + stderrors[1] ** 2) * 1000

# Plot
import matplotlib.pyplot as plt

# colors = ["black", "dimgray", "lightgray",  "navy", "darkslateblue",  "royalblue", "forestgreen",  "firebrick", "red"]
colors = ["black"] + ["dimgray"] * 2 + ["navy"] * 3 + ["forestgreen"] + ["red"] * 4

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
for idx, color, (name, E) in zip(range(len(colors)), colors, rel_energies.items()):
    ax.barh(idx, E, color=color, label=name, zorder=3)
    if name in rel_stderrors:
        ax.errorbar(E, idx, xerr=rel_stderrors[name], fmt="none", ecolor="black", zorder=3, capsize=5)
    text_pos = -2 if E < 0 else 0.5
    ax.text(text_pos, idx, f"{E:.1f}", va="center", ha="left", color="white", zorder=4)
ax.set_xlabel("Binding energy $E_{4.95A} - E_{10.0A}$ / mHa")
ax.invert_yaxis()
ax.set_yticks(range(len(rel_energies)))
ax.set_yticklabels(rel_energies.keys())
ax.axvline(0, color="black", lw=1.5, zorder=4)
ax.axvline(rel_energies["Experiment"], color="black", lw=1, ls="--", zorder=-1)
ax.grid(axis="x", linestyle="--", zorder=-1)
ax.set_title("Benzene dimer binding energy")
fig.tight_layout()
fig_fname = "benzene_dimer_barchart.png"
fig.savefig(fig_fname, dpi=300, bbox_inches="tight")
fig.savefig(fig_fname.replace(".png", ".pdf"), bbox_inches="tight")
