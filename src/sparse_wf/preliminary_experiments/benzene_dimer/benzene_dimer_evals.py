# %%
import wandb
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt

reload_data = False
if reload_data:
    api = wandb.Api()
    all_runs = api.runs("tum_daml_nicholas/benzene")
    runs = [r for r in all_runs if re.match("HLR.*eval\d*k", r.name)]

    data = []
    for r in runs:
        print(r.name)
        dist = float(re.search(r"(\d+\.\d+)A", r.name).group(1))
        cutoff = r.name.split("_")[1]
        if r.name.startswith("HLRTransfer"):
            cutoff = "Transfer" + cutoff
        metadata = dict(
            dist=dist,
            cutoff=cutoff,
            run_name=r.name,
        )

        hist = pd.DataFrame([h for h in r.scan_history(["eval/step", "eval/E"], page_size=10_000)])
        hist = hist[hist["eval/step"] > 200]
        if len(hist) < 1000:
            print(f"Not enough data for {r.name}")
        energies = hist["eval/E"].values
        data.append(metadata | dict(E=np.mean(energies), E_stderr=np.std(energies) / np.sqrt(len(energies))))

    df_all = pd.DataFrame(data)
    df_all.to_csv("benzene_eval.csv", index=False)


df_all = pd.read_csv("benzene_eval.csv")
df_swann = df_all.pivot_table(index="cutoff", columns=["dist"], values=["E", "E_stderr"])
df_swann["delta"] = (df_swann["E"][4.95] - df_swann["E"][10.0]) * 1000
df_swann["err"] = np.sqrt(df_swann["E_stderr"][4.95] ** 2 + df_swann["E_stderr"][10.0] ** 2) * 1000
df_swann = df_swann.reset_index()[["cutoff", "delta", "err"]]
df_swann["cutoff"] = df_swann["cutoff"].str.replace("Transfer", "")
df_swann.columns = ["cutoff", "delta", "err"]

fig, ax = plt.subplots(1, 1, figsize=(8, 6))

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
for _, r in df_swann.iterrows():
    cutoff, delta, error = r["cutoff"], r["delta"], r["err"]
    rel_energies[f"SWANN {cutoff}"] = delta
    rel_stderrors[f"SWANN {cutoff}"] = error


import matplotlib.pyplot as plt

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

