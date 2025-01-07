#%%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

MILLIHARTREE_PER_KCAL_PER_MOL = 1.59362

n_steps_min = 42_000
n_steps_eval = 20_000

def stderror(x):
    return np.std(x) / np.sqrt(len(x))


dists = [4.95, 5.5, 6, 8, 10]
df_ref = pd.read_csv("pitonak_et_al.csv")
df_swann = pd.read_csv("benzene_energies.csv")
df_swann = df_swann[df_swann["cutoff"] == 5.0]
df_swann = df_swann[df_swann["dist"].isin(dists)]
for d in dists:
    assert df_swann.loc[df_swann.dist == d, "opt/step"].max() >= n_steps_min + n_steps_eval

df_swann = df_swann[(df_swann["opt/step"] >= n_steps_min) & (df_swann["opt/step"] <= n_steps_min + n_steps_eval)]

df_ref["E"] = df_ref["E_kcal_per_mol"] * MILLIHARTREE_PER_KCAL_PER_MOL
df_swann = df_swann.groupby("dist").agg({"opt/E": ("mean", stderror)}).reset_index()
df_swann["delta_E"] = (df_swann["opt/E"]["mean"] - df_swann["opt/E"]["mean"].iloc[0]) * 1000
df_swann["delta_E_std"] = np.sqrt(df_swann["opt/E"]["stderror"]**2 + df_swann["opt/E"]["stderror"]**2) * 1000



fig, ax = plt.subplots(1,1, figsize=(8, 6))
ax.errorbar(df_ref["dist_angstrom"], df_ref["E"] - df_ref["E"].min(), label="CCSD(T) Pitonak et al.", marker="o")
ax.errorbar(df_swann["dist"], df_swann["delta_E"], yerr=df_swann["delta_E_std"], label=f"SWANN (cutoff 5.0, {n_steps_min/1000:.0f}k steps)", marker="s", capsize=4)

refs = {
# "SWANN dist=100.0": 1.57,
"Experiment": -3.8,
"PsiFormer": 5.0,
"FermiNet VMC (Glehn et al)": -4.6,
"FermiNet DMC (Ren et al)": -9.2,
}
for i, (ref, E) in enumerate(refs.items()):
    ax.scatter([10.0], -E, label=ref, marker="X", s=100, color=f"C{i+2}")

ax.legend()
ax.grid(alpha=0.5)
ax.set_xlabel("Distance (Angstrom)")
ax.set_ylabel("(E - E_min) / mHa")
fig.suptitle(f"SWANN, cutoff 5.0, {n_steps_min/1000:.0f}k steps")
fig.savefig("swann_vs_pitonak.png", dpi=200)
