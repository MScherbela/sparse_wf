# %%
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use(["science", "grid"])

orca_s2_values = {
    "cumulene_C12H4_90deg_triplet": 3.835049,
    "cumulene_C16H4_90deg_triplet": 4.838533,
    "cumulene_C20H4_90deg_triplet": 5.853606,
    "cumulene_C24H4_90deg_triplet": 6.870610,
    "cumulene_C2H4_90deg_triplet": 2.010821,
    "cumulene_C30H4_90deg_triplet": 8.397287,
    "cumulene_C36H4_90deg_triplet": 9.924735,
    "cumulene_C4H4_90deg_triplet": 2.223554,
    "cumulene_C6H4_90deg_triplet": 2.515511,
    "cumulene_C8H4_90deg_triplet": 2.901217,
}

n_values = [2, 4, 6, 8, 12, 16, 20, 24, 36]

df_pyscf = pd.read_csv("cumulene_spin_data.csv")
df_pyscf["method"] = "pyscf"
df_orca = pd.read_csv("orca_energies.csv")
df_orca = df_orca[
    (df_orca["method"] == "UHF")
    & (df_orca["basis_set"] == "cc-pVDZ")
    & (~df_orca["comment"].str.contains("quintuplet"))
]
df_orca.rename(columns={"comment": "geom", "E_final": "E"}, inplace=True)
df_orca["s2"] = df_orca["geom"].apply(lambda x: orca_s2_values.get(x, 0))
df_orca["method"] = "orca"
df = pd.concat([df_pyscf[["method", "geom", "E", "s2"]], df_orca[["method", "geom", "E", "s2"]]], ignore_index=True)

df["n"] = df["geom"].str.extract(r"C(\d+)H").astype(int)
df["angle"] = df["geom"].str.extract(r"_(\d+)deg").astype(int)
df = df[df.n.isin(n_values)]

pivot = df.pivot(index="n", columns=["angle", "method"], values="E")
deltaE = (pivot[90] - pivot[0]) * 1000
pivot_s2 = df.pivot(index=["n", "angle"], columns=["method"], values="s2").reset_index()


fig, axes = plt.subplots(1, 2, figsize=(8, 4))
# axes[0].plot(pivot_orca.index, pivot_orca["deltaE"], marker="o", label="ORCA")
axes[0].plot(deltaE.index, deltaE.pyscf, marker="s", label="PySCF")
axes[0].plot(deltaE.index, deltaE.orca, marker="o", label="ORCA")
axes[0].set_xlabel("Cumulene length")
axes[0].set_ylabel("triplet - singlet / mHa")
axes[0].legend()

for angle, ls in zip([0, 90], ["-", "--"]):
    pivot_angle = pivot_s2[pivot_s2["angle"] == angle]
    axes[1].plot(pivot_angle["n"], pivot_angle["pyscf"], marker="s", label=f"PySCF {angle}deg", ls=ls)
    axes[1].plot(pivot_angle["n"], pivot_angle["orca"], marker="o", label=f"ORCA {angle}deg", ls=ls)
axes[1].set_xlabel("Cumulene length")
axes[1].set_ylabel("S2")
axes[1].legend()


# axes[1].plot(deltaE.orca, deltaE.pyscf, marker="o")
# axes[1].plot(pivot_orca["deltaE"], pivot_pyscf["deltaE"], marker="o")
# axes[1].set_xlabel("ORCA energy difference (mHa)")
# axes[1].set_ylabel("PySCF energy difference (mHa)")
# plt.tight_layout()
