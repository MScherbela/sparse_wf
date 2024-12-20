#%%
import numpy as np
from sparse_wf.scf import run_cas, run_hf, get_most_important_determinants
import pyscf.gto
import sparse_wf.system
import matplotlib.pyplot as plt
import itertools
import pandas as pd

BOHR_PER_ANGSTROM = 1.8897259886

def get_mol(distance_in_angstrom, basis):
    mol_template = sparse_wf.system.database(comment="benzene_dimer_T_4.95A")
    Z = mol_template.atom_charges()
    R = mol_template.atom_coords()
    R[12:, 2] += (distance_in_angstrom - 4.95) * BOHR_PER_ANGSTROM
    mol = pyscf.gto.M(atom=list(zip(Z, R)), unit="bohr", basis=basis)
    return mol



# basis_set = "sto-6g"
basis_sets = ["sto-6g", "cc-pvdz", "cc-pvtz"]
# distances = [3, 4, 4.95, 6, 8, 10]
distances = [4.5, 4.75]
all_data = []
for (dist, basis) in itertools.product(distances, basis_sets):
    mol = get_mol(dist, basis)

    hf = run_hf(mol)
    cas = run_cas(hf, n_orbitals=12, n_electrons=12, s2=0)

    idx_orb, ci_coeffs = get_most_important_determinants(cas, n_dets=16, threshold=0.001)
    all_identical = np.all(idx_orb == idx_orb[0], axis=0)
    idx_orb = idx_orb[:, ~all_identical]

    print(f"dist={dist}, basis={basis}, E_HF = {hf.e_tot:.4f}, E_CAS = {cas.energy:.4f}, s2 = {cas.s2}")
    for ci_coeff, idx in zip(ci_coeffs, idx_orb):
        idx_up, idx_dn = np.split(idx, 2)
        idx_up_string = ",".join([str(x) for x in idx_up])
        idx_dn_string = ",".join([str(x) for x in idx_dn])
        print(f"{ci_coeff:+3f}: {idx_up_string} | {idx_dn_string}")
    all_data.append(dict(E_HF = hf.e_tot, E_CAS=cas.energy, s2=cas.s2, idx_orb=idx_orb, ci_coeffs=ci_coeffs, basis=basis, dist=dist))
df = pd.DataFrame(all_data)
df.to_csv("benzene_dimer_pyscf_more.csv", index=False)


#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("benzene_dimer_pyscf.csv")
df = df[["basis", "dist", "E_CAS", "E_HF"]]
distances = np.sort(np.unique(df["dist"]))
df = pd.wide_to_long(df, stubnames=["E"], i=["basis", "dist"], j="method", sep="_", suffix=".*").reset_index()
df_10 = df[df.dist==10].rename(columns={"E": "E_10"}).drop(columns="dist")
df = df.merge(df_10, on=["basis", "method"], how="left")
df = df.sort_values(["basis", "dist", "method"])
df["delta_E"] = (df["E"] - df["E_10"]) * 1000
df = df[df.dist > 4]
print(df)

fig, ax = plt.subplots(1,1, figsize=(10, 8))
basis_sets = ["sto-6g", "cc-pvdz", "cc-pvtz"]
# colors = [f"C{i}" for i in range(len(basis_sets))]
colors = [("darkblue", "lightblue"), ("darkgreen", "lightgreen"), ("darkred", "salmon")]
for color, basis in zip(colors, basis_sets):
    df_hf = df[(df.basis == basis) & (df.method == "HF")]
    df_cas = df[(df.basis == basis) & (df.method == "CAS")]
    ax.plot(df_cas["dist"], df_cas["delta_E"], label=f"{basis} CAS", color=color[0], ls="-", marker="s")
    ax.plot(df_hf["dist"], df_hf["delta_E"], label=f"{basis} HF", color=color[1], ls="--", marker="o")
ax.set_xlabel("Distance (Angstrom)")
ax.set_ylabel("Delta E / mHa")
ax.grid(alpha=0.5)
ax.axvline(4.95, color="black", ls="--", label="4.95 Angstrom", zorder=-1)


ref_values = {
    "Experiment": -3.8,
    "SWANN (cutoff=3.0, 30k steps)": -9.2,
    "SWANN (cutoff=5.0, 180k steps)": 1.0,
    "PsiFormer": 5.0,
    "FermiNet VMC (Glehn et al)": -4.6,
    "FermiNet DMC (Ren et al)": -9.2,
}

ref_colors = ["k"] + [f"C{i}" for i in range(5)]
for (ref, E_ref), color in zip(ref_values.items(), ref_colors):
    ax.plot([4.95], [E_ref], color=color, label=ref, marker='X', ms=10)
ax.legend()
fig.tight_layout()
fig.savefig("benzene_dimer_pyscf.png")




# E_10 =


# plt.close("all")
# for basis in b




