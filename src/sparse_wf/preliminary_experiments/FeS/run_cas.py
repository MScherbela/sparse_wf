#%%
import os
os.environ["OMP_NUM_THREADS"] = "20"
os.environ["MKL_NUM_THREADS"] = "20"
from sparse_wf.scf import run_hf, run_cas, get_most_important_determinants
import sparse_wf.system
import numpy as np

def get_excitations(idx_occ):
    idx_occ = set(idx_occ)
    idx_hf = set([i for i in range(len(idx_occ))])

    idx_excited = idx_occ - idx_hf
    idx_holes = idx_hf - idx_occ
    return (sorted(list(idx_holes)), sorted(list(idx_excited)))

def get_molecule(name, basis, ecp):
    mol = sparse_wf.system.database(name=name)
    mol.basis = basis
    if ecp:
        mol.ecp = {atom: "ccecp" for atom in ["C", "S", "Fe"]}
    mol.build()
    return mol




geoms = ["zhai_et_al_2023_HFe2_df2-svp", "zhai_et_al_2023_HS_df2-svp"]
basis = "ccecpccpvdz"
use_ecp = True

for geom in geoms:
    print(f"{geom=}, {basis=}, {use_ecp=}")
    mol = get_molecule(geom, basis, use_ecp)

    hf = run_hf(mol)
    print("Converged HF, running CAS...")
    cas = run_cas(hf, n_orbitals=12, n_electrons=12, s2=0)
    idx_orb, ci_coeffs = get_most_important_determinants(cas, n_dets=16, threshold=0.01)

    for ci_coeff, idx in zip(ci_coeffs, idx_orb):
        idx_up, idx_dn = np.split(idx, 2)
        holes_up, excited_up = get_excitations(idx_up)
        holes_dn, excited_dn = get_excitations(idx_dn)
        print(f"{ci_coeff:+3f}: {holes_up}->{excited_up} | {holes_dn}->{excited_dn}")
    print("="*40)
