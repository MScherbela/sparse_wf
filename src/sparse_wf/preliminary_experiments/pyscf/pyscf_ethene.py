#%%
from sparse_wf.geometry import load_geometries
import pyscf.scf
import matplotlib.pyplot as plt
from pyscf import gto, mp, mcscf
import numpy as np
from pyscf.fci.cistring import _gen_occslst

all_geoms = load_geometries()
geom = [g for g in all_geoms.values() if g.comment == 'cumulene_C2H4_90deg']
assert len(geom) == 1
geom = geom[0]

print("Running Hartree-Fock...")
mol = geom.as_pyscf_molecule('cc-pvdz')
hf = pyscf.scf.RHF(mol)
hf.kernel()
print("Finished running Hartree-Fock.")

print("Running CAS...")
cas = hf.CASSCF(2, 2).run()
print("Finished running CAS.")

#%%
ci_coeffs = cas.ci.flatten()
idx_sorted = np.argsort(ci_coeffs**2)[::-1]
n_coeffs_max = 5

assert cas.ci.shape[0] == cas.ci.shape[1]
n_dets_per_spin = cas.ci.shape[0]

weights_cum = 0
print("i, idx_up, idx_dn, weight, weights_cum")
print("======================================")
for i in range(n_coeffs_max):
    weight = ci_coeffs[idx_sorted[i]]**2
    weights_cum += weight
    idx_det_up, idx_det_dn = np.divmod(idx_sorted[i], n_dets_per_spin)
    print(f"{i:2d} | {idx_det_up:3d},{idx_det_dn:3d} |     {weight:.4f}, {weights_cum:.4f}")






