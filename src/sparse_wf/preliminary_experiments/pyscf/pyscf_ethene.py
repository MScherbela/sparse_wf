#%%
from sparse_wf.geometry import load_geometries
import pyscf.scf
import matplotlib.pyplot as plt
from pyscf import gto, mp, mcscf
import numpy as np
from pyscf.fci.cistring import _gen_occslst
import pyscf.ci
import pyscf.scf

all_geoms = load_geometries()
geom = [g for g in all_geoms.values() if g.comment == 'cumulene_C2H4_90deg']
assert len(geom) == 1
geom = geom[0]
basis_set = "sto-6g"
mol = geom.as_pyscf_molecule(basis_set)

def run_hf(smearing):
    mol = geom.as_pyscf_molecule(basis_set)
    mol.output = "/dev/null"
    hf = pyscf.scf.RHF(mol)
    hf.verbose = 0
    if smearing:
        hf = pyscf.scf.addons.smearing_(hf, sigma=smearing, method='fermi')
        hf.sigma = 0.05
        hf.max_cycle = 5
        hf.kernel()

    hf.sigma = 1e-6
    hf.max_cycle = 50
    hf.kernel()
    return hf.e_tot


for smearing in [True, False]:
    for i in range(10):
        energy = run_hf(smearing)
        print(f"Smearing: {smearing:<5}, Energy: {energy:.3f}")
# print("Running Hartree-Fock...")
hf = pyscf.scf.RHF(mol)
hf = pyscf.scf.addons.smearing_(hf, sigma=.1, method='fermi')

hf.sigma = .1
hf.max_cycle = 10
hf.kernel()

hf.sigma = 1e-6
hf.max_cycle = 50
hf.kernel()

print(f"Finished HF: {hf.e_tot:.3f}")

# print("Running CAS...")
# cas = hf.CASSCF(4, 4).run()
# cas.fix_spin_(ss=0.0)
# print("Finished running CAS.")

#%%
ci_coeffs = cas.ci.flatten()
idx_sorted = np.argsort(ci_coeffs**2)[::-1]
n_coeffs_max = 5

assert cas.ci.shape[0] == cas.ci.shape[1]
n_dets_per_spin = cas.ci.shape[0]

weights_cum = 0
print("i, idx_up, idx_dn, coeff, weight, weights_cum")
print("=============================================")
for i in range(min(n_coeffs_max, len(ci_coeffs))):
    coeff = ci_coeffs[idx_sorted[i]]
    weight = ci_coeffs[idx_sorted[i]]**2
    weights_cum += weight
    idx_det_up, idx_det_dn = np.divmod(idx_sorted[i], n_dets_per_spin)
    print(f"{i:2d} | {idx_det_up:3d},{idx_det_dn:3d} |    {coeff:+.3f}   {weight:.3f}, {weights_cum:.3f}")






