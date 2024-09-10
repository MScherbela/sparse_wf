#%%
from sparse_wf.geometry import load_geometries
import pyscf.scf
import matplotlib.pyplot as plt
from pyscf import gto, mp, mcscf
import numpy as np
from pyscf.fci.cistring import _gen_occslst
import pyscf.ci
import pyscf.scf
import pyscf.mcscf

all_geoms = load_geometries()
geoms = [g for g in all_geoms.values() if g.comment == 'cumulene_C4H4_90deg']
assert len(geoms) == 1
geom = geoms[0]
basis_set = "sto-6g"

mol = geom.as_pyscf_molecule(basis_set)
mol.output = "/dev/null"
hf = pyscf.scf.RHF(mol)
hf.verbose = 0
hf = pyscf.scf.addons.smearing_(hf, sigma=0.05, method='fermi')
hf.max_cycle = 5
hf.kernel()

hf.sigma = 1e-6
hf.max_cycle = 50
hf.kernel()

print(f"Finished HF: {hf.e_tot:.3f}")

print("Running CAS...")
cas = hf.CASSCF(4, 4)
cas.fix_spin_(ss=0)
cas.kernel()
print("Finished running CAS.")

is_large = (cas.ci**2) > 0.05
ind_large = np.where(is_large)
ind_large = np.array(ind_large).T
large_ci_coeffs = cas.ci[ind_large[:, 0], ind_large[:, 1]]
idx_sort = np.argsort(large_ci_coeffs**2)[::-1]
ind_large = ind_large[idx_sort]

det_strings = [f"{coeff:+.2f}|{u},{d}>" for (u,d), coeff in zip(ind_large, large_ci_coeffs)]
state_string = " ".join(det_strings)
s2 = pyscf.mcscf.spin_square(cas)[0]
print(f"S2={s2:.1f}, {state_string}")
print(f"E(CAS) = {cas.e_tot:.4f}")




