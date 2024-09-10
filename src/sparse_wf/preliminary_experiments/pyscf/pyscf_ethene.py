#%%
from sparse_wf.geometry import load_geometries
import pyscf.scf
import matplotlib.pyplot as plt
import numpy as np

all_geoms = load_geometries()
geom = [g for g in all_geoms.values() if g.comment == 'cumulene_C2H4_90deg']
assert len(geom) == 1
geom = geom[0]

mol = geom.as_pyscf_molecule('cc-pvtz')
hf = pyscf.scf.RHF(mol)
hf.kernel()

#%%
n_el = geom.n_el
n_orb_occ = n_el // 2
n_orb_vrt = len(hf.mo_energy) - n_orb_occ
plt.close("all")
plt.plot(hf.mo_energy, marker='o')
plt.axvline(n_orb_occ - 0.5, color='k')
plt.axhline(0, color='k')

E_occ = hf.mo_energy[:n_orb_occ]
E_vrt = hf.mo_energy[n_orb_occ:]
E_excitation = (E_vrt[:, None] - E_occ[None, :]).flatten()
ind_excitation = np.argsort(E_excitation)
ind_virt, ind_occ = np.divmod(ind_excitation, n_orb_occ)
E_excitation = E_vrt[ind_virt] - E_occ[ind_occ]

n_states = 10
for i in range(n_states):
    print(f"State {i:2d}: {ind_occ[i]:2d}->{ind_virt[i] + n_orb_occ:2d}: {E_excitation[i]:.3f} Ha")







