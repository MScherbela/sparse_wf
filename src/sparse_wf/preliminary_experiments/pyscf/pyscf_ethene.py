#%%
from sparse_wf.geometry import load_geometries
import pyscf.scf
import matplotlib.pyplot as plt

all_geoms = load_geometries()
geom = [g for g in all_geoms.values() if g.comment == 'cumulene_C2H4_90deg']
assert len(geom) == 1
geom = geom[0]

mol = geom.as_pyscf_molecule('cc-pvtz')
hf = pyscf.scf.RHF(mol)
hf.kernel()

#%%
plt.close("all")
plt.plot(hf.mo_energy)





