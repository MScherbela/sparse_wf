#%%
from sparse_wf.geometry import load_geometries
import pyscf.scf
import numpy as np

geom_names = ["WG_01", "WG_09", "WG_15"]
all_geoms = load_geometries()
geoms = [g for g in all_geoms.values() if g.comment in geom_names]

ecp = {k: "ccecp" for k in ["C", "N", "O"]}
for spin in [0, 2]:
    for basis in ["STO-6G"]:
        energies = []
        for g in geoms:
            print(g.comment)
            g.spin = spin
            mol = g.as_pyscf_molecule(basis, ecp)
            hf = pyscf.scf.RHF(mol)
            hf.kernel()
            print(hf.e_tot)
            energies.append(hf.e_tot)
        delta_E = np.array(energies)
        delta_E = 1000 * (delta_E - delta_E.mean())
        print("="*20)
        print(f"{basis}, ecp={ecp}")
        for name, E in zip(geom_names, delta_E):
            print(f"{name}: {E:.2f} mHa")
        print("="*20)
