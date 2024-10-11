#%%
import numpy as np
from sparse_wf.geometry import Geometry, save_geometries

all_geoms = []
for n_atoms in [10, 20, 30]:
    Z = [1] * n_atoms
    for lattice in [3.0, 3.2]:
        for d_short in np.arange(1.2, lattice/2+0.01, 0.1):
            d_long = lattice - d_short
            R_unitcell = np.array([[0, 0, 0], [d_short, 0, 0]])
            shifts = np.arange(n_atoms//2)[:, None, None] * np.array([lattice, 0, 0])
            R = R_unitcell + shifts
            R = R.reshape(-1, 3)
            geom = Geometry(Z=Z, R=R, name=f"HChain{n_atoms}", comment=f"HChain{n_atoms}_{d_short:.2f}_{d_long:.2f}")
            all_geoms.append(geom)
save_geometries({g.hash: g for g in all_geoms})

