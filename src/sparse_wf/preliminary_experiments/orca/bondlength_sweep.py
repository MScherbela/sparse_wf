#%%
import numpy as np
from sparse_wf.preliminary_experiments.orca.run_orca import submit_to_slurm
from sparse_wf.geometry import load_geometries

all_geoms = list(load_geometries().values())

orca_path = "/home/scherbelam20/orca_6_0_1/orca"
n_cores = 16
total_memory = 25
frozen_core = False

for method, basis_set in [("PBE0", "def2-TZVP"), ("B3LYP", "def2-TZVP"), ("UHF", "cc-pVTZ")]:
    for n_carbon in [2, 4, 8, 12, 16, 20, 24, 30]:
        geom_names = [f"cumulene_C{n_carbon}H4_0deg_singlet", f"cumulene_C{n_carbon}H4_90deg_triplet"]
        geom_basis = [[g for g in all_geoms if g.comment == name][0] for name in geom_names]
        for bond_length in np.linspace(2.3, 2.6, 7):
            for g in geom_basis:
                R = g.R.copy()
                dx_hydrogen = np.abs(R[-4, 0])
                R[:-4, 0] = np.arange(n_carbon) * bond_length
                R[-2:, 0] = (n_carbon - 1) * bond_length + dx_hydrogen
                g_dict = dict(R=R.tolist(), Z=g.Z.tolist(), comment=f"{g.name}_{bond_length:.3f}", charge=g.charge, spin=g.spin)
                run_args = (None, g.hash, g_dict, method, basis_set, n_cores, total_memory, orca_path, frozen_core)
                submit_to_slurm(run_args)















