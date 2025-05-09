#%%
from sparse_wf.pseudopotentials import eval_ecp_on_grid
from sparse_wf.geometry import PERIODIC_TABLE, BOHR_IN_ANGSTROM
import pyscf.gto
import numpy as np
import matplotlib.pyplot as plt

ecp = "ccecp"
Z_values = [6, 7, 8, 25]
ecp_data = {}
for Z in Z_values:
    symbol = PERIODIC_TABLE[Z-1]
    ecp_data[Z] = pyscf.gto.basis.load_ecp(ecp, symbol)
n_cores, v_grid_dict, grid_radius, max_channels = eval_ecp_on_grid(ecp_data)

fig, axes = plt.subplots(2, 2)
for Z, ax in zip(Z_values, axes.flat):
    non_loc_grid_values, loc_grid_values = np.split(v_grid_dict[Z], (max_channels - 1,), axis=0)

    for cutoff in [1e-3]:
        larger_than_cutoff = np.abs(non_loc_grid_values) > cutoff
        cutoff_idx = np.max(larger_than_cutoff * np.arange(grid_radius.shape[-1])[None, :])
        r_cut = grid_radius[cutoff_idx]
        print(f"Z={Z}, f^-1({cutoff})={r_cut * BOHR_IN_ANGSTROM:.3f} A")

    r_plot_max = 2.0
    idx_plot_max = np.where(grid_radius>r_plot_max)[0][0]
    ax.plot(grid_radius[:idx_plot_max], np.abs(non_loc_grid_values[:, :idx_plot_max].T))
    ax.set_yscale("log")
    # ax.set_ylim([1e-6, 1e3])