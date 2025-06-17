#%%
import numpy as np
from sparse_wf.model.utils import cutoff_function
from sparse_wf.plot_utils import COLOR_FIRE
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(["science"])

x = np.linspace(0, 1.2, 200)
y = cutoff_function(x)

fig, ax = plt.subplots(1, 1, figsize=(5, 2), dpi=300)
ax.plot(x, y, lw=3, color=COLOR_FIRE)
ax.set_ylabel("cutoff function", fontsize=16)
ax.set_xlabel("$r / r_\\text{cut}$", fontsize=16)


