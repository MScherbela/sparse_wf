#%%
import numpy as np
from sparse_wf.model.utils import cutoff_function
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(["science"])
from sparse_wf.plot_utils import COLOR_PALETTE, COLOR_FIRE

data = [
    ("FermiNet\nDeepMind", 59.310190, COLOR_PALETTE[0]),
    ("Psiformer\nDeepMind", 67.171134, COLOR_PALETTE[0]),
    ("LapNet\nByteDance", 47.870060, COLOR_PALETTE[0]),
    ("Naive FiRE", 45.425663, COLOR_FIRE),
    ("FiRE", 2.917086, COLOR_FIRE),
]

fig, ax = plt.subplots(1, 1, figsize=(6, 4.2), dpi=400)
for i, (name, t, color) in enumerate(data):
    ax.bar([i], [t], color=color, zorder=3)
ax.bar([0], [0], color=COLOR_PALETTE[0], label="fully connected")
ax.bar([0], [0], color=COLOR_FIRE, label="cutoff")
ax.legend(frameon=True)
ax.set_xticks(np.arange(len(data)), [d[0] for d in data])
ax.set_ylabel("timer per step [sec]")
ax.set_xlim([-0.5, 3.5])
ax.grid(axis="y", alpha=0.5)
ax.grid(False, axis="x")
ax.set_ylim([0, 75])


