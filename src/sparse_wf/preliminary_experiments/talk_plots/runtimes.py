#%%
import numpy as np
from sparse_wf.model.utils import cutoff_function
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(["science"])
from sparse_wf.plot_utils import COLOR_PALETTE, COLOR_FIRE

data = [
    ("\\textbf{FermiNet}\nDeepMind\n2020", 59.310190, COLOR_PALETTE[0]),
    ("\\textbf{Psiformer}\nDeepMind\n2022", 67.171134, COLOR_PALETTE[0]),
    ("\\textbf{LapNet}\nByteDance\n2023", 47.870060, COLOR_PALETTE[0]),
    ("\\textbf{Naive FiRE}\nOurs", 45.425663, COLOR_FIRE),
    ("\\textbf{FiRE}\nOurs", 2.917086, COLOR_FIRE),
]

fig, ax = plt.subplots(1, 1, figsize=(6, 4.2), dpi=400)
for i, (name, t, color) in enumerate(data):
    ax.bar([i], [t], color=color, zorder=3)
ax.bar([0], [0], color=COLOR_PALETTE[0], label="fully connected\nSOTA models")
ax.bar([0], [0], color=COLOR_FIRE, label="cutoff")
ax.legend(frameon=True)
ax.set_xticks(np.arange(len(data)), [d[0] for d in data])
ax.set_ylabel("time per step [sec]")
ax.set_xlim([-0.5, 3.5])
ax.grid(axis="y", alpha=0.5)
ax.grid(False, axis="x")
ax.set_ylim([0, 75])


