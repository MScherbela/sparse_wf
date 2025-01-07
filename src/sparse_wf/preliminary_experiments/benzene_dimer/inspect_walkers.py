#%%
import numpy as np
from collections import Counter
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import os

# extra_top_electrons = np.arange(-3, 4)
# counts = []

df_data = []

run_names = [
    "5.0_benzene_dimer_T_10.00A_single-electron_0.5_2",
    # "5.0_benzene_dimer_T_10.00A"
    ]
for run_name in run_names:
    for n_steps in [0, 2282, 5000, 10_000]:
        fname = f"/storage/scherbelam20/runs/sparse_wf/benzene_dimer/cutoff5.0/{run_name}/optchkpt{n_steps:06d}/electrons.npz"
        if not os.path.exists(fname):
            continue

        data = np.load(fname)

        r = data[""][0]
        n_el = r.shape[-2]

        R = np.array([[0.0, 2.63955, 0.0], [0.0, -2.63955, 0.0], [2.28592, 1.31978, 0.0], [-2.28592, -1.31978, 0.0], [-2.28592, 1.31978, 0.0], [2.28592, -1.31978, 0.0], [0.0, 4.69448, 0.0], [4.06554, 2.34724, 0.0], [-4.06554, -2.34724, 0.0], [-4.06554, 2.34724, 0.0], [4.06554, -2.34724, 0.0], [0.0, -4.69448, 0.0], [0.0, 0.0, 13.97791], [0.0, 0.0, 8.6988], [2.28592, 0.0, 12.65813], [-2.28592, 0.0, 10.01858], [-2.28592, 0.0, 12.65813], [2.28592, 0.0, 10.01858], [0.0, 0.0, 16.03284], [4.06554, 0.0, 13.6856], [-4.06554, 0.0, 8.99112], [-4.06554, 0.0, 13.6856], [4.06554, 0.0, 8.99112], [0.0, 0.0, 6.64388]])
        is_bottom_atom = R[:, 2] == 0
        R_bottom = R[is_bottom_atom]
        R_top = R[~is_bottom_atom]

        dist_bottom = np.min(np.linalg.norm(r[:, :, None, :] - R_bottom, axis=-1), axis=-1)
        dist_top = np.min(np.linalg.norm(r[:, :, None, :] - R_top, axis=-1), axis=-1)
        n_top = np.sum(dist_bottom > dist_top, axis=-1)

        hist = Counter(n_top - n_el // 2)
        for k, v in hist.items():
            df_data.append(dict(n_steps=n_steps, extra_top_electrons=k, count=v, sampling="single-electron" if "single-electron" in run_name else "cluster-update"))
df = pd.DataFrame(df_data)

df_extra = df[df.extra_top_electrons != 0]
df_extra["percent_of_batch"] = df_extra["count"] / 4096 * 100
fig, ax = plt.subplots(1,1, figsize=(8, 6))
ax.yaxis.set_major_formatter(PercentFormatter())
sns.lineplot(data=df_extra, x="n_steps", y="percent_of_batch", hue="extra_top_electrons", style="sampling", ax=ax, palette="tab10", markers="o")
ax.axhline(0, color="black", linestyle="-", zorder=-1)
fig.savefig("extra_top_electrons.png", dpi=200)





