# %%
import matplotlib.pyplot as plt
import pandas as pd
from sparse_wf.plot_utils import COLOR_PALETTE, COLOR_FIRE, MILLIHARTREE
import matplotlib as mpl

mpl.style.use(["science", "grid"])

colors = {
    "UCCSD(T)": COLOR_PALETTE[0],
    "AFQMC": COLOR_PALETTE[1],
    "MR-ACPF": COLOR_PALETTE[2],
    "Conventional VMC": "gray",
    "NN-VMC": COLOR_FIRE,
    "NN-VMC (FermiNet)": COLOR_FIRE,
}

fnames = ["n2_data.csv", "h10_data.csv"]
titles = ["N$_2$", "H$_{10}$"]
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4), sharey=True, dpi=200)
for i, (ax, fname, title) in enumerate(zip(axes, fnames, titles)):
    df = pd.read_csv(fname)
    df.columns = ["method", "d", "error"]
    for method in df.method.unique():
        df_method = df[df.method == method]
        ax.plot(df_method.d, df_method.error, "o-", label=method, color=colors[method])
    ax.set_xlabel("bond length [bohr]")
    ax.legend()
    ax.set_title(title)
    ax.axhline(0, color="black", linestyle="-", zorder=-1)
axes[0].set_ylabel("energy error " + MILLIHARTREE)
