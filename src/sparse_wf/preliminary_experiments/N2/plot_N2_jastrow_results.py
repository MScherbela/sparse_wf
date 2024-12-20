#%%
import wandb
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
import seaborn as sns

api = wandb.Api()

runs = api.runs("tum_daml_nicholas/N2_jastrow")

# regex that matches anything of the format "N2_jas_50k_2.068_4.0_True",
# where the numbers are arbietrary floats and the last token is either "True" or "False"
runs = [r for r in runs if re.match(r"N2_jas_50k_\d+\.\d+_\d+\.\d+_(True|False)", r.name)]

df = []
for r in runs:
    print(r.name)
    tokens = r.name.split("_")[-3:]
    energies = r.history(keys=["opt/step", "opt/E", "opt/E_std"], pandas=True, samples=10_000)
    energies = energies[energies["opt/step"] >= 45_000]
    if len(energies) < 50:
        print("Skipping", r.name)
    E = energies["opt/E"].mean()
    E_sigma = energies["opt/E"].std() / np.sqrt(len(energies))
    E_std = energies["opt/E_std"].mean()

    df.append(
        dict(
            d=float(tokens[0]),
            cutoff=float(tokens[1]),
            jastrow="true" in tokens[2].lower(),
            E=E,
            E_sigma=E_sigma,
            E_std = E_std,
            n_samples=len(energies),
        ))
df = pd.DataFrame(df)

E_ref = {
    2.068: -109.536991,
    4.0: -109.194726,
    6.0: -109.173851
}
df["E_ref"] = df["d"].map(E_ref)
df["error"] = (df["E"] - df["E_ref"]) * 1000
df["error_sigma"] = df["E_sigma"] * 1000

#%%
plt.close("all")
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

for idx_cutoff, cutoff in enumerate(df["cutoff"].unique()):
    for jastrow in [False, True]:
        color = "C" + str(idx_cutoff)
        ls = "-" if jastrow else "--"
        label = f"cutoff={cutoff}"
        if jastrow:
            label += " + pairwise MLP jastrow"

        df_filt = df[(df["cutoff"] == cutoff) & (df["jastrow"] == jastrow)]
        axes[0].errorbar(df_filt["d"], df_filt["error"], yerr=df_filt["error_sigma"], label=label, color=color, ls=ls, capsize=5)
        axes[1].plot(df_filt["d"], df_filt["E_std"], label=label, color=color, ls=ls)

for ax in axes:
    ax.legend()
    ax.set_xlabel("bond length / a.u.")
    ax.grid(alpha=0.5)
axes[0].set_ylabel("error vs. MRCI / mHa")
axes[1].set_ylabel("std(E) / Ha")
fig.suptitle("N2, new sparse embedding, spin-restricted, 50k steps")
fig.tight_layout()
fig.savefig("/home/mscherbela/ucloud/results/N2_jastrow.png", dpi=200)



