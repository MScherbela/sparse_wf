# %%
import wandb
import pandas as pd
import re

api = wandb.Api()
all_runs = api.runs("tum_daml_nicholas/benzene")
runs = [r for r in all_runs if re.match("HLR.*", r.name)]
runs = [r for r in runs if "eval" not in r.name]

data = []
for r in runs:
    print(r.name)
    dist = float(re.search(r"(\d+\.\d+)A", r.name).group(1))
    cutoff = r.name.split("_")[1]
    if r.name.startswith("HLRTransfer"):
        cutoff = "Transfer" + cutoff
    metadata = dict(
        dist=dist,
        cutoff=cutoff,
        run_name=r.name,
    )

    history = []
    for h in r.scan_history(["opt/step", "opt/E"], page_size=10_000):
        history.append(h | metadata)
    df = pd.DataFrame(history)
    if len(df):
        df = df.sort_values("opt/step").iloc[1:] # drop first step
        data.append(df)
    else:
        print(f"No data for {r.name}")

df_all = pd.concat(data)
df_all.to_csv("benzene_energies.csv", index=False)

# %%
import matplotlib.pyplot as plt
import numpy as np

def get_outlier_mask(x):
    qlow = x.quantile(0.01)
    qhigh = x.quantile(0.99)
    med = x.median()
    included_range = 5 * (qhigh - qlow)
    is_outlier = (x < med - included_range) | (x > med + included_range)
    return is_outlier


window_kwargs = dict(    window = 5000,    min_periods = 1000)
# cutoffs = [3.0, 5.0]
cutoffs = ["3.0", "Transfer5.0" ,"Transfer7.0"]
dists = [4.95, 10.0]

df_all = pd.read_csv("benzene_energies.csv")
# molecules = sorted(df_all["molecule"].unique())
pivot = df_all.pivot_table(index="opt/step", columns=["cutoff", "dist"], values="opt/E", aggfunc="mean")
pivot = pivot.ffill(limit=10)
for cutoff in cutoffs:
    # for dist in dists:
    #     smoothed = pivot[(cutoff, dist)].fillna(method="ffill", limit=20).rolling(**window_kwargs).mean()
    #     pivot.loc[:, (cutoff, f"E{dist}_smooth")] = smoothed
    # pivot.loc[:, (cutoff, "delta_smooth")] = (pivot[cutoff][f"E{dists[0]}_smooth"] - pivot[cutoff][f"E{dists[1]}_smooth"]) * 1000
    pivot.loc[:, (cutoff, "delta")] = (pivot[(cutoff, dists[0])] - pivot[(cutoff, dists[1])]) * 1000
    is_outlier = get_outlier_mask(pivot[(cutoff, "delta")])
    pivot.loc[is_outlier, (cutoff, "delta")] = np.nan
    pivot.loc[:, (cutoff, "delta")] = pivot.loc[:, (cutoff, "delta")].ffill(limit=10)
    pivot.loc[:, (cutoff, "delta_smooth")] = pivot.loc[:, (cutoff, "delta")].rolling(**window_kwargs).mean()
    pivot.loc[:, (cutoff, "delta_stderr")] = pivot.loc[:, (cutoff, "delta")].rolling(**window_kwargs).std() / np.sqrt(window_kwargs["window"])


refs = {
    "Experiment": (-3.8, "k"),
    "PsiFormer": (5.0, "forestgreen"),
    "FermiNet VMC (Glehn et al)": (-4.6, "C0"),
    "FermiNet DMC (Ren et al)": (-9.2, "navy"),
}

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
for ref, (E_ref, color) in refs.items():
    ax.axhline(E_ref, color=color, linestyle="dashed")
    if ref == "Experiment":
        ax.axhspan(E_ref - 0.6, E_ref + 0.6, color=color, alpha=0.2, zorder=-1)
    ax.text(0.1, E_ref, ref, color=color, va="bottom", ha="left")

cmap = plt.get_cmap("YlOrRd")
colors = [cmap(i) for i in np.linspace(0.3, 0.9, len(cutoffs))]
steps_max = [25, 50, 75]
for cutoff, max_opt_step, color in zip(cutoffs, steps_max, colors):
    df_cutoff = pivot[cutoff]
    df_cutoff = df_cutoff[df_cutoff.index < max_opt_step * 1000]
    delta_E = df_cutoff["delta_smooth"]
    delta_Estd = df_cutoff["delta_stderr"]
    delta_E_final = delta_E[delta_E.notna()].iloc[-1]
    ax.plot(df_cutoff.index / 1000,  delta_E, label=f"SWANN cutoff={cutoff}", color=color)
    ax.axhline(delta_E_final, color=color, zorder=0, ls="--")
    ax.fill_between(
        df_cutoff.index / 1000,
        delta_E - 2 * delta_Estd,
        delta_E + 2 * delta_Estd,
        color=color,
        alpha=0.2,
    )
    print(f"{cutoff}: {delta_E_final:.1f} mHa")
ax.legend(loc="upper right")
ax.set_ylim([-10, 6])
ax.set_xlim([0, None])
ax.set_xlabel("Opt Step / k")
ax.set_ylabel("E_4.95 - E_10.0 / mHa")
ax.set_title("Benzene dimer binding energy")
fig.savefig("benzene_dimer.png")
