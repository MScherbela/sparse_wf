#%%
from matplotlib import lines
import wandb
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scienceplots
import colorsys
import re

from collections import defaultdict
import numpy as np
import jax


plt.style.use(["science", "grid"])

energies_kcal_per_mol = {
    "naphthalene": [61.0, -3.4, 64.4, 68.0, 62.6, 65.8, 66.2, 70.6, 64.7, 62.2, 67.1],
    "anthracene": [43.1, -2.3, 45.4, 46.2, 41.8, 48.2, 45.7, 45.5, 43.1, 43.2, 46.1],
    "tetracene": [29.4, -1.8, 31.2, 34.0, 27.7, 33.5, 32.1, 33.6, 28.8, 28.3, 31.6],
    "pentacene": [19.8, -1.5, 21.3, 25.2, 17.9, 25.3, 22.6, 25.4, 20.5, 18.0, 22.6],
    "hexacene": [12.4, -1.3, 13.7, np.nan, 10.9, 17.7, np.nan, 19.7, 15.0, 11.4, 16.8],
    # "heptacene": [np.nan, np.nan, np.nan, np.nan, 5.6, 13.4, np.nan, 16.5, 10.0, 7.7, 14.3]
}
methods = [
    "exp",
    "ZPE",
    "ZPE-corr'd exp",
    "AFQMC",
    "UB3LYP",
    "CCSD(T)/FPA",
    "B3LYP/pp-RPA",
    "GAS-pDFT (FP-1)",
    "GAS-pDFT (WFP-3)",
    "ACI-DSRG-MRPT2",
    "DMRG-pDFT",
]
afqmc_error = np.array([1.2, 1.2, 1.6, 1.6, 0]) * 1.6
reference = pd.DataFrame(energies_kcal_per_mol, index=methods) * 1.6

api = wandb.Api()
runs = ["tum_daml_nicholas/acene/kiq6t01c", "tum_daml_nicholas/acene/5lsw3pnf"]
# ending = "_ccecpccpvdz"
ending = "_splus_new"
pattern = re.compile(f"{ending}(?:_from\\d{{6}})?$")
# ending = "_b3lyp"

runs = list(api.runs("tum_daml_nicholas/acene"))
runs = [r for r in runs if pattern.search(r.name) and 'opt/E' in r.summary]
runs = sorted(runs, key=lambda r: r.summary['_timestamp'])
_runs = defaultdict(lambda: defaultdict(list))
for r in runs:
    _runs[r.name.split("_")[1]][r.name.split("_")[2]].append(r)
runs = {k: dict(v) for k, v in _runs.items()}
energies = jax.tree.map(
    lambda r: pd.DataFrame(r.scan_history(keys=["opt/E", 'opt/E_std', "opt/step"])).set_index('opt/step').sort_index(), runs, is_leaf=lambda x: isinstance(x, wandb.apis.public.Run)
)
energies = {
    k: {
        s: pd.concat(d)
        for s, d in v.items()
    }
    for k, v in energies.items()
}

full_df = pd.concat([
    d[['opt/E_std']].rename(columns={'opt/E_std': (k, s)})[~d.index.duplicated(keep='first')]
    for k, v in energies.items()
    for s, d in v.items()
], axis=1)
tuples = full_df.transpose().index
new_columns = pd.MultiIndex.from_tuples(tuples, names=['Molecule', 'State'])
full_df.columns = new_columns
full_df.to_csv('acene_std.csv')

full_df = pd.concat([
    d[['opt/E']].rename(columns={'opt/E': (k, s)})[~d.index.duplicated(keep='first')]
    for k, v in energies.items()
    for s, d in v.items()
], axis=1)
tuples = full_df.transpose().index
new_columns = pd.MultiIndex.from_tuples(tuples, names=['Molecule', 'State'])
full_df.columns = new_columns
full_df.to_csv('acene.csv')

# %%
full_df = pd.read_csv('acene.csv', header=[0, 1], index_col=0).sort_index()
full_df_std = pd.read_csv('acene_std.csv', header=[0, 1], index_col=0).sort_index()
final_energies = {}
final_errors = {}
for k in full_df.keys():
    mean = full_df[k].dropna().rolling(5000).mean().dropna().iloc[-1]
    err = full_df_std[k].dropna().rolling(5000).mean().dropna().iloc[-1] / np.sqrt(5000 * 4096)
    final_energies[k] = mean
    final_errors[k] = err
final_energies = pd.Series(final_energies)
final_errors = pd.Series(final_errors)
errors = (final_errors.unstack()['triplet'] + final_errors.unstack()['singlet']) * 1000

def scale_lightness(rgb, scale_l):
    rgb = matplotlib.colors.ColorConverter.to_rgb(rgb)
    # convert rgb to hls
    h, l, s = colorsys.rgb_to_hls(*rgb)
    # manipulate h, l, s values and return as rgb
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s = s)
final_deltas = {
    m: (final_energies[m]['triplet'] - final_energies[m]['singlet']) * 1000
    for m in final_energies.index.levels[0]
    if 'singlet' in final_energies[m] and 'triplet' in final_energies[m]
}
order = list(energies_kcal_per_mol.keys())
colors = ['e15759', '4e79a7', 'f28e2b', '59a14f', '9c755f', 'b07aa1', '76b7b2', 'ff9da7', 'edc948', 'bab0ac']
colors = [f"#{c}" for c in colors]
line_styles = [
    'solid',
    'dashed',
    'dotted',
    'dashdot',
    (0, (5, 1)),
    (0, (3, 5, 1, 5, 1, 5))
]

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 5), width_ratios=[0.9, 1], dpi=200)
ax.plot(pd.Series(final_deltas)[order], 's', color=colors[0], label='FiRE', zorder=5, linestyle=line_styles[0])
ax.plot(reference.loc['ZPE-corr\'d exp'][order], '*', color='black', label='exp', zorder=-1, linestyle=line_styles[1])
ax.fill_between(range(len(order)), reference.loc['ZPE-corr\'d exp'][order] - 1.6, reference.loc['ZPE-corr\'d exp'][order] + 1.6, color='black', alpha=0.1, zorder=-10, label='exp $\pm$ chem. acc')
ax.plot(reference.loc['CCSD(T)/FPA'][order], '^', color=colors[1], label='CCSD(T)/FPA', zorder=3, linestyle=line_styles[2])
ax.plot(reference.loc['ACI-DSRG-MRPT2'][order], 'v', color=colors[2], label='ACI-DSRG-MRPT2', zorder=4, linestyle=line_styles[3])
ax.plot(reference.loc['AFQMC'][order], 'o', color=colors[3], label='AFQMC', zorder=2, linestyle=line_styles[4])
# ax.errorbar(range(len(order)), reference.loc['AFQMC'][order], afqmc_error, color=colors[3], zorder=2, capsize=4)
# ax.set_xlabel('$n$-acene')
ax.set_ylabel(r"$E_\text{triplet} - E_\text{singlet}$ [m$E_\text{h}$]")
ax.tick_params(axis='x', which='minor', bottom=False, top=False)
ax.set_title('Absolute singlet-triplet gap')
ax.set_xticks(range(len(order)), order)
handles, labels = ax.get_legend_handles_labels()
legend_dict = dict(zip(labels, handles))
legend_dict['exp $\pm$ chem. acc'] = (legend_dict.pop('exp'), legend_dict['exp $\pm$ chem. acc'])
leg = ax.legend(legend_dict.values(), legend_dict.keys(), loc='upper right')

# ax2 = ax.inset_axes([0.45, 0.5, 0.55, 0.5])
x = np.arange(len(order))
w = 0.175
n = 4
pos = x + (n+1)/2 * w
ax2.bar(pos := pos - w, pd.Series(final_deltas)[order] -reference.loc['ZPE-corr\'d exp'][order], width=w, color=colors[0], label='FiRE', zorder=10, linestyle=line_styles[0])
ax2.bar(pos := pos - w, reference.loc['CCSD(T)/FPA'][order] - reference.loc['ZPE-corr\'d exp'][order], width=w, color=colors[1], label='CCSD(T)/FPA', zorder=5, linestyle=line_styles[2])
ax2.bar(pos := pos - w, reference.loc['ACI-DSRG-MRPT2'][order] - reference.loc['ZPE-corr\'d exp'][order], width=w, color=colors[2], label='ACI-DSRG-MRPT2', zorder=6, linestyle=line_styles[3])
ax2.bar(pos := pos - w, reference.loc['AFQMC'][order] - reference.loc['ZPE-corr\'d exp'][order], width=w, color=colors[3], label='AFQMC', zorder=4, linestyle=line_styles[4])
for container in ax2.containers:
    # pad = 14 if container.get_label() == 'AFQMC' else 4
    pad = 4
    ax2.bar_label(container, fmt='%.1f', padding=pad, zorder=11, fontsize=8)
# ax2.errorbar(pd.Series(final_deltas)[order] - reference.loc['ZPE-corr\'d exp'][order], x + (n+1)/2 * w - w, xerr=errors[order], color=scale_lightness(colors[0], 0.7), capsize=2, label='FiRE', linestyle='', zorder=10)
# ax2.errorbar(reference.loc['AFQMC'][order] - reference.loc['ZPE-corr\'d exp'][order], pos, xerr=afqmc_error, color=scale_lightness(colors[3], 0.7), capsize=2, label='AFQMC', linestyle='', zorder=5)
ax2.axhline(0, color='black', zorder=100, linestyle=line_styles[1])
ax2.axhspan(-1.6, 1.6, color='black', alpha=0.1, zorder=-10, label='exp$\pm$ chem. acc')
ax2.set_xticks(x, reference.columns)
# ax2.set_xticklabels([])
ax2.set_ylim(-7.5, 7.5)
ax2.grid(False, axis="x")
ax2.tick_params(axis='x', which='minor', left=False, right=False)
ax2.set_title(r"$\Delta - \Delta_\text{exp}$ [m$E_\text{h}$]")
# ax2.set_xlabel('$n$-acene')

# for ax, label in zip((ax, ax2), "cd"):
#     ax.text(0, 1.02, f"{label})", transform=ax.transAxes, va="bottom", ha="left", fontweight="bold", fontsize=12)

# fig.subplots_adjust(wspace=0.35)
fig.tight_layout()
plt.savefig("acene_final.pdf", bbox_inches="tight")
#%%


# #%%
# our = final_energies.unstack()
# our[r'FiRE'] = (our['triplet'] - our['singlet']) * 1000
# our = our.rename(columns={'triplet': r'$E^\text{FiRE}_\text{triplet}$ (Ha)', 'singlet': r'$E^\text{FiRE}_\text{singlet}$ (Ha)'})
# our = our.dropna().T
# columns = ['ZPE-corr\'d exp', 'FiRE', 'CCSD(T)/FPA', 'ACI-DSRG-MRPT2', 'AFQMC']
# print(pd.concat([our, reference]).T[columns].loc[order].to_latex(float_format="%.1f"))
# # %%

# #%%
# #%%
# window = 5000
# fig, axes = plt.subplots(1, len(energies_kcal_per_mol), figsize=(10, 3))
# axes = np.array([axes]).reshape(-1)
# colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
# ref_data = reference.loc[["AFQMC", "CCSD(T)/FPA", "ACI-DSRG-MRPT2"]]
# for ax, k in zip(axes, energies_kcal_per_mol.keys()):
#     states = full_df[k]
#     print(k)
#     if "triplet" not in states:
#         continue
#     diff = (states["triplet"] - states["singlet"]) * 1000
#     diff = diff.dropna()
#     avg = diff.rolling(window, min_periods=1).mean()
#     mad = np.abs(diff - avg).mean()
#     diff = diff[np.abs(diff - avg) < 10 * mad]
#     diff = diff.rolling(window, min_periods=1).mean()
#     diff = diff.dropna()
#     x = diff.index
#     c_iter = iter(colors)
#     x_max = x.max() * 1.05
#     ax.plot(x, diff, label="SWANN (c=3)", color=next(c_iter))
#     exp = reference[k]["ZPE-corr'd exp"]
#     ax.axhline(exp, label="exp", color="black")
#     ax.fill_between(
#         [x.min(), x_max], exp - 1.6, exp + 1.6, color="black", alpha=0.1, label="exp $\pm$ chem. acc "
#     )
#     for ref, name in ref_data[k].items():
#         ax.axhline(name, label=ref, color=next(c_iter))
#     ax.set_title(k)
#     ax.set_xlabel("Step")
#     ax.set_xlim(x.min(), x_max)
#     ax.set_ylim(ref_data[k].min() - 5, ref_data[k].max() + 5)
#     handles, labels = ax.get_legend_handles_labels()
#     legend_dict = dict(zip(labels, handles))
# # axes[0].set_ylabel("Energy difference [mHa]")
# axes[0].set_ylabel(r"$E_\text{triplet} - E_\text{singlet}$ / mHa")
# fig.legend(legend_dict, loc="upper center", bbox_to_anchor=(0.5, 0), ncol=6)
# plt.savefig("acene_convergence.pdf", bbox_inches="tight")

# # %%
# # %%
# final_energies = jax.tree.map(
#     lambda x: x.rolling(5000).mean().iloc[-1]['opt/E'],
#     energies,
# )
# final_deltas = {
#     k: (v['triplet'] - v['singlet']) * 1000
#     for k, v in final_energies.items()
#     if 'singlet' in v and 'triplet' in v
# }
# fig, axes = plt.subplots(1, len(energies_kcal_per_mol), figsize=(8, 3), sharey=True, sharex=True)
# w = .75
# n = 4
# colors = ['4e79a7', 'f28e2b', '59a14f', '9c755f', 'e15759', 'b07aa1', '76b7b2', 'ff9da7', 'edc948', 'bab0ac']
# colors = [f"#{c}" for c in colors]
# print(colors)
# for (i, k), ax in zip(enumerate(energies_kcal_per_mol), axes):
#     c_iter = iter(colors)
#     pos = 0 - (n-1)/2 * w
#     # ax.bar(i-2*w, reference[k]["ZPE-corr'd exp"], width=w, color=next(c_iter))
#     ax.bar(0, final_deltas[k] - reference[k]["ZPE-corr'd exp"], width=w, color=next(c_iter), label='SWANN')
#     ax.bar(1, reference[k]["CCSD(T)/FPA"] - reference[k]["ZPE-corr'd exp"], width=w, color=next(c_iter), label='CCSD(T)/FPA')
#     ax.bar(2, reference[k]["ACI-DSRG-MRPT2"] - reference[k]["ZPE-corr'd exp"], width=w, color=next(c_iter), label='ACI-DSRG-MRPT2')
#     ax.bar(3, reference[k]["AFQMC"] - reference[k]["ZPE-corr'd exp"], width=w, color=next(c_iter), label='AFQMC')
#     ax.axhspan(-1.6, 1.6, color='black', alpha=0.1, zorder=-10, label='exp$\pm$ chem. acc')
#     for container in ax.containers:
#         ax.bar_label(container, fmt='%.1f', padding=3)
#     if i == 0:
#         handles, labels = ax.get_legend_handles_labels()
#         legend_dict = dict(zip(labels, handles))
#     ax.set_xticks(np.arange(4), [])
#     ax.set_title(k)
# # ax.set_xticks(range(len(energies_kcal_per_mol)), energies_kcal_per_mol.keys());
# fig.legend(legend_dict, loc="upper center", bbox_to_anchor=(0.5, 0.1), ncol=6)
# axes[0].set_ylim(-8, 8)
# axes[0].set_ylabel(r"$(E_\text{triplet} - E_\text{singlet}) - \Delta_\text{exp}$ / mHa")
# plt.savefig("acene_relative.pdf", bbox_inches="tight")
# # plt.xlabel("Molecule")
# # %%
# reference
# # %%
# print('our MAE:', np.mean(np.abs([final_deltas[k]- reference[k]["ZPE-corr'd exp"] for k in energies_kcal_per_mol])))
# print('CCSD(T)/FPA MAE:', np.mean(np.abs([reference[k]["CCSD(T)/FPA"] - reference[k]["ZPE-corr'd exp"] for k in energies_kcal_per_mol])))
# print('ACI-DSRG-MRPT2 MAE:', np.mean(np.abs([reference[k]["ACI-DSRG-MRPT2"] - reference[k]["ZPE-corr'd exp"] for k in energies_kcal_per_mol])))
# print('AFQMC MAE:', np.nanmean(np.abs([reference[k]["AFQMC"] - reference[k]["ZPE-corr'd exp"] for k in energies_kcal_per_mol])))


# # %%
# final_energies = jax.tree.map(
#     lambda x: x.rolling(5000).mean().iloc[-1]['opt/E'],
#     energies,
# )
# final_deltas = {
#     k: (v['triplet'] - v['singlet']) * 1000
#     for k, v in final_energies.items()
#     if 'singlet' in v and 'triplet' in v
# }
# fig, ax = plt.subplots(figsize=(8, 3))
# w = 0.1
# n = 4
# colors = ['4e79a7', 'f28e2b', '59a14f', '9c755f', 'e15759', 'b07aa1', '76b7b2', 'ff9da7', 'edc948', 'bab0ac']
# colors = [f"#{c}" for c in colors]
# for i, k in enumerate(energies_kcal_per_mol):
#     c_iter = iter(colors)
#     pos = i - (n)/2 * w
#     ax.bar(pos, reference[k]["ZPE-corr'd exp"], width=w, color=next(c_iter))
#     ax.bar(pos := pos + w, final_deltas[k], width=w, color=next(c_iter), label='SWANN')
#     ax.bar(pos := pos + w, reference[k]["CCSD(T)/FPA"], width=w, color=next(c_iter), label='CCSD(T)/FPA')
#     ax.bar(pos := pos + w, reference[k]["ACI-DSRG-MRPT2"], width=w, color=next(c_iter), label='ACI-DSRG-MRPT2')
#     ax.bar(pos := pos + w, reference[k]["AFQMC"], width=w, color=next(c_iter), label='AFQMC')
#     if i == 0:
#         handles, labels = ax.get_legend_handles_labels()
#         legend_dict = dict(zip(labels, handles))
# ax.set_xticks(range(len(energies_kcal_per_mol)), energies_kcal_per_mol.keys());
# fig.legend(legend_dict, loc="upper center", bbox_to_anchor=(0.5, 0.05), ncol=6)
# plt.ylabel(r"$E_\text{triplet} - E_\text{singlet}$ / mHa")
# plt.savefig("acene_gap.pdf", bbox_inches="tight")

# # %%
# full_df[full_df.keys().levels[1][:1]]
# # %%
# full_df.keys().levels[1][:1]
# # %%
