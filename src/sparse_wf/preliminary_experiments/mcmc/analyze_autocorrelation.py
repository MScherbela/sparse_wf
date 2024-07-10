#%%
import pandas as pd
from pathlib import Path
from ast import literal_eval
import numpy as np
import matplotlib.pyplot as plt

all_data = {}
logfiles = Path.glob(Path("/storage/scherbelam20/runs/sparse_wf/cluster_update"), "C16H4_eval_batch16*/log.txt")
for logfile in logfiles:
    run_name = logfile.parent.name
    eval_data = []
    print(run_name)
    with open(logfile, "r") as f:
        for line in f:
            if "molecule_args" in line:
                config = literal_eval(line)
            if "eval/E" in line:
                eval_data.append(literal_eval(line))
    all_data[run_name] = (config, pd.DataFrame(eval_data))

#%%
def get_autocorrelation(energy, max_lag=1000):
    energy = energy - np.mean(energy)
    corr = [np.corrcoef(energy[:-lag], energy[lag:])[0, 1] for lag in range(1, max_lag+1)]
    return np.array([1, *corr])

def get_ind_cut(corr, cutoff=1e-2):
    return np.where(corr < cutoff)[0][5]

def get_integrated_corr_time(corr, cutoff=1e-2):
    ind_cut = get_ind_cut(corr, cutoff)
    return 1 + 2 * np.sum(corr[1:ind_cut])


corr_cutoff = 0.5e-2
t_energy = 0.1
n_carbon_atoms = 16
n_el = 6*n_carbon_atoms + 4
n_atoms = n_carbon_atoms + 4

plt.close("all")
df_corr = []
proposal_colors = {"single-electron": "C0", "single-electron-sweep": "navy", "all-electron": "C1", "cluster-update": "C2"}
for run_name in sorted(all_data.keys()):
    config, data = all_data[run_name]
    run_name = run_name.replace("C16H4_eval_batch16_", "")
    proposal = config["mcmc_args"]["proposal"]
    if "sweep" in run_name:
        proposal += "-sweep"
    cluster_radius = np.nan
    if proposal == "cluster-update":
        steps = n_atoms * config["mcmc_args"]["cluster_update_args"]["sweeps"]
        cluster_radius = config["mcmc_args"]["cluster_update_args"]["cluster_radius"]
    elif proposal == "all-electron":
        steps = config["mcmc_args"]["all_electron_args"]["steps"]
    elif proposal in ["single-electron", "single-electron-sweep"]:
        steps = n_el * config["mcmc_args"]["single_electron_args"]["sweeps"]

    color = proposal_colors[proposal]
    if "eval/E" not in data.columns:
        continue
    energy = data["eval/E"].values
    if len(energy) < 1000:
        continue

    corr = get_autocorrelation(energy)
    ind_cut = get_ind_cut(corr, corr_cutoff)
    tau = get_integrated_corr_time(corr, corr_cutoff)
    t_step = np.median(data["eval/t_step"])
    t_sampling = t_step - t_energy
    df_corr.append(dict(run_name=run_name, tau=tau, cluster_radius=cluster_radius, proposal=proposal, steps=steps))


    print(f"{run_name:<20}: tau={tau:4.1f}, t={t_step:4.2f} sec, t_s={t_sampling:4.2f} sec, tau*t_s={tau*t_sampling:4.2f} sec")
    fig, ax = plt.subplots(1,1)
    ax.semilogy(corr, label=run_name, color=color)
    ax.set_ylim([corr_cutoff, 1.1])
    ax.set_xlim([0, 300])
    ax.fill_between(np.arange(ind_cut), np.ones(ind_cut)*corr_cutoff, corr[:ind_cut], color=color, alpha=0.2)
    ax.set_title(run_name)
    ax.text(100, 0.3, f"$\\tau$={tau:4.1f}", color=color, fontsize=20)
    ax.set_xlabel("MCMC steps")
    ax.set_ylabel("Correlation coeff.")
    fig.savefig(f"ac_{run_name}.png", dpi=100)

df_corr = pd.DataFrame(df_corr)
df_corr["cluster_radius"] = df_corr["cluster_radius"].fillna(0)
df_corr = df_corr.sort_values(["proposal", "cluster_radius", "steps"])
fig, axes = plt.subplots(1,2, figsize=(9, 4))
for prop in df_corr.proposal.unique():
    df_filt = df_corr[df_corr.proposal == prop]
    color = proposal_colors[prop]
    for cluster_radius in df_filt.cluster_radius.unique():
        df_plot = df_filt[df_filt.cluster_radius == cluster_radius]
        if prop == "cluster-update":
            ls = {2.0: '-', 4.0: '--', 6.0: ':'}[cluster_radius]
            label = f"{prop} {cluster_radius}"
        else:
            ls = '-'
            label = prop
        axes[0].plot(df_plot.steps, df_plot.tau, label=label, color=color, ls=ls, marker="o")
        axes[1].plot(df_plot.steps, df_plot.tau * df_plot.steps, label=label, color=color, ls=ls, marker="o")
axes[0].legend(loc="upper right")
for ax in axes:
    ax.set_xlabel("MCMC inter-steps")
    ax.set_ylabel("Autocorrelation time / opt_steps")
    ax.set_ylabel("Autocorrelation time / single MCMC update")
    # ax.set_ylim([0, None])
    ax.set_yscale("log")
    ax.grid(alpha=0.5)
fig.suptitle("Autcorrelation times, C16H4")
fig.tight_layout()
fig.savefig("correlation_summary.png", dpi=400)











