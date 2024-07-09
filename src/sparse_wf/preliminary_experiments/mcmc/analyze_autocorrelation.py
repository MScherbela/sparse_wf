#%%
import pandas as pd
from pathlib import Path
from ast import literal_eval
import numpy as np
import matplotlib.pyplot as plt

all_data = {}
logfiles = Path.glob(Path("/storage/scherbelam20/runs/sparse_wf/cluster_update"), "C6H4_eval_batch32*/log.txt")
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

def get_integrated_corr_time(corr, cutoff=1e-2):
    ind_cut = np.where(corr < cutoff)[0][0]
    return 1 + 2 * np.sum(corr[1:ind_cut])


corr_cutoff = 0.5e-2
t_energy = 0.1

plt.close("all")
for run_name, (config, data) in all_data.items():
    run_name = run_name.replace("C6H4_eval_batch32_", "")
    proposal = config["mcmc_args"]["proposal"]
    color = {"single-electron": "C0", "all-electron": "C1", "cluster-update": "C2"}[proposal]
    energy = data["eval/E"].values
    energy = energy[1000:]

    corr = get_autocorrelation(energy)
    tau = get_integrated_corr_time(corr, corr_cutoff)
    t_step = np.median(data["eval/t_step"])
    t_sampling = t_step - t_energy
    print(f"{run_name:<20}: tau={tau:4.1f}, t={t_step:4.2f} sec, t_s={t_sampling:4.2f} sec, tau*t_s={tau*t_sampling:4.2f} sec")
    fig, ax = plt.subplots(1,1)
    ax.semilogy(corr, label=run_name, color=color)
    ax.set_ylim([corr_cutoff, 1.1])
    ax.set_xlim([0, 300])
    ind_cut = np.where(corr < corr_cutoff)[0][0]
    ax.fill_between(np.arange(ind_cut), np.ones(ind_cut)*corr_cutoff, corr[:ind_cut], color=color, alpha=0.2)
    ax.set_title(run_name)
    ax.text(100, 0.3, f"$\\tau$={tau:4.1f}", color=color, fontsize=20)
    ax.set_xlabel("MCMC steps")
    ax.set_ylabel("Correlation coeff.")
    fig.savefig(f"ac_{run_name}.png", dpi=100)









