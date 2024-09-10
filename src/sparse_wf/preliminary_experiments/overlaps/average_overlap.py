#%%
import numpy as np
from ast import literal_eval
import pandas as pd
import matplotlib.pyplot as plt

def get_overlaps(fname):
    data = []
    with open(fname, "r") as f:
        for line in f:
            if "eval/" in line:
                eval_data = literal_eval(line.strip())
                eval_data = {k: v for k, v in eval_data.items() if "overlap" in k}
                data.append(eval_data)
    df = pd.DataFrame(data)
    n_steps = len(df)

    if "eval/overlap_0" in list(df):
        # Work in lin-space
        df_summary = df.agg(["mean", "std"]).transpose()
        normalization = np.sqrt((df_summary["mean"]**2).sum())
        df_summary = df_summary / normalization
        std_err = df_summary["std"] / np.sqrt(n_steps)
        overlap_mean = df_summary["mean"].values
        overlap_stderr = std_err.values
    else:
        assert "eval/log_overlap_0" in list(df)
        # Work in log-space
        n_states = len(list(df)) // 2
        logs = df[[f"eval/log_overlap_{i}" for i in range(n_states)]].values
        signs = df[[f"eval/sign_overlap_{i}" for i in range(n_states)]].values
        logs -= np.max(logs)
        overlaps = np.exp(logs) * signs
        overlap_mean = overlaps.mean(axis=0)
        overlap_stderr = overlaps.std(axis=0) / np.sqrt(n_steps)
        normalization = np.sqrt(np.sum(overlap_mean**2))
        overlap_mean /= normalization
        overlap_stderr /= normalization
    return overlap_mean, overlap_stderr, n_steps

overlap_data = []
for n_carbon in [4, 6, 8]:
    for twist in [0, 90]:
        for cutoff in [3, 5, 7]:
            log_fname = f"/storage/scherbelam20/runs/sparse_wf/overlaps/eval_C{n_carbon}H4_{twist}deg_{cutoff:.1f}/log.txt"
            overlap_mean, overlap_stderr, n_steps = get_overlaps(log_fname)
            overlaps = {f"overlap_{i}" : overlap_mean[i] for i in range(len(overlap_mean))}
            overlap_data.append(dict(n_carbon=n_carbon, twist=twist, cutoff=cutoff, **overlaps))
df = pd.DataFrame(overlap_data)

#%%
state_labels = ["HH (HF)", "HL", "LH", "LL"]
fig, axes = plt.subplots(3, 2, figsize=(10, 6))
for ind_cutoff, cutoff in enumerate([3, 5, 7]):
    for ind_twist, twist in enumerate([0, 90]):
        ax = axes[ind_cutoff, ind_twist]
        df_plot = df[(df.cutoff == cutoff) & (df.twist == twist)]
        for ind_state in range(4):
            ax.plot(df_plot.n_carbon, df_plot[f"overlap_{ind_state}"]**2, marker='o', label=f"{ind_state}: {state_labels[ind_state]}")
        ax.grid(alpha=0.5)
        ax.axhline(0, color="black", linestyle="-")
        ax.set_ylim([0, 1.1])
        ax.set_xlabel("n_carbon")
        if ind_twist == 0:
            ax.set_ylabel("cutoff = {:.1f}".format(cutoff))
            ax.legend(loc="center left")
axes[0, 0].set_title("0 deg")
axes[0, 1].set_title("90 deg")

fname = "/home/scherbelam20/develop/sparse_wf/src/sparse_wf/preliminary_experiments/overlaps/overlaps_vs_n_carbon.png"
fig.suptitle("Cumulene, CI weights, 50k steps")
fig.tight_layout()
fig.savefig(fname, dpi=400)




# n_opt_steps = [1, 2, 3, 4]

# plt.close("all")
# fig, axes = plt.subplots(1, 3, figsize=(12, 5))

# fname_templates = [
#     "/storage/scherbelam20/runs/sparse_wf/overlaps/eval_C2H4_90deg_groundstate_{n_steps}k/log.txt",
#     "/storage/scherbelam20/runs/sparse_wf/overlaps/eval_C2H4_90deg_excited_{n_steps}k/log.txt",
#     "/storage/scherbelam20/runs/sparse_wf/overlaps/eval_C2H4_90deg_transition_{n_steps}k/log.txt",
# ]

# state_labels = ["HH (HF)", "HL", "LH", "LL"]


# for fname_template, ax in zip(fname_templates, axes):
#     means = []
#     std_errs = []
#     for i in n_opt_steps:
#         mean, std_err, n_steps = get_overlaps(fname_template.format(n_steps=i))
#         means.append(mean)
#         std_errs.append(std_err)
#     means = np.array(means)
#     std_errs = np.array(std_errs)


#     for state in range(4):
#         ax.plot(n_opt_steps, means[:, state]**2, marker='o', label=f"{state}: {state_labels[state]}")
#     ax.plot(n_opt_steps, (means[:, 4:]**2).sum(axis=-1), marker='o', label="Others", color="gray")

# # Plot C8 overlaps
# fname = "/storage/scherbelam20/runs/sparse_wf/overlaps/eval_C8H4_90deg/log.txt"
# overlap_mean, overlap_stderr, n_steps = get_overlaps(fname)
# for i, (mean, err) in enumerate(zip(overlap_mean, overlap_stderr)):
#     if i >= 4:
#         break
#     print(f"{i}: {mean:-.4f} +- {err:.4f}")
#     axes[0].axhline(mean**2, color=f"C{i}", linestyle="--", label=f"{i} (C8)")
# axes[0].axhline((overlap_mean[4:]**2).sum(), color="gray", linestyle="--", label="Others (C8)")


# for ax in axes:
#     ax.axhline(0, color="black", linestyle="-")
#     ax.legend()
#     ax.set_xlabel("Opt steps / k")
#     ax.set_ylim([0, 1.1])
#     ax.grid(alpha=0.2)
#     ax.set_ylabel("Overlap")



# axes[0].set_title("Ground state")
# axes[1].set_title("First excited state?")
# axes[2].set_title("Transition during opt")
# fig.suptitle("C2H4, CI coefficients")
# fig.tight_layout()
# fig.savefig("overlaps_vs_opt_steps.png")



# #%%









# # data = []
# # with open(fname, "r") as f:
# #     for line in f:
# #         if "eval/" in line:
# #             eval_data = literal_eval(line.strip())
# #             eval_data = {k: v for k, v in eval_data.items() if "eval/overlap_" in k}
# #             data.append(eval_data)
# # df = pd.DataFrame(data)
# # df_summary = df.agg(["mean", "std"]).transpose()

# # mean_max = df_summary["mean"].abs().max()
# # df_summary = df_summary / mean_max
# # print(f"{len(df)} eval steps; max overlap {mean_max:.2e}")
# # print(df_summary)