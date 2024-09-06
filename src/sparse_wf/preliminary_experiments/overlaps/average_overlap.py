#%%
import numpy as np
from ast import literal_eval
import pandas as pd
import matplotlib.pyplot as plt

# fname = "/storage/scherbelam20/runs/sparse_wf/overlaps/eval_bad_3k/log.txt"
# fname = "/storage/scherbelam20/runs/sparse_wf/overlaps/eval_badV2_2k/log.txt"
# fname = "/storage/scherbelam20/runs/sparse_wf/overlaps/eval_good_5k/log.txt"
fname = "/storage/scherbelam20/runs/sparse_wf/overlaps/ovlp_C2H4_90deg/log.txt"

def get_overlaps(fname):
    data = []
    with open(fname, "r") as f:
        for line in f:
            if "eval/" in line:
                eval_data = literal_eval(line.strip())
                eval_data = {k: v for k, v in eval_data.items() if "eval/overlap_" in k}
                data.append(eval_data)
    df = pd.DataFrame(data)
    df_summary = df.agg(["mean", "std"]).transpose()
    n_steps = len(df)

    normalization = np.sqrt((df_summary["mean"]**2).sum())
    df_summary = df_summary / normalization
    std_err = df_summary["std"] / np.sqrt(n_steps)
    return df_summary["mean"].values, std_err, n_steps



n_opt_steps = [1, 2, 3, 4]

plt.close("all")
fig, axes = plt.subplots(1, 3, figsize=(12, 5))

fname_templates = [
    "/storage/scherbelam20/runs/sparse_wf/overlaps/eval_C2H4_90deg_groundstate_{n_steps}k/log.txt",
    "/storage/scherbelam20/runs/sparse_wf/overlaps/eval_C2H4_90deg_excited_{n_steps}k/log.txt",
    "/storage/scherbelam20/runs/sparse_wf/overlaps/eval_C2H4_90deg_transition_{n_steps}k/log.txt",
]

state_labels = ["HH (HF)", "HL", "LH", "LL"]

for fname_template, ax in zip(fname_templates, axes):
    means = []
    std_errs = []
    for i in n_opt_steps:
        mean, std_err, n_steps = get_overlaps(fname_template.format(n_steps=i))
        means.append(mean)
        std_errs.append(std_err)
    means = np.array(means)
    std_errs = np.array(std_errs)


    for state in range(4):
        ax.plot(n_opt_steps, means[:, state]**2, marker='o', label=f"{state}: {state_labels[state]}")
    ax.plot(n_opt_steps, (means[:, 4:]**2).sum(axis=-1), marker='o', label="Others", color="gray")
    ax.axhline(0, color="black", linestyle="-")
    ax.legend()
    ax.set_xlabel("Opt steps / k")
    ax.set_ylim([0, 1.1])
    ax.grid(alpha=0.2)
    ax.set_ylabel("Overlap")
axes[0].set_title("Ground state")
axes[1].set_title("First excited state?")
axes[2].set_title("Transition during opt")
fig.suptitle("C2H4, CI coefficients")
fig.tight_layout()
fig.savefig("overlaps_vs_opt_steps.png")








# data = []
# with open(fname, "r") as f:
#     for line in f:
#         if "eval/" in line:
#             eval_data = literal_eval(line.strip())
#             eval_data = {k: v for k, v in eval_data.items() if "eval/overlap_" in k}
#             data.append(eval_data)
# df = pd.DataFrame(data)
# df_summary = df.agg(["mean", "std"]).transpose()

# mean_max = df_summary["mean"].abs().max()
# df_summary = df_summary / mean_max
# print(f"{len(df)} eval steps; max overlap {mean_max:.2e}")
# print(df_summary)