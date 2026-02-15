# %%
import pathlib
from collections import defaultdict
import pandas as pd

INPUT_DIR = "/home/scherbelam20/runs/lapnet_scaling/full_code"
KEYS_TO_EXTRACT = ["t_opt", "t_burnin"]


def get_hyperparams(dirname: str):
    tokens = dirname.split("_")
    return dict(model=tokens[0], n_carbon=int(tokens[1]), batch_size=int(tokens[2]))


def get_runtimes(log_fname):
    with open(log_fname, "r") as f:
        lines = f.readlines()

    runtimes = defaultdict(list)
    for line in lines:
        for key in KEYS_TO_EXTRACT:
            if key in line:
                value_string = line.split(f"{key}=")[-1]
                runtimes[key].append(float(value_string))

    min_runtimes = {}
    for key in KEYS_TO_EXTRACT:
        if key in runtimes:
            min_runtimes[key] = min(runtimes[key])
        else:
            min_runtimes[key] = None
    return min_runtimes


all_data = []
run_dirs = pathlib.Path(INPUT_DIR).glob("*/")
for run_dir in run_dirs:
    hyperparams = get_hyperparams(run_dir.stem)
    runtimes = get_runtimes(run_dir / "stdout.txt")
    all_data.append(hyperparams | runtimes)

df = (
    pd.DataFrame(all_data)
    .sort_values(by=["n_carbon", "batch_size"])
    .rename(columns={"t_opt": "opt/t_step", "t_burnin": "burnin/t_step"})
)
df.to_csv("data/full_run_data_lapnet.csv", index=False)

# %%
