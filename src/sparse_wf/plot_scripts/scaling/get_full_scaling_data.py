# %%
import yaml
import pathlib
from collections import defaultdict
import pandas as pd

INPUT_DIR = "/home/scherbelam20/develop/sparse_wf/runs/scaling/full_code"
KEYS_TO_EXTRACT = ["burnin/t_step", "opt/t_step"]


def get_hyperparams(config_fname):
    with open(config_fname, "r") as f:
        config = yaml.safe_load(f)
    return dict(
        n_carbon=config["molecule_args"]["cumulene_args"]["n_carbon"],
        cutoff=config["model_args"]["embedding"]["new"]["cutoff"],
        batch_size=config["batch_size"],
        opt_batch_size=config["optimization"]["max_batch_size"],
    )


def get_runtimes(log_fname):
    with open(log_fname, "r") as f:
        lines = f.readlines()

    runtimes = defaultdict(list)
    for line in lines:
        for key in KEYS_TO_EXTRACT:
            if key in line:
                value_string = line.split(key + "': ")[-1].split(",")[0]
                runtimes[key].append(float(value_string))

    min_runtimes = {}
    for key in KEYS_TO_EXTRACT:
        if key in runtimes:
            min_runtimes[key] = min(runtimes[key])
        else:
            min_runtimes[key] = None
    return min_runtimes


all_data = []
run_dirs = pathlib.Path(INPUT_DIR).glob("FiRE*")
for run_dir in run_dirs:
    config_fname = run_dir / "config.yaml"
    log_fname = run_dir / "log.txt"
    if not config_fname.exists() or not log_fname.exists():
        continue
    hyperparams = get_hyperparams(config_fname)
    runtimes = get_runtimes(log_fname)
    all_data.append(hyperparams | runtimes)

df = pd.DataFrame(all_data)
df = df.sort_values(by=["n_carbon", "cutoff", "batch_size", "opt_batch_size"])
df["model"] = "FiRE"
df.to_csv("data/full_run_data.csv", index=False)

# %%
