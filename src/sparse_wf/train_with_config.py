import os

os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
os.environ["JAX_DEFAULT_DTYPE_BITS"] = "32"

import yaml
from sparse_wf.train import main
import argparse
import os
import subprocess
import logging


def get_git_commit():
    try:
        path = os.path.dirname(__file__)
        msg = subprocess.check_output(["git", "log", "-1"], cwd=path, encoding="utf-8")
        return msg.replace("\n", "; ")
    except Exception as e:
        print(e)
        return ""


def train_with_config(config_fname=None):
    if int(os.environ.get("SLURM_PROCID", 0)) == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler()],
        )
    if config_fname is None:
        parser = argparse.ArgumentParser(description="Train a model from a config file")
        parser.add_argument("config", type=str, help="Path to the config file")
        args = parser.parse_args()
        config_fname = args.config

    with open(config_fname, "r") as f:
        config = yaml.load(f, Loader=yaml.CLoader)

    config["logging_args"]["name"] = os.getcwd().split("/")[-1]
    metadata = dict(commit=get_git_commit(), cwd=os.getcwd())
    config["metadata"] = config.get("metadata", {}) | metadata
    main(**config)


if __name__ == "__main__":
    train_with_config()
