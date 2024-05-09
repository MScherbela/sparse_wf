import yaml
from sparse_wf.train import main
import argparse
import os
import subprocess


def get_git_commit():
    try:
        path = os.path.dirname(__file__)
        msg = subprocess.check_output(["git", "log", "-1"], cwd=path, encoding="utf-8")
        return msg.replace("\n", "; ")
    except Exception as e:
        print(e)
        return ""


def train_with_config():
    parser = argparse.ArgumentParser(description="Train a model from a config file")
    parser.add_argument("config", type=str, help="Path to the config file")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.CLoader)

    config["logging_args"]["name"] = os.getcwd().split("/")[-1]
    metadata = dict(commit=get_git_commit(), cwd=os.getcwd())
    config["metadata"] = config.get("metadata", {}) | metadata
    main(**config)


if __name__ == "__main__":
    train_with_config()
