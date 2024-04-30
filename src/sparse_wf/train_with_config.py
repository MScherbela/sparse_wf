import yaml
from sparse_wf.train import main
import argparse
import os


def train_with_config():
    parser = argparse.ArgumentParser(description="Train a model from a config file")
    parser.add_argument("config", type=str, help="Path to the config file")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.CLoader)

    config["logging_args"]["name"] = os.getcwd().split("/")[-1]
    main(**config)


if __name__ == "__main__":
    train_with_config()
