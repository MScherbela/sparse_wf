import atexit
import os
from typing import Any

import numpy as np

import wandb
from sparse_wf.api import Logger, LoggingArgs
from sparse_wf.jax_utils import only_on_main_process


class FileLogger(Logger):
    @only_on_main_process
    def __init__(self, file_name: str, collection: str, out_directory: str, name: str, comment: str, **_) -> None:
        # TODO: fix this for seml
        # if collection:
        #     self.path = os.path.join(out_directory, collection, name, file_name)
        # else:
        #     self.path = os.path.join(out_directory, name, file_name)
        self.path = file_name
        self.file = open(self.path, "w")
        atexit.register(self.file.close)
        if comment:
            self.log(comment)

    @only_on_main_process
    def log(self, data: Any) -> None:
        self.file.write(str(data) + "\n")

    @only_on_main_process
    def log_config(self, config: dict) -> None:
        self.file.write(str(config) + "\n")


class WandBLogger(Logger):
    @only_on_main_process
    def __init__(self, project: str, entity: str, name: str, comment: str, **_) -> None:
        wandb.init(project=project, entity=entity, name=name, notes=comment)
        atexit.register(wandb.finish)

    @only_on_main_process
    def log(self, data: dict) -> None:
        wandb.log(data)

    @only_on_main_process
    def log_config(self, config: dict) -> None:
        wandb.config.update(config)


class MultiLogger(Logger):
    METRICS_TO_SMOOTH = ["opt/E"]

    def __init__(self, logging_args: LoggingArgs) -> None:
        self.loggers: list[Logger] = []
        self.smoothing_history: dict[str, np.ndarray] = {}
        self.smoothing_length = logging_args["smoothing"]
        self.args = logging_args

        # TODO: fix this for seml
        # with only_on_main_process():
        #     if self.run_directory != ".":
        #         os.makedirs(self.run_directory, exist_ok=False)

        if ("wandb" in logging_args) and (logging_args["wandb"]["use"]):
            self.loggers.append(WandBLogger(**(logging_args | logging_args["wandb"])))  # type: ignore
        if ("file" in logging_args) and (logging_args["file"]["use"]):
            self.loggers.append(FileLogger(**(logging_args | logging_args["file"])))  # type: ignore

    # TODO: This enforces that the run directory always ends with the name of the run and does not support setting the cwd as run_directory
    @property
    def run_directory(self):
        if self.args.get("collection", None):
            return os.path.join(self.args["out_directory"], self.args["collection"], self.args["name"])
        return os.path.join(self.args["out_directory"], self.args["name"])

    def smoothen_data(self, data: dict) -> dict:
        # This implementation is a bit ugly, but does the job for now
        smoothed_data = {}

        step = data.get("opt_step")
        if step is not None:
            smoothing_length = int(np.clip(step * 0.1, 1, self.smoothing_length))
        else:
            smoothing_length = self.smoothing_length

        for key, val in data.items():
            if key not in self.METRICS_TO_SMOOTH:
                continue
            if key not in self.smoothing_history:
                self.smoothing_history[key] = np.ones(self.smoothing_length) * np.nan
            self.smoothing_history[key] = np.roll(self.smoothing_history[key], 1)
            self.smoothing_history[key][0] = val
            smoothed_data[key + "_smooth"] = np.nanmean(self.smoothing_history[key][:smoothing_length])
        data.update(smoothed_data)
        return data

    def log(self, data: dict) -> None:
        data = self.smoothen_data(data)
        for logger in self.loggers:
            logger.log(data)

    def log_config(self, config: dict) -> None:
        for logger in self.loggers:
            logger.log_config(config)

    @only_on_main_process
    def store_blob(self, data: bytes, file_name: str):
        with open(os.path.join(self.run_directory, file_name), "wb") as f:
            f.write(data)
