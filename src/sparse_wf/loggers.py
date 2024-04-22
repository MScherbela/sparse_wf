from sparse_wf.api import LoggingArgs, Logger
import atexit
import wandb
import numpy as np
from sparse_wf.jax_utils import only_on_main_process


class FileLogger(Logger):
    @only_on_main_process
    def __init__(self, path: str) -> None:
        self.path = path
        self.file = open(path, "w")
        atexit.register(self.file.close)

    @only_on_main_process
    def log(self, data: dict) -> None:
        self.file.write(str(data) + "\n")

    @only_on_main_process
    def log_config(self, config: dict) -> None:
        self.file.write(str(config) + "\n")


class WandBLogger(Logger):
    @only_on_main_process
    def __init__(self, project: str, entity: str, name: str = None) -> None:
        wandb.init(project=project, entity=entity, name=name)
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
        self.loggers = []
        self.smoothing_history: dict[str, np.ndarray] = {}
        self.smoothing_length = logging_args["smoothing"]

        if ("wandb" in logging_args) and (logging_args["wandb"]["use"]):
            wandb_args = {k: v for k, v in logging_args["wandb"].items() if k != "use"}
            self.loggers.append(WandBLogger(**wandb_args))  # type: ignore
        if ("file" in logging_args) and (logging_args["file"]["use"]):
            file_args = {k: v for k, v in logging_args["file"].items() if k != "use"}
            self.loggers.append(FileLogger(**file_args))  # type: ignore

    def smoothen_data(self, data: dict) -> dict:
        # This implementation is a bit ugly, but does the job for now
        smoothed_data = {}
        for key, val in data.items():
            if key not in self.METRICS_TO_SMOOTH:
                continue
            if key not in self.smoothing_history:
                self.smoothing_history[key] = np.ones(self.smoothing_length) * np.nan
            self.smoothing_history[key] = np.roll(self.smoothing_history[key], 1)
            self.smoothing_history[key][0] = val
            smoothed_data[key + "_smooth"] = np.nanmean(self.smoothing_history[key])
        data.update(smoothed_data)
        return data

    def log(self, data: dict) -> None:
        data = self.smoothen_data(data)
        for logger in self.loggers:
            logger.log(data)

    def log_config(self, config: dict) -> None:
        for logger in self.loggers:
            logger.log_config(config)
