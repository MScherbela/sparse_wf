from sparse_wf.api import LoggingArgs, WandBArgs, FileLoggingArgs, Logger
import atexit
import wandb

class FileLogger(Logger):
    def __init__(self, path: str) -> None:
        self.path = path
        self.file = open(path, "w")
        atexit.register(self.file.close)

    def log(self, data: dict) -> None:
        self.file.write(str(data) + "\n")

    def log_config(self, config: dict) -> None:
        self.file.write(str(config) + "\n")

class WandBLogger(Logger):
    def __init__(self, project: str, entity: str) -> None:
        wandb.init(project=project, entity=entity)

    def log(self, data: dict) -> None:
        wandb.log(data)

    def log_config(self, config: dict) -> None:
        wandb.config.update(config)

class MultiLogger(Logger):
    def __init__(self, logging_args: LoggingArgs) -> None:
        self.loggers = []
        if ("wandb" in logging_args) and (logging_args["wandb"]["use"]):
            wandb_args = {k: v for k, v in logging_args["wandb"].items() if k != "use"}
            self.loggers.append(WandBLogger(**wandb_args)) # type: ignore
        if ("file" in logging_args) and (logging_args["file"]["use"]):
            file_args = {k: v for k, v in logging_args["file"].items() if k != "use"}
            self.loggers.append(FileLogger(**file_args)) # type: ignore

    def log(self, data: dict) -> None:
        for logger in self.loggers:
            logger.log(data)

    def log_config(self, config: dict) -> None:
        for logger in self.loggers:
            logger.log_config(config)