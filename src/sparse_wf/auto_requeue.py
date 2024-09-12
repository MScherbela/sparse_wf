import signal
from pathlib import Path
import shutil
import yaml
import subprocess
from sparse_wf.jax_utils import only_on_main_process
import re
import os
import logging

global SPARSEWF_ABORT_CALCULATION
SPARSEWF_ABORT_CALCULATION = False


def signal_handler(signum, frame):
    global SPARSEWF_ABORT_CALCULATION
    SPARSEWF_ABORT_CALCULATION = True


def register_signal_handler():
    old_handler = signal.getsignal(signal.SIGUSR1)
    if old_handler == signal.SIG_IGN:
        logging.warning("SIGUSR1 was ignored")
    if old_handler == signal.SIG_DFL:
        logging.warning("SIGUSR1 was set to default")
    if old_handler is None:
        logging.warning("SIGUSR1 was not set")
    else:
        logging.warning(f"Existing handler for SIGUSR1: {str(old_handler)}")

    signal.signal(signal.SIGUSR1, signal_handler)
    logging.info("Registered signal handler for SIGUSR1")


def requeue_and_exit(opt_step, chkpt_fname):
    with only_on_main_process():
        # Create new run directory
        chkpt_fname = Path(chkpt_fname)
        run_dir = chkpt_fname.resolve().parent
        old_run_dir_name = re.sub(r"_from\d{6}\d*$", "", run_dir.name)
        new_run_dir_name = old_run_dir_name + f"_from{opt_step:06d}"
        new_run_dir = run_dir / ".." / new_run_dir_name
        new_run_dir.mkdir()

        # Copy the full config with minor modifications
        with open("full_config.yaml", "r") as f:
            full_config = yaml.safe_load(f)
        full_config["load_checkpoint"] = str(chkpt_fname.resolve())
        full_config["pretraining"]["steps"] = 0
        full_config["optimization"]["burn_in"] = 0
        full_config["auto_requeue"] = max(0, full_config["auto_requeue"] - 1)
        with open(new_run_dir / "full_config.yaml", "w") as f:
            yaml.safe_dump(full_config, f)

        # Copy the jobfile and submit the job
        shutil.copy("job.sh", new_run_dir / "job.sh")
        env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
        subprocess.call(["sbatch", "job.sh"], cwd=new_run_dir, env=env)
    raise SystemExit(0)
