from pathlib import Path
import shutil
import yaml
import subprocess
from sparse_wf.jax_utils import only_on_main_process
import re
import os


def should_abort():
    return Path("SPARSEWF_ABORT").is_file()


def requeue_job(opt_step, chkpt_fname):
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
        full_config["optimization"]["burn_in"] = 50  # hack to avoid storing mcmc_stepsize and statics
        full_config["auto_requeue"] = max(0, full_config["auto_requeue"] - 1)
        with open(new_run_dir / "full_config.yaml", "w") as f:
            yaml.safe_dump(full_config, f)

        # Copy the jobfile and submit the job
        shutil.copy("job.sh", new_run_dir / "job.sh")
        env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
        subprocess.call(["sbatch", "job.sh"], cwd=new_run_dir, env=env)
