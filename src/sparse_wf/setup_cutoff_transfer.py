#!/usr/bin/env python
import argparse
import pathlib
import yaml
import numpy as np
from sparse_wf.setup_calculations import setup_calculations


def load_yaml(fname):
    with open(fname, "r") as f:
        return yaml.load(f, Loader=yaml.CLoader)


def main():
    parser = argparse.ArgumentParser(description="Setup cutoff transfer calculations")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint file")
    parser.add_argument("--n-gpus", type=int, required=True, help="Number of GPUs to use")
    parser.add_argument("--cutoff-final", type=float, required=True, help="Final cutoff value")
    parser.add_argument("--cutoff-stepsize", type=float, default=0.5, help="Step size for cutoff increments")
    parser.add_argument("--opt-steps-per-cutoff", type=int, default=1000, help="Number of optimization steps per job")
    parser.add_argument("--burn-in", type=int, default=20, help="Number of burn-in steps")
    parser.add_argument("--experiment-name", type=str, default="cutoff_transfer", help="Experiment name")
    args = parser.parse_args()

    # Get checkpoint path and config
    checkpoint_path = pathlib.Path(args.checkpoint).resolve()
    config_path = checkpoint_path.parent / "full_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    # Load original config
    config = load_yaml(config_path)

    # Get current cutoff
    current_cutoff = config["model_args"]["embedding"]["new"]["cutoff"]
    # Infer current nr of opt-steps from checkpoint filename, eg. "optchpt005000.msgpk
    current_opt_steps = int(checkpoint_path.stem.split("optchkpt")[1])

    # Generate sequence of cutoffs
    n_transition_steps = int(np.ceil((args.cutoff_final - current_cutoff) / args.cutoff_stepsize))
    cutoffs = np.linspace(current_cutoff, args.cutoff_final, n_transition_steps + 1)[1:]
    opt_steps = current_opt_steps + np.arange(1, n_transition_steps + 1) * args.opt_steps_per_cutoff

    # Setup chain of jobs
    prev_job_ids = None
    for cutoff, opt_step in zip(cutoffs, opt_steps):
        # Update config for this step
        config["model_args"]["embedding"]["new"]["cutoff"] = float(cutoff)  # Convert from numpy float
        config["load_checkpoint"] = str(checkpoint_path)
        config["optimization"]["steps"] = opt_step
        config["optimization"]["burn_in"] = args.burn_in
        config["pretraining"]["steps"] = 0
        config["evaluation"]["steps"] = 0
        config["metadata"]["experiment_name"] = f"{args.experiment_name}_{cutoff:.3f}"
        config["slurm"] = dict(n_gpus=args.n_gpus)

        # Write temporary config
        tmp_config = pathlib.Path("tmp_config.yaml")
        with open(tmp_config, "w") as f:
            yaml.dump(config, f)

        # Setup calculation with dependency on previous job
        job_ids = setup_calculations(["-i", "tmp_config.yaml", "--no-commit-check"], depends_on=prev_job_ids)

        # Update for next iteration
        prev_job_ids = job_ids
        checkpoint_path = pathlib.Path(config["metadata"]["experiment_name"]) / f"optchkpt{opt_step:06d}.msgpk"
        checkpoint_path = checkpoint_path.resolve()

    # Cleanup
    tmp_config.unlink()


if __name__ == "__main__":
    main()
