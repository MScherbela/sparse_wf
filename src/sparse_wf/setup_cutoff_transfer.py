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
    parser.add_argument("--cutoff-final", type=float, required=True, help="Final cutoff value")
    parser.add_argument("--cutoff-stepsize", type=float, default=0.5, help="Step size for cutoff increments")
    parser.add_argument("--opt-steps", type=int, default=2000, help="Number of optimization steps per job")
    args = parser.parse_args()

    # Get checkpoint path and config
    checkpoint_path = pathlib.Path(args.checkpoint)
    config_path = checkpoint_path.parent / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    # Load original config
    config = load_yaml(config_path)
    
    # Get current cutoff
    current_cutoff = config.get("model", {}).get("cutoff", 0.0)
    
    # Generate sequence of cutoffs
    cutoffs = np.arange(current_cutoff + args.cutoff_stepsize, 
                       args.cutoff_final + args.cutoff_stepsize/2, 
                       args.cutoff_stepsize)

    # Setup chain of jobs
    prev_job_ids = None
    for cutoff in cutoffs:
        # Update config for this step
        config["model"]["cutoff"] = float(cutoff)  # Convert from numpy float
        config["checkpoint"] = str(checkpoint_path)
        config["optimization"]["steps"] = args.opt_steps
        
        # Write temporary config
        tmp_config = pathlib.Path("tmp_config.yaml")
        with open(tmp_config, "w") as f:
            yaml.dump(config, f)
            
        # Setup calculation with dependency on previous job
        job_ids = setup_calculations(depends_on=prev_job_ids)
        
        # Update for next iteration
        prev_job_ids = job_ids
        checkpoint_path = pathlib.Path(f"exp_{cutoff}/checkpoint.npz")

    # Cleanup
    tmp_config.unlink()

if __name__ == "__main__":
    main()
