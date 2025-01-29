# Build a simple CLI tool, which
# - takes a checkpoint file path as input
# - as has as cli inputs cutoff_final, cutoff_stepsize=0.5, opt_steps=2000
# - finds the config.yaml located in the same directory
# - sets up multiple jobs using setup_calculations()
# - For each job use the existing config.yaml, but modify it it to have the new cutoff, new checkpoint from previous calculation, and extend the nr of steps in optimization.steps accordingly
# - Specify the SLURM dependencies approrpiately
# AI!
