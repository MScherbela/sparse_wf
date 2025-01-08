#!/bin/bash
#SBATCH -J {job_name}
#SBATCH -N 1
#SBATCH -n {n_gpus}
#SBATCH --cpus-per-task 8
#SBATCH --partition {partition}
#SBATCH --qos=default
#SBATCH --output stdout.txt
#SBATCH --time {time}
#SBATCH --gres=gpu:{n_gpus}
#SBATCH --mem={n_gpus*100_000}
#SBATCH --export=NONE
SLURM_EXPORT_ENV=ALL

trap 'touch SPARSEWF_ABORT && wait' SIGUSR1

export NVIDIA_TF32_OVERRIDE=0

srun uv run sparse-wf-run full_config.yaml &
wait
