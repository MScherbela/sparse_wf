#!/bin/bash
#SBATCH -J {job_name}
#SBATCH --nodes {nodes}
#SBATCH --ntasks-per-node {ntasks_per_node}
#SBATCH --cpus-per-task {cpus_per_task}
#SBATCH --partition {partition}
#SBATCH --qos={qos}
#SBATCH --output stdout.txt
#SBATCH --time {time}
#SBATCH --gres=gpu:{n_gpus}
#SBATCH --mem={mem}
#SBATCH --signal=B:USR1@300
#SBATCH --export=NONE
SLURM_EXPORT_ENV=ALL

trap 'touch SPARSEWF_ABORT && wait' SIGUSR1

export NVIDIA_TF32_OVERRIDE=0
export JAX_DEFAULT_DTYPE_BITS=32

srun uv run sparse-wf-run full_config.yaml &
wait