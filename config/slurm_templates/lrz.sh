#!/bin/bash
#SBATCH -J {job_name}
#SBATCH -N 1
#SBATCH -n 1
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

source $HOME/repos/sparse_wf/.venv/bin/activate
export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10
export NVIDIA_TF32_OVERRIDE=0
#export WANDB_MODE=offline

srun uv run sparse-wf-run full_config.yaml &
wait
