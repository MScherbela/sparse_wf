#!/bin/bash
#SBATCH -J {job_name}
#SBATCH -N {n_nodes}
#SBATCH -p boost_usr_prod
#SBATCH --qos normal
#SBATCH -A l-aut_005
#SBATCH --output stdout.txt
#SBATCH --time {time}
#SBATCH --gres=gpu:4

source $HOME/venv/sparse_wf/bin/activate
export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10
export NVIDIA_TF32_OVERRIDE=0
export WANDB_MODE=offline

srun sparse-wf-run full_config.yaml