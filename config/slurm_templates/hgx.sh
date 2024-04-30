#!/bin/bash
#SBATCH -J {job_name}
#SBATCH -N 1
#SBATCH -n {n_gpus}
#SBATCH --cpus-per-task 8
#SBATCH --partition hgx
#SBATCH --qos normal
#SBATCH --output stdout.txt
#SBATCH --time {time}
#SBATCH --gres=gpu:{n_gpus}
#SBATCH --mem=100000

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate sparse_wf
export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10
export NVIDIA_TF32_OVERRIDE=0
export WANDB_DIR="${{HOME}}/tmp"
srun sparse-wf-run full_config.yaml
