#!/bin/bash
#SBATCH -J scaling
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task 8
#SBATCH --partition hgx
#SBATCH --qos normal
#SBATCH --output stdout.txt
#SBATCH --time 1440
#SBATCH --gres=gpu:1
#SBATCH --mem=100000

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate sparse_wf
export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10
export NVIDIA_TF32_OVERRIDE=0
export WANDB_DIR="${HOME}/tmp"
srun python main.py
