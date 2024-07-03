#!/bin/bash
#SBATCH -J scaling_benchmark
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task 8
#SBATCH --partition hgx
#SBATCH --qos=normal
#SBATCH --output stdout.txt
#SBATCH --time 6:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=100000

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate sparse_wf
export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10
export NVIDIA_TF32_OVERRIDE=0
export WANDB_DIR="${HOME}/tmp"
python /home/scherbelam20/develop/sparse_wf/src/sparse_wf/preliminary_experiments/sparse_mpnn/benchmark.py
