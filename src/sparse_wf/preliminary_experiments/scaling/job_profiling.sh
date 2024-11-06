#!/bin/bash
#SBATCH -J swann_profiling
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task 8
#SBATCH --partition hgx
#SBATCH --qos=normal
#SBATCH --output stdout_profiling.txt
#SBATCH --time 20-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=100000

source $HOME/develop/sparse_wf/.venv/bin/activate
export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10
export NVIDIA_TF32_OVERRIDE=0
export WANDB_DIR="${HOME}/tmp"

PROGRAM="/home/scherbelam20/develop/sparse_wf/src/sparse_wf/preliminary_experiments/scaling/run_scaling_tests.py"
python $PROGRAM --system_size_min 50 --system_size_max 50 --system_size_steps 1 --profile -o /dev/null
