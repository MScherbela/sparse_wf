#!/bin/bash
#SBATCH -J scaling
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task 8
#SBATCH --partition hgx
#SBATCH --qos=normal
#SBATCH --output stdout_no_ecp.txt
#SBATCH --time 20-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=100000

source $HOME/develop/sparse_wf/.venv/bin/activate
export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10
export NVIDIA_TF32_OVERRIDE=0
export WANDB_DIR="${HOME}/tmp"

PROGRAM="/home/scherbelam20/develop/sparse_wf/src/sparse_wf/preliminary_experiments/scaling/run_scaling_tests.py"
python $PROGRAM -o timings_no_ecp.txt
