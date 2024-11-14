#!/bin/bash
#SBATCH -J {job_name}
#SBATCH -N 1
#SBATCH -n {n_gpus}
#SBATCH --cpus-per-task 8
#SBATCH --partition {partition}
#SBATCH --qos=mcml
#SBATCH --output stdout.txt
#SBATCH --time {time}
#SBATCH --gres=gpu:{n_gpus}
#SBATCH --mem={n_gpus*100_000}
#SBATCH --signal=B:USR1@300

trap 'touch SPARSEWF_ABORT && wait' SIGUSR1

# if $HOME/develop/sparse_wf/.venv exists, activate it, else fall back to conda
if [ -d $HOME/develop/sparse_wf/.venv ]; then
    source $HOME/repos/sparse_wf/.venv/bin/activate
fi
export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10
export NVIDIA_TF32_OVERRIDE=0
export WANDB_DIR="${{HOME}}/tmp"
srun sparse-wf-run full_config.yaml &
wait
