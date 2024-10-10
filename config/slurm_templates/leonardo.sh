#!/bin/bash
#SBATCH -J {job_name}
#SBATCH -N {(n_gpus-1)//4 + 1}
#SBATCH -p boost_usr_prod
#SBATCH --qos normal
#SBATCH -A L-AUT_Sch-Hoef
#SBATCH --output stdout.txt
#SBATCH --time {time}
#SBATCH --gres=gpu:{min(n_gpus, 4)}
#SBATCH --ntasks-per-node={min(n_gpus, 4)}
#SBATCH --signal=B:USR1@300
#SBATCH --export=NONE
SLURM_EXPORT_ENV=ALL
# The export=None ensures that no environment variables from submission are inherited by the job
# The SLURM_EXPORT_ENV=ALL ensures that all environment variables present in this file
# (and the ones set by SLURM) are exported to the srun tasks

trap 'touch SPARSEWF_ABORT && wait' SIGUSR1

source $HOME/develop/sparse_wf/.venv/bin/activate
export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10
export NVIDIA_TF32_OVERRIDE=0
export WANDB_MODE=offline

srun sparse-wf-run full_config.yaml &
wait