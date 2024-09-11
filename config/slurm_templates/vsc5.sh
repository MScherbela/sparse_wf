#!/bin/bash
#SBATCH -J {job_name}
#SBATCH -N {(n_gpus-1)//2 + 1}
#SBATCH --partition {partition}
#SBATCH --qos {qos}
#SBATCH --output stdout.txt
#SBATCH --time {time}
#SBATCH --gres=gpu:{min(n_gpus, 2)}
#SBATCH --ntasks-per-node={min(n_gpus, 2)}
#SBATCH --signal=USR1@300
#SBATCH --export=NONE
SLURM_EXPORT_ENV=ALL
# The export=None ensures that no environment variables from submission are inherited by the job
# The SLURM_EXPORT_ENV=ALL ensures that all environment variables present in this file
# (and the ones set by SLURM) are exported to the srun tasks


module purge
source /gpfs/opt/sw/spack-0.17.1/opt/spack/linux-almalinux8-zen3/gcc-11.2.0/miniconda3-4.12.0-ap65vga66z2rvfcfmbqopba6y543nnws/etc/profile.d/conda.sh
conda activate sparse_wf
export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10
export NVIDIA_TF32_OVERRIDE=0
export WANDB_DIR="${{HOME}}/tmp"
srun sparse-wf-run full_config.yaml