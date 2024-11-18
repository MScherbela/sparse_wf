#!/bin/bash
#SBATCH -J {job_name}
#SBATCH -N 1
#SBATCH --gpus-per-task={n_gpus}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task 16
#SBATCH --partition hgx
#SBATCH --qos={qos}
#SBATCH --output stdout.txt
#SBATCH --time {time}
#SBATCH --mem={n_gpus*100_000}
#SBATCH --signal=B:USR1@300
#SBATCH --export=NONE
SLURM_EXPORT_ENV=ALL
# The export=None ensures that no environment variables from submission are inherited by the job
# The SLURM_EXPORT_ENV=ALL ensures that all environment variables present in this file
# (and the ones set by SLURM) are exported to the srun tasks

trap 'touch SPARSEWF_ABORT && wait' SIGUSR1

# if $HOME/develop/sparse_wf/.venv exists, activate it, else fall back to conda
if [ -d $HOME/develop/sparse_wf/.venv ]; then
    source $HOME/develop/sparse_wf/.venv/bin/activate
else
    source /opt/anaconda3/etc/profile.d/conda.sh
    conda activate sparse_wf
fi
export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10
export NVIDIA_TF32_OVERRIDE=0
export WANDB_DIR="${{HOME}}/tmp"
srun sparse-wf-run full_config.yaml &
wait
