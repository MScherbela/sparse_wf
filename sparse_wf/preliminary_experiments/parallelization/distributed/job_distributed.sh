#!/bin/bash
#SBATCH -J test_distributed
#SBATCH -N 1
#SBATCH --partition hgx
#SBATCH --qos normal
#SBATCH --output stdout_test_distributed.txt
#SBATCH --time 5
#SBATCH --tasks-per-node=2
#SBATCH --gpus-per-task=1

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate sparse_wf
srun python test_distributed.py