#!/bin/bash
#SBATCH -J test_sharding
#SBATCH -N 1
#SBATCH --partition hgx
#SBATCH --qos normal
#SBATCH --output stdout_test_sharding.txt
#SBATCH --time 5
#SBATCH --tasks-per-node=1

# Change the following line to gpus-per-task=0 to run on CPUs
#SBATCH --gpus-per-task=2

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate sparse_wf

export XLA_FLAGS=--xla_force_host_platform_device_count=2
python test_sharding.py