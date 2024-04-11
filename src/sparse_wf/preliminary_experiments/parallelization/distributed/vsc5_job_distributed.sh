#!/bin/bash
#SBATCH -J test_distributed
#SBATCH -N 2
#SBATCH --partition zen3_0512_a100x2
#SBATCH --qos zen3_0512_a100x2
#SBATCH --output stdout_test_distributed.txt
#SBATCH --time 5
#SBATCH --tasks-per-node=2
#SBATCH --gres=gpu:2

module purge
source /gpfs/opt/sw/spack-0.17.1/opt/spack/linux-almalinux8-zen3/gcc-11.2.0/miniconda3-4.12.0-ap65vga66z2rvfcfmbqopba6y543nnws/etc/profile.d/conda.sh
conda activate sparse_wf
srun python test_distributed.py
