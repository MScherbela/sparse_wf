#!/bin/bash
#SBATCH -J test
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task 1
#SBATCH --partition cpu_all
#SBATCH --qos=default
#SBATCH --output stdout.txt
#SBATCH --time 00:00:05
#SBATCH --gres=gpu:0
#SBATCH --mem=16M
#SBATCH --signal=B:USR1@300

trap 'touch SPARSEWF_ABORT && wait' SIGUSR1
sleep 30