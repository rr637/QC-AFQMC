#!/bin/bash
#SBATCH -A m4916
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -o afqmc_output-%A_%a.out
#SBATCH -J AFQMC

module load cray-mpich
module load conda
conda activate CUDAQ


srun python afqmc.py