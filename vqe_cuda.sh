#!/bin/bash              
#SBATCH -A m4916         # e.g., m4916
#SBATCH -C gpu
#SBATCH -q regular                 # or debug for short tests
#SBATCH -t 04:00:00
#SBATCH -N 4                   # 2 nodes
#SBATCH --ntasks-per-node=1        # 1 rank per node
#SBATCH --gpus-per-task=4          # 4 GPUs per rank (so 8 total)
#SBATCH -o slurm-%j_%t.out

module load cray-mpich
module load conda
conda activate CUDAQ

# Launch
srun python vqe_cuda.py --cudaq-full-stack-trace
