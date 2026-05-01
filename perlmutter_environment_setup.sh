#!/usr/bin/env bash

# Run:
#   bash perlmutter_environment_setup.sh

set -e

module load conda
conda create -y -n QC-AFQMC_env python=3.12.11
conda activate QC-AFQMC_env

# Compile mpi4py against Perlmutter MPI
MPICC="cc -shared" pip install --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py==4.1.1

# Modified ipie fork used by QC-AFQMC
pip install "ipie @ git+https://github.com/rr637/ipie.git@develop"

pip install \
  pyscf==2.13.0 \
  openfermion==1.7.1 \
  qiskit-nature==0.7.2 \
  cudaq==0.14.2 \
  ffsim==0.0.79