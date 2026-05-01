#!/usr/bin/env bash

# Run:
#   bash perlmutter_environment_setup.sh

set -e

module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"

conda create -y -n QC-AFQMC_env python=3.12.11
conda activate QC-AFQMC_env

python -m pip install --upgrade pip

# Compile mpi4py against Perlmutter MPI
MPICC="cc -shared" pip install --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py==4.1.1

# Modified ipie fork used by QC-AFQMC
pip install "ipie @ git+https://github.com/rr637/ipie.git@develop"

pip install \
  numpy==2.3.0 \
  scipy==1.17.1 \
  h5py==3.16.0 \
  pandas==2.3.3 \
  matplotlib==3.10.9 \
  pyscf==2.13.0 \
  openfermion==1.7.1 \
  qiskit==2.4.1 \
  qiskit-algorithms==0.4.0 \
  qiskit-nature==0.7.2 \
  cudaq==0.14.2 \
  ffsim==0.0.79