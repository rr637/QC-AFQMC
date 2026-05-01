import sys; sys.path.append("../")

from System.MoleculeBuilder import BuildMoleculeProblem
from AFQMC.AFQMCRunner import AFQMCParams

from pathlib import Path
from ipie.config import config
from AFQMC_Pipeline import AFQMC_Event, unpack_vqe_data
from mpi4py import MPI
import random
import os
import copy
import re
from datetime import datetime
from Trial_wfn import TrialWfn
import numpy as np
from mpi_info import MPIInfo
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
mpi_info = MPIInfo(comm=comm, rank=rank, size=size)
config.update_option("use_gpu", False)

comm_size = comm.size if comm is not None else 1
current_datetime = datetime.now()
formatted_datetime = current_datetime.strftime("%y-%m-%d_%H-%M-%S")

# VQE folder containing trial wavefunction files
VQE_DIR = "VQE_Trials"  
SAVING_TO_FILE = True
max_dets= [None] #default
compute_trial_energy = True # gets expensive for 10k+ determinants


# Select which ansatz_types and mol_ids to run AFQMC on.
# group_id controls output file naming:
# results are written to "estimators_group{group_id}".
# Multiple AFQMC simulations can't share the same group_id.
group_id = 0    
total_h = 4
bond_lengths = {1.0}
ansatz_types = {"UCCSD"}
mol_ids = {f"H{total_h}_{bond_length}_chain" for bond_length in bond_lengths}




if rank == 0:
    print(f"[INFO] group_id     = {group_id}")
    print(f"[INFO] total_h      = {total_h}")
    print(f"[INFO] bond_lengths = {sorted(bond_lengths)}")
    print(f"[INFO] ansatz_types = {sorted(ansatz_types)}")




verbose = False
if rank == 0:
    verbose = True
    print("***************************************************")
# 1024 wakers, 10 steps per block, 400 blocks is default
DEFAULT_AFQMC_PARAMS = AFQMCParams(num_total_walkers=256,
    num_steps_per_block=10,
    num_blocks=20,
    timestep=0.01,
    stabilize_freq=5,
    pop_control_freq=5,
    comm_size=comm_size,
    verbose=verbose)

beta = DEFAULT_AFQMC_PARAMS.num_steps_per_block * DEFAULT_AFQMC_PARAMS.timestep * DEFAULT_AFQMC_PARAMS.num_blocks


variation_grid = [
dict(num_steps_per_block=10)]


for variation in variation_grid:
    THIS_AFQMC_PARAMS = DEFAULT_AFQMC_PARAMS.copy_and_update(variation)

    # deterministic ordering across all jobs
    for entry_name in sorted(os.listdir(VQE_DIR)):
        full_path = os.path.join(VQE_DIR, entry_name)
        if not os.path.isfile(full_path):
            continue
        # load and filter by mol id and ansatz type, as before
        mol_problem, (coeffs,occa,occb), vqe_run_data = unpack_vqe_data(
            full_path, mpi_info=mpi_info
        )
        if str(vqe_run_data["mol_identifier"]) not in mol_ids:
            continue

        raw = vqe_run_data["vqe_params"]
        vqe_params = raw.item() if isinstance(raw, np.ndarray) and raw.dtype == object else raw
        ansatz_type = vqe_params.get("ansatz_type", "UNKNOWN")
        k = vqe_params.get("k","UNKNOWN")
        init_state = vqe_params.get("init_state", "UNKNOWN")
        if ansatz_type not in ansatz_types:
            continue
        num_dets = len(coeffs)

        if not rank:
            print(f'# Reading from: {full_path}')
            print("Num Full Determinents: ", num_dets)

        for max_det in max_dets:
            t0 = time.time()

            Trial = TrialWfn(
                coeffs=coeffs,
                occa=occa,
                occb=occb,
                mol_problem=mol_problem,
                max_det=max_det,
                compute_trial_energy=compute_trial_energy,
            )

            this_afqmc_run = AFQMC_Event(
                mol_problem=mol_problem,
                afqmc_params=THIS_AFQMC_PARAMS,
                saving_to_file=SAVING_TO_FILE,
            )
            if rank == 0:
                print(f'Running AFQMC on: {full_path}')
                print(f"Num Truncated Determinents: {Trial.num_dets}")
                print("HF Energy: ", mol_problem.get_hf_energy())
                print(f"# Final Energy from VQE: {vqe_run_data['energy_list'][-1]}")

            this_afqmc_run.run_afqmc(Trial, group_id=group_id)

            if rank == 0 and SAVING_TO_FILE:
                filename = f"max_det{max_det}_{entry_name}_beta{beta}"
                output_directory = Path(VQE_DIR) / f"AFQMC_{str(vqe_run_data["mol_identifier"])}"
                output_directory.mkdir(parents=True, exist_ok=True)
                this_afqmc_run.saveAFQMC(filename=filename, output_directory=output_directory)
                t1 = time.time()
                print(f"max_dets: {max_det}, time = {t1-t0}")
            if max_det is not None and max_det >= num_dets:
                break