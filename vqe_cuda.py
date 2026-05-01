import cudaq
import sys,os
sys.path.append('../')
from mpi_info import MPIInfo
from mpi4py import MPI
import socket
from datetime import datetime
from pathlib import Path

now = datetime.now().astimezone()


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
mpi_info = MPIInfo(comm=comm,rank=rank,size=size)


framework = "mqpu_gradients"


SAVING = True  # Save VQE Results
OUTPUT_DIR = "VQE_Trials"  



if SAVING:
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)



option = 'mqpu,fp64'

cudaq.set_target("nvidia", option = option)
num_qpus_local = cudaq.get_target().num_qpus()

verbose = (rank == 0)
if verbose:
    print(f"Started: {now:%Y-%m-%d %H:%M:%S %Z} | JobID: {os.getenv('SLURM_JOB_ID','N/A')}")

print(f"[rank {rank}/{size}] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
      f"num_qpus={cudaq.get_target().num_qpus()}, host={socket.gethostname()}", flush=True)


from CUDAQ_Pipeline import CUDA_VQE_Event
import time
import numpy as np
from System.MoleculeBuilder import BuildMoleculeProblem
from System.geometry_helpers import build_h_chain_atom_string

from CUDAQ.VQE_CUDARunner import CUDA_VQE,CUDA_VQE_Params
from Trial_wfn import TrialWfn

              


               


R = 1.23
coords = [
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 1.23),
    (R  , 0.0, 0.0),
    (R  , 0.0, 1.23),
]

# h4_eq= "; ".join(f"H {x} {y} {z}" for x, y, z in coords)
# n2_eq = f"N 0 0 0; N 0 0 1.098"
# n2_stretched = f"N 0 0 0; N 0 0 2.5"
# h2_eq = f"H 0 0 0; H 0 0 0.7474"

total_h_atoms = 4
spin =0

bond_lengths = [1.0]
mol_problems = []
for bond_length in bond_lengths:
    h_chain = build_h_chain_atom_string(total_atoms=total_h_atoms,spacing=bond_length)
    mol_problems.append(BuildMoleculeProblem(atom=h_chain,basis="sto-6g", spin=spin,mol_identifier=f"H{total_h_atoms}_{bond_length}_chain"))



optimizer = 'L-BFGS-B'
ftol = 1e-6
gtol=1e-3
k=1


ansatz_types = [
                ("UCCSD","ccsd"), 
                ("UpCCGSD", "zeros"), 
                ("ADAPT-UCCSD", "zeros"), 
                ("HVA", "zeros"), 
                ("LUCJ", "ccsd")
                ]


optim_runtime_budget = None
epsilon = 1e-3      # epsilon for finite difference gradients
noise_sigma = 0.0   # noise added to initial point
amp_threshold = 1e-16   # save amplitudes larger than threshold
adapt_thresh = 1e-3     # gradient norm threshold for adapt
use_symmetry = False    # symm
init_state = "RHF"   # UHF or RHF
for mol_problem in mol_problems:
    system_name = mol_problem.mol_identifier
    for ansatz_type,init_method in ansatz_types:
        vqe_params = CUDA_VQE_Params(ansatz_type=ansatz_type,
                                    k = k,
                                    init_method=init_method,
                                    use_symmetry = use_symmetry,
                                    optimizer=optimizer,
                                    ftol=ftol,
                                    gtol=gtol,
                                    init_state = init_state,
                                    optim_runtime_budget=optim_runtime_budget,
                                    adapt_thresh = adapt_thresh,
                                    framework=framework,
                                    eps=epsilon,
                                    verbose=verbose,
                                    noise_sigma = noise_sigma,
                                    amp_threshold=amp_threshold)
        vqe_event = CUDA_VQE_Event(mol_problem=mol_problem,
                                vqe_params=vqe_params,
                                mpi_info=mpi_info,
                                saving_to_file=SAVING)
        filename = f'{system_name}_{ansatz_type}_{init_state}_{init_method}_k{k}'

        if verbose:
            print("Optimizer: ", optimizer)
            print("ftol:  ", ftol)
            print("gtol: ", gtol)
            print("epsilon: ", epsilon)
            print("noise_sigma: ", noise_sigma)
            print(filename)
            print("HF Energy: ", mol_problem.get_hf_energy())
            # print("UHF Energy: ", mol_problem.mf_uhf.e_tot)
        t0 = time.time()
        cuda_state = vqe_event.run_vqe()
        t1 =  time.time()
        # print(cuda_state.wavefunction)
        if SAVING and rank == 0:
            vqe_event.saveVQE(file_name=filename, output_directory=OUTPUT_DIR, run_group="")










        







                                        

            

        

        

