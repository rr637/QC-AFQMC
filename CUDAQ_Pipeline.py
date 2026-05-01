from System.MoleculeBuilder import BuildMoleculeProblem
from CUDAQ.VQE_CUDARunner import CUDA_VQE,CUDA_VQE_Params #HERE
from time import time
from pathlib import Path
import numpy as np
from datetime import datetime
from numpy import load
from mpi4py import MPI  #
# from tracking import RunTracker
from mpi_info import MPIInfo
class CUDA_VQE_Event:
    def __init__(self,
        mol_problem : BuildMoleculeProblem,
        vqe_params : CUDA_VQE_Params,
        mpi_info: MPIInfo =  None,
        saving_to_file : bool = False):
        self.mol_problem = mol_problem
        self.mpi_info = mpi_info
        self.vqe_params = vqe_params
        self.wf = None
        # file saving variables
        self.saving_to_file = saving_to_file
        self.runtimes = None
        self.energy_list = []
        self.steps_to_conv = -1
        self.converged =False
        self.vqe_dict = None
        self.num_params = None
        self.cx_count = None
        self.statevector = None

    def run_vqe(self):

        vqe_runner = CUDA_VQE(self.mol_problem, self.vqe_params,self.mpi_info)
        # timing
        start_time = time()

        # RUN VQE KERNEL
        Statevector = vqe_runner.run()
        if self.mpi_info.rank == 0:
            self.runtimes = vqe_runner.runtimes

            self.converged = vqe_runner.converged

            self.energy_list = vqe_runner.energy_list
            self.wf = Statevector.getIPIEWavefunction()
            self.statevector = Statevector.statevector
            self.vqe_dict = self.vqe_params.get_dict()
            self.num_params = vqe_runner.num_ansatz_params
            self.two_qubit_count = vqe_runner.two_qubit_count
            self.amp_threshold = vqe_runner.vqe_params.amp_threshold
            self.steps_to_conv = len(self.energy_list)
            self.init_trial = vqe_runner.init_trial.getIPIEWavefunction() if vqe_runner.init_trial is not None else None
            self.init_point = vqe_runner.init_point
        return Statevector
    
    def saveVQE(self, file_name, output_directory, run_group="", verbose=True, save_init_trial = True, save_init_point = False, trial_fidelity=None):
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%y-%m-%d_%H-%M-%S")

        output_directory = Path(output_directory)

        # append storage time
        file_name += f'_{formatted_datetime}.npz'
        file_name = Path(file_name)
        dest = output_directory / file_name   

        atom = self.mol_problem.atom
        if self.init_trial is None:
            save_init_trial  = False
        if self.mol_problem.active_space:
            active_orbitals = self.mol_problem.active_orbitals
            active_mol_nelec = self.mol_problem.active_mol_nelec
        else:
            active_orbitals =  None
            active_mol_nelec = None
        if save_init_trial:
            init_coeffs = self.init_trial[0]
            init_occ_a = self.init_trial[1]
            init_occ_b = self.init_trial[2]
        else:
            init_coeffs = None
            init_occ_a = None
            init_occ_b = None
        if save_init_point:
            init_point = self.init_point
        else:
            init_point = None
        if self.saving_to_file:
            np.savez_compressed(
                dest,
                atom=atom,
                basis=self.mol_problem.basis,
                spin=self.mol_problem.spin,
                active_orbitals=active_orbitals,
                active_mol_nelec=active_mol_nelec,
                mol_identifier=self.mol_problem.mol_identifier,
                coeffs=self.wf[0],
                occ_a=self.wf[1],
                occ_b=self.wf[2],
                init_coeffs = init_coeffs,
                init_occ_a = init_occ_a,
                init_occ_b = init_occ_b,
                group=run_group,
                runtimes=self.runtimes,
                converged=self.converged,
                steps_to_conv=self.steps_to_conv,
                energy_list=self.energy_list,
                vqe_params=self.vqe_dict,
                num_params=self.num_params,
                two_qubit_count = self.two_qubit_count,
                amp_threshold = self.amp_threshold,
                init_point = init_point,
                trial_fidelity=trial_fidelity
            )

