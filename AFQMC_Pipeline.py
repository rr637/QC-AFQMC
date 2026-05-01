from System.MoleculeBuilder import BuildMoleculeProblem
from AFQMC.AFQMCRunner import AFQMCRunner,AFQMCParams
from time import time
from pathlib import Path
import numpy as np
from datetime import datetime
from numpy import load
from mpi4py import MPI  #
from ipie.analysis.autocorr import reblock_by_autocorr  
from Trial_wfn import TrialWfn
import pandas as pd
from  pathlib import Path
from System.geometry_helpers import build_h_chain_atom_string

class AFQMC_Event:
    def __init__(self,
                 mol_problem : BuildMoleculeProblem,
                 afqmc_params : AFQMCParams,
                 saving_to_file : bool = False):
        self.mol_problem = mol_problem
        self.afqmc_params = afqmc_params
        self.saving_to_file = saving_to_file

    def run_afqmc(self, trial_wfn:TrialWfn, global_offset=0,comm=MPI.COMM_WORLD,group_id = 0, compute_FCI_overlap=False):
        self.trial_wfn = trial_wfn.trial_wfn
        self.coeffs = trial_wfn.t_coeffs
        self.occa = trial_wfn.t_occa
        self.occb = trial_wfn.t_occb
        self.num_dets = len(self.coeffs)
        afqmc_run = AFQMCRunner(params=self.afqmc_params, mol_problem=self.mol_problem, Trial=trial_wfn, global_offset=global_offset,comm=comm, group_id=group_id,compute_FCI_overlap=compute_FCI_overlap)
        self.ETotals = afqmc_run.run()
        self.trial_energy,self.afqmc_runtime = afqmc_run.trial_energy,afqmc_run.runtimes
        # print("Trial Energy",trial_energy)
        self.afqmc_dict = self.afqmc_params.__dict__
        self.mean_ac, self.err_ac,_,_ = self.reblocking_analysis()
    def reblocking_analysis(self, burnin=0.25):
        energy_list = self.ETotals
        start = int(burnin * len(energy_list))
        y = np.asarray(energy_list[start:], dtype=float)

        mean_ac = err_ac = None
        block_size = None
        block_df = None

        if len(y) > 10:
            # 1) your existing reblocking
            df = reblock_by_autocorr(y, name="ETotal", verbose=False)
            mean_ac = df["ETotal_ac"].iat[0]
            err_ac  = df["ETotal_error_ac"].iat[0]

            # 2) choose a concrete integer block size
            ac = df["ac"].iat[0]  # integrated autocorr or suggested block length
            block_size = max(1, int(np.ceil(ac)))  # round up to be safe

            # 3) split into blocks
            n = len(y)
            n_blocks = n // block_size
            if n_blocks >= 1:
                y_trim = y[: n_blocks * block_size]
                blocks = y_trim.reshape(n_blocks, block_size)

                # per-block stats
                block_means = blocks.mean(axis=1)
                block_stds  = blocks.std(axis=1, ddof=1) if block_size > 1 else np.zeros(n_blocks)

                # overall from block means (this is the independent-sample SEM)
                overall_mean = block_means.mean()
                overall_sem  = block_means.std(ddof=1) / np.sqrt(n_blocks) if n_blocks > 1 else 0.0

                # pack a tidy table
                starts = start + np.arange(n_blocks) * block_size
                ends   = starts + block_size - 1
                block_df = pd.DataFrame({
                    "block_id":   np.arange(n_blocks),
                    "start_idx":  starts,
                    "end_idx":    ends,
                    "block_size": block_size,
                    "mean":       block_means,
                    "std":        block_stds,
                })
                # You can also return overall_mean/overall_sem if helpful:
                block_df.attrs["overall_mean_from_blocks"] = overall_mean
                block_df.attrs["overall_sem_from_blocks"]  = overall_sem

        return mean_ac, err_ac, block_size, block_df
    def saveAFQMC(self, filename, output_directory, overlap = None,run_group="", verbose=True):
        # current_datetime = datetime.now()
        # formatted_datetime = current_datetime.strftime("%y-%m-%d_%H-%M-%S")

        
        # append storage time
        # filename += f'_{formatted_datetime}.npz'
        filename = Path(filename)
        dest = output_directory / filename
        self.overlap = overlap   
        if self.saving_to_file:
            np.savez_compressed(dest,
                atom=self.mol_problem.atom,
                basis=self.mol_problem.basis,
                spin=self.mol_problem.spin,
                mol_identifier=self.mol_problem.mol_identifier,
                active_orbitals=self.mol_problem.active_orbitals,
                active_mol_nelec=self.mol_problem.active_mol_nelec,
                afqmc_params = self.afqmc_dict,
                ETotals = self.ETotals,
                trial_energy = self.trial_energy,
                coeffs = self.coeffs,
                occa = self.occa,
                occb = self.occb,
                num_dets  = self.num_dets,
                runtime =  self.afqmc_runtime,
                mean_ac = self.mean_ac,
                err_ac = self.err_ac,
                overlap = self.overlap)
import os



def unpack_vqe_data(full_path,mpi_info=None):
    vqe_run_data = load(full_path, allow_pickle=True)
    fname = os.path.basename(full_path)

    # unpack and format data
    if "atom" in vqe_run_data.files:
        atom = str(vqe_run_data['atom'])
    else:
        atom =  build_h_chain_atom_string(total_atoms=10, spacing=2.0)



    basis = str(vqe_run_data['basis'])
    spin = int(vqe_run_data['spin'])
    mol_identifier = str(vqe_run_data['mol_identifier'])
    # print("mol_identifier: ", mol_identifier)
    # temporary, remove  later
    # mol_identifier  =  grab_middle(fname) 
    active_orbitals = None
    active_mol_nelec = None

    if vqe_run_data['active_orbitals'] is not None:
        active_orbitals = vqe_run_data['active_orbitals']
        active_mol_nelec = vqe_run_data['active_mol_nelec']

    mol_problem = BuildMoleculeProblem(atom, basis, spin, active_orbitals, active_mol_nelec,mol_identifier=mol_identifier,mpi_info=mpi_info, verbose=False)

    coeffs = vqe_run_data['coeffs']
    occ_a = vqe_run_data['occ_a']
    occ_b = vqe_run_data['occ_b']
    wf = (np.asarray(coeffs, dtype=np.complex128),
        occ_a,
        occ_b)
    
    return mol_problem, wf, vqe_run_data