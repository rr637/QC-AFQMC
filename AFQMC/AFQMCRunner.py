import sys; sys.path.append('../')

from System.MoleculeBuilder import BuildMoleculeProblem
from ipie.trial_wavefunction.particle_hole import ParticleHole
from ipie.utils.from_pyscf import generate_integrals
from ipie.hamiltonians.generic  import Generic as HamGeneric
from ipie.walkers.uhf_walkers   import UHFWalkersParticleHole
from ipie.utils.mpi             import MPIHandler
from ipie.qmc.afqmc             import AFQMC
from ipie.analysis.extraction  import extract_observable
from ipie.estimators.estimator_base import EstimatorBase
from ipie.estimators.greens_function import greens_function
import numpy as np
from Trial_wfn import TrialWfn
from pyscf.fci import cistring
import numpy as np
from types import SimpleNamespace
from mpi4py import MPI
import time

class AFQMCParams(SimpleNamespace):
    def __init__(self, num_total_walkers,
        num_steps_per_block,
        num_blocks,
        timestep,
        stabilize_freq,
        pop_control_freq,
        comm_size,
        verbose
    ):
        self.num_total_walkers = num_total_walkers
        self.num_steps_per_block = num_steps_per_block
        self.num_blocks = num_blocks
        self.timestep = timestep
        self.stabilize_freq = stabilize_freq
        self.pop_control_freq = pop_control_freq
        self.comm_size = comm_size
        self.verbose = verbose
        self.num_walkers = num_total_walkers // comm_size

    def copy_and_update(self, update_dict : dict):
        new_params =  AFQMCParams(
        self.num_total_walkers,
        self.num_steps_per_block,
        self.num_blocks,
        self.timestep,
        self.stabilize_freq,
        self.pop_control_freq,
        self.comm_size,
        self.verbose
       )
        
        for key, value in update_dict.items():
           if not hasattr(new_params, key):
              raise KeyError(f'unknown param field: {key}')
           setattr(new_params, key, value)

        new_params.num_walkers = (
           new_params.num_total_walkers // new_params.comm_size
        )
    
        return new_params

class AFQMCRunner:
  def __init__(self, 
        params: AFQMCParams,
        mol_problem: BuildMoleculeProblem,
        Trial: TrialWfn,
        global_offset=0,
        comm=MPI.COMM_WORLD,
        group_id = 0,
        compute_FCI_overlap = False): 
    self.params = params
    self.mol_problem = mol_problem
    self.Trial = Trial
    self.global_offset = global_offset
    self.comm  = comm
    self.group_id = group_id
    self.compute_FCI_overlap = compute_FCI_overlap

    
  def run(self):
    t0 = time.time()
    
    # ham  = self.mol_problem.build_ipie_ham()
    ham = self.mol_problem.build_ipie_ham_from_fcidump()
    PH_trial = self.Trial.build_PH_trial() #half rotate (ham), calculates energy
    t1 = time.time()
    if self.comm.Get_rank() == 0:
       print("Trial build time: ", t1-t0)
       

    # Explicitiy making init_walker state RHF state can cause error (singular matrix error) if trial overlap with RHF is zero 
    rhf_trial = self.mol_problem.get_HF_trial()
    fake_ph_trial = rhf_trial.build_PH_trial() 
    initial_walker = np.hstack([fake_ph_trial.psi0a, fake_ph_trial.psi0b])
    
    # Making init_walker first determinent of trial (usually RHF) 
    # initial_walker = np.hstack([PH_trial.psi0a, PH_trial.psi0b])


    
    np.random.seed(12345678)



    #optional add a small pertubation to each walker so each walker starts slightly different 
    # random_perturbation = np.random.random(initial_walker.shape)
    # initial_walker = initial_walker + random_perturbation
    # initial_walker, _ = np.linalg.qr(initial_walker)
    #constructus walker population object
    # print("Trial CI coefficients:")
    # print(PH_trial.coeffs)

    # # or, more verbosely, print each determinant’s index, coefficient, and occupation
    # for i, (c, occ) in enumerate(zip(PH_trial.coeffs, PH_trial.spin_occs)):
    #     print(f"determinant {i:3d}:  c = {c:.6e}    occ = {occ}")
    walkers = UHFWalkersParticleHole(
        initial_walker,
        self.mol_problem.mol_nelec[0], #num of alpha and beta electrons
        self.mol_problem.mol_nelec[1],
        PH_trial.psi0a.shape[0],
        self.params.num_walkers,
        MPIHandler(self.comm))#handles any MPI    )
    walkers.build(PH_trial)


    afqmc_msd = AFQMC.build(
        self.mol_problem.mol_nelec,
        ham,
        PH_trial, 
        walkers=walkers, # initial walker populationo object
        num_walkers=self.params.num_walkers, 
        num_steps_per_block=self.params.num_steps_per_block, # number of dt steps per block, each step: sample from AF, apply propogatoor, comptute overlaps, and update walker weightss
        num_blocks=self.params.num_blocks, # each block aaverages local energies (or any observable) over all (num_steps_per_block) steps 
        timestep=self.params.timestep, # dt
        stabilize_freq=self.params.stabilize_freq, # after (stapilize_freq) steps, we re-orthonormalize from repeatedly propogating
        seed=11,
        pop_control_freq=self.params.pop_control_freq, # after (pop_control_freq) steps, we kill low-weight walkers and clone high weight walkers, to keep # of walkers roughly the same
        verbose=self.params.verbose,
        mpi_handler = MPIHandler(self.comm))
    estimator_filename = f"estimators/estimators_group{self.group_id}.h5"


    t0 = time.time()
    afqmc_msd.run(estimator_filename = estimator_filename)
    runtime = time.time()-t0
    ETotals = None  # <-- define on ALL ranks first

    if self.comm.Get_rank() == 0:
        qmc_data = extract_observable(estimator_filename, "energy")
        # robust: make it a plain numpy array (handles pandas Series/list/etc)
        ETotals = np.asarray(qmc_data["ETotal"], dtype=float)

    ETotals = self.comm.bcast(ETotals, root=0)
    ETotals = ETotals.tolist() 

    afqmc_msd.finalise(verbose=True)
    self.trial_energy = self.Trial.var_energy
    self.runtimes = runtime
    return ETotals
    


class S2Mixed(EstimatorBase):
    def __init__(self, ham):
        self._data = {"S2": np.zeros((1,), dtype=np.complex128)}
        self._shape = (1,)
        self.scalar_estimator = False
        self.print_to_stdout = True
        self.ascii_filename = None

    def compute_estimator(self, system, walkers, hamiltonian, trial):
        greens_function(walkers, trial, build_full=True)
        # if walkers.mpi_handler.rank == 0:
        #     for w in range(min(20, walkers.nwalkers)):
        #         print("||phia-phib||", w, np.linalg.norm(walkers.phia[w] - walkers.phib[w]))
        ndown = system.ndown
        nup = system.nup
        Ms = (nup - ndown) / 2.0
        two_body = -np.einsum("wij,wji->w", walkers.Ga, walkers.Gb)
        two_body = two_body * walkers.weight

        denom = np.sum(walkers.weight)
        numer = np.sum(two_body) + denom * (Ms * (Ms + 1) + ndown)

        self["S2"] = numer / (denom + 1e-16)
  


def slater_norm_from_orbs(phi_a, phi_b, eps=1e-16):
    # phi_a: (nbasis, nocc_a), phi_b: (nbasis, nocc_b)
    sa = np.linalg.det(phi_a.conj().T @ phi_a).real
    sb = np.linalg.det(phi_b.conj().T @ phi_b).real
    sa = max(sa, eps)
    sb = max(sb, eps)
    return np.sqrt(sa * sb)



  

