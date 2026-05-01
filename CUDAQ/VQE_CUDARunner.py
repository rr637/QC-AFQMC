import sys
sys.path.append("../")
import cudaq
import numpy as np
from System.MoleculeBuilder import BuildMoleculeProblem
from CUDAQ.qchem_hamiltonian import jordan_wigner_fermion
from CUDAQ.hva import *
from mpi_info import MPIInfo
from CUDAQ.ucc import get_ucc_two_qubit_gate_count,ucc_circuit,get_excitation_list,collect_pauli_from_excitation_list, ansatz_type_to_kwargs, get_symmetry_excitation_list
from CUDAQ.adapt_ucc import *
from CUDAQ.CUDAQStatevector import BlockStatevector
from types import SimpleNamespace
import copy
import sys
from CUDAQ.optimizers import *
from System.MoleculeBuilder import BuildMoleculeProblem
from CUDAQ.uhf import fock_unitary_apply_from_U
import time
from CUDAQ.utils import compute_pyscf_initial_point,  order_excitations_by_init
from CUDAQ.lucj import *


class CUDA_VQE_Params(SimpleNamespace):
	def __init__(self,ansatz_type,k, init_method, ordered = True, use_symmetry = False,optimizer = "L-BFGS-B", ftol = 1e-5,gtol=1e-3,init_state = "RHF",optim_runtime_budget=None,max_iters = None,adapt_thresh = 1e-3,noise_sigma =  0, framework='batched_gradients', eps=1e-3,amp_threshold=1e-16,verbose =True, custom_ucc = None):
		
		ansatz_types = {"UCCSD", "UCCGSD", "UpCCGSD", "HEA", "HVA", "ADAPT-UCCSD", "UpCCD","QNP", "LUCJ"}
		if ansatz_type not in ansatz_types:
			raise ValueError(f"method must be one of {ansatz_types}, got {ansatz_type!r}")
		self.UCC = (ansatz_type in {"UCCSD", "UCCGSD", "UpCCGSD", "UpCCD"})
		self.ansatz_type = ansatz_type
		self.k =  k
		self.init_method = init_method
		self.use_symmetry = use_symmetry
		self.ordered = ordered
		self.framework = framework    # "batched_gradients" or  "parameter_shift"
		self.optimizer = optimizer
		self.ftol = ftol
		self.gtol = gtol
		self.max_iters = max_iters
		self.adapt_thresh = adapt_thresh
		self.noise_sigma  = noise_sigma
		self.eps = eps
		self.amp_threshold = amp_threshold
		self.verbose = verbose
		self.optim_runtime_budget = optim_runtime_budget
		# excitataion list, init_point
		self.custom_ucc = custom_ucc
		self.init_state = init_state
	def get_dict(self):
		log_dict = copy.copy(self.__dict__)
		return log_dict

class CUDA_VQE: 
	def __init__(self,mol_problem : BuildMoleculeProblem,
							 vqe_params: CUDA_VQE_Params,
							 mpi_info: MPIInfo = None):
		self.mol_problem = mol_problem
		self.vqe_params = vqe_params

		self.mpi_info = mpi_info
		self.molecular_data = mol_problem.get_mol_hamiltonian_from_fcidump()
		self.obi = self.molecular_data[0]
		self.tbi = self.molecular_data[1]
		self.econst = self.molecular_data[2]
		self.electron_count = self.molecular_data[3]
		self.norbitals = self.molecular_data[4]
		self.fer_ham = self.molecular_data[5]
		self.qubits_num = 2 * self.norbitals
		self.spin_mult  = self.mol_problem.spin
		if self.vqe_params.init_state == "UHF":
			_ = self.mol_problem.mf 
			self.spin_op_ham_rot, self.U_so = self.rotate_ham()
			if self.mpi_info.rank == 0:
				print("UHF Energy: ", self.mol_problem.mf_uhf.e_tot)

		self.spin_op_ham = jordan_wigner_fermion(self.obi,self.tbi,ecore=0.0,tolerance=1e-15)
		
		self.num_qpus = cudaq.get_target().num_qpus()
		self.init_trial = None
		self.amp_threshold = self.vqe_params.amp_threshold
		self.runtimes = {}
		self.converged = False
		self.energy_list = []
		self.num_ansatz_params = 0
		


	def run(self):

		if self.vqe_params.UCC:
			cuda_state = self.run_ucc()
		elif self.vqe_params.ansatz_type == 'HVA':
			cuda_state = self.run_hva()
		elif self.vqe_params.ansatz_type == 'ADAPT-UCCSD':
			cuda_state = self.run_adapt_ucc()
		elif self.vqe_params.ansatz_type == "LUCJ":
			cuda_state = self.run_lucj()
		else:
			raise NotImplementedError


		return cuda_state

	

	
	def run_ucc(self):

		if self.vqe_params.ansatz_type == "UpCCGSD" and self.vqe_params.k > 1:
			return self.upccgsd_layered_optimize()
		t0 = time.time()
		verbose = self.vqe_params.verbose
		runtimes = {}
		k = self.vqe_params.k
		mol_problem = self.mol_problem
		if self.vqe_params.custom_ucc is not None:
			excitation_list,init_point  = self.vqe_params.custom_ucc
		else:

			excitations, extra_kwargs = ansatz_type_to_kwargs(self.vqe_params.ansatz_type)
			excitation_list = get_excitation_list(excitations=excitations,mol_problem=mol_problem,
																					extra_kwargs=extra_kwargs)
			# if self.vqe_params.test:
			# 	excitation_list = excitation_list[:16]

			full_excitation_list = excitation_list
			if self.vqe_params.use_symmetry:
				excitation_list,group = get_symmetry_excitation_list(excitation_list=excitation_list,mol_problem=mol_problem)
				if verbose:
					print(f"Full num excitation_operators: {len(full_excitation_list)}, kept {len(excitation_list)} preserving point group {group}")
			if self.vqe_params.init_state == "RHF":
				mf = self.mol_problem.mf
			else:
				mf = self.mol_problem.mf_uhf
			init_point = compute_pyscf_initial_point(method = self.vqe_params.init_method,
																						mol_problem = self.mol_problem,
																						active_spatial = mol_problem.active_spatial,
																						excitation_list = excitation_list,
																						UCC = True,
																						noise_sigma=self.vqe_params.noise_sigma)
		if self.vqe_params.ordered:
			excitation_list,init_point, _ = order_excitations_by_init(excitation_list,init_point)
		excitation_list = excitation_list 
		if k > 1:
			init_point = np.concatenate([init_point] + [np.zeros_like(init_point) for _ in range(k - 1)])

		self.init_point = init_point

				

 
		pauli_words,coeffs,block_ids = collect_pauli_from_excitation_list(excitation_list=excitation_list,n_spin_orbitals=mol_problem.active_spin_orbitals, nreps=self.vqe_params.k)


		self.two_qubit_count = get_ucc_two_qubit_gate_count(pauli_words)
		self.num_ansatz_params = block_ids[-1]+1
		assert len(init_point) == block_ids[-1]+1
		num_qubit = self.qubits_num
		if self.vqe_params.init_state == "RHF":
			ucc_spec = (num_qubit,THETA,pauli_words,coeffs,block_ids,mol_problem.active_orbitals,mol_problem.active_mol_nelec[0],mol_problem.active_mol_nelec[1])

			init_spec = (num_qubit,init_point,pauli_words,coeffs,block_ids,mol_problem.active_orbitals,mol_problem.active_mol_nelec[0],mol_problem.active_mol_nelec[1])
			kernel = ucc_circuit
			spin_op_ham = self.spin_op_ham
		elif self.vqe_params.init_state == "UHF":
			ucc_spec = (num_qubit,THETA,pauli_words,coeffs,block_ids,mol_problem.active_orbitals,mol_problem.active_mol_nelec[0],mol_problem.active_mol_nelec[1])

			init_spec = (num_qubit,init_point,pauli_words,coeffs,block_ids,mol_problem.active_orbitals,mol_problem.active_mol_nelec[0],mol_problem.active_mol_nelec[1])
			kernel = ucc_circuit
			spin_op_ham = self.spin_op_ham_rot


		if self.vqe_params.init_state == "UHF":
			init_state = np.array(cudaq.get_state(
							kernel,*init_spec), dtype=complex)
			init_state_rot = fock_unitary_apply_from_U(init_state, self.U_so, dagger=False)
			init_state_rot = self.convert_state_big_endian(init_state_rot)
			init_state = init_state_rot

			
		else:
			init_state = self.convert_state_big_endian(
						np.array(cudaq.get_state(
								kernel,*init_spec), dtype=complex))

		self.init_trial = Statevector = BlockStatevector(statevector=init_state, mol_problem=self.mol_problem,ampl_eps=self.amp_threshold)
		
		
	
		t1 = time.time()
		runtimes['pre-process'] = t1 - t0

		opt = make_optimizer(
				framework=self.vqe_params.framework,
				kernel=kernel,
				hamiltonian=spin_op_ham,
				kernel_arg_spec=ucc_spec,
				init_theta=init_point,
				econst=self.econst,
				max_walltime = self.vqe_params.optim_runtime_budget,
				epsilon=self.vqe_params.eps,
				optimizer_method=self.vqe_params.optimizer,
				ftol = self.vqe_params.ftol,
				gtol = self.vqe_params.gtol,
				mpi_info=self.mpi_info
		)

		if verbose:
			print("two_qubit_count", self.two_qubit_count)

			print("HF Energy: ", self.mol_problem.get_hf_energy())
			if self.vqe_params.init_state == "UHF":
				print("UHF Energy: ", self.mol_problem.mf_uhf.e_tot)
			print(f"QUBITS: {self.mol_problem.active_orbitals*2}")
			print("Num params: ", self.num_ansatz_params)
			print("Runtime Budget: ", self.vqe_params.optim_runtime_budget)

			# print(extra_kwargs)


		t1_ = time.time()
		results, energies = opt.optimize()
		t2 = time.time()
		self.converged = results.success
		total_energy = results.fun + self.econst
		opt_params = results.x
		runtimes = opt.timing_summary()
		if self.vqe_params.framework == "batched_gradients":
			eval_counts = opt.gather_eval_counts()
			runtimes["eval_counts"] = eval_counts
		runtimes['pre-process'] = t1 - t0
		runtimes["full_optim"] = t2 - t1_

		self.amp_threshold = self.vqe_params.amp_threshold

		final_spec = (num_qubit,opt_params,pauli_words,coeffs,block_ids,mol_problem.active_orbitals,mol_problem.active_mol_nelec[0],mol_problem.active_mol_nelec[1])
		final_kernel = kernel

		
		self.energy_list = energies
		if verbose:

			state = np.array(cudaq.get_state(
							final_kernel,*final_spec), dtype=complex)
			if self.vqe_params.init_state == "UHF":
				state = fock_unitary_apply_from_U(state, self.U_so, dagger=False)

			state = self.convert_state_big_endian(state)
		

			Statevector = BlockStatevector(statevector=state, mol_problem=self.mol_problem,ampl_eps=self.amp_threshold)
		else:
			Statevector = None
		t3 = time.time()
		runtimes["post_process"] = t3 - t2
		self.runtimes = runtimes

		if verbose:


			print(f"Converged: {self.converged}")




			print("PRE PROCESS Runtime: ",runtimes['pre-process'] )
			print("FULL OPTIMIZATION TIME: ", runtimes["full_optim"])
			print("POST-PROCESS Runtime: ", runtimes["post_process"])
			for k,v in self.runtimes.items():
				print(f"{k}: {v}")
			if self.vqe_params.framework == "batched_gradients":

				print("num_energy_evals (global):", eval_counts["total"])
				print("num_energy_evals_per_rank (global):", eval_counts["per_rank"])
		return Statevector	
	
	def upccgsd_layered_optimize(self):
		t0 = time.time()
		verbose = self.vqe_params.verbose
		if verbose:
			print("Running layered optimization with k = ", self.vqe_params.k)
		runtimes = {}
		k_final = int(self.vqe_params.k)
		mol_problem = self.mol_problem

		if self.vqe_params.custom_ucc is not None:
			excitation_list, init_point_k1 = self.vqe_params.custom_ucc
		else:
			excitations, extra_kwargs = ansatz_type_to_kwargs(self.vqe_params.ansatz_type)
			excitation_list = get_excitation_list(
				excitations=excitations,
				mol_problem=mol_problem,
				extra_kwargs=extra_kwargs
			)

			full_excitation_list = excitation_list
			if self.vqe_params.use_symmetry:
				excitation_list, group = get_symmetry_excitation_list(
					excitation_list=excitation_list,
					mol_problem=mol_problem
				)
				if verbose:
					print(
						f"Full num excitation_operators: {len(full_excitation_list)}, "
						f"kept {len(excitation_list)} preserving point group {group}"
					)

			init_point_k1 = compute_pyscf_initial_point(
				method=self.vqe_params.init_method,
				mol_problem=mol_problem,
				active_spatial=mol_problem.active_spatial,
				excitation_list=excitation_list,
				UCC=True,
				noise_sigma=self.vqe_params.noise_sigma
			)

		# Order once by |CCSD amps|; later blocks inherit same ordering via nreps replication
		if self.vqe_params.ordered:
			excitation_list, init_point_k1, _ = order_excitations_by_init(excitation_list, init_point_k1)

		p = len(init_point_k1)

		def build_specs(nreps: int, theta: np.ndarray):
			pauli_words, coeffs, block_ids = collect_pauli_from_excitation_list(
				excitation_list=excitation_list,
				n_spin_orbitals=mol_problem.active_spin_orbitals,
				nreps=nreps
			)
			num_params = int(block_ids[-1] + 1)
			assert len(theta) == num_params, (len(theta), num_params)

			num_qubit = self.qubits_num
			ucc_spec = (
				num_qubit, THETA, pauli_words, coeffs, block_ids,
				mol_problem.active_orbitals,
				mol_problem.active_mol_nelec[0],
				mol_problem.active_mol_nelec[1]
			)
			theta_spec = (
				num_qubit, theta, pauli_words, coeffs, block_ids,
				mol_problem.active_orbitals,
				mol_problem.active_mol_nelec[0],
				mol_problem.active_mol_nelec[1]
			)
			return pauli_words, coeffs, block_ids, ucc_spec, theta_spec

		energies_all = []
		theta_prev = None
		results = None

		t_pre = time.time()

		for k_stage in range(1, k_final + 1):
			if k_stage == 1:
				theta0 = np.array(init_point_k1, dtype=float, copy=True)
			else:
				theta0 = np.concatenate([theta_prev, np.zeros(p, dtype=float)])

			pauli_words, coeffs, block_ids, ucc_spec, theta_spec = build_specs(k_stage, theta0)

			self.two_qubit_count = get_ucc_two_qubit_gate_count(pauli_words)
			self.num_ansatz_params = int(block_ids[-1] + 1)

			if verbose:
				e0 = cudaq.observe(ucc_circuit, self.spin_op_ham, *theta_spec).expectation() + self.econst
				print(f"[k={k_stage}] init energy: {e0:.12f} Ha   num_params={self.num_ansatz_params}")

			opt = make_optimizer(
				framework=self.vqe_params.framework,
				kernel=ucc_circuit,
				hamiltonian=self.spin_op_ham,
				kernel_arg_spec=ucc_spec,
				init_theta=theta0,
				econst=self.econst,
				max_walltime=self.vqe_params.optim_runtime_budget,
				epsilon=self.vqe_params.eps,
				optimizer_method=self.vqe_params.optimizer,
				ftol=self.vqe_params.ftol,
				gtol=self.vqe_params.gtol,
				mpi_info=self.mpi_info
			)

			results, energies = opt.optimize()

			theta_prev = np.array(results.x, dtype=float)
			energies_all.extend(list(energies))

			if verbose:
				E_stage = float(results.fun) + self.econst
				print(f"[k={k_stage}] success={getattr(results,'success',True)}  E={E_stage:.12f} Ha")

		t_post = time.time()
		runtimes["pre-process"] = t_pre - t0
		runtimes["full_optim"] = t_post - t_pre

		# final
		opt_params = theta_prev
		self.energy_list = np.array(energies_all, dtype=float)
		self.converged = bool(getattr(results, "success", True))

		# no init trial requested
		self.init_point = None
		self.init_trial = None

		# optional final state
		if verbose:
			_, _, _, _, final_spec = build_specs(k_final, opt_params)
			state = self.convert_state_big_endian(
				np.array(cudaq.get_state(ucc_circuit, *final_spec), dtype=complex)
			)
			Statevector = BlockStatevector(
				statevector=state,
				mol_problem=self.mol_problem,
				ampl_eps=self.vqe_params.amp_threshold
			)
		else:
			Statevector = None

		t_end = time.time()
		runtimes["post_process"] = t_end - t_post
		self.runtimes = runtimes

		if verbose:
			print(f"Converged: {self.converged}")
			print("two_qubit_count", self.two_qubit_count)
			print("HF Energy: ", self.mol_problem.get_hf_energy())
			print(f"QUBITS: {self.mol_problem.active_orbitals*2}")
			print("Num params: ", self.num_ansatz_params)
			print("Runtime Budget: ", self.vqe_params.optim_runtime_budget)

		return Statevector
		
	def run_lucj(self):
		t0 = time.time()
		verbose = self.vqe_params.verbose
		mol_problem = self.mol_problem
		norb = mol_problem.active_orbitals
		n_reps = self.vqe_params.k
		# if self.vqe_params.init_method  != "ccsd":
		# 	raise NotImplementedError

		norb = mol_problem.active_orbitals
		pairs_aa = [(p, p + 1) for p in range(norb - 1)]
		pairs_ab = [(p, p) for p in range(norb)]
		interaction_pairs = (pairs_aa,pairs_ab)
		if self.vqe_params.init_method in ["ccsd", "rccsd", "uccsd","random"]:
			if self.vqe_params.init_method in ["rccsd","ccsd"]:
				restricted = True
			elif self.vqe_params.init_method in ["uccsd"]:
				restricted = False
			ccsd = self.mol_problem.build_ccsd(restricted=restricted)
			ucj_op_init = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
			ccsd.t2,
			t1=ccsd.t1,
			n_reps=n_reps,
			interaction_pairs=interaction_pairs,
			# Setting optimize=True enables the "compressed" factorization
			optimize=True,
			options=dict(maxiter=100),
				regularization=1e-2
				)
		elif self.vqe_params.init_method == "zeros":
			n_params = ffsim.UCJOpSpinBalanced.n_params(
            norb = norb,
            n_reps = n_reps,
            interaction_pairs=interaction_pairs,
            with_final_orbital_rotation=True,
        )
			x = np.zeros(n_params)
			ucj_op_init = ffsim.UCJOpSpinBalanced.from_parameters(
            x, norb=norb, n_reps=n_reps,
            interaction_pairs=interaction_pairs,
            with_final_orbital_rotation=True
        )

		self.num_ansatz_params = len(ucj_op_init.to_parameters())
		x0 = ucj_op_init.to_parameters(interaction_pairs=interaction_pairs)
		if self.vqe_params.noise_sigma > 0:
			x0 += np.random.normal(0.0,self.vqe_params.noise_sigma, x0.shape)
		if self.vqe_params.init_method == "random":
			x0 = np.random.uniform(-np.pi, np.pi, x0.shape)
		self.init_point = x0
		lucj_spec_init =  build_lucj_packed_args(ucj_op_init, norb)

		args_init = (norb, self.mol_problem.mol_nelec[0], self.mol_problem.mol_nelec[1],*lucj_spec_init)
		init_state = self.convert_state_big_endian(
						np.array(cudaq.get_state(
								lucj_circuit, *args_init), dtype=complex)
				)
		self.init_trial = BlockStatevector(statevector=init_state, mol_problem=self.mol_problem,ampl_eps=self.vqe_params.amp_threshold)
		self.two_qubit_count = twoq_count_lucj_from_packed(lucj_spec_init)
		# made zeros init point
		# x0 =  np.zeros_like(x0)


		if verbose:
			print("two_qubit_count", self.two_qubit_count)
			print("HF Energy: ", self.mol_problem.get_hf_energy())
			print(f"QUBITS: {self.mol_problem.active_orbitals*2}")
			print("Num params: ", self.num_ansatz_params)
			print("Runtime Budget: ", self.vqe_params.optim_runtime_budget)
			# print(extra_kwargs)

		opt  = LUCJBatchedGradOptimizer(kernel = lucj_circuit,
																	hamiltonian=self.spin_op_ham,
																	init_theta = x0,
																	interaction_pairs=interaction_pairs,
																	norb = norb,
																	mol_nelec = mol_problem.mol_nelec,
																	n_reps = n_reps,
																	max_walltime=self.vqe_params.optim_runtime_budget,
																	econst  = self.econst,
																epsilon=self.vqe_params.eps,
																optimizer_method=self.vqe_params.optimizer,
																	opt_tol=self.vqe_params.ftol,
																	mpi_info=self.mpi_info)



		# Qiskit test

		# qubits = QuantumRegister(2 * norb)
		# circuit = QuantumCircuit(qubits)
		# circuit.append(ffsim.qiskit.PrepareHartreeFockJW(norb, mol_problem.mol_nelec), qubits)
		# circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op_init), qubits)
		# qiskit_ham = mol_problem.build_qiskit_ham() #Sparse Pauli Op


		# est = Estimator()  

		# job = est.run(circuits=[circuit], observables=[qiskit_ham])
		# res = job.result()
		# qiskit_energy = np.real(res.values[0] +  self.econst)
		# print("qiskit initial energy:  ", qiskit_energy)

		zero_vector = np.zeros_like(x0)
		zero_energy = opt._observe_energy_call(zero_vector).expectation() + self.econst

		t1_ = time.time()
		initial_energy = opt._observe_energy_call(x0).expectation() + self.econst
		# print("cudaq_initial_energy: ", initial_energy)
		results, energies = opt.optimize()
		t2 = time.time()
		self.converged = results.success
		opt_params = np.array(results.x)
		runtimes = opt.timing_summary()
		if self.vqe_params.framework == "batched_gradients":
			eval_counts = opt.gather_eval_counts()
			runtimes["eval_counts"] = eval_counts
		runtimes['pre-process'] = t1_ - t0
		runtimes["full_optim"] = t2 - t1_

		self.amp_threshold = self.vqe_params.amp_threshold


		self.energy_list = energies

		if verbose:
			final_ucj_op = ffsim.UCJOpSpinBalanced.from_parameters(
			opt_params, norb=norb, n_reps=n_reps,
			interaction_pairs=interaction_pairs,
			with_final_orbital_rotation=True
			)

			lucj_spec =  build_lucj_packed_args(final_ucj_op, norb)

			args = (norb, self.mol_problem.mol_nelec[0], self.mol_problem.mol_nelec[1],*lucj_spec)

			state = self.convert_state_big_endian(
						np.array(cudaq.get_state(
								lucj_circuit, *args), dtype=complex)
				)

			Statevector = BlockStatevector(statevector=state, mol_problem=self.mol_problem,ampl_eps=self.amp_threshold)
		else:
			Statevector = None
		t3 = time.time()
		runtimes["post_process"] = t3 - t2
		self.runtimes = runtimes

		if verbose:

			print("Converged: ", self.converged)

			print("PRE PROCESS Runtime: ",runtimes['pre-process'] )
			print("FULL OPTIMIZATION TIME: ", runtimes["full_optim"])
			print("POST-PROCESS Runtime: ", runtimes["post_process"])
			for k,v in self.runtimes.items():
				print(f"{k}: {v}")
			if self.vqe_params.framework == "batched_gradients":

				print("num_energy_evals (global):", eval_counts["total"])
				print("num_energy_evals_per_rank (global):", eval_counts["per_rank"])
		return Statevector	
		
		
	def run_hva(self):

		t0 = time.time()
		verbose = self.vqe_params.verbose
		runtimes = {}
		k = self.vqe_params.k
		mol_problem = self.mol_problem


				
		# print(f"Excitation_list: {excitation_list}")
		# print(f"pauli words:  {pauli_words}")
		# # print(f"Hamiltonian {self.spin_op_ham}")
		# print(f"({self.vqe_params.init_method}) initial point: {init_point}")
 
		pauli_words,coeffs,block_ids = build_HVA(mol_problem, nreps=k)
		self.num_ansatz_params = block_ids[-1]+1

		self.init_point = np.ones(self.num_ansatz_params)
		init_point = self.init_point
		self.two_qubit_count = None




		num_qubit = self.qubits_num
		if self.vqe_params.init_state == "RHF":
			hva_spec = (num_qubit,THETA,pauli_words,coeffs,block_ids,mol_problem.active_orbitals,mol_problem.active_mol_nelec[0],mol_problem.active_mol_nelec[1])

			init_spec = (num_qubit,init_point,pauli_words,coeffs,block_ids,mol_problem.active_orbitals,mol_problem.active_mol_nelec[0],mol_problem.active_mol_nelec[1])
			kernel = hva_circuit
			spin_op_ham = self.spin_op_ham
		elif self.vqe_params.init_state == "UHF":
			hva_spec = (num_qubit,THETA,pauli_words,coeffs,block_ids,mol_problem.active_orbitals,mol_problem.active_mol_nelec[0],mol_problem.active_mol_nelec[1])

			init_spec = (num_qubit,init_point,pauli_words,coeffs,block_ids,mol_problem.active_orbitals,mol_problem.active_mol_nelec[0],mol_problem.active_mol_nelec[1])
			kernel = hva_circuit
			spin_op_ham = self.spin_op_ham_rot
		initial_energy = cudaq.observe(kernel,spin_op_ham,*init_spec).expectation()
		# print(cudaq.draw(hva_circuit,*init_spec))
		print("Initial Energy: ", initial_energy + self.econst)
		# raise Exception
		init_state = self.convert_state_big_endian(
					np.array(cudaq.get_state(
							kernel,
							*init_spec
					), dtype=complex)
			)
		self.init_trial = BlockStatevector(statevector=init_state, mol_problem=self.mol_problem,ampl_eps=self.vqe_params.amp_threshold)
		
		
	
		t1 = time.time()
		runtimes['pre-process'] = t1 - t0

		opt = make_optimizer(
				framework=self.vqe_params.framework,
				kernel=kernel,
				hamiltonian=spin_op_ham,
				kernel_arg_spec=hva_spec,
				init_theta=init_point,
				econst=self.econst,
				max_walltime = self.vqe_params.optim_runtime_budget,
				epsilon=self.vqe_params.eps,
				optimizer_method=self.vqe_params.optimizer,
				ftol = self.vqe_params.ftol,
				gtol = self.vqe_params.gtol,
				mpi_info=self.mpi_info
		)

		if verbose:

			print("HF Energy: ", self.mol_problem.get_hf_energy())
			print(f"QUBITS: {self.mol_problem.active_orbitals*2}")
			print("Num params: ", self.num_ansatz_params)
			print("Runtime Budget: ", self.vqe_params.optim_runtime_budget)
			# raise Exception
			# print(extra_kwargs)


		t1_ = time.time()
		results, energies = opt.optimize()
		t2 = time.time()
		self.converged = results.success
		total_energy = results.fun + self.econst
		opt_params = results.x
		# print(f"optimized parameters: {opt_params}")
		runtimes = opt.timing_summary()
		if self.vqe_params.framework == "batched_gradients":
			eval_counts = opt.gather_eval_counts()
			runtimes["eval_counts"] = eval_counts
		runtimes['pre-process'] = t1 - t0
		runtimes["full_optim"] = t2 - t1_

		self.amp_threshold = self.vqe_params.amp_threshold

		final_spec = (num_qubit,opt_params,pauli_words,coeffs,block_ids,mol_problem.active_orbitals,mol_problem.active_mol_nelec[0],mol_problem.active_mol_nelec[1])

		self.energy_list = energies
		if verbose:
			state = np.array(cudaq.get_state(
							kernel,*final_spec), dtype=complex)
			if self.vqe_params.init_state == "UHF":
				state = fock_unitary_apply_from_U(state, self.U_so, dagger=False)
				# E_final_unrot = expectation_from_spinop_dense(state, self.spin_op_ham,n_qubits=self.qubits_num,state_is_big_endian=False, econst=self.econst)
				# print("Energy from rotated statevector with unrotated Ham: ", E_final_unrot)
			state = self.convert_state_big_endian(state)
		

			Statevector = BlockStatevector(statevector=state, mol_problem=self.mol_problem,ampl_eps=self.amp_threshold)
		else:
			Statevector = None
		t3 = time.time()
		runtimes["post_process"] = t3 - t2
		self.runtimes = runtimes

		if verbose:
			print(f"Converged: {self.converged}")
			print("PRE PROCESS Runtime: ",runtimes['pre-process'] )
			print("FULL OPTIMIZATION TIME: ", runtimes["full_optim"])
			print("POST-PROCESS Runtime: ", runtimes["post_process"])
			for k,v in self.runtimes.items():
				print(f"{k}: {v}")
			if self.vqe_params.framework == "batched_gradients":

				print("num_energy_evals (global):", eval_counts["total"])
				print("num_energy_evals_per_rank (global):", eval_counts["per_rank"])
		return Statevector	



	def run_adapt_ucc(self):
			start = time.perf_counter()
			runtime_budget = self.vqe_params.optim_runtime_budget
			if runtime_budget is None:
				runtime_budget = 12*3600
			runtimes = {}
			self.init_trial = None
			self.init_point = None
			comm = None
			rank = 0
			size = 1
			if self.mpi_info is not None:
					comm = self.mpi_info.comm
					rank = self.mpi_info.rank
					size = self.mpi_info.size

			mol_problem = self.mol_problem
			excitations, extra_kwargs = ansatz_type_to_kwargs("UCCSD")
			excitation_list = get_excitation_list(
					excitations=excitations,
					mol_problem=mol_problem,
					extra_kwargs=extra_kwargs,
			)

			pauli_words, coeffs, block_ids = collect_pauli_from_excitation_list(
					excitation_list=excitation_list,
					n_spin_orbitals=mol_problem.active_spin_orbitals,
			)

			self.amp_threshold = self.vqe_params.amp_threshold

			exp_vals_per_adapt = []
			spin_ham = self.spin_op_ham
			n_qubits = self.qubits_num

			pools = build_cudaq_operator_pool_dense(
					pauli_words=pauli_words, coeffs=coeffs, block_ids=block_ids
			)
			init_point = compute_pyscf_initial_point(
					method=self.vqe_params.init_method,
					mol_problem=mol_problem,
					active_spatial=mol_problem.active_spatial,
					excitation_list=excitation_list,
					UCC=True,
			)
			if self.vqe_params.verbose:
					print("Number of operator pool: ", len(pools))
					print("HF Energy: ", self.mol_problem.get_hf_energy())
					print("Adapt threshold: ", self.vqe_params.adapt_thresh)

			mod_pool, sign_pool = [], []
			for op_i in pools:
					words, coefs = [], []
					for term in op_i:
							coefs.append(float(term.evaluate_coefficient().real))
							words.append(term.get_pauli_word(n_qubits))  # pauli_word
					mod_pool.append(words)
					sign_pool.append(coefs)

			# Commutator-based gradient operators
			grad_op = adapt_commutator(pools, spin_ham)
			GradVecComputer = ADAPTGradVec(
					grad_op=grad_op, mpi_info=self.mpi_info, kernel=adapt_gradient
			)

			# Initial reference states (HF) — only works with mqpu backend for now
			states_futures = [
					cudaq.get_state_async(
							adapt_initial_state,
							n_qubits,
							mol_problem.active_orbitals,
							mol_problem.mol_nelec[0],
							mol_problem.mol_nelec[1],
							qpu_id=i,
					)
					for i in range(self.num_qpus)]

			threshold = self.vqe_params.adapt_thresh
			e_stop = 1e-5
			E_prev = 0.0

			# Growing ansatz bookkeeping
			theta_single, theta_double = [], []
			pool_single, pool_double = [], []
			coef_single, coef_double = [], []
			selected_pool = []

			optimize_eval_counts = {"total": 0, "per_rank": {}}
			grad_runtime = 0.0
			optimize_runtime = 0.0
			self.converged = True
			def add_eval_counts(dst, src):
					"""Accumulate eval-counts dicts of the shape:
					{"total": int, "per_rank": {rank: int, ...}}. No-op if src is None."""
					if not src:
							return
					dst["total"] += int(src.get("total", 0))
					per_dst = dst.setdefault("per_rank", {})
					for r, c in src.get("per_rank", {}).items():
							per_dst[r] = per_dst.get(r, 0) + int(c)
			step =  0
			while True:
					
					# 1) Compute gradient magnitudes (Allreduce inside get_full_grad_vec)
					t0 = time.time()
					states = [f.get() for f in states_futures]
					gradient_vec = GradVecComputer.get_full_grad_vec(states)
					t1 = time.time()
					dt = t1 - t0
					grad_runtime += dt
					if runtime_budget is not None:
						runtime_budget -= dt
					norm = np.linalg.norm(np.array(gradient_vec, dtype=float))

					if norm <= threshold:
							break

					# 2) Pick the largest component (same across ranks)
					idx = int(np.argmax(np.abs(gradient_vec)))
					max_grad = float(gradient_vec[idx])

					# The chosen pool at this step
					temp_pool = [mod_pool[idx]]
					temp_sign = [sign_pool[idx]]
					selected_pool += temp_pool

					# Split singles vs doubles by term count (2 words => single; else double)
					tot_single = 0
					tot_double = 0
					for p in temp_pool:
							if len(p) == 2:
									tot_single += 1
									pool_single.extend(p)
							else:
									tot_double += 1
									pool_double.extend(p)

					for cf in temp_sign:
							if len(cf) == 2:
									coef_single.extend(cf)
							else:
									coef_double.extend(cf)

					base_theta_val = init_point[idx]
					# 3) Grow theta with base init
					theta_single += [base_theta_val] * tot_single
					theta_double += [base_theta_val] * tot_double
					theta = theta_single + theta_double

					# 4) Optimize current ansatz
					adapt_spec = (
							THETA,
							n_qubits,
							mol_problem.active_orbitals,
							mol_problem.mol_nelec[0],
							mol_problem.mol_nelec[1],
							pool_single,
							coef_single,
							pool_double,
							coef_double,
					)
				
					if runtime_budget < 0:
						runtime_budget = None
					opt = make_optimizer(
							framework=self.vqe_params.framework,
							kernel=adapt_kernel,
							hamiltonian=self.spin_op_ham,
							kernel_arg_spec=adapt_spec,
							init_theta=theta,
							max_walltime=runtime_budget,
							econst=self.econst,
							epsilon=self.vqe_params.eps,
							optimizer_method=self.vqe_params.optimizer,
							ftol=self.vqe_params.ftol,
							gtol=self.vqe_params.gtol,
							mpi_info=self.mpi_info,
							verbose = False
					)

					t1 = time.time()
					results, energies = opt.optimize()
					t2 = time.time() 
					self.converged=results.success


						
					dt = t2-t1
					optimize_runtime += dt
					if runtime_budget is not None:
						runtime_budget -= dt

					# Synchronize optimized theta and energy from root --
					if self.mpi_info is None:
							theta_opt = np.asarray(results.x, float)
							e_opt_corr = float(results.fun)
					else:
							theta_opt = np.asarray(results.x, float) if rank == 0 else None
							theta_opt = comm.bcast(theta_opt, root=0)
							e_opt_corr = float(results.fun) if rank == 0 else None
							e_opt_corr = comm.bcast(e_opt_corr, root=0)
					e_opt_tot = e_opt_corr + self.econst

					local_eval_counts = opt.gather_eval_counts()
					add_eval_counts(optimize_eval_counts, local_eval_counts)

					if self.mpi_info is None or rank == 0:
							exp_vals_per_adapt.append(e_opt_tot)

					theta = theta_opt.tolist()
					theta_single = theta[: len(theta_single)]
					theta_double = theta[len(theta_single) :]

					if self.vqe_params.verbose and (self.mpi_info is None or rank == 0):
							print(f"Step {step}, Energy = {e_opt_tot}, Time  = {optimize_runtime+grad_runtime}, Converged = {self.converged}, GradNorm = {norm}, theta_len = {len(theta)}" )

					# Stop criterion (identical on all ranks after broadcast)
					dE = e_opt_corr - E_prev
					E_prev = e_opt_corr
					if self.vqe_params.verbose:
							print("dE: ", dE, "\n")
					if abs(dE) <= e_stop:
							if self.vqe_params.verbose:
									print("\nFinal Result (energy stop):\n")
							break
					if not self.converged:
						if self.vqe_params.verbose:
							print("Runtime Budget exceeded on optim")
						break 
					
					if runtime_budget is None:
						self.converged = False
						if self.vqe_params.verbose:
							print("Runtime Budget exceeded on grad")
						break
					states_futures = [cudaq.get_state_async(adapt_kernel,
									theta,
									n_qubits,
									mol_problem.active_orbitals,
									mol_problem.mol_nelec[0],
									mol_problem.mol_nelec[1],
									pool_single,
									coef_single,
									pool_double,
									coef_double, qpu_id = i) for i in range(self.num_qpus)]
					step += 1
			if self.vqe_params.verbose:
					# print("Full pool: ", selected_pool)
					print("total operators: ", len(selected_pool))
					print("Num Singles: ", len(pool_single))
					print("Num doubles: ", len(pool_double))
					for i in range(len(exp_vals_per_adapt)):
						print(f"ADAPT Iteration {i}: Energy =  {exp_vals_per_adapt[i]}")

			runtimes["optimize_eval_counts"] = optimize_eval_counts
			runtimes["optimize_runtime"] = optimize_runtime
			runtimes["grad_runtime"] = grad_runtime
			runtimes["grad_eval_counts"] = GradVecComputer.gather_eval_counts()

			self.runtimes = runtimes
			pauli_words = pool_single + pool_double
			self.two_qubit_count = get_ucc_two_qubit_gate_count(pauli_words)
			# ---- wrap up / projection ----
			if self.vqe_params.verbose:
					state = cudaq.get_state(
							adapt_kernel,
							theta,
							n_qubits,
							mol_problem.active_orbitals,
							mol_problem.mol_nelec[0],
							mol_problem.mol_nelec[1],
							pool_single,
							coef_single,
							pool_double,
							coef_double,
					)
					if self.vqe_params.verbose:
							self.num_ansatz_params = len(theta)

							# for i, energy in enumerate(exp_vals_per_adapt):
							# 		print(f"Energy: ADAPT Iteration {i}: {energy}")
							for k, v in self.runtimes.items():
									print(f"{k}:{v}")
					self.energy_list = exp_vals_per_adapt
					state_final = np.array(state, dtype=complex)
					state = self.convert_state_big_endian(state_final)
					return BlockStatevector(
							statevector=state,
							mol_problem=self.mol_problem,
							ampl_eps=self.amp_threshold,
					)
			else:
					return None
	
	def rotate_ham(self):
		_, X_a, X_b = self.mol_problem.get_uhf()
		U_a = complete_unitary_from_occupied(X_a)   # (10,10)
		U_b = complete_unitary_from_occupied(X_b)   # (10,10)
		U_so = block_diag(U_a, U_b)    
		obi_so = np.asarray(self.obi, dtype=np.complex128)
		tbi_so = np.asarray(self.tbi, dtype=np.complex128)
		obi_uhf_so = rotate_1e_spinorb(obi_so, U_so)
		tbi_uhf_so = rotate_2e_spinorb_chemist(tbi_so, U_so)
		spin_op_ham_uhf = jordan_wigner_fermion(obi_uhf_so, tbi_uhf_so, ecore=0.0, tolerance=1e-15)
		return spin_op_ham_uhf, U_so
	    



	# convert little endian to big endian for statevector obtained from cudaq.get_state, so that it can be compatible with expectation value calculation and other post-processing that assumes big endian ordering. This is because cudaq.get_state returns statevector in little endian format, where the basis states are ordered as |q_{n-1} q_{n-2} ... q_0>, while many quantum chemistry tools (including our post-processing) assume big endian format |q_0 q_1 ... q_{n-1}>.
	def convert_state_big_endian(self,state_little_endian):

			state_big_endian = 0. * state_little_endian

			n_qubits = int(np.log2(state_big_endian.size))
			for j, val in enumerate(state_little_endian):
					little_endian_pos = np.binary_repr(j, n_qubits)
					big_endian_pos = little_endian_pos[::-1]
					int_big_endian_pos = int(big_endian_pos, 2)
					state_big_endian[int_big_endian_pos] = state_little_endian[j]

			return state_big_endian
	

def complete_unitary_from_occupied(X_occ: np.ndarray) -> np.ndarray:
    norb, nocc = X_occ.shape

    # Projector onto complement
    P_perp = np.eye(norb) - X_occ @ X_occ.conj().T

    # Apply projector to identity basis
    Y = P_perp @ np.eye(norb)

    # Remove first nocc columns (these lie in occupied space)
    Y = Y[:, nocc:]

    # Orthonormalize complement
    Q2, _ = np.linalg.qr(Y)

    # Assemble full unitary
    U = np.hstack([X_occ, Q2])

    return U
def rotate_1e_spinorb(h1_so: np.ndarray, U_so: np.ndarray) -> np.ndarray:
    # h' = U† h U
    return U_so.conj().T @ h1_so @ U_so


def rotate_2e_spinorb_chemist(eri_so: np.ndarray, U_so: np.ndarray) -> np.ndarray:
    """
    eri_so in chemist notation: eri[p,q,r,s] = (pq|rs) in a SPIN-ORBITAL basis.
    Rotation: eri'_{pqrs} = sum_{abcd} U*_{a p} U*_{b q} U_{c r} U_{d s} eri_{a b c d}
    """
    Uc = U_so.conj()
    return np.einsum('ap,bq,cr,ds,abcd->pqrs', Uc, Uc, U_so, U_so, eri_so, optimize=True)

def block_diag(A, B):
    """Simple block diagonal for (n×n) A and B -> (2n×2n)."""
    n = A.shape[0]
    out = np.zeros((2*n, 2*n), dtype=np.complex128)
    out[:n, :n] = A
    out[n:, n:] = B
    return out

