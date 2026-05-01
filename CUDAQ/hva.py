from qiskit.quantum_info import SparsePauliOp
from CUDAQ.utils import sparse_pauli_op_to_spinop
import cudaq

def build_HVA(mol_problem, nreps=1):

  ham_spo=mol_problem.build_qiskit_ham()
  ham_group_commute = ham_spo.group_commuting()
  pauli_words = []
  coeffs = []
  block_ids = []

  n = len(ham_group_commute)
  for j in range(nreps):
    for i in range(n):
      spin_op = sparse_pauli_op_to_spinop(ham_group_commute[i])
      block_op = cudaq.spin.canonicalized(spin_op, set(range(mol_problem.active_spin_orbitals)))
      for term in block_op:
        pw = cudaq.pauli_word(term.get_pauli_word())
        coeff = float(term.evaluate_coefficient().real)
        pauli_words.append(pw)
        coeffs.append(coeff)
        block_ids.append(i+j*n)
  
  return pauli_words, coeffs, block_ids


@cudaq.kernel
def hva_circuit(qubit_num: int, 
                theta: list[float],
                pauli_words: list[cudaq.pauli_word],
                coeffs: list[float],
                block_ids: list[int],
                n_spat: int,
                n_alpha: int,
                n_beta: int):

  q = cudaq.qvector(qubit_num)  # Expect qubit_num == 2 * n_spat

  # --- Hartree–Fock prep (RHF), consistent with q = N-1-j mapping ---
  # alpha occupied spin-orbitals: j = 0..n_alpha-1  -> qubits q = N-1 - j
  for i in range(n_alpha):
      x(q[i])
  # beta occupied spin-orbitals:  j = n_spat..n_spat+n_beta-1 -> q = N-1 - j = n_spat - 1 - (j - n_spat)
  for j in range(n_beta):
      x(q[n_spat +j])

  # --- UCC exponentials ---
  for i in range(len(pauli_words)):
    pidx  = block_ids[i]                 # which parameter controls this term
    exp_pauli(-theta[pidx] * coeffs[i], q, pauli_words[i])    

