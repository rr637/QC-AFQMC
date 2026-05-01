import cudaq
import time
from typing import List, Dict
import cudaq
from cudaq import spin
import numpy as np
try:
    from mpi4py import MPI  # optional; only needed if mpi is used
except Exception:
    MPI = None

class ADAPTGradVec:
    def __init__(self,grad_op,mpi_info,kernel):
        self.grad_op = grad_op
        self.mpi_info =  mpi_info
        self.kernel =  kernel
        self.num_qpus = cudaq.get_target().num_qpus()
        self.total_energy_evals = 0
        self.get_states_evals  = 0
        self.get_states_runtime = 0
        if self.mpi_info is None:
        # print("IN GRADIENT CALCULTAION - NUM-QPU: ", self.num_qpus)
            self.energy_evals = {r:0 for r in range(1)}
            self.rank = 0
            self.size = 1
        else:
            self.energy_evals = {r:0 for r in range(self.mpi_info.size)}
            self.rank = self.mpi_info.rank
            self.size = self.mpi_info.size
    def gather_eval_counts(self):
        if self.mpi_info is None:
            return {"total": self.total_energy_evals, "per_rank": {0: self.total_energy_evals}}

        comm, rank, size = self.mpi_info.comm, self.mpi_info.rank, self.mpi_info.size
        # send just the scalar count from each rank
        local_count = self.energy_evals[rank]
        counts = comm.gather(local_count, root=0)
        total_local = self.total_energy_evals
        totals = comm.gather(total_local, root=0)

        if rank == 0:
            per_rank = {r: counts[r] for r in range(size)}
            return {"total": sum(totals), "per_rank": per_rank}
        else:
            return None
    def _my_indices(self, n):
        """Shard indices across ranks; in serial, take all."""
        if self.mpi_info is None:
            return range(n)
        r, p = self.mpi_info.rank, self.mpi_info.size
        return range(r, n, p)

    def batched_gradient(self, states, indices=None):


        if indices is None:
            indices = range(len(self.grad_op))

        qid = 0
        grad_vec = np.zeros(len(self.grad_op), dtype=float)
        futures  = []
        idxs = []
        indices = list(indices)  # in case it's a range/ndarray
        t0 = time.time()
        for i in indices:

            f = cudaq.observe_async(
                self.kernel, self.grad_op[i], states[qid % self.num_qpus],
                qpu_id=(qid % self.num_qpus)
            )
            self.energy_evals[self.rank] += 1
            qid += 1
            futures.append(f)
            idxs.append(i)
        # print("grad A: ", time.time()-t0)
            # Collect this chunk (handles group size < num_qpus)
        for idx, f in zip(idxs, futures):
            grad_vec[idx] = f.get().expectation()
        # print("grad B: ", time.time() - t0)
        self.total_energy_evals += len(futures)

        return grad_vec
    def get_full_grad_vec(self, grad_spec):
        if self.mpi_info is None:
            return self.batched_gradient(grad_spec)

        comm = self.mpi_info.comm
        g_local = self.batched_gradient(grad_spec, indices=self._my_indices(len(self.grad_op)))

        # everyone gets the summed full vector
        g_full = np.zeros_like(g_local, dtype=g_local.dtype)
        comm.Allreduce(g_local, g_full, op=MPI.SUM)
        return g_full


    



def _spin_from_dense_label(label: str, *, reverse: bool = False) -> cudaq.SpinOperator:
    """
    Convert a dense 'I/X/Y/Z' string (one char per qubit) into a cudaq.SpinOperator.
    Example: 'YXII' -> spin.y(0) * spin.x(1)  (if reverse=False)
    Set reverse=True if your bit order is opposite to what you want.
    """
    s = label.replace(' ', '')
    if reverse:
        s = s[::-1]

    op = 1.0  # scalar identity; multiplying by spin.* yields SpinOperator
    for q, ch in enumerate(s):
        c = ch.upper()
        if c == 'I':
            continue
        elif c == 'X':
            op = op * spin.x(q)
        elif c == 'Y':
            op = op * spin.y(q)
        elif c == 'Z':
            op = op * spin.z(q)
        else:
            raise ValueError(f"Unexpected Pauli char '{ch}' in label '{label}'")
    return op

def build_cudaq_operator_pool_dense(
    pauli_words: List[str],
    coeffs: List[complex],
    block_ids: List[int],
    *,
    reverse_labels: bool = False,   # flip endianness if needed
    drop_zero_tol: float = 0.0,     # prune tiny |coeff| terms
    realify_tol: float = 1e-12      # cast ~real coeffs to float
) -> List[cudaq.SpinOperator]:
    """
    Returns a list of cudaq.SpinOperator, one per unique block_id (sorted).
    """


    # Accumulate sum per block id
    acc: Dict[int, cudaq.SpinOperator] = {}

    for word, c, bid in zip(pauli_words, coeffs, block_ids):
        if drop_zero_tol and abs(c) < drop_zero_tol:
            continue
        # Make nearly-real coefficients exactly real for nicer printing/Hermiticity
        if isinstance(c, complex) and abs(c.imag) < realify_tol:
            c = float(c.real)

        term_op = _spin_from_dense_label(word, reverse=reverse_labels)
        term_op = c * term_op  # scale

        if bid in acc:
            acc[bid] = acc[bid] + term_op
        else:
            acc[bid] = term_op

    return [acc[k] for k in sorted(acc.keys())]







def adapt_commutator(pools, ham):
    com_op = []
    
    for i in range(len(pools)):
        # We add the imaginary number that we excluded when generating the operator pool.
        op = 1j * pools[i]
        
        com_op.append(ham * op - op * ham)
         
    return com_op
@cudaq.kernel
def adapt_initial_state(n_qubits:int, n_spat: int,
                n_alpha: int,
                n_beta: int):
    
    q = cudaq.qvector(n_qubits)  # Expect qubit_num == 2 * n_spat

    # --- Hartree–Fock prep (RHF), consistent with q = N-1-j mapping ---
    # alpha occupied spin-orbitals: j = 0..n_alpha-1  -> qubits q = N-1 - j
    for i in range(n_alpha):
        x(q[i])
    # beta occupied spin-orbitals:  j = n_spat..n_spat+n_beta-1 -> q = N-1 - j = n_spat - 1 - (j - n_spat)
    for j in range(n_beta):
        x(q[n_spat +j])

@cudaq.kernel
def adapt_gradient(state:cudaq.State):
    q = cudaq.qvector(state)


@cudaq.kernel
def adapt_kernel(theta: list[float], qubits_num: int, n_spat: int,n_alpha: int,n_beta: int, pool_single: list[cudaq.pauli_word], 
           coef_single: list[float], pool_double: list[cudaq.pauli_word], coef_double: list[float]):
    q = cudaq.qvector(qubits_num)
    
    # for i in range(nelectrons):
    #     x(q[i])
    for i in range(n_alpha):
        x(q[i])
    # beta occupied spin-orbitals:  j = n_spat..n_spat+n_beta-1 -> q = N-1 - j = n_spat - 1 - (j - n_spat)
    for j in range(n_beta):
        x(q[n_spat +j])
    
    count=0
    for  i in range(0, len(coef_single), 2):
        exp_pauli(-coef_single[i] * theta[count], q, pool_single[i])
        exp_pauli(-coef_single[i+1] * theta[count], q, pool_single[i+1])
        count+=1

    for i in range(0, len(coef_double), 8):
        exp_pauli(-coef_double[i] * theta[count], q, pool_double[i])
        exp_pauli(-coef_double[i+1] * theta[count], q, pool_double[i+1])
        exp_pauli(-coef_double[i+2] * theta[count], q, pool_double[i+2])
        exp_pauli(-coef_double[i+3] * theta[count], q, pool_double[i+3])
        exp_pauli(-coef_double[i+4] * theta[count], q, pool_double[i+4])
        exp_pauli(-coef_double[i+5] * theta[count], q, pool_double[i+5])
        exp_pauli(-coef_double[i+6] * theta[count], q, pool_double[i+6])
        exp_pauli(-coef_double[i+7] * theta[count], q, pool_double[i+7])
        count+=1