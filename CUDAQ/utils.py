
import numpy as np
from pyscf import mp, cc


def order_excitations_by_init(excitation_list, init_point, descending=True):
    """
    Reorder excitations by |init_point|.
    """


    init_point = np.asarray(init_point, dtype=float)
    if len(excitation_list) != init_point.size:
        raise ValueError(f"length mismatch: {len(excitation_list)=} vs {init_point.size=}")

    mags = np.abs(init_point)
    mags = np.nan_to_num(mags, nan=0.0)  # treat NaNs as 0 for ordering

    # argsort ascending; flip if descending
    perm = np.argsort(mags, kind="stable")
    if descending:
        perm = perm[::-1]

    excitation_list_sorted = [excitation_list[i] for i in perm]
    init_point_sorted = init_point[perm].copy()
    return excitation_list_sorted, init_point_sorted, perm



def compute_pyscf_initial_point(method, mol_problem, active_spatial, excitation_list, UCC,noise_sigma=0.0):

    mf = mol_problem.mf
    num_parameters = len(excitation_list)
    init_methods = {"zeros","ones","normal","random","mp2","ccsd", "rccsd"}
    if method not in init_methods:
        raise ValueError(f"method must be one of {init_methods}, got {method!r}")   
    if UCC == True:
        base_length = len(excitation_list)
    else:
        base_length = num_parameters
    if method == 'zeros':
        return np.zeros(base_length)
        
    if method == 'ones':
        return np.ones(base_length)
    if method == 'normal':
        return np.random.normal(0.0, noise_sigma, size=base_length)
    if method == 'random':
        return np.random.uniform(-np.pi, np.pi, base_length)


    # --- for mp2 / ccsd we build the MP amplitudes ---
    nspatial = mf.mo_coeff.shape[1]
    occ_full = np.where(mf.mo_occ > 0)[0]
    frozen   = [i for i in range(nspatial) if i not in active_spatial]

    if method == 'mp2':
        mp2 = mp.MP2(mf, frozen=frozen)
        mp2.kernel()
        t2_act, t1_act = mp2.t2, None
    elif method ==  "ccsd":  # ccsd
        return ccsd_amplitudes_for_excitations(mol_problem,excitation_list, noise_sigma)
    elif method == "rccsd":
        ccsd = cc.CCSD(mf, frozen=frozen)
        ccsd.kernel()
        t1_act, t2_act = ccsd.t1, ccsd.t2

    # map global→active indices
    active_occ = [i for i in active_spatial if i in occ_full]
    active_vir = [i for i in active_spatial if i not in occ_full]
    map_occ = {g:i for i,g in enumerate(active_occ)}
    map_vir = {g:i for i,g in enumerate(active_vir)}

    n_act = len(active_spatial)
    base_init = []
    
    for occ_spin, virt_spin in excitation_list:
        # singles
        if len(occ_spin) == 1:
            if method == 'rccsd':
                i_s, a_s = occ_spin[0] % n_act, virt_spin[0] % n_act
                g_i, g_a = active_spatial[i_s], active_spatial[a_s]
                amp = float(t1_act[map_occ[g_i], map_vir[g_a]]) if (g_i in map_occ and g_a in map_vir) else 0.0
            else:
                amp = 0.0
            base_init.append(amp)
            continue

        # doubles
        i_s, j_s = [x % n_act for x in occ_spin]
        a_s, b_s = [x % n_act for x in virt_spin]
        g_i, g_j = active_spatial[i_s], active_spatial[j_s]
        g_a, g_b = active_spatial[a_s], active_spatial[b_s]
        try:
            oi, oj = map_occ[g_i], map_occ[g_j]
            va, vb = map_vir[g_a], map_vir[g_b]
            amp = float(t2_act[oi, oj, va, vb])
        except KeyError:
            amp = 0.0
        base_init.append(amp)

    base_init = np.array(base_init, dtype=float)

    # #  sprinkle small Gaussian noise on every zero entry ---
    # # new: only entries exactly equal to 0.0 get noise
    # print(f"Pre-noise {method} point",base_init)
    # zero_mask = (base_init == 0.0)
    # base_init[zero_mask] = np.random.normal(0.0, noise_sigma, size=zero_mask.sum())
    # base_init += np.random.normal(0.0, noise_sigma, size=base_init.shape)

    # base_init[zero_mask] = np.random.uniform(-np.pi,np.pi,size = zero_mask.sum())
    # base_init_point = [float(base_init[i]) for i in range((len(base_init)))]

    return base_init


def ccsd_amplitudes_for_excitations(mol_problem, excitation_list, noise_sigma=0.0):
    """
    Compute CCSD amplitudes aligned with a Qiskit-style excitation_list,
    following the same choices as in your ccsd.py:
      • UCCSD on top of RHF (pyscf.cc.UCCSD)
      • Freeze everything NOT in active_spatial
      • Spin blocks: t1 = (alpha, beta), t2 = (aa, ab, bb)
      • Occupied/virtual determined from mf.mo_occ (spatial occupancy)
    
    Parameters
    ----------
    mol_problem : System.BuildMoleculeProblem object

    excitation_list : list[tuple[tuple[int,...], tuple[int,...]]]
        UCC excitation list written in *active spin-orbital* indices:
          0..n_act-1          => alpha on active_spatial[0..n_act-1]
          n_act..2*n_act-1    => beta  on active_spatial[0..n_act-1]
        Examples:
          ((0,), (3,))                  -> alpha single
          ((10,), (11,))                -> beta single (if n_act=10)
          ((0,10), (1,11))              -> alpha-beta double (ab)
    
    Returns
    -------
    np.ndarray
        1D array of amplitudes in the same order as excitation_list.
    """
    # -------- active-space & freezing (same semantics as ccsd.py) --------


    mycc = mol_problem.build_ccsd(restricted=False)
    mf = mol_problem.mf
    active_spatial = mol_problem.active_spatial
    n_act = len(mol_problem.active_spatial)


    t1a, t1b = mycc.t1
    t2aa, t2ab, t2bb = mycc.t2

    # Partition occupied / virtual (spatial) using RHF mo_occ
    occ_full = set(np.where(mf.mo_occ > 1e-8)[0])
    active_occ = [g for g in active_spatial if g in occ_full]
    active_vir = [g for g in active_spatial if g not in occ_full]

    # Maps from global spatial index -> active occ/vir index
    map_occ = {g: i for i, g in enumerate(active_occ)}
    map_vir = {g: i for i, g in enumerate(active_vir)}
    nocc_act = len(active_occ)
    nvir_act = len(active_vir)

    # -------- helpers to map active spin-orbital index -> (spin, global spatial) ------
    def so_to_spin_global(so_idx):
        """Interpret excitation_list index in active spin-orbital convention."""
        if not (0 <= so_idx < 2 * n_act):
            return None, None
        if so_idx < n_act:
            return 'a', active_spatial[so_idx]
        else:
            return 'b', active_spatial[so_idx - n_act]

    # -------- amplitude accessors (match aa/ab/bb semantics in ccsd.py) --------
    def get_t1(spin, g_i, g_a):
        # singles must preserve spin
        if g_i not in map_occ or g_a not in map_vir:
            return 0.0
        oi = map_occ[g_i]
        va = map_vir[g_a]
        if not (0 <= oi < nocc_act and 0 <= va < nvir_act):
            return 0.0
        return float(t1a[oi, va] if spin == 'a' else t1b[oi, va])

    def get_t2_aa(g_i, g_j, g_a, g_b):
        # canonicalize (i<j, a<b) and track sign under swaps (same as ccsd.py behavior)
        if (g_i not in map_occ) or (g_j not in map_occ) or (g_a not in map_vir) or (g_b not in map_vir):
            return 0.0
        oi, oj = map_occ[g_i], map_occ[g_j]
        va, vb = map_vir[g_a], map_vir[g_b]
        if oi == oj or va == vb:
            return 0.0
        sign = 1
        if oi > oj:
            oi, oj = oj, oi
            sign *= -1
        if va > vb:
            va, vb = vb, va
            sign *= -1
        if not (0 <= oi < nocc_act and 0 <= oj < nocc_act and 0 <= va < nvir_act and 0 <= vb < nvir_act):
            return 0.0
        return sign * float(t2aa[oi, oj, va, vb])

    def get_t2_bb(g_i, g_j, g_a, g_b):
        # identical logic to aa for beta-beta block
        if (g_i not in map_occ) or (g_j not in map_occ) or (g_a not in map_vir) or (g_b not in map_vir):
            return 0.0
        oi, oj = map_occ[g_i], map_occ[g_j]
        va, vb = map_vir[g_a], map_vir[g_b]
        if oi == oj or va == vb:
            return 0.0
        sign = 1
        if oi > oj:
            oi, oj = oj, oi
            sign *= -1
        if va > vb:
            va, vb = vb, va
            sign *= -1
        if not (0 <= oi < nocc_act and 0 <= oj < nocc_act and 0 <= va < nvir_act and 0 <= vb < nvir_act):
            return 0.0
        return sign * float(t2bb[oi, oj, va, vb])

    def get_t2_ab(g_i_alpha, g_j_beta, g_a_alpha, g_b_beta):
        # mixed-spin block: no (i<j)/(a<b) canonicalization across different spins
        if (g_i_alpha not in map_occ) or (g_j_beta not in map_occ) or (g_a_alpha not in map_vir) or (g_b_beta not in map_vir):
            return 0.0
        oi = map_occ[g_i_alpha]
        oj = map_occ[g_j_beta]
        va = map_vir[g_a_alpha]
        vb = map_vir[g_b_beta]
        if not (0 <= oi < nocc_act and 0 <= oj < nocc_act and 0 <= va < nvir_act and 0 <= vb < nvir_act):
            return 0.0
        return float(t2ab[oi, oj, va, vb])

    # -------- build amplitudes in exactly the order of excitation_list --------
    amps = []
    for occ_tuple, vir_tuple in excitation_list:
        # singles
        if len(occ_tuple) == 1 and len(vir_tuple) == 1:
            s_o, g_i = so_to_spin_global(occ_tuple[0])
            s_v, g_a = so_to_spin_global(vir_tuple[0])
            if None in (s_o, g_i, s_v, g_a) or s_o != s_v:
                amps.append(0.0)
            else:
                amps.append(get_t1(s_o, g_i, g_a))
            continue

        # doubles
        if len(occ_tuple) == 2 and len(vir_tuple) == 2:
            s_i, g_i = so_to_spin_global(occ_tuple[0])
            s_j, g_j = so_to_spin_global(occ_tuple[1])
            s_a, g_a = so_to_spin_global(vir_tuple[0])
            s_b, g_b = so_to_spin_global(vir_tuple[1])
            if None in (s_i, g_i, s_j, g_j, s_a, g_a, s_b, g_b):
                amps.append(0.0)
                continue

            # aa
            if s_i == s_j == s_a == s_b == 'a':
                amps.append(get_t2_aa(g_i, g_j, g_a, g_b))
                continue
            # bb
            if s_i == s_j == s_a == s_b == 'b':
                amps.append(get_t2_bb(g_i, g_j, g_a, g_b))
                continue
            # ab (one alpha, one beta on each side)
            if {'a','b'} == {s_i, s_j} and {'a','b'} == {s_a, s_b}:
                # put alpha first for occ, and alpha first for vir before calling ab accessor
                if s_i == 'a':
                    g_i_a, g_j_b = g_i, g_j
                else:
                    g_i_a, g_j_b = g_j, g_i
                if s_a == 'a':
                    g_a_a, g_b_b = g_a, g_b
                else:
                    g_a_a, g_b_b = g_b, g_a
                amps.append(get_t2_ab(g_i_a, g_j_b, g_a_a, g_b_b))
                continue

            # anything else not recognized
            amps.append(0.0)
            continue

        # not a single or double
        amps.append(0.0)
    amps = np.array(amps, dtype=float)
    if noise_sigma > 0:
        amps += np.random.normal(0.0, noise_sigma, size=amps.shape)




    return amps





def sparse_pauli_op_to_spinop(spo):
    """
    Convert qiskit.quantum_info.SparsePauliOp -> cudaq.SpinOperator
    using the spin builder API. Preserves constants and complex coeffs.
    """
    from qiskit.quantum_info import SparsePauliOp
    import cudaq

    if not isinstance(spo, SparsePauliOp):
        raise TypeError("Expected a qiskit.quantum_info.SparsePauliOp")

    # Combine duplicates / drop tiny terms
    spo = spo.simplify()

    H = cudaq.SpinOperator()
    n = spo.num_qubits

    for label, coeff in spo.to_list():
        c = complex(coeff)

        # Constant term: all identities
        if all(ch == 'I' for ch in label):
            H += c  # CUDA-Q supports scalar shifts
            continue

        # Build product like X0 * Y1 * Z3 using little-endian reversal
        term = None
        for q, p in enumerate(reversed(label)):  # label[0] -> qubit n-1
            if p == 'I':
                continue
            op = {'X': cudaq.spin.x, 'Y': cudaq.spin.y, 'Z': cudaq.spin.z}[p](q)
            term = op if term is None else term * op

        # Multiply by coefficient and add
        H = H + (c * term)

    return H


def expectation_from_spinop_dense(state: np.ndarray,
                                  spin_op,
                                  n_qubits: int,
                                  *,
                                  state_is_big_endian: bool = False,
                                  econst: float = 0.0):
    # Get dense H in CUDA-Q's (little-endian) ordering
    try:
        H = spin_op.to_matrix()
    except TypeError:
        H = spin_op.to_matrix(n_qubits)

    # Make sure state matches H's ordering
    if state_is_big_endian:
        state = to_little_endian_state(state, n_qubits)

    # Normalize just in case (projection/rounding can drift)
    norm = np.vdot(state, state).real
    if not np.isclose(norm, 1.0):
        state = state / np.sqrt(norm)

    E = np.vdot(state, H @ state).real + econst
    return E

def _reverse_bits_indices(n_qubits: int):
    dim = 1 << n_qubits
    idx = np.arange(dim, dtype=np.uint32)
    rev = np.zeros_like(idx)
    for i in range(dim):
        x = i; r = 0
        for _ in range(n_qubits):
            r = (r << 1) | (x & 1)
            x >>= 1
        rev[i] = r
    return rev

def to_little_endian_state(state_big: np.ndarray, n_qubits: int) -> np.ndarray:
    """Convert a big-endian statevector to CUDA-Q's little-endian ordering."""
    perm = _reverse_bits_indices(n_qubits)
    return state_big[perm]

    
