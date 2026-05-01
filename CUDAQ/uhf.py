import numpy as np
from ffsim.linalg import givens_decomposition


def fock_unitary_apply_from_U(state: np.ndarray, U: np.ndarray, dagger: bool = False) -> np.ndarray:


    U = np.asarray(U, dtype=np.complex128)
    n_modes = U.shape[0]
    assert U.shape == (n_modes, n_modes)
    assert state.size == (1 << n_modes)

    psi = np.array(state, dtype=np.complex128, copy=True)
    Umode = U.conj().T if dagger else U

    rots, diag_phases = givens_decomposition(Umode)

    parity = _precompute_parity(psi.size)  # once per call

    for (c, s, i, j) in rots:
        _apply_fermionic_givens_jw_fast(psi, int(i), int(j), complex(c), complex(s), parity)

    diag_phases = np.asarray(diag_phases).ravel().astype(np.complex128)
    phases = np.angle(diag_phases)
    _apply_diag_phases_jw_fast(psi, phases)

    return psi


def _precompute_parity(dim: int) -> np.ndarray:
    """parity[b] = popcount(b) mod 2"""
    parity = np.zeros(dim, dtype=np.uint8)
    for b in range(1, dim):
        parity[b] = parity[b >> 1] ^ (b & 1)
    return parity

def _indices_i1_j0(n_modes: int, i: int, j: int) -> np.ndarray:
    """
    Return all basis indices b with bit i=1, bit j=0 (i<j).
    Vectorized construction by iterating over the "other bits" space.
    """
    assert i < j
    dim = 1 << n_modes
    mi = 1 << i
    mj = 1 << j

    # We build indices by sweeping blocks of size 2^(j+1),
    # within each block, j=0 half then j=1 half. We only take j=0 half.
    block = 1 << (j + 1)
    half  = 1 << j

    # Inside the j=0 half, bit i toggles every 2^i. We only want i=1 portion.
    stride = 1 << (i + 1)
    on     = 1 << i

    idxs = []
    for base in range(0, dim, block):
        # j=0 half: [base, base+half)
        start = base
        end = base + half
        # within this half, take i=1 segments
        for s in range(start, end, stride):
            idxs.append(np.arange(s + on, min(s + stride, end), dtype=np.int64))

    return np.concatenate(idxs) if idxs else np.empty(0, dtype=np.int64)

def _apply_fermionic_givens_jw_fast(psi: np.ndarray, i: int, j: int, c: complex, s: complex,
                                   parity: np.ndarray):
    """
    Vectorized 2-mode fermionic Givens update in JW ordering.
    parity is parity[b] = popcount(b) mod 2 precomputed for dim.
    """
    if i == j:
        return
    if i > j:
        i, j = j, i
        c = np.conj(c)
        s = -s

    n_modes = int(np.log2(psi.size))
    dim = psi.size
    mi = 1 << i
    mj = 1 << j
    mid_mask = ((1 << j) - 1) ^ ((1 << (i + 1)) - 1)

    idx10 = _indices_i1_j0(n_modes, i, j)
    if idx10.size == 0:
        return
    idx01 = idx10 ^ mi ^ mj

    # JW sign from parity of bits between i and j
    signs = 1.0 - 2.0 * parity[idx10 & mid_mask].astype(np.float64)  # 0->+1, 1->-1

    a10 = psi[idx10]
    a01 = psi[idx01]

    # Update (use temporaries of size ~dim/4, not huge)
    psi[idx10] = c * a10 + (s * signs) * a01
    psi[idx01] = -(np.conj(s) * signs) * a10 + np.conj(c) * a01

def _apply_diag_phases_jw_fast(psi: np.ndarray, phases: np.ndarray):
    """
    Apply product_p f[p]^{n_p} to each basis amplitude using DP recurrence.
    O(dim) time, no per-b popcount loop.
    """
    phases = np.asarray(phases)
    n_modes = phases.shape[0]
    dim = psi.size
    f = np.exp(1j * phases)

    # phase_factor[b] = product of f[p] for set bits p in b
    phase_factor = np.empty(dim, dtype=np.complex128)
    phase_factor[0] = 1.0 + 0.0j
    for b in range(1, dim):
        lsb = b & -b
        p = lsb.bit_length() - 1
        phase_factor[b] = phase_factor[b ^ lsb] * f[p]

    psi *= phase_factor

