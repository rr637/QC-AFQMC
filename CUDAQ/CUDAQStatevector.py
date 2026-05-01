
import numpy as np
from pyscf.fci import cistring

class BlockStatevector:
    """
    Build an ipie-style multi-determinant trial from a BIG-ENDIAN statevector.

    Assumptions
    ----------
    - statevector : 1D complex numpy array of length 2**n_qubits (BIG-ENDIAN).
      Big-endian here means index j corresponds to the bitstring with the MSB on
      the left; to map to qubit indices (q=0 is LSB), we flip the bits first.
    - Jordan–Wigner (block spin ordering): qubits 0..(norb-1) = alpha,
      qubits norb..(2*norb-1) = beta, where norb = n_spin_orbitals // 2.
    """

    def __init__(self, statevector: np.ndarray, mol_problem, ampl_eps: float | None = None,vectorized=True):
        self.mol_problem = mol_problem
        self.statevector = np.asarray(statevector, dtype=np.complex128)

        self.n_spin_orbitals = self.mol_problem.active_spin_orbitals     # = 2 * norb
        self.n_alpha = self.mol_problem.active_mol_nelec[0]
        self.n_beta  = self.mol_problem.active_mol_nelec[1]

        self.coeffs: list[np.complex128] = []
        self.occ_a:  list[np.ndarray]    = []
        self.occ_b:  list[np.ndarray]    = []
        self.vectorized = vectorized
        if self.vectorized:
            self.__compute_wavefunction2(ampl_eps)
        else:
            self.__compute_wavefunction(ampl_eps)
    def getIPIEWavefunction(self):
        """Return (coeffs, occ_alpha_list, occ_beta_list)."""
        return self.wavefunction

    # -------- internals --------

    @staticmethod
    def _flip_bits_to_little_endian(j: int, n_qubits: int) -> int:
        """Convert big-endian index j into a little-endian integer mask."""
        return int(f"{j:0{n_qubits}b}"[::-1], 2)

    def _decode_occ_block_spin(self, j_le: int, n_qubits: int):
        """Decode occupations from a LITTLE-ENDIAN integer j_le (block spin JW)."""
        norb = n_qubits // 2
        a_occ, b_occ = [], []
        # qubit q is set iff (j_le >> q) & 1 == 1
        for q in range(n_qubits):
            if (j_le >> q) & 1:
                if q < norb:
                    a_occ.append(q)           # alpha orbital index
                else:
                    b_occ.append(q - norb)    # beta orbital index
        return a_occ, b_occ

    def __compute_wavefunction(self, eps: float | None):
        psi = self.statevector
        n_dim = psi.size
        n_qubits = int(round(np.log2(n_dim)))
        assert (1 << n_qubits) == n_dim, "Statevector length must be a power of 2."
        assert n_qubits == self.n_spin_orbitals, \
            f"Qubit count ({n_qubits}) != spin-orbital count ({self.n_spin_orbitals})."

        coeffs, occ_a, occ_b = [], [], []

        for j, amp in enumerate(psi):
            if eps is not None and abs(amp) < eps:
                continue

            # input is BIG-ENDIAN → flip to little-endian integer mask
            j_le = self._flip_bits_to_little_endian(j, n_qubits)
            a_list, b_list = self._decode_occ_block_spin(j_le, n_qubits)

            if len(a_list) == self.n_alpha and len(b_list) == self.n_beta:
                coeffs.append(amp)                                    # no extra phase
                occ_a.append(np.asarray(sorted(a_list), dtype=np.int32))
                occ_b.append(np.asarray(sorted(b_list), dtype=np.int32))

        if not coeffs:
            raise ValueError(
                "No determinants selected (check electron counts, threshold, or spin-orbital mapping)."
            )

        coeffs = np.asarray(coeffs, dtype=np.complex128)
        order  = np.argsort(-np.abs(coeffs))  # sort by magnitude, descending

        coeffs = coeffs[order]
        occ_a  = [occ_a[i] for i in order]
        occ_b  = [occ_b[i] for i in order]

        # normalize the selected subspace coefficients
        coeffs = coeffs / np.linalg.norm(coeffs)

        self.coeffs = coeffs
        self.occ_a  = occ_a
        self.occ_b  = occ_b
        self.wavefunction = (self.coeffs, self.occ_a, self.occ_b)
    


    def __compute_wavefunction2(self, eps: float | None):
        psi = self.statevector
        n_dim = psi.size
        n_qubits = int(round(np.log2(n_dim)))
        assert (1 << n_qubits) == n_dim, "Statevector length must be a power of 2."
        assert n_qubits == self.n_spin_orbitals, \
            f"Qubit count ({n_qubits}) != spin-orbital count ({self.n_spin_orbitals})."

        norb = n_qubits // 2
        n_alpha, n_beta = self.n_alpha, self.n_beta

        # Optional amplitude mask (pre-filter)
        if eps is not None:
            amp_mask = np.abs(psi) >= eps
            if not np.any(amp_mask):
                raise ValueError("No amplitudes above threshold eps.")
            idx_all = np.nonzero(amp_mask)[0]
        else:
            idx_all = np.arange(n_dim, dtype=np.int64)

        # We will build results only for selected indices to avoid 2^n memory.
        sel_coeffs = []
        sel_occ_a  = []
        sel_occ_b  = []

        # Vectorized bit extraction helper (little-endian bit order!)
        # bits_le[i, q] = ((idx[i] >> q) & 1)
        def bits_le_from_indices(idx):
            # idx are BIG-endian indices; produce LITTLE-endian bit matrix
            # bits_le[:, q] = ((idx >> (n_qubits - 1 - q)) & 1)
            shifts = (n_qubits - 1 - np.arange(n_qubits, dtype=idx.dtype))
            return (idx[:, None] >> shifts) & 1

        # Process in chunks to avoid huge (len(idx) x n_qubits) allocations
        CHUNK = 1_000_000 // max(n_qubits, 1)  # rough heuristic; tune as needed
        for start in range(0, idx_all.size, CHUNK):
            sl = slice(start, min(start + CHUNK, idx_all.size))
            idx = idx_all[sl]
            if idx.size == 0: break

            bits = bits_le_from_indices(idx)              # (m, n_qubits), little-endian
            bits_a = bits[:, :norb]                       # alpha block (q=0..norb-1)
            bits_b = bits[:, norb:]                       # beta  block (q=norb..2norb-1)

            # Electron-count filter (vectorized)
            mask_elec = (bits_a.sum(axis=1) == n_alpha) & (bits_b.sum(axis=1) == n_beta)
            if not np.any(mask_elec):
                continue

            idx_kept = idx[mask_elec]
            # Pull amplitudes
            amps_kept = psi[idx_kept]

            # Gather occupations (loop only over kept determinants, typically sparse)
            a_lists = [np.flatnonzero(row).astype(np.int32, copy=False) for row in bits_a[mask_elec]]
            b_lists = [np.flatnonzero(row).astype(np.int32, copy=False) for row in bits_b[mask_elec]]

            sel_coeffs.append(amps_kept)
            sel_occ_a.extend(a_lists)
            sel_occ_b.extend(b_lists)

        if not sel_coeffs:
            raise ValueError(
                "No determinants selected (check electron counts, threshold, or spin-orbital mapping)."
            )

        coeffs = np.concatenate(sel_coeffs).astype(np.complex128, copy=False)

        # Sort by |coeff| descending
        order = np.argsort(-np.abs(coeffs))
        coeffs = coeffs[order]
        occ_a  = [sel_occ_a[i] for i in order]
        occ_b  = [sel_occ_b[i] for i in order]

        # Normalize the selected subspace coefficients
        norm = np.linalg.norm(coeffs)
        if norm == 0.0:
            raise ValueError("Selected amplitudes have zero norm after filtering.")
        coeffs = coeffs / norm

        self.coeffs = coeffs
        self.occ_a  = occ_a
        self.occ_b  = occ_b
        self.wavefunction = (self.coeffs, self.occ_a, self.occ_b)

    # -------- utilities --------

    def order_truncate(self, max_det: int | None = None):
        """Optionally truncate to top-|coeff| determinants and renormalize."""
        if max_det is None or max_det >= len(self.coeffs):
            return self.wavefunction
        coeffs = self.coeffs[:max_det]
        occ_a  = self.occ_a[:max_det]
        occ_b  = self.occ_b[:max_det]
        coeffs = coeffs / np.linalg.norm(coeffs)
        return (coeffs, occ_a, occ_b)

    def overlap_with_fci(self):
        """⟨Ψ_FCI | Ψ_trial⟩ using PySCF’s block-spin FCI vector layout."""
        _, fcivec = self.mol_problem.get_FCI()
        coeffs, occ_a, occ_b = self.wavefunction
        norb   = self.n_spin_orbitals // 2
        nalpha = self.n_alpha
        nbeta  = self.n_beta

        def det_address(a_occ, b_occ):
            bit_a = sum(1 << i for i in a_occ)
            bit_b = sum(1 << i for i in b_occ)
            idx_a = cistring.str2addr(norb, nalpha, bit_a)
            idx_b = cistring.str2addr(norb, nbeta,  bit_b)
            return idx_a * cistring.num_strings(norb, nbeta) + idx_b

        ovlp = 0.0 + 0.0j
        for c, a, b in zip(coeffs, occ_a, occ_b):
            ovlp += np.conjugate(fcivec[det_address(a, b)]) * c
        return ovlp

    



