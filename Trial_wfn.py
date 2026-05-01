from __future__ import annotations
from ipie.trial_wavefunction.particle_hole import ParticleHole
from pyscf.fci.addons import large_ci
from ipie.systems.generic import Generic
from typing import Tuple, Optional
import numpy as np
from pyscf.fci import cistring
from typing import List, Tuple
import math
import re
import numpy as np
from pathlib import Path
from pyscf.fci.spin_op import spin_square
from math import sqrt
import time
class TrialWfn:
    def __init__(self,coeffs,occa,occb,mol_problem,max_det = None,verbose=False, compute_trial_energy=False):
        self.coeffs = coeffs
        self.occa  = occa
        self.occb = occb
        self.full_num_dets = len(self.coeffs)
        self.mol_problem = mol_problem
        self.verbose = verbose
        self.order_truncate(max_det)
        self.num_dets = len(self.t_coeffs)
        self.compute_trial_energy = compute_trial_energy

    def get_PH_trial(self):
        trial = ParticleHole(self.trial_wfn, 
                            self.mol_problem.mol_nelec,
                            self.mol_problem.full_spatial_orbitals, 
                            num_dets_for_props=self.num_dets,
                            use_active_space= self.mol_problem.active_space,
                            verbose = self.verbose)

        return trial
    def build_PH_trial(self):
        start = time.time()

        trial = self.get_PH_trial()
        # print("A: ", time.time()-start)

        ipie_ham = self.mol_problem.build_ipie_ham_from_fcidump()
        # print("B: ", time.time()-start)
        # trial.build()
        # print("C: ", time.time()-start)
        trial.half_rotate(ipie_ham)
        # print("D: ", time.time()-start)
        if self.compute_trial_energy:
            system = Generic(self.mol_problem.mol_nelec)

            trial.compute_trial_energy = True
            self.var_energy, _, _ = trial.calculate_energy(system, ipie_ham)
            # print("E:  ", time.time()-start)
        else:
            trial.compute_trial_energy = False
            self.var_energy = None
        return trial
    def order_truncate(self, max_det=None):
        c = self.coeffs
        a = self.occa
        b = self.occb

        # sort by |coeff| descending
        order = np.argsort(-np.abs(c))
        c = c[order]
        a = [a[i] for i in order]
        b = [b[i] for i in order]

        if max_det is not None:
            if max_det < len(c):
                c = c[:max_det]
                a = a[:max_det]
                b = b[:max_det]
            else: 
                print(f"num dets {len(c)} < max dets {max_det}, returning full trial")
        n = np.linalg.norm(c)
        c = c / n
        self.t_coeffs,self.t_occa,self.t_occb = c,a,b
        self.trial_wfn = (self.t_coeffs,self.t_occa,self.t_occb)
    def get_fci_fidelity(self, fci_trial:TrialWfn):
        """
        Return |⟨Ψ_FCI | Ψ_trial⟩|^2 using the (coeffs, occa, occb)
        expansions stored in TrialWfn.trial_wfn, without using ParticleHole.

        Parameters
        ----------
        fci_trial : TrialWfn or (coeffs, occa, occb)
            Either a TrialWfn instance for the FCI wavefunction, or directly
            its (coeffs, occa, occb) tuple.
        """

        # --- unpack FCI trial ---

        fci_coeffs, fci_occa, fci_occb = fci_trial.trial_wfn

        # --- unpack this trial ---
        trial_coeffs, trial_occa, trial_occb = self.trial_wfn

        # Ensure NumPy arrays for coefficients
        fci_coeffs   = np.asarray(fci_coeffs,   dtype=np.complex128)
        trial_coeffs = np.asarray(trial_coeffs, dtype=np.complex128)

        # --- sizes must match the determinant basis used by both expansions ---
        norb   = self.mol_problem.full_spatial_orbitals
        nalpha, nbeta = self.mol_problem.mol_nelec

        def det_address(alpha_occ, beta_occ, norb, nalpha, nbeta):
            """Alpha-major linear index (Na * Nb grid) matching PySCF ordering."""
            bit_a = sum(1 << i for i in alpha_occ)  # little-endian: orb 0 is LSB
            bit_b = sum(1 << i for i in beta_occ)
            idx_a = cistring.str2addr(norb, nalpha, bit_a)
            idx_b = cistring.str2addr(norb, nbeta,  bit_b)
            nbeta_dim = cistring.num_strings(norb, nbeta)
            return idx_a * nbeta_dim + idx_b

        # --- map FCI determinants to coefficients for O(1) lookup ---
        fci_map = {}
        for cF, a_occF, b_occF in zip(fci_coeffs, fci_occa, fci_occb):
            addrF = det_address(a_occF, b_occF, norb, nalpha, nbeta)
            # If duplicates appear (shouldn't), accumulate
            fci_map[addrF] = fci_map.get(addrF, 0.0 + 0.0j) + cF

        # --- accumulate overlap over the trial determinants ---
        ovlp = 0.0 + 0.0j
        for cT, a_occT, b_occT in zip(trial_coeffs, trial_occa, trial_occb):
            addrT = det_address(a_occT, b_occT, norb, nalpha, nbeta)
            cF = fci_map.get(addrT, 0.0 + 0.0j)
            if cF != 0.0:
                ovlp += np.conjugate(cF) * cT

        # --- normalize by ||Ψ_FCI|| and ||Ψ_trial|| (defensive, in case not 1) ---
        norm_fci   = float(np.linalg.norm(fci_coeffs))
        norm_trial = float(np.linalg.norm(trial_coeffs))
        if norm_fci == 0.0 or norm_trial == 0.0:
            raise ValueError("Zero-norm wavefunction encountered in fidelity calculation.")

        normalized_ovlp = ovlp / (norm_fci * norm_trial)
        return abs(normalized_ovlp)**2
    

    def get_S_squared(self):
        """
        Exact <S^2> from sparse determinant expansion only.
        """
        nalpha, nbeta = self.mol_problem.mol_nelec
        norb = self.mol_problem.full_spatial_orbitals
        psi = build_state_dict(self.coeffs, self.occa, self.occb, norb, normalize=True)
        s2psi = apply_S2_to_state_dict(psi, norb, nalpha, nbeta)
        s2 = overlap_sparse(psi, s2psi).real
        return float(s2)
    
    def get_singlet_projected_fidelity(self, fci_trial, return_weight=False):
        """
        Project trial_obj onto S^2=0 using the sparse determinant-basis S^2 operator,
        then compute fidelity with FCI.
        """
        norb = self.mol_problem.full_spatial_orbitals
        nelec = self.mol_problem.mol_nelec

        trial_coeffs, trial_occa, trial_occb = self.trial_wfn
        fci_coeffs, fci_occa, fci_occb = fci_trial.trial_wfn

        psi_trial = build_state_dict(trial_coeffs, trial_occa, trial_occb, norb, normalize=True)
        psi_fci   = build_state_dict(fci_coeffs,   fci_occa,   fci_occb,   norb, normalize=True)

        s2_before = self.get_S_squared()
        print("Non-Projected <S^2> =", s2_before)

        psi_proj = project_to_singlet_state_dict(psi_trial, norb, nelec)

        w0 = float(sum(abs(c)**2 for c in psi_proj.values()).real)

        if w0 < 1e-14:
            if return_weight:
                return 0.0, w0
            return 0.0

        psi_proj = normalize_state_dict(psi_proj)

        c_proj, oa_proj, ob_proj = state_dict_to_coeffs_occs(psi_proj, norb)
        projected_trial_obj = TrialWfn(
            coeffs=c_proj,
            occa=oa_proj,
            occb=ob_proj,
            mol_problem=self.mol_problem,
            max_det=None,
            compute_trial_energy=False
        )
        s2_after = projected_trial_obj.get_S_squared()
        print("Projected <S^2> =", s2_after)

        fid_proj = sparse_fidelity(psi_fci, psi_proj)

        if return_weight:
            return fid_proj, w0
        return fid_proj





def scale_state_dict(psi, a):
    return {b: a * c for b, c in psi.items()}

def add_state_dicts(psi1, psi2, tol=1e-15):
    out = dict(psi1)
    for b, c in psi2.items():
        out[b] = out.get(b, 0.0 + 0.0j) + c
        if abs(out[b]) < tol:
            del out[b]
    return out

def norm_state_dict(psi):
    return float(np.sqrt(sum(abs(c)**2 for c in psi.values())))

def normalize_state_dict(psi, tol=1e-15):
    nrm = norm_state_dict(psi)
    if nrm < tol:
        raise ValueError("Cannot normalize near-zero state.")
    return {b: c / nrm for b, c in psi.items()}

def bitstring_to_occs_alpha_beta(bitstr, norb):
    occa = [i for i in range(norb) if (bitstr >> i) & 1]
    occb = [i for i in range(norb) if (bitstr >> (norb + i)) & 1]
    return occa, occb

def state_dict_to_coeffs_occs(psi, norb):
    coeffs = []
    occa = []
    occb = []
    for b, c in psi.items():
        oa, ob = bitstring_to_occs_alpha_beta(b, norb)
        coeffs.append(c)
        occa.append(oa)
        occb.append(ob)
    return np.array(coeffs, dtype=np.complex128), occa, occb

def project_to_singlet_state_dict(psi, norb, nelec, tol=1e-15):
    """
    Exact singlet projector in fixed-(N_alpha, N_beta) sector:
        P0 = prod_{s=1}^{Smax} ( I - S^2 / [s(s+1)] )
    implemented using the sparse S^2 operator.
    """
    nalpha, nbeta = nelec
    sz = abs(nalpha - nbeta) / 2.0
    smax = (nalpha + nbeta) / 2.0

    if sz != 0:
        raise ValueError(
            f"Singlet projection requires Sz=0, but got Sz={0.5*(nalpha-nbeta)}"
        )

    proj = dict(psi)

    # for H10 half-filling this runs s = 1,2,3,4,5
    for s in range(1, int(round(smax)) + 1):
        lam = s * (s + 1.0)
        s2_proj = apply_S2_to_state_dict(proj, norb, nalpha, nbeta)
        proj = add_state_dicts(proj, scale_state_dict(s2_proj, -1.0 / lam), tol=tol)

    return proj

def sparse_overlap(psi1, psi2):
    return overlap_sparse(psi1, psi2)

def sparse_fidelity(psi1, psi2):
    ovlp = sparse_overlap(psi1, psi2)
    return float(abs(ovlp)**2)











def popcount(x: int) -> int:
    return x.bit_count()

def occs_to_bitstring(occa, occb, norb):
    """
    Spin-orbital ordering:
      alpha orbitals: 0 .. norb-1
      beta  orbitals: norb .. 2*norb-1
    """
    b = 0
    for i in occa:
        b |= (1 << i)
    for i in occb:
        b |= (1 << (norb + i))
    return b

def annihilate(bitstr: int, orb: int):
    """Apply a_orb to |bitstr>. Returns (phase, new_bitstr) or (0, None)."""
    if ((bitstr >> orb) & 1) == 0:
        return 0.0, None
    phase = -1.0 if (popcount(bitstr & ((1 << orb) - 1)) % 2) else 1.0
    return phase, bitstr ^ (1 << orb)

def create(bitstr: int, orb: int):
    """Apply a^\dagger_orb to |bitstr>. Returns (phase, new_bitstr) or (0, None)."""
    if ((bitstr >> orb) & 1) == 1:
        return 0.0, None
    phase = -1.0 if (popcount(bitstr & ((1 << orb) - 1)) % 2) else 1.0
    return phase, bitstr | (1 << orb)

def apply_op_sequence(bitstr: int, ops):
    """
    Apply a sequence of fermionic ops from right to left.
    ops is a list like:
      [("create", orb1), ("annihilate", orb2), ...]
    meaning leftmost operator first in algebraic expression.
    """
    phase = 1.0
    state = bitstr
    for kind, orb in reversed(ops):
        if kind == "annihilate":
            s, state = annihilate(state, orb)
        elif kind == "create":
            s, state = create(state, orb)
        else:
            raise ValueError(f"Unknown op kind: {kind}")
        if state is None:
            return 0.0, None
        phase *= s
    return phase, state

def build_state_dict(coeffs, occa, occb, norb, normalize=True):
    """
    Build sparse dict: bitstring -> coefficient
    """
    psi = {}
    for c, oa, ob in zip(coeffs, occa, occb):
        b = occs_to_bitstring(oa, ob, norb)
        psi[b] = psi.get(b, 0.0 + 0.0j) + complex(c)

    if normalize:
        norm = np.sqrt(sum(abs(v)**2 for v in psi.values()))
        if norm == 0:
            raise ValueError("Zero-norm state.")
        psi = {k: v / norm for k, v in psi.items()}
    return psi

def apply_S2_to_state_dict(psi, norb, nalpha, nbeta):
    """
    Apply S^2 exactly to a sparse determinant expansion.
    Returns sparse dict for S^2 |psi>.
    """
    out = {}

    # Sz is constant in a fixed (N_alpha, N_beta) sector
    sz = 0.5 * (nalpha - nbeta)
    sz2 = sz * sz

    for bitstr, coeff in psi.items():
        # Sz^2 term
        out[bitstr] = out.get(bitstr, 0.0 + 0.0j) + sz2 * coeff

        # 1/2 (S+S- + S-S+)
        for p in range(norb):
            a_p = p
            b_p = norb + p
            for q in range(norb):
                a_q = q
                b_q = norb + q

                # S+ S- = (a†_{pα} a_{pβ})(a†_{qβ} a_{qα})
                ops_sp_sm = [
                    ("create", a_p),
                    ("annihilate", b_p),
                    ("create", b_q),
                    ("annihilate", a_q),
                ]
                phase, newb = apply_op_sequence(bitstr, ops_sp_sm)
                if newb is not None:
                    out[newb] = out.get(newb, 0.0 + 0.0j) + 0.5 * phase * coeff

                # S- S+ = (a†_{pβ} a_{pα})(a†_{qα} a_{qβ})
                ops_sm_sp = [
                    ("create", b_p),
                    ("annihilate", a_p),
                    ("create", a_q),
                    ("annihilate", b_q),
                ]
                phase, newb = apply_op_sequence(bitstr, ops_sm_sp)
                if newb is not None:
                    out[newb] = out.get(newb, 0.0 + 0.0j) + 0.5 * phase * coeff

    return out

def overlap_sparse(psi1, psi2):
    """
    <psi1|psi2> for sparse dicts
    """
    if len(psi1) > len(psi2):
        psi1, psi2 = psi2, psi1
        conj_first = False
    else:
        conj_first = True

    val = 0.0 + 0.0j
    if conj_first:
        for b, c in psi1.items():
            d = psi2.get(b, 0.0 + 0.0j)
            if d != 0:
                val += np.conjugate(c) * d
    else:
        for b, d in psi1.items():
            c = psi2.get(b, 0.0 + 0.0j)
            if c != 0:
                val += np.conjugate(c) * d
    return val


  
   

  











def parse_swcs_wf(path: str | Path) -> Tuple[np.ndarray, List[List[int]], List[List[int]]]:
    """
    Parse an SWCS/PySCF .wf file into (coeffs, occa, occb).

    Returns
    -------
    coeffs : (n_det,) float64
        CI coefficients for each determinant.
    occa : list[list[int]]
        For each determinant, the list of occupied alpha *spatial* orbital indices (0-based).
    occb : list[list[int]]
        For each determinant, the list of occupied beta *spatial* orbital indices (0-based).

    Notes
    -----
    - The file format:
        * First non-empty line may be an integer giving total # of spin orbitals (qubits).
          If absent, we infer it from the largest determinant bit-length and round up to an even number.
        * Remaining non-empty lines each contain: <det_int> <amplitude_float>
    - Bitstring convention (per your description):
        * Read right-to-left (LSB first).
        * First half (positions 0..n_spat-1) → alpha spin orbitals.
        * Next half (positions n_spat..2*n_spat-1) → beta spin orbitals.
        * Within each spin block, indices increase with bit position (0-based).
    - Coefficients are renormalized defensively if they are not already unit-normalized.
    """
    path = Path(path)
    dets: List[int] = []
    coeffs: List[complex] = [] 

    with path.open("r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    # Try to read the first line as an explicit "nspin" if it looks like a single integer.
    nspin: int | None = None
    first = lines[0]
    if re.fullmatch(r"[+-]?\d+", first):
        nspin = int(first)
        data_lines = lines[1:]
    else:
        data_lines = lines

    # Parse remaining lines of "<int> <float>" (tolerant to extra spacing)
    pair_re = re.compile(
        r"""
        ^\s*([+-]?\d+)\s+            # determinant integer
        ([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)
        \s*$""",
        re.VERBOSE,
    )

    for ln in data_lines:
        m = pair_re.match(ln)
        if not m:
            # Ignore non-matching lines (headers, comments), or raise if you prefer strict
            continue
        dets.append(int(m.group(1)))
        coeffs.append(complex(float(m.group(2)), 0.0))  # NEW: make complex with imag=0

    if not dets:
        raise ValueError("No determinant/coeff pairs parsed from file.")

    # Infer nspin if not explicitly given
    if nspin is None:
        max_bits = max(d.bit_length() for d in dets)
        # Round up to the next even number to split evenly into alpha/beta
        nspin = max_bits if (max_bits % 2 == 0) else (max_bits + 1)

    if nspin <= 0 or (nspin % 2) != 0:
        raise ValueError(f"Total spin orbitals (nspin={nspin}) must be a positive even integer.")

    n_spat = nspin // 2

    # Build occupation lists
    occa: List[List[int]] = []
    occb: List[List[int]] = []
    print("in parrser number of dets: ", len(dets))

    for n,D in enumerate(dets):
        alpha_occ = [i for i in range(n_spat) if (D >> i) & 1]
        beta_occ  = [i for i in range(n_spat) if (D >> (n_spat + i)) & 1]
        if n < 5:
           print(f"int: {D}, alpha_occ: {alpha_occ}, beta_occ: {beta_occ}")

        occa.append(alpha_occ)
        occb.append(beta_occ)

    coeffs_arr = np.asarray(coeffs, dtype=np.complex128)  # CHANGED: complex128 array

    # Defensive normalization (your files *should* already be normalized)
    norm = float(np.linalg.norm(coeffs_arr))
    if norm == 0.0:
        raise ValueError("All coefficients are zero; cannot normalize.")
    if abs(norm - 1.0) > 1e-10:
        coeffs_arr /= norm

    return coeffs_arr, occa, occb






