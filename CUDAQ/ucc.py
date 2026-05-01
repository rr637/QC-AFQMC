
from __future__ import annotations
import cudaq
from typing import Callable, Sequence
from functools import partial
from CUDAQ.fermionic_excitation_generator import generate_fermionic_excitations, get_alpha_excitations
from typing import List, Tuple
from openfermion.ops import FermionOperator, QubitOperator
from openfermion.transforms.opconversions.jordan_wigner import jordan_wigner
from openfermion.transforms.opconversions.term_reordering import normal_ordered
from pyscf import gto
from pyscf.scf import hf_symm
import numpy as np



############ Without importing qiskit, excitation_list is adapted from qiskit ###################

_EXCITATION_TYPE = {
    "s": 1,
    "d": 2,
    "t": 3,
    "q": 4,
}

def ansatz_type_to_kwargs(ansatz_type):
    
    if ansatz_type not in {"UCCSD","UCCGSD","UCCD","UpCCD", "UpCCSD", "UpCCGSD", "UpCCGD"}:
        raise ValueError("not suitbale ansatz_type")
    excitation = ""
    paired_doubles = False
    alpha_spin = True
    beta_spin = True
    max_spin_excitation = None
    generalized = False
    preserve_spin = True

    if "p" in ansatz_type:
        paired_doubles = True
        # beta_spin = False
    if "G" in ansatz_type:
        generalized = True
    if "S" in ansatz_type:
        excitation += "s"
    if 'D' in ansatz_type:
        excitation += "d"
    extra_kwargs = {
        "alpha_spin": alpha_spin,
        "beta_spin": beta_spin,
        "max_spin_excitation": max_spin_excitation,
        "generalized": generalized,
        "preserve_spin": preserve_spin,
        "paired_doubles": paired_doubles
    }
    return excitation,extra_kwargs
 

def _get_excitation_generators(excitations,extra_kwargs) -> list[Callable]:
    generators: list[Callable] = []


    if isinstance(excitations, str):
        for exc in excitations:
            if exc == "d" and extra_kwargs["paired_doubles"]:
                continue
            else:
                generators.append(
                    partial(
                        generate_fermionic_excitations,
                        num_excitations=_EXCITATION_TYPE[exc],
                        **extra_kwargs,
                    )
                )
    return generators


def get_excitation_list(excitations,mol_problem,
                        extra_kwargs) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    num_spatial_orbitals = mol_problem.active_orbitals
    num_particles = mol_problem.active_mol_nelec
    excitation_list = []

    # print(extra_kwargs["paired_doubles"])
    if extra_kwargs["paired_doubles"]:
        num_electrons = num_particles[0]
        beta_index_shift = num_spatial_orbitals

        # generate alpha-spin orbital indices for occupied and unoccupied ones
        alpha_excitations = get_alpha_excitations(
            num_spatial_orbitals, num_electrons, generalized=extra_kwargs["generalized"]
        )
        # print(f"alpha_excitations: {alpha_excitations}")
        for alpha_exc in alpha_excitations:
            # create the beta-spin excitation by shifting into the upper block-spin orbital indices
            beta_exc = (
                alpha_exc[0] + beta_index_shift,
                alpha_exc[1] + beta_index_shift,
            )
            # add the excitation tuple
            occ: tuple[int, ...]
            unocc: tuple[int, ...]
            occ, unocc = zip(alpha_exc, beta_exc)
            exc_tuple = (occ, unocc)
            excitation_list.append(exc_tuple)
    
    generators = _get_excitation_generators(excitations,extra_kwargs)

    for gen in generators:
        excitation_list.extend(
            gen(  # pylint: disable=not-callable
                num_spatial_orbitals=num_spatial_orbitals,
                num_particles=num_particles,
            )
        )

    return excitation_list

def count_singles_doubles(excitation_list):
    """
    Count single- and double-excitations in an excitation_list.

    Each item in excitation_list is ((occ_indices...), (virt_indices...)).
    A 'single' has len(occ)==len(virt)==1; a 'double' has ==2.

    Returns
    -------
    n_singles : int
    n_doubles : int
    """
    singles = doubles = 0
    for ex in excitation_list:
        if not isinstance(ex, (tuple, list)) or len(ex) != 2:
            raise ValueError(f"Bad excitation tuple: {ex!r}")
        occ, virt = ex
        r_occ, r_virt = len(occ), len(virt)
        if r_occ != r_virt:
            raise ValueError(f"Rank mismatch in excitation {ex!r}")
        if r_occ == 1:
            singles += 1
        elif r_occ == 2:
            doubles += 1
        # higher ranks are ignored
    return singles, doubles


def get_ucc_two_qubit_gate_count(pauli_words:List[str]):
    count = 0
    for pw in pauli_words:
        weight = sum(1 for ch in pw if ch != 'I')
        if weight >= 2:
            count += 2*(weight-1)
    return count



def _hermitian_conjugated(op: FermionOperator) -> FermionOperator:
    out = FermionOperator()
    for term, coeff in op.terms.items():
        # reverse order & flip creation/annihilation; conjugate coefficient
        hc_term = tuple((i, 1 - a) for i, a in reversed(term))
        out += FermionOperator(hc_term, complex(coeff).conjugate())
    return out

from typing import List, Tuple


def _excitation_to_fermion_generator(ex) -> FermionOperator:
    """
    Hermitian generator:
      single:  a_a^\dagger a_i - h.c.
      double:  a_a^\dagger a_b^\dagger a_j a_i - h.c.
    """
    occ, virt = ex
    if len(occ) == 1 and len(virt) == 1:
        (i,), (a,) = occ, virt
        up = FermionOperator(((a, 1), (i, 0)))
    elif len(occ) == 2 and len(virt) == 2:
        i, j = occ
        a, b = virt
        up = FermionOperator(((a, 1), (b, 1), (j, 0), (i, 0)))
    else:
        raise ValueError(f"Only singles/doubles supported. Got occ={occ}, virt={virt}")
    rank = len(ex[0])  # 1 for singles, 2 for doubles
    phase = 1j
    gen_f = phase * (up - _hermitian_conjugated(up))
    return normal_ordered(gen_f)

def _qubit_term_to_label(term: Tuple[Tuple[int, str], ...], n_qubits: int) -> str:
    letters = ['I'] * n_qubits
    for q, p in term:
        letters[q] = p
    return ''.join(letters)

def collect_pauli_from_excitation_list(
    excitation_list,
    n_spin_orbitals: int,
    *,
    make_hermitian: bool = True,
    real_tol: float = 1e-12,
    nreps = 1
):
    pauli_words: List[str] = []
    coeffs: List[complex] = []
    block_ids: List[int] = []

    # sanity on index range
    max_index = -1
    for occ, virt in excitation_list:
        if occ: max_index = max(max_index, max(occ))
        if virt: max_index = max(max_index, max(virt))
    if max_index >= n_spin_orbitals:
        raise ValueError(f"Found index {max_index}, but n_spin_orbitals={n_spin_orbitals}.")
    for i in range(nreps):
        for j, ex in enumerate(excitation_list):
            gen_f = _excitation_to_fermion_generator(ex)  # anti-Hermitian                     

            qop: QubitOperator = jordan_wigner(gen_f)

            for term, coef in qop.terms.items():
                label = _qubit_term_to_label(term, n_spin_orbitals)
                c = complex(coef)
                if make_hermitian and abs(c.imag) < real_tol:
                    c = float(c.real)
                pauli_words.append(label)
                coeffs.append(c)
                block_ids.append(j+i*len(excitation_list))


    return pauli_words, coeffs, block_ids

def get_symmetry_excitation_list(excitation_list, mol_problem):
    """excittaion list is list of single ((a),(i)) or double ((a,b),(i,j)) excitations"""
    
    pyscf_mol = mol_problem.mol
    mf = mol_problem.mf
    orbsym, group = get_orbsym_for_existing_mf(pyscf_mol, mf) 
    # print("Detected group for labeling:", group)

    kept_excitations = sym_filter_excitations_with_orbsym(
        excitation_list, orbsym, mol_problem.active_orbitals
    )
    return kept_excitations, group

def get_orbsym_for_existing_mf(mol, mf):
    """Build a symmetry-enabled copy just to label MOs; do NOT change your main mol/mf."""
    mol_lbl = gto.M(atom=mol.atom, basis=mol.basis, unit=mol.unit,
                    charge=mol.charge, spin=mol.spin,
                    symmetry=True)
    orbsym = np.asarray(hf_symm.get_orbsym(mol_lbl, mf.mo_coeff), dtype=int)
    return orbsym, getattr(mol_lbl, 'groupname', 'C1')


def so_to_spatial(idx, norb):
        """ spin orbital to spatial orbital (Assume blocked ordering) """
        return idx if idx < norb else idx - norb

def sym_filter_excitations_with_orbsym(excitation_list, orbsym, norb):
    """
    Keep only excitations that preserve the reference state's spatial irrep.
    orbsym: length-norb int array of MO irreps from PySCF (hf_symm.get_orbsym).
    """
    # orbsym = np.asarray(orbsym, dtype=int)
    print("ORBSYM: ",orbsym)
    assert orbsym.size == norb, f"orbsym length {orbsym.size} != norb {norb}"

    kept = []
    for occ, virt in excitation_list:
        # detect if list is in spin-orbital indices
        use_so = max((*occ, *virt)) >= norb
        occ_sp = [so_to_spatial(i, norb) for i in occ] if use_so else list(occ)
        virt_sp = [so_to_spatial(a, norb) for a in virt] if use_so else list(virt)

        if len(occ_sp) == 1:                      # singles: Γ(a) == Γ(i)
            i, a = occ_sp[0], virt_sp[0]

            if orbsym[i] == orbsym[a]:
                # print(f"Kept, occ: {orbsym[i]},  vir: {orbsym[a]} | Equal")
                kept.append((occ, virt))

        elif len(occ_sp) == 2:                    # doubles: Γ(a) XOR Γ(b) == Γ(i)  XOR Γ(j)
            i, j = occ_sp
            a, b = virt_sp
            if (orbsym[i] ^ orbsym[j]) == (orbsym[a] ^ orbsym[b]):  
                # print(f"Kept, occs: {orbsym[i],orbsym[j]},  vir: {orbsym[a],orbsym[b] } by XOR")

                kept.append((occ, virt))
        else:
            raise ValueError("Only singles/doubles supported.")
    return kept




@cudaq.kernel
def ucc_circuit(qubit_num: int, 
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







