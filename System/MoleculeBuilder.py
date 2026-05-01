from __future__ import annotations
import pathlib, re, tempfile
from pyscf.tools import fcidump as pyscf_fcidump
from pyscf import ao2mo
from pyscf import gto, scf,fci, ao2mo,mcscf,lib,tools
from typing import Tuple, Optional
from pyscf.mcscf import CASCI
from pyscf.fci.addons import large_ci
import hashlib
import numpy as np
from pyscf.fci.addons import large_ci
from functools import reduce
from pyscf.mcscf import CASSCF
import time
from Trial_wfn  import TrialWfn
from pyscf.tools.fcidump import read
from ipie.utils.from_pyscf import generate_integrals
from ipie.hamiltonians.generic  import Generic as HamGeneric
from  pathlib import Path
import os
import pyscf.cc as cc
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.operators import ElectronicIntegrals
from qiskit_nature.second_q.mappers import JordanWignerMapper

class BuildMoleculeProblem:

  def __init__(self,
               atom : str,
               basis : str ,
               spin: int,
               active_orbitals: Optional[int] = None,
               active_mol_nelec: Optional[Tuple[int,int]] = None,
               charge: int = 0,
               unit : Optional[str] = 'angstrom',
               mol_identifier: Optional[str] = None,
               mpi_info = None,
               fci_dump_path = None,
               verbose = True):
    self.atom = atom
    self.basis = basis
    self.spin = spin
    self.mol_identifier = mol_identifier
    self.mpi_info = mpi_info
    self.fci_dump_path = fci_dump_path if fci_dump_path is None else Path(fci_dump_path)
    self.verbose = verbose
    self.unit = unit
    Path("chk_files").mkdir(parents=True, exist_ok=True)
    Path("fci_dump_files").mkdir(parents=True, exist_ok=True)
    
    if self.mol_identifier is None:
       self.mol_identifier = f"{self.atom}"
    if self.mpi_info is None:
        self.rank = 0
        self.size = 1
    else:
        self.rank = self.mpi_info.rank
        self.size  = self.mpi_info.size
    if self.rank > 0:
       self.verbose = False
    if self.unit == 'Bohr':
       self.mol_identifier = f"{self.mol_identifier}_bohr"

    current_file_directory = os.path.dirname(os.path.abspath(__file__))  # for path generalization


    self.chk_file_path = Path(f"chk_files/{self.mol_identifier}_{self.basis}_{self.spin}.chk")
        
    self.charge = charge

    self.mol = self.PySCF_mol()
    if active_orbitals == None and active_mol_nelec == None:
      self.active_orbitals = self.mol.nao_nr()
      self.active_mol_nelec = self.mol.nelec
      self.active_space = False
    else:
      self.active_orbitals = active_orbitals
      self.active_mol_nelec = active_mol_nelec
      self.active_space = not self.active_orbitals == self.mol.nao_nr() and self.active_mol_nelec[0] == self.mol.nelec[0] and self.active_mol_nelec[1] == self.mol.nelec[1]
    if self.fci_dump_path is None:
        if self.active_space:
            self.fci_dump_path = Path(f"fci_dump_files/{self.mol_identifier}_{self.basis}_{self.spin}_AO{self.active_orbitals}_AE{self.active_mol_nelec[0]}-{self.active_mol_nelec[1]}.FCIDUMP")
        else:
            self.fci_dump_path = Path(f"fci_dump_files/{self.mol_identifier}_{self.basis}_{self.spin}.FCIDUMP")

    self.mf = self.build_scf()
    self.active_electrons = self.active_mol_nelec[0] + self.active_mol_nelec[1]
    self.active_spin_orbitals = self.active_orbitals * 2
    self.active_spatial = self.get_active_spatial()
    self.mf_uhf = None
    if self.rank == 0:
        if not self.fci_dump_path.exists():
            if self.verbose:
                print(f"Creating FCI_Dump full path: {self.fci_dump_path.resolve()}")

            self.fci_dump_path.parent.mkdir(parents=True, exist_ok=True)

            self.build_fci_dump()
        else:
           if self.verbose:
            print(f"Using FCI_Dump full path: {self.fci_dump_path.resolve()}")

    self.FCI = None
    self.ipie_ham = None
    self.ccsd = None
  def PySCF_mol(self):
    self.mol = gto.M(
          atom=self.atom,
          basis=self.basis,
          unit=self.unit,
          verbose=0,
          spin=self.spin,
          charge = self.charge
      )
    return self.mol
  @property
  def full_spatial_orbitals(self):
    return self.mol.nao_nr()
  @property
  def mol_nelec(self):
    return self.mol.nelec
  @property
  def full_electrons(self):
    return sum(self.mol.nelec)
  
  def frozen_spatial_orbitals(self):
    nspatial = self.mf.mo_coeff.shape[1]
    return [i for i in range(nspatial) if i not in self.active_spatial]

  def build_ccsd(self,restricted=True):
      if self.ccsd is None:
        frz = self.frozen_spatial_orbitals()

        if restricted:
            c = cc.CCSD(self.mf, frozen=frz)
            c.kernel()
            self.ccsd = c
            if self.verbose:
                print("RCCSD CONVERGED: ",c.converged)
        else:
            c = cc.UCCSD(self.mf, frozen=frz)
            c.kernel()
            self.ccsd = c
            if self.verbose:
                print("UCCSD CONVERGED: ",c.converged)
      return self.ccsd
  def get_ccsd_energy(self):
     ccsd = self.build_ccsd()
     return ccsd.e_tot

  
  def build_scf(self):
    # Pick RHF/ROHF based on spin
    chk = Path(self.chk_file_path)

    scf_cls = scf.ROHF if self.mol.spin else scf.RHF

    if chk.exists():
        if self.rank == 0:
            if self.verbose:
                print(f"Checkpoint full path: {chk.resolve()}")

        data = lib.chkfile.load(str(chk), 'scf')  # dict of arrays/scalars
        mf = scf_cls(self.mol)
        # choose fields you need
        for k in ('mo_coeff','mo_occ','mo_energy','e_tot','converged', "hcore", "X","mol"):
            if k in data:
                # print(f"{k}:{data[k]}")
                v = data[k]
                setattr(mf, k, np.asarray(v) if hasattr(v, 'shape') else v)
        return mf
    self.chk_file_path.parent.mkdir(parents=True, exist_ok=True)
    mf = scf_cls(self.mol)
    if self.rank == 0:
        if self.verbose:
            print(f"chk doesn't exist, writing in {os.getcwd()}", flush=True)
        self.chk_file_path.parent.mkdir(parents=True, exist_ok=True)
        mf.chkfile = str(self.chk_file_path)
    mf.kernel()
    return mf

  def build_fci_dump(self):
    fci_path = self.fci_dump_path
    if self.active_space:
        ncas = self.active_orbitals
        nelecas_total = self.active_electrons  # int is fine
        mc = mcscf.CASSCF(self.mf, ncas, nelecas_total).run()
        tools.fcidump.from_mcscf(mc, str(fci_path))
    else:
        tools.fcidump.from_scf(self.mf, str(fci_path))
  def get_uhf_energy(self):
    mf_uhf, _, _ = self.get_uhf()
    return mf_uhf.e_tot
       
  def get_uhf(self):
    """
    Converge AFM-UHF in AO basis, then express occupied UHF orbitals in the
    RHF MO basis used by the FCIDUMP Hamiltonian.

    Returns:
      psi: (norb, nalpha+nbeta) for ipie SingleDet
      nelec: (nalpha, nbeta)
      nbasis: norb
      mf_uhf: the converged PySCF UHF object (useful for diagnostics)
    """
    # originally was m=0, max_cycle = 200
    def run_uhf_afm(mol, m=0.8, max_cycle=500, conv_tol=1e-10, verbose=0):
        """Converge the broken-symmetry (AFM) UHF solution for an even H chain."""
        mf = scf.UHF(mol)
        mf.verbose = verbose
        mf.max_cycle = max_cycle
        mf.conv_tol = conv_tol

        # Spin-summed initial density from atomic guess
        dm_rhf = scf.hf.init_guess_by_atom(mol)

        # Alternating local spin polarization in AO blocks per atom
        nao = mol.nao_nr()
        dms = np.zeros((nao, nao))
        aoslices = mol.aoslice_by_atom()

        for A in range(mol.natm):
            p0, p1 = aoslices[A, 2], aoslices[A, 3]   # AO range for atom A
            sign = +1.0 if (A % 2 == 0) else -1.0
            dms[p0:p1, p0:p1] += sign * m * np.eye(p1 - p0)

        dm_a0 = dm_rhf + 0.5 * dms
        dm_b0 = dm_rhf - 0.5 * dms

        mf.kernel(dm0=(dm_a0, dm_b0))
        return mf
    if self.mf_uhf is None:

        mol = self.mol

        mf_rhf = self.mf
        C_ref = np.asarray(mf_rhf.mo_coeff)     # AO -> RHF-MO
        norb = C_ref.shape[1]

        # *** IMPORTANT: get the broken-symmetry UHF solution ***
        mf_uhf = run_uhf_afm(mol)
        self.mf_uhf = mf_uhf
        Ca = np.asarray(mf_uhf.mo_coeff[0])     # AO -> UHF alpha MOs
        Cb = np.asarray(mf_uhf.mo_coeff[1])     # AO -> UHF beta MOs
        occa_mask = np.asarray(mf_uhf.mo_occ[0]) > 1e-8
        occb_mask = np.asarray(mf_uhf.mo_occ[1]) > 1e-8

        S = mol.intor("int1e_ovlp_sph")

        # Project occupied UHF orbitals into RHF-MO basis
        X_a = (C_ref.T @ S @ Ca)[:, occa_mask]  # (norb, nalpha)
        X_b = (C_ref.T @ S @ Cb)[:, occb_mask]  # (norb, nbeta)

        # Orthonormalize within RHF basis
        X_a, _ = np.linalg.qr(X_a)
        X_b, _ = np.linalg.qr(X_b)
        self.X_a = X_a
        self.X_b = X_b


    return self.mf_uhf, self.X_a, self.X_b
    

  
  def get_hf_energy(self):
    return self.mf.energy_tot()
  # need .build() to be called
  

  def build_ipie_ham(self):
    if self.ipie_ham is None:
      mf = self.mf
      mol = self.mol
      rhf_coeff_matrix = mf.mo_coeff
      h1e, chol, nuc = generate_integrals(mol, mf.get_hcore(), rhf_coeff_matrix)
      # build hamiltonian
      n_full_basis = rhf_coeff_matrix.shape[0]
      nchol = chol.shape[0]
      nelec = mol.nelec
      chol = chol.transpose(1, 2, 0).reshape(n_full_basis * n_full_basis, nchol)
      ham = HamGeneric(np.array([h1e, h1e]), chol, nuc)
      self.ipie_ham = ham
    return self.ipie_ham
  def build_ipie_ham_from_fcidump(self, eig_thresh: float = 1e-12, verbose: bool = False) -> HamGeneric:
    if self.ipie_ham is None:
        fcidump_path = self.fci_dump_path
        res = read(str(fcidump_path), verbose=verbose)  # your read() from above
        norb  = int(res['NORB'])
        h1e   = np.asarray(res['H1'], dtype=float)
        h2pk  = np.asarray(res['H2'], dtype=float)      # packed pair-space
        ecore = float(res.get('ECORE', 0.0))

        assert h1e.shape == (norb, norb)

        # Unpack pair-space supermatrix
        npair = norb * (norb + 1) // 2
        V_pair = np.zeros((npair, npair), dtype=float)
        k = 0
        for a in range(npair):
            for b in range(a + 1):
                val = h2pk[k]
                V_pair[a, b] = val
                V_pair[b, a] = val
                k += 1
        V_pair = 0.5 * (V_pair + V_pair.T)

        # Map unordered pair index to ordered (p,q)
        def pair_index(i: int, j: int) -> int:
            if i < j:
                i, j = j, i
            return i * (i + 1) // 2 + j

        pair_idx = np.empty((norb, norb), dtype=int)
        for p in range(norb):
            for q in range(norb):
                pair_idx[p, q] = pair_index(p, q)

        flat_map = pair_idx.reshape(norb * norb)
        V_full = V_pair[np.ix_(flat_map, flat_map)]
        V_full = 0.5 * (V_full + V_full.T)

        # Eigendecomp → Cholesky vectors
        w, U = np.linalg.eigh(V_full)          # U: (norb^2, norb^2)
        keep = w > max(eig_thresh, 0.0)
        if not np.any(keep):
            raise RuntimeError("No positive eigenvalues in ERI supermatrix; check FCIDUMP.")
        w_keep = w[keep]                        # (k,)
        U_keep = U[:, keep]                     # (norb^2, k)

        # FIXED ORIENTATION: (k,1) * (k, n^2) via U_keep.T
        L_flat = (np.sqrt(w_keep)[:, None] * U_keep.T)  # (k, norb^2)
        chol_flat = L_flat.T                              # (norb^2, k)

        h1e_spin = np.stack([h1e, h1e], axis=0)         # RHF
        ham = HamGeneric(h1e_spin, chol_flat, ecore)
        # print("h1e shape:", ham.H1.shape)      # should be (2, norb, norb)
        # print("chol shape:", ham.chol.shape)   # should be (norb*norb, nchol)
        # print("norb inferred:", ham.H1.shape[-1])
        if verbose:
            print(f"[ham_from_fcidump] norb={norb}, npair={npair}, "
                f"nchol={chol_flat.shape[1]}, ecore={ecore:.12f}")
        self.ipie_ham = ham
    return self.ipie_ham


  def build_qiskit_ham(self):
  
    if self.active_space:       # ---- build CASCI on the active space ----
      ncas = self.active_orbitals             # number of spatial orbitals in the active space
      nelecas = self.active_mol_nelec                 # (n_alpha, n_beta)

      # Correct CASCI signature: CASCI(mf, ncas, nelecas)
      casci = mcscf.CASCI(self.mf, ncas, nelecas)
      casci.kernel()

      # Effective 1e and 2e integrals in the *active MO* basis + core constant
      h1, ecore = casci.get_h1eff()              # h1 shape: (ncas, ncas). ecore is a float
      h2 = casci.get_h2eff()                     # compressed 2e; chemist's notation
      eri = ao2mo.restore(1, h2, ncas)           # shape (ncas, ncas, ncas, ncas), (pq|rs) **chemist's**

      # ---- build a Qiskit-Nature Hamiltonian in the same active basis ----
      ints = ElectronicIntegrals.from_raw_integrals(h1, eri)  # restricted case: alpha=beta
      H_el = ElectronicEnergy(ints, constants=float(ecore))   # include core constant here

      second_q_op = H_el.second_q_op()
      qubit_spo = JordanWignerMapper().map(second_q_op)

    else:
      
      # ---- full space (no active space truncation) ----
      # MO coefficients from your SCF object (RHF/ROHF assumed here)
      C = self.mf.mo_coeff                     # shape (nao, norb)
      norb = C.shape[1]

      # 1-electron core Hamiltonian in AO, then transform to MO
      hcore_ao = self.mol.intor("int1e_kin") + self.mol.intor("int1e_nuc")
      h1 = C.T @ hcore_ao @ C                  # shape (norb, norb)

      # 2-electron integrals in MO (chemist's notation, pq|rs)
      # ao2mo.kernel returns a compressed tensor; restore() gives full (norb,norb,norb,norb)
      eri_comp = ao2mo.kernel(self.mol, C, aosym=1)   # 's4' compressed
      eri = ao2mo.restore(1, eri_comp, norb)          # chemist's (pq|rs)

      # Core constant: nuclear repulsion energy (no frozen core in full space)
      ecore = float(self.mol.energy_nuc())

      # Qiskit-Nature electronic Hamiltonian (spin-restricted)
      ints = ElectronicIntegrals.from_raw_integrals(h1, eri)
      H_el = ElectronicEnergy(ints, constants=ecore)

      second_q_op = H_el.second_q_op()
      qubit_spo = JordanWignerMapper().map(second_q_op)
    return qubit_spo


  def get_mol_hamiltonian_from_fcidump(
    self,
    return_fermion_string: bool = True
) -> Tuple[np.ndarray, np.ndarray, float, int, int, Optional[str]]:
    """
    Load 1e/2e integrals (+ core energy) from an FCIDUMP and build the
    spin-orbital, blocked-ordering Hamiltonian expected by
    `generate_molecular_spin_ham_restricted_blocked`.
    """
    fcidump_path = str(self.fci_dump_path)

    # Robust parse with canonicalization fallback
    res = _read_fcidump_resilient(fcidump_path)

    h1e = np.asarray(res['H1'])            # (norb, norb), spatial
    h2e_packed = np.asarray(res['H2'])     # packed ERIs (ij|kl)
    ecore = float(res.get('ECORE', 0.0))
    norb = int(res['NORB'])
    nelec = int(res['NELEC'])

    # Full chemist (pq|rs)
    h2e_chem = ao2mo.restore(1, h2e_packed, norb)  # -> (norb, norb, norb, norb)

    # If your downstream builder wants (p, r, s, q), keep this transpose; otherwise adjust.
    h2e_pqrs = np.asarray(h2e_chem.transpose(0, 2, 3, 1), order='C')

    obi, tbi, core_energy, ferm_ham = self.generate_molecular_spin_ham_restricted_blocked(
        h1e, h2e_pqrs, ecore
    )

    if not return_fermion_string:
        ferm_ham = None

    return (obi, tbi, core_energy, nelec, norb, ferm_ham)

  
  # From Nvidia Qchem
  def generate_molecular_spin_ham_restricted_blocked(self, h1e, h2e, ecore):
      """
      Generate the molecular spin Hamiltonian with **blocked spin-orbital ordering**:
          [α0, α1, ..., α_{N-1}, β0, β1, ..., β_{N-1}]
      """

      n_spatial = h1e.shape[0]
      nqubits = 2 * n_spatial

      one_body_coeff = np.zeros((nqubits, nqubits))
      two_body_coeff = np.zeros((nqubits, nqubits, nqubits, nqubits))
      ferm_ham = []

      # α(i) = i, β(i) = n_spatial + i
      def a(i): return i
      def b(i): return n_spatial + i

      for p in range(n_spatial):
          for q in range(n_spatial):

              # Same-spin one-body terms
              one_body_coeff[a(p), a(q)] = h1e[p, q]
              ferm_ham.append(f"{h1e[p, q]} a_{p}^dagger a_{q}")
              one_body_coeff[b(p), b(q)] = h1e[p, q]
              ferm_ham.append(f"{h1e[p, q]} b_{p}^dagger b_{q}")

              for r in range(n_spatial):
                  for s in range(n_spatial):
                      val = 0.5 * h2e[p, q, r, s]

                      # Same-spin αααα and ββββ
                      two_body_coeff[a(p), a(q), a(r), a(s)] = val
                      ferm_ham.append(f"{val} a_{p}^dagger a_{q}^dagger a_{r} a_{s}")

                      two_body_coeff[b(p), b(q), b(r), b(s)] = val
                      ferm_ham.append(f"{val} b_{p}^dagger b_{q}^dagger b_{r} b_{s}")

                      # Mixed-spin αββα and βααβ
                      two_body_coeff[a(p), b(q), b(r), a(s)] = val
                      ferm_ham.append(f"{val} a_{p}^dagger a_{q}^dagger b_{r} b_{s}")

                      two_body_coeff[b(p), a(q), a(r), b(s)] = val
                      ferm_ham.append(f"{val} b_{p}^dagger b_{q}^dagger a_{r} a_{s}")

      full_hamiltonian = " + ".join(ferm_ham)
      return one_body_coeff, two_body_coeff, ecore, full_hamiltonian



  def get_mol_hamiltonian(self):
    nele_cas = self.active_electrons
    mol = self.mol
    myhf = self.mf
    norb_cas = self.active_orbitals
    if nele_cas is None:
      h1e_ao = mol.intor("int1e_kin") + mol.intor("int1e_nuc")

      h1e = reduce(np.dot, (myhf.mo_coeff.T, h1e_ao, myhf.mo_coeff))

      # Compute the 2e integrals then convert to HF basis
      h2e_ao = mol.intor("int2e_sph", aosym='1')
      h2e = ao2mo.incore.full(h2e_ao, myhf.mo_coeff)


      h2e = h2e.transpose(0, 2, 3, 1)

      nuclear_repulsion = myhf.energy_nuc()


      obi, tbi, e_nn, ferm_ham = self.generate_molecular_spin_ham_restricted_blocked(
          h1e, h2e, nuclear_repulsion)

    else:


    
        mc = mcscf.CASCI(myhf, norb_cas, nele_cas)
        h1e_cas, ecore = mc.get_h1eff(myhf.mo_coeff)
        self.ecore = ecore
        h2e_cas = mc.get_h2eff(myhf.mo_coeff)
        h2e_cas = ao2mo.restore('1', h2e_cas, norb_cas)
        h2e_cas = np.asarray(h2e_cas.transpose(0, 2, 3, 1), order='C')


    obi, tbi, core_energy, ferm_ham = self.generate_molecular_spin_ham_restricted_blocked(
        h1e_cas, h2e_cas, ecore)

    nelec = mol.nelectron
    norb = self.full_spatial_orbitals
    if nele_cas is None:

        return (obi, tbi, e_nn, nelec, norb, ferm_ham)

    else:

        return (obi, tbi, ecore, nele_cas, norb_cas, ferm_ham)


  def get_HF_trial(self,verbose=False):
      """Return a single‑determinant HF trial  (coeffs, occα‑list, occβ‑list)."""
      mf  = self.mf
      occ = np.where(mf.mo_occ > 1e-8)[0]          # all occupied spatial MOs

      # RHF 
      if hasattr(mf, "mo_occ") and mf.mol.spin == 0:
          occa = [occ]            # α and β identical
          occb = [occ]

      # ROHF 
      elif hasattr(mf, "mo_occ"):                  # mo_occ is 0/1/2
          occa = [occ]                             # α has all occupied
          occb = [occ[mf.mo_occ[occ] > 1.5]]       # β has only the doubly‑occ

      # UHF 
      else:                                        # mo_occ is a tuple (α,β)
          occa = [np.where(mf.mo_occ[0] > 1e-8)[0]]
          occb = [np.where(mf.mo_occ[1] > 1e-8)[0]]

      coeffs = np.array([1.0 + 0.0j])        
      return TrialWfn(coeffs=coeffs, occa=occa, occb=occb,mol_problem=self,verbose=verbose, compute_trial_energy=True)


  
  def get_FCI_energy(self):
     return self.get_FCI()[0]
  def get_FCI(self):
      if getattr(self, "FCI", None) is not None:
          return self.FCI

      mf  = self.mf
      mol = self.mol
      chk = Path(self.chk_file_path)

      # Current run's metadata
      mo_coeff = np.asarray(mf.mo_coeff)
      norb     = mo_coeff.shape[1]
      nelec    = mol.nelectron if isinstance(mol.nelectron, tuple) else (mol.nelectron//2, mol.nelectron - mol.nelectron//2)
      def _sha1(a: np.ndarray) -> str:
        a = np.ascontiguousarray(a)
        return hashlib.sha1(a.view(np.uint8)).hexdigest()
      meta_now = {
          "norb": int(norb),
          "nelec": tuple(nelec),
          "mo_sha1": _sha1(mo_coeff),
      }

      # Try to load from chk if compatible
      if chk.exists():
          try:
              meta_saved = lib.chkfile.load(str(chk), "fci/meta")
              if (int(meta_saved.get("norb", -1)) == meta_now["norb"] and
                  tuple(meta_saved.get("nelec", ())) == meta_now["nelec"] and
                  meta_saved.get("mo_sha1", "") == meta_now["mo_sha1"]):
                  efci   = float(lib.chkfile.load(str(chk), "fci/e_tot"))
                  fcivec = np.asarray(lib.chkfile.load(str(chk), "fci/ci"))
                  self.FCI = (efci, fcivec)
                  return self.FCI
          except Exception:
              pass  # group/keys not present → fall through to compute

      # Compute FCI in the current MO basis
      hcore_ao = mf.get_hcore()
      h1e = mo_coeff.T @ hcore_ao @ mo_coeff                     # (ij) in MO basis
      eri = ao2mo.kernel(mol, mo_coeff)                          # (pq|rs) in MO basis (chemist's)
      cisolver = fci.FCI(mol, mo_coeff)
      efci, fcivec = cisolver.kernel(h1e, eri, norb, nelec)
    #   if self.mpi_info.rank == 0:
    #   # Save to chk for reuse
    #     chk.parent.mkdir(parents=True, exist_ok=True)
    #     lib.chkfile.dump(str(chk), "fci/e_tot", float(efci))
    #     lib.chkfile.dump(str(chk), "fci/ci",   np.asarray(fcivec))
    #     lib.chkfile.dump(str(chk), "fci/meta", meta_now)

      self.FCI = (efci, fcivec)
      return self.FCI


    
  def get_CASSCF_trial(self, max_det: Optional[int] = None,verbose = False):

    from pyscf.mcscf import CASSCF
    from pyscf.fci.addons import large_ci
    # 1) Set up active space
    mf = self.mf
    M = self.active_orbitals
    N = self.active_electrons
    nocca,noccb = self.active_mol_nelec
    t0 = time.time()
    mc = mcscf.CASSCF(mf, M, N)

    mc.chkfile = "scf.chk"
    e_tot, e_cas, fcivec, mo, mo_energy = mc.kernel()
    print("PySCF CASSCF Energy: ", e_tot)
    self.casscf_energy = e_tot
    # print("CASSCF Energy: ",e_tot)
    # print("CASSCF runtime: ",time.time()-t0)
    coeffs, occa, occb = zip(
        *fci.addons.large_ci(fcivec, M, (nocca, noccb), tol=1e-8, return_strs=False)
    )
    trial_wfn = TrialWfn(coeffs=coeffs,occa=occa,occb = occb,mol_problem=self,max_det=max_det,verbose=verbose)
 
    return trial_wfn

  def get_CASCI_trial(self, max_det: Optional[int] = None,verbose =  False):
      from pyscf import mcscf
      from pyscf.fci.addons import large_ci
      import time

      # 1) Set up active space
      mf = self.mf
      M  = self.active_orbitals
      N  = self.active_electrons
      nocca, noccb = self.active_mol_nelec

      t0 = time.time()
      mc = mcscf.CASCI(mf, M, N)
      mc.chkfile = "scf.chk"

      # CASCI does not optimize orbitals; kernel returns energy and CI vector
      # (Return signature can include extras; unpack the ones we need.)
      e_tot, e_cas, fcivec, mo, mo_energy= mc.kernel()
      self.casci_energy = e_tot
      print("CASCI Energy: ", e_tot)
      # Build MSD trial from the CI expansion in the active space
      coeffs, occa, occb = zip(
          *large_ci(fcivec, M, (nocca, noccb), tol=1e-8, return_strs=False)
      )

      trial_wfn = TrialWfn(coeffs=coeffs,occa=occa,occb = occb,mol_problem=self,max_det=max_det,verbose=verbose)
  # keep same behavior as your CASSCF version
      return trial_wfn

  
  def get_FCI_trial(self, max_det: Optional[int] = None):
        """
        Build a trial wavefunction by doing FCI in the FULL MO space.

        Returns (coeffs, occa, occb) truncated to n_det determinants if requested.
        """
        efci, fcivec = self.get_FCI()
        print("FCI energy (full space):", efci)

        nmo   = self.full_spatial_orbitals
        nelec = self.mol_nelec

        # Extract the CI expansion
        coeffs, occa, occb = zip(
            *large_ci(fcivec, nmo, nelec, tol=0, return_strs=False)
        )
        print("FCI Pre-truncated # of determinants:", len(coeffs))

        # Convert to the same types you expect elsewhere
        coeffs = np.asarray(coeffs, dtype=np.complex128)
        occa   = list(occa)
        occb   = list(occb)

        trial_wfn = TrialWfn(
            coeffs=coeffs,
            occa=occa,
            occb=occb,
            mol_problem=self,
            max_det=max_det,
        )
        return trial_wfn

  def get_active_spatial(self):
      mol = self.mol
      norb_total = self.mf.mo_coeff.shape[1]                  # total spatial MOs
      n_alpha_act, n_beta_act = self.active_mol_nelec
      n_elec_total = sum(mol.nelec)                      # total electrons in full system
      n_elec_act   = n_alpha_act + n_beta_act
      n_elec_inactive = n_elec_total - n_elec_act

      # sanity checks mirroring Qiskit’s ActiveSpaceTransformer
      if n_elec_inactive < 0 or (n_elec_inactive % 2) != 0:
          raise ValueError(
              f"Invalid active electron count: total={n_elec_total}, active={n_elec_act} "
              f"→ inactive={n_elec_inactive} (must be ≥0 and even)."
          )

      n_core = n_elec_inactive // 2
      if n_core + self.active_orbitals > norb_total:
          raise ValueError(
              f"Requested {self.active_orbitals } active orbitals but only "
              f"{norb_total - n_core} are available above the frozen core."
          )

      # Canonical PySCF orbitals are already energy-ordered; choose the window above the core.
      active_spatial = list(range(n_core, n_core + self.active_orbitals ))
      return active_spatial
  



  def generate_molecular_spin_ham_restricted(self,h1e, h2e, ecore):

      # This function generates the molecular spin Hamiltonian
      # H = E_core+sum_{`pq`}  h_{`pq`} a_p^dagger a_q +
      #                          0.5 * h_{`pqrs`} a_p^dagger a_q^dagger a_r a_s
      # h1e: one body integrals h_{`pq`}
      # h2e: two body integrals h_{`pqrs`}
      # `ecore`: constant (nuclear repulsion or core energy in the active space Hamiltonian)

      # Total number of qubits equals the number of spin molecular orbitals
      nqubits = 2 * h1e.shape[0]

      # Initialization
      one_body_coeff = np.zeros((nqubits, nqubits))
      two_body_coeff = np.zeros((nqubits, nqubits, nqubits, nqubits))

      ferm_ham = []

      for p in range(nqubits // 2):
          for q in range(nqubits // 2):

              # p & q have the same spin <a|a>= <b|b>=1
              # <a|b>=<b|a>=0 (orthogonal)
              one_body_coeff[2 * p, 2 * q] = h1e[p, q]
              temp = str(h1e[p, q]) + ' a_' + str(p) + '^dagger ' + 'a_' + str(q)
              ferm_ham.append(temp)
              one_body_coeff[2 * p + 1, 2 * q + 1] = h1e[p, q]
              temp = str(h1e[p, q]) + ' b_' + str(p) + '^dagger ' + 'b_' + str(q)
              ferm_ham.append(temp)

              for r in range(nqubits // 2):
                  for s in range(nqubits // 2):

                      # Same spin (`aaaa`, `bbbbb`) <a|a><a|a>, <b|b><b|b>
                      two_body_coeff[2 * p, 2 * q, 2 * r,
                                    2 * s] = 0.5 * h2e[p, q, r, s]
                      temp = str(0.5 * h2e[p, q, r, s]) + ' a_' + str(
                          p) + '^dagger ' + 'a_' + str(
                              q) + '^dagger ' + 'a_' + str(r) + ' a_' + str(s)
                      ferm_ham.append(temp)
                      two_body_coeff[2 * p + 1, 2 * q + 1, 2 * r + 1,
                                    2 * s + 1] = 0.5 * h2e[p, q, r, s]
                      temp = str(0.5 * h2e[p, q, r, s]) + ' b_' + str(
                          p) + '^dagger ' + 'b_' + str(
                              q) + '^dagger ' + 'b_' + str(r) + ' b_' + str(s)
                      ferm_ham.append(temp)

                      # Mixed spin(`abab`, `baba`) <a|a><b|b>, <b|b><a|a>
                      #<a|b>= 0 (orthogonal)
                      two_body_coeff[2 * p, 2 * q + 1, 2 * r + 1,
                                    2 * s] = 0.5 * h2e[p, q, r, s]
                      temp = str(0.5 * h2e[p, q, r, s]) + ' a_' + str(
                          p) + '^dagger ' + 'a_' + str(
                              q) + '^dagger ' + 'b_' + str(r) + ' b_' + str(s)
                      ferm_ham.append(temp)
                      two_body_coeff[2 * p + 1, 2 * q, 2 * r,
                                    2 * s + 1] = 0.5 * h2e[p, q, r, s]
                      temp = str(0.5 * h2e[p, q, r, s]) + ' b_' + str(
                          p) + '^dagger ' + 'b_' + str(
                              q) + '^dagger ' + 'a_' + str(r) + ' a_' + str(s)
                      ferm_ham.append(temp)

      full_hamiltonian = " + ".join(ferm_ham)

      return one_body_coeff, two_body_coeff, ecore, full_hamiltonian
  




_HEADER_KEYS = ("NORB", "NELEC", "MS2", "ORBSYM", "ISYM")

def _canonicalize_header_block(lines: list[str]) -> str:
    """
    Given raw lines *between* &FCI and &END (exclusive), produce a single-line
    canonical header that PySCF can always parse.
    """
    # Collect key->raw value strings (no trailing commas), last one wins if repeated
    fields: dict[str, str] = {}

    def strip_trailing_comma(s: str) -> str:
        return re.sub(r',\s*$', '', s.strip())

    # Concatenate continuation lines that might belong to ORBSYM, etc.
    # Strategy: join all header lines into a single string, split on commas
    # only for separating fields, NOT inside a value. Since each line here
    # is “one field per line” in your files, it's safe to parse per-line.
    buf = []
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith('!'):  # skip blank/comment
            continue
        s = strip_trailing_comma(s)
        if not s:
            continue
        buf.append(s)

    # Now each s in buf should look like KEY=VALUE or KEY=val1,val2,...
    for s in buf:
        if '=' not in s:
            # if a bizarre orphan appears, skip it (prevents PySCF crash)
            continue
        key, val = s.split('=', 1)
        key = key.strip().upper()
        val = val.strip()
        if key in _HEADER_KEYS:
            # normalize whitespace inside value; keep commas in ORBSYM
            val = re.sub(r'\s+', '', val) if key == 'ORBSYM' else re.sub(r'\s+', ' ', val).strip()
            fields[key] = val

    # Required fields
    missing = [k for k in ("NORB", "NELEC", "MS2") if k not in fields]
    if missing:
        raise ValueError(f"FCIDUMP header missing required keys: {missing}")

    # Rebuild canonical single-line header
    parts = []
    for k in _HEADER_KEYS:
        if k in fields:
            parts.append(f"{k}={fields[k]}")
    canonical = "&FCI " + ", ".join(parts) + ", &END"
    return canonical

def _sanitize_fcidump_header_strict(in_path: str | pathlib.Path) -> str:
    """
    Create a temp copy with a canonical single-line header. Leaves integral body unchanged.
    """
    in_path = str(in_path)
    with open(in_path, 'r') as f:
        all_lines = f.readlines()

    # find &FCI ... &END region
    start = next((i for i, ln in enumerate(all_lines) if ln.strip().upper().startswith('&FCI')), None)
    end   = next((i for i, ln in enumerate(all_lines) if ln.strip().upper().startswith('&END')), None)
    if start is None or end is None or end <= start:
        # No header block—return original path
        return in_path

    # Build canonical header
    header_body = all_lines[start+1:end]
    canonical = _canonicalize_header_block(header_body)

    # Assemble new file: canonical header on one line; body unchanged
    out_lines = []
    out_lines.append(canonical + "\n")
    # Keep everything *outside* the old header block (body starts after &END)
    out_lines.extend(all_lines[end+1:])

    tmp = tempfile.NamedTemporaryFile('w', suffix='.fcidump', delete=False)
    tmp.write("".join(out_lines))
    tmp.flush()
    tmp.close()
    return tmp.name

def _read_fcidump_resilient(path: str | pathlib.Path):
    """
    Try PySCF read(); if header tokenization fails, canonicalize header and retry.
    """
    path = str(path)
    try:
        return pyscf_fcidump.read(path, verbose=False)
    except Exception:
        safe = _sanitize_fcidump_header_strict(path)
        return pyscf_fcidump.read(safe, verbose=False)
