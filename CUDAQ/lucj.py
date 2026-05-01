
import ffsim
from ffsim import gates, linalg, protocols
import cudaq
import itertools
import numpy as np
from typing import List, Tuple, Optional
from openfermion import FermionOperator
from openfermion.transforms.opconversions.term_reordering import normal_ordered
from ffsim.linalg import givens_decomposition
from openfermion.transforms import jordan_wigner
import time
from scipy.optimize import minimize
import math, cmath
from scipy.optimize import OptimizeResult

try:
    from mpi4py import MPI  # optional; only needed if mpi is used
except Exception:
    MPI = None

import cudaq, numpy as np, math, cmath
from ffsim.linalg import givens_decomposition
from typing import List, Tuple

# -----------------------
# Host-side param builders
# -----------------------


def _xx_yy_labels(n_qubits: int, i: int, j: int) -> tuple[str, str]:
    sX = ['I'] * n_qubits; sY = ['I'] * n_qubits
    sX[i] = 'X'; sX[j] = 'X'
    sY[i] = 'Y'; sY[j] = 'Y'
    return ''.join(sX), ''.join(sY)


def prepare_orbital_rotation_params(U: np.ndarray):
    rots, diag_phases = givens_decomposition(U)
    thetas, deltas, is_, js_ = [], [], [], []
    for c, s, i, j in rots:
        thetas.append(2.0 * math.acos(float(c)))
        deltas.append(cmath.phase(s) - 0.5 * math.pi)
        is_.append(int(i)); js_.append(int(j))
    diag = [float(cmath.phase(z)) for z in np.asarray(diag_phases).ravel()]

    return thetas, deltas, is_, js_, diag

def prepare_all_orbital_layers_with_pws(orbital_rotations: np.ndarray, inverse:bool):
    """
    Returns per layer l:
      thetasL[l], deltasL[l], isL[l], jsL[l], diagL[l],
      xx_aL[l], yy_aL[l], xx_bL[l], yy_bL[l]  (as lists of cudaq.pauli_word)
    """
    L, norb, _ = orbital_rotations.shape
    n_qubits = 2 * norb

    thetasL, deltasL, isL, jsL, diagL = [], [], [], [], []
    xx_aL, yy_aL, xx_bL, yy_bL = [], [], [], []

    for l in range(L):
        if inverse:
            thetas, deltas, is_, js_, diag = prepare_orbital_rotation_params(orbital_rotations[l].T.conj())
        else:
            thetas, deltas, is_, js_, diag = prepare_orbital_rotation_params(orbital_rotations[l])

        # Build Pauli words (host-side)
        xx_a, yy_a, xx_b, yy_b = [], [], [], []
        for i, j in zip(is_, js_):
            sx, sy = _xx_yy_labels(n_qubits, i, j)               # alpha
            xx_a.append(cudaq.pauli_word(sx))
            yy_a.append(cudaq.pauli_word(sy))
            sx, sy = _xx_yy_labels(n_qubits, i + norb, j + norb) # beta
            xx_b.append(cudaq.pauli_word(sx))
            yy_b.append(cudaq.pauli_word(sy))

        thetasL.append([float(t) for t in thetas])
        deltasL.append([float(d) for d in deltas])
        isL.append(is_); jsL.append(js_)
        diagL.append([float(x) for x in diag])
        xx_aL.append(xx_a); yy_aL.append(yy_a)
        xx_bL.append(xx_b); yy_bL.append(yy_b)

    return (thetasL, deltasL, isL, jsL, diagL, xx_aL, yy_aL, xx_bL, yy_bL)



def prepare_diag_coulomb_num_rep_params(norb: int, mat, time: float):
    """
    Returns parallel lists:
      singles_qs, singles_angles,
      pairs_qi, pairs_qj, pairs_phis
    """
    if isinstance(mat, np.ndarray) and mat.ndim == 2:
        mat_aa = mat_ab = mat_bb = mat
    else:
        mat_aa, mat_ab, mat_bb = mat

    singles_qs, singles_angles = [], []
    pairs_qi, pairs_qj, pairs_phis = [], [], []

    # aa
    if mat_aa is not None:
        for i in range(norb):
            z = float(mat_aa[i, i])
            if z:
                singles_qs.append(i); singles_angles.append(-0.5 * z * time)
        for i in range(norb):
            for j in range(i + 1, norb):
                z = float(mat_aa[i, j])
                if z:
                    pairs_qi.append(i); pairs_qj.append(j); pairs_phis.append(-z * time)

    # bb
    if mat_bb is not None:
        for i in range(norb):
            z = float(mat_bb[i, i])
            if z:
                singles_qs.append(i + norb); singles_angles.append(-0.5 * z * time)
        for i in range(norb):
            for j in range(i + 1, norb):
                z = float(mat_bb[i, j])
                if z:
                    pairs_qi.append(i + norb); pairs_qj.append(j + norb); pairs_phis.append(-z * time)

    # ab
    if mat_ab is not None:
        for i in range(norb):
            z = float(mat_ab[i, i])
            if z:
                pairs_qi.append(i); pairs_qj.append(i + norb); pairs_phis.append(-z * time)
        for i in range(norb):
            for j in range(i + 1, norb):
                z1 = float(mat_ab[i, j]); z2 = float(mat_ab[j, i])
                if z1:
                    pairs_qi.append(i); pairs_qj.append(j + norb); pairs_phis.append(-z1 * time)
                if z2:
                    pairs_qi.append(j); pairs_qj.append(i + norb); pairs_phis.append(-z2 * time)

    return singles_qs, singles_angles, pairs_qi, pairs_qj, pairs_phis

def prepare_all_diag_layers_numrep(norb: int, diag_coulomb_layers, time: float):
    K = diag_coulomb_layers.shape[0]
    singles_qsL, singles_angL = [], []
    zz_phisL, zz_qiL, zz_qjL = [], [], []

    for l in range(K):
        block = diag_coulomb_layers[l]
        if block.ndim == 2:
            mat_aa = mat_ab = mat_bb = block
        elif block.ndim == 3:
            if block.shape[0] == 2:
                mat_aa, mat_ab = block[0], block[1]; mat_bb = mat_aa
            elif block.shape[0] == 3:
                mat_aa, mat_ab, mat_bb = block[0], block[1], block[2]
            else:
                raise ValueError(...)
        else:
            raise ValueError(...)

        sq, sa, pqi, pqj, pphi = prepare_diag_coulomb_num_rep_params(
            norb, (mat_aa, mat_ab, mat_bb), time
        )
        singles_qsL.append([int(q) for q in sq])
        singles_angL.append([float(a) for a in sa])
        zz_qiL.append([int(q) for q in pqi])
        zz_qjL.append([int(q) for q in pqj])
        zz_phisL.append([float(p) for p in pphi])

    return singles_qsL, singles_angL, zz_phisL, zz_qiL, zz_qjL

def pack_layers(layers):
    flat, starts, sizes = [], [], []
    cur = 0
    for L in layers:
        starts.append(cur)
        sz = len(L)
        sizes.append(sz)
        flat.extend(L)
        cur += sz
    return flat, starts, sizes




     

@cudaq.kernel
def hf_prepare(q: cudaq.qview, norb: int, n_alpha: int, n_beta: int):
    for i in range(n_alpha): x(q[i])
    for j in range(n_beta):  x(q[norb + j])

@cudaq.kernel
def orbital_rot_layer_view(
    q: cudaq.qview,
    norb: int,
    thetas: list[float], deltas: list[float],
    is_: list[int], js_: list[int], diag: list[float],
    xx_a: list[cudaq.pauli_word], yy_a: list[cudaq.pauli_word],
    xx_b: list[cudaq.pauli_word], yy_b: list[cudaq.pauli_word],
    ts: int, tz: int, ds: int, is0: int, js0: int,
    dg0: int, dgz: int, xa0: int, ya0: int, xb0: int, yb0: int
):
    # alpha spin
    for k in range(tz):
        i = is_[is0 + k]
        j = js_[js0 + k]
        th = thetas[ts + k]
        beta = deltas[ds + k]

        rz(beta, q[i])

        exp_pauli(-th / 4.0, q, xx_a[xa0 + k])
        exp_pauli(-th / 4.0, q, yy_a[ya0 + k])

        # RZ_i(-beta)
        rz(-beta, q[i])

    # beta spin
    for k in range(tz):
        i = is_[is0 + k] + norb
        j = js_[js0 + k] + norb
        th = thetas[ts + k]
        beta = deltas[ds + k]
        # originally run rz on qubit i
        rz(beta, q[i])
        exp_pauli(-th / 4.0, q, xx_b[xb0 + k])
        exp_pauli(-th / 4.0, q, yy_b[yb0 + k])
        rz(-beta, q[i])
    # phases shifts
    for p in range(dgz):                         
        phi = diag[dg0 + p]
        rz(phi, q[p])       
        rz(phi, q[p + norb]) 


@cudaq.kernel
def final_orbital_rotation(
    q: cudaq.qview,
    norb: int,
    # rotation parameters
    thetasF: list[float],    # length = final_tz
    deltasF: list[float],    # length = final_tz
    isF:     list[int],      # length = final_tz
    jsF:     list[int],      # length = final_tz
    diagF:   list[float],    # length = final_dgz (typically = norb)
    # Pauli words for α and β spin blocks (aligned 1:1 with thetasF/isF/jsF)
    xx_aF: list[cudaq.pauli_word],
    yy_aF: list[cudaq.pauli_word],
    xx_bF: list[cudaq.pauli_word],
    yy_bF: list[cudaq.pauli_word],
    # explicit counts to avoid len() in the kernel
    final_tz: int,
    final_dgz: int
):

    for k in range(final_tz):
        i = isF[k]
        # j = jsF[k]  # not used explicitly; encoded in the pauli words
        th   = thetasF[k]
        beta = deltasF[k]

        # RZ_i(β)
        rz(beta, q[i])
        # exp{-i (θ/4) XX} · exp{-i (θ/4) YY}
        exp_pauli(-th / 4.0, q, xx_aF[k])
        exp_pauli(-th / 4.0, q, yy_aF[k])
        # RZ_i(-β)
        rz(-beta, q[i])

    for k in range(final_tz):
        i = isF[k] + norb
        th   = thetasF[k]
        beta = deltasF[k]

        rz(beta, q[i])
        exp_pauli(-th / 4.0, q, xx_bF[k])
        exp_pauli(-th / 4.0, q, yy_bF[k])
        rz(-beta, q[i])

    for p in range(final_dgz):     
        phi = diagF[p]
        rz(phi, q[p])         # α
        rz(phi, q[p + norb])  # β



@cudaq.kernel
def diag_coulomb_layer_view(
    q: cudaq.qview,
    s_qs: list[int], s_ang: list[float],
    z_phi: list[float], z_qi: list[int], z_qj: list[int],
    s0: int, sn: int, z0: int, zn: int
):
    for k in range(sn):
        rz(s_ang[s0 + k], q[s_qs[s0 + k]])
    for k in range(zn):
        a = z_qi[z0 + k]; b = z_qj[z0 + k]; lam = z_phi[z0 + k]
        rz(lam/2.0, q[a]); cx(q[a], q[b]); rz(-lam/2.0, q[b]); cx(q[a], q[b]); rz(lam/2.0, q[b])



@cudaq.kernel
def lucj_circuit(
    norb: int, n_alpha: int, n_beta: int,
    # K = number of UCJ layers (explicit)
    K_layers: int,

    # orbital rot layers (flat + offsets)
    thetasi: list[float],  thetas_sti: list[int],  thetas_szi: list[int],
    deltasi: list[float],  deltas_sti: list[int],  deltas_szi: list[int],
    is_i:   list[int],     is_sti:     list[int],  is_szi:     list[int],
    js_i:   list[int],     js_sti:     list[int],  js_szi:     list[int],
    diagi:  list[float],   diag_sti:   list[int],  diag_szi:   list[int],
    xx_ai:  list[cudaq.pauli_word], xx_a_sti: list[int], xx_a_szi: list[int],
    yy_ai:  list[cudaq.pauli_word], yy_a_sti: list[int], yy_a_szi: list[int],
    xx_bi:  list[cudaq.pauli_word], xx_b_sti: list[int], xx_b_szi: list[int],
    yy_bi:  list[cudaq.pauli_word], yy_b_sti: list[int], yy_b_szi: list[int],

    # diag Coulomb per layer (flat + offsets)
    s_qs: list[int], s_ang: list[float], s_st: list[int], s_sz: list[int], z_phi: list[float],
    z_qi: list[int], z_qj: list[int], z_st: list[int], z_sz: list[int],

    thetas: list[float],  thetas_st: list[int],  thetas_sz: list[int],
    deltas: list[float],  deltas_st: list[int],  deltas_sz: list[int],
    is_:   list[int],     is_st:     list[int],  is_sz:     list[int],
    js_:   list[int],     js_st:     list[int],  js_sz:     list[int],
    diag:  list[float],   diag_st:   list[int],  diag_sz:   list[int],
    xx_a:  list[cudaq.pauli_word], xx_a_st: list[int], xx_a_sz: list[int],
    yy_a:  list[cudaq.pauli_word], yy_a_st: list[int], yy_a_sz: list[int],
    xx_b:  list[cudaq.pauli_word], xx_b_st: list[int], xx_b_sz: list[int],
    yy_b:  list[cudaq.pauli_word], yy_b_st: list[int], yy_b_sz: list[int],

    # final rotation (flat view) + explicit counts to avoid len()
    thetasF: list[float], deltasF: list[float], isF: list[int], jsF: list[int], diagF: list[float],
    xx_aF: list[cudaq.pauli_word], yy_aF: list[cudaq.pauli_word],
    xx_bF: list[cudaq.pauli_word], yy_bF: list[cudaq.pauli_word],
    final_tz: int, final_dgz: int
):
    q = cudaq.qvector(2 * norb)
    hf_prepare(q, norb, n_alpha, n_beta)

    # loop with explicit K_layers, not len(thetas_st)
    for l in range(K_layers):
        orbital_rot_layer_view(
            q, norb,
            thetasi, deltasi, is_i, js_i, diagi,
            xx_ai, yy_ai, xx_bi, yy_bi,
            thetas_sti[l], thetas_szi[l],   # ts, tz
            deltas_sti[l],                 # ds
            is_sti[l],                     # is0
            js_sti[l],                     # js0
            diag_sti[l], diag_szi[l],       # dg0, dgz
            xx_a_sti[l], yy_a_sti[l],       # xa0, ya0
            xx_b_sti[l], yy_b_sti[l]        # xb0, yb0
        )
        diag_coulomb_layer_view(
            q,
            s_qs, s_ang, z_phi, z_qi, z_qj,
            s_st[l], s_sz[l],
            z_st[l], z_sz[l]
        )
        orbital_rot_layer_view(
            q, norb,
            thetas, deltas, is_, js_, diag,
            xx_a, yy_a, xx_b, yy_b,
            thetas_st[l], thetas_sz[l],   # ts, tz
            deltas_st[l],                 # ds
            is_st[l],                     # is0
            js_st[l],                     # js0
            diag_st[l], diag_sz[l],       # dg0, dgz
            xx_a_st[l], yy_a_st[l],       # xa0, ya0
            xx_b_st[l], yy_b_st[l]        # xb0, yb0
        )

    # always call final rotation once; it’s a no-op if final_tz=0 and final_dgz=0
    final_orbital_rotation(q, norb,
                        thetasF, deltasF, isF, jsF, diagF,
                        xx_aF, yy_aF, xx_bF, yy_bF,
                        final_tz, final_dgz)



    


def build_lucj_packed_args(ucj_op, norb: int, time: float = -1.0):
    U_layers = ucj_op.orbital_rotations          # (K, norb, norb)
    Z_layers = ucj_op.diag_coulomb_mats          # (K, 2, norb, norb) or similar
    U_final  = ucj_op.final_orbital_rotation     # (norb, norb) or None

    K_layers = int(U_layers.shape[0])  # explicit

    # ---- orbital layers (unchanged) ----
    (thetasLi, deltasLi, isLi, jsLi, diagLi,
     xx_aLi, yy_aLi, xx_bLi, yy_bLi) = prepare_all_orbital_layers_with_pws(U_layers,inverse=True)

    thetasi, thetas_sti, thetas_szi = pack_layers(thetasLi)
    deltasi, deltas_sti, deltas_szi = pack_layers(deltasLi)
    is_fi,   is_sti,     is_szi     = pack_layers(isLi)
    js_fi,   js_sti,     js_szi     = pack_layers(jsLi)
    diag_fi, diag_sti,   diag_szi   = pack_layers(diagLi)
    xx_a_fi, xx_a_sti,   xx_a_szi   = pack_layers(xx_aLi)
    yy_a_fi, yy_a_sti,   yy_a_szi   = pack_layers(yy_aLi)
    xx_b_fi, xx_b_sti,   xx_b_szi   = pack_layers(xx_bLi)
    yy_b_fi, yy_b_sti,   yy_b_szi   = pack_layers(yy_bLi)

    # ---- diagonal Coulomb (unchanged) ----
    (singles_qsL, singles_angL,  zz_phisL, zz_qiL, zz_qjL) = prepare_all_diag_layers_numrep(norb, Z_layers,time)


    s_qs, s_st, s_sz   = pack_layers(singles_qsL)
    s_ang, _, _        = pack_layers(singles_angL)
    z_phi, z_st, z_sz  = pack_layers(zz_phisL)
    z_qi,  _, _        = pack_layers(zz_qiL)
    z_qj,  _, _        = pack_layers(zz_qjL)

    (thetasL, deltasL, isL, jsL, diagL,
     xx_aL, yy_aL, xx_bL, yy_bL) = prepare_all_orbital_layers_with_pws(U_layers,inverse=False)

    thetas, thetas_st, thetas_sz = pack_layers(thetasL)
    deltas, deltas_st, deltas_sz = pack_layers(deltasL)
    is_f,   is_st,     is_sz     = pack_layers(isL)
    js_f,   js_st,     js_sz     = pack_layers(jsL)
    diag_f, diag_st,   diag_sz   = pack_layers(diagL)
    xx_a_f, xx_a_st,   xx_a_sz   = pack_layers(xx_aL)
    yy_a_f, yy_a_st,   yy_a_sz   = pack_layers(yy_aL)
    xx_b_f, xx_b_st,   xx_b_sz   = pack_layers(xx_bL)
    yy_b_f, yy_b_st,   yy_b_sz   = pack_layers(yy_bL)

    # ---- final rotation (provide counts explicitly, avoid len() in kernel) ----
    if U_final is None:
        thetasF = []; deltasF = []; isF = []; jsF = []; diagF = []
        xx_aF = []; yy_aF = []; xx_bF = []; yy_bF = []
        final_tz = 0
        final_dgz = 0
    else:
        t, d, iL, jL, dg = prepare_orbital_rotation_params(U_final)
        n_qubits = 2 * norb
        xx_aF = []; yy_aF = []; xx_bF = []; yy_bF = []
        for i, j in zip(iL, jL):
            sx, sy = _xx_yy_labels(n_qubits, i, j)
            xx_aF.append(cudaq.pauli_word(sx)); yy_aF.append(cudaq.pauli_word(sy))
            sx, sy = _xx_yy_labels(n_qubits, i + norb, j + norb)
            xx_bF.append(cudaq.pauli_word(sx)); yy_bF.append(cudaq.pauli_word(sy))
        thetasF = [float(x) for x in t]
        deltasF = [float(x) for x in d]
        isF     = [int(x) for x in iL]
        jsF     = [int(x) for x in jL]
        diagF   = [float(x) for x in dg]
        final_tz  = len(thetasF)
        final_dgz = len(diagF)

    args = (
        K_layers,
        # orbital inverse
        thetasi, thetas_sti, thetas_szi,
        deltasi, deltas_sti, deltas_szi,
        is_fi,   is_sti,     is_szi,
        js_fi,   js_sti,     js_szi,
        diag_fi, diag_sti,   diag_szi,
        xx_a_fi, xx_a_sti,   xx_a_szi,
        yy_a_fi, yy_a_sti,   yy_a_szi,
        xx_b_fi, xx_b_sti,   xx_b_szi,
        yy_b_fi, yy_b_sti,   yy_b_szi,
        # diag Coulomb
        s_qs, s_ang, s_st, s_sz,
        z_phi, z_qi, z_qj, z_st, z_sz,
        # orbital
        thetas, thetas_st, thetas_sz,
        deltas, deltas_st, deltas_sz,
        is_f,   is_st,     is_sz,
        js_f,   js_st,     js_sz,
        diag_f, diag_st,   diag_sz,
        xx_a_f, xx_a_st,   xx_a_sz,
        yy_a_f, yy_a_st,   yy_a_sz,
        xx_b_f, xx_b_st,   xx_b_sz,
        yy_b_f, yy_b_st,   yy_b_sz,

        # final
        thetasF, deltasF, isF, jsF, diagF, xx_aF, yy_aF, xx_bF, yy_bF,
        final_tz, final_dgz
    )
    # print("[CHK] first five (θ,β):", list(zip(thetas[:5], deltas[:5])))
    # print("[CHK] sample pair (i,j):", list(zip(is_f[:5], js_f[:5])))
    # print("[CHK] first five pair λ (diag Coulomb):", z_phi[:5])
    return args


def twoq_count_lucj_from_packed(lucj_packed_args):
    """
    Compute naive two-qubit gate count for lucj_circuit
    using ONLY the packed kernel inputs.

    lucj_packed_args is what build_lucj_packed_args(...) returns,
    i.e. the tuple that lucj_circuit receives after (norb, nα, nβ).
    """
    (
        K_layers,

        # inverse orbital flats + per-layer sizes
        thetasi, thetas_sti, thetas_szi,
        deltasi, deltas_sti, deltas_szi,
        is_fi,   is_sti,     is_szi,
        js_fi,   js_sti,     js_szi,
        diagi,   diag_sti,   diag_szi,
        xx_ai,   xx_a_sti,   xx_a_szi,
        yy_ai,   yy_a_sti,   yy_a_szi,
        xx_bi,   xx_b_sti,   xx_b_szi,
        yy_bi,   yy_b_sti,   yy_b_szi,

        # diag Coulomb flats + per-layer sizes
        s_qs, s_ang, s_st, s_sz,
        z_phi, z_qi, z_qj, z_st, z_sz,

        # forward orbital flats + per-layer sizes
        thetas, thetas_st, thetas_sz,
        deltas, deltas_st, deltas_sz,
        is_f,   is_st,     is_sz,
        js_f,   js_st,     js_sz,
        diag,   diag_st,   diag_sz,
        xx_a,   xx_a_st,   xx_a_sz,
        yy_a,   yy_a_st,   yy_a_sz,
        xx_b,   xx_b_st,   xx_b_sz,
        yy_b,   yy_b_st,   yy_b_sz,

        # final rotation + explicit counts
        thetasF, deltasF, isF, jsF, diagF,
        xx_aF, yy_aF, xx_bF, yy_bF,
        final_tz, final_dgz
    ) = lucj_packed_args
    print("K_layers =", K_layers)
    print("tz_i =", thetas_szi[0])
    print("tz_f =", thetas_sz[0])
    print("zn   =", z_sz[0])
    print("final_tz =", final_tz)

    total_twoq = 0

    for l in range(K_layers):
        tz_i = thetas_szi[l]   # number of Givens in inverse orbital layer l
        tz_f = thetas_sz[l]    # number of Givens in forward orbital layer l
        zn   = z_sz[l]         # number of ZZ pairs in Coulomb layer l

        total_twoq += 8 * (tz_i + tz_f) + 2 * zn

    total_twoq += 8 * final_tz

    return total_twoq

















class LUCJBatchedGradOptimizer:
    def __init__(
        self,
        *,
        kernel,
        hamiltonian,
        init_theta,
        interaction_pairs,
        norb,
        mol_nelec,
        n_reps,
        econst,
        max_walltime = None,
        epsilon=1e-3,
        optimizer_method="L-BFGS-B",
        opt_tol=1e-10,
        max_iters = None,
        mpi_info=None,              # <- optional: {'comm': ..., 'rank': int, 'size': int}
        qpus_per_rank=4,
        verbose = True       # <- max concurrent QPUs this rank will use
    ):
        self.kernel = kernel
        self.hamiltonian = hamiltonian
        self.interaction_pairs = interaction_pairs
        self.init_theta = init_theta
        self.norb =  norb
        self.n_reps = n_reps
        self.mol_nelec  = mol_nelec
        self.verbose= verbose
        # self.init_theta = np.asarray(init_theta, dtype=float).ravel()
        self.econst = float(econst)
        self.epsilon = float(epsilon)
        self.optimizer_method = optimizer_method
        self.opt_tol = float(opt_tol)
        self.max_iters = max_iters
        # Optional MPI context
        self.mpi = mpi_info
        self.max_walltime = max_walltime

        if self.mpi is not None and MPI is None:
            raise RuntimeError("mpi4py is required when 'mpi' context is provided.")

        # Local QPU discovery (per-process / per-node)
        self.num_qpus = cudaq.get_target().num_qpus()
        self.qpus_avail = max(1, min(int(qpus_per_rank), int(self.num_qpus)))

        if self.mpi is None:
            # print("IN GRADIENT CALCULTAION - NUM-QPU: ", self.num_qpus)
            self.energy_evals = {r:0 for r in range(1)}
            self.rank = 0
            self.szie = 1
        else:
            self.energy_evals = {r:0 for r in range(self.mpi.size)}
            self.rank = self.mpi.rank
            self.size = self.mpi.size
            # print("IN GRADIENT Calculation")
            # if self.mpi.rank == 0:
                
            #     print(f"[MPI size={self.mpi.size}] Rank 0 sees {self.num_qpus} QPUs; using up to {self.qpus_avail} per rank.")
            # else:
            #     print(f"[rank {self.mpi.rank}] local QPUs: {self.num_qpus}; using up to {self.qpus_avail}")

        # ---- timing & counters ----
        self.exp_vals = []
        self.time_cost_total = 0.0
        self.time_grad_total = 0.0
        self.num_cost_calls = 0
        self.num_grad_calls = 0
        self.total_energy_evals = 0
        self.iter = 1

        # ---------------------------
    def gather_eval_counts(self):
        if self.mpi is None:
            return {"total": self.total_energy_evals, "per_rank": {0: self.total_energy_evals}}

        comm, rank, size = self.mpi.comm, self.mpi.rank, self.mpi.size
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
    # ------------ helpers ------------
    def _my_indices(self, n):
        """Shard indices across ranks; in serial, take all."""
        if self.mpi is None:
            return range(n)
        r, p = self.mpi.rank, self.mpi.size
        return range(r, n, p)

    # ------------ core evals ------------


    def _batched_gradient(self, x: np.ndarray, indices=None) -> np.ndarray:
        """
        Central-difference gradient. If 'indices' is provided, only compute those
        components and return a full-size vector with zeros elsewhere.
        """
        x = np.asarray(x, dtype=float).ravel()
        n = x.size
        eps = self.epsilon
        if indices is None:
            indices = range(n)

        # Build perturbed points only for selected indices
        futures_p, futures_m = [], []
        idx_list = []
        qid = 0
        # print(f"Rank/Node: {self.mpi.rank}, parameter indicies: {indices} ")
        for i in indices:
            ei = np.zeros(n)
            ei[i] = 1.0
            xp = x + eps * ei
            xm = x - eps * ei

            f_plus = self._observe_energy_call(xp,qpu_id=(qid%self.qpus_avail))
            self.energy_evals[self.rank] += 1
            qid += 1

            f_minus =  self._observe_energy_call(xm,qpu_id=(qid%self.qpus_avail))
            self.energy_evals[self.rank] += 1
            qid += 1

            futures_p.append(f_plus)
            futures_m.append(f_minus)
            idx_list.append(i)

        # Gather and assemble full gradient vector (zeros outside my indices)
        g_full = np.zeros(n, dtype=float)
        for i, f_p, f_m in zip(idx_list, futures_p, futures_m):
            ep = f_p.get().expectation()
            em = f_m.get().expectation()
            self.total_energy_evals += 2
            g_full[i] = (ep - em) / (2.0 * eps)

        return g_full
    
    def _observe_energy_call(self, x, qpu_id=None):
        ucj_op = ffsim.UCJOpSpinBalanced.from_parameters(
            x, norb=self.norb, n_reps=self.n_reps,
            interaction_pairs=self.interaction_pairs,
            with_final_orbital_rotation=True
        )
        lucj_args = build_lucj_packed_args(ucj_op, self.norb)

        # DEBUG PRINTS (safe, brief)
        # print("\n[DBG] About to call cudaq.observe for LUCJ...")
        # _debug_dump_packing(self.norb, self.mol_nelec[0], self.mol_nelec[1], lucj_args)

        args = (self.norb, self.mol_nelec[0], self.mol_nelec[1], *lucj_args)

        if qpu_id is None:
            return cudaq.observe(lucj_circuit, self.hamiltonian, *args)
        else:
            return cudaq.observe_async(lucj_circuit, self.hamiltonian, *args, qpu_id=qpu_id)

    # ------------ public API used by SciPy ------------
    def cost(self, x: np.ndarray) -> float:
        if self.mpi is None:
            return float(self._observe_energy_call(x).expectation())
        # MPI: only root needs the value; workers never call cost().
        if self.mpi.rank == 0:
            return float(self._observe_energy_call(x).expectation())

        else:
            return 0.0

    def jac(self, x: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()

        if self.mpi is None:
            g = self._batched_gradient(x)
            self.time_grad_total += (time.perf_counter() - t0)
            self.num_grad_calls += 1
            return g

        # MPI mode: root orchestrates; workers are in a loop inside optimize()
        comm = self.mpi.comm; rank = self.mpi.rank

        # Notify workers we are about to do a JAC step, then broadcast x
        comm.bcast("JAC", root=0)
        x = np.asarray(x, float).ravel()
        comm.bcast(x, root=0)

        # Root also computes its shard; workers do theirs in their loop, not here.
        # To keep a single code path, root computes its local shard and participates
        # in a Reduce(SUM) to assemble the full gradient.
        n = x.size
        g_local = self._batched_gradient(x, indices=self._my_indices(n))

        g_full = np.zeros(n, dtype=float) if rank == 0 else None
        comm.Reduce([g_local, MPI.DOUBLE], [g_full, MPI.DOUBLE] if rank == 0 else None,
                    op=MPI.SUM, root=0)

        self.time_grad_total += (time.perf_counter() - t0)
        if rank == 0:
            self.num_grad_calls += 1
            return g_full
        else:
            # Return value is ignored by SciPy on non-root; still return something
            return g_local
    def _make_callback_intermediate(self, *, max_walltime=None):
        self.start = time.perf_counter()

        # store last iterate so we can build a partial OptimizeResult if we stop
        self._last_intermediate = None
        self._stopped_reason = None

        def callback(*, intermediate_result: OptimizeResult):
            # record & (optionally) keep your own trace
            self._last_intermediate = intermediate_result
            xk = intermediate_result.x
            self.callback(xk)
            if self.verbose:  # your existing bookkeeping (appends exp_vals, etc.)
                print(f"Iteration  {self.iter}, Energy = {self.exp_vals[self.iter]},  Time  = {time.perf_counter()-self.start}")
            self.iter += 1
            # budgets
            over_wall = (max_walltime is not None and
                         time.perf_counter() - self.start >= max_walltime)
            if over_wall:
                self._stopped_reason = "walltime"
                raise StopIteration  # <- mandated by docs to terminate
        return callback
    def callback(self, xk, *_, **__):
        # Only rank 0 keeps a trace in MPI mode
        if (self.mpi is None) or (self.mpi.rank == 0):
            self.exp_vals.append(self.cost(xk) + self.econst)

    def timing_summary(self) -> dict:
        return {
            "time_cost_total": self.time_cost_total,
            "time_grad_total": self.time_grad_total,
            "time_quantum_total": self.time_cost_total + self.time_grad_total,
            "num_cost_calls": self.num_cost_calls,
            "num_grad_calls": self.num_grad_calls,
            "avg_cost_time": (self.time_cost_total / self.num_cost_calls) if self.num_cost_calls else 0.0,
            "avg_grad_time": (self.time_grad_total / self.num_grad_calls) if self.num_grad_calls else 0.0,
            "num_energy_evals": self.total_energy_evals,
        }

    def optimize(self):
        theta0 = self.init_theta.copy()
        if self.max_iters is not None:
            options = {'maxiter':self.max_iters}
        else:
            options = None
        # record initial value (not included in avg timing below)
        if (self.mpi is None) or (self.mpi.rank == 0):
            self.exp_vals.append(self.cost(theta0) + self.econst)
            if self.verbose:
                print(f"Iteration  0, Energy = {self.exp_vals[0]},  Time  = 0")

        eval_times = []  # backward-compat

        def timed_cost(x):
            t0 = time.perf_counter()
            val = self.cost(x)
            dt = time.perf_counter() - t0
            eval_times.append(dt)
            self.time_cost_total += dt
            self.num_cost_calls += 1
            return val
        # -------- MPI path --------
        comm = self.mpi.comm; rank = self.mpi.rank

        if rank == 0:
            cb = self._make_callback_intermediate(max_walltime=self.max_walltime)
            # start worker loops by broadcasting a no-op so they enter bcast
            # (not strictly necessary, workers will block waiting for first cmd)
            # Run SciPy on root only
            try:
                result = minimize(
                    fun=timed_cost,   # no broadcasts for cost
                    x0=theta0,
                    method=self.optimizer_method,
                    jac=self.jac,     # jac will handle "JAC" broadcasts to workers
                    callback=cb,
                    tol=self.opt_tol,
                    options=options
                )
            except StopIteration:
    
                    ir = getattr(self, "_last_intermediate", None)
                    if ir is not None:
                        result = OptimizeResult(x=ir.x, fun=ir.fun, success=False,
                                                message=f"Stopped early ({self._stopped_reason}).")
                    else:
                        # fall back to current theta0 if somehow nothing was recorded
                        result = OptimizeResult(x=theta0, fun=self.cost(theta0), success=False,
                                                message=f"Stopped early ({self._stopped_reason}).")
            finally:
                # Tell workers to stop
                comm.bcast("STOP", root=0)

            self.exp_vals.append(self.cost(result.x) + self.econst)
            return result, np.array(self.exp_vals, dtype=float)

        else:
            # Worker loop: respond to root's commands
            while True:
                cmd = comm.bcast(None, root=0)  # "JAC" or "STOP"
                if cmd == "STOP":
                    break
                elif cmd == "JAC":
                    x = comm.bcast(None, root=0)  # receive x
                    n = x.size
                    g_local = self._batched_gradient(x, indices=self._my_indices(n))
                    comm.Reduce([g_local, MPI.DOUBLE], None, op=MPI.SUM, root=0)
                else:
                    pass

            # Return a placeholder result to keep API consistent on workers
            class _Res: pass
            result = _Res()
            result.x = theta0
            result.fun = self.cost(theta0)
            result.success = True
            return result, np.array(self.exp_vals, dtype=float)

