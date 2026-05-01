import json
from pathlib import Path
from typing import Sequence

# Minimal periodic table mapping (index = Z)
PERIODIC = [
    None, "H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar","K","Ca",
    "Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr",
    "Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe","Cs","Ba","La","Ce","Pr","Nd",
    "Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg",
    "Tl","Pb","Bi","Po","At","Rn","Fr","Ra","Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm",
    "Md","No","Lr","Rf","Db","Sg","Bh","Hs","Mt","Ds","Rg","Cn","Nh","Fl","Mc","Lv","Ts","Og"
]

def parse_pubchem_json(
    path: str | Path,
    compound_index: int = 0,
    coords_set_index: int = 0,
    conformer_index: int = 0,
    float_fmt: str = "g",      # e.g., ".6f" if you want fixed decimals
    sep: str = "; "
) -> str:
    """
    Parse a PubChem-style JSON (as shown) and return:
      'H x y z; C x y z; ...'

    Args:
        path: Path to the JSON file.
        compound_index: Which compound in PC_Compounds to use.
        coords_set_index: Which coords set to use (e.g., first entry in 'coords').
        conformer_index: Which conformer (x/y/z triplet) to use.
        float_fmt: Python format spec for coordinates (default 'g').
        sep: Separator between atoms in the output string.

    Returns:
        A single string joining 'SYMBOL x y z' with `sep`.
    """
    with open(path, "r") as f:
        data = json.load(f)

    compounds = data.get("PC_Compounds")
    if not compounds:
        raise ValueError("No 'PC_Compounds' found in JSON.")
    compound = compounds[compound_index]

    atoms = compound.get("atoms")
    if not atoms:
        raise ValueError("No 'atoms' section found.")

    atom_aids: Sequence[int] = atoms.get("aid", [])
    atom_Z:   Sequence[int] = atoms.get("element", [])
    if not atom_aids or not atom_Z or len(atom_aids) != len(atom_Z):
        raise ValueError("Atoms 'aid' and 'element' arrays are missing or mismatched.")

    # Map atom ID (aid) -> atomic number (Z)
    aid_to_Z = {aid: Z for aid, Z in zip(atom_aids, atom_Z)}

    coords_sets = compound.get("coords", [])
    if not coords_sets:
        raise ValueError("No 'coords' section found.")
    coords = coords_sets[coords_set_index]

    conformers = coords.get("conformers", [])
    if not conformers:
        raise ValueError("No 'conformers' found under 'coords'.")
    conf = conformers[conformer_index]

    xs = conf.get("x", [])
    ys = conf.get("y", [])
    zs = conf.get("z", [])
    aids_for_coords: Sequence[int] = coords.get("aid", [])

    if not (len(xs) == len(ys) == len(zs) == len(aids_for_coords)):
        raise ValueError("Coordinate arrays (x,y,z) and 'coords.aid' have different lengths.")

    # Build output as 'SYMBOL x y z'
    out_pieces = []
    for i, aid in enumerate(aids_for_coords):
        Z = aid_to_Z.get(aid)
        if not Z or Z >= len(PERIODIC) or PERIODIC[Z] is None:
            sym = f"Z{Z}"  # fallback if unknown Z
        else:
            sym = PERIODIC[Z]
        x, y, z = xs[i], ys[i], zs[i]
        out_pieces.append(f"{sym} {format(x, float_fmt)} {format(y, float_fmt)} {format(z, float_fmt)}")

    return sep.join(out_pieces)
