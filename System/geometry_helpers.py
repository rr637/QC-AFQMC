def build_h_chain_atom_string(total_atoms: int, spacing: float = 0.74, axis: str = "z") -> str:
    """
    Create a linear hydrogen chain "H x y z; ..." with given spacing (Å)
    along the chosen axis ('x', 'y', 'z'). Origin at index 0.
    """
    if total_atoms < 1:
        raise ValueError("total_atoms must be >= 1.")
    # Unit axis vector
    ax = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}[axis.lower()]
    atoms = []
    for i in range(total_atoms):
        x = i * spacing * ax[0]
        y = i * spacing * ax[1]
        z = i * spacing * ax[2]
        atoms.append(f"H {x:.6f} {y:.6f} {z:.6f}")
    return "; ".join(atoms)