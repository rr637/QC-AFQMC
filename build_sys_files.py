from System.MoleculeBuilder import BuildMoleculeProblem
from System.geometry_helpers import build_h_chain_atom_string
total_h_atoms_list = [4]
h_spacing_list = [1.0]
FCIs = []
HFs = []
for total_h_atoms in total_h_atoms_list:
  for h_spacing in h_spacing_list:
    h_chain = build_h_chain_atom_string(total_atoms=total_h_atoms,spacing=h_spacing)
    mol_problem = BuildMoleculeProblem(atom=h_chain,basis="sto-6g", spin=0,mol_identifier=f"H{total_h_atoms}_{h_spacing}_chain")
    FCI = mol_problem.get_FCI_energy()
    HF = mol_problem.get_hf_energy()
    FCIs.append(FCI)
    HFs.append(HF)

    print(f"R = {h_spacing}")
    print("HF", HF)
    print("FCI",FCI)
    print(f"Coorelation energy {(HF - FCI)*1e3} mH")
