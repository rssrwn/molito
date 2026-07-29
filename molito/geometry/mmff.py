from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import AllChem

from molito.geometry.common import possibly_add_hs


def calc_energy_mmff(mol: Chem.Mol, per_atom: bool = False) -> float | list[float]:
    """Calculate the energy for an RDKit molecule using the MMFF forcefield.

    The molecule is copied so the original is not modified. If multiple conformers exist in the molecule the energies
    are calculated independently and returned as a list. The conformer ids must be continuous and start from 0.

    Args:
        mol (Chem.Mol): RDKit molecule
        per_atom (bool): Whether to normalise by number of atoms in mol, default False

    Returns:
        float: Energy of the molecule or None if the energy could not be calculated
    """

    mol_copy = possibly_add_hs(mol)

    if mol_copy is None:
        return None

    n_atoms = mol_copy.GetNumAtoms()

    energies = []
    for c_idx in range(mol_copy.GetNumConformers()):
        try:
            mmff_props = AllChem.MMFFGetMoleculeProperties(mol_copy, mmffVariant="MMFF94")
            ff = AllChem.MMFFGetMoleculeForceField(mol_copy, mmff_props, confId=c_idx)
            energy = ff.CalcEnergy()
            energy = energy / n_atoms if per_atom else energy
        except Exception:
            energy = None

        energies.append(energy)

    energy = energies[0] if len(energies) == 1 else energies
    return energy


def optimise_mol_mmff(
    mol: Chem.rdchem.Mol,
    max_iters: int = 1000,
    n_threads: int = 1,
    allow_unconverged: bool = True,
    return_energy: bool = False,
) -> Chem.rdchem.Mol | tuple[Chem.rdchem.Mol, float | list[float]] | None:
    """Optimise the conformation of an RDKit molecule using the MMFF forcefield.

    The molecule is copied so the original is not modified. If the input molecule contains multiple conformers,
    this function will attempt to optimise all conformers and return a new molecule object with the same number
    of conformers.

    When return_energy is True, energies are computed on the optimised with-Hs structure (same forcefield state used
    for optimisation) before copying coords back, giving accurate post-optimisation energies.

    Args:
        mol (Chem.Mol): RDKit molecule
        max_iters (int): Max iterations for the conformer optimisation algorithm
        n_threads (int): Number of threads to ask RDKit to use (0 means use all available processors)
        allow_unconverged (bool): Whether to allow returning partially converged molecules
        return_energy (bool): If True, also compute and return MMFF energies for each conformer

    Returns:
        The optimised molecule, or None on failure. With return_energy, a tuple of
        (optimised molecule, energy per conformer) instead - a float for a single
        conformer, a list otherwise.
    """

    mol_copy = Chem.Mol(mol)

    try:
        Chem.SanitizeMol(mol_copy)
    except Exception:
        return None

    contains_hs = mol_copy.GetNumAtoms() != mol_copy.GetNumHeavyAtoms()
    mol_copy = Chem.AddHs(mol_copy, addCoords=True) if not contains_hs else mol_copy

    opt_mol = Chem.Mol(mol_copy)

    try:
        out = AllChem.MMFFOptimizeMoleculeConfs(opt_mol, maxIters=max_iters, numThreads=n_threads)
    except Exception:
        return None

    if len(out) == 0:
        return None

    exitcodes, _ = tuple(zip(*out, strict=True))
    converged = [code == 0 for code in exitcodes]

    if not allow_unconverged and not all(converged):
        return None

    # Compute energies on the optimised with-Hs structure before copying coords back
    energies = None
    if return_energy:
        energies = []
        for c_idx in range(opt_mol.GetNumConformers()):
            try:
                mmff_props = AllChem.MMFFGetMoleculeProperties(opt_mol, mmffVariant="MMFF94")
                ff = AllChem.MMFFGetMoleculeForceField(opt_mol, mmff_props, confId=c_idx)
                energies.append(ff.CalcEnergy())
            except Exception:
                energies.append(None)

        energies = energies[0] if len(energies) == 1 else energies

    # Copy the mol and pass the opt conf info since MMFF will change aromatic atom props
    mol_copy.RemoveAllConformers()

    for conf in opt_mol.GetConformers():
        mol_copy.AddConformer(conf, assignId=True)

    mol_out = mol_copy if contains_hs else Chem.RemoveAllHs(mol_copy)

    if return_energy:
        return mol_out, energies

    return mol_out
