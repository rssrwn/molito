from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from molito.core._checks import check_dim_shape, check_shape_len, check_shapes_equal
from molito.core.bonds import BondEncoding
from molito.core.vocab import INT_TO_RDKIT_CHIRAL

TArr = np.ndarray


def mol_is_valid(mol: Chem.rdchem.Mol, with_hs: bool = True, connected: bool = True) -> bool:
    """Whether the mol can be sanitised and, optionally, whether it's fully connected

    Args:
        mol (Chem.Mol): RDKit molecule to check
        with_hs (bool): Whether to check validity including hydrogens (if they are in the input mol), default True
        connected (bool): Whether to also assert that the mol must not have disconnected atoms, default True

    Returns:
        bool: Whether the mol is valid
    """

    if mol is None:
        return False

    mol_copy = Chem.Mol(mol)
    if not with_hs:
        mol_copy = Chem.RemoveAllHs(mol_copy)

    try:
        AllChem.SanitizeMol(mol_copy)
    except Exception:
        return False

    n_frags = len(AllChem.GetMolFrags(mol_copy))
    if connected and n_frags != 1:
        return False

    return True


# TODO could allow more args
def smiles_from_mol(mol: Chem.rdchem.Mol, canonical: bool = True, explicit_hs: bool = False) -> str | None:
    """Create a SMILES string from a molecule

    Args:
        mol (Chem.Mol): RDKit molecule object
        canonical (bool): Whether to create a canonical SMILES, default True
        explicit_hs (bool): Whether to embed hydrogens in the mol before creating a SMILES, default False. If True
                this will create a new mol with all hydrogens embedded. Note that the SMILES created by doing this
                is not necessarily the same as creating a SMILES showing implicit hydrogens.

    Returns:
        str: SMILES string which could be None if the SMILES generation failed
    """

    if mol is None:
        return None

    if explicit_hs:
        mol = Chem.AddHs(mol)

    try:
        smiles = Chem.MolToSmiles(mol, canonical=canonical)
    except Exception:
        smiles = None

    return smiles


def mol_from_smiles(smiles: str, preserve_hs: bool = True, embed_hs: bool = False) -> Chem.rdchem.Mol | None:
    """Create a RDKit molecule from a SMILES string

    Args:
        smiles (str): SMILES string
        preserve_hs (bool): Whether to preserve hydrogen atoms that are present in the given SMILES.
        embed_hs (bool): Whether to embed explicit hydrogens into the mol. This could change the number of atoms in
                the molecule, even if hydrogens are provided in the input SMILES.

    Returns:
        Chem.Mol: RDKit molecule object or None if one cannot be created from the SMILES
    """

    if smiles is None:
        return None

    smi_params = Chem.SmilesParserParams()
    if preserve_hs:
        smi_params.removeHs = False

    try:
        mol = Chem.MolFromSmiles(smiles, smi_params)
        mol = Chem.AddHs(mol) if embed_hs else mol
    except Exception:
        mol = None

    return mol


def mol_from_atoms(
    atomics: TArr,
    bonds: TArr,
    coords: TArr | None = None,
    charges: TArr | None = None,
    chirality: TArr | None = None,
    sanitise: bool = True,
) -> Chem.rdchem.Mol | None:
    """Create RDKit mol from atomic numbers, bonds and, optionally, coords, charges, and chirality.

    Args:
        atomics (np.ndarray): Atomic numbers, length must be n_atoms
        bonds (np.ndarray): Bond indices and types, shape [n_bonds, 3]
        coords (np.ndarray, optional): Coordinate tensor, shape [n_atoms, 3] or [n_confs, n_atoms, 3]
        charges (np.ndarray, optional): Charge for each atom, shape [n_atoms]
        chirality (np.ndarray, optional): Chirality for each atom, shape [n_atoms]
        sanitise (bool): Whether to apply RDKit sanitization to the molecule, default True

    Returns:
        Chem.rdchem.Mol: RDKit molecule or None if one cannot be created
    """

    check_shape_len(atomics, 1, "atomics")
    check_shape_len(bonds, 2, "bonds")
    check_dim_shape(bonds, 1, 3, "bonds")

    if coords is not None:
        if len(coords.shape) == 2:
            check_dim_shape(coords, 1, 3, "coords")
            check_shapes_equal(atomics, coords, 0)
            coords = np.expand_dims(coords, (0))

        elif len(coords.shape) == 3:
            check_dim_shape(coords, 2, 3, "coords")
            if coords.shape[1] != atomics.shape[0]:
                raise ValueError("Coords and atomics must have the same number of atoms.")

        else:
            raise ValueError("Coords must have shape either [n_confs, n_atoms, 3] or [n_atoms, 3].")

    if charges is not None:
        check_shape_len(charges, 1, "charges")
        check_shapes_equal(atomics, charges, 0)

    charges = charges.tolist() if charges is not None else [0] * atomics.shape[0]
    chirality_list = chirality.tolist() if chirality is not None else None

    mol = Chem.RWMol()

    for idx, atomic in enumerate(atomics.tolist()):
        atom = Chem.Atom(atomic)
        atom.SetFormalCharge(charges[idx])

        if chirality_list is not None and chirality_list[idx] in INT_TO_RDKIT_CHIRAL:
            atom.SetChiralTag(INT_TO_RDKIT_CHIRAL[chirality_list[idx]])

        mol.AddAtom(atom)

    for bond in bonds.astype(np.int32).tolist():
        start, end, bond_index = bond

        if start == end:
            continue

        bond_type, is_arom, direction = BondEncoding.decode(bond_index)

        # Ignore non-RDKit bonds (eg. NONE or MASK)
        if not isinstance(bond_type, Chem.BondType):
            continue

        b_idx = mol.AddBond(start, end, bond_type)

        bond_obj = mol.GetBondWithIdx(b_idx - 1)
        bond_obj.SetIsAromatic(is_arom)

        if direction is not None:
            bond_obj.SetBondDir(direction)

    try:
        mol = mol.GetMol()
    except Exception:
        return None

    if coords is not None:
        for conf_coords in list(coords):
            conf = Chem.Conformer(conf_coords.shape[0])
            for idx, coord in enumerate(conf_coords.tolist()):
                conf.SetAtomPosition(idx, coord)

            mol.AddConformer(conf, assignId=True)

        # Only derive stereo from 3D when the caller hasn't supplied it — otherwise
        # the caller's chirality tags and bond directions (which came from a known-good
        # source like a SMILES string) would be silently overwritten by 3D perception,
        # which can disagree with the intended stereochemistry if the coords were
        # generated by an ML model or have approximate geometry.
        if chirality is None:
            try:
                Chem.AssignStereochemistryFrom3D(mol)
            except Exception:
                pass

    if sanitise:
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            return None

    return mol
