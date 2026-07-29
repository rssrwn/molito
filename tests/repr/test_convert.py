import unittest

import numpy as np
from rdkit import Chem

from molito.convert import mol_from_atoms, mol_from_smiles, mol_is_valid, smiles_from_mol
from molito.core.bonds import BondEncoding


class TestMolIsValid(unittest.TestCase):
    def test_valid_mol(self):
        mol = Chem.MolFromSmiles("CCO")
        self.assertTrue(mol_is_valid(mol))

    def test_none_is_invalid(self):
        self.assertFalse(mol_is_valid(None))

    def test_disconnected_mol_invalid_by_default(self):
        mol = Chem.MolFromSmiles("C.C")
        self.assertFalse(mol_is_valid(mol))

    def test_disconnected_mol_valid_when_connected_false(self):
        mol = Chem.MolFromSmiles("C.C")
        self.assertTrue(mol_is_valid(mol, connected=False))

    def test_with_hs_true(self):
        mol = Chem.MolFromSmiles("C")
        mol = Chem.AddHs(mol)
        self.assertTrue(mol_is_valid(mol, with_hs=True))

    def test_with_hs_false_strips_hs(self):
        mol = Chem.MolFromSmiles("C")
        mol = Chem.AddHs(mol)
        self.assertTrue(mol_is_valid(mol, with_hs=False))

    def test_invalid_chemistry(self):
        mol = Chem.RWMol()
        mol.AddAtom(Chem.Atom(8))
        mol.AddAtom(Chem.Atom(8))
        mol.AddAtom(Chem.Atom(8))
        mol.AddAtom(Chem.Atom(8))
        mol.AddAtom(Chem.Atom(8))
        mol.AddBond(0, 1, Chem.BondType.TRIPLE)
        mol.AddBond(0, 2, Chem.BondType.TRIPLE)
        mol.AddBond(0, 3, Chem.BondType.TRIPLE)
        mol.AddBond(0, 4, Chem.BondType.TRIPLE)
        self.assertFalse(mol_is_valid(mol.GetMol()))


class TestSmilesFromMol(unittest.TestCase):
    def test_basic_smiles(self):
        mol = Chem.MolFromSmiles("CCO")
        smi = smiles_from_mol(mol)
        self.assertEqual(smi, "CCO")

    def test_canonical_true(self):
        mol = Chem.MolFromSmiles("OCC")
        smi = smiles_from_mol(mol, canonical=True)
        self.assertEqual(smi, "CCO")

    def test_none_mol_returns_none(self):
        self.assertIsNone(smiles_from_mol(None))

    def test_explicit_hs(self):
        mol = Chem.MolFromSmiles("C")
        smi = smiles_from_mol(mol, explicit_hs=True)
        self.assertIn("[H]", smi)


class TestMolFromSmiles(unittest.TestCase):
    def test_basic_mol(self):
        mol = mol_from_smiles("CCO")
        self.assertIsNotNone(mol)
        self.assertEqual(mol.GetNumHeavyAtoms(), 3)

    def test_none_smiles_returns_none(self):
        self.assertIsNone(mol_from_smiles(None))

    def test_invalid_smiles_returns_none(self):
        self.assertIsNone(mol_from_smiles("not_a_smiles_XYZ"))

    def test_preserve_hs(self):
        mol = mol_from_smiles("[H]C([H])([H])[H]", preserve_hs=True)
        self.assertEqual(mol.GetNumAtoms(), 5)

    def test_embed_hs(self):
        mol = mol_from_smiles("C", embed_hs=True)
        self.assertGreater(mol.GetNumAtoms(), 1)


class TestMolFromAtoms(unittest.TestCase):
    def setUp(self):
        self.atomics = np.array([6, 6, 8], dtype=np.uint8)
        self.bonds = np.array(
            [
                [0, 1, BondEncoding.encode(Chem.BondType.SINGLE)],
                [1, 2, BondEncoding.encode(Chem.BondType.SINGLE)],
            ]
        )

    def test_basic_creation(self):
        mol = mol_from_atoms(self.atomics, self.bonds)
        self.assertIsNotNone(mol)
        self.assertEqual(mol.GetNumAtoms(), 3)
        self.assertEqual(mol.GetNumBonds(), 2)

    def test_with_charges(self):
        charges = np.array([0, 0, -1], dtype=np.int8)
        mol = mol_from_atoms(self.atomics, self.bonds, charges=charges)
        self.assertEqual(mol.GetAtomWithIdx(2).GetFormalCharge(), -1)

    def test_with_chirality(self):
        # Tetrahedral carbon: C with 4 different neighbours
        atomics = np.array([6, 9, 17, 35, 53], dtype=np.uint8)
        bonds = np.array(
            [
                [0, 1, BondEncoding.encode(Chem.BondType.SINGLE)],
                [0, 2, BondEncoding.encode(Chem.BondType.SINGLE)],
                [0, 3, BondEncoding.encode(Chem.BondType.SINGLE)],
                [0, 4, BondEncoding.encode(Chem.BondType.SINGLE)],
            ]
        )
        chirality = np.array([1, 0, 0, 0, 0], dtype=np.int8)
        mol = mol_from_atoms(atomics, bonds, chirality=chirality)
        self.assertEqual(mol.GetAtomWithIdx(0).GetChiralTag(), Chem.ChiralType.CHI_TETRAHEDRAL_CW)

    def test_with_2d_coords(self):
        coords = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
        mol = mol_from_atoms(self.atomics, self.bonds, coords=coords)
        self.assertEqual(mol.GetNumConformers(), 1)

    def test_with_3d_coords(self):
        coords = np.random.rand(3, 3, 3).astype(np.float32)
        mol = mol_from_atoms(self.atomics, self.bonds, coords=coords)
        self.assertEqual(mol.GetNumConformers(), 3)

    def test_self_bonds_skipped(self):
        bonds = np.array(
            [
                [0, 0, BondEncoding.encode(Chem.BondType.SINGLE)],
                [0, 1, BondEncoding.encode(Chem.BondType.SINGLE)],
            ]
        )
        mol = mol_from_atoms(self.atomics, bonds)
        self.assertEqual(mol.GetNumBonds(), 1)

    def test_none_bond_skipped(self):
        bonds = np.array(
            [
                [0, 1, BondEncoding.encode("NONE")],
                [1, 2, BondEncoding.encode(Chem.BondType.SINGLE)],
            ]
        )
        mol = mol_from_atoms(self.atomics, bonds)
        self.assertEqual(mol.GetNumBonds(), 1)

    def test_aromatic_bond(self):
        # Benzene ring: 6 carbons
        atomics = np.array([6, 6, 6, 6, 6, 6], dtype=np.uint8)
        bonds = np.array(
            [
                [0, 1, BondEncoding.encode(Chem.BondType.DOUBLE, is_aromatic=True)],
                [1, 2, BondEncoding.encode(Chem.BondType.SINGLE, is_aromatic=True)],
                [2, 3, BondEncoding.encode(Chem.BondType.DOUBLE, is_aromatic=True)],
                [3, 4, BondEncoding.encode(Chem.BondType.SINGLE, is_aromatic=True)],
                [4, 5, BondEncoding.encode(Chem.BondType.DOUBLE, is_aromatic=True)],
                [5, 0, BondEncoding.encode(Chem.BondType.SINGLE, is_aromatic=True)],
            ]
        )
        mol = mol_from_atoms(atomics, bonds)
        self.assertIsNotNone(mol)

    def test_direction_bonds(self):
        # E/Z: C/C=C/C (but-2-ene with E/Z)
        atomics = np.array([6, 6, 6, 6], dtype=np.uint8)
        bonds = np.array(
            [
                [0, 1, BondEncoding.encode(Chem.BondType.SINGLE, direction=Chem.BondDir.ENDUPRIGHT)],
                [1, 2, BondEncoding.encode(Chem.BondType.DOUBLE)],
                [2, 3, BondEncoding.encode(Chem.BondType.SINGLE, direction=Chem.BondDir.ENDUPRIGHT)],
            ]
        )
        mol = mol_from_atoms(atomics, bonds, sanitise=False)
        self.assertIsNotNone(mol)
        bond_01 = mol.GetBondBetweenAtoms(0, 1)
        self.assertEqual(bond_01.GetBondDir(), Chem.BondDir.ENDUPRIGHT)

    def test_sanitise_false_allows_bad_chemistry(self):
        # Oxygen with too many bonds
        atomics = np.array([8, 6, 6, 6], dtype=np.uint8)
        bonds = np.array(
            [
                [0, 1, BondEncoding.encode(Chem.BondType.DOUBLE)],
                [0, 2, BondEncoding.encode(Chem.BondType.DOUBLE)],
                [0, 3, BondEncoding.encode(Chem.BondType.DOUBLE)],
            ]
        )
        mol = mol_from_atoms(atomics, bonds, sanitise=False)
        self.assertIsNotNone(mol)

    def test_sanitise_true_rejects_bad_chemistry(self):
        atomics = np.array([8, 6, 6, 6], dtype=np.uint8)
        bonds = np.array(
            [
                [0, 1, BondEncoding.encode(Chem.BondType.DOUBLE)],
                [0, 2, BondEncoding.encode(Chem.BondType.DOUBLE)],
                [0, 3, BondEncoding.encode(Chem.BondType.DOUBLE)],
            ]
        )
        mol = mol_from_atoms(atomics, bonds, sanitise=True)
        self.assertIsNone(mol)

    def test_invalid_coords_shape_raises(self):
        coords = np.random.rand(3).astype(np.float32)
        with self.assertRaises(ValueError):
            mol_from_atoms(self.atomics, self.bonds, coords=coords)

    def test_coords_atom_mismatch_raises(self):
        coords = np.random.rand(5, 3).astype(np.float32)
        with self.assertRaises(RuntimeError):
            mol_from_atoms(self.atomics, self.bonds, coords=coords)


class TestMolFromAtomsPreservesStereoUnderCoords(unittest.TestCase):
    """Regression: supplied chirality / bond directions must NOT be overwritten by
    3D perception when coords are also provided. This used to silently corrupt stereo
    whenever the 3D coords disagreed with the stored stereo (e.g. ML-generated poses
    that don't faithfully preserve chirality or E/Z planarity from the source SMILES).
    """

    def _tetrahedral_atomics_bonds(self):
        atomics = np.array([6, 9, 17, 35, 53], dtype=np.uint8)
        bonds = np.array(
            [
                [0, 1, BondEncoding.encode(Chem.BondType.SINGLE)],
                [0, 2, BondEncoding.encode(Chem.BondType.SINGLE)],
                [0, 3, BondEncoding.encode(Chem.BondType.SINGLE)],
                [0, 4, BondEncoding.encode(Chem.BondType.SINGLE)],
            ]
        )
        return atomics, bonds

    def test_stored_chirality_survives_contradicting_coords(self):
        atomics, bonds = self._tetrahedral_atomics_bonds()
        chirality = np.array([1, 0, 0, 0, 0], dtype=np.int8)  # CW

        # Coords that, if perceived from 3D, would assign the OPPOSITE chirality.
        # Neighbours at tetrahedral corners, but with two swapped (mirror-image layout).
        coords = np.array(
            [
                [0.0, 0.0, 0.0],  # central C
                [1.0, 1.0, 1.0],  # F
                [1.0, -1.0, -1.0],  # Cl
                [-1.0, -1.0, 1.0],  # Br  (swapped vs mirror image)
                [-1.0, 1.0, -1.0],  # I
            ],
            dtype=np.float32,
        )

        mol = mol_from_atoms(atomics, bonds, coords=coords, chirality=chirality)
        # Stored chirality must still be CW
        self.assertEqual(mol.GetAtomWithIdx(0).GetChiralTag(), Chem.ChiralType.CHI_TETRAHEDRAL_CW)

    def test_ccw_stored_chirality_survives_coords(self):
        atomics, bonds = self._tetrahedral_atomics_bonds()
        chirality = np.array([2, 0, 0, 0, 0], dtype=np.int8)  # CCW

        coords = np.random.RandomState(42).rand(5, 3).astype(np.float32)
        mol = mol_from_atoms(atomics, bonds, coords=coords, chirality=chirality)
        self.assertEqual(mol.GetAtomWithIdx(0).GetChiralTag(), Chem.ChiralType.CHI_TETRAHEDRAL_CCW)

    def test_bond_direction_survives_coords(self):
        atomics = np.array([6, 6, 6, 6], dtype=np.uint8)
        bonds = np.array(
            [
                [0, 1, BondEncoding.encode(Chem.BondType.SINGLE, direction=Chem.BondDir.ENDUPRIGHT)],
                [1, 2, BondEncoding.encode(Chem.BondType.DOUBLE)],
                [2, 3, BondEncoding.encode(Chem.BondType.SINGLE, direction=Chem.BondDir.ENDUPRIGHT)],
            ]
        )
        chirality = np.zeros(4, dtype=np.int8)  # triggers the "chirality supplied" gate

        # Non-planar coords that don't clearly encode E/Z (would previously confuse
        # AssignStereochemistryFrom3D and silently reset directions to NONE).
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.3, 0.2, 0.1],
                [2.6, 0.0, -0.1],
                [3.9, 0.3, 0.3],
            ],
            dtype=np.float32,
        )

        mol = mol_from_atoms(atomics, bonds, coords=coords, chirality=chirality, sanitise=False)
        bond_01 = mol.GetBondBetweenAtoms(0, 1)
        bond_23 = mol.GetBondBetweenAtoms(2, 3)
        self.assertEqual(bond_01.GetBondDir(), Chem.BondDir.ENDUPRIGHT)
        self.assertEqual(bond_23.GetBondDir(), Chem.BondDir.ENDUPRIGHT)

    def test_fallback_to_3d_when_chirality_is_none(self):
        # When chirality is NOT supplied, 3D perception should still run so callers
        # that only have coords (e.g. protein ingestion) get chirality inferred.
        smi = "[C@@H](F)(Cl)Br"
        rdkit_mol = Chem.MolFromSmiles(smi)
        rdkit_mol = Chem.AddHs(rdkit_mol)
        from rdkit.Chem import AllChem

        AllChem.EmbedMolecule(rdkit_mol, randomSeed=42)
        rdkit_mol = Chem.RemoveHs(rdkit_mol)

        atomics = np.array([a.GetAtomicNum() for a in rdkit_mol.GetAtoms()], dtype=np.uint8)
        bonds_list = []
        for b in rdkit_mol.GetBonds():
            s, e = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if s > e:
                s, e = e, s
            bonds_list.append([s, e, BondEncoding.encode(b.GetBondType())])

        bonds = np.array(bonds_list)
        coords = rdkit_mol.GetConformer().GetPositions().astype(np.float32)

        # chirality=None -> 3D should be used to derive chirality
        mol = mol_from_atoms(atomics, bonds, coords=coords, chirality=None)
        central = mol.GetAtomWithIdx(0)
        # Perceived chirality from 3D should not be UNSPECIFIED
        self.assertIn(
            central.GetChiralTag(),
            (Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW),
        )
