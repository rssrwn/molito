import unittest

import numpy as np
from rdkit import Chem

from molito.core.atoms import AtomSet


class TestAtoms(unittest.TestCase):
    def setUp(self):
        atomics = np.array([6, 1, 6, 7, 1])
        charges = np.array([0, 0, 1, -1, 0])
        atoms = AtomSet(atomics, charges=charges)
        self.atoms = atoms

    def test_length(self):
        expected_length = 5

        self.assertEqual(len(self.atoms), expected_length)
        self.assertEqual(self.atoms.seq_length, expected_length)

    def test_charged_symbols(self):
        expected_symbols = ["C_0", "H_0", "C_1", "N_-1", "H_0"]

        self.assertEqual(self.atoms.charged_symbols, expected_symbols)

    def test_permute(self):
        expected_symbols_1 = ["C_1", "N_-1", "H_0", "C_0", "H_0"]
        expected_symbols_2 = ["N_-1", "H_0", "C_1", "H_0", "C_0"]
        expected_length = 5

        indices_1 = [2, 3, 1, 0, 4]
        permuted_1 = self.atoms.permute_atoms(indices_1)

        indices_2 = np.array([3, 1, 2, 4, 0])
        permuted_2 = self.atoms.permute_atoms(indices_2)

        self.assertEqual(len(permuted_1), expected_length)
        self.assertEqual(len(permuted_2), expected_length)

        self.assertEqual(permuted_1.charged_symbols, expected_symbols_1)
        self.assertEqual(permuted_2.charged_symbols, expected_symbols_2)

    def test_permute_takes_subset(self):
        expected_symbols = ["N_-1", "H_0", "H_0"]
        expected_length = 3

        indices = [3, 4, 1]
        permuted = self.atoms.permute_atoms(indices)

        self.assertEqual(len(permuted), expected_length)
        self.assertEqual(permuted.charged_symbols, expected_symbols)

    def test_permute_checks_min_max(self):
        neg_indices = [2, 3, 0, -1, 1]
        oob_indices = [2, 3, 0, 6]

        self.assertRaises(ValueError, self.atoms.permute_atoms, neg_indices)
        self.assertRaises(ValueError, self.atoms.permute_atoms, oob_indices)

    def test_permute_finds_duplicates(self):
        indices = [2, 3, 0, 1, 1]

        self.assertRaises(ValueError, self.atoms.permute_atoms, indices)

    def test_getitem_integer_index(self):
        expected_atomic = 6
        expected_charge = 1

        index = 2

        atomic_1, charge_1, chirality_1 = self.atoms[index]
        atomic_2, charge_2, chirality_2 = self.atoms.__getitem__(index)

        self.assertEqual(atomic_1, expected_atomic)
        self.assertEqual(charge_1, expected_charge)
        self.assertEqual(chirality_1, 0)

        self.assertEqual(atomic_2, expected_atomic)
        self.assertEqual(charge_2, expected_charge)
        self.assertEqual(chirality_2, 0)

    def test_pad_equal_length(self):
        expected_symbols = ["C_0", "H_0", "C_1", "N_-1", "H_0"]

        padded = self.atoms.pad(len(self.atoms))

        self.assertEqual(padded.charged_symbols, expected_symbols)

    def test_pad_throws_error_on_small_length(self):
        self.assertRaises(ValueError, self.atoms.pad, len(self.atoms) - 1)

    def test_pad_pads_atomics_and_charges_with_zeros(self):
        expected_atomics = [6, 1, 6, 7, 1, 0, 0]
        expected_charges = [0, 0, 1, -1, 0, 0, 0]

        padded = self.atoms.pad(len(self.atoms) + 2)
        atomics = padded.atomics.tolist()
        charges = padded.charges.tolist()

        self.assertEqual(len(atomics), len(expected_atomics))
        self.assertEqual(atomics, expected_atomics)
        self.assertEqual(charges, expected_charges)

    def test_pad_pads_atomics_and_charges_with_given_val(self):
        expected_atomics = [6, 1, 6, 7, 1, 99, 99, 99]
        expected_charges = [0, 0, 1, -1, 0, 74, 74, 74]

        padded = self.atoms.pad(len(self.atoms) + 3, pad_atomic=99, pad_charge=74)
        atomics = padded.atomics.tolist()
        charges = padded.charges.tolist()

        self.assertEqual(len(atomics), len(expected_atomics))
        self.assertEqual(atomics, expected_atomics)
        self.assertEqual(charges, expected_charges)


class TestAtomsWithAnnotations(unittest.TestCase):
    def setUp(self):
        atomics = np.array([6, 7, 8, 6, 7])
        charges = np.array([0, 0, -1, 0, 1])
        res_names = np.array(["ALA", "ALA", "ALA", "GLY", "GLY"])
        atom_names = np.array(["CA", "N", "O", "CA", "N"])
        res_ids = np.array([1, 1, 1, 2, 2])

        self.atoms = AtomSet(atomics, charges=charges, res_names=res_names, atom_names=atom_names, res_ids=res_ids)

    def test_optional_properties_accessible(self):
        self.assertEqual(self.atoms.res_names.tolist(), ["ALA", "ALA", "ALA", "GLY", "GLY"])
        self.assertEqual(self.atoms.atom_names.tolist(), ["CA", "N", "O", "CA", "N"])
        self.assertEqual(self.atoms.res_ids.tolist(), [1, 1, 1, 2, 2])

    def test_has_residue_annotations_true(self):
        self.assertTrue(self.atoms.has_residue_annotations)

    def test_has_residue_annotations_false_when_missing(self):
        atoms_no_annotations = AtomSet(np.array([6, 7, 8]))
        self.assertFalse(atoms_no_annotations.has_residue_annotations)

    def test_permute_preserves_annotations(self):
        indices = [2, 0, 4]
        permuted = self.atoms.permute_atoms(indices)

        self.assertEqual(permuted.res_names.tolist(), ["ALA", "ALA", "GLY"])
        self.assertEqual(permuted.atom_names.tolist(), ["O", "CA", "N"])
        self.assertEqual(permuted.res_ids.tolist(), [1, 1, 2])

    def test_getitem_array_preserves_annotations(self):
        indices = np.array([1, 3])
        subset = self.atoms[indices]

        self.assertEqual(len(subset), 2)
        self.assertEqual(subset.res_names.tolist(), ["ALA", "GLY"])
        self.assertEqual(subset.atom_names.tolist(), ["N", "CA"])
        self.assertEqual(subset.res_ids.tolist(), [1, 2])

    def test_pad_preserves_and_extends_annotations(self):
        padded = self.atoms.pad(7)

        self.assertEqual(len(padded), 7)
        self.assertEqual(padded.res_names.tolist(), ["ALA", "ALA", "ALA", "GLY", "GLY", "PAD", "PAD"])
        self.assertEqual(padded.atom_names.tolist(), ["CA", "N", "O", "CA", "N", "PAD", "PAD"])
        self.assertEqual(padded.res_ids.tolist(), [1, 1, 1, 2, 2, -1, -1])

    def test_copy_preserves_annotations(self):
        copied = self.atoms.copy()

        self.assertEqual(copied.res_names.tolist(), self.atoms.res_names.tolist())
        self.assertEqual(copied.atom_names.tolist(), self.atoms.atom_names.tolist())
        self.assertEqual(copied.res_ids.tolist(), self.atoms.res_ids.tolist())


class TestAtomSetChainIds(unittest.TestCase):
    def setUp(self):
        self.atomics = np.array([6, 7, 8, 6, 7])
        self.chain_ids = np.array(["A", "A", "A", "B", "B"])

    def test_chain_ids_default_none(self):
        atoms = AtomSet(self.atomics)
        self.assertIsNone(atoms.chain_ids)

    def test_chain_ids_stored_and_accessible(self):
        atoms = AtomSet(self.atomics, chain_ids=self.chain_ids)
        self.assertEqual(atoms.chain_ids.tolist(), ["A", "A", "A", "B", "B"])

    def test_chain_ids_permute(self):
        atoms = AtomSet(self.atomics, chain_ids=self.chain_ids)
        permuted = atoms.permute_atoms([3, 0, 4])
        self.assertEqual(permuted.chain_ids.tolist(), ["B", "A", "B"])

    def test_chain_ids_pad(self):
        atoms = AtomSet(self.atomics, chain_ids=self.chain_ids)
        padded = atoms.pad(7)
        self.assertEqual(padded.chain_ids.tolist(), ["A", "A", "A", "B", "B", "PAD", "PAD"])

    def test_chain_ids_copy(self):
        atoms = AtomSet(self.atomics, chain_ids=self.chain_ids)
        copied = atoms.copy()
        self.assertEqual(copied.chain_ids.tolist(), atoms.chain_ids.tolist())

    def test_chain_ids_roundtrip_arrays(self):
        atoms = AtomSet(self.atomics, chain_ids=self.chain_ids)
        arrays = AtomSet.arrays_from_atoms([atoms])
        self.assertIn("chain_ids", arrays)

        restored = AtomSet.atoms_from_arrays(arrays)
        self.assertEqual(len(restored), 1)
        self.assertEqual(restored[0].chain_ids.tolist(), ["A", "A", "A", "B", "B"])

    def test_legacy_batch_without_chain_ids(self):
        atoms = AtomSet(self.atomics)
        arrays = AtomSet.arrays_from_atoms([atoms])
        self.assertNotIn("chain_ids", arrays)

        restored = AtomSet.atoms_from_arrays(arrays)
        self.assertIsNone(restored[0].chain_ids)

    def test_chain_id_length_validated(self):
        too_long = np.array(["LONGCHAIN", "A", "B", "C", "D"])
        with self.assertRaises(ValueError):
            AtomSet(self.atomics, chain_ids=too_long)


class TestAtomSetFromRdkit(unittest.TestCase):
    def test_basic_from_rdkit(self):
        mol = Chem.MolFromSmiles("CCO")
        atoms = AtomSet.from_rdkit(mol)
        self.assertEqual(len(atoms), 3)
        self.assertEqual(atoms.atomics.tolist(), [6, 6, 8])

    def test_charges_from_rdkit(self):
        mol = Chem.MolFromSmiles("[NH3+]CC([O-])=O")
        atoms = AtomSet.from_rdkit(mol)
        charges = atoms.charges.tolist()
        self.assertIn(1, charges)
        self.assertIn(-1, charges)

    def test_chirality_from_rdkit(self):
        mol = Chem.MolFromSmiles("[C@@H](F)(Cl)Br")
        atoms = AtomSet.from_rdkit(mol)
        chirals = atoms.chirality.tolist()
        self.assertTrue(any(c != 0 for c in chirals))

    def test_dtypes(self):
        mol = Chem.MolFromSmiles("CCO")
        atoms = AtomSet.from_rdkit(mol)
        self.assertEqual(atoms.atomics.dtype, np.uint8)
        self.assertEqual(atoms.charges.dtype, np.int8)
        self.assertEqual(atoms.chirality.dtype, np.int8)


class TestAtomSetTokens(unittest.TestCase):
    def test_tokens_without_chirality(self):
        atomics = np.array([6, 7, 8])
        charges = np.array([0, -1, 0])
        atoms = AtomSet(atomics, charges=charges)
        self.assertEqual(atoms.tokens, ["C_0", "N_-1", "O_0"])

    def test_tokens_with_chirality(self):
        atomics = np.array([6, 6])
        charges = np.array([0, 0])
        chirality = np.array([1, 2], dtype=np.int8)
        atoms = AtomSet(atomics, charges=charges, chirality=chirality)
        self.assertEqual(atoms.tokens, ["C_0_CW", "C_0_CCW"])

    def test_tokens_mixed_chirality(self):
        atomics = np.array([6, 6, 7])
        charges = np.array([0, 0, 0])
        chirality = np.array([0, 1, 0], dtype=np.int8)
        atoms = AtomSet(atomics, charges=charges, chirality=chirality)
        self.assertEqual(atoms.tokens, ["C_0", "C_0_CW", "N_0"])


class TestAtomSetDictRoundtrip(unittest.TestCase):
    def test_basic_roundtrip(self):
        atomics = np.array([6, 7, 8])
        charges = np.array([0, -1, 0])
        chirality = np.array([0, 0, 1], dtype=np.int8)
        atoms = AtomSet(atomics, charges=charges, chirality=chirality)

        d = atoms.to_dict()
        restored = AtomSet.from_dict(d)

        np.testing.assert_array_equal(restored.atomics, atoms.atomics)
        np.testing.assert_array_equal(restored.charges, atoms.charges)
        np.testing.assert_array_equal(restored.chirality, atoms.chirality)

    def test_roundtrip_with_annotations(self):
        atomics = np.array([6, 7])
        charges = np.array([0, 0])
        res_names = np.array(["ALA", "GLY"])
        atom_names = np.array(["CA", "N"])
        res_ids = np.array([1, 2])
        atoms = AtomSet(atomics, charges=charges, res_names=res_names, atom_names=atom_names, res_ids=res_ids)

        d = atoms.to_dict()
        restored = AtomSet.from_dict(d)

        np.testing.assert_array_equal(restored.atomics, atoms.atomics)
        np.testing.assert_array_equal(restored.res_names, atoms.res_names)
        np.testing.assert_array_equal(restored.atom_names, atoms.atom_names)
        np.testing.assert_array_equal(restored.res_ids, atoms.res_ids)


class TestAtomSetArraysMerge(unittest.TestCase):
    def test_arrays_from_atoms_roundtrip(self):
        atoms1 = AtomSet(np.array([6, 7]), charges=np.array([0, -1]))
        atoms2 = AtomSet(np.array([8, 16, 1]), charges=np.array([0, 0, 0]))

        arrays = AtomSet.arrays_from_atoms([atoms1, atoms2])
        restored = AtomSet.atoms_from_arrays(arrays)

        self.assertEqual(len(restored), 2)
        np.testing.assert_array_equal(restored[0].atomics, atoms1.atomics)
        np.testing.assert_array_equal(restored[1].atomics, atoms2.atomics)
        np.testing.assert_array_equal(restored[0].charges, atoms1.charges)
        np.testing.assert_array_equal(restored[1].charges, atoms2.charges)
