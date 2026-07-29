import unittest

import numpy as np

from molito.core.atoms import AtomSet
from molito.core.bonds import BondSet
from molito.core.confs import ConfSet
from molito.mol.protein import Protein


class TestProtein(unittest.TestCase):
    def setUp(self):
        # Create a simple protein with 5 atoms across 2 residues
        atomics = np.array([6, 7, 8, 6, 7])
        charges = np.array([0, 0, -1, 0, 1])
        res_names = np.array(["ALA", "ALA", "ALA", "GLY", "GLY"])
        atom_names = np.array(["CA", "N", "O", "CA", "N"])
        res_ids = np.array([1, 1, 1, 2, 2])

        atoms = AtomSet(atomics, charges=charges, res_names=res_names, atom_names=atom_names, res_ids=res_ids)

        # Bonds: 0-1, 1-2, 2-3, 3-4
        bonds = np.array([[0, 1, 1], [1, 2, 1], [2, 3, 1], [3, 4, 1]])
        bonds = BondSet(bonds)

        coords = np.array([[[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]]], dtype=np.float32)
        confs = ConfSet(coords)

        self.protein = Protein(atoms, bonds, confs, meta={"name": "test"})

    def test_basic_properties(self):
        self.assertEqual(self.protein.n_atoms, 5)
        self.assertEqual(self.protein.n_bonds, 4)
        self.assertEqual(self.protein.n_residues, 2)
        self.assertEqual(self.protein.seq_length, 5)
        self.assertEqual(len(self.protein), 5)

    def test_atom_property_accessors(self):
        self.assertEqual(self.protein.atomics.tolist(), [6, 7, 8, 6, 7])
        self.assertEqual(self.protein.charges.tolist(), [0, 0, -1, 0, 1])
        self.assertEqual(self.protein.res_names, ["ALA", "ALA", "ALA", "GLY", "GLY"])
        self.assertEqual(self.protein.atom_names, ["CA", "N", "O", "CA", "N"])
        self.assertEqual(self.protein.res_ids.tolist(), [1, 1, 1, 2, 2])

    def test_bond_property_accessors(self):
        self.assertEqual(self.protein.bond_indices.shape, (4, 2))
        self.assertEqual(self.protein.bond_types.shape, (4,))

    def test_coords_accessor(self):
        self.assertEqual(self.protein.coords.shape, (5, 3))

    def test_adjacency_matrix(self):
        adj = self.protein.adjacency
        self.assertEqual(adj.shape, (5, 5))
        # Check symmetry
        self.assertTrue((adj == adj.T).all())
        # Check diagonal is zero
        self.assertTrue((np.diag(adj) == 0).all())

    def test_copy(self):
        copied = self.protein.copy()

        self.assertEqual(copied.n_atoms, self.protein.n_atoms)
        self.assertEqual(copied.atomics.tolist(), self.protein.atomics.tolist())
        self.assertEqual(copied.meta, self.protein.meta)

        # Verify it's a deep copy
        copied.meta["new_key"] = "new_value"
        self.assertNotIn("new_key", self.protein.meta)

    def test_permute(self):
        indices = [4, 3, 2]
        permuted = self.protein.permute(indices)

        self.assertEqual(permuted.n_atoms, 3)
        self.assertEqual(permuted.atomics.tolist(), [7, 6, 8])
        self.assertEqual(permuted.res_names, ["GLY", "GLY", "ALA"])

    def test_remove_hs(self):
        # Add hydrogen atoms to test removal
        atomics = np.array([6, 1, 7, 1, 8])
        charges = np.zeros(5, dtype=np.int16)
        res_names = np.array(["ALA", "ALA", "ALA", "ALA", "ALA"])
        atom_names = np.array(["CA", "H1", "N", "H2", "O"])
        res_ids = np.array([1, 1, 1, 1, 1])

        atoms = AtomSet(atomics, charges=charges, res_names=res_names, atom_names=atom_names, res_ids=res_ids)
        bonds = BondSet(np.array([[0, 2, 1]]))
        coords = np.random.rand(1, 5, 3).astype(np.float32)
        confs = ConfSet(coords)

        protein = Protein(atoms, bonds, confs)
        no_h = protein.remove_hs()

        self.assertEqual(no_h.n_atoms, 3)
        self.assertEqual(no_h.atomics.tolist(), [6, 7, 8])

    def test_zero_com(self):
        zeroed = self.protein.zero_com()
        com = zeroed.coords.mean(axis=0)
        np.testing.assert_array_almost_equal(com, np.zeros(3), decimal=5)

    def test_shift(self):
        shift_vec = np.array([10, 20, 30])
        shifted = self.protein.shift(shift_vec)

        expected_coords = self.protein.coords + shift_vec
        np.testing.assert_array_almost_equal(shifted.coords, expected_coords)

    def test_bytes_serialization(self):
        data = self.protein.to_bytes()
        restored = Protein.from_bytes(data)

        self.assertEqual(restored.n_atoms, self.protein.n_atoms)
        self.assertEqual(restored.atomics.tolist(), self.protein.atomics.tolist())
        self.assertEqual(restored.res_names, self.protein.res_names)
        self.assertEqual(restored.meta, self.protein.meta)
        np.testing.assert_array_almost_equal(restored.coords, self.protein.coords)

    def test_requires_residue_annotations(self):
        atoms_no_annotations = AtomSet(np.array([6, 7, 8]))
        bonds = BondSet(np.array([[0, 1, 1]]))
        coords = np.random.rand(1, 3, 3).astype(np.float32)
        confs = ConfSet(coords)

        with self.assertRaises(ValueError):
            Protein(atoms_no_annotations, bonds, confs)
