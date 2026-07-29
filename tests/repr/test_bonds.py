import unittest

import numpy as np
from rdkit import Chem

from molito.core.bonds import BondEncoding, BondSet


class TestBondSet(unittest.TestCase):
    def setUp(self):
        bonds = np.array([[0, 1, 1], [1, 2, 2], [2, 3, 1], [3, 4, 3], [0, 2, 1]])
        bonds = BondSet(bonds)
        self.bonds = bonds

    def test_adj_provides_correct_connections(self):
        expected_shape = (5, 5)
        expected_adj = np.array([[0, 1, 1, 0, 0], [1, 0, 2, 0, 0], [1, 2, 0, 1, 0], [0, 0, 1, 0, 3], [0, 0, 0, 3, 0]])

        adj = self.bonds.adj_matrix(5)
        arr_equal = (adj == expected_adj).all()

        self.assertEqual(adj.shape, expected_shape)
        self.assertTrue(arr_equal)

    def test_permute_atoms(self):
        expected_shape = (5, 5)
        expected_adj = np.array([[0, 2, 0, 1, 0], [2, 0, 0, 1, 1], [0, 0, 0, 0, 3], [1, 1, 0, 0, 0], [0, 1, 3, 0, 0]])

        permute_indices = [1, 2, 4, 0, 3]

        permuted = self.bonds.permute_atoms(permute_indices)
        adj = permuted.adj_matrix(5)
        arr_equal = (adj == expected_adj).all()

        self.assertEqual(adj.shape, expected_shape)
        self.assertTrue(arr_equal)

    def test_permute_atoms_takes_bond_subset(self):
        expected_shape = (3, 3)
        expected_adj = np.array([[0, 2, 0], [2, 0, 0], [0, 0, 0]])
        expected_bonds = np.array([[0, 1, 2]])

        permute_indices = [1, 2, 4]

        permuted = self.bonds.permute_atoms(permute_indices)
        adj = permuted.adj_matrix(3)
        arr_equal = (adj == expected_adj).all()

        bond_equal = (permuted.bonds == expected_bonds).all()

        self.assertEqual(adj.shape, expected_shape)
        self.assertTrue(arr_equal)
        self.assertTrue(bond_equal)


class TestNeighbourRanks(unittest.TestCase):
    """Per-atom neighbour rank matrix: rank[i, j] = rank of j in i's neighbour
    list in bond row order (1-indexed). 0 means "no rank info".
    """

    def setUp(self):
        # Same fixture as TestBondSet for easy cross-referencing.
        bonds = np.array(
            [
                [0, 1, 1],  # row 0
                [1, 2, 2],  # row 1
                [2, 3, 1],  # row 2
                [3, 4, 3],  # row 3
                [0, 2, 1],  # row 4
            ]
        )
        self.bonds = BondSet(bonds)

    def test_full_ranks_match_bond_row_order(self):
        # Atom 0 sees its neighbours in row order: 1 (row 0), 2 (row 4) → ranks 1, 2.
        # Atom 2 sees: 1 (row 1), 3 (row 2), 0 (row 4) → ranks 1, 2, 3.
        # So rank[0, 2]=2 but rank[2, 0]=3 — the matrix is intentionally asymmetric.
        expected = np.array(
            [
                [0, 1, 2, 0, 0],
                [1, 0, 2, 0, 0],
                [3, 1, 0, 2, 0],
                [0, 0, 1, 0, 2],
                [0, 0, 0, 1, 0],
            ],
            dtype=np.int16,
        )

        ranks = self.bonds.neighbour_ranks(5)
        np.testing.assert_array_equal(ranks, expected)

    def test_ranks_are_asymmetric(self):
        # rank[0, 2] = 2 (2 is 2nd neighbour of 0)
        # rank[2, 0] = 3 (0 is 3rd neighbour of 2)
        ranks = self.bonds.neighbour_ranks(5)
        self.assertEqual(ranks[0, 2], 2)
        self.assertEqual(ranks[2, 0], 3)
        self.assertNotEqual(ranks[0, 2], ranks[2, 0])

    def test_no_bond_entries_are_zero(self):
        ranks = self.bonds.neighbour_ranks(5)
        # Atoms 0 and 4 aren't bonded → rank[0, 4] and rank[4, 0] should be 0
        self.assertEqual(ranks[0, 4], 0)
        self.assertEqual(ranks[4, 0], 0)

    def test_chiral_mask_only_populates_chiral_rows(self):
        # chiral_mask marks atom 2 as chiral → only row 2 is populated.
        chiral_mask = np.array([False, False, True, False, False])
        ranks = self.bonds.neighbour_ranks(5, chiral_mask=chiral_mask)

        expected = np.zeros((5, 5), dtype=np.int16)
        expected[2] = [3, 1, 0, 2, 0]
        np.testing.assert_array_equal(ranks, expected)

    def test_directional_bonds_force_ranks_at_endpoints(self):
        # No chiral atoms, but bond row 1 is directional (encoding 7 = "1_F_U").
        # That should still populate ranks at the bond's endpoints (atoms 1 and 2).
        bonds = BondSet(
            np.array(
                [
                    [0, 1, 1],  # plain single
                    [1, 2, 7],  # 1_F_U — directional
                    [2, 3, 1],  # plain single
                ]
            )
        )
        chiral_mask = np.zeros(4, dtype=bool)
        ranks = bonds.neighbour_ranks(4, chiral_mask=chiral_mask)

        # Atom 1 sees neighbours in row order: 0 (row 0), 2 (row 1)
        # Atom 2 sees neighbours in row order: 1 (row 1), 3 (row 2)
        expected = np.array(
            [
                [0, 0, 0, 0],
                [1, 0, 2, 0],
                [0, 1, 0, 2],
                [0, 0, 0, 0],
            ],
            dtype=np.int16,
        )
        np.testing.assert_array_equal(ranks, expected)

    def test_chiral_atom_and_directional_bond_combine(self):
        # Atom 0 is chiral AND atom 2 is in a directional bond → both rows populated.
        bonds = BondSet(
            np.array(
                [
                    [0, 1, 1],
                    [1, 2, 7],  # directional
                    [2, 3, 1],
                ]
            )
        )
        chiral_mask = np.array([True, False, False, False])
        ranks = bonds.neighbour_ranks(4, chiral_mask=chiral_mask)

        # Atom 0: chiral → ranks populated. Sees: 1 (row 0) only.
        # Atoms 1, 2: directional endpoints → ranks populated.
        # Atom 3: not chiral, not directional → all zero.
        expected = np.array(
            [
                [0, 1, 0, 0],
                [1, 0, 2, 0],
                [0, 1, 0, 2],
                [0, 0, 0, 0],
            ],
            dtype=np.int16,
        )
        np.testing.assert_array_equal(ranks, expected)

    def test_empty_bonds_returns_zeros(self):
        bs = BondSet(np.zeros((0, 3), dtype=np.int16))
        ranks = bs.neighbour_ranks(3)
        np.testing.assert_array_equal(ranks, np.zeros((3, 3), dtype=np.int16))

    def test_raises_when_atom_index_exceeds_n_atoms(self):
        with self.assertRaises(ValueError):
            self.bonds.neighbour_ranks(3)  # mol has atom index 4 in bonds

    def test_n_atoms_larger_than_used_pads_with_zeros(self):
        # Padding rows/cols beyond the actual atoms must stay 0.
        ranks = self.bonds.neighbour_ranks(7)
        self.assertEqual(ranks.shape, (7, 7))
        np.testing.assert_array_equal(ranks[5:], 0)
        np.testing.assert_array_equal(ranks[:, 5:], 0)


class TestBondSetFromRdkit(unittest.TestCase):
    def test_basic_from_rdkit(self):
        mol = Chem.MolFromSmiles("CCO")
        bonds = BondSet.from_rdkit(mol)
        self.assertEqual(len(bonds), 2)

    def test_aromatic_bonds(self):
        mol = Chem.MolFromSmiles("c1ccccc1")
        bonds = BondSet.from_rdkit(mol)
        self.assertEqual(len(bonds), 6)
        # All bonds should be aromatic
        for i in range(len(bonds)):
            _, is_arom, _ = BondEncoding.decode(bonds.types[i].item())
            self.assertTrue(is_arom)

    def test_upper_triangular(self):
        mol = Chem.MolFromSmiles("CCCC")
        bonds = BondSet.from_rdkit(mol)
        for i in range(len(bonds)):
            start, end, _ = bonds[i]
            self.assertLess(start, end)

    def test_ez_bonds_preserved(self):
        mol = Chem.MolFromSmiles("C/C=C/C")
        bonds = BondSet.from_rdkit(mol)
        has_direction = False
        for i in range(len(bonds)):
            _, _, direction = BondEncoding.decode(bonds.types[i].item())
            if direction is not None:
                has_direction = True

        self.assertTrue(has_direction)


class TestBondSetProperties(unittest.TestCase):
    def test_indices_and_types(self):
        bonds = np.array([[0, 1, 1], [1, 2, 2]])
        bs = BondSet(bonds)
        np.testing.assert_array_equal(bs.indices, [[0, 1], [1, 2]])
        np.testing.assert_array_equal(bs.types, [1, 2])

    def test_min_max_index(self):
        bonds = np.array([[2, 5, 1], [3, 7, 2]])
        bs = BondSet(bonds)
        self.assertEqual(bs.min_index, 2)
        self.assertEqual(bs.max_index, 7)

    def test_empty_bonds_min_max_none(self):
        bs = BondSet(np.zeros((0, 3), dtype=np.int16))
        self.assertIsNone(bs.min_index)
        self.assertIsNone(bs.max_index)

    def test_getitem_int(self):
        bonds = np.array([[0, 1, 3], [1, 2, 5]])
        bs = BondSet(bonds)
        result = bs[0]
        self.assertEqual(result, (0, 1, 3))

    def test_getitem_array(self):
        bonds = np.array([[0, 1, 1], [1, 2, 2], [2, 3, 3]])
        bs = BondSet(bonds)
        subset = bs[np.array([0, 2])]
        self.assertEqual(len(subset), 2)

    def test_copy(self):
        bonds = np.array([[0, 1, 1]])
        bs = BondSet(bonds)
        copied = bs.copy()
        np.testing.assert_array_equal(copied.bonds, bs.bonds)
        # Verify independence
        copied.bonds[0, 2] = 99
        self.assertNotEqual(bs.bonds[0, 2], 99)

    def test_dtype_is_int16(self):
        bonds = np.array([[0, 1, 1]], dtype=np.int32)
        bs = BondSet(bonds)
        self.assertEqual(bs.bonds.dtype, np.int16)


class TestBondSetDictRoundtrip(unittest.TestCase):
    def test_roundtrip(self):
        bonds = np.array([[0, 1, 1], [1, 2, 5]])
        bs = BondSet(bonds)
        d = bs.to_dict()
        restored = BondSet.from_dict(d)
        np.testing.assert_array_equal(restored.bonds, bs.bonds)


class TestBondSetArraysMerge(unittest.TestCase):
    def test_arrays_from_bonds_roundtrip(self):
        bs1 = BondSet(np.array([[0, 1, 1], [1, 2, 2]]))
        bs2 = BondSet(np.array([[0, 1, 3]]))

        arrays = BondSet.arrays_from_bonds([bs1, bs2])
        restored = BondSet.bonds_from_arrays(arrays)

        self.assertEqual(len(restored), 2)
        np.testing.assert_array_equal(restored[0].bonds, bs1.bonds)
        np.testing.assert_array_equal(restored[1].bonds, bs2.bonds)

    def test_permute_atoms_duplicate_raises(self):
        bs = BondSet(np.array([[0, 1, 1], [1, 2, 2]]))
        with self.assertRaises(ValueError):
            bs.permute_atoms([0, 0, 1])
