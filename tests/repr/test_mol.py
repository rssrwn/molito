import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from molito.core.atoms import AtomSet
from molito.core.bonds import BondSet
from molito.core.confs import ConfSet
from molito.mol.graph import GraphBatch, GraphMol


class TestGraphMol(unittest.TestCase):
    def setUp(self):
        # C-C=C-N with valid valences
        atomics1 = np.array([6, 6, 6, 7])
        charges1 = np.array([0, 0, 0, 0])
        atoms1 = AtomSet(atomics1, charges=charges1)

        atomics2 = np.array([6, 6, 6, 7, 1, 1])
        charges2 = np.array([0, 0, 0, 0, 0, 0])
        atoms2 = AtomSet(atomics2, charges=charges2)

        bonds = np.array(
            [
                [0, 1, 1],
                [1, 2, 2],
                [2, 3, 1],
            ]
        )
        bonds = BondSet(bonds)

        n_confs = 8
        n_atoms = 4
        confs = np.random.rand(n_confs, n_atoms, 3)
        confs = ConfSet(confs)

        mol = GraphMol(atoms1, bonds, confs=confs)
        graph_mol = GraphMol(atoms2, bonds.copy())

        self.mol = mol
        self.graph_mol = graph_mol

    def test_basic_mol_properties(self):
        exp_n_atoms = 6
        exp_n_heavy_atoms = 4
        exp_n_bonds = 3
        exp_n_confs = 8
        exp_n_confs_graph = 0

        self.assertEqual(exp_n_atoms, self.graph_mol.n_atoms)
        self.assertEqual(exp_n_heavy_atoms, self.graph_mol.n_heavy_atoms)
        self.assertEqual(exp_n_bonds, self.mol.n_bonds)
        self.assertEqual(exp_n_confs, self.mol.n_conformers)
        self.assertEqual(exp_n_confs_graph, self.graph_mol.n_conformers)

    def test_adj_provides_correct_connections(self):
        expected_shape = (4, 4)
        expected_adj = np.array([[0, 1, 0, 0], [1, 0, 2, 0], [0, 2, 0, 1], [0, 0, 1, 0]])

        adj = self.mol.adjacency
        arr_equal = (adj == expected_adj).all()

        self.assertEqual(adj.shape, expected_shape)
        self.assertTrue(arr_equal)

    def test_bytes_conversion(self):
        mol_bytes = self.mol.to_bytes()
        new_mol = GraphMol.from_bytes(mol_bytes)

        adj = self.mol.adjacency
        new_adj = new_mol.adjacency

        coords = self.mol.coords
        new_coords = new_mol.coords

        equal_adj = (adj == new_adj).all()
        equal_coords = (coords == new_coords).all()

        self.assertEqual(len(self.mol), len(new_mol))
        self.assertEqual(self.mol.charged_symbols, new_mol.charged_symbols)

        self.assertTrue(equal_adj)
        self.assertTrue(equal_coords)

    def test_rdkit_conversion(self):
        rdkit_mol = self.mol.to_rdkit()
        new_mol = GraphMol.from_rdkit(rdkit_mol)

        self.assertEqual(len(self.mol), len(new_mol))
        self.assertEqual(self.mol.n_conformers, new_mol.n_conformers)
        self.assertEqual(sorted(self.mol.charged_symbols), sorted(new_mol.charged_symbols))

    def test_pad_mol_equal_size(self):
        padded = self.mol.pad(len(self.mol))

        adj = self.mol.adjacency
        new_adj = padded.adjacency

        coords = self.mol.coords
        new_coords = padded.coords

        equal_adj = (adj == new_adj).all()
        equal_coords = (coords == new_coords).all()

        self.assertEqual(len(self.mol), len(padded))
        self.assertEqual(self.mol.charged_symbols, padded.charged_symbols)

        self.assertTrue(equal_adj)
        self.assertTrue(equal_coords)

    def test_pad_mol_smaller_size_throws_error(self):
        self.assertRaises(ValueError, self.mol.pad, len(self.mol) - 1)

    def test_pad_mol_pads_correctly(self):
        exp_charged_symbols = ["C_0", "C_0", "C_0", "N_0", "*_0", "*_0"]
        pad_coords = np.zeros((self.mol.n_conformers, 2, 3))
        exp_coords = np.concatenate((self.mol.coords, pad_coords), axis=1)

        padded = self.mol.pad(len(self.mol) + 2)

        bonds = self.mol.bonds.bonds
        new_bonds = padded.bonds.bonds
        new_coords = padded.coords

        equal_bonds = (bonds == new_bonds).all()
        correct_coords = (exp_coords == new_coords).all()

        self.assertTrue(equal_bonds)
        self.assertTrue(correct_coords)
        self.assertEqual(exp_charged_symbols, padded.charged_symbols)


class TestGraphMolFromRdkit(unittest.TestCase):
    def test_from_rdkit_preserves_hs(self):
        mol = Chem.MolFromSmiles("C")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        graph = GraphMol.from_rdkit(mol)
        self.assertEqual(graph.n_atoms, 5)

    def test_from_rdkit_preserves_conformers(self):
        mol = Chem.MolFromSmiles("CCCC")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMultipleConfs(mol, 3, randomSeed=42)
        graph = GraphMol.from_rdkit(mol)
        self.assertEqual(graph.n_conformers, 3)

    def test_from_rdkit_without_canonicalise(self):
        mol = Chem.MolFromSmiles("CCO")
        graph = GraphMol.from_rdkit(mol, canonicalise=False)
        self.assertEqual(graph.n_atoms, 3)

    def test_from_rdkit_ez_roundtrip(self):
        smi = "C/C=C/C"
        mol = Chem.MolFromSmiles(smi)
        graph = GraphMol.from_rdkit(mol)
        rdkit_mol = graph.to_rdkit()
        Chem.SanitizeMol(rdkit_mol)
        Chem.SetBondStereoFromDirections(rdkit_mol)
        result_smi = Chem.MolToSmiles(rdkit_mol)
        # The stereo should be preserved in the round-trip
        self.assertIn("/", result_smi)

    def test_from_rdkit_chirality_roundtrip(self):
        smi = "[C@@H](F)(Cl)Br"
        mol = Chem.MolFromSmiles(smi)
        graph = GraphMol.from_rdkit(mol)
        rdkit_mol = graph.to_rdkit()
        Chem.SanitizeMol(rdkit_mol)
        Chem.AssignStereochemistry(rdkit_mol, force=True)
        has_chiral = any(a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for a in rdkit_mol.GetAtoms())
        self.assertTrue(has_chiral)


class TestGraphMolTokens(unittest.TestCase):
    def test_tokens_basic(self):
        atomics = np.array([6, 7, 8])
        charges = np.array([0, 0, 0])
        atoms = AtomSet(atomics, charges=charges)
        bonds = BondSet(np.array([[0, 1, 1], [1, 2, 1]]))
        mol = GraphMol(atoms, bonds)
        self.assertEqual(mol.tokens, ["C_0", "N_0", "O_0"])

    def test_tokens_with_chirality(self):
        atomics = np.array([6, 6])
        charges = np.array([0, 0])
        chirality = np.array([1, 0], dtype=np.int8)
        atoms = AtomSet(atomics, charges=charges, chirality=chirality)
        bonds = BondSet(np.array([[0, 1, 1]]))
        mol = GraphMol(atoms, bonds)
        self.assertEqual(mol.tokens, ["C_0_CW", "C_0"])


class TestGraphMolTransforms(unittest.TestCase):
    def setUp(self):
        atomics = np.array([6, 6, 8, 1])
        charges = np.array([0, 0, 0, 0])
        atoms = AtomSet(atomics, charges=charges)
        bonds = BondSet(np.array([[0, 1, 1], [1, 2, 1], [0, 3, 1]]))
        coords = np.array([[[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0]]], dtype=np.float32)
        confs = ConfSet(coords)
        self.mol = GraphMol(atoms, bonds, confs=confs)

    def test_remove_hs(self):
        no_hs = self.mol.remove_hs()
        self.assertEqual(no_hs.n_atoms, 3)
        # Should only have C, C, O
        self.assertTrue(all(a != 1 for a in no_hs.atomics.tolist()))

    def test_remove_hs_preserves_coords(self):
        no_hs = self.mol.remove_hs()
        self.assertIsNotNone(no_hs.coords)
        self.assertEqual(no_hs.coords.shape, (1, 3, 3))

    def test_zero_com(self):
        zeroed = self.mol.zero_com()
        com = zeroed.coords.mean(axis=1)
        np.testing.assert_array_almost_equal(com, np.zeros((1, 3)), decimal=5)

    def test_shift(self):
        shift_vec = np.array([10, 20, 30], dtype=np.float32)
        shifted = self.mol.shift(shift_vec)
        expected = self.mol.coords + shift_vec
        np.testing.assert_array_almost_equal(shifted.coords, expected)

    def test_scale(self):
        scaled = self.mol.scale(2.0)
        np.testing.assert_array_almost_equal(scaled.coords, self.mol.coords * 2.0)

    def test_drop_3d(self):
        no_3d = self.mol.drop_3d()
        self.assertIsNone(no_3d.confs)
        self.assertEqual(no_3d.n_atoms, self.mol.n_atoms)

    def test_mol_with_conformer(self):
        # Build a mol with multiple conformers and pick one out
        rdkit_mol = Chem.MolFromSmiles("CCO")
        rdkit_mol = Chem.AddHs(rdkit_mol)
        AllChem.EmbedMultipleConfs(rdkit_mol, numConfs=3, randomSeed=0)
        multi = GraphMol.from_rdkit(rdkit_mol)
        self.assertEqual(multi.n_conformers, 3)

        single = multi.mol_with_conformer(1)
        self.assertEqual(single.n_conformers, 1)
        np.testing.assert_array_equal(single.atomics, multi.atomics)
        np.testing.assert_array_almost_equal(single.coords[0], multi.coords[1])

    def test_mol_with_conformer_raises_when_no_confs(self):
        rdkit_mol = Chem.MolFromSmiles("CCO")
        mol = GraphMol.from_rdkit(rdkit_mol)  # No confs from SMILES
        with self.assertRaises(ValueError):
            mol.mol_with_conformer(0)

    def test_neighbour_ranks_full(self):
        # Default (stereo_only=False) returns ranks for every neighbour pair.
        from molito.core.atoms import AtomSet
        from molito.core.bonds import BondSet

        atoms = AtomSet(np.array([6, 6, 6], dtype=np.uint8))
        bonds = BondSet(np.array([[0, 1, 1], [1, 2, 1]], dtype=np.int16))
        mol = GraphMol(atoms, bonds)

        ranks = mol.neighbour_ranks()
        expected = np.array(
            [
                [0, 1, 0],
                [1, 0, 2],
                [0, 1, 0],
            ],
            dtype=np.int16,
        )
        np.testing.assert_array_equal(ranks, expected)

    def test_neighbour_ranks_stereo_only_uses_chirality(self):
        # With stereo_only=True, only chiral atoms (and directional-bond endpoints)
        # get their ranks populated.
        from molito.core.atoms import AtomSet
        from molito.core.bonds import BondSet

        # Atom 1 is chiral (CW=1); others are not.
        chirality = np.array([0, 1, 0, 0], dtype=np.int8)
        atoms = AtomSet(np.array([6, 6, 6, 6], dtype=np.uint8), chirality=chirality)
        bonds = BondSet(
            np.array(
                [
                    [0, 1, 1],
                    [1, 2, 1],
                    [1, 3, 1],
                ],
                dtype=np.int16,
            )
        )
        mol = GraphMol(atoms, bonds)

        ranks = mol.neighbour_ranks(stereo_only=True)

        # Only row 1 should be non-zero — atom 1 sees neighbours 0, 2, 3 in row order.
        expected = np.zeros((4, 4), dtype=np.int16)
        expected[1] = [1, 0, 2, 3]
        np.testing.assert_array_equal(ranks, expected)

    def test_neighbour_ranks_stereo_only_includes_directional_endpoints(self):
        # Directional bonds force their endpoints to have ranks even with no chiral atoms.
        from molito.core.atoms import AtomSet
        from molito.core.bonds import BondSet

        atoms = AtomSet(np.array([6, 6, 6, 6], dtype=np.uint8))  # chirality all 0
        bonds = BondSet(
            np.array(
                [
                    [0, 1, 1],
                    [1, 2, 7],  # 1_F_U
                    [2, 3, 1],
                ],
                dtype=np.int16,
            )
        )
        mol = GraphMol(atoms, bonds)

        ranks = mol.neighbour_ranks(stereo_only=True)

        # Atoms 1, 2 (directional bond endpoints) populated. Atoms 0, 3 untouched.
        self.assertTrue((ranks[0] == 0).all())
        self.assertTrue((ranks[3] == 0).all())
        self.assertEqual(ranks[1, 0], 1)
        self.assertEqual(ranks[1, 2], 2)

    def test_mol_with_conformer_drops_weights(self):
        # Confs with zero weight (extreme-energy underflow, masked entries, etc.) must
        # still extract cleanly. Weights are dropped from the singleton result because
        # a per-conf weight only carries info relative to the rest of the ensemble.
        from molito.core.atoms import AtomSet
        from molito.core.bonds import BondSet
        from molito.core.confs import ConfSet

        coords = np.random.randn(3, 4, 3).astype(np.float32)
        weights = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # one dominant, two zeroed

        atoms = AtomSet(np.array([6, 6, 8, 1], dtype=np.uint8))
        bonds = BondSet(np.zeros((0, 3), dtype=np.int16))
        confs = ConfSet(coords, weights=weights)
        mol = GraphMol(atoms, bonds, confs=confs)

        # Index 1 has weight 0; previously this raised in the ConfSet constructor
        single = mol.mol_with_conformer(1)
        self.assertEqual(single.n_conformers, 1)
        self.assertFalse(single.confs.has_weights)
        np.testing.assert_array_almost_equal(single.coords[0], coords[1])

    def test_copy_is_independent(self):
        copied = self.mol.copy()
        self.assertEqual(copied.n_atoms, self.mol.n_atoms)
        np.testing.assert_array_equal(copied.atomics, self.mol.atomics)


class TestGraphMolInvalidInput(unittest.TestCase):
    def _make_invalid_rdkit_mol(self):
        """Create an RDKit mol with invalid chemistry (oxygen with valence 5)."""

        mol = Chem.RWMol()
        mol.AddAtom(Chem.Atom(6))  # C
        mol.AddAtom(Chem.Atom(8))  # O
        mol.AddAtom(Chem.Atom(6))  # C
        mol.AddAtom(Chem.Atom(6))  # C
        mol.AddBond(0, 1, Chem.BondType.DOUBLE)
        mol.AddBond(1, 2, Chem.BondType.SINGLE)
        mol.AddBond(1, 3, Chem.BondType.TRIPLE)
        return mol.GetMol()

    def test_from_rdkit_invalid_mol_raises_with_canonicalise(self):
        mol = self._make_invalid_rdkit_mol()
        with self.assertRaises(ValueError):
            GraphMol.from_rdkit(mol, canonicalise=True)

    def test_from_rdkit_invalid_mol_loads_without_canonicalise(self):
        mol = self._make_invalid_rdkit_mol()
        graph_mol = GraphMol.from_rdkit(mol, canonicalise=False)

        self.assertEqual(graph_mol.n_atoms, 4)
        self.assertEqual(graph_mol.n_bonds, 3)

    def test_order_by_bonds_invalid_mol_raises(self):
        mol = self._make_invalid_rdkit_mol()
        graph_mol = GraphMol.from_rdkit(mol, canonicalise=False)

        with self.assertRaises(ValueError):
            graph_mol.order_by_bonds()


class TestGraphBatchMixedWeights(unittest.TestCase):
    """Test batching/unbatching GraphMols with mixed weight configurations and conformer counts."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def _create_mol_with_confs(self, n_atoms: int, n_confs: int, has_weights: bool) -> GraphMol:
        """Create a test molecule with specified number of conformers and optional weights."""

        atomics = np.array([6, 7, 8, 1] * n_atoms)[:n_atoms]
        charges = np.zeros(n_atoms, dtype=np.int16)
        atoms = AtomSet(atomics, charges=charges)

        bonds = BondSet(np.array([[i, i + 1, 1] for i in range(n_atoms - 1)]))

        coords = np.random.rand(n_confs, n_atoms, 3).astype(np.float32)
        weights = np.random.rand(n_confs).astype(np.float32) if has_weights else None
        confs = ConfSet(coords, weights=weights)

        return GraphMol(atoms, bonds, confs=confs)

    def _create_mol_no_confs(self, n_atoms: int) -> GraphMol:
        """Create a test molecule without conformers."""

        atomics = np.array([6, 7, 8, 1] * n_atoms)[:n_atoms]
        charges = np.zeros(n_atoms, dtype=np.int16)
        atoms = AtomSet(atomics, charges=charges)

        bonds = BondSet(np.array([[i, i + 1, 1] for i in range(n_atoms - 1)]))

        return GraphMol(atoms, bonds, confs=None)

    def test_batch_unbatch_mixed_weights_same_n_confs(self):
        """Test mols with mixed weight configs but same number of conformers."""

        mol_with_weights_1 = self._create_mol_with_confs(5, 3, has_weights=True)
        mol_no_weights = self._create_mol_with_confs(6, 3, has_weights=False)
        mol_with_weights_2 = self._create_mol_with_confs(4, 3, has_weights=True)

        mols = [mol_with_weights_1, mol_no_weights, mol_with_weights_2]
        batch = GraphBatch(mols)

        save_path = Path(self.temp_dir) / "mixed_weights"
        batch.save(save_path)

        loaded = GraphBatch.load(save_path)

        self.assertEqual(len(loaded), 3)

        # Check mol 0 (had weights)
        self.assertEqual(loaded[0].n_atoms, 5)
        self.assertEqual(loaded[0].n_conformers, 3)
        self.assertTrue(loaded[0].confs.has_weights)
        np.testing.assert_array_almost_equal(loaded[0].coords, mol_with_weights_1.coords)
        np.testing.assert_array_almost_equal(loaded[0].confs.weights, mol_with_weights_1.confs.weights)

        # Check mol 1 (no weights)
        self.assertEqual(loaded[1].n_atoms, 6)
        self.assertEqual(loaded[1].n_conformers, 3)
        self.assertFalse(loaded[1].confs.has_weights)
        np.testing.assert_array_almost_equal(loaded[1].coords, mol_no_weights.coords)

        # Check mol 2 (had weights)
        self.assertEqual(loaded[2].n_atoms, 4)
        self.assertEqual(loaded[2].n_conformers, 3)
        self.assertTrue(loaded[2].confs.has_weights)
        np.testing.assert_array_almost_equal(loaded[2].coords, mol_with_weights_2.coords)
        np.testing.assert_array_almost_equal(loaded[2].confs.weights, mol_with_weights_2.confs.weights)

        loaded.close_hdf5()

    def test_batch_unbatch_different_n_confs_all_weights(self):
        """Test mols with different numbers of conformers, all with weights."""

        mol1 = self._create_mol_with_confs(5, 2, has_weights=True)
        mol2 = self._create_mol_with_confs(6, 5, has_weights=True)
        mol3 = self._create_mol_with_confs(4, 1, has_weights=True)

        mols = [mol1, mol2, mol3]
        batch = GraphBatch(mols)

        save_path = Path(self.temp_dir) / "diff_confs_weights"
        batch.save(save_path)

        loaded = GraphBatch.load(save_path)

        self.assertEqual(len(loaded), 3)

        for i, (orig, restored) in enumerate(zip(mols, loaded, strict=True)):
            self.assertEqual(restored.n_atoms, orig.n_atoms, f"Mol {i} n_atoms mismatch")
            self.assertEqual(restored.n_conformers, orig.n_conformers, f"Mol {i} n_confs mismatch")
            self.assertTrue(restored.confs.has_weights, f"Mol {i} should have weights")
            np.testing.assert_array_almost_equal(restored.coords, orig.coords, err_msg=f"Mol {i} coords mismatch")
            np.testing.assert_array_almost_equal(
                restored.confs.weights, orig.confs.weights, err_msg=f"Mol {i} weights mismatch"
            )

        loaded.close_hdf5()

    def test_batch_unbatch_different_n_confs_no_weights(self):
        """Test mols with different numbers of conformers, none with weights."""

        mol1 = self._create_mol_with_confs(5, 4, has_weights=False)
        mol2 = self._create_mol_with_confs(6, 1, has_weights=False)
        mol3 = self._create_mol_with_confs(4, 7, has_weights=False)

        mols = [mol1, mol2, mol3]
        batch = GraphBatch(mols)

        save_path = Path(self.temp_dir) / "diff_confs_no_weights"
        batch.save(save_path)

        loaded = GraphBatch.load(save_path)

        self.assertEqual(len(loaded), 3)

        for i, (orig, restored) in enumerate(zip(mols, loaded, strict=True)):
            self.assertEqual(restored.n_atoms, orig.n_atoms, f"Mol {i} n_atoms mismatch")
            self.assertEqual(restored.n_conformers, orig.n_conformers, f"Mol {i} n_confs mismatch")
            self.assertFalse(restored.confs.has_weights, f"Mol {i} should not have weights")
            np.testing.assert_array_almost_equal(restored.coords, orig.coords, err_msg=f"Mol {i} coords mismatch")

        loaded.close_hdf5()

    def test_batch_unbatch_different_n_confs_mixed_weights(self):
        """Test mols with different numbers of conformers and mixed weight configs."""

        mol1 = self._create_mol_with_confs(5, 2, has_weights=True)
        mol2 = self._create_mol_with_confs(6, 5, has_weights=False)
        mol3 = self._create_mol_with_confs(4, 1, has_weights=True)
        mol4 = self._create_mol_with_confs(7, 3, has_weights=False)

        mols = [mol1, mol2, mol3, mol4]
        batch = GraphBatch(mols)

        save_path = Path(self.temp_dir) / "diff_confs_mixed"
        batch.save(save_path)

        loaded = GraphBatch.load(save_path)

        self.assertEqual(len(loaded), 4)

        # Check each mol
        expected_weights = [True, False, True, False]
        expected_n_confs = [2, 5, 1, 3]

        for i, (orig, restored) in enumerate(zip(mols, loaded, strict=True)):
            self.assertEqual(restored.n_atoms, orig.n_atoms, f"Mol {i} n_atoms mismatch")
            self.assertEqual(restored.n_conformers, expected_n_confs[i], f"Mol {i} n_confs mismatch")
            self.assertEqual(restored.confs.has_weights, expected_weights[i], f"Mol {i} has_weights mismatch")
            np.testing.assert_array_almost_equal(restored.coords, orig.coords, err_msg=f"Mol {i} coords mismatch")

            if expected_weights[i]:
                np.testing.assert_array_almost_equal(
                    restored.confs.weights, orig.confs.weights, err_msg=f"Mol {i} weights mismatch"
                )

        loaded.close_hdf5()

    def test_batch_unbatch_all_no_confs(self):
        """Test batch where all mols have no conformers."""

        mols = [
            self._create_mol_no_confs(5),
            self._create_mol_no_confs(6),
            self._create_mol_no_confs(4),
        ]
        batch = GraphBatch(mols)

        save_path = Path(self.temp_dir) / "all_no_confs"
        batch.save(save_path)

        loaded = GraphBatch.load(save_path)

        self.assertEqual(len(loaded), 3)

        for i, (orig, restored) in enumerate(zip(mols, loaded, strict=True)):
            self.assertEqual(restored.n_atoms, orig.n_atoms, f"Mol {i} n_atoms mismatch")
            self.assertEqual(restored.n_conformers, 0, f"Mol {i} should have no conformers")
            self.assertIsNone(restored.confs, f"Mol {i} confs should be None")

        loaded.close_hdf5()

    def test_batch_unbatch_preserves_all_properties(self):
        """Test that all mol properties are preserved through batch/unbatch."""

        mols = [
            self._create_mol_with_confs(5, 2, has_weights=True),
            self._create_mol_with_confs(7, 4, has_weights=False),
            self._create_mol_with_confs(4, 1, has_weights=True),
        ]
        batch = GraphBatch(mols)

        save_path = Path(self.temp_dir) / "full_check"
        batch.save(save_path)

        loaded = GraphBatch.load(save_path)

        for i, (orig, restored) in enumerate(zip(mols, loaded, strict=True)):
            # Check atom properties
            np.testing.assert_array_equal(restored.atomics, orig.atomics, err_msg=f"Mol {i} atomics mismatch")
            np.testing.assert_array_equal(restored.charges, orig.charges, err_msg=f"Mol {i} charges mismatch")

            # Check bonds
            np.testing.assert_array_equal(
                restored.bond_indices, orig.bond_indices, err_msg=f"Mol {i} bond_indices mismatch"
            )
            np.testing.assert_array_equal(restored.bond_types, orig.bond_types, err_msg=f"Mol {i} bond_types mismatch")

            # Check conformers
            self.assertEqual(restored.n_conformers, orig.n_conformers, f"Mol {i} n_confs mismatch")
            np.testing.assert_array_almost_equal(restored.coords, orig.coords, err_msg=f"Mol {i} coords mismatch")

            if orig.confs.has_weights:
                np.testing.assert_array_almost_equal(
                    restored.confs.weights, orig.confs.weights, err_msg=f"Mol {i} weights mismatch"
                )

        loaded.close_hdf5()


class TestGraphBatchNeighbourRanks(unittest.TestCase):
    """GraphBatch.neighbour_ranks pads per-mol rank matrices to max atom count
    across the batch and stacks them into a [B, max_atoms, max_atoms] tensor.
    """

    def test_padding_across_different_sized_mols(self):
        from molito.core.atoms import AtomSet
        from molito.core.bonds import BondSet

        # Mol 1: 3 atoms in a chain
        mol1 = GraphMol(
            AtomSet(np.array([6, 6, 8], dtype=np.uint8)),
            BondSet(np.array([[0, 1, 1], [1, 2, 1]], dtype=np.int16)),
        )

        # Mol 2: 5 atoms in a chain (larger)
        mol2 = GraphMol(
            AtomSet(np.array([6, 6, 6, 6, 8], dtype=np.uint8)),
            BondSet(np.array([[0, 1, 1], [1, 2, 1], [2, 3, 1], [3, 4, 1]], dtype=np.int16)),
        )

        batch = GraphBatch([mol1, mol2])
        ranks = batch.neighbour_ranks()

        # Padded to max_length = 5
        self.assertEqual(ranks.shape, (2, 5, 5))

        # Mol 1 ranks fill its 3x3 block; padding rows/cols past index 3 stay 0.
        self.assertTrue((ranks[0, 3:] == 0).all())
        self.assertTrue((ranks[0, :, 3:] == 0).all())
        # The chain pattern: rank[i, i+1] = 1; rank[i+1, i] varies by row order.
        self.assertEqual(ranks[0, 0, 1], 1)
        self.assertEqual(ranks[0, 1, 0], 1)

    def test_stereo_only_respects_per_mol_chirality(self):
        from molito.core.atoms import AtomSet
        from molito.core.bonds import BondSet

        # Mol 1: no chirality, no directional bonds → empty ranks in stereo-only mode
        mol1 = GraphMol(
            AtomSet(np.array([6, 6, 8], dtype=np.uint8)),
            BondSet(np.array([[0, 1, 1], [1, 2, 1]], dtype=np.int16)),
        )

        # Mol 2: atom 1 is chiral → only row 1 populated in stereo-only mode
        chirality = np.array([0, 1, 0, 0, 0], dtype=np.int8)
        mol2 = GraphMol(
            AtomSet(np.array([6, 6, 6, 6, 8], dtype=np.uint8), chirality=chirality),
            BondSet(np.array([[0, 1, 1], [1, 2, 1], [2, 3, 1], [3, 4, 1]], dtype=np.int16)),
        )

        ranks = GraphBatch([mol1, mol2]).neighbour_ranks(stereo_only=True)

        # Mol 1: all zeros (no chiral, no directional)
        np.testing.assert_array_equal(ranks[0], 0)

        # Mol 2: only row 1 has rank values (atom 1 sees atom 0 then atom 2)
        self.assertEqual(ranks[1, 1, 0], 1)
        self.assertEqual(ranks[1, 1, 2], 2)
        # Other rows zero
        for r in (0, 2, 3, 4):
            self.assertTrue((ranks[1, r] == 0).all(), f"row {r} should be zero")


class TestGraphBatchBytes(unittest.TestCase):
    """Byte round-trip for a whole batch, mirroring GraphMol.to_bytes/from_bytes."""

    def setUp(self):
        smis = ["CCO", "c1ccccc1", "C/C=C/C"]
        self.mols = [GraphMol.from_rdkit(Chem.MolFromSmiles(smi)) for smi in smis]

    def test_bytes_conversion(self):
        batch = GraphBatch(self.mols)
        restored = GraphBatch.from_bytes(batch.to_bytes())

        self.assertIsInstance(restored, GraphBatch)
        self.assertEqual(len(restored), len(batch))

        for original, new in zip(batch, restored, strict=True):
            self.assertEqual(len(original), len(new))
            self.assertEqual(original.charged_symbols, new.charged_symbols)
            np.testing.assert_array_equal(original.adjacency, new.adjacency)

    def test_bytes_conversion_preserves_coords(self):
        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        AllChem.EmbedMolecule(mol, randomSeed=42)

        batch = GraphBatch([GraphMol.from_rdkit(mol)])
        restored = GraphBatch.from_bytes(batch.to_bytes())

        np.testing.assert_array_equal(batch[0].coords, restored[0].coords)

    def test_bytes_conversion_preserves_meta(self):
        for idx, mol in enumerate(self.mols):
            mol.meta = {"id": str(idx)}

        restored = GraphBatch.from_bytes(GraphBatch(self.mols).to_bytes())
        self.assertEqual([dict(mol.meta) for mol in restored], [{"id": "0"}, {"id": "1"}, {"id": "2"}])
