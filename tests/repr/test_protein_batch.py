import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from molito.core.atoms import AtomSet
from molito.core.bonds import BondSet
from molito.core.confs import ConfSet
from molito.mol.protein import Protein, ProteinBatch


def create_test_protein(n_atoms: int = 5, n_residues: int = 2, meta: dict | None = None) -> Protein:
    """Helper function to create a test protein."""

    atomics = np.array([6, 7, 8] * n_atoms)[:n_atoms]
    charges = np.zeros(n_atoms, dtype=np.int16)

    # Create residue assignments
    atoms_per_res = n_atoms // n_residues
    res_ids = np.array([i // atoms_per_res + 1 for i in range(n_atoms)])
    res_names = np.array(["ALA" if r % 2 == 1 else "GLY" for r in res_ids])
    atom_names = np.array(["CA", "N", "O", "CB", "C"][:n_atoms] * (n_atoms // 5 + 1))[:n_atoms]

    atoms = AtomSet(atomics, charges=charges, res_names=res_names, atom_names=atom_names, res_ids=res_ids)

    # Create bonds: linear chain
    bond_arr = np.array([[i, i + 1, 1] for i in range(n_atoms - 1)])
    bonds = BondSet(bond_arr)

    # Create single conformer
    coords = np.random.rand(1, n_atoms, 3).astype(np.float32)
    confs = ConfSet(coords)

    return Protein(atoms, bonds, confs, meta=meta)


class TestProteinBatch(unittest.TestCase):
    def setUp(self):
        self.protein1 = create_test_protein(5, 2, meta={"name": "protein1"})
        self.protein2 = create_test_protein(7, 3, meta={"name": "protein2"})
        self.protein3 = create_test_protein(4, 2, meta={"name": "protein3"})

        self.batch = ProteinBatch([self.protein1, self.protein2, self.protein3])

    def test_length(self):
        self.assertEqual(len(self.batch), 3)

    def test_getitem(self):
        protein = self.batch[0]
        self.assertIsInstance(protein, Protein)
        self.assertEqual(protein.n_atoms, 5)

        protein = self.batch[1]
        self.assertEqual(protein.n_atoms, 7)

    def test_lengths_property(self):
        lengths = self.batch.lengths
        self.assertEqual(lengths, [5, 7, 4])

    def test_mask_property(self):
        mask = self.batch.mask
        self.assertEqual(mask.shape, (3, 7))  # batch_size x max_length
        np.testing.assert_array_equal(mask[0, :5], np.ones(5))
        np.testing.assert_array_equal(mask[0, 5:], np.zeros(2))

    def test_atomics_property(self):
        atomics = self.batch.atomics
        self.assertEqual(atomics.shape, (3, 7))  # padded to max length
        np.testing.assert_array_equal(atomics[0, :5], self.protein1.atomics)

    def test_charges_property(self):
        charges = self.batch.charges
        self.assertEqual(charges.shape, (3, 7))
        np.testing.assert_array_equal(charges[0, :5], self.protein1.charges)

    def test_res_names_property(self):
        res_names = self.batch.res_names
        self.assertEqual(res_names.shape, (3, 7))
        self.assertEqual(res_names[0, :5].tolist(), self.protein1.res_names)

    def test_atom_names_property(self):
        atom_names = self.batch.atom_names
        self.assertEqual(atom_names.shape, (3, 7))
        self.assertEqual(atom_names[0, :5].tolist(), self.protein1.atom_names)

    def test_res_ids_property(self):
        res_ids = self.batch.res_ids
        self.assertEqual(res_ids.shape, (3, 7))
        np.testing.assert_array_equal(res_ids[0, :5], self.protein1.res_ids)

    def test_coords_property(self):
        coords = self.batch.coords
        # Shape: [batch, max_atoms, 3] - no conformer dim since Protein.coords is 2D
        self.assertEqual(coords.shape, (3, 7, 3))
        np.testing.assert_array_equal(coords[0, :5, :], self.protein1.coords)

    def test_bond_indices_property(self):
        bond_indices = self.batch.bond_indices
        self.assertEqual(bond_indices.shape[0], 3)  # batch size

    def test_bond_types_property(self):
        bond_types = self.batch.bond_types
        self.assertEqual(bond_types.shape[0], 3)  # batch size

    def test_adjacency_property(self):
        adj = self.batch.adjacency
        self.assertEqual(adj.shape, (3, 7, 7))
        # Check symmetry
        self.assertTrue((adj == adj.transpose(0, 2, 1)).all())

    def test_subset(self):
        subset = self.batch.subset([0, 2])
        self.assertEqual(len(subset), 2)
        self.assertEqual(subset[0].n_atoms, 5)
        self.assertEqual(subset[1].n_atoms, 4)

    def test_iteration(self):
        proteins = list(self.batch)
        self.assertEqual(len(proteins), 3)
        self.assertIsInstance(proteins[0], Protein)


class TestProteinBatchSerialization(unittest.TestCase):
    def setUp(self):
        self.protein1 = create_test_protein(5, 2, meta={"name": "protein1"})
        self.protein2 = create_test_protein(7, 3, meta={"name": "protein2"})
        self.batch = ProteinBatch([self.protein1, self.protein2])
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_bytes_serialization(self):
        data = self.batch.to_bytes()
        restored = ProteinBatch.from_bytes(data)

        self.assertEqual(len(restored), len(self.batch))
        self.assertEqual(restored[0].n_atoms, self.protein1.n_atoms)
        self.assertEqual(restored[1].n_atoms, self.protein2.n_atoms)
        np.testing.assert_array_equal(restored[0].atomics, self.protein1.atomics)

    def test_save_load_single_shard(self):
        save_path = Path(self.temp_dir) / "proteins"
        self.batch.save(save_path)

        loaded = ProteinBatch.load(save_path)

        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0].n_atoms, self.protein1.n_atoms)
        self.assertEqual(loaded[1].n_atoms, self.protein2.n_atoms)
        np.testing.assert_array_almost_equal(loaded[0].coords, self.protein1.coords)
        np.testing.assert_array_almost_equal(loaded[1].coords, self.protein2.coords)

        # Check metadata
        self.assertEqual(loaded[0].meta["name"], "protein1")
        self.assertEqual(loaded[1].meta["name"], "protein2")

        loaded.close_hdf5()

    def test_save_load_multiple_shards(self):
        # Create a larger batch
        proteins = [create_test_protein(5 + i, 2, meta={"idx": str(i)}) for i in range(5)]
        batch = ProteinBatch(proteins)

        save_path = Path(self.temp_dir) / "sharded_proteins"
        batch.save(save_path, shard_size=2)

        # Verify shards were created
        shard_files = list(save_path.glob("*.hdf5"))
        self.assertEqual(len(shard_files), 3)  # 5 proteins / 2 per shard = 3 shards

        loaded = ProteinBatch.load(save_path)

        self.assertEqual(len(loaded), 5)
        for i in range(5):
            self.assertEqual(loaded[i].n_atoms, 5 + i)
            self.assertEqual(loaded[i].meta["idx"], str(i))

        loaded.close_hdf5()

    def test_save_load_preserves_annotations(self):
        save_path = Path(self.temp_dir) / "annotated_proteins"
        self.batch.save(save_path)

        loaded = ProteinBatch.load(save_path)

        # Check all annotation arrays are preserved
        np.testing.assert_array_equal(loaded[0].res_names, self.protein1.res_names)
        np.testing.assert_array_equal(loaded[0].atom_names, self.protein1.atom_names)
        np.testing.assert_array_equal(loaded[0].res_ids, self.protein1.res_ids)

        loaded.close_hdf5()

    def test_save_load_fixed_length_strings(self):
        """Test that fixed-length strings are preserved correctly."""

        # Create atoms with strings at max allowed lengths
        atomics = np.array([6, 7, 8])
        charges = np.zeros(3, dtype=np.int16)
        res_names = np.array(["ALA", "GLY", "A"])
        atom_names = np.array(["CA", "N", "O"])
        res_ids = np.array([1, 1, 2])

        atoms = AtomSet(atomics, charges=charges, res_names=res_names, atom_names=atom_names, res_ids=res_ids)

        bonds = BondSet(np.array([[0, 1, 1], [1, 2, 1]]))
        coords = np.random.rand(1, 3, 3).astype(np.float32)
        confs = ConfSet(coords)
        protein = Protein(atoms, bonds, confs)

        batch = ProteinBatch([protein])
        save_path = Path(self.temp_dir) / "fixedlen_strings"
        batch.save(save_path)

        loaded = ProteinBatch.load(save_path)

        # Verify all strings are preserved
        np.testing.assert_array_equal(loaded[0].res_names, res_names)
        np.testing.assert_array_equal(loaded[0].atom_names, atom_names)

        loaded.close_hdf5()

    def test_save_to_existing_nonempty_dir_raises_error(self):
        save_path = Path(self.temp_dir) / "existing"
        save_path.mkdir()
        (save_path / "dummy.txt").write_text("dummy")

        with self.assertRaises(RuntimeError):
            self.batch.save(save_path)

    def test_load_nonexistent_path_raises_error(self):
        with self.assertRaises(RuntimeError):
            ProteinBatch.load(Path(self.temp_dir) / "nonexistent")


class TestProteinBatchFromBatches(unittest.TestCase):
    def test_from_batches(self):
        protein1 = create_test_protein(5, 2)
        protein2 = create_test_protein(7, 3)
        protein3 = create_test_protein(4, 2)

        batch1 = ProteinBatch([protein1])
        batch2 = ProteinBatch([protein2, protein3])

        combined = ProteinBatch.from_batches([batch1, batch2])

        self.assertEqual(len(combined), 3)
        self.assertEqual(combined[0].n_atoms, 5)
        self.assertEqual(combined[1].n_atoms, 7)
        self.assertEqual(combined[2].n_atoms, 4)


class TestProteinBatchValidation(unittest.TestCase):
    def test_non_protein_raises_error(self):
        with self.assertRaises(TypeError):
            ProteinBatch(["not a protein"])

    def test_different_conformer_counts_still_works(self):
        """Proteins with different conformer counts can be batched since Protein.coords returns 2D."""

        protein1 = create_test_protein(5, 2)

        # Create protein with 2 conformers
        atoms2 = protein1.atoms.copy()
        bonds2 = protein1.bonds.copy()
        coords2 = np.random.rand(2, 5, 3).astype(np.float32)
        confs2 = ConfSet(coords2)
        protein2 = Protein(atoms2, bonds2, confs2)

        batch = ProteinBatch([protein1, protein2])

        # Should work since Protein.coords only returns first conformer (2D)
        coords = batch.coords
        self.assertEqual(coords.shape, (2, 5, 3))


class TestProteinBatchMixedWeights(unittest.TestCase):
    """Test batching/unbatching proteins with mixed weight configurations."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def _create_protein_with_confs(self, n_atoms: int, n_confs: int, has_weights: bool) -> Protein:
        """Create a test protein with specified number of conformers and optional weights."""

        atomics = np.array([6, 7, 8] * n_atoms)[:n_atoms]
        charges = np.zeros(n_atoms, dtype=np.int16)
        res_names = np.array(["ALA"] * n_atoms)
        atom_names = np.array(["CA", "N", "O", "CB"][:n_atoms] * (n_atoms // 4 + 1))[:n_atoms]
        res_ids = np.array([1] * n_atoms)

        atoms = AtomSet(atomics, charges=charges, res_names=res_names, atom_names=atom_names, res_ids=res_ids)

        bonds = BondSet(np.array([[i, i + 1, 1] for i in range(n_atoms - 1)]))

        coords = np.random.rand(n_confs, n_atoms, 3).astype(np.float32)
        weights = np.random.rand(n_confs).astype(np.float32) if has_weights else None
        confs = ConfSet(coords, weights=weights)

        return Protein(atoms, bonds, confs)

    def test_batch_unbatch_mixed_weights_same_n_confs(self):
        """Test that proteins with mixed weight configs but same n_confs are preserved through save/load."""

        # Create proteins: some with weights, some without
        protein_with_weights_1 = self._create_protein_with_confs(5, 2, has_weights=True)
        protein_no_weights = self._create_protein_with_confs(6, 2, has_weights=False)
        protein_with_weights_2 = self._create_protein_with_confs(4, 2, has_weights=True)

        proteins = [protein_with_weights_1, protein_no_weights, protein_with_weights_2]
        batch = ProteinBatch(proteins)

        save_path = Path(self.temp_dir) / "mixed_weights"
        batch.save(save_path)

        loaded = ProteinBatch.load(save_path)

        self.assertEqual(len(loaded), 3)

        # Check protein 0 (had weights)
        self.assertEqual(loaded[0].n_atoms, 5)
        self.assertEqual(loaded[0].confs.n_conformers, 2)
        self.assertTrue(loaded[0].confs.has_weights)
        np.testing.assert_array_almost_equal(loaded[0].coords, protein_with_weights_1.coords)
        np.testing.assert_array_almost_equal(loaded[0].confs.weights, protein_with_weights_1.confs.weights)

        # Check protein 1 (no weights)
        self.assertEqual(loaded[1].n_atoms, 6)
        self.assertEqual(loaded[1].confs.n_conformers, 2)
        self.assertFalse(loaded[1].confs.has_weights)
        np.testing.assert_array_almost_equal(loaded[1].coords, protein_no_weights.coords)

        # Check protein 2 (had weights)
        self.assertEqual(loaded[2].n_atoms, 4)
        self.assertEqual(loaded[2].confs.n_conformers, 2)
        self.assertTrue(loaded[2].confs.has_weights)
        np.testing.assert_array_almost_equal(loaded[2].coords, protein_with_weights_2.coords)
        np.testing.assert_array_almost_equal(loaded[2].confs.weights, protein_with_weights_2.confs.weights)

        loaded.close_hdf5()

    def test_batch_unbatch_different_n_confs_all_weights(self):
        """Test proteins with different numbers of conformers, all with weights."""

        proteins = [
            self._create_protein_with_confs(5, 2, has_weights=True),
            self._create_protein_with_confs(6, 5, has_weights=True),
            self._create_protein_with_confs(4, 1, has_weights=True),
        ]
        batch = ProteinBatch(proteins)

        save_path = Path(self.temp_dir) / "diff_confs_all_weights"
        batch.save(save_path)

        loaded = ProteinBatch.load(save_path)

        for i, (orig, restored) in enumerate(zip(proteins, loaded, strict=True)):
            self.assertEqual(restored.n_atoms, orig.n_atoms, f"Protein {i} n_atoms mismatch")
            self.assertEqual(restored.confs.n_conformers, orig.confs.n_conformers, f"Protein {i} n_confs mismatch")
            self.assertTrue(restored.confs.has_weights, f"Protein {i} should have weights")
            np.testing.assert_array_almost_equal(restored.coords, orig.coords, err_msg=f"Protein {i} coords mismatch")
            np.testing.assert_array_almost_equal(
                restored.confs.weights, orig.confs.weights, err_msg=f"Protein {i} weights mismatch"
            )

        loaded.close_hdf5()

    def test_batch_unbatch_different_n_confs_no_weights(self):
        """Test proteins with different numbers of conformers, none with weights."""

        proteins = [
            self._create_protein_with_confs(5, 4, has_weights=False),
            self._create_protein_with_confs(6, 1, has_weights=False),
            self._create_protein_with_confs(4, 7, has_weights=False),
        ]
        batch = ProteinBatch(proteins)

        save_path = Path(self.temp_dir) / "diff_confs_no_weights"
        batch.save(save_path)

        loaded = ProteinBatch.load(save_path)

        for i, (orig, restored) in enumerate(zip(proteins, loaded, strict=True)):
            self.assertEqual(restored.n_atoms, orig.n_atoms, f"Protein {i} n_atoms mismatch")
            self.assertEqual(restored.confs.n_conformers, orig.confs.n_conformers, f"Protein {i} n_confs mismatch")
            self.assertFalse(restored.confs.has_weights, f"Protein {i} should not have weights")
            np.testing.assert_array_almost_equal(restored.coords, orig.coords, err_msg=f"Protein {i} coords mismatch")

        loaded.close_hdf5()

    def test_batch_unbatch_different_n_confs_mixed_weights(self):
        """Test proteins with different numbers of conformers and mixed weight configs."""

        proteins = [
            self._create_protein_with_confs(5, 2, has_weights=True),
            self._create_protein_with_confs(6, 5, has_weights=False),
            self._create_protein_with_confs(4, 1, has_weights=True),
            self._create_protein_with_confs(7, 3, has_weights=False),
        ]
        batch = ProteinBatch(proteins)

        save_path = Path(self.temp_dir) / "diff_confs_mixed"
        batch.save(save_path)

        loaded = ProteinBatch.load(save_path)

        self.assertEqual(len(loaded), 4)

        expected_weights = [True, False, True, False]
        expected_n_confs = [2, 5, 1, 3]

        for i, (orig, restored) in enumerate(zip(proteins, loaded, strict=True)):
            self.assertEqual(restored.n_atoms, orig.n_atoms, f"Protein {i} n_atoms mismatch")
            self.assertEqual(restored.confs.n_conformers, expected_n_confs[i], f"Protein {i} n_confs mismatch")
            self.assertEqual(restored.confs.has_weights, expected_weights[i], f"Protein {i} has_weights mismatch")
            np.testing.assert_array_almost_equal(restored.coords, orig.coords, err_msg=f"Protein {i} coords mismatch")

            if expected_weights[i]:
                np.testing.assert_array_almost_equal(
                    restored.confs.weights, orig.confs.weights, err_msg=f"Protein {i} weights mismatch"
                )

        loaded.close_hdf5()

    def test_batch_unbatch_preserves_all_atom_annotations(self):
        """Test that all atom annotations are preserved through batch/unbatch."""

        proteins = [
            self._create_protein_with_confs(5, 2, has_weights=True),
            self._create_protein_with_confs(7, 2, has_weights=False),
        ]
        batch = ProteinBatch(proteins)

        save_path = Path(self.temp_dir) / "full_check"
        batch.save(save_path)

        loaded = ProteinBatch.load(save_path)

        for i, (orig, restored) in enumerate(zip(proteins, loaded, strict=True)):
            # Check all atom properties
            np.testing.assert_array_equal(restored.atomics, orig.atomics, err_msg=f"Protein {i} atomics mismatch")
            np.testing.assert_array_equal(restored.charges, orig.charges, err_msg=f"Protein {i} charges mismatch")
            np.testing.assert_array_equal(restored.res_names, orig.res_names, err_msg=f"Protein {i} res_names mismatch")
            np.testing.assert_array_equal(
                restored.atom_names, orig.atom_names, err_msg=f"Protein {i} atom_names mismatch"
            )

            np.testing.assert_array_equal(restored.res_ids, orig.res_ids, err_msg=f"Protein {i} res_ids mismatch")

            # Check bonds
            np.testing.assert_array_equal(
                restored.bond_indices, orig.bond_indices, err_msg=f"Protein {i} bond_indices mismatch"
            )

            np.testing.assert_array_equal(
                restored.bond_types, orig.bond_types, err_msg=f"Protein {i} bond_types mismatch"
            )

            # Check coords
            np.testing.assert_array_almost_equal(restored.coords, orig.coords, err_msg=f"Protein {i} coords mismatch")

        loaded.close_hdf5()


class TestStringLengthValidation(unittest.TestCase):
    def test_res_name_too_long_raises_error(self):
        atomics = np.array([6, 7])
        res_names = np.array(["AAAA", "B"])  # "AAAA" exceeds max of 3
        atom_names = np.array(["CA", "N"])
        res_ids = np.array([1, 1])

        with self.assertRaises(ValueError) as ctx:
            AtomSet(atomics, res_names=res_names, atom_names=atom_names, res_ids=res_ids)

        self.assertIn("res_names", str(ctx.exception))
        self.assertIn("AAAA", str(ctx.exception))

    def test_atom_name_too_long_raises_error(self):
        atomics = np.array([6, 7])
        res_names = np.array(["ALA", "GLY"])
        atom_names = np.array(["ABCDE", "N"])  # "ABCDE" exceeds max of 4
        res_ids = np.array([1, 1])

        with self.assertRaises(ValueError) as ctx:
            AtomSet(atomics, res_names=res_names, atom_names=atom_names, res_ids=res_ids)

        self.assertIn("atom_names", str(ctx.exception))
        self.assertIn("ABCDE", str(ctx.exception))

    def test_valid_string_lengths_accepted(self):
        atomics = np.array([6, 7])
        res_names = np.array(["ALA", "G"])  # 3 and 1 char - ok
        atom_names = np.array(["1HB1", "N"])  # 4 and 1 char - ok
        res_ids = np.array([1, 1])

        # Should not raise
        atoms = AtomSet(atomics, res_names=res_names, atom_names=atom_names, res_ids=res_ids)
        self.assertEqual(len(atoms), 2)
