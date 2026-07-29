import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from molito.core.atoms import AtomSet
from molito.core.bonds import BondSet
from molito.core.confs import ConfSet
from molito.mol.complex import BindingComplex, ComplexBatch
from molito.mol.graph import GraphMol
from molito.mol.interactions import Interaction, InteractionSet
from molito.mol.protein import Protein

# *** Helper functions ***


def create_test_protein(n_atoms: int = 5, n_confs: int = 1) -> Protein:
    """Create a simple test protein."""

    # Cycle through element types and other properties for arbitrary n_atoms
    base_atomics = [6, 7, 8, 6, 7]
    base_charges = [0, 0, -1, 0, 1]
    base_res_names = ["ALA", "ALA", "ALA", "GLY", "GLY"]
    base_atom_names = ["CA", "N", "O", "CA", "N"]
    base_res_ids = [1, 1, 1, 2, 2]

    atomics = np.array([base_atomics[i % len(base_atomics)] for i in range(n_atoms)])
    charges = np.array([base_charges[i % len(base_charges)] for i in range(n_atoms)])
    res_names = np.array([base_res_names[i % len(base_res_names)] for i in range(n_atoms)])
    atom_names = np.array([base_atom_names[i % len(base_atom_names)] for i in range(n_atoms)])
    res_ids = np.array([base_res_ids[i % len(base_res_ids)] for i in range(n_atoms)])

    atoms = AtomSet(atomics, charges=charges, res_names=res_names, atom_names=atom_names, res_ids=res_ids)

    empty_bonds = np.array([]).reshape(0, 3).astype(np.int32)
    bonds_arr = np.array([[i, i + 1, 1] for i in range(n_atoms - 1)]) if n_atoms > 1 else empty_bonds
    bonds = BondSet(bonds_arr)

    coords = np.random.rand(n_confs, n_atoms, 3).astype(np.float32) * 10
    confs = ConfSet(coords)

    return Protein(atoms, bonds, confs)


def create_test_ligand(n_atoms: int = 4, n_confs: int = 1) -> GraphMol:
    """Create a simple test ligand."""

    base_atomics = [6, 6, 8, 7]
    base_charges = [0, -1, 0, 1]

    atomics = np.array([base_atomics[i % len(base_atomics)] for i in range(n_atoms)])
    charges = np.array([base_charges[i % len(base_charges)] for i in range(n_atoms)])
    atoms = AtomSet(atomics, charges=charges)

    empty_bonds = np.array([]).reshape(0, 3).astype(np.int32)
    bonds_arr = np.array([[i, i + 1, 1] for i in range(n_atoms - 1)]) if n_atoms > 1 else empty_bonds
    bonds = BondSet(bonds_arr)

    coords = np.random.rand(n_confs, n_atoms, 3).astype(np.float32) * 10
    confs = ConfSet(coords)

    return GraphMol(atoms, bonds, confs=confs)


def create_test_interactions(n_protein: int, n_ligand: int) -> InteractionSet:
    """Create a simple test InteractionSet."""

    interactions = [
        Interaction((0,), (0,), "Hydrophobic"),
        Interaction((1,), (1,), "HBAcceptor"),
    ]
    return InteractionSet(interactions, n_protein, n_ligand)


# **************************
# *** ComplexBatch Tests ***
# **************************


class TestComplexBatchInit(unittest.TestCase):
    """Test ComplexBatch initialization."""

    def test_basic_creation(self):
        complexes = [
            BindingComplex(create_test_protein(), create_test_ligand()),
            BindingComplex(create_test_protein(), create_test_ligand()),
        ]
        batch = ComplexBatch(complexes)

        self.assertEqual(len(batch), 2)

    def test_invalid_type_raises(self):
        with self.assertRaises(TypeError):
            ComplexBatch(["not a complex"])


class TestComplexBatchProperties(unittest.TestCase):
    """Test ComplexBatch properties."""

    def setUp(self):
        self.complexes = [
            BindingComplex(create_test_protein(n_atoms=5), create_test_ligand(n_atoms=4), meta={"id": "cx1"}),
            BindingComplex(create_test_protein(n_atoms=3), create_test_ligand(n_atoms=2), meta={"id": "cx2"}),
        ]
        self.batch = ComplexBatch(self.complexes)

    def test_lengths(self):
        self.assertEqual(self.batch.lengths, [9, 5])

    def test_protein_lengths(self):
        self.assertEqual(self.batch.protein_lengths, [5, 3])

    def test_ligand_lengths(self):
        self.assertEqual(self.batch.ligand_lengths, [4, 2])

    def test_mask_shape(self):
        mask = self.batch.mask
        self.assertEqual(mask.shape, (2, 9))

    def test_protein_mask_shape(self):
        mask = self.batch.protein_mask
        self.assertEqual(mask.shape, (2, 5))

    def test_ligand_mask_shape(self):
        mask = self.batch.ligand_mask
        self.assertEqual(mask.shape, (2, 4))

    def test_getitem(self):
        cx = self.batch[0]
        self.assertEqual(cx.meta["id"], "cx1")

    def test_subset(self):
        subset = self.batch.subset([1])
        self.assertEqual(len(subset), 1)
        self.assertEqual(subset[0].meta["id"], "cx2")


class TestComplexBatchSerialization(unittest.TestCase):
    """Test ComplexBatch save/load."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def _create_test_batch(self, n_complexes: int = 3, with_interactions: bool = False):
        complexes = []
        for i in range(n_complexes):
            protein = create_test_protein(n_atoms=5 + i)
            ligand = create_test_ligand(n_atoms=4 + i)

            interactions = None
            if with_interactions:
                interactions = create_test_interactions(5 + i, 4 + i)

            cx = BindingComplex(protein, ligand, interactions=interactions, meta={"id": f"cx_{i}"})
            complexes.append(cx)

        return ComplexBatch(complexes)

    def test_save_load_basic(self):
        batch = self._create_test_batch(n_complexes=3)
        save_path = Path(self.temp_dir) / "batch"

        batch.save(save_path)
        loaded = ComplexBatch.load(save_path)

        self.assertEqual(len(loaded), 3)

        for i, cx in enumerate(loaded):
            self.assertEqual(cx.meta["id"], f"cx_{i}")
            self.assertEqual(len(cx.protein), 5 + i)
            self.assertEqual(len(cx.ligand), 4 + i)

        loaded.close_hdf5()

    def test_save_load_with_interactions(self):
        batch = self._create_test_batch(n_complexes=2, with_interactions=True)
        save_path = Path(self.temp_dir) / "batch_int"

        batch.save(save_path)
        loaded = ComplexBatch.load(save_path)

        self.assertEqual(len(loaded), 2)

        for cx in loaded:
            self.assertIsNotNone(cx.interactions)
            self.assertEqual(len(cx.interactions), 2)  # create_test_interactions creates 2

        loaded.close_hdf5()

    def test_save_load_mixed_interactions(self):
        """Test batch where some complexes have interactions, others don't."""

        protein1 = create_test_protein(n_atoms=5)
        ligand1 = create_test_ligand(n_atoms=4)
        interactions1 = create_test_interactions(5, 4)
        cx1 = BindingComplex(protein1, ligand1, interactions=interactions1)

        protein2 = create_test_protein(n_atoms=6)
        ligand2 = create_test_ligand(n_atoms=3)
        cx2 = BindingComplex(protein2, ligand2)  # No interactions

        batch = ComplexBatch([cx1, cx2])
        save_path = Path(self.temp_dir) / "batch_mixed"

        batch.save(save_path)
        loaded = ComplexBatch.load(save_path)

        self.assertEqual(len(loaded), 2)
        self.assertIsNotNone(loaded[0].interactions)
        self.assertIsNone(loaded[1].interactions)

        loaded.close_hdf5()

    def test_save_load_multiple_shards(self):
        batch = self._create_test_batch(n_complexes=5)
        save_path = Path(self.temp_dir) / "shards"

        batch.save(save_path, shard_size=2)
        loaded = ComplexBatch.load(save_path)

        self.assertEqual(len(loaded), 5)

        # Check that shards were created
        shard_files = list(save_path.glob("*.hdf5"))
        self.assertEqual(len(shard_files), 3)  # 5 complexes / 2 per shard = 3 shards

        loaded.close_hdf5()

    def test_save_load_preserves_protein_data(self):
        batch = self._create_test_batch(n_complexes=2)
        save_path = Path(self.temp_dir) / "protein_data"

        batch.save(save_path)
        loaded = ComplexBatch.load(save_path)

        for orig, rest in zip(batch, loaded, strict=True):
            np.testing.assert_array_equal(rest.protein.atomics, orig.protein.atomics)
            np.testing.assert_array_equal(rest.protein.charges, orig.protein.charges)
            np.testing.assert_array_equal(rest.protein.res_names, orig.protein.res_names)
            np.testing.assert_array_almost_equal(rest.protein.coords, orig.protein.coords)

        loaded.close_hdf5()

    def test_save_load_preserves_ligand_data(self):
        batch = self._create_test_batch(n_complexes=2)
        save_path = Path(self.temp_dir) / "ligand_data"

        batch.save(save_path)
        loaded = ComplexBatch.load(save_path)

        for orig, rest in zip(batch, loaded, strict=True):
            np.testing.assert_array_equal(rest.ligand.atomics, orig.ligand.atomics)
            np.testing.assert_array_equal(rest.ligand.charges, orig.ligand.charges)
            np.testing.assert_array_almost_equal(rest.ligand.coords, orig.ligand.coords)

        loaded.close_hdf5()

    def test_save_to_existing_nonempty_dir_raises(self):
        save_path = Path(self.temp_dir) / "nonempty"
        save_path.mkdir()
        (save_path / "existing.txt").touch()

        batch = self._create_test_batch(n_complexes=1)

        with self.assertRaises(RuntimeError):
            batch.save(save_path)

    def test_load_nonexistent_path_raises(self):
        with self.assertRaises(RuntimeError):
            ComplexBatch.load("/nonexistent/path")


class TestComplexBatchFromBatches(unittest.TestCase):
    """Test ComplexBatch.from_batches."""

    def test_merge_batches(self):
        complexes1 = [
            BindingComplex(create_test_protein(), create_test_ligand(), meta={"id": "a"}),
        ]
        complexes2 = [
            BindingComplex(create_test_protein(), create_test_ligand(), meta={"id": "b"}),
            BindingComplex(create_test_protein(), create_test_ligand(), meta={"id": "c"}),
        ]

        batch1 = ComplexBatch(complexes1)
        batch2 = ComplexBatch(complexes2)

        merged = ComplexBatch.from_batches([batch1, batch2])

        self.assertEqual(len(merged), 3)
        self.assertEqual(merged[0].meta["id"], "a")
        self.assertEqual(merged[1].meta["id"], "b")
        self.assertEqual(merged[2].meta["id"], "c")
