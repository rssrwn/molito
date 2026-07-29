"""Tests for ComplexBatch's padded array accessors and byte round-trip.

GraphBatch and ProteinBatch already exposed model-ready padded arrays; ComplexBatch
could store and reload complexes but gave no way to get arrays back out of a batch.
"""

import unittest

import numpy as np

from molito.core.atoms import AtomSet
from molito.core.bonds import BondSet
from molito.core.confs import ConfSet
from molito.mol.complex import BindingComplex, ComplexBatch
from molito.mol.graph import GraphMol
from molito.mol.protein import Protein


def _protein(n_atoms: int = 5) -> Protein:
    atoms = AtomSet(
        np.array([6, 7, 8, 6, 7][:n_atoms], dtype=np.uint8),
        charges=np.zeros(n_atoms, dtype=np.int8),
        res_names=np.array(["ALA"] * n_atoms),
        atom_names=np.array(["CA"] * n_atoms),
        res_ids=np.ones(n_atoms, dtype=np.int32),
    )
    bonds = BondSet(np.array([[i, i + 1, 1] for i in range(n_atoms - 1)], dtype=np.int16))
    confs = ConfSet(np.random.rand(1, n_atoms, 3).astype(np.float32))
    return Protein(atoms, bonds, confs)


def _ligand(n_atoms: int = 4) -> GraphMol:
    atoms = AtomSet(np.array([6, 6, 8, 7][:n_atoms], dtype=np.uint8))
    bonds = BondSet(np.array([[i, i + 1, 1] for i in range(n_atoms - 1)], dtype=np.int16))
    confs = ConfSet(np.random.rand(1, n_atoms, 3).astype(np.float32))
    return GraphMol(atoms, bonds, confs=confs)


def _batch() -> ComplexBatch:
    # Deliberately ragged so padding is exercised
    return ComplexBatch(
        [
            BindingComplex(_protein(5), _ligand(4), meta={"system_id": "a"}),
            BindingComplex(_protein(3), _ligand(2), meta={"system_id": "b"}),
        ]
    )


class TestComplexBatchArrays(unittest.TestCase):
    def setUp(self):
        self.batch = _batch()
        self.max_atoms = max(self.batch.lengths)  # 9 and 5 -> 9

    def test_lengths(self):
        self.assertEqual(self.batch.lengths, [9, 5])

    def test_atomics_shape_and_padding(self):
        atomics = self.batch.atomics
        self.assertEqual(atomics.shape, (2, self.max_atoms))
        self.assertTrue((atomics[1, 5:] == 0).all(), "short complex must be zero-padded")

    def test_atomics_ligand_first(self):
        cx = self.batch[0]
        np.testing.assert_array_equal(self.batch.atomics[0, :4], cx.ligand.atomics)
        np.testing.assert_array_equal(self.batch.atomics[0, 4:9], cx.protein.atomics)

    def test_charges_shape(self):
        self.assertEqual(self.batch.charges.shape, (2, self.max_atoms))

    def test_coords_shape_and_order(self):
        coords = self.batch.coords
        self.assertEqual(coords.shape, (2, self.max_atoms, 3))
        np.testing.assert_allclose(coords[0, :9], self.batch[0].coords)

    def test_res_names_marks_ligand_atoms(self):
        res_names = self.batch.res_names
        self.assertEqual(res_names.shape, (2, self.max_atoms))
        self.assertTrue((res_names[0, :4] == "LIG").all())
        self.assertTrue((res_names[0, 4:9] == "ALA").all())

    def test_bond_arrays_shapes_agree(self):
        self.assertEqual(self.batch.bond_indices.shape[0], 2)
        self.assertEqual(self.batch.bond_types.shape[0], 2)
        self.assertEqual(self.batch.bonds.shape[0], 2)
        self.assertEqual(self.batch.bonds.shape[2], 3)

    def test_adjacency_shape_and_symmetry(self):
        adj = self.batch.adjacency
        self.assertEqual(adj.shape, (2, self.max_atoms, self.max_atoms))
        np.testing.assert_array_equal(adj, adj.transpose(0, 2, 1))

    def test_adjacency_matches_per_complex(self):
        # The first complex fills the whole padded block, so it should match exactly
        np.testing.assert_array_equal(self.batch.adjacency[0], self.batch[0].adjacency)

    def test_adjacency_padding_is_empty(self):
        self.assertTrue((self.batch.adjacency[1, 5:] == 0).all())

    def test_masks_agree_with_lengths(self):
        self.assertEqual(self.batch.mask.shape, (2, self.max_atoms))
        self.assertEqual(self.batch.protein_mask.sum(), 8)  # 5 + 3
        self.assertEqual(self.batch.ligand_mask.sum(), 6)  # 4 + 2

    def test_meta_column(self):
        self.assertEqual(self.batch.meta_column("system_id").tolist(), ["a", "b"])

    def test_meta_column_missing_key(self):
        self.assertEqual(self.batch.meta_column("nope").tolist(), ["", ""])

    def test_accessors_match_graphbatch_conventions(self):
        # Same leading batch dim across every accessor, as for GraphBatch/ProteinBatch
        for name in ("atomics", "charges", "coords", "res_names", "bond_indices", "bond_types", "adjacency", "mask"):
            self.assertEqual(getattr(self.batch, name).shape[0], 2, f"{name} has the wrong batch dim")


class TestComplexBatchMultiConformerLigand(unittest.TestCase):
    """coords has no single geometry to return if a ligand has an ensemble."""

    def test_multi_conformer_ligand_raises(self):
        atoms = AtomSet(np.array([6, 6], dtype=np.uint8))
        bonds = BondSet(np.array([[0, 1, 1]], dtype=np.int16))
        ligand = GraphMol(atoms, bonds, confs=ConfSet(np.random.rand(3, 2, 3).astype(np.float32)))

        batch = ComplexBatch([BindingComplex(_protein(3), ligand)])

        with self.assertRaises(ValueError):
            _ = batch.coords


class TestComplexBatchBytes(unittest.TestCase):
    def test_bytes_roundtrip(self):
        batch = _batch()
        restored = ComplexBatch.from_bytes(batch.to_bytes())

        self.assertIsInstance(restored, ComplexBatch)
        self.assertEqual(len(restored), len(batch))
        self.assertEqual(restored.lengths, batch.lengths)

    def test_bytes_roundtrip_preserves_arrays(self):
        batch = _batch()
        restored = ComplexBatch.from_bytes(batch.to_bytes())

        np.testing.assert_array_equal(restored.atomics, batch.atomics)
        np.testing.assert_allclose(restored.coords, batch.coords)

    def test_bytes_roundtrip_preserves_meta(self):
        batch = _batch()
        restored = ComplexBatch.from_bytes(batch.to_bytes())
        self.assertEqual(restored.meta_column("system_id").tolist(), ["a", "b"])
