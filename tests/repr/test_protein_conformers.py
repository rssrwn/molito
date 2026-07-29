"""Tests for multi-conformer proteins.

ConfSet always stored [n_confs, n_atoms, 3] and the HDF5 layout already carried the
conformer count, so storage needed no change - only the Protein accessors, which
assumed exactly one structure.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from molito.core.atoms import AtomSet
from molito.core.bonds import BondSet
from molito.core.confs import ConfSet
from molito.mol.protein import Protein, ProteinBatch

N_ATOMS = 5


def _atoms(n_atoms: int = N_ATOMS) -> AtomSet:
    return AtomSet(
        np.array([6, 7, 8, 6, 7][:n_atoms], dtype=np.uint8),
        res_names=np.array(["ALA"] * n_atoms),
        atom_names=np.array(["CA"] * n_atoms),
        res_ids=np.ones(n_atoms, dtype=np.int32),
    )


def _bonds(n_atoms: int = N_ATOMS) -> BondSet:
    return BondSet(np.array([[i, i + 1, 1] for i in range(n_atoms - 1)], dtype=np.int16))


def _protein(n_confs: int = 1, n_atoms: int = N_ATOMS, seed: int = 0) -> Protein:
    rng = np.random.default_rng(seed)
    coords = rng.random((n_confs, n_atoms, 3), dtype=np.float32)
    return Protein(_atoms(n_atoms), _bonds(n_atoms), ConfSet(coords))


class TestSingleConformerUnchanged(unittest.TestCase):
    """The common case must behave exactly as before."""

    def setUp(self):
        self.protein = _protein(n_confs=1)

    def test_n_conformers(self):
        self.assertEqual(self.protein.n_conformers, 1)

    def test_coords_is_2d(self):
        self.assertEqual(self.protein.coords.shape, (N_ATOMS, 3))

    def test_all_coords_is_3d(self):
        self.assertEqual(self.protein.all_coords.shape, (1, N_ATOMS, 3))

    def test_coords_matches_all_coords(self):
        np.testing.assert_array_equal(self.protein.coords, self.protein.all_coords[0])


class TestMultipleConformers(unittest.TestCase):
    def setUp(self):
        self.protein = _protein(n_confs=4)

    def test_n_conformers(self):
        self.assertEqual(self.protein.n_conformers, 4)

    def test_all_coords_shape(self):
        self.assertEqual(self.protein.all_coords.shape, (4, N_ATOMS, 3))

    def test_coords_returns_the_first_conformer(self):
        # Documented behaviour, not an accident: it is what allows ProteinBatch to batch
        # proteins whose conformer counts differ.
        np.testing.assert_array_equal(self.protein.coords, self.protein.get_conformer(0))
        self.assertEqual(self.protein.coords.shape, (N_ATOMS, 3))

    def test_get_conformer(self):
        for idx in range(4):
            conf = self.protein.get_conformer(idx)
            self.assertEqual(conf.shape, (N_ATOMS, 3))
            np.testing.assert_array_equal(conf, self.protein.all_coords[idx])

    def test_conformers_are_distinct(self):
        self.assertFalse(np.allclose(self.protein.get_conformer(0), self.protein.get_conformer(1)))

    def test_protein_with_conformer(self):
        single = self.protein.protein_with_conformer(2)

        self.assertEqual(single.n_conformers, 1)
        self.assertEqual(single.coords.shape, (N_ATOMS, 3))
        np.testing.assert_array_equal(single.coords, self.protein.get_conformer(2))

    def test_protein_with_conformer_keeps_annotations(self):
        single = self.protein.protein_with_conformer(1)
        np.testing.assert_array_equal(single.atoms.res_names, self.protein.atoms.res_names)
        np.testing.assert_array_equal(single.atomics, self.protein.atomics)
        self.assertEqual(single.n_bonds, self.protein.n_bonds)

    def test_transforms_apply_to_all_conformers(self):
        shift = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        shifted = self.protein.shift(shift)
        np.testing.assert_allclose(shifted.all_coords, self.protein.all_coords + shift, atol=1e-5)

    def test_permute_applies_to_all_conformers(self):
        order = [4, 3, 2, 1, 0]
        permuted = self.protein.permute(order)

        self.assertEqual(permuted.n_conformers, 4)
        np.testing.assert_array_equal(permuted.all_coords, self.protein.all_coords[:, order, :])


class TestMultiConformerPersistence(unittest.TestCase):
    """Storage already carried the conformer count -- confirm it survives a round trip."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_hdf5_roundtrip_keeps_every_conformer(self):
        protein = _protein(n_confs=3, seed=1)
        ProteinBatch([protein]).save(self.tmp / "ds")

        loaded = ProteinBatch.load(self.tmp / "ds")
        self.assertEqual(loaded[0].n_conformers, 3)
        np.testing.assert_allclose(loaded[0].all_coords, protein.all_coords, atol=1e-6)
        loaded.close_hdf5()

    def test_bytes_roundtrip_keeps_every_conformer(self):
        protein = _protein(n_confs=3, seed=2)
        restored = Protein.from_bytes(protein.to_bytes())

        self.assertEqual(restored.n_conformers, 3)
        np.testing.assert_allclose(restored.all_coords, protein.all_coords, atol=1e-6)

    def test_mixed_conformer_counts_roundtrip(self):
        proteins = [_protein(n_confs=n, seed=n) for n in (1, 3, 2)]
        ProteinBatch(proteins).save(self.tmp / "mixed")

        loaded = ProteinBatch.load(self.tmp / "mixed")
        self.assertEqual([p.n_conformers for p in loaded], [1, 3, 2])

        for before, after in zip(proteins, loaded, strict=True):
            np.testing.assert_allclose(after.all_coords, before.all_coords, atol=1e-6)

        loaded.close_hdf5()


class TestProteinBatchCoords(unittest.TestCase):
    def test_batch_coords_for_single_conformer_proteins(self):
        batch = ProteinBatch([_protein(n_confs=1, n_atoms=5), _protein(n_confs=1, n_atoms=3)])
        self.assertEqual(batch.coords.shape, (2, 5, 3))

    def test_batch_coords_works_across_differing_conformer_counts(self):
        # Relies on Protein.coords being the first conformer -- see its docstring
        batch = ProteinBatch([_protein(n_confs=1, seed=3), _protein(n_confs=3, seed=4)])

        coords = batch.coords
        self.assertEqual(coords.shape, (2, N_ATOMS, 3))
        np.testing.assert_array_equal(coords[1], batch[1].get_conformer(0))
