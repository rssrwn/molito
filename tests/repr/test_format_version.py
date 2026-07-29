"""Tests for the on-disk format version stamp.

Covers that shards are stamped on write, that pre-versioning shards still load, and
that a shard from a newer molito is refused rather than misread.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
from rdkit import Chem

from molito.core.atoms import AtomSet
from molito.core.bonds import BondSet
from molito.core.confs import ConfSet
from molito.core.format import (
    FORMAT_VERSION,
    FORMAT_VERSION_ATTR,
    LEGACY_FORMAT_VERSION,
    PACKAGE_VERSION_ATTR,
    check_format,
)
from molito.mol.complex import BindingComplex, ComplexBatch
from molito.mol.graph import GraphBatch, GraphMol
from molito.mol.protein import Protein, ProteinBatch


def _mols(n: int = 3) -> list[GraphMol]:
    smis = ["CCO", "c1ccccc1", "C/C=C/C"]
    return [GraphMol.from_rdkit(Chem.MolFromSmiles(smis[i % len(smis)])) for i in range(n)]


def _protein(n_atoms: int = 5) -> Protein:
    atoms = AtomSet(
        np.array([6, 7, 8, 6, 7][:n_atoms], dtype=np.uint8),
        res_names=np.array(["ALA"] * n_atoms),
        atom_names=np.array(["CA"] * n_atoms),
        res_ids=np.array([1] * n_atoms, dtype=np.int32),
    )
    bonds = BondSet(np.array([[i, i + 1, 1] for i in range(n_atoms - 1)], dtype=np.int16))
    confs = ConfSet(np.random.rand(1, n_atoms, 3).astype(np.float32))
    return Protein(atoms, bonds, confs)


def _strip_stamp(shard_dir: Path) -> None:
    """Remove the version attributes, reproducing a shard written before stamping."""

    for shard in shard_dir.glob("*.hdf5"):
        with h5py.File(shard, "r+") as f:
            del f.attrs[FORMAT_VERSION_ATTR]
            del f.attrs[PACKAGE_VERSION_ATTR]


def _set_version(shard_dir: Path, version: int) -> None:
    for shard in shard_dir.glob("*.hdf5"):
        with h5py.File(shard, "r+") as f:
            f.attrs[FORMAT_VERSION_ATTR] = version


class TestStampWritten(unittest.TestCase):
    """Every batch type stamps the shard root on save."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _assert_stamped(self, path: Path):
        shards = list(path.glob("*.hdf5"))
        self.assertGreater(len(shards), 0)

        for shard in shards:
            with h5py.File(shard, "r") as f:
                self.assertEqual(f.attrs[FORMAT_VERSION_ATTR], FORMAT_VERSION)
                self.assertIn(PACKAGE_VERSION_ATTR, f.attrs)

    def test_graph_batch_stamped(self):
        path = self.tmp / "mols"
        GraphBatch(_mols()).save(path)
        self._assert_stamped(path)

    def test_graph_batch_stamped_every_shard(self):
        path = self.tmp / "sharded"
        GraphBatch(_mols(6)).save(path, shard_size=2)
        self.assertEqual(len(list(path.glob("*.hdf5"))), 3)
        self._assert_stamped(path)

    def test_protein_batch_stamped(self):
        path = self.tmp / "proteins"
        ProteinBatch([_protein(), _protein(4)]).save(path)
        self._assert_stamped(path)

    def test_complex_batch_stamped(self):
        path = self.tmp / "complexes"
        complexes = [BindingComplex(_protein(), mol) for mol in _mols(2)]
        ComplexBatch(complexes).save(path)
        self._assert_stamped(path)


class TestRoundTripWithStamp(unittest.TestCase):
    """Stamping must not disturb the data it sits alongside."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_graph_roundtrip(self):
        mols = _mols()
        path = self.tmp / "mols"
        GraphBatch(mols).save(path)

        loaded = GraphBatch.load(path)
        self.assertEqual(len(loaded), len(mols))

        for original, new in zip(mols, loaded, strict=True):
            np.testing.assert_array_equal(original.atomics, new.atomics)
            np.testing.assert_array_equal(original.adjacency, new.adjacency)

        loaded.close_hdf5()

    def test_lazy_roundtrip(self):
        path = self.tmp / "lazy"
        GraphBatch(_mols()).save(path)

        lazy = GraphBatch.load(path, materialise=False)
        self.assertEqual(len(lazy), 3)
        lazy.close_hdf5()


class TestLegacyShards(unittest.TestCase):
    """Shards written before the stamp existed are still readable."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.mols = _mols()
        self.path = self.tmp / "legacy"
        GraphBatch(self.mols).save(self.path)
        _strip_stamp(self.path)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_legacy_reports_version_zero(self):
        with h5py.File(next(self.path.glob("*.hdf5")), "r") as f:
            self.assertEqual(check_format(f), LEGACY_FORMAT_VERSION)

    def test_legacy_loads_eagerly(self):
        loaded = GraphBatch.load(self.path)
        self.assertEqual(len(loaded), len(self.mols))
        np.testing.assert_array_equal(loaded[0].atomics, self.mols[0].atomics)
        loaded.close_hdf5()

    def test_legacy_loads_lazily(self):
        lazy = GraphBatch.load(self.path, materialise=False)
        self.assertEqual(len(lazy), len(self.mols))
        np.testing.assert_array_equal(lazy[0].atomics, self.mols[0].atomics)
        lazy.close_hdf5()


class TestFutureShardsRefused(unittest.TestCase):
    """A shard from a newer molito is refused rather than silently misread."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _future_graph_shards(self, name: str) -> Path:
        path = self.tmp / name
        GraphBatch(_mols()).save(path)
        _set_version(path, FORMAT_VERSION + 1)
        return path

    def test_eager_load_raises(self):
        path = self._future_graph_shards("future")

        with self.assertRaises(RuntimeError) as ctx:
            GraphBatch.load(path)

        self.assertIn(str(FORMAT_VERSION + 1), str(ctx.exception))
        self.assertIn("upgrade", str(ctx.exception).lower())

    def test_lazy_load_raises(self):
        path = self._future_graph_shards("future_lazy")

        with self.assertRaises(RuntimeError):
            GraphBatch.load(path, materialise=False)

    def test_protein_load_raises(self):
        path = self.tmp / "future_proteins"
        ProteinBatch([_protein()]).save(path)
        _set_version(path, FORMAT_VERSION + 1)

        with self.assertRaises(RuntimeError):
            ProteinBatch.load(path)

    def test_complex_load_raises(self):
        path = self.tmp / "future_complexes"
        ComplexBatch([BindingComplex(_protein(), _mols(1)[0])]).save(path)
        _set_version(path, FORMAT_VERSION + 1)

        with self.assertRaises(RuntimeError):
            ComplexBatch.load(path)

    def test_current_version_accepted(self):
        path = self.tmp / "current"
        GraphBatch(_mols()).save(path)
        _set_version(path, FORMAT_VERSION)

        loaded = GraphBatch.load(path)
        self.assertEqual(len(loaded), 3)
        loaded.close_hdf5()
