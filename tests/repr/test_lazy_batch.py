"""Tests for LazyGraphBatch: on-demand GraphMol construction, multi-shard handling,
subset materialisation, and the meta_column fast path.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from rdkit import Chem

from molito.core.atoms import AtomSet
from molito.core.bonds import BondSet
from molito.mol.graph import GraphBatch, GraphMol, LazyGraphBatch


def _build(smi: str, meta: dict) -> GraphMol:
    mol = GraphMol.from_rdkit(Chem.MolFromSmiles(smi))
    mol.meta = meta
    return mol


def _sample_mols(n: int = 5) -> list[GraphMol]:
    smis = ["CCO", "c1ccccc1", "CC(=O)O", "N[C@@H](C)C(=O)O", "C/C=C/C"]
    return [
        _build(smis[i % len(smis)], {"id": str(i), "pIC50": str(5.0 + i * 0.1), "source": "test"}) for i in range(n)
    ]


class TestLazyLoadEquivalence(unittest.TestCase):
    """Lazy and eager loads produce mols that compare equal in content."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _compare_mols(self, a: GraphMol, b: GraphMol):
        self.assertTrue(np.array_equal(a.atomics, b.atomics))
        self.assertTrue(np.array_equal(a.charges, b.charges))
        self.assertTrue(np.array_equal(a.bond_indices, b.bond_indices))
        self.assertTrue(np.array_equal(a.bond_types, b.bond_types))
        self.assertEqual(dict(a.meta), dict(b.meta))

    def _assert_equivalent_single_shard(self, columnar: bool):
        mols = _sample_mols(5)
        path = self.tmp / ("col" if columnar else "blob")
        GraphBatch(mols).save(path, columnar_meta=columnar)

        eager = GraphBatch.load(path, materialise=True)
        lazy = GraphBatch.load(path, materialise=False)

        self.assertIsInstance(lazy, LazyGraphBatch)
        self.assertIsInstance(lazy, GraphBatch)  # subclass relationship
        self.assertEqual(len(lazy), len(eager))

        for i in range(len(eager)):
            self._compare_mols(lazy[i], eager[i])

        eager.close_hdf5()
        lazy.close_hdf5()

    def test_equivalence_blob(self):
        self._assert_equivalent_single_shard(columnar=False)

    def test_equivalence_columnar(self):
        self._assert_equivalent_single_shard(columnar=True)

    def test_multi_shard_navigation(self):
        mols = _sample_mols(10)
        path = self.tmp / "multi"
        GraphBatch(mols).save(path, shard_size=3, columnar_meta=True)

        # Sanity: we actually made multiple shards
        self.assertGreater(len(list(path.iterdir())), 1)

        eager = GraphBatch.load(path, materialise=True)
        lazy = GraphBatch.load(path, materialise=False)

        self.assertEqual(len(lazy), 10)
        self.assertEqual(len(eager), 10)

        for i in range(10):
            self._compare_mols(lazy[i], eager[i])

        eager.close_hdf5()
        lazy.close_hdf5()


class TestLazyNoMaterialisation(unittest.TestCase):
    """Verify that LazyGraphBatch doesn't build GraphMols until asked."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        GraphBatch(_sample_mols(5)).save(self.tmp / "s", columnar_meta=True)
        self.lazy = GraphBatch.load(self.tmp / "s", materialise=False)

    def tearDown(self):
        self.lazy.close_hdf5()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_getitem_returns_fresh_instance(self):
        # Read-only semantics: two accesses return different objects
        self.assertIsNot(self.lazy[0], self.lazy[0])

    def test_iteration_works(self):
        mols = list(self.lazy)
        self.assertEqual(len(mols), 5)
        for mol in mols:
            self.assertIsInstance(mol, GraphMol)

    def test_properties_work_via_inheritance(self):
        # These delegate to self._mols iteration and should Just Work
        self.assertEqual(len(self.lazy.lengths), 5)
        self.assertEqual(self.lazy.atomics.shape[0], 5)
        self.assertEqual(self.lazy.bond_indices.shape[0], 5)
        self.assertEqual(self.lazy.adjacency.shape[0], 5)


class TestSubsetMaterialises(unittest.TestCase):
    """subset(idxs) on a lazy batch returns an eager GraphBatch with just those mols."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.mols = _sample_mols(10)
        GraphBatch(self.mols).save(self.tmp / "s", shard_size=4, columnar_meta=True)
        self.lazy = GraphBatch.load(self.tmp / "s", materialise=False)

    def tearDown(self):
        self.lazy.close_hdf5()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_subset_returns_eager_batch(self):
        subset = self.lazy.subset([0, 3, 7])
        self.assertIsInstance(subset, GraphBatch)
        self.assertNotIsInstance(subset, LazyGraphBatch)
        self.assertEqual(len(subset), 3)

    def test_subset_has_correct_mols(self):
        subset = self.lazy.subset([1, 4, 9])
        self.assertEqual(dict(subset[0].meta), {"id": "1", "pIC50": "5.1", "source": "test"})
        self.assertEqual(dict(subset[1].meta), {"id": "4", "pIC50": "5.4", "source": "test"})
        self.assertEqual(dict(subset[2].meta), {"id": "9", "pIC50": "5.9", "source": "test"})


class TestMetaColumn(unittest.TestCase):
    """Fast path for reading a single meta key as an array."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.mols = _sample_mols(8)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _assert_column_matches_mols(self, batch: GraphBatch):
        pic50_col = batch.meta_column("pIC50")
        expected = [m.meta["pIC50"] for m in self.mols]
        self.assertEqual(pic50_col.tolist(), expected)

    def test_meta_column_eager_blob(self):
        path = self.tmp / "e_blob"
        GraphBatch(self.mols).save(path, columnar_meta=False)
        batch = GraphBatch.load(path)
        self._assert_column_matches_mols(batch)
        batch.close_hdf5()

    def test_meta_column_lazy_blob(self):
        path = self.tmp / "l_blob"
        GraphBatch(self.mols).save(path, columnar_meta=False)
        batch = GraphBatch.load(path, materialise=False)
        self._assert_column_matches_mols(batch)
        batch.close_hdf5()

    def test_meta_column_lazy_columnar(self):
        path = self.tmp / "l_col"
        GraphBatch(self.mols).save(path, shard_size=3, columnar_meta=True)
        batch = GraphBatch.load(path, materialise=False)
        self._assert_column_matches_mols(batch)
        batch.close_hdf5()

    def test_meta_column_multi_shard(self):
        path = self.tmp / "multi"
        GraphBatch(self.mols).save(path, shard_size=3, columnar_meta=True)
        batch = GraphBatch.load(path, materialise=False)

        ids = batch.meta_column("id")
        self.assertEqual(ids.tolist(), [str(i) for i in range(8)])
        batch.close_hdf5()


class TestMetaIsImmutable(unittest.TestCase):
    """Meta returned from a loaded mol is read-only (same policy as atomics/coords/etc.,
    which return fresh copies on each property read). Users who want to mutate meta
    should call `dict(mol.meta)` for a mutable copy, or reassign `mol.meta = {...}`.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _assert_meta_rejects_mutation(self, columnar: bool):
        mols = [_build("CCO", {"k": "original"})]
        path = self.tmp / ("col" if columnar else "blob")
        GraphBatch(mols).save(path, columnar_meta=columnar)

        batch = GraphBatch.load(path, materialise=False)
        mol = batch[0]

        with self.assertRaises(TypeError):
            mol.meta["k"] = "mutated"

        with self.assertRaises(TypeError):
            del mol.meta["k"]

        batch.close_hdf5()

    def test_meta_immutable_blob(self):
        self._assert_meta_rejects_mutation(columnar=False)

    def test_meta_immutable_columnar(self):
        self._assert_meta_rejects_mutation(columnar=True)

    def _assert_mutable_copy_works(self, columnar: bool):
        mols = [_build("CCO", {"k": "original"})]
        path = self.tmp / ("col2" if columnar else "blob2")
        GraphBatch(mols).save(path, columnar_meta=columnar)

        batch = GraphBatch.load(path, materialise=False)
        mol = batch[0]

        # Escape hatch 1: dict(mol.meta) gives a mutable copy
        d = dict(mol.meta)
        d["k"] = "local"
        d["new"] = "value"
        self.assertEqual(d["k"], "local")

        # Original meta is untouched
        self.assertEqual(mol.meta["k"], "original")

        # Escape hatch 2: reassign mol.meta
        mol.meta = d
        self.assertEqual(mol.meta["k"], "local")
        self.assertEqual(mol.meta["new"], "value")

        batch.close_hdf5()

    def test_mutable_copy_blob(self):
        self._assert_mutable_copy_works(columnar=False)

    def test_mutable_copy_columnar(self):
        self._assert_mutable_copy_works(columnar=True)


class TestLazyPreservesAtomAnnotations(unittest.TestCase):
    """Every optional AtomSet annotation must survive a lazy load, not just the ones
    the eager loader happens to read. chain_ids was silently dropped by the lazy path.
    """

    ANNOTATIONS = ("chirality", "res_names", "atom_names", "res_ids", "chain_ids")

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

        atoms = AtomSet(
            np.array([6, 6, 8], dtype=np.uint8),
            charges=np.array([0, 0, -1], dtype=np.int8),
            chirality=np.array([0, 1, 0], dtype=np.int8),
            res_names=np.array(["ALA", "ALA", "GLY"]),
            atom_names=np.array(["CA", "CB", "O"]),
            res_ids=np.array([1, 1, 2], dtype=np.int32),
            chain_ids=np.array(["A", "A", "B"]),
        )
        bonds = BondSet(np.array([[0, 1, 1], [1, 2, 1]], dtype=np.int16))

        self.mol = GraphMol(atoms, bonds)
        GraphBatch([self.mol]).save(self.tmp / "s")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_annotations_survive_lazy_load(self):
        lazy = GraphBatch.load(self.tmp / "s", materialise=False)

        for name in self.ANNOTATIONS:
            expected = getattr(self.mol.atoms, name)
            actual = getattr(lazy[0].atoms, name)
            self.assertIsNotNone(actual, f"{name} was dropped by the lazy loader")
            np.testing.assert_array_equal(actual, expected, err_msg=f"{name} mismatch")

        lazy.close_hdf5()

    def test_lazy_matches_eager(self):
        eager = GraphBatch.load(self.tmp / "s", materialise=True)
        lazy = GraphBatch.load(self.tmp / "s", materialise=False)

        for name in self.ANNOTATIONS:
            np.testing.assert_array_equal(
                getattr(lazy[0].atoms, name),
                getattr(eager[0].atoms, name),
                err_msg=f"{name} differs between lazy and eager loads",
            )

        eager.close_hdf5()
        lazy.close_hdf5()

    def test_absent_annotations_stay_none(self):
        # A mol with no annotations shouldn't gain empty ones on the way through
        path = self.tmp / "plain"
        GraphBatch([_build("CCO", {})]).save(path)

        lazy = GraphBatch.load(path, materialise=False)
        atoms = lazy[0].atoms

        self.assertIsNone(atoms.res_names)
        self.assertIsNone(atoms.atom_names)
        self.assertIsNone(atoms.res_ids)
        self.assertIsNone(atoms.chain_ids)

        lazy.close_hdf5()


class TestLazyIndexValidation(unittest.TestCase):
    """Out-of-range access raises IndexError rather than silently returning garbage."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        GraphBatch(_sample_mols(3)).save(self.tmp / "s", columnar_meta=True)
        self.lazy = GraphBatch.load(self.tmp / "s", materialise=False)

    def tearDown(self):
        self.lazy.close_hdf5()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_out_of_range_raises(self):
        with self.assertRaises(IndexError):
            _ = self.lazy[5]

        with self.assertRaises(IndexError):
            _ = self.lazy[-1]  # negative indexing not supported in this impl
