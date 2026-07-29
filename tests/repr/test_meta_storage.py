"""Tests for the columnar + blob meta storage paths added to GraphBatch / ProteinBatch /
ComplexBatch. Covers:

- Round-trip equivalence between blob and columnar formats.
- Missing keys in columnar mode fall back to empty string values.
- `__slots__` restricts attribute assignment on hot classes.
- The fast-path load produces mols equivalent to an eager (user-space) reconstruction.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from rdkit import Chem

from molito.core.atoms import AtomSet
from molito.core.bonds import BondSet
from molito.core.confs import ConfSet
from molito.mol.graph import GraphBatch, GraphMol


def _build_graphmol(smi: str, meta: dict) -> GraphMol:
    mol = GraphMol.from_rdkit(Chem.MolFromSmiles(smi))
    mol.meta = meta
    return mol


class TestMetaRoundtrip(unittest.TestCase):
    """Both save formats round-trip the same data."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.mols = [
            _build_graphmol("CCO", {"name": "ethanol", "pIC50": "7.1", "source": "chembl"}),
            _build_graphmol("c1ccccc1", {"name": "benzene", "pIC50": "6.3", "source": "chembl"}),
            _build_graphmol("CC(=O)O", {"name": "acetic_acid", "pIC50": "8.5", "source": "bindingdb"}),
        ]

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _assert_roundtrip(self, columnar: bool):
        save_path = self.tmp / ("col" if columnar else "blob")
        GraphBatch(self.mols).save(save_path, columnar_meta=columnar)

        loaded = GraphBatch.load(save_path)
        self.assertEqual(len(loaded), 3)
        for i, mol in enumerate(loaded):
            expected = self.mols[i].meta
            got = dict(mol.meta)
            self.assertEqual(got, expected, f"mol {i}: {got} != {expected}")

        loaded.close_hdf5()

    def test_blob_format(self):
        self._assert_roundtrip(columnar=False)

    def test_columnar_format(self):
        self._assert_roundtrip(columnar=True)


class TestColumnarMissingKeys(unittest.TestCase):
    """Mols missing a key in columnar mode load back as empty string for that key."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_missing_keys_become_empty_strings(self):
        mols = [
            _build_graphmol("CCO", {"a": "1", "b": "x"}),
            _build_graphmol("CC(=O)O", {"a": "2"}),  # missing 'b'
            _build_graphmol("c1ccccc1", {"a": "3", "b": "y", "c": "z"}),  # extra 'c'
        ]

        save_path = self.tmp / "out"
        GraphBatch(mols).save(save_path, columnar_meta=True)

        loaded = GraphBatch.load(save_path)
        self.assertEqual(loaded[0].meta["a"], "1")
        self.assertEqual(loaded[0].meta["b"], "x")
        self.assertEqual(loaded[0].meta["c"], "")

        self.assertEqual(loaded[1].meta["a"], "2")
        self.assertEqual(loaded[1].meta["b"], "")
        self.assertEqual(loaded[1].meta["c"], "")

        self.assertEqual(loaded[2].meta["a"], "3")
        self.assertEqual(loaded[2].meta["b"], "y")
        self.assertEqual(loaded[2].meta["c"], "z")

        loaded.close_hdf5()

    def test_columnar_view_is_read_only(self):
        # Meta on a loaded mol is a read-only view. Call dict(mol.meta) for a mutable
        # copy, or reassign mol.meta = {...} to replace it outright.
        mols = [_build_graphmol("CCO", {"a": "1"})]
        save_path = self.tmp / "ro"
        GraphBatch(mols).save(save_path, columnar_meta=True)

        loaded = GraphBatch.load(save_path)
        self.assertEqual(loaded[0].meta["a"], "1")
        self.assertIn("a", loaded[0].meta)

        with self.assertRaises(TypeError):
            loaded[0].meta["a"] = "new"

        # Escape hatch works: dict(...) returns a mutable copy
        d = dict(loaded[0].meta)
        d["a"] = "new"
        self.assertEqual(d["a"], "new")

        loaded.close_hdf5()


class TestNonAsciiMeta(unittest.TestCase):
    """Non-ASCII characters (µ, Å, ...) round-trip through both save formats."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.mols = [
            _build_graphmol("CCO", {"units": "\u00b5M", "name": "\u00e5ngstr\u00f6m"}),
            _build_graphmol("c1ccccc1", {"units": "nM", "name": "benzene"}),
        ]

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _assert_roundtrip(self, columnar: bool):
        save_path = self.tmp / ("col" if columnar else "blob")
        GraphBatch(self.mols).save(save_path, columnar_meta=columnar)

        loaded = GraphBatch.load(save_path)
        self.assertEqual(loaded[0].meta["units"], "\u00b5M")
        self.assertEqual(loaded[0].meta["name"], "\u00e5ngstr\u00f6m")
        self.assertEqual(loaded[1].meta["units"], "nM")
        loaded.close_hdf5()

    def test_non_ascii_blob(self):
        self._assert_roundtrip(columnar=False)

    def test_non_ascii_columnar(self):
        self._assert_roundtrip(columnar=True)

    def test_non_ascii_column_array(self):
        # column_array batch-reads a whole column; must also decode UTF-8 correctly.
        import h5py

        from molito.core.meta import column_array

        save_path = self.tmp / "cb"
        GraphBatch(self.mols).save(save_path, columnar_meta=True)

        with h5py.File(save_path / "0.hdf5", "r") as f:
            arr = column_array(f["meta"], "units")

        self.assertEqual(list(arr), ["\u00b5M", "nM"])


class TestEmptyMetas(unittest.TestCase):
    """Mols with empty / None metas survive both save formats."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.mols = [
            _build_graphmol("CCO", {}),
            _build_graphmol("c1ccccc1", {}),
        ]

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_empty_metas_blob(self):
        save_path = self.tmp / "b"
        GraphBatch(self.mols).save(save_path, columnar_meta=False)
        loaded = GraphBatch.load(save_path)
        self.assertEqual(dict(loaded[0].meta), {})
        loaded.close_hdf5()

    def test_empty_metas_columnar(self):
        save_path = self.tmp / "c"
        GraphBatch(self.mols).save(save_path, columnar_meta=True)
        loaded = GraphBatch.load(save_path)
        self.assertEqual(dict(loaded[0].meta), {})
        loaded.close_hdf5()


class TestSlotsEnforcement(unittest.TestCase):
    """`__slots__` should forbid setting undeclared attributes on the hot classes."""

    def test_graphmol_rejects_unknown_attr(self):
        mol = GraphMol.from_rdkit(Chem.MolFromSmiles("CCO"))
        with self.assertRaises(AttributeError):
            mol.some_new_attr = 42

    def test_atomset_rejects_unknown_attr(self):
        atoms = AtomSet(np.array([6, 6, 8], dtype=np.uint8))
        with self.assertRaises(AttributeError):
            atoms.foo = 1

    def test_bondset_rejects_unknown_attr(self):
        bonds = BondSet(np.zeros((0, 3), dtype=np.int16))
        with self.assertRaises(AttributeError):
            bonds.bar = 2

    def test_confset_rejects_unknown_attr(self):
        confs = ConfSet(np.zeros((1, 3, 3), dtype=np.float32))
        with self.assertRaises(AttributeError):
            confs.baz = 3


class TestLoadUncheckedEquivalence(unittest.TestCase):
    """The fast-path HDF5 load produces mols that behave like ones built via the public API."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write_and_reload(self, columnar: bool):
        mol = GraphMol.from_rdkit(Chem.MolFromSmiles("N[C@@H](C)C(=O)O"))
        mol.meta = {"key": "value", "n": "3"}
        save_path = self.tmp / ("c" if columnar else "b")
        GraphBatch([mol]).save(save_path, columnar_meta=columnar)

        loaded = GraphBatch.load(save_path)[0]

        # Data round-trips faithfully
        self.assertTrue(np.array_equal(loaded.atomics, mol.atomics))
        self.assertTrue(np.array_equal(loaded.bond_indices, mol.bond_indices))
        self.assertTrue(np.array_equal(loaded.bond_types, mol.bond_types))
        self.assertTrue(np.array_equal(loaded.charges, mol.charges))

        # Meta roundtrips
        self.assertEqual(dict(loaded.meta), mol.meta)

        # to_rdkit still works (tests that the fast-loaded mol is fully functional)
        recovered = loaded.to_rdkit()
        self.assertIsNotNone(recovered)

    def test_fast_load_equivalence_blob(self):
        self._write_and_reload(columnar=False)

    def test_fast_load_equivalence_columnar(self):
        self._write_and_reload(columnar=True)


class TestColumnarTypePreservation(unittest.TestCase):
    """Columnar format auto-promotes numeric columns to int64/float64 so callers don't
    have to manually cast on every read. Mixed or sparse columns fall back to S-dtype.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _save_and_load(self, mols):
        save_path = self.tmp / "out"
        GraphBatch(mols).save(save_path, columnar_meta=True)
        return GraphBatch.load(save_path)

    def test_int_column_roundtrips_as_int(self):
        mols = [
            _build_graphmol("CCO", {"n_heavy_atoms": 3}),
            _build_graphmol("c1ccccc1", {"n_heavy_atoms": 6}),
            _build_graphmol("CC(=O)O", {"n_heavy_atoms": 4}),
        ]
        loaded = self._save_and_load(mols)

        for i, expected in enumerate([3, 6, 4]):
            v = loaded[i].meta["n_heavy_atoms"]
            self.assertIsInstance(v, int)
            self.assertEqual(v, expected)

        loaded.close_hdf5()

    def test_float_column_roundtrips_as_float(self):
        mols = [
            _build_graphmol("CCO", {"pIC50": 7.1}),
            _build_graphmol("c1ccccc1", {"pIC50": 6.3}),
        ]
        loaded = self._save_and_load(mols)

        v0 = loaded[0].meta["pIC50"]
        self.assertIsInstance(v0, float)
        self.assertAlmostEqual(v0, 7.1)
        self.assertAlmostEqual(loaded[1].meta["pIC50"], 6.3)

        loaded.close_hdf5()

    def test_mixed_int_float_promotes_to_float(self):
        mols = [
            _build_graphmol("CCO", {"x": 1}),  # int
            _build_graphmol("c1ccccc1", {"x": 2.5}),  # float
        ]
        loaded = self._save_and_load(mols)

        self.assertIsInstance(loaded[0].meta["x"], float)
        self.assertAlmostEqual(loaded[0].meta["x"], 1.0)
        self.assertAlmostEqual(loaded[1].meta["x"], 2.5)

        loaded.close_hdf5()

    def test_string_values_stay_as_strings(self):
        # Strings (including stringified numbers) are NOT auto-promoted — only native
        # int/float types are.
        mols = [
            _build_graphmol("CCO", {"name": "ethanol", "stringy_num": "42"}),
            _build_graphmol("c1ccccc1", {"name": "benzene", "stringy_num": "84"}),
        ]
        loaded = self._save_and_load(mols)

        self.assertEqual(loaded[0].meta["name"], "ethanol")
        self.assertEqual(loaded[0].meta["stringy_num"], "42")
        self.assertIsInstance(loaded[0].meta["stringy_num"], str)

        loaded.close_hdf5()

    def test_missing_values_force_string_fallback(self):
        # Any missing rows in a column → fall back to S-dtype with empty-string fill,
        # because there's no clean way to represent "missing" in an int dataset.
        mols = [
            _build_graphmol("CCO", {"x": 1}),
            _build_graphmol("c1ccccc1", {}),  # missing x
            _build_graphmol("CC(=O)O", {"x": 3}),
        ]
        loaded = self._save_and_load(mols)

        self.assertEqual(loaded[0].meta["x"], "1")
        self.assertEqual(loaded[1].meta["x"], "")
        self.assertEqual(loaded[2].meta["x"], "3")

        loaded.close_hdf5()

    def test_mixed_numeric_and_string_falls_back_to_string(self):
        mols = [
            _build_graphmol("CCO", {"x": 1}),
            _build_graphmol("c1ccccc1", {"x": "two"}),
        ]
        loaded = self._save_and_load(mols)

        self.assertEqual(loaded[0].meta["x"], "1")
        self.assertEqual(loaded[1].meta["x"], "two")

        loaded.close_hdf5()

    def test_bool_values_fall_back_to_string(self):
        # Bools deliberately don't auto-promote to int so True/False don't silently
        # collapse to 1/0 on disk. They land in the string fallback.
        mols = [
            _build_graphmol("CCO", {"flag": True}),
            _build_graphmol("c1ccccc1", {"flag": False}),
        ]
        loaded = self._save_and_load(mols)

        self.assertEqual(loaded[0].meta["flag"], "True")
        self.assertEqual(loaded[1].meta["flag"], "False")

        loaded.close_hdf5()

    def test_pickle_columnar_loaded_mol(self):
        # to_bytes() pickles the mol's dict repr — previously broke when meta was a
        # live _ColumnMetaView holding h5py datasets.
        import pickle

        mols = [_build_graphmol("CCO", {"n": 3, "name": "ethanol"})]
        loaded = self._save_and_load(mols)

        blob = loaded[0].to_bytes()
        # Roundtripping through pickle should yield a normal dict for meta
        restored = GraphMol.from_bytes(blob)
        self.assertEqual(dict(restored.meta), {"n": 3, "name": "ethanol"})

        # Direct pickle.dumps on the loaded mol's _to_core_repr() output should also work
        pickle.dumps(loaded[0]._to_core_repr())

        loaded.close_hdf5()

    def test_resave_columnar_loaded_batch(self):
        # Saving a freshly-loaded columnar batch (whose mols carry _ColumnMetaView
        # metas) must not trip the blob/columnar save paths.
        mols = [
            _build_graphmol("CCO", {"n": 3, "name": "ethanol"}),
            _build_graphmol("c1ccccc1", {"n": 6, "name": "benzene"}),
        ]
        first_save = self.tmp / "first"
        GraphBatch(mols).save(first_save, columnar_meta=True)

        loaded = GraphBatch.load(first_save)

        # Re-save in both formats — both must succeed and preserve values
        for fmt_columnar in (False, True):
            resave_path = self.tmp / f"resave_{'col' if fmt_columnar else 'blob'}"
            GraphBatch(list(loaded)).save(resave_path, columnar_meta=fmt_columnar)

            reloaded = GraphBatch.load(resave_path)
            self.assertEqual(reloaded[0].meta["n"], 3)
            self.assertEqual(reloaded[1].meta["n"], 6)
            reloaded.close_hdf5()

        loaded.close_hdf5()

    def test_copy_with_on_columnar_loaded_mol(self):
        # Regression: copy_with deepcopies meta, which previously broke when meta was
        # a live _ColumnMetaView backed by h5py.Dataset handles (not picklable).
        mols = [
            _build_graphmol("CCO", {"n": 3, "name": "ethanol"}),
            _build_graphmol("c1ccccc1", {"n": 6, "name": "benzene"}),
        ]
        loaded = self._save_and_load(mols)

        copy = loaded[0].copy()
        self.assertEqual(dict(copy.meta), {"n": 3, "name": "ethanol"})

        # Mutating the copy must not see the live HDF5 columns
        mutable = dict(loaded[1].meta)
        mutable["n"] = 999
        self.assertEqual(loaded[1].meta["n"], 6)  # original view unchanged

        loaded.close_hdf5()

    def test_column_array_returns_numeric_dtype(self):
        import h5py

        from molito.core.meta import column_array

        mols = [
            _build_graphmol("CCO", {"n": 3}),
            _build_graphmol("c1ccccc1", {"n": 6}),
        ]
        save_path = self.tmp / "ca"
        GraphBatch(mols).save(save_path, columnar_meta=True)

        with h5py.File(save_path / "0.hdf5", "r") as f:
            arr = column_array(f["meta"], "n")

        self.assertTrue(np.issubdtype(arr.dtype, np.integer))
        self.assertTrue(np.array_equal(arr, np.array([3, 6])))
