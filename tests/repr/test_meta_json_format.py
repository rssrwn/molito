"""Tests for the JSON meta format and the pickle gate on legacy blob shards.

A non-columnar save used to write a pickled blob, which meant loading a shared HDF5 file
executed arbitrary code. It now writes JSON, and reading a legacy blob shard requires
allow_pickle=True.
"""

import json
import pickle
import shutil
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
from rdkit import Chem

from molito.core._checks import PICKLE_PROTOCOL
from molito.core.meta import (
    BLOB_FORMAT,
    COLUMNAR_FORMAT,
    FORMAT_ATTR,
    JSON_FORMAT,
    load_meta,
)
from molito.mol.graph import GraphBatch, GraphMol


def _mols(metas: list[dict]) -> list[GraphMol]:
    smis = ["CCO", "c1ccccc1", "CC(=O)O", "C/C=C/C"]
    mols = []

    for idx, meta in enumerate(metas):
        mol = GraphMol.from_rdkit(Chem.MolFromSmiles(smis[idx % len(smis)]))
        mol.meta = meta
        mols.append(mol)

    return mols


def _rewrite_meta_as_blob(shard_dir: Path) -> None:
    """Replace each shard's meta group with the legacy pickled blob, as older molito wrote it."""

    for shard in shard_dir.glob("*.hdf5"):
        with h5py.File(shard, "r") as f:
            n = len(f["atoms"]["sizes"])
            metas = [dict(m) for m in load_meta(f["meta"], n)]

        with h5py.File(shard, "r+") as f:
            del f["meta"]
            group = f.create_group("meta", track_order=True)
            group.attrs[FORMAT_ATTR] = BLOB_FORMAT
            group.attrs["metas"] = np.void(pickle.dumps(metas, protocol=PICKLE_PROTOCOL))


class TestJsonIsTheNonColumnarFormat(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.metas = [{"id": "a", "score": 1.5}, {"id": "b", "score": 2.5}]
        GraphBatch(_mols(self.metas)).save(self.tmp / "ds", columnar_meta=False)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_format_marker_is_json(self):
        with h5py.File(self.tmp / "ds" / "0.hdf5", "r") as f:
            self.assertEqual(f["meta"].attrs[FORMAT_ATTR], JSON_FORMAT)

    def test_no_pickle_anywhere_in_the_shard(self):
        # The pickle opcode for a protocol-4 stream; a smoke check that nothing pickled slipped in
        raw = (self.tmp / "ds" / "0.hdf5").read_bytes()
        self.assertNotIn(b"\x80\x04", raw)

    def test_roundtrip(self):
        loaded = GraphBatch.load(self.tmp / "ds")
        self.assertEqual([dict(m.meta) for m in loaded], self.metas)
        loaded.close_hdf5()

    def test_loads_without_allow_pickle(self):
        loaded = GraphBatch.load(self.tmp / "ds", allow_pickle=False)
        self.assertEqual(dict(loaded[0].meta), self.metas[0])
        loaded.close_hdf5()

    def test_deferred_load_works(self):
        loaded = GraphBatch.load(self.tmp / "ds", materialise=False)
        self.assertEqual(dict(loaded[0].meta), self.metas[0])
        loaded.close_hdf5()

    def test_meta_is_read_only(self):
        loaded = GraphBatch.load(self.tmp / "ds")

        with self.assertRaises(TypeError):
            loaded[0].meta["id"] = "mutated"

        loaded.close_hdf5()


class TestJsonHandlesWhatColumnarCannot(unittest.TestCase):
    """The reason JSON exists: nested and ragged metadata."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _roundtrip(self, metas: list[dict]) -> list[dict]:
        path = self.tmp / f"ds{len(list(self.tmp.iterdir()))}"
        GraphBatch(_mols(metas)).save(path, columnar_meta=False)

        loaded = GraphBatch.load(path)
        got = [dict(m.meta) for m in loaded]
        loaded.close_hdf5()
        return got

    def test_nested_dict(self):
        metas = [{"metrics": {"rmsd": 1.2, "ok": True}}, {"metrics": {"rmsd": 0.4, "ok": False}}]
        self.assertEqual(self._roundtrip(metas), metas)

    def test_list_values(self):
        metas = [{"residues": [11, 12, 13]}, {"residues": [1]}]
        self.assertEqual(self._roundtrip(metas), metas)

    def test_differing_key_sets(self):
        metas = [{"a": 1}, {"b": "two", "c": 3.0}]
        self.assertEqual(self._roundtrip(metas), metas)

    def test_none_and_bool_survive(self):
        metas = [{"x": None, "flag": True}, {"x": None, "flag": False}]
        self.assertEqual(self._roundtrip(metas), metas)

    def test_empty_metas(self):
        self.assertEqual(self._roundtrip([{}, {}]), [{}, {}])

    def test_numpy_arrays_become_lists(self):
        # Documented lossiness: JSON has no array type, so dtype is not preserved
        metas = [{"seq": np.array(["ALA", "GLY"])}, {"seq": np.array(["SER"])}]
        got = self._roundtrip(metas)

        self.assertEqual(got, [{"seq": ["ALA", "GLY"]}, {"seq": ["SER"]}])

    def test_numpy_scalars_become_python_scalars(self):
        metas = [{"n": np.int64(7)}, {"n": np.float32(1.5)}]
        got = self._roundtrip(metas)

        self.assertEqual(got[0]["n"], 7)
        self.assertIsInstance(got[0]["n"], int)
        self.assertAlmostEqual(got[1]["n"], 1.5, places=5)

    def test_unencodable_value_raises_clearly(self):
        with self.assertRaises(TypeError) as ctx:
            self._roundtrip([{"obj": object()}, {"obj": object()}])

        self.assertIn("not JSON serialisable", str(ctx.exception))


class TestLegacyBlobRequiresOptIn(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.metas = [{"id": "a", "n": 1}, {"id": "b", "n": 2}]
        self.path = self.tmp / "legacy"

        GraphBatch(_mols(self.metas)).save(self.path, columnar_meta=False)
        _rewrite_meta_as_blob(self.path)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_default_load_refuses(self):
        with self.assertRaises(ValueError) as ctx:
            GraphBatch.load(self.path)

        message = str(ctx.exception)
        self.assertIn("allow_pickle=True", message)
        self.assertIn("pickled", message)

    def test_error_names_the_file(self):
        with self.assertRaises(ValueError) as ctx:
            GraphBatch.load(self.path)

        self.assertIn(".hdf5", str(ctx.exception))

    def test_allow_pickle_loads_it(self):
        loaded = GraphBatch.load(self.path, allow_pickle=True)
        self.assertEqual([dict(m.meta) for m in loaded], self.metas)
        loaded.close_hdf5()

    def test_deferred_load_also_refuses(self):
        with self.assertRaises(ValueError):
            GraphBatch.load(self.path, materialise=False)

    def test_deferred_load_with_allow_pickle(self):
        loaded = GraphBatch.load(self.path, materialise=False, allow_pickle=True)
        self.assertEqual(dict(loaded[0].meta), self.metas[0])
        loaded.close_hdf5()

    def test_load_hdf5_shard_refuses_too(self):
        with self.assertRaises(ValueError):
            GraphBatch.load_hdf5_shard(next(self.path.glob("*.hdf5")))


class TestColumnarUnaffected(unittest.TestCase):
    """Columnar shards contain no pickle, so nothing about them changes."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.metas = [{"id": "a", "n": 1}, {"id": "b", "n": 2}]
        self.path = self.tmp / "col"
        GraphBatch(_mols(self.metas)).save(self.path, columnar_meta=True)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_still_columnar(self):
        with h5py.File(self.path / "0.hdf5", "r") as f:
            self.assertEqual(f["meta"].attrs[FORMAT_ATTR], COLUMNAR_FORMAT)

    def test_loads_with_pickle_disallowed(self):
        loaded = GraphBatch.load(self.path, allow_pickle=False)
        self.assertEqual([dict(m.meta) for m in loaded], self.metas)
        loaded.close_hdf5()

    def test_meta_column_still_works(self):
        loaded = GraphBatch.load(self.path, materialise=False)
        self.assertEqual(loaded.meta_column("id").tolist(), ["a", "b"])
        loaded.close_hdf5()


class TestJsonEncodingOnDisk(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        GraphBatch(_mols([{"id": "a"}, {"id": "b"}])).save(self.tmp / "ds", columnar_meta=False)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_stored_as_a_compressed_dataset(self):
        with h5py.File(self.tmp / "ds" / "0.hdf5", "r") as f:
            dataset = f["meta"]["json"]
            self.assertEqual(dataset.compression, "gzip")

    def test_payload_is_plain_readable_json(self):
        # Anything can read this without molito, which is part of the point
        with h5py.File(self.tmp / "ds" / "0.hdf5", "r") as f:
            payload = f["meta"]["json"][()].tobytes().decode("utf-8")

        self.assertEqual(json.loads(payload), [{"id": "a"}, {"id": "b"}])

    def test_entry_count_mismatch_is_caught(self):
        shard = self.tmp / "ds" / "0.hdf5"
        with h5py.File(shard, "r") as f:
            n = len(f["atoms"]["sizes"])

        with h5py.File(shard, "r") as f:
            with self.assertRaises(RuntimeError):
                load_meta(f["meta"], n + 5)
