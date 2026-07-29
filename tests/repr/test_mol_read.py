"""Tests for GraphMol.read: detaching a loaded molecule from its HDF5 file."""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from molito.mol.graph import GraphBatch, GraphMol


def _mol_with_conformer(smi: str = "CCO", seed: int = 3) -> GraphMol:
    rdkit_mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    AllChem.EmbedMolecule(rdkit_mol, randomSeed=seed)
    return GraphMol.from_rdkit(rdkit_mol)


class TestReadInMemory(unittest.TestCase):
    """read() on an already in-memory molecule is a no-op in effect."""

    def test_returns_equivalent_mol(self):
        mol = _mol_with_conformer()
        got = mol.read()

        self.assertIsInstance(got, GraphMol)
        np.testing.assert_array_equal(got.atomics, mol.atomics)
        np.testing.assert_array_equal(got.bond_types, mol.bond_types)
        np.testing.assert_allclose(got.coords, mol.coords)

    def test_returns_a_new_object(self):
        mol = _mol_with_conformer()
        self.assertIsNot(mol.read(), mol)

    def test_handles_mol_without_conformers(self):
        mol = GraphMol.from_smiles("c1ccccc1")
        got = mol.read()

        self.assertIsNone(got.confs)
        self.assertIsNone(got.coords)
        np.testing.assert_array_equal(got.atomics, mol.atomics)

    def test_preserves_meta(self):
        mol = GraphMol.from_smiles("CCO")
        mol.meta = {"id": "x"}
        self.assertEqual(dict(mol.read().meta), {"id": "x"})

    def test_preserves_stereo(self):
        mol = GraphMol.from_smiles("N[C@@H](C)C(=O)O")
        self.assertEqual(mol.read().to_smiles(), mol.to_smiles())


class TestReadDetachesFromFile(unittest.TestCase):
    """The point of read(): surviving close_hdf5()."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.mols = [_mol_with_conformer("CCO", 1), _mol_with_conformer("CCN", 2)]

        for idx, mol in enumerate(self.mols):
            mol.meta = {"id": str(idx)}

        GraphBatch(self.mols).save(self.tmp / "ds", columnar_meta=True)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _assert_survives_close(self, materialise: bool):
        loaded = GraphBatch.load(self.tmp / "ds", materialise=materialise)
        detached = [mol.read() for mol in loaded]
        loaded.close_hdf5()

        for original, got in zip(self.mols, detached, strict=True):
            np.testing.assert_array_equal(got.atomics, original.atomics)
            np.testing.assert_array_equal(got.bond_types, original.bond_types)
            np.testing.assert_allclose(got.coords, original.coords, atol=1e-6)
            self.assertEqual(dict(got.meta), dict(original.meta))

    def test_survives_close_when_materialised(self):
        self._assert_survives_close(materialise=True)

    def test_survives_close_when_deferred(self):
        self._assert_survives_close(materialise=False)

    def test_without_read_the_mol_breaks_after_close(self):
        # The behaviour read() exists to work around. h5py reports a closed file as either
        # OSError or RuntimeError depending on whether it was read before closing, so this
        # deliberately does not pin the class.
        loaded = GraphBatch.load(self.tmp / "ds")
        mol = loaded[0]
        loaded.close_hdf5()

        with self.assertRaises((OSError, RuntimeError)):
            _ = mol.atomics

    def test_detached_batch_is_fully_usable(self):
        loaded = GraphBatch.load(self.tmp / "ds")
        train = GraphBatch([mol.read() for mol in loaded.subset([0, 1])])
        loaded.close_hdf5()

        self.assertEqual(train.atomics.shape[0], 2)
        self.assertEqual(train.adjacency.shape[0], 2)
        self.assertEqual(train.coords.shape[0], 2)

    def test_detached_mol_meta_is_mutable(self):
        # dict(...) of a read-only view gives a plain dict, so the detached copy is writable
        loaded = GraphBatch.load(self.tmp / "ds")
        mol = loaded[0].read()
        loaded.close_hdf5()

        mol.meta["extra"] = "value"
        self.assertEqual(mol.meta["extra"], "value")
