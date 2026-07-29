"""Regression tests for bugs surfaced by adding a type checker.

Each of these was a case where the declared types and the runtime behaviour disagreed, in a
way that only bit a caller who took the signature at its word.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from molito.core.confs import ConfSet
from molito.geometry.mmff import calc_energy_mmff
from molito.mol.graph import GraphBatch, GraphMol
from molito.mol.protein import ProteinBatch
from tests.repr.test_protein_conformers import _protein


class TestStringPathsAccepted(unittest.TestCase):
    """save_hdf5_shard advertises `str | Path` but checked `save_file.exists()` on the
    unconverted argument, so passing a string raised AttributeError.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_graph_batch_accepts_str_path(self):
        batch = GraphBatch([GraphMol.from_smiles("CCO")])
        batch.save_hdf5_shard(str(self.tmp / "mols.hdf5"))
        self.assertTrue((self.tmp / "mols.hdf5").is_file())

    def test_protein_batch_accepts_str_path(self):
        ProteinBatch([_protein()]).save_hdf5_shard(str(self.tmp / "prots.hdf5"))
        self.assertTrue((self.tmp / "prots.hdf5").is_file())

    def test_str_path_still_refuses_to_overwrite(self):
        batch = GraphBatch([GraphMol.from_smiles("CCO")])
        target = str(self.tmp / "mols.hdf5")
        batch.save_hdf5_shard(target)

        with self.assertRaises(RuntimeError):
            batch.save_hdf5_shard(target)

    def test_path_and_str_produce_the_same_file(self):
        batch = GraphBatch([GraphMol.from_smiles("CCO")])
        batch.save_hdf5_shard(str(self.tmp / "a.hdf5"))
        batch.save_hdf5_shard(self.tmp / "b.hdf5")

        self.assertEqual((self.tmp / "a.hdf5").stat().st_size, (self.tmp / "b.hdf5").stat().st_size)


class TestGeometryWithoutConformers(unittest.TestCase):
    """Geometric transforms reached through `self.confs`, which is None for a graph-only
    molecule, giving "'NoneType' object has no attribute 'rotate'".
    """

    def setUp(self):
        self.mol = GraphMol.from_smiles("CCO")  # no conformers

    def test_rotate_explains_itself(self):
        from scipy.spatial.transform import Rotation

        with self.assertRaises(ValueError) as ctx:
            self.mol.rotate(Rotation.identity())

        self.assertIn("no conformers", str(ctx.exception))

    def test_shift_explains_itself(self):
        with self.assertRaises(ValueError) as ctx:
            self.mol.shift(np.zeros(3, dtype=np.float32))

        self.assertIn("no conformers", str(ctx.exception))

    def test_scale_explains_itself(self):
        with self.assertRaises(ValueError):
            self.mol.scale(2.0)

    def test_zero_com_explains_itself(self):
        with self.assertRaises(ValueError):
            self.mol.zero_com()

    def test_get_conformer_explains_itself(self):
        with self.assertRaises(ValueError):
            self.mol.get_conformer(0)

    def test_transforms_still_work_with_conformers(self):
        mol = GraphMol(self.mol.atoms, self.mol.bonds, confs=ConfSet(np.zeros((1, 3, 3), dtype=np.float32)))
        self.assertEqual(mol.shift(np.ones(3, dtype=np.float32)).coords.shape, (1, 3, 3))


class TestOptionalReturnsAreDeclared(unittest.TestCase):
    """Signatures that promised a value but returned None on failure."""

    def test_conf_weights_may_be_none(self):
        confs = ConfSet(np.zeros((2, 3, 3), dtype=np.float32))
        self.assertIsNone(confs.weights)

    def test_conf_weights_present_when_given(self):
        confs = ConfSet(np.zeros((2, 3, 3), dtype=np.float32), weights=np.array([0.4, 0.6], dtype=np.float32))
        self.assertIsNotNone(confs.weights)

    def test_calc_energy_returns_none_on_unusable_mol(self):
        from rdkit import Chem

        # No conformer, so there is no geometry to score and hydrogen placement fails
        self.assertIsNone(calc_energy_mmff(Chem.MolFromSmiles("CCO")))

    def test_calc_energy_returns_a_float_for_a_real_conformer(self):
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        AllChem.EmbedMolecule(mol, randomSeed=11)
        self.assertIsInstance(calc_energy_mmff(mol), float)

    def test_to_rdkit_may_return_none_for_an_unsanitisable_graph(self):
        from molito.core.atoms import AtomSet
        from molito.core.bonds import BondSet

        # Carbon with five single bonds - RDKit rejects it on sanitisation
        atoms = AtomSet(np.array([6, 1, 1, 1, 1, 1], dtype=np.uint8))
        bonds = BondSet(np.array([[0, i, 1] for i in range(1, 6)], dtype=np.int16))
        mol = GraphMol(atoms, bonds)

        self.assertIsNone(mol.to_rdkit(sanitise=True))

    def test_to_smiles_raises_rather_than_returning_none(self):
        from molito.core.atoms import AtomSet
        from molito.core.bonds import BondSet

        atoms = AtomSet(np.array([6, 1, 1, 1, 1, 1], dtype=np.uint8))
        bonds = BondSet(np.array([[0, i, 1] for i in range(1, 6)], dtype=np.int16))
        mol = GraphMol(atoms, bonds)

        with self.assertRaises(ValueError):
            mol.to_smiles()

    def test_to_rdkit_normal_case_still_returns_a_mol(self):
        self.assertIsNotNone(GraphMol.from_smiles("CCO").to_rdkit())
