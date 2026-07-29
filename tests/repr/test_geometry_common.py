import unittest

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from molito.geometry.common import possibly_add_hs, sample_conformers, sample_ensemble


class TestPossiblyAddHs(unittest.TestCase):
    def test_adds_hs_when_missing(self):
        mol = Chem.MolFromSmiles("CCO")
        AllChem.EmbedMolecule(mol, randomSeed=42)
        result = possibly_add_hs(mol)
        self.assertIsNotNone(result)
        self.assertGreater(result.GetNumAtoms(), mol.GetNumAtoms())

    def test_does_not_add_hs_when_present(self):
        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        result = possibly_add_hs(mol)
        self.assertEqual(result.GetNumAtoms(), mol.GetNumAtoms())

    def test_does_not_modify_input(self):
        mol = Chem.MolFromSmiles("CCO")
        AllChem.EmbedMolecule(mol, randomSeed=42)
        n_atoms_before = mol.GetNumAtoms()
        possibly_add_hs(mol)
        self.assertEqual(mol.GetNumAtoms(), n_atoms_before)


class TestSampleConformers(unittest.TestCase):
    def test_single_conformer(self):
        mol = Chem.MolFromSmiles("CCCC")
        result = sample_conformers(mol, n_confs=1)
        self.assertIsNotNone(result)
        self.assertEqual(result.GetNumConformers(), 1)

    def test_multiple_conformers(self):
        mol = Chem.MolFromSmiles("CCCCCC")
        result = sample_conformers(mol, n_confs=5)
        self.assertIsNotNone(result)
        self.assertEqual(result.GetNumConformers(), 5)

    def test_does_not_modify_input(self):
        mol = Chem.MolFromSmiles("CCCC")
        n_confs_before = mol.GetNumConformers()
        sample_conformers(mol, n_confs=3)
        self.assertEqual(mol.GetNumConformers(), n_confs_before)

    def test_with_opt_iters(self):
        mol = Chem.MolFromSmiles("CCCC")
        result = sample_conformers(mol, n_confs=1, opt_iters=100)
        self.assertIsNotNone(result)
        self.assertEqual(result.GetNumConformers(), 1)

    def test_returns_mol_with_hs(self):
        mol = Chem.MolFromSmiles("CCO")
        result = sample_conformers(mol, n_confs=1)
        self.assertIsNotNone(result)
        self.assertGreater(result.GetNumAtoms(), mol.GetNumHeavyAtoms())


class TestSampleEnsemble(unittest.TestCase):
    def test_basic_ensemble(self):
        mol = Chem.MolFromSmiles("CCCCCC")
        result = sample_ensemble(mol, max_confs=8)
        self.assertIsNotNone(result)
        final_mol, weights, e_min = result
        self.assertGreater(final_mol.GetNumConformers(), 0)
        self.assertEqual(len(weights), final_mol.GetNumConformers())
        self.assertIsInstance(e_min, float)

    def test_weights_sum_to_one(self):
        mol = Chem.MolFromSmiles("CCCCCC")
        result = sample_ensemble(mol, max_confs=8)
        _, weights, _ = result
        self.assertAlmostEqual(float(np.sum(weights)), 1.0, places=5)

    def test_weights_are_positive(self):
        mol = Chem.MolFromSmiles("CCCCCC")
        result = sample_ensemble(mol, max_confs=8)
        _, weights, _ = result
        self.assertTrue(np.all(weights > 0))
