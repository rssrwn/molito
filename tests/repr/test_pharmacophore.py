import unittest

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from molito.core.pharmacophore import PharmacophoreFeature, PharmacophoreFinder


class TestPharmacophoreFeature(unittest.TestCase):
    def test_basic_creation(self):
        feat = PharmacophoreFeature(type=0, atom_ids=(0, 1))
        self.assertEqual(feat.type, 0)
        self.assertEqual(feat.atom_ids, (0, 1))
        self.assertIsNone(feat.position)
        self.assertIsNone(feat.direction)

    def test_with_position(self):
        pos = np.array([1.0, 2.0, 3.0])
        feat = PharmacophoreFeature(type=0, atom_ids=(0,), position=pos)
        np.testing.assert_array_equal(feat.position, pos)

    def test_with_direction(self):
        direction = np.array([0.0, 0.0, 1.0])
        feat = PharmacophoreFeature(type=0, atom_ids=(0,), direction=direction)
        np.testing.assert_array_equal(feat.direction, direction)


class TestPharmacophoreFinder(unittest.TestCase):
    def setUp(self):
        PharmacophoreFinder.set_default_features()

    def test_get_vocab_size(self):
        self.assertGreater(PharmacophoreFinder.get_vocab_size(), 0)

    def test_get_feature_vocab(self):
        vocab = PharmacophoreFinder.get_feature_vocab()
        self.assertIsInstance(vocab, list)
        self.assertGreater(len(vocab), 0)

    def test_get_feature_index_and_name_roundtrip(self):
        for name in PharmacophoreFinder.get_feature_vocab():
            idx = PharmacophoreFinder.get_feature_index(name)
            recovered = PharmacophoreFinder.get_feature_name(idx)
            self.assertEqual(name, recovered)

    def test_get_feature_name_out_of_bounds(self):
        with self.assertRaises(RuntimeError):
            PharmacophoreFinder.get_feature_name(999)

    def test_set_rdkit_features(self):
        PharmacophoreFinder.set_rdkit_features()
        vocab = PharmacophoreFinder.get_feature_vocab()
        self.assertGreater(len(vocab), 0)
        PharmacophoreFinder.set_default_features()

    def test_set_custom_features_restores(self):
        original_vocab = PharmacophoreFinder.get_feature_vocab()
        PharmacophoreFinder.set_rdkit_features()
        PharmacophoreFinder.set_default_features()
        restored_vocab = PharmacophoreFinder.get_feature_vocab()
        self.assertEqual(original_vocab, restored_vocab)


class TestRunMolWithoutConformer(unittest.TestCase):
    def setUp(self):
        PharmacophoreFinder.set_default_features()

    def test_returns_features(self):
        mol = Chem.MolFromSmiles("c1ccccc1O")
        features = PharmacophoreFinder.run_mol(mol)
        self.assertIsInstance(features, list)
        self.assertGreater(len(features), 0)

    def test_positions_are_none(self):
        mol = Chem.MolFromSmiles("c1ccccc1O")
        features = PharmacophoreFinder.run_mol(mol)
        for feat in features:
            self.assertIsNone(feat.position)

    def test_directions_are_none(self):
        mol = Chem.MolFromSmiles("c1ccccc1O")
        features = PharmacophoreFinder.run_mol(mol)
        for feat in features:
            self.assertIsNone(feat.direction)

    def test_atom_ids_populated(self):
        mol = Chem.MolFromSmiles("c1ccccc1O")
        features = PharmacophoreFinder.run_mol(mol)
        for feat in features:
            self.assertIsInstance(feat.atom_ids, tuple)
            self.assertGreater(len(feat.atom_ids), 0)


class TestRunMolWithConformer(unittest.TestCase):
    def setUp(self):
        PharmacophoreFinder.set_default_features()
        self.mol = Chem.MolFromSmiles("c1ccccc1O")
        self.mol = Chem.AddHs(self.mol)
        AllChem.EmbedMolecule(self.mol, randomSeed=42)

    def test_positions_populated(self):
        features = PharmacophoreFinder.run_mol(self.mol)
        for feat in features:
            self.assertIsNotNone(feat.position)
            self.assertEqual(feat.position.shape, (3,))

    def test_directions_false_leaves_directions_none(self):
        features = PharmacophoreFinder.run_mol(self.mol, directions=False)
        for feat in features:
            self.assertIsNone(feat.direction)


class TestRunMolWithDirections(unittest.TestCase):
    def setUp(self):
        PharmacophoreFinder.set_default_features()

    def test_directions_true_no_conformer_raises(self):
        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.AddHs(mol)
        with self.assertRaises(ValueError):
            PharmacophoreFinder.run_mol(mol, directions=True)

    def test_directions_true_no_hs_raises(self):
        mol = Chem.MolFromSmiles("CCO")
        AllChem.EmbedMolecule(mol, randomSeed=42)
        with self.assertRaises(ValueError):
            PharmacophoreFinder.run_mol(mol, directions=True)

    def test_donor_directions_populated(self):
        # Ethanol has an OH donor
        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)

        features = PharmacophoreFinder.run_mol(mol, directions=True)
        donor_idx = PharmacophoreFinder.get_feature_index("Donor")
        donor_feats = [f for f in features if f.type == donor_idx]

        self.assertGreater(len(donor_feats), 0)
        for feat in donor_feats:
            self.assertIsNotNone(feat.direction)
            norm = np.linalg.norm(feat.direction)
            self.assertAlmostEqual(norm, 1.0, places=5)

    def test_aromatic_directions_populated(self):
        mol = Chem.MolFromSmiles("c1ccccc1")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)

        features = PharmacophoreFinder.run_mol(mol, directions=True)
        aromatic_idx = PharmacophoreFinder.get_feature_index("Aromatic")
        aromatic_feats = [f for f in features if f.type == aromatic_idx]

        self.assertGreater(len(aromatic_feats), 0)
        for feat in aromatic_feats:
            self.assertIsNotNone(feat.direction)
            norm = np.linalg.norm(feat.direction)
            self.assertAlmostEqual(norm, 1.0, places=5)

    def test_non_donor_non_aromatic_directions_none(self):
        # Acceptor features should not get directions
        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)

        features = PharmacophoreFinder.run_mol(mol, directions=True)
        donor_idx = PharmacophoreFinder.get_feature_index("Donor")
        aromatic_idx = PharmacophoreFinder.get_feature_index("Aromatic")

        other_feats = [f for f in features if f.type not in (donor_idx, aromatic_idx)]
        for feat in other_feats:
            self.assertIsNone(feat.direction)
