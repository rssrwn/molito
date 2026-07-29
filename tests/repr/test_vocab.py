import unittest

import numpy as np
from rdkit import Chem

from molito.core.bonds import BondEncoding
from molito.core.vocab import AtomVocab, BondVocab, Vocabulary


class TestVocabulary(unittest.TestCase):
    def setUp(self):
        self.tokens = ["apple", "banana", "cherry", "date"]
        self.vocab = Vocabulary(self.tokens)

    def test_length(self):
        self.assertEqual(len(self.vocab), 4)

    def test_get_index(self):
        self.assertEqual(self.vocab.get_index("apple"), 0)
        self.assertEqual(self.vocab.get_index("banana"), 1)
        self.assertEqual(self.vocab.get_index("cherry"), 2)
        self.assertEqual(self.vocab.get_index("date"), 3)

    def test_get_token(self):
        self.assertEqual(self.vocab.get_token(0), "apple")
        self.assertEqual(self.vocab.get_token(1), "banana")
        self.assertEqual(self.vocab.get_token(2), "cherry")
        self.assertEqual(self.vocab.get_token(3), "date")

    def test_getitem_returns_index(self):
        self.assertEqual(self.vocab["apple"], 0)
        self.assertEqual(self.vocab["cherry"], 2)

    def test_contains_token(self):
        self.assertTrue(self.vocab.contains_token("apple"))
        self.assertTrue("banana" in self.vocab)
        self.assertFalse(self.vocab.contains_token("grape"))
        self.assertFalse("grape" in self.vocab)

    def test_contains_index(self):
        self.assertTrue(self.vocab.contains_index(0))
        self.assertTrue(self.vocab.contains_index(3))
        self.assertFalse(self.vocab.contains_index(4))
        self.assertFalse(self.vocab.contains_index(-1))

    def test_tokens_from_indices(self):
        tokens = self.vocab.tokens_from_indices([2, 0, 3])
        self.assertEqual(tokens, ["cherry", "apple", "date"])

    def test_indices_from_tokens(self):
        indices = self.vocab.indices_from_tokens(["date", "banana"])
        self.assertEqual(indices, [3, 1])

    def test_iter_tokens(self):
        tokens = list(self.vocab.iter_tokens())
        self.assertEqual(tokens, ["apple", "banana", "cherry", "date"])

    def test_iter_indices(self):
        indices = list(self.vocab.iter_indices())
        self.assertEqual(indices, [0, 1, 2, 3])

    def test_iter_uses_tokens(self):
        tokens = list(self.vocab)
        self.assertEqual(tokens, ["apple", "banana", "cherry", "date"])

    def test_bytes_serialization(self):
        data = self.vocab.to_bytes()
        restored = Vocabulary.from_bytes(data)

        self.assertEqual(len(restored), len(self.vocab))
        self.assertEqual(list(restored), list(self.vocab))

    def test_duplicate_tokens_raises_error(self):
        with self.assertRaises(RuntimeError):
            Vocabulary(["a", "b", "a"])


class TestBondEncoding(unittest.TestCase):
    def test_size(self):
        self.assertEqual(BondEncoding.size(), 12)

    def test_encode_basic_bonds(self):
        self.assertEqual(BondEncoding.encode("NONE"), 0)
        self.assertEqual(BondEncoding.encode(Chem.BondType.SINGLE), 1)
        self.assertEqual(BondEncoding.encode(Chem.BondType.DOUBLE), 2)
        self.assertEqual(BondEncoding.encode(Chem.BondType.TRIPLE), 3)

    def test_encode_aromatic_bonds(self):
        self.assertEqual(BondEncoding.encode(Chem.BondType.SINGLE, is_aromatic=True), 4)
        self.assertEqual(BondEncoding.encode(Chem.BondType.DOUBLE, is_aromatic=True), 5)
        self.assertEqual(BondEncoding.encode(Chem.BondType.TRIPLE, is_aromatic=True), 6)

    def test_encode_direction_bonds(self):
        self.assertEqual(BondEncoding.encode(Chem.BondType.SINGLE, direction=Chem.BondDir.ENDUPRIGHT), 7)
        self.assertEqual(BondEncoding.encode(Chem.BondType.SINGLE, direction=Chem.BondDir.ENDDOWNRIGHT), 8)

    def test_encode_mask_bond(self):
        self.assertEqual(BondEncoding.encode("MASK"), 11)

    def test_decode_roundtrip(self):
        for idx in range(BondEncoding.size()):
            bond_type, is_arom, direction = BondEncoding.decode(idx)
            re_encoded = BondEncoding.encode(bond_type, is_arom, direction=direction)
            self.assertEqual(re_encoded, idx, f"Roundtrip failed for index {idx}")

    def test_decode_basic(self):
        bond_type, is_arom, direction = BondEncoding.decode(1)
        self.assertEqual(bond_type, Chem.BondType.SINGLE)
        self.assertFalse(is_arom)
        self.assertIsNone(direction)

    def test_decode_aromatic(self):
        bond_type, is_arom, direction = BondEncoding.decode(5)
        self.assertEqual(bond_type, Chem.BondType.DOUBLE)
        self.assertTrue(is_arom)
        self.assertIsNone(direction)


class TestBondVocab(unittest.TestCase):
    def test_build_with_directions(self):
        vocab = BondVocab.build(directions=True)
        self.assertEqual(len(vocab), 12)

    def test_build_without_directions(self):
        vocab = BondVocab.build(directions=False)
        self.assertEqual(len(vocab), 8)

    def test_encode_basic(self):
        vocab = BondVocab.build(directions=True)
        idx = vocab.encode(Chem.BondType.SINGLE)
        token = vocab.get_token(idx)
        self.assertEqual(token, "1_F")

    def test_get_mask_index(self):
        vocab = BondVocab.build(directions=True)
        mask_idx = vocab.get_mask_index()
        self.assertEqual(vocab.get_token(mask_idx), "-1_F")

    def test_get_bond_type(self):
        vocab = BondVocab.build(directions=True)
        self.assertEqual(vocab.get_bond_type(vocab["1_F"]), Chem.BondType.SINGLE)
        self.assertEqual(vocab.get_bond_type(vocab["2_F"]), Chem.BondType.DOUBLE)

    def test_get_is_aromatic(self):
        vocab = BondVocab.build(directions=True)
        self.assertFalse(vocab.get_is_aromatic(vocab["1_F"]))
        self.assertTrue(vocab.get_is_aromatic(vocab["1_T"]))


class TestAtomVocab(unittest.TestCase):
    def test_build_with_chirality(self):
        vocab = AtomVocab.build(chirality=True)
        self.assertIn("C_0", vocab)
        self.assertIn("C_0_CW", vocab)
        self.assertIn("C_0_CCW", vocab)

    def test_build_without_chirality(self):
        vocab = AtomVocab.build(chirality=False)
        self.assertIn("C_0", vocab)
        self.assertNotIn("C_0_CW", vocab)

    def test_resolve_token_with_fallback(self):
        vocab = AtomVocab.build(chirality=False)
        idx = vocab.resolve_token("C_0_CW")
        self.assertEqual(idx, vocab["C_0"])

    def test_resolve_token_direct(self):
        vocab = AtomVocab.build(chirality=True)
        idx = vocab.resolve_token("C_0_CW")
        self.assertEqual(idx, vocab["C_0_CW"])

    def test_resolve_token_unknown_raises(self):
        vocab = AtomVocab.build(chirality=True)
        with self.assertRaises(KeyError):
            vocab.resolve_token("Xx_99")

    def test_resolve_tokens_list(self):
        vocab = AtomVocab.build(chirality=True)
        tokens = ["C_0", "C_0_CW", "N_0"]
        indices = vocab.resolve_tokens(tokens)
        self.assertEqual(len(indices), 3)
        self.assertEqual(indices[0], vocab["C_0"])
        self.assertEqual(indices[1], vocab["C_0_CW"])

    def test_has_pad_and_mask(self):
        vocab = AtomVocab.build()
        self.assertIn("PAD", vocab)
        self.assertIn("MASK", vocab)
        self.assertEqual(vocab["PAD"], 0)

    def test_custom_tokens(self):
        vocab = AtomVocab.build(tokens=["C_0", "N_0"], chirality=False)
        self.assertIn("C_0", vocab)
        self.assertIn("N_0", vocab)
        self.assertNotIn("O_0", vocab)


class TestBondVocabDirectionFallback(unittest.TestCase):
    def test_direction_bond_falls_back_without_directions(self):
        vocab = BondVocab.build(directions=False)
        # Encoding index for single non-aromatic ENDUPRIGHT (index 7 in BondEncoding)
        enc_idx = BondEncoding.encode(Chem.BondType.SINGLE, direction=Chem.BondDir.ENDUPRIGHT)
        model_idx = vocab.encoding_to_model_index(enc_idx)
        token = vocab.get_token(model_idx)
        self.assertEqual(token, "1_F")

    def test_direction_bond_keeps_direction_with_directions(self):
        vocab = BondVocab.build(directions=True)
        enc_idx = BondEncoding.encode(Chem.BondType.SINGLE, direction=Chem.BondDir.ENDUPRIGHT)
        model_idx = vocab.encoding_to_model_index(enc_idx)
        token = vocab.get_token(model_idx)
        self.assertEqual(token, "1_F_U")


class TestBondVocabResolveTypes(unittest.TestCase):
    def test_resolve_types_matches_scalar(self):
        vocab = BondVocab.build(directions=False)
        enc = np.arange(BondEncoding.size(), dtype=np.int64)
        expected = np.array([vocab.encoding_to_model_index(int(i)) for i in enc])
        np.testing.assert_array_equal(vocab.resolve_types(enc), expected)

    def test_resolve_types_preserves_shape(self):
        vocab = BondVocab.build(directions=True)
        enc = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
        out = vocab.resolve_types(enc)
        self.assertEqual(out.shape, enc.shape)

    def test_resolve_types_raises_on_unmapped(self):
        # Vocab missing the base "1_F" so encoded indices for any single bond have no mapping.
        vocab = BondVocab(["0_F", "2_F", "-1_F"])
        bad_enc = np.array([1])
        with self.assertRaises(KeyError):
            vocab.resolve_types(bad_enc)


class TestVocabConfig(unittest.TestCase):
    def setUp(self):
        from molito.core.vocab import VocabConfig

        VocabConfig.reset()
        self.vc = VocabConfig

    def tearDown(self):
        self.vc.reset()

    def test_default_has_chirality(self):
        self.assertIn("C_0_CW", self.vc.atoms)

    def test_default_has_directions(self):
        self.assertIn("1_F_U", self.vc.bonds)

    def test_disable_chirality(self):
        self.vc.set_chirality(False)
        self.assertNotIn("C_0_CW", self.vc.atoms)
        self.assertIn("C_0", self.vc.atoms)

    def test_disable_directions(self):
        self.vc.set_directions(False)
        self.assertNotIn("1_F_U", self.vc.bonds)
        self.assertIn("1_F", self.vc.bonds)

    def test_reset_restores_defaults(self):
        self.vc.set_chirality(False)
        self.vc.set_directions(False)
        self.vc.reset()
        self.assertIn("C_0_CW", self.vc.atoms)
        self.assertIn("1_F_U", self.vc.bonds)

    def test_set_atom_tokens(self):
        self.vc.set_atom_tokens(["C_0", "N_0"])
        self.assertIn("C_0", self.vc.atoms)
        self.assertIn("N_0", self.vc.atoms)
        self.assertNotIn("O_0", self.vc.atoms)

    def test_set_atom_tokens_respects_chirality(self):
        self.vc.set_chirality(True)
        self.vc.set_atom_tokens(["C_0", "N_0"])
        self.assertIn("C_0_CW", self.vc.atoms)
