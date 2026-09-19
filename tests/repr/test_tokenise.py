import json
import re
import tempfile
import unittest
from importlib.resources import files
from pathlib import Path

from rdkit import Chem, RDConfig

from molito.core.vocab import SMILES_CHARACTERS, STRING_SPECIAL_TOKENS, StringVocab
from molito.tokenise import SMILES_REGEX, CharTokeniser, RegexTokeniser, Tokeniser

# Independent examples exercise syntax rather than merely enumerating vocabulary entries.
SMILES_EXAMPLES = [
    "",
    "CCO",
    "ClCBr",
    "c1cc[nH]c1",
    "N[C@@H](C)C(=O)O",
    "C/C=C/C",
    "C/C=C\\C",
    "[Na+].[Cl-]",
    "*CC(*)O",
    "[13CH3:147][C@H:2]([NH3+])C(=O)[O-]",
    "[2H]O[2H]",
    "C%12CCCCC%12",
    "C%(123)CCCCC%(123)",
    "[Fe+2]",
    "[U]",
    "[Og]",
    "[se]1cccc1",
    "N->[Cu+2]<-N",
    "[C]",
    "[NH4+]",
    "F[Pt@SP1](Cl)(Br)I",
    "S=[S@TB1](F)(Cl)Br",
    "O[Co@OH1](Cl)(Cl)(Cl)(Cl)N",
    "[H][C@@](F)(Cl)Br",
    "[Xe]",
    "[SiH2]1CC1",
]


class TestStringVocabulary(unittest.TestCase):
    def test_alphabet_covers_all_elements_and_smiles_syntax(self):
        vocab = StringVocab.build()
        periodic_table = Chem.GetPeriodicTable()
        for number in range(1, 119):
            symbol = periodic_table.GetElementSymbol(number)
            self.assertTrue(set(symbol).issubset(vocab), symbol)
        self.assertTrue(set("bcnopsasTHALSPTBOH0123456789[]()@+-=#$:/\\.%*~<>").issubset(vocab))
        self.assertNotIn(" ", vocab)
        self.assertNotIn("!", vocab)
        self.assertNotIn("é", vocab)
        self.assertLess(len(SMILES_CHARACTERS), 80)
        self.assertEqual(vocab["<PAD>"], 0)
        self.assertTrue(set(STRING_SPECIAL_TOKENS).issubset(vocab))

    def test_presets_store_complete_ordered_vocabularies(self):
        presets = [
            ("smiles_char_vocab.json", StringVocab.characters(), 78),
            ("smiles_regex_vocab.json", StringVocab.smiles(), 169),
        ]
        for filename, vocab, size in presets:
            data = json.loads(files("molito").joinpath(f"defs/{filename}").read_text())
            self.assertEqual(list(vocab), data["tokens"])
            self.assertEqual(len(vocab), size)
            self.assertEqual(data["special_tokens"], list(STRING_SPECIAL_TOKENS))
            self.assertNotIn("filter", data)
        self.assertEqual(list(StringVocab.build()), list(StringVocab.characters()))
        vocab = StringVocab.smiles()
        for token in ["Cl", "Br", "[nH]", "[C@H]", "[C@@H]", "[NH3+]", ".", "*", "@", "/", "\\"]:
            self.assertIn(token, vocab)
        for token in ["[13CH3]", "[CH3:147]", "[223Ra+2]", "<UNUSED_0>", "LogD_change_(-0.1, 0.1]"]:
            self.assertNotIn(token, vocab)

    def test_extension_preserves_indices_and_original(self):
        original = StringVocab.smiles()
        extended = original.extend(["[13CH3]", "<CUSTOM>", "C"])
        self.assertNotIn("[13CH3]", original)
        self.assertEqual(len(extended), len(original) + 2)
        for token in original:
            self.assertEqual(original[token], extended[token])
        self.assertEqual(list(StringVocab.from_bytes(extended.to_bytes())), list(extended))

    def test_invalid_vocab_definitions(self):
        with self.assertRaises(ValueError):
            StringVocab.build(characters=["Cl"])
        with self.assertRaises(ValueError):
            StringVocab.build([""])
        with self.assertRaises(ValueError):
            StringVocab(["C"])
        with self.assertRaises(ValueError):
            CharTokeniser(StringVocab.smiles())


class TestTokenisers(unittest.TestCase):
    def test_targeted_examples_are_valid_smiles(self):
        for text in SMILES_EXAMPLES:
            with self.subTest(text=text):
                self.assertIsNotNone(Chem.MolFromSmiles(text))

    def test_exact_roundtrips(self):
        for tokeniser in [CharTokeniser(), RegexTokeniser.smiles()]:
            for text in SMILES_EXAMPLES:
                with self.subTest(tokeniser=type(tokeniser).__name__, text=text):
                    self.assertEqual(tokeniser.detokenise(tokeniser.tokenise(text)), text)
                    self.assertEqual(tokeniser.decode(tokeniser.encode(text)), text)

    def test_unknown_bracket_atoms_fall_back_without_losing_information(self):
        tokeniser = RegexTokeniser.smiles()
        self.assertEqual(tokeniser.tokenise("[CH3:147]Cl"), [*"[CH3:147]", "Cl"])
        self.assertEqual(tokeniser.tokenise("[13CH3][nH]"), [*"[13CH3]", "[nH]"])
        self.assertEqual(tokeniser.tokenise("[C@@H](Cl)Br"), ["[C@@H]", "(", "Cl", ")", "Br"])

    def test_invalid_strings_preserved_and_unsupported_characters_reported(self):
        tokeniser = RegexTokeniser.smiles()
        for text in ["C1(", "[C@@", "C%", "C(C)(C)(C)(C)C", "C..O", "[]"]:
            self.assertEqual(tokeniser.decode(tokeniser.encode(text)), text)
        text = "C !é\n"
        self.assertEqual("".join(tokeniser.tokenise(text)), text)
        with self.assertRaisesRegex(ValueError, "extend"):
            tokeniser.encode(text)
        extended = RegexTokeniser(SMILES_REGEX, tokeniser.vocab.extend([" ", "!", "é", "\n"]))
        self.assertEqual(extended.decode(extended.encode(text)), text)

    def test_control_tokens_are_explicit(self):
        for tokeniser in [CharTokeniser(), RegexTokeniser.smiles()]:
            text = "C<MASK>O"
            ids = tokeniser.encode(text)
            self.assertNotIn(tokeniser.vocab["<MASK>"], ids)
            self.assertEqual(tokeniser.decode(ids, skip_special_tokens=True), text)
            ids = tokeniser.encode("C.O", add_special_tokens=True)
            self.assertEqual(ids[0], tokeniser.vocab["<BOS>"])
            self.assertEqual(ids[-1], tokeniser.vocab["<EOS>"])
            self.assertEqual(tokeniser.decode(ids), "<BOS>C.O<EOS>")
            self.assertEqual(tokeniser.decode([*ids, tokeniser.vocab["<PAD>"]], skip_special_tokens=True), "C.O")

    def test_custom_regex_captures_and_gaps(self):
        tokeniser = RegexTokeniser(r"(Cl)|(Br)", StringVocab.build(["Cl", "Br"]))
        self.assertEqual(tokeniser.split("ClC!?Br"), ["Cl", "C", "!", "?", "Br"])
        self.assertEqual(tokeniser.tokenise("ClCBr"), ["Cl", "C", "Br"])
        with self.assertRaises(ValueError):
            RegexTokeniser(r"C*")
        with self.assertRaises(ValueError):
            RegexTokeniser(r"(?=C)").tokenise("C")

    def test_extra_tokens_are_literals_and_longest_match_wins(self):
        tokeniser = RegexTokeniser.smiles(extra_tokens=["[13CH3]", "CC", "CCC", "<CUSTOM>"])
        self.assertEqual(tokeniser.tokenise("CCC[13CH3]<CUSTOM>"), ["CCC", "[13CH3]", "<CUSTOM>"])
        with self.assertRaises(ValueError):
            RegexTokeniser.smiles(extra_tokens=["<MASK>"])

    def test_fit_is_deterministic_filtered_and_frozen(self):
        original = RegexTokeniser(SMILES_REGEX)
        data = ["[nH]", "[nH]", "Br", "Cl", "[13CH3]", "[CH3:1]", "[CH3:2]"]
        fitted = original.fit(data, max_tokens=2)
        reverse = original.fit(reversed(data), max_tokens=2)
        self.assertEqual(list(fitted.vocab), list(reverse.vocab))
        self.assertIn("[nH]", fitted.vocab)
        self.assertIn("Br", fitted.vocab)
        self.assertNotIn("Cl", fitted.vocab)
        self.assertNotIn("[13CH3]", fitted.vocab)
        self.assertNotIn("[nH]", original.vocab)
        self.assertEqual(original.fit(data, min_frequency=2).tokenise("[nH]Cl"), ["[nH]", "C", "l"])
        snapshot = list(fitted.vocab)
        fitted.encode("[13CH3:999]C")
        self.assertEqual(snapshot, list(fitted.vocab))
        unfiltered = original.fit(data, token_filter=None)
        self.assertIn("[13CH3]", unfiltered.vocab)
        included = original.fit(data, token_filter=lambda token: False, extra_tokens=["[CH3:1]"])
        self.assertEqual(included.tokenise("[CH3:1]"), ["[CH3:1]"])

    def test_save_load_preserves_full_configuration(self):
        tokenisers = [
            CharTokeniser(),
            RegexTokeniser.smiles(),
            RegexTokeniser(r"(Cl)|(Br)", flags=re.IGNORECASE, extra_tokens=["CC"]),
        ]
        with tempfile.TemporaryDirectory() as directory:
            for idx, tokeniser in enumerate(tokenisers):
                path = Path(directory) / f"{idx}.json"
                tokeniser.save(path)
                loaded = Tokeniser.load(path)
                self.assertEqual(loaded.to_bytes(), tokeniser.to_bytes())
                self.assertEqual(loaded.encode("CCClBr"), tokeniser.encode("CCClBr"))
                with self.assertRaises(FileExistsError):
                    tokeniser.save(path)

    def test_nci_corpus_roundtrips(self):
        path = Path(RDConfig.RDDataDir) / "NCI" / "first_5K.smi"
        if not path.exists():
            self.skipTest("RDKit installation does not include the NCI SMILES corpus.")
        texts = [line.split()[0] for line in path.read_text().splitlines() if line.strip()]
        self.assertGreater(len(texts), 4000)
        for tokeniser in [CharTokeniser(), RegexTokeniser.smiles()]:
            for idx, text in enumerate(texts):
                with self.subTest(tokeniser=type(tokeniser).__name__, record=idx):
                    self.assertEqual(tokeniser.decode(tokeniser.encode(text)), text)
