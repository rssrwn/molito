import inspect
import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem

from molito import ConversionError, GraphBatch, GraphMol, MolBatch, MolRepr, RDKitMol, SmilesMol, StringMol
from molito.core import AtomSet, BondSet, ConfSet
from molito.tokenise import RegexTokeniser


def _rich_mol():
    mol = Chem.AddHs(Chem.MolFromSmiles("[13CH3:7][C@H](F)Cl"))
    AllChem.EmbedMultipleConfs(mol, numConfs=2, randomSeed=42)
    mol.SetProp("source", "example")
    mol.SetIntProp("number", 17)
    mol.SetProp("_private", "keep")
    mol.GetAtomWithIdx(0).SetDoubleProp("score", 1.25)
    mol.GetBondWithIdx(0).SetBoolProp("selected", True)
    for idx, conf in enumerate(mol.GetConformers()):
        conf.SetProp("weight", str(0.25 + idx * 0.5))
        conf.SetIntProp("rank", idx)
    return mol


class TestMoleculeRepresentations(unittest.TestCase):
    def test_common_interface_and_string_abstraction(self):
        self.assertTrue(inspect.isabstract(MolRepr))
        self.assertTrue(inspect.isabstract(StringMol))
        for mol in [SmilesMol("CCO"), RDKitMol.from_smiles("CCO"), GraphMol.from_smiles("CCO")]:
            self.assertIsInstance(mol, MolRepr)
            self.assertIsNone(mol.validate())

    def test_invalid_text_storage_and_tokenisation(self):
        mol = SmilesMol("C1[", meta={"scores": [1, 2]})
        self.assertEqual(str(mol), "C1[")
        self.assertEqual("".join(mol.tokenise(RegexTokeniser.smiles())), "C1[")
        self.assertEqual(SmilesMol.from_bytes(mol.to_bytes()).text, mol.text)
        copied = mol.to(SmilesMol, strict=True)
        copied.meta["scores"].append(3)
        self.assertEqual(mol.meta["scores"], [1, 2])
        self.assertEqual(copied.text, mol.text)
        with rdBase.BlockLogs():
            self.assertIsNone(mol.to_rdkit())
            with self.assertRaises(ConversionError):
                mol.validate()
            with self.assertRaises(ConversionError):
                mol.to(GraphMol)

    def test_all_conversion_pairs_preserve_supported_chemistry_and_metadata(self):
        for text in ["CCO", "C/C=C/C", "C/C=C\\C", "N[C@@H](C)C(=O)O", "[Na+].[Cl-]", "*CCO"]:
            expected = Chem.MolToSmiles(Chem.MolFromSmiles(text))
            for source_type in [SmilesMol, RDKitMol, GraphMol]:
                source = SmilesMol(text).to(source_type)
                source.meta = {"id": "example", "nested": {"x": [1]}}
                for target_type in [SmilesMol, RDKitMol, GraphMol]:
                    with self.subTest(text=text, source=source_type, target=target_type):
                        target = source.to(target_type)
                        self.assertIsInstance(target, target_type)
                        self.assertEqual(target.to_smiles(), expected)
                        target.meta["nested"]["x"].append(2)
                        self.assertEqual(source.meta["nested"]["x"], [1])

    def test_smiles_construction_preserves_original_text(self):
        mol = SmilesMol("OCC")
        self.assertEqual(mol.text, "OCC")
        self.assertEqual(mol.copy().text, "OCC")
        self.assertEqual(mol.to_smiles(), "CCO")
        self.assertEqual(mol.text, "OCC")

    def test_parser_preserves_explicit_hydrogens_and_rejects_suffixes(self):
        self.assertEqual(SmilesMol("[H]C([H])([H])[H]").to_rdkit().GetNumAtoms(), 5)
        with rdBase.BlockLogs():
            self.assertIsNone(SmilesMol("CCO name").to_rdkit())
            self.assertIsNone(SmilesMol("CCO |atomProp:0.foo.bar|").to_rdkit())

    def test_rdkit_ownership_and_properties(self):
        raw = _rich_mol()
        mol = RDKitMol(raw, meta={"nested": [1]})
        raw.SetProp("source", "changed")
        self.assertEqual(mol.rdkit_mol.GetProp("source"), "example")
        exported = mol.to_rdkit()
        exported.SetProp("source", "exported")
        self.assertEqual(mol.rdkit_mol.GetProp("source"), "example")
        copied = mol.copy()
        copied.rdkit_mol.SetProp("source", "copy")
        copied.meta["nested"].append(2)
        self.assertEqual(mol.meta["nested"], [1])
        self.assertEqual(mol.n_conformers, 2)
        self.assertEqual(mol.coords.shape, (2, mol.n_atoms, 3))
        self.assertEqual(mol.n_heavy_atoms, 4)
        self.assertGreater(mol.n_bonds, 0)

    def test_strict_loss_detection(self):
        for text in ["[13CH3]CO", "[CH3:7]CO", "[CH2]C"]:
            with self.subTest(text=text):
                with self.assertRaises(ConversionError):
                    SmilesMol(text).to(GraphMol, strict=True)
                preserved = SmilesMol(text).to(RDKitMol, strict=True)
                self.assertEqual(preserved.to_smiles(), Chem.MolToSmiles(Chem.MolFromSmiles(text)))
        with self.assertRaises(ConversionError):
            RDKitMol(_rich_mol()).to(SmilesMol, strict=True)
        raw = Chem.MolFromSmiles("CCO")
        raw.SetProp("source", "example")
        with self.assertRaisesRegex(ConversionError, "properties"):
            RDKitMol(raw).to(GraphMol, strict=True)
        self.assertEqual(SmilesMol("CCO").to(GraphMol, strict=True).to_smiles(), "CCO")

    def test_conformers_and_weights_graph_rdkit(self):
        raw = _rich_mol()
        graph = GraphMol.from_rdkit(raw)
        result = graph.to(RDKitMol).to(GraphMol)
        np.testing.assert_allclose(result.coords, graph.coords)
        np.testing.assert_allclose(result.conf_weights, graph.conf_weights)

    def test_strict_conversion_allows_float32_coordinate_rounding(self):
        raw = Chem.MolFromSmiles("CCO")
        conf = Chem.Conformer(3)
        # Values on a CXSMILES formatting boundary must be compared numerically.
        for idx in range(3):
            conf.SetAtomPosition(idx, [1.234549999 + idx, -0.123456789, 0.0])
        raw.AddConformer(conf)
        result = RDKitMol(raw).to(GraphMol, strict=True)
        np.testing.assert_allclose(result.coords[0], raw.GetConformer().GetPositions(), atol=1e-6)

    def test_unsanitised_rdkit_can_be_stored_without_validation(self):
        raw = Chem.MolFromSmiles("C(C)(C)(C)(C)C", sanitize=False)
        mol = RDKitMol.from_bytes(RDKitMol(raw).to_bytes())
        self.assertEqual(mol.n_atoms, 6)
        with rdBase.BlockLogs(), self.assertRaises(ConversionError):
            mol.validate()

    def test_failed_sanitisation_with_weights_returns_none(self):
        graph = GraphMol(
            AtomSet(np.array([6, 1, 1, 1, 1, 1], dtype=np.uint8)),
            BondSet(np.array([[0, i, 1] for i in range(1, 6)], dtype=np.int16)),
            ConfSet(np.zeros((1, 6, 3), dtype=np.float32), weights=np.array([1.0], dtype=np.float32)),
        )
        with rdBase.BlockLogs():
            self.assertIsNone(graph.to_rdkit(sanitise=True))
            with self.assertRaises(ConversionError):
                graph.to(SmilesMol)

    def test_unknown_target_and_format(self):
        with self.assertRaises(TypeError):
            SmilesMol("C").to(str)
        obj = json.loads(SmilesMol("C").to_bytes())
        obj["version"] = 99
        with self.assertRaises(ValueError):
            SmilesMol.from_bytes(json.dumps(obj).encode())
        with self.assertRaises(ValueError):
            RDKitMol.from_bytes(SmilesMol("C").to_bytes())


class TestNativePersistence(unittest.TestCase):
    def test_failed_serialisation_does_not_create_or_overwrite_file(self):
        with tempfile.TemporaryDirectory() as directory:
            for mol in [SmilesMol("C"), RDKitMol.from_smiles("C")]:
                path = Path(directory) / type(mol).__name__
                mol.meta = {"unsupported": object()}
                with self.assertRaises(TypeError):
                    mol.save(path)

                self.assertFalse(path.exists())
                path.write_bytes(b"existing data")
                with self.assertRaises((TypeError, FileExistsError)):
                    mol.save(path)

                self.assertEqual(path.read_bytes(), b"existing data")

    def assert_rich_mol(self, mol):
        raw = mol.rdkit_mol
        self.assertEqual(raw.GetProp("source"), "example")
        self.assertEqual(raw.GetIntProp("number"), 17)
        self.assertEqual(raw.GetProp("_private"), "keep")
        self.assertEqual(raw.GetAtomWithIdx(0).GetDoubleProp("score"), 1.25)
        self.assertEqual(raw.GetAtomWithIdx(0).GetIsotope(), 13)
        self.assertEqual(raw.GetAtomWithIdx(0).GetAtomMapNum(), 7)
        self.assertTrue(raw.GetBondWithIdx(0).GetBoolProp("selected"))
        self.assertEqual(raw.GetNumConformers(), 2)
        self.assertEqual(raw.GetConformer(1).GetIntProp("rank"), 1)

    def test_single_molecule_roundtrips(self):
        with tempfile.TemporaryDirectory() as directory:
            mols = [SmilesMol("invalid text\x00é"), RDKitMol(_rich_mol()), GraphMol.from_smiles("CCO")]
            for idx, mol in enumerate(mols):
                mol.meta = {"nested": {"x": [1, True]}}
                path = Path(directory) / f"{idx}.molito"
                mol.save(path)
                loaded = type(mol).load(path)
                self.assertEqual(loaded.meta, mol.meta)
                if isinstance(mol, SmilesMol):
                    self.assertEqual(loaded.text, mol.text)
                elif isinstance(mol, RDKitMol):
                    self.assert_rich_mol(loaded)
                    np.testing.assert_array_equal(loaded.coords, mol.coords)
                else:
                    self.assertEqual(loaded.to_smiles(), mol.to_smiles())
                with self.assertRaises(FileExistsError):
                    mol.save(path)

    def test_native_shards_keep_strings_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            texts = ["OCC", "C[", "", "invalid\x00é"]
            batch = MolBatch([SmilesMol(text, meta={"id": idx}) for idx, text in enumerate(texts)])
            for columnar in (False, True):
                path = Path(directory) / str(columnar)
                batch.save(path, shard_size=2, columnar_meta=columnar)
                loaded = MolBatch.load(path)
                self.assertEqual([mol.text for mol in loaded], texts)
                np.testing.assert_array_equal(loaded.meta_column("id"), np.arange(4))
                self.assertEqual([mol.text for mol in loaded.subset([3, 0])], [texts[3], texts[0]])
                with self.assertRaises(RuntimeError):
                    batch.save(path)

    def test_native_rdkit_shards_preserve_properties_and_coordinates(self):
        mol = RDKitMol(_rich_mol(), meta={"x": {"y": [1]}})
        with tempfile.TemporaryDirectory() as directory:
            MolBatch([mol, mol.copy()]).save(directory, shard_size=1)
            for loaded in MolBatch.load(directory):
                self.assert_rich_mol(loaded)
                np.testing.assert_array_equal(loaded.coords, mol.coords)
                self.assertEqual(loaded.meta, mol.meta)

    def test_empty_and_invalid_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            batch = MolBatch([], mol_type=SmilesMol)
            batch.save(directory)
            loaded = MolBatch.load(directory)
            self.assertEqual(len(loaded), 0)
            self.assertIs(loaded.mol_type, SmilesMol)
            self.assertEqual(len(loaded.to(GraphMol)), 0)
        with self.assertRaises(ValueError):
            MolBatch([])
        with self.assertRaises(TypeError):
            MolBatch([SmilesMol("C"), RDKitMol.from_smiles("C")])

    def test_batch_conversion_preserves_order_and_reports_failure_index(self):
        batch = MolBatch([SmilesMol("OCC", meta={"id": 0}), SmilesMol("C", meta={"id": 1})])
        graphs = batch.to(GraphMol)
        self.assertIsInstance(graphs, GraphBatch)
        loaded = graphs.to(SmilesMol)
        self.assertEqual([mol.text for mol in loaded], ["CCO", "C"])
        self.assertEqual([mol.meta["id"] for mol in loaded], [0, 1])
        with rdBase.BlockLogs(), self.assertRaisesRegex(ConversionError, "batch index 1"):
            MolBatch([SmilesMol("C"), SmilesMol("C[")]).to(GraphMol)

    def test_corrupt_or_mismatched_shards_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "0.hdf5"
            MolBatch([SmilesMol("C")]).save_hdf5_shard(path)
            with self.assertRaises(ValueError):
                MolBatch.load_hdf5_shard(path, mol_type=RDKitMol)
            with h5py.File(path, "r+") as f:
                f["offsets"][1] = 999
            with self.assertRaisesRegex(ValueError, "offsets"):
                MolBatch.load_hdf5_shard(path)
