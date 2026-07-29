"""Tests for the file and string level IO helpers: SMILES and SDF in and out."""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from molito.mol.graph import GraphBatch, GraphMol


class TestGraphMolSmiles(unittest.TestCase):
    def test_from_smiles_basic(self):
        mol = GraphMol.from_smiles("CCO")
        self.assertEqual(mol.n_atoms, 3)
        self.assertEqual(mol.n_conformers, 0)

    def test_from_smiles_matches_from_rdkit(self):
        smi = "N[C@@H](C)C(=O)O"
        direct = GraphMol.from_smiles(smi)
        via_rdkit = GraphMol.from_rdkit(Chem.MolFromSmiles(smi))

        np.testing.assert_array_equal(direct.atomics, via_rdkit.atomics)
        np.testing.assert_array_equal(direct.bond_types, via_rdkit.bond_types)
        self.assertEqual(direct.tokens, via_rdkit.tokens)

    def test_from_smiles_invalid_raises(self):
        with self.assertRaises(ValueError) as ctx:
            GraphMol.from_smiles("not_a_smiles_XYZ")

        self.assertIn("not_a_smiles_XYZ", str(ctx.exception))

    def test_from_smiles_explicit_hs(self):
        implicit = GraphMol.from_smiles("CCO")
        explicit = GraphMol.from_smiles("CCO", explicit_hs=True)
        self.assertEqual(implicit.n_atoms, 3)
        self.assertEqual(explicit.n_atoms, 9)

    def test_from_smiles_canonicalise(self):
        mol = GraphMol.from_smiles("C(O)C", canonicalise=True)
        self.assertEqual(mol.n_atoms, 3)

    def test_roundtrip_preserves_ez(self):
        for smi in [r"C/C=C/C", r"C/C=C\C"]:
            mol = GraphMol.from_smiles(smi)
            expected = Chem.CanonSmiles(smi)
            self.assertEqual(mol.to_smiles(), expected, f"E/Z lost for {smi}")

    def test_roundtrip_preserves_chirality(self):
        for smi in ["N[C@@H](C)C(=O)O", "N[C@H](C)C(=O)O"]:
            mol = GraphMol.from_smiles(smi)
            self.assertEqual(mol.to_smiles(), Chem.CanonSmiles(smi), f"chirality lost for {smi}")

    def test_e_and_z_stay_distinct_through_roundtrip(self):
        e = GraphMol.from_smiles(r"C/C=C/C").to_smiles()
        z = GraphMol.from_smiles(r"C/C=C\C").to_smiles()
        self.assertNotEqual(e, z)

    def test_to_smiles_explicit_hs(self):
        mol = GraphMol.from_smiles("CCO")
        self.assertIn("[H]", mol.to_smiles(explicit_hs=True))


class TestGraphBatchSmiles(unittest.TestCase):
    SMIS = ["CCO", "c1ccccc1", "CC(=O)O"]

    def test_from_smiles_batch(self):
        batch = GraphBatch.from_smiles(self.SMIS)
        self.assertEqual(len(batch), 3)
        self.assertEqual([m.to_smiles() for m in batch], [Chem.CanonSmiles(s) for s in self.SMIS])

    def test_invalid_raises_by_default(self):
        with self.assertRaises(ValueError):
            GraphBatch.from_smiles(["CCO", "garbage_XYZ", "c1ccccc1"])

    def test_skip_invalid(self):
        batch = GraphBatch.from_smiles(["CCO", "garbage_XYZ", "c1ccccc1"], skip_invalid=True)
        self.assertEqual(len(batch), 2)

    def test_empty_input(self):
        self.assertEqual(len(GraphBatch.from_smiles([])), 0)


class TestSdf(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.sdf = self.tmp / "mols.sdf"

        writer = Chem.SDWriter(str(self.sdf))
        for idx, smi in enumerate(["CCO", "c1ccccc1", "N[C@@H](C)C(=O)O"]):
            mol = Chem.AddHs(Chem.MolFromSmiles(smi))
            AllChem.EmbedMolecule(mol, randomSeed=42 + idx)
            mol.SetProp("mol_id", f"m{idx}")
            mol.SetProp("pIC50", str(5.0 + idx))
            writer.write(mol)
        writer.close()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_from_sdf_reads_all_records(self):
        batch = GraphBatch.from_sdf(self.sdf)
        self.assertEqual(len(batch), 3)

    def test_from_sdf_keeps_conformers(self):
        batch = GraphBatch.from_sdf(self.sdf)
        for mol in batch:
            self.assertEqual(mol.n_conformers, 1)
            self.assertEqual(mol.coords.shape, (1, mol.n_atoms, 3))

    def test_from_sdf_keeps_hs_by_default(self):
        batch = GraphBatch.from_sdf(self.sdf)
        self.assertGreater(batch[0].n_atoms, batch[0].n_heavy_atoms)

    def test_from_sdf_remove_hs(self):
        batch = GraphBatch.from_sdf(self.sdf, remove_hs=True)
        self.assertEqual(batch[0].n_atoms, batch[0].n_heavy_atoms)

    def test_from_sdf_reads_props_into_meta(self):
        batch = GraphBatch.from_sdf(self.sdf)
        self.assertEqual(batch[0].meta["mol_id"], "m0")
        self.assertEqual(batch[2].meta["mol_id"], "m2")

    def test_numeric_props_keep_their_type(self):
        batch = GraphBatch.from_sdf(self.sdf)
        self.assertIsInstance(batch[0].meta["pIC50"], float)

    def test_props_can_be_skipped(self):
        batch = GraphBatch.from_sdf(self.sdf, read_props=False)
        self.assertEqual(dict(batch[0].meta), {})

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            GraphBatch.from_sdf(self.tmp / "nope.sdf")

    def test_stereo_survives_sdf_roundtrip(self):
        batch = GraphBatch.from_sdf(self.sdf, remove_hs=True)
        self.assertEqual(batch[2].to_smiles(), Chem.CanonSmiles("N[C@@H](C)C(=O)O"))

    def test_to_sdf_roundtrip(self):
        original = GraphBatch.from_sdf(self.sdf)
        out = self.tmp / "out.sdf"
        original.to_sdf(out)

        reloaded = GraphBatch.from_sdf(out)
        self.assertEqual(len(reloaded), len(original))

        for before, after in zip(original, reloaded, strict=True):
            self.assertEqual(before.n_atoms, after.n_atoms)
            np.testing.assert_array_equal(before.atomics, after.atomics)
            np.testing.assert_allclose(before.coords, after.coords, atol=1e-3)

    def test_to_sdf_writes_meta_as_tags(self):
        batch = GraphBatch.from_sdf(self.sdf)
        out = self.tmp / "tagged.sdf"
        batch.to_sdf(out)

        reloaded = GraphBatch.from_sdf(out)
        self.assertEqual(reloaded[1].meta["mol_id"], "m1")

    def test_to_sdf_without_meta(self):
        batch = GraphBatch.from_sdf(self.sdf)
        out = self.tmp / "untagged.sdf"
        batch.to_sdf(out, write_meta=False)

        reloaded = GraphBatch.from_sdf(out)
        self.assertNotIn("mol_id", dict(reloaded[0].meta))

    def test_sdf_to_hdf5_pipeline(self):
        # The path most users will actually take: SDF in, sharded HDF5 out, read back.
        batch = GraphBatch.from_sdf(self.sdf, remove_hs=True)
        store = self.tmp / "dataset"
        batch.save(store, columnar_meta=True)

        loaded = GraphBatch.load(store)
        self.assertEqual(len(loaded), 3)
        self.assertEqual(loaded.meta_column("mol_id").tolist(), ["m0", "m1", "m2"])
        np.testing.assert_array_equal(loaded[0].atomics, batch[0].atomics)
        loaded.close_hdf5()


class TestTwoDimensionalSdf(unittest.TestCase):
    """SDFs holding 2D depictions rather than real geometry are common. They must load as
    graphs without conformers, not as a collapsed structure and not as an error.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.sdf = self.tmp / "flat.sdf"

        writer = Chem.SDWriter(str(self.sdf))
        for smi in ["CCO", "c1ccccc1"]:
            writer.write(Chem.MolFromSmiles(smi))  # SDWriter lays out 2D coordinates
        writer.close()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_loads_without_error(self):
        batch = GraphBatch.from_sdf(self.sdf)
        self.assertEqual(len(batch), 2)

    def test_no_conformers_kept(self):
        for mol in GraphBatch.from_sdf(self.sdf):
            self.assertEqual(mol.n_conformers, 0)
            self.assertIsNone(mol.coords)

    def test_graph_is_still_correct(self):
        batch = GraphBatch.from_sdf(self.sdf)
        np.testing.assert_array_equal(batch[0].atomics, GraphMol.from_smiles("CCO").atomics)

    def test_still_saveable(self):
        batch = GraphBatch.from_sdf(self.sdf)
        batch.save(self.tmp / "ds")
        loaded = GraphBatch.load(self.tmp / "ds")
        self.assertEqual(len(loaded), 2)
        loaded.close_hdf5()


class TestInvalidSdfRecords(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.sdf = self.tmp / "broken.sdf"

        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        AllChem.EmbedMolecule(mol, randomSeed=7)
        valid = Chem.MolToMolBlock(mol) + "$$$$\n"

        # A record declaring an element that does not exist - RDKit yields None for it
        broken = valid.replace(" C   0", " Zz  0", 1)
        self.sdf.write_text(valid + broken + valid)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_invalid_record_raises_by_default(self):
        with self.assertRaises(ValueError) as ctx:
            GraphBatch.from_sdf(self.sdf)

        self.assertIn("record", str(ctx.exception))

    def test_error_names_the_record_index(self):
        with self.assertRaises(ValueError) as ctx:
            GraphBatch.from_sdf(self.sdf)

        self.assertIn("1", str(ctx.exception))

    def test_skip_invalid_drops_bad_records(self):
        batch = GraphBatch.from_sdf(self.sdf, skip_invalid=True)
        self.assertEqual(len(batch), 2)
