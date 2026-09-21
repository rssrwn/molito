import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from molito import ConversionError, GraphBatch, GraphMol, MolBatch, RDKitMol, SmilesMol
from molito.mol.batch import LazyMolBatch
from molito.tokenise import RegexTokeniser


class CustomSmiles(SmilesMol):
    pass


class TestNativeLazyBatch(unittest.TestCase):
    def setUp(self):
        self.path = Path(self.enterContext(tempfile.TemporaryDirectory())) / "dataset"

    @staticmethod
    def _batch(mol_type):
        if issubclass(mol_type, SmilesMol):
            mols = [mol_type(text) for text in ["CCO", "C[", "", "invalid\x00é"]]
        else:
            mol = Chem.AddHs(Chem.MolFromSmiles("C[C@H](O)F"))
            AllChem.EmbedMultipleConfs(mol, numConfs=2, randomSeed=42)
            mol.SetProp("name", "stereo")
            mol.GetAtomWithIdx(0).SetIntProp("label", 7)
            mol.GetConformers()[0].SetId(7)
            mol.GetConformers()[1].SetId(19)
            mols = [mol_type(mol) for _ in range(4)]

        for idx, mol in enumerate(mols):
            mol.meta = {"id": idx, "label": f"mol-{idx}"}

        return MolBatch(mols)

    def test_payload_reads_are_per_molecule_and_wrappers_can_be_deferred(self):
        getitem = h5py.Dataset.__getitem__
        for mol_type in [SmilesMol, RDKitMol]:
            original = self._batch(mol_type)
            path = self.path / mol_type.__name__
            original.save(path, shard_size=2, columnar_meta=True)

            for materialise in [True, False]:
                reads = []

                def tracked_read(dataset, key, reads=reads):
                    if dataset.name == "/payload" or dataset.name.startswith("/meta/columns/"):
                        reads.append((dataset.name, key))

                    return getitem(dataset, key)

                with (
                    self.subTest(mol_type=mol_type, materialise=materialise),
                    patch.object(h5py.Dataset, "__getitem__", tracked_read),
                    patch.object(mol_type, "_from_lazy", wraps=mol_type._from_lazy) as factory,
                    MolBatch.load(path, materialise=materialise) as loaded,
                ):
                    self.assertEqual(reads, [])
                    self.assertEqual(factory.call_count, 4 if materialise else 0)
                    np.testing.assert_array_equal(loaded.meta_column("id"), np.arange(4))
                    self.assertFalse(any(name == "/payload" for name, _ in reads))
                    self.assertEqual(factory.call_count, 4 if materialise else 0)

                    mol = loaded[2]
                    self.assertFalse(any(name == "/payload" for name, _ in reads))
                    self.assertEqual(factory.call_count, 4 if materialise else 1)
                    if mol_type is SmilesMol:
                        self.assertEqual(mol.text, "")
                    else:
                        self.assertEqual(mol.n_atoms, original[2].n_atoms)
                        mol.rdkit_mol.SetProp("changed", "yes")
                        self.assertEqual(mol.to_rdkit().GetProp("changed"), "yes")

                    payload_reads = [key for name, key in reads if name == "/payload"]
                    self.assertEqual(len(payload_reads), 1)
                    self.assertIsInstance(payload_reads[0], slice)
                    self.assertEqual(payload_reads[0].start, 0)
                    with h5py.File(path / "1.hdf5", "r") as shard:
                        self.assertEqual(payload_reads[0].stop, int(shard["offsets"][1]))
                        self.assertLess(payload_reads[0].stop, len(shard["payload"]))

    def test_identity_mutation_and_text_replacement(self):
        for mol_type in [SmilesMol, RDKitMol]:
            path = self.path / mol_type.__name__
            self._batch(mol_type).save(path)

            for materialise in [True, False]:
                with (
                    self.subTest(mol_type=mol_type, materialise=materialise),
                    MolBatch.load(path, materialise=materialise) as loaded,
                ):
                    mol = loaded[0]
                    self.assertEqual(mol is loaded[0], materialise)
                    if mol_type is SmilesMol:
                        mol.text = "different invalid text"
                        self.assertEqual(loaded[0].text, mol.text if materialise else "CCO")
                    else:
                        mol.rdkit_mol.SetProp("changed", "yes")
                        self.assertEqual(bool(loaded[0].rdkit_mol.HasProp("changed")), materialise)

                    with self.assertRaises(TypeError):
                        mol.meta["id"] = 99

    def test_read_detaches_payload_metadata_and_rdkit_state(self):
        for mol_type in [SmilesMol, RDKitMol]:
            for columnar in [True, False]:
                original = self._batch(mol_type)
                path = self.path / f"{mol_type.__name__}-{columnar}"
                original.save(path, shard_size=2, columnar_meta=columnar)

                for materialise in [True, False]:
                    with self.subTest(mol_type=mol_type, columnar=columnar, materialise=materialise):
                        with MolBatch.load(path, materialise=materialise) as loaded:
                            subset = loaded.subset([3, 0])
                            detached = subset.read()
                            single = loaded[1].read()
                            subset.close_hdf5()
                            self.assertEqual(loaded[0].meta["id"], 0)
                            files = list(loaded._open_fps)

                        self.assertTrue(all(not fp.id.valid for fp in files))
                        self.assertEqual([mol.meta["id"] for mol in detached], [3, 0])
                        self.assertEqual(single.meta["id"], 1)
                        for actual, expected in zip(detached, [original[3], original[0]], strict=True):
                            if mol_type is SmilesMol:
                                self.assertEqual(actual.text, expected.text)
                            else:
                                GraphMol._check_conversion(expected.to_rdkit(), actual.to_rdkit())
                                np.testing.assert_array_equal(actual.coords, expected.coords)

                        detached[0].meta["label"] = "changed"
                        destination = path / f"detached-{materialise}"
                        detached.save(destination)
                        with MolBatch.load(destination) as restored:
                            self.assertEqual(restored[0].meta["label"], "changed")

    def test_unread_payload_requires_open_file(self):
        for mol_type in [SmilesMol, RDKitMol]:
            path = self.path / mol_type.__name__
            self._batch(mol_type).save(path)

            with MolBatch.load(path) as loaded:
                mol = loaded[0]

            with self.subTest(mol_type=mol_type), self.assertRaises((RuntimeError, ValueError)):
                _ = mol.text if mol_type is SmilesMol else mol.n_atoms

    def test_closed_payload_is_not_reported_as_invalid_chemistry(self):
        for mol_type in [SmilesMol, RDKitMol]:
            path = self.path / mol_type.__name__
            self._batch(mol_type).save(path)
            for materialise in [True, False]:
                with MolBatch.load(path, materialise=materialise) as loaded:
                    mol = loaded[0]

                with self.assertRaisesRegex(RuntimeError, "closed HDF5 file"):
                    mol.to_rdkit()

                with self.assertRaisesRegex(ConversionError, "closed HDF5 file"):
                    mol.validate()

                destination = path / f"failed-{materialise}.molito"
                with self.assertRaises(RuntimeError):
                    mol.save(destination)

                self.assertFalse(destination.exists())

    def test_empty_loaded_metadata_is_read_only(self):
        for mol_type, batch_type in [(SmilesMol, MolBatch), (RDKitMol, MolBatch), (GraphMol, GraphBatch)]:
            for columnar in [True, False]:
                path = self.path / f"{mol_type.__name__}-{columnar}"
                batch_type([SmilesMol("C").to(mol_type)]).save(path, columnar_meta=columnar)
                for materialise in [True, False]:
                    with batch_type.load(path, materialise=materialise) as loaded:
                        mol = loaded[0]
                        self.assertEqual(dict(mol.meta), {})
                        with self.assertRaises(TypeError):
                            mol.meta["new"] = "value"

                        detached = loaded.read()

                    detached[0].meta["new"] = "value"
                    self.assertEqual(detached[0].meta["new"], "value")

    def test_indexing_iteration_and_empty_shards(self):
        self.path.mkdir(parents=True)
        MolBatch([], mol_type=SmilesMol).save_hdf5_shard(self.path / "0.hdf5")
        MolBatch([SmilesMol("C"), SmilesMol("N")]).save_hdf5_shard(self.path / "2.hdf5")
        MolBatch([], mol_type=SmilesMol).save_hdf5_shard(self.path / "3.hdf5")
        MolBatch([SmilesMol("O")]).save_hdf5_shard(self.path / "10.hdf5")

        with MolBatch.load(self.path, materialise=False) as loaded:
            self.assertIsInstance(loaded, LazyMolBatch)
            self.assertEqual([mol.text for mol in loaded], ["C", "N", "O"])
            self.assertEqual([mol.text for mol in loaded[::-1]], ["O", "N", "C"])
            self.assertEqual(loaded[np.int64(-1)].text, "O")
            self.assertEqual([mol.text for mol in loaded.subset([2, 0, 2])], ["O", "C", "O"])
            self.assertEqual(len(loaded.subset([]).read()), 0)
            for idx in [3, -4]:
                with self.assertRaises(IndexError):
                    _ = loaded[idx]

            with self.assertRaises(TypeError):
                _ = loaded[1.5]

        with MolBatch.load_hdf5_shard(self.path / "0.hdf5", materialise=False) as loaded:
            self.assertEqual(len(loaded), 0)
            self.assertIs(loaded.read().mol_type, SmilesMol)
            self.assertEqual(loaded.meta_column("missing").size, 0)

    def test_conversion_tokenisation_and_custom_string_subclasses(self):
        self._batch(CustomSmiles).save(self.path, shard_size=2)
        with self.assertRaisesRegex(ValueError, "representation"):
            MolBatch.load(self.path)

        for materialise in [True, False]:
            with MolBatch.load(self.path, mol_type=CustomSmiles, materialise=materialise) as loaded:
                self.assertIs(type(loaded[0]), CustomSmiles)
                tokeniser = RegexTokeniser.smiles()
                self.assertEqual(tokeniser.decode(loaded[1].encode(tokeniser)), "C[")
                self.assertEqual(loaded.subset([0]).to(GraphMol)[0].to_smiles(), "CCO")
                self.assertEqual(loaded.read()[3].text, "invalid\x00é")

                with self.assertRaisesRegex(ConversionError, "batch index 1"):
                    loaded.to(GraphMol)

    def test_failed_load_and_context_exit_close_all_files(self):
        self._batch(SmilesMol).save(self.path, shard_size=2)
        real_file = h5py.File
        opened = []

        class TrackedFile(real_file):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                opened.append(self)

        for materialise in [True, False]:
            with patch("h5py.File", TrackedFile), self.assertRaisesRegex(RuntimeError, "processing"):
                with MolBatch.load(self.path, materialise=materialise):
                    raise RuntimeError("processing")

            self.assertTrue(all(not fp.id.valid for fp in opened))

        with real_file(self.path / "1.hdf5", "r+") as f:
            f["offsets"][-1] = 999999

        for materialise in [True, False]:
            with patch("h5py.File", TrackedFile), self.assertRaisesRegex(ValueError, "offsets"):
                MolBatch.load(self.path, materialise=materialise)

            self.assertTrue(all(not fp.id.valid for fp in opened))

    def test_mixed_representation_shards_are_rejected(self):
        self.path.mkdir(parents=True)
        MolBatch([SmilesMol("C")]).save_hdf5_shard(self.path / "0.hdf5")
        MolBatch([RDKitMol.from_smiles("C")]).save_hdf5_shard(self.path / "1.hdf5")
        for materialise in [True, False]:
            with self.assertRaisesRegex(ValueError, "representation"):
                MolBatch.load(self.path, materialise=materialise)

    def test_invalid_payload_is_only_decoded_when_accessed(self):
        for mol_type in [SmilesMol, RDKitMol]:
            path = self.path / mol_type.__name__
            self._batch(mol_type).save(path)
            with h5py.File(path / "0.hdf5", "r+") as shard:
                start, end = shard["offsets"][:2]
                shard["payload"][int(start) : int(end)] = 255

            for materialise in [True, False]:
                with (
                    self.subTest(mol_type=mol_type, materialise=materialise),
                    MolBatch.load(path, materialise=materialise) as loaded,
                ):
                    self.assertEqual(loaded[0].meta["id"], 0)
                    with self.assertRaises((UnicodeDecodeError, RuntimeError, ValueError)):
                        _ = loaded[0].text if mol_type is SmilesMol else loaded[0].rdkit_mol

                    with self.assertRaises((UnicodeDecodeError, RuntimeError, ValueError)):
                        loaded[0].to_rdkit()

                    if mol_type is SmilesMol:
                        self.assertEqual(loaded[1].text, "C[")
                    else:
                        self.assertGreater(loaded[1].n_atoms, 0)
