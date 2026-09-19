import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import h5py
import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem

from molito.arrays import one_hot_encode
from molito.core import AtomSet, BondSet, ConfSet
from molito.core.vocab import BondVocab
from molito.geometry.common import _calc_weights
from molito.geometry.mmff import calc_energy_mmff, optimise_mol_mmff
from molito.geometry.xtb import _make_xtb_calculator, optimise_mol_xtb
from molito.mol import ComplexBatch, GraphBatch, Protein, ProteinBatch
from molito.mol.interactions import Interaction, InteractionSet
from molito.tokenise import RegexTokeniser


class TestArrayRegressions(unittest.TestCase):
    def test_individual_conformer_shifts_and_unweighted_permutation(self):
        coords = np.arange(18).reshape(2, 3, 3).astype(float)
        confs = ConfSet(coords)
        shifts = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        np.testing.assert_allclose(confs.shift(shifts).coords, coords + np.array(shifts)[:, None, :])
        permuted = confs.permute_confs([1, 0])
        np.testing.assert_array_equal(permuted.coords, coords[[1, 0]])
        self.assertIsNone(permuted.weights)

    def test_conformer_array_roundtrip_retains_ensemble_shape(self):
        confs = [ConfSet(np.arange(18).reshape(2, 3, 3)), ConfSet(np.arange(24).reshape(3, 2, 4)[:, :, :3])]
        restored = ConfSet.confs_from_arrays(ConfSet.arrays_from_confs(confs))
        for expected, actual in zip(confs, restored, strict=True):
            np.testing.assert_array_equal(actual.coords, expected.coords)

    def test_invalid_chirality_shape_and_weights(self):
        with self.assertRaises(RuntimeError):
            AtomSet(np.array([6, 8]), chirality=np.array([0]))
        for weights in [[-1, 2], [np.nan, 1], [np.inf, 1]]:
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                ConfSet(np.zeros((2, 3, 3)), weights=np.array(weights))

    def test_invalid_indices_do_not_wrap(self):
        for indices in [np.array([-1]), np.array([10000]), np.array([1.2])]:
            with self.subTest(indices=indices):
                with self.assertRaises(ValueError):
                    one_hot_encode(indices, 4)
                with self.assertRaises(ValueError):
                    BondVocab.build().resolve_types(indices)

    def test_inline_regex_flags_and_literal_precedence(self):
        tokeniser = RegexTokeniser("(?i)Cl", extra_tokens=["ClCl"])
        self.assertEqual(tokeniser.split("ClClclC"), ["ClCl", "cl", "C"])


class TestPersistenceRegressions(unittest.TestCase):
    def test_in_memory_batch_subset_and_padded_bonds(self):
        batch = GraphBatch.from_smiles(["CCO", "C"])
        self.assertEqual(batch.subset([0])[0].to_smiles(), "CCO")
        self.assertEqual(batch.bonds.shape, (2, 2, 3))
        np.testing.assert_array_equal(batch.bonds[0], batch[0].bonds.bonds)

    def test_legacy_atoms_without_chirality(self):
        with tempfile.TemporaryDirectory() as tmp:
            GraphBatch.from_smiles(["CCO"]).save(tmp)
            path = next(Path(tmp).glob("*.hdf5"))
            with h5py.File(path, "r+") as f:
                del f["atoms/chirality"]
            for materialise in [True, False]:
                batch = GraphBatch.load(tmp, materialise=materialise)
                try:
                    np.testing.assert_array_equal(batch[0].atoms.chirality, np.zeros(3))
                    self.assertEqual(batch[0].to_smiles(), "CCO")
                finally:
                    batch.close_hdf5()

    def test_protein_shard_limit_includes_final_shard(self):
        atoms = AtomSet(
            np.array([6, 7]),
            res_names=np.array(["ALA", "ALA"]),
            atom_names=np.array(["CA", "N"]),
            res_ids=np.array([1, 1]),
        )
        protein = Protein(atoms, BondSet(np.array([[0, 1, 1]])), ConfSet(np.zeros((1, 2, 3))))
        with tempfile.TemporaryDirectory() as tmp:
            ProteinBatch([protein, protein]).save(tmp, shard_size=np.int64(1))
            for limit, expected in [(1, 1), (2, 2), (10, 2), (np.int64(1), 1)]:
                loaded = ProteinBatch.load(tmp, n_shards=limit)
                try:
                    self.assertEqual(len(loaded), expected)
                finally:
                    loaded.close_hdf5()

    def test_failed_load_closes_file_for_every_batch_type(self):
        real_file = h5py.File
        opened = []

        class TrackedFile(real_file):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                opened.append(self)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.hdf5"
            with real_file(path, "w"):
                pass
            for batch_type in [GraphBatch, ProteinBatch, ComplexBatch]:
                with self.subTest(batch_type=batch_type), patch("h5py.File", TrackedFile):
                    with self.assertRaises((KeyError, RuntimeError)):
                        batch_type.load_hdf5_shard(path)
                    self.assertFalse(opened[-1].id.valid)

    def test_multishard_failure_closes_previous_files(self):
        real_file = h5py.File
        opened = []

        class TrackedFile(real_file):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                opened.append(self)

        with tempfile.TemporaryDirectory() as tmp:
            GraphBatch.from_smiles(["CCO", "C"]).save(tmp, shard_size=1)
            paths = sorted(Path(tmp).glob("*.hdf5"))
            with real_file(paths[-1], "r+") as f:
                del f["atoms"]
            for materialise in [True, False]:
                with patch("h5py.File", TrackedFile), self.assertRaises(KeyError):
                    GraphBatch.load(tmp, materialise=materialise)
                self.assertTrue(all(not f.id.valid for f in opened))

    def test_interactions_json_and_legacy_pickle_gate(self):
        interactions = InteractionSet([Interaction((0,), (0,), "Hydrophobic")], 1, 1)
        with tempfile.TemporaryDirectory() as tmp, h5py.File(Path(tmp) / "interactions.hdf5", "w") as f:
            group = f.create_group("json")
            InteractionSet.save_to_group([interactions, None, interactions], group)
            with patch("molito.mol.interactions.pickle.loads", side_effect=AssertionError("Unexpected unpickle")):
                restored = InteractionSet.load_from_group(group)
            self.assertIsNone(restored[1])
            self.assertEqual(restored[0]._to_core_repr(), interactions._to_core_repr())
            self.assertEqual(restored[2]._to_core_repr(), interactions._to_core_repr())
            legacy = f.create_group("legacy")
            payload = interactions.to_bytes()
            legacy.create_dataset("data", data=np.frombuffer(payload, dtype=np.uint8))
            legacy.create_dataset("sizes", data=[len(payload)])
            with patch("molito.mol.interactions.pickle.loads") as loads:
                with self.assertRaisesRegex(ValueError, "allow_pickle"):
                    InteractionSet.load_from_group(legacy)
                loads.assert_not_called()
            restored = InteractionSet.load_from_group(legacy, allow_pickle=True)
            self.assertEqual(restored[0]._to_core_repr(), interactions._to_core_repr())
            legacy["sizes"][0] = len(payload) + 1
            with self.assertRaisesRegex(ValueError, "sizes"):
                InteractionSet.load_from_group(legacy, allow_pickle=True)


class TestGeometryRegressions(unittest.TestCase):
    def test_mmff_accepts_noncontiguous_conformer_ids(self):
        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        AllChem.EmbedMolecule(mol, randomSeed=42)
        expected = calc_energy_mmff(mol)
        mol.GetConformer().SetId(7)
        self.assertAlmostEqual(calc_energy_mmff(mol), expected)
        optimised, energy = optimise_mol_mmff(mol, return_energy=True)
        self.assertIsNotNone(optimised)
        self.assertTrue(np.isfinite(energy))

    def test_mmff_unsupported_parameters_are_not_unconverged_success(self):
        mol = Chem.MolFromSmiles("[U]")
        mol.AddConformer(Chem.Conformer(1))
        with rdBase.BlockLogs():
            self.assertIsNone(optimise_mol_mmff(mol, allow_unconverged=True))

    def test_invalid_boltzmann_inputs(self):
        for temp in [0, -1, np.nan, np.inf]:
            with self.subTest(temp=temp), self.assertRaises(ValueError):
                _calc_weights(np.array([0.0, 1.0]), temp)
        for energies in [[], [np.nan], [np.inf]]:
            with self.subTest(energies=energies), self.assertRaises(ValueError):
                _calc_weights(np.array(energies), 300)

    def test_xtb_calculator_charge_spin_and_solvent(self):
        calculator = Mock()
        solvent_enum = object()
        modules = {
            "xtb.interface": SimpleNamespace(Calculator=calculator),
            "xtb.libxtb": SimpleNamespace(VERBOSITY_MUTED=0),
            "xtb.utils": SimpleNamespace(get_solvent=Mock(return_value=solvent_enum)),
        }
        with (
            patch.dict(sys.modules, modules),
            patch("molito.geometry.xtb._get_xtb_method_map", return_value={"test": 1}),
        ):
            _make_xtb_calculator([7], np.zeros((1, 3)), "test", 1.0, 300.0, "water", 1, 2)
            self.assertEqual(calculator.call_args.kwargs, {"charge": 1, "uhf": 2})
            calculator.return_value.set_solvent.assert_called_once_with(solvent_enum)
            modules["xtb.utils"].get_solvent.return_value = None
            with self.assertRaisesRegex(ValueError, "Unknown xTB solvent"):
                _make_xtb_calculator([7], np.zeros((1, 3)), "test", 1.0, 300.0, "invalid", 1, 2)

    def test_xtb_passes_molecular_charge_and_rejects_failed_optimisation(self):
        mol = Chem.AddHs(Chem.MolFromSmiles("[NH4+]"))
        AllChem.EmbedMolecule(mol, randomSeed=42)
        result = SimpleNamespace(success=False, status=1, x=mol.GetConformer().GetPositions().flatten(), fun=-1.0)
        with (
            patch("molito.geometry.xtb._get_xtb_method_map", return_value={"GFN2-xTB": 1}),
            patch("molito.geometry.xtb._make_xtb_calculator") as factory,
            patch("molito.geometry.xtb._xtb_energy_and_gradient", return_value=(-1.0, np.zeros(mol.GetNumAtoms() * 3))),
            patch("scipy.optimize.minimize", return_value=result),
        ):
            self.assertIsNone(optimise_mol_xtb(mol, allow_unconverged=False))
            self.assertEqual(factory.call_args.args[-2:], (1, 0))
            self.assertIsNotNone(optimise_mol_xtb(mol))
            result.status = 2
            self.assertIsNone(optimise_mol_xtb(mol))
            result.success = True
            self.assertIsNotNone(optimise_mol_xtb(mol, uhf=2))
            self.assertEqual(factory.call_args.args[-2:], (1, 2))
