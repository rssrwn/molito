"""Live optional xTB integration tests; skipped when xtb-python is not installed."""

import importlib.util
import os
import tempfile
import unittest
from contextlib import chdir
from unittest.mock import Mock, patch

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from molito.geometry import calc_energy_xtb, optimise_mol_xtb
from molito.geometry.xtb import BOHR_PER_ANGSTROM, KCAL_MOL_PER_HARTREE


@unittest.skipUnless(importlib.util.find_spec("xtb") is not None, "xtb-python is not installed")
class TestXTBIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    def setUp(self):
        # GFN-FF writes a topology file into the current working directory.
        tmp = self.enterContext(tempfile.TemporaryDirectory())
        self.enterContext(chdir(tmp))

    @staticmethod
    def _molecule(smiles):
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
            raise RuntimeError("Could not embed test molecule.")
        return mol

    @staticmethod
    def _energy(mol, method, solvent=None):
        from xtb.interface import Calculator, Param
        from xtb.libxtb import VERBOSITY_MUTED
        from xtb.utils import get_solvent

        methods = {"GFN1-xTB": Param.GFN1xTB, "GFN2-xTB": Param.GFN2xTB, "GFN-FF": Param.GFNFF}
        calc = Calculator(
            methods[method],
            np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()]),
            mol.GetConformer().GetPositions() * BOHR_PER_ANGSTROM,
            charge=Chem.GetFormalCharge(mol),
            uhf=sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()),
        )
        calc.set_verbosity(VERBOSITY_MUTED)
        if solvent is not None:
            calc.set_solvent(get_solvent(solvent))
        return calc.singlepoint().get_energy() * KCAL_MOL_PER_HARTREE

    def test_energy_units_change_values_not_optimised_geometry(self):
        mol = self._molecule("O")
        native_energy = calc_energy_xtb(mol, units="hartree")
        self.assertAlmostEqual(KCAL_MOL_PER_HARTREE, 627.509474, places=6)
        self.assertAlmostEqual(calc_energy_xtb(mol), native_energy * KCAL_MOL_PER_HARTREE, places=7)
        self.assertAlmostEqual(calc_energy_xtb(mol, per_atom=True), native_energy * KCAL_MOL_PER_HARTREE / 3, places=7)

        kcal = optimise_mol_xtb(mol, allow_unconverged=False)
        hartree = optimise_mol_xtb(mol, allow_unconverged=False, units="hartree")
        np.testing.assert_allclose(kcal[1:], np.array(hartree[1:]) * KCAL_MOL_PER_HARTREE, atol=1e-7, rtol=0)
        np.testing.assert_allclose(
            kcal[0].GetConformer().GetPositions(), hartree[0].GetConformer().GetPositions(), atol=1e-12, rtol=0
        )

    def test_neutral_charge_radical_and_solvent_energies(self):
        for smiles, solvent in [("CCO", None), ("[NH4+]", "water"), ("[CH3]", None)]:
            with self.subTest(smiles=smiles, solvent=solvent):
                mol = self._molecule(smiles)
                original_coords = mol.GetConformer().GetPositions().copy()
                original = Chem.MolToSmiles(mol)
                energy = calc_energy_xtb(mol, solvent=solvent)
                self.assertAlmostEqual(energy, self._energy(mol, "GFN2-xTB", solvent), places=7)
                self.assertAlmostEqual(calc_energy_xtb(mol, per_atom=True, solvent=solvent), energy / mol.GetNumAtoms())
                result = optimise_mol_xtb(mol, solvent=solvent, allow_unconverged=False)
                self.assertIsNotNone(result)
                optimised, final_energy, initial_energy = result
                self.assertAlmostEqual(initial_energy, self._energy(mol, "GFN2-xTB", solvent), places=7)
                self.assertAlmostEqual(final_energy, self._energy(optimised, "GFN2-xTB", solvent), places=7)
                self.assertAlmostEqual(final_energy, calc_energy_xtb(optimised, solvent=solvent), places=7)
                self.assertLessEqual(final_energy, initial_energy + 1e-8)
                self.assertTrue(np.isfinite(optimised.GetConformer().GetPositions()).all())
                np.testing.assert_array_equal(mol.GetConformer().GetPositions(), original_coords)
                self.assertEqual(Chem.MolToSmiles(optimised), original)

    def test_supported_methods(self):
        for method in ["GFN1-xTB", "GFN2-xTB", "GFN-FF"]:
            with self.subTest(method=method):
                mol = self._molecule("O")
                self.assertAlmostEqual(calc_energy_xtb(mol, method=method), self._energy(mol, method), places=7)
                result = optimise_mol_xtb(mol, method=method, allow_unconverged=False)
                self.assertIsNotNone(result)
                self.assertTrue(np.isfinite(result[1]))
                self.assertLessEqual(result[1], result[2] + 1e-8)

    def test_iteration_limit_is_explicit(self):
        mol = self._molecule("CCO")
        self.assertIsNotNone(optimise_mol_xtb(mol, max_iters=1))
        self.assertIsNone(optimise_mol_xtb(mol, max_iters=1, allow_unconverged=False))

    def test_implicit_hydrogens_and_nonzero_conformer_id(self):
        mol = Chem.RemoveHs(self._molecule("CCO"))
        mol.GetConformer().SetId(7)
        result = optimise_mol_xtb(mol, conf_idx=7, allow_unconverged=False)
        self.assertIsNotNone(result)
        self.assertEqual(result[0].GetNumAtoms(), mol.GetNumAtoms())
        self.assertIsNone(optimise_mol_xtb(mol, conf_idx=999))

    def test_invalid_solvent_fails_without_gas_phase_fallback(self):
        self.assertIsNone(optimise_mol_xtb(self._molecule("O"), solvent="not-a-solvent"))
        self.assertIsNone(calc_energy_xtb(self._molecule("O"), solvent="not-a-solvent"))

    def test_energy_conformer_order_and_failed_conformer(self):
        mol = self._molecule("O")
        first = mol.GetConformer()
        first.SetId(7)
        second = Chem.Conformer(first)
        second.SetId(19)
        second.SetAtomPosition(1, second.GetAtomPosition(1) * 1.1)
        mol.AddConformer(second, assignId=False)
        failed = Chem.Conformer(first)
        failed.SetId(2)
        failed.SetAtomPosition(0, (float("nan"), 0.0, 0.0))
        mol.AddConformer(failed, assignId=False)

        expected = []
        for conf in [first, second]:
            single = Chem.Mol(mol)
            single.RemoveAllConformers()
            single.AddConformer(conf, assignId=True)
            expected.append(self._energy(single, "GFN2-xTB"))

        energies = calc_energy_xtb(mol)
        np.testing.assert_allclose(energies[:2], expected, atol=1e-7, rtol=0)
        self.assertIsNone(energies[2])
        native_energies = calc_energy_xtb(mol, units="hartree")
        np.testing.assert_allclose(native_energies[:2], np.array(expected) / KCAL_MOL_PER_HARTREE, atol=1e-7, rtol=0)
        self.assertIsNone(native_energies[2])
        self.assertEqual([conf.GetId() for conf in mol.GetConformers()], [7, 19, 2])

    def test_energy_adds_missing_hydrogens_without_optimising(self):
        mol = Chem.RemoveHs(self._molecule("CCO"))
        original = mol.GetConformer().GetPositions().copy()
        prepared = Chem.AddHs(mol, addCoords=True)
        expected = self._energy(prepared, "GFN2-xTB")
        self.assertAlmostEqual(calc_energy_xtb(mol), expected, places=7)
        self.assertAlmostEqual(calc_energy_xtb(mol, per_atom=True), expected / prepared.GetNumAtoms(), places=7)
        np.testing.assert_array_equal(mol.GetConformer().GetPositions(), original)
        self.assertEqual(mol.GetNumAtoms(), 3)

    def test_energy_without_coordinates_or_with_invalid_options(self):
        self.assertIsNone(calc_energy_xtb(Chem.MolFromSmiles("CCO")))
        self.assertIsNone(calc_energy_xtb(Chem.Mol()))
        mol = self._molecule("O")
        with self.assertRaisesRegex(ValueError, "method"):
            calc_energy_xtb(mol, method="invalid")
        with self.assertRaisesRegex(ValueError, "uhf"):
            calc_energy_xtb(mol, uhf=-1)


class TestXTBEnergyFailures(unittest.TestCase):
    def test_invalid_units_raise_before_calculation(self):
        mol = Chem.MolFromSmiles("O")
        for function in [calc_energy_xtb, optimise_mol_xtb]:
            with self.subTest(function=function), self.assertRaisesRegex(ValueError, "units"):
                function(mol, units="invalid")

    def test_missing_backend_raises_import_error(self):
        with patch("molito.geometry.xtb._get_xtb_method_map", side_effect=ImportError("xtb-python")):
            with self.assertRaisesRegex(ImportError, "xtb-python"):
                calc_energy_xtb(Chem.MolFromSmiles("O"))

    def test_backend_failures_are_isolated_and_settings_forwarded(self):
        mol = TestXTBIntegration._molecule("[NH4+]")
        mol.AddConformer(Chem.Conformer(mol.GetConformer()), assignId=True)
        calculator = Mock()
        calculator.singlepoint.return_value.get_energy.return_value = -1.0
        with (
            patch("molito.geometry.xtb._get_xtb_method_map", return_value={"GFN2-xTB": 1}),
            patch(
                "molito.geometry.xtb._make_xtb_calculator", side_effect=[RuntimeError("failed"), calculator]
            ) as factory,
        ):
            self.assertEqual(
                calc_energy_xtb(mol, accuracy=0.5, electronic_temperature=400, uhf=2, units="hartree"), [None, -1.0]
            )
            self.assertEqual(factory.call_args.args[2:], ("GFN2-xTB", 0.5, 400, None, 1, 2))

    def test_nonfinite_energy_returns_none(self):
        mol = TestXTBIntegration._molecule("O")
        with (
            patch("molito.geometry.xtb._get_xtb_method_map", return_value={"GFN2-xTB": 1}),
            patch("molito.geometry.xtb._make_xtb_calculator") as factory,
        ):
            factory.return_value.singlepoint.return_value.get_energy.return_value = float("nan")
            self.assertIsNone(calc_energy_xtb(mol))
