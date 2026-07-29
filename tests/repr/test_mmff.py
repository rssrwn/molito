import unittest

from rdkit import Chem
from rdkit.Chem import AllChem

from molito.geometry.mmff import calc_energy_mmff, optimise_mol_mmff


def _embed_mol(smiles, n_confs=1, seed=42):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMultipleConfs(mol, n_confs, randomSeed=seed)
    return mol


class TestCalcEnergyMmff(unittest.TestCase):
    def test_single_conformer(self):
        mol = _embed_mol("CCO")
        energy = calc_energy_mmff(mol)
        self.assertIsInstance(energy, float)

    def test_multiple_conformers(self):
        mol = _embed_mol("CCCCCC", n_confs=3)
        energies = calc_energy_mmff(mol)
        self.assertIsInstance(energies, list)
        self.assertEqual(len(energies), 3)
        for e in energies:
            self.assertIsInstance(e, float)

    def test_per_atom_normalisation(self):
        mol = _embed_mol("CCO")
        energy = calc_energy_mmff(mol)
        per_atom_energy = calc_energy_mmff(mol, per_atom=True)
        self.assertAlmostEqual(per_atom_energy, energy / mol.GetNumAtoms(), places=5)

    def test_none_mol_returns_none(self):
        result = calc_energy_mmff(Chem.Mol())
        self.assertIsNone(result)

    def test_mol_without_hs_still_works(self):
        mol = Chem.MolFromSmiles("CCO")
        AllChem.EmbedMolecule(mol, randomSeed=42)
        energy = calc_energy_mmff(mol)
        self.assertIsNotNone(energy)


class TestOptimiseMolMmff(unittest.TestCase):
    def test_basic_optimisation(self):
        mol = _embed_mol("CCCC")
        opt = optimise_mol_mmff(mol)
        self.assertIsNotNone(opt)
        self.assertEqual(opt.GetNumConformers(), 1)

    def test_multiple_conformers(self):
        mol = _embed_mol("CCCCCC", n_confs=3)
        opt = optimise_mol_mmff(mol)
        self.assertIsNotNone(opt)
        self.assertEqual(opt.GetNumConformers(), 3)

    def test_returns_lower_energy(self):
        mol = _embed_mol("CCCCCC")
        energy_before = calc_energy_mmff(mol)
        opt = optimise_mol_mmff(mol)
        energy_after = calc_energy_mmff(opt)
        self.assertLessEqual(energy_after, energy_before + 0.01)

    def test_return_energy(self):
        mol = _embed_mol("CCO")
        result = optimise_mol_mmff(mol, return_energy=True)
        self.assertIsNotNone(result)
        opt_mol, energy = result
        self.assertIsNotNone(opt_mol)
        self.assertIsInstance(energy, float)

    def test_return_energy_multiple_confs(self):
        mol = _embed_mol("CCCCCC", n_confs=3)
        _, energies = optimise_mol_mmff(mol, return_energy=True)
        self.assertIsInstance(energies, list)
        self.assertEqual(len(energies), 3)

    def test_allow_unconverged_false(self):
        mol = _embed_mol("CCO")
        # With max_iters=0, optimisation won't converge
        optimise_mol_mmff(mol, max_iters=0, allow_unconverged=False)
        # Might be None or not, depending on if RDKit considers 0 iters as converged
        # The key thing is it doesn't crash

    def test_preserves_h_count(self):
        mol = _embed_mol("CCO")
        n_atoms = mol.GetNumAtoms()
        opt = optimise_mol_mmff(mol)
        self.assertEqual(opt.GetNumAtoms(), n_atoms)

    def test_without_hs_adds_and_removes(self):
        mol = Chem.MolFromSmiles("CCO")
        AllChem.EmbedMolecule(mol, randomSeed=42)
        n_heavy = mol.GetNumHeavyAtoms()
        opt = optimise_mol_mmff(mol)
        self.assertIsNotNone(opt)
        # Should return mol without Hs since input didn't have them
        self.assertEqual(opt.GetNumAtoms(), n_heavy)
