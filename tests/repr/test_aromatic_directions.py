"""Regression tests for residual SMILES directions on aromatic ring bonds."""

import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from molito.core.bonds import BondEncoding, BondSet
from molito.mol import GraphBatch, GraphMol


def canonical_smiles(mol):
    mol = Chem.RemoveHs(Chem.Mol(mol))
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return Chem.MolToSmiles(mol)


class TestAromaticDirections(unittest.TestCase):
    def test_benzene_directions_in_both_cleaning_modes(self):
        for direction in ("/", "\\"):
            for clean in (False, True):
                with self.subTest(direction=direction, clean=clean):
                    mol = Chem.MolFromSmiles(f"C1=C{direction}C=C/C=C1")
                    self.assertTrue(
                        any(b.GetIsAromatic() and b.GetBondDir() != Chem.BondDir.NONE for b in mol.GetBonds())
                    )
                    graph = GraphMol.from_rdkit(mol, clean_stereo=clean)
                    self.assertEqual(graph.to_smiles(), "c1ccccc1")
                    self.assertTrue(all(BondEncoding.decode(int(t))[2] is None for t in graph.bonds.types))

    def test_direct_bond_import(self):
        mol = Chem.MolFromSmiles(r"C1=C\C=C/C=C1")
        bonds = BondSet.from_rdkit(mol)
        self.assertEqual(len(bonds.types), 6)
        self.assertEqual(set(bonds.types), {4, 5})

    def test_real_stereo_survives_roundtrip_and_permutation(self):
        smiles = [
            r"C1=C\C=C/C=C1/C=C/C",
            r"C1=C\C=C/C=C1/C=C\C",
            r"C1=C\C=C/C=C1/C=C/C=C\C",
            r"C1=C\C=C/C=C1/C=C/[C@H](F)Cl",
            r"C1=C\C=C/C=C1/C=C/[C@@H](F)Cl",
        ]
        rng = random.Random(7)
        for smi in smiles:
            for explicit_hs in (False, True):
                mol = Chem.MolFromSmiles(smi)
                if explicit_hs:
                    mol = Chem.AddHs(mol)
                expected = canonical_smiles(mol)
                for canonicalise in (False, True):
                    for clean in (False, True):
                        with self.subTest(smi=smi, hs=explicit_hs, canonical=canonicalise, clean=clean):
                            graph = GraphMol.from_rdkit(mol, canonicalise=canonicalise, clean_stereo=clean)
                            self.assertEqual(canonical_smiles(graph.to_rdkit()), expected)
                            for _ in range(10):
                                perm = list(range(graph.n_atoms))
                                rng.shuffle(perm)
                                self.assertEqual(canonical_smiles(graph.permute(perm).to_rdkit()), expected)

    def test_input_and_coordinates_unchanged(self):
        mol = Chem.AddHs(Chem.MolFromSmiles(r"C1=C\C=C/C=C1/C=C/[C@H](F)Cl"))
        self.assertEqual(AllChem.EmbedMolecule(mol, randomSeed=42), 0)
        mol.SetProp("source", "regression")
        before = mol.ToBinary(Chem.PropertyPickleOptions.AllProps)
        graph = GraphMol.from_rdkit(mol)
        self.assertEqual(mol.ToBinary(Chem.PropertyPickleOptions.AllProps), before)
        recovered = graph.to_rdkit()
        # ConfSet stores coordinates as float32; atom order and positions survive
        # up to that existing storage precision.
        np.testing.assert_array_equal(
            recovered.GetConformer().GetPositions(), mol.GetConformer().GetPositions().astype(np.float32)
        )
        self.assertEqual([a.GetAtomicNum() for a in recovered.GetAtoms()], [a.GetAtomicNum() for a in mol.GetAtoms()])
        self.assertEqual([a.GetChiralTag() for a in recovered.GetAtoms()], [a.GetChiralTag() for a in mol.GetAtoms()])

    def test_default_preserves_ghost_atom_tag(self):
        mol = Chem.MolFromSmiles(r"C1=C\C=C/C=C1.CCC(CC)O")
        atom = mol.GetAtomWithIdx(8)
        atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
        graph = GraphMol.from_rdkit(mol)
        self.assertTrue((graph.atoms.chirality != 0).any())
        cleaned = GraphMol.from_rdkit(mol, clean_stereo=True)
        self.assertFalse((cleaned.atoms.chirality != 0).any())

    def test_hdf5_roundtrip_uses_existing_encodings(self):
        mol = Chem.MolFromSmiles(r"C1=C\C=C/C=C1/C=C\C")
        graph = GraphMol.from_rdkit(mol)
        self.assertEqual(BondEncoding.size(), 12)
        self.assertTrue((graph.bonds.types < 12).all())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "poses"
            GraphBatch([graph]).save(path)
            loaded = GraphBatch.load(path)
            self.assertEqual(canonical_smiles(loaded[0].to_rdkit()), canonical_smiles(mol))
