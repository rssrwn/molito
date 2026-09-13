"""RDKit atom numbering must not change the stereo stored by Molito."""

import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
from rdkit import Chem

from molito.mol import GraphBatch, GraphMol


def canonical_smiles(mol):
    mol = Chem.RemoveHs(Chem.Mol(mol))
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return Chem.MolToSmiles(mol)


class TestBondEndpoints(unittest.TestCase):
    def test_atom_numbering_preserves_stereo(self):
        # Includes an alkene, imine, conjugated system, ring-closure directions,
        # and examples from the production poses that lost or inverted stereo.
        smiles = [
            "C/C=C/C",
            "C/C=C\\C",
            "C/C=N/C",
            "C/C=N\\C",
            "C/C=C/C=C\\C",
            "Cc1cccc(/C=C/C(=O)N2CCCCC2=O)c1",
            "CCN(CC)c1ccnc2c1CCC/C2=C\\c1ccccc1",
            "Cc1ccc(S(=O)(=O)/N=c2\\ncn(C)c3ccccc23)cc1",
            "N[C@@H](C)/C=C/[C@H](O)C",
        ]
        rng = random.Random(11)
        for smi in smiles:
            expected = canonical_smiles(Chem.MolFromSmiles(smi))
            for explicit_hs in (False, True):
                mol = Chem.MolFromSmiles(smi)
                if explicit_hs:
                    mol = Chem.AddHs(mol)
                for _ in range(10):
                    permutation = list(range(mol.GetNumAtoms()))
                    rng.shuffle(permutation)
                    renumbered = Chem.RenumberAtoms(mol, permutation)
                    before = renumbered.ToBinary(Chem.PropertyPickleOptions.AllProps)
                    for canonicalise in (False, True):
                        with self.subTest(smi=smi, hs=explicit_hs, canonicalise=canonicalise, order=permutation):
                            graph = GraphMol.from_rdkit(renumbered, canonicalise=canonicalise)
                            self.assertEqual(canonical_smiles(graph.to_rdkit()), expected)
                            self.assertEqual(renumbered.ToBinary(Chem.PropertyPickleOptions.AllProps), before)
                            rng.shuffle(permutation)
                            self.assertEqual(canonical_smiles(graph.permute(permutation).to_rdkit()), expected)

    def test_hdf5_keeps_directed_endpoints_and_stereo(self):
        mol = Chem.RenumberAtoms(Chem.MolFromSmiles("C/C=C/C"), [3, 1, 2, 0])
        graph = GraphMol.from_rdkit(mol)
        self.assertTrue((graph.bonds.indices[:, 0] > graph.bonds.indices[:, 1]).any())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "poses"
            GraphBatch([graph]).save(path)
            for materialise in (False, True):
                loaded = GraphBatch.load(path, materialise=materialise)[0]
                np.testing.assert_array_equal(loaded.bonds.bonds, graph.bonds.bonds)
                self.assertEqual(canonical_smiles(loaded.to_rdkit()), canonical_smiles(mol))
