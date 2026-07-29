"""Tests that chirality and E/Z bond stereo are preserved through atom reordering and permutation.

This is a critical property of the representation — if stereo is lost or flipped during
canonicalisation or arbitrary permutation, the molecule identity changes silently.
"""

import itertools
import random
import unittest

from rdkit import Chem
from rdkit.Chem import AllChem

from molito.core.bonds import BondSet
from molito.mol.graph import GraphMol


def _canonical_smiles(rdkit_mol):
    """Get canonical SMILES from an RDKit mol, ensuring stereo is assigned."""

    mol = Chem.Mol(rdkit_mol)
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass

    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return Chem.MolToSmiles(mol, canonical=True)


def _roundtrip_smiles(smiles):
    """SMILES -> GraphMol.from_rdkit (with canonicalisation) -> to_rdkit -> canonical SMILES."""

    mol = Chem.MolFromSmiles(smiles)
    graph = GraphMol.from_rdkit(mol, canonicalise=True)
    recovered = graph.to_rdkit()
    return _canonical_smiles(recovered)


class TestEZPreservationThroughCanonicalization(unittest.TestCase):
    """Verify E/Z stereo is preserved through from_rdkit (which does canonical reordering)."""

    def test_e_but2ene(self):
        smi = r"C/C=C/C"
        result = _roundtrip_smiles(smi)
        expected = _canonical_smiles(Chem.MolFromSmiles(smi))
        self.assertEqual(result, expected)

    def test_z_but2ene(self):
        smi = r"C/C=C\C"
        result = _roundtrip_smiles(smi)
        expected = _canonical_smiles(Chem.MolFromSmiles(smi))
        self.assertEqual(result, expected)

    def test_e_and_z_are_different(self):
        e_smi = r"C/C=C/C"
        z_smi = r"C/C=C\C"
        e_result = _roundtrip_smiles(e_smi)
        z_result = _roundtrip_smiles(z_smi)
        self.assertNotEqual(e_result, z_result)

    def test_stilbene_e(self):
        smi = r"C(/C=C/c1ccccc1)c1ccccc1"
        result = _roundtrip_smiles(smi)
        expected = _canonical_smiles(Chem.MolFromSmiles(smi))
        self.assertEqual(result, expected)

    def test_stilbene_z(self):
        smi = r"C(/C=C\c1ccccc1)c1ccccc1"
        result = _roundtrip_smiles(smi)
        expected = _canonical_smiles(Chem.MolFromSmiles(smi))
        self.assertEqual(result, expected)

    def test_multiple_double_bonds(self):
        smi = r"C/C=C/C=C/C"
        result = _roundtrip_smiles(smi)
        expected = _canonical_smiles(Chem.MolFromSmiles(smi))
        self.assertEqual(result, expected)

    def test_mixed_ez(self):
        smi = r"C/C=C\C=C/C"
        result = _roundtrip_smiles(smi)
        expected = _canonical_smiles(Chem.MolFromSmiles(smi))
        self.assertEqual(result, expected)

    def test_ring_with_ez(self):
        smi = r"C/C=C\CC/C=C\C"
        result = _roundtrip_smiles(smi)
        expected = _canonical_smiles(Chem.MolFromSmiles(smi))
        self.assertEqual(result, expected)


class TestChiralityPreservationThroughCanonicalization(unittest.TestCase):
    """Verify tetrahedral chirality is preserved through from_rdkit (canonical reordering)."""

    def test_cw_chirality(self):
        smi = "[C@@H](F)(Cl)Br"
        result = _roundtrip_smiles(smi)
        expected = _canonical_smiles(Chem.MolFromSmiles(smi))
        self.assertEqual(result, expected)

    def test_ccw_chirality(self):
        smi = "[C@H](F)(Cl)Br"
        result = _roundtrip_smiles(smi)
        expected = _canonical_smiles(Chem.MolFromSmiles(smi))
        self.assertEqual(result, expected)

    def test_cw_and_ccw_are_different(self):
        cw_smi = "[C@@H](F)(Cl)Br"
        ccw_smi = "[C@H](F)(Cl)Br"
        cw_result = _roundtrip_smiles(cw_smi)
        ccw_result = _roundtrip_smiles(ccw_smi)
        self.assertNotEqual(cw_result, ccw_result)

    def test_alanine(self):
        smi = "N[C@@H](C)C(=O)O"
        result = _roundtrip_smiles(smi)
        expected = _canonical_smiles(Chem.MolFromSmiles(smi))
        self.assertEqual(result, expected)

    def test_multiple_chiral_centres(self):
        smi = "[C@@H](F)(Cl)[C@H](Br)I"
        result = _roundtrip_smiles(smi)
        expected = _canonical_smiles(Chem.MolFromSmiles(smi))
        self.assertEqual(result, expected)

    def test_chirality_with_ring(self):
        smi = "O[C@@H]1CC[C@@H](O)CC1"
        result = _roundtrip_smiles(smi)
        expected = _canonical_smiles(Chem.MolFromSmiles(smi))
        self.assertEqual(result, expected)


class TestStereoPreservationThroughPermutation(unittest.TestCase):
    """Verify that stereo is preserved when atoms are explicitly permuted (not just canonicalised)."""

    def _permute_and_recover_smiles(self, smiles, indices):
        mol = Chem.MolFromSmiles(smiles)
        graph = GraphMol.from_rdkit(mol, canonicalise=False)
        permuted = graph.permute(indices)
        recovered = permuted.to_rdkit()
        return _canonical_smiles(recovered)

    def test_ez_survives_reverse_permutation(self):
        smi = r"C/C=C/C"
        expected = _canonical_smiles(Chem.MolFromSmiles(smi))
        result = self._permute_and_recover_smiles(smi, [3, 2, 1, 0])
        self.assertEqual(result, expected)

    def test_chirality_survives_reverse_permutation(self):
        smi = "[C@@H](F)(Cl)Br"
        expected = _canonical_smiles(Chem.MolFromSmiles(smi))
        result = self._permute_and_recover_smiles(smi, [3, 2, 1, 0])
        self.assertEqual(result, expected)

    def test_ez_survives_all_permutations_small_mol(self):
        """Exhaustively test all 24 permutations of a 4-atom E/Z molecule."""

        smi = r"C/C=C/C"
        expected = _canonical_smiles(Chem.MolFromSmiles(smi))

        for perm in itertools.permutations(range(4)):
            result = self._permute_and_recover_smiles(smi, list(perm))
            self.assertEqual(result, expected, f"E/Z lost with permutation {perm}")

    def test_chirality_survives_all_permutations_small_mol(self):
        """Exhaustively test all 24 permutations of a 4-atom chiral molecule."""

        smi = "[C@@H](F)(Cl)Br"
        expected = _canonical_smiles(Chem.MolFromSmiles(smi))

        for perm in itertools.permutations(range(4)):
            result = self._permute_and_recover_smiles(smi, list(perm))
            self.assertEqual(result, expected, f"Chirality lost with permutation {perm}")

    def test_z_survives_all_permutations_small_mol(self):
        smi = r"C/C=C\C"
        expected = _canonical_smiles(Chem.MolFromSmiles(smi))

        for perm in itertools.permutations(range(4)):
            result = self._permute_and_recover_smiles(smi, list(perm))
            self.assertEqual(result, expected, f"Z stereo lost with permutation {perm}")

    def test_ccw_chirality_survives_all_permutations(self):
        smi = "[C@H](F)(Cl)Br"
        expected = _canonical_smiles(Chem.MolFromSmiles(smi))

        for perm in itertools.permutations(range(4)):
            result = self._permute_and_recover_smiles(smi, list(perm))
            self.assertEqual(result, expected, f"CCW chirality lost with permutation {perm}")


class TestStereoPreservationThroughOrderByBonds(unittest.TestCase):
    """Verify stereo is preserved specifically through order_by_bonds (the canonical reordering step)."""

    def _order_and_recover_smiles(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        graph = GraphMol.from_rdkit(mol, canonicalise=False)
        ordered = graph.order_by_bonds()
        recovered = ordered.to_rdkit()
        return _canonical_smiles(recovered)

    def test_ez_through_order_by_bonds(self):
        smi = r"C/C=C/C"
        expected = _canonical_smiles(Chem.MolFromSmiles(smi))
        result = self._order_and_recover_smiles(smi)
        self.assertEqual(result, expected)

    def test_z_through_order_by_bonds(self):
        smi = r"C/C=C\C"
        expected = _canonical_smiles(Chem.MolFromSmiles(smi))
        result = self._order_and_recover_smiles(smi)
        self.assertEqual(result, expected)

    def test_chirality_through_order_by_bonds(self):
        smi = "[C@@H](F)(Cl)Br"
        expected = _canonical_smiles(Chem.MolFromSmiles(smi))
        result = self._order_and_recover_smiles(smi)
        self.assertEqual(result, expected)

    def test_double_order_by_bonds_is_idempotent(self):
        smi = r"C/C=C/C"
        mol = Chem.MolFromSmiles(smi)
        graph = GraphMol.from_rdkit(mol, canonicalise=False)
        once = graph.order_by_bonds()
        twice = once.order_by_bonds()

        smi1 = _canonical_smiles(once.to_rdkit())
        smi2 = _canonical_smiles(twice.to_rdkit())
        self.assertEqual(smi1, smi2)


class TestStereoPreservationWithConformers(unittest.TestCase):
    """Verify stereo is preserved when the molecule has 3D coordinates."""

    def test_ez_with_3d(self):
        smi = r"C/C=C/C"
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)

        graph = GraphMol.from_rdkit(mol, canonicalise=True)
        recovered = graph.to_rdkit()
        result = _canonical_smiles(recovered)
        expected = _canonical_smiles(mol)
        self.assertEqual(result, expected)

    def test_chirality_with_3d(self):
        smi = "[C@@H](F)(Cl)Br"
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)

        graph = GraphMol.from_rdkit(mol, canonicalise=True)
        recovered = graph.to_rdkit()
        result = _canonical_smiles(recovered)
        expected = _canonical_smiles(mol)
        self.assertEqual(result, expected)


class TestStereoSurvivesDisagreeingCoords(unittest.TestCase):
    """Regression for SAIR-style ingestion: when a mol carries SMILES-derived stereo
    but has 3D coords that don't faithfully encode that stereo (e.g. ML-generated
    poses with non-planar double bonds or inverted chirality), the GraphMol roundtrip
    must still preserve the SMILES stereo -- the chirality tags and bond directions
    are the authoritative record, the coords are just an approximate pose annotation.
    """

    def _mol_with_foreign_coords(self, tmpl_smi, coord_smi):
        """Build a mol whose atom chirality / bond directions come from tmpl_smi,
        but whose 3D coords come from embedding coord_smi. tmpl_smi and coord_smi
        must have the same heavy-atom graph (differ only in stereo)."""

        template = Chem.MolFromSmiles(tmpl_smi)
        coord_mol = Chem.AddHs(Chem.MolFromSmiles(coord_smi))
        AllChem.EmbedMolecule(coord_mol, randomSeed=42)
        coord_mol = Chem.RemoveHs(coord_mol)

        assert template.GetNumAtoms() == coord_mol.GetNumAtoms()
        conf = Chem.Conformer(template.GetNumAtoms())
        cc = coord_mol.GetConformer()
        for i in range(template.GetNumAtoms()):
            p = cc.GetAtomPosition(i)
            conf.SetAtomPosition(i, (p.x, p.y, p.z))

        result = Chem.RWMol(template)
        result.RemoveAllConformers()
        result.AddConformer(conf, assignId=True)
        return result.GetMol()

    def _assert_roundtrip_matches_template(self, tmpl_smi, coord_smi):
        mol = self._mol_with_foreign_coords(tmpl_smi, coord_smi)
        expected = _canonical_smiles(Chem.MolFromSmiles(tmpl_smi))
        graph = GraphMol.from_rdkit(mol, canonicalise=True)
        recovered = graph.to_rdkit()
        got = _canonical_smiles(recovered)
        self.assertEqual(got, expected, f"tmpl={tmpl_smi} coords_from={coord_smi}: {got} != {expected}")

    def test_ez_survives_flipped_coords(self):
        self._assert_roundtrip_matches_template(r"C/C=C/C", r"C/C=C\C")

    def test_ez_z_survives_flipped_coords(self):
        self._assert_roundtrip_matches_template(r"C/C=C\C", r"C/C=C/C")

    def test_ez_survives_stereoless_coords(self):
        self._assert_roundtrip_matches_template(r"C/C=C/C", "CC=CC")

    def test_chirality_survives_inverted_coords(self):
        self._assert_roundtrip_matches_template("N[C@@H](C)C(=O)O", "N[C@H](C)C(=O)O")

    def test_chirality_survives_stereoless_coords(self):
        self._assert_roundtrip_matches_template("N[C@@H](C)C(=O)O", "NC(C)C(=O)O")

    def test_stilbene_ez_survives_flipped_coords(self):
        self._assert_roundtrip_matches_template(
            r"C(/C=C/c1ccccc1)c1ccccc1",
            r"C(/C=C\c1ccccc1)c1ccccc1",
        )


class TestStereoSurvivesRandomPermutations(unittest.TestCase):
    """Reordering (via permute or canonical ordering) must preserve stereo for
    molecules with multiple chiral centres, mixed E/Z + chirality, ring stereo,
    and aromatic substituents. Covers regression space beyond the 4-atom molecules
    in TestStereoPreservationThroughPermutation."""

    COMPLEX_SMILES = [
        "N[C@@H](C)C(=O)O",
        "O[C@@H]1CC[C@@H](O)CC1",
        "C[C@H]([C@@H](O)C)O",
        r"C/C=C/[C@H](N)C",
        r"C/C=C\[C@@H](F)C(=O)O",
        r"N[C@@H](C)/C=C/[C@H](O)C",
        "CC(=O)N[C@@H](CC1=CC=CC=C1)C(=O)O",
        "OC(=O)[C@@H](N)Cc1c[nH]c2ccccc12",
        r"C/C=C/C1CCCCC1",
        "C[C@H](N)C(=O)N[C@H](Cc1ccccc1)C(=O)O",
    ]

    N_PERMS_PER_MOL = 20

    def test_random_permutations_preserve_stereo(self):
        rng = random.Random(0)
        for smi in self.COMPLEX_SMILES:
            mol = Chem.MolFromSmiles(smi)
            n_atoms = mol.GetNumAtoms()
            expected = _canonical_smiles(mol)

            for _ in range(self.N_PERMS_PER_MOL):
                perm = list(range(n_atoms))
                rng.shuffle(perm)

                graph = GraphMol.from_rdkit(mol, canonicalise=False)
                recovered = graph.permute(perm).to_rdkit()
                got = _canonical_smiles(recovered)
                self.assertEqual(got, expected, f"{smi} perm={perm}: {got} != {expected}")

    def test_canonicalise_true_matches_canonicalise_false(self):
        """Both code paths should produce the same stereo-preserving result."""

        for smi in self.COMPLEX_SMILES:
            mol = Chem.MolFromSmiles(smi)
            expected = _canonical_smiles(mol)

            got_t = _canonical_smiles(GraphMol.from_rdkit(mol, canonicalise=True).to_rdkit())
            got_f = _canonical_smiles(GraphMol.from_rdkit(mol, canonicalise=False).to_rdkit())

            self.assertEqual(got_t, expected, f"{smi}: canon=True -> {got_t}")
            self.assertEqual(got_f, expected, f"{smi}: canon=False -> {got_f}")


class TestBytesRoundtripPreservesStereo(unittest.TestCase):
    """Verify stereo survives serialisation to bytes and back."""

    def test_ez_bytes_roundtrip(self):
        smi = r"C/C=C/C"
        mol = Chem.MolFromSmiles(smi)
        graph = GraphMol.from_rdkit(mol)
        restored = GraphMol.from_bytes(graph.to_bytes())
        result = _canonical_smiles(restored.to_rdkit())
        expected = _canonical_smiles(mol)
        self.assertEqual(result, expected)

    def test_chirality_bytes_roundtrip(self):
        smi = "[C@@H](F)(Cl)Br"
        mol = Chem.MolFromSmiles(smi)
        graph = GraphMol.from_rdkit(mol)
        restored = GraphMol.from_bytes(graph.to_bytes())
        result = _canonical_smiles(restored.to_rdkit())
        expected = _canonical_smiles(mol)
        self.assertEqual(result, expected)

    def test_e_z_distinguished_after_bytes(self):
        e_mol = Chem.MolFromSmiles(r"C/C=C/C")
        z_mol = Chem.MolFromSmiles(r"C/C=C\C")

        e_graph = GraphMol.from_rdkit(e_mol)
        z_graph = GraphMol.from_rdkit(z_mol)

        e_restored = GraphMol.from_bytes(e_graph.to_bytes())
        z_restored = GraphMol.from_bytes(z_graph.to_bytes())

        e_smi = _canonical_smiles(e_restored.to_rdkit())
        z_smi = _canonical_smiles(z_restored.to_rdkit())
        self.assertNotEqual(e_smi, z_smi)

    def test_cw_ccw_distinguished_after_bytes(self):
        cw_mol = Chem.MolFromSmiles("[C@@H](F)(Cl)Br")
        ccw_mol = Chem.MolFromSmiles("[C@H](F)(Cl)Br")

        cw_graph = GraphMol.from_rdkit(cw_mol)
        ccw_graph = GraphMol.from_rdkit(ccw_mol)

        cw_restored = GraphMol.from_bytes(cw_graph.to_bytes())
        ccw_restored = GraphMol.from_bytes(ccw_graph.to_bytes())

        cw_smi = _canonical_smiles(cw_restored.to_rdkit())
        ccw_smi = _canonical_smiles(ccw_restored.to_rdkit())
        self.assertNotEqual(cw_smi, ccw_smi)


class TestBondRowOrderInvariant(unittest.TestCase):
    """Pins down the structural invariant that chirality preservation transitively relies on:
    `BondSet.permute_atoms` must preserve the relative row order of the bond array. The stored
    chirality is just a CW/CCW label whose meaning depends on the order RDKit encounters a chiral
    atom's neighbours during `mol_from_atoms` (which iterates bonds in row order). If this
    invariant ever breaks, the all-permutations chirality tests will start flapping in
    hard-to-debug ways — this test isolates the cause."""

    def _bond_row_keys(self, bonds: BondSet, perm: list[int]) -> list[tuple]:
        """Return the (start, end, type) tuples each row maps to under `perm`, in row order."""

        index_map = {old: new for new, old in enumerate(perm)}
        rows = []
        for start, end, btype in bonds.bonds.tolist():
            if start in index_map and end in index_map:
                rows.append((index_map[start], index_map[end], btype))

        return rows

    def test_row_order_preserved_under_reverse_permutation(self):
        smi = "N[C@@H](C)C(=O)O"
        mol = Chem.MolFromSmiles(smi)
        graph = GraphMol.from_rdkit(mol, canonicalise=False)
        perm = list(range(graph.n_atoms))[::-1]

        expected_rows = self._bond_row_keys(graph.bonds, perm)
        permuted = graph.bonds.permute_atoms(perm)
        actual_rows = [tuple(row) for row in permuted.bonds.tolist()]
        self.assertEqual(actual_rows, expected_rows)

    def test_row_order_preserved_under_random_permutations(self):
        rng = random.Random(0)
        smiles_list = [
            "N[C@@H](C)C(=O)O",
            "C[C@H]([C@@H](O)C)O",
            r"N[C@@H](C)/C=C/[C@H](O)C",
            "OC(=O)[C@@H](N)Cc1c[nH]c2ccccc12",
        ]
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            graph = GraphMol.from_rdkit(mol, canonicalise=False)
            n = graph.n_atoms

            for _ in range(10):
                perm = list(range(n))
                rng.shuffle(perm)
                expected_rows = self._bond_row_keys(graph.bonds, perm)
                permuted = graph.bonds.permute_atoms(perm)
                actual_rows = [tuple(row) for row in permuted.bonds.tolist()]
                self.assertEqual(actual_rows, expected_rows, f"{smi} perm={perm}")

    def test_row_order_preserved_under_atom_subset(self):
        """Filtering atoms (e.g. remove_hs) must keep surviving bonds in their original relative order."""

        smi = "[C@@H](F)(Cl)Br"
        mol = Chem.AddHs(Chem.MolFromSmiles(smi))
        graph = GraphMol.from_rdkit(mol, canonicalise=False)

        keep = [i for i, a in enumerate(graph.atomics.tolist()) if a != 1]
        expected_rows = self._bond_row_keys(graph.bonds, keep)
        permuted = graph.bonds.permute_atoms(keep)
        actual_rows = [tuple(row) for row in permuted.bonds.tolist()]
        self.assertEqual(actual_rows, expected_rows)


class TestChiralityWithSubsetAndHs(unittest.TestCase):
    """Gaps in coverage: chirality survival when explicit Hs are added/removed, and when
    multiple chiral centres are simultaneously permuted in different directions."""

    def test_chirality_survives_remove_hs(self):
        smi = "N[C@@H](C)C(=O)O"
        mol = Chem.AddHs(Chem.MolFromSmiles(smi))
        graph = GraphMol.from_rdkit(mol, canonicalise=False)
        stripped = graph.remove_hs()
        recovered = stripped.to_rdkit()
        result = _canonical_smiles(recovered)
        expected = _canonical_smiles(Chem.MolFromSmiles(smi))
        self.assertEqual(result, expected)

    def test_chirality_survives_remove_hs_then_permute(self):
        rng = random.Random(0)
        smi = "C[C@H]([C@@H](O)C)O"
        mol = Chem.AddHs(Chem.MolFromSmiles(smi))
        graph = GraphMol.from_rdkit(mol, canonicalise=False).remove_hs()
        expected = _canonical_smiles(Chem.MolFromSmiles(smi))

        for _ in range(20):
            perm = list(range(graph.n_atoms))
            rng.shuffle(perm)
            recovered = graph.permute(perm).to_rdkit()
            self.assertEqual(_canonical_smiles(recovered), expected, f"perm={perm}")

    def test_two_chiral_centres_independent_permutations(self):
        """Both centres must round-trip correctly across many random permutations."""

        rng = random.Random(0)
        for smi in ["C[C@H]([C@@H](O)C)O", "N[C@@H](C)[C@H](O)C(=O)O"]:
            mol = Chem.MolFromSmiles(smi)
            graph = GraphMol.from_rdkit(mol, canonicalise=False)
            expected = _canonical_smiles(mol)

            for _ in range(30):
                perm = list(range(graph.n_atoms))
                rng.shuffle(perm)
                recovered = graph.permute(perm).to_rdkit()
                self.assertEqual(_canonical_smiles(recovered), expected, f"{smi} perm={perm}")

    def test_chirality_survives_double_permute(self):
        """Chaining two permutations must compose correctly (no cumulative drift)."""

        rng = random.Random(0)
        smi = r"N[C@@H](C)/C=C/[C@H](O)C"
        mol = Chem.MolFromSmiles(smi)
        graph = GraphMol.from_rdkit(mol, canonicalise=False)
        expected = _canonical_smiles(mol)
        n = graph.n_atoms

        for _ in range(20):
            perm1 = list(range(n))
            perm2 = list(range(n))
            rng.shuffle(perm1)
            rng.shuffle(perm2)
            recovered = graph.permute(perm1).permute(perm2).to_rdkit()
            self.assertEqual(_canonical_smiles(recovered), expected, f"perms={perm1},{perm2}")


class TestCleanStereo(unittest.TestCase):
    """Verify clean_stereo=True strips ghost tetrahedral tags but keeps real CIP centres."""

    def _ghost_chir_mol(self):
        # Build a mol with a CHI_TETRAHEDRAL tag manually placed on an atom that isn't
        # a real CIP centre. Simulates how the GEOM-drugs pickled mols come in: tags
        # got applied by `AssignStereochemistryFrom3D` (or were drawn) but the chemistry
        # can't actually resolve them. Central C has two identical ethyl substituents.
        mol = Chem.MolFromSmiles("CCC(CC)O")
        self.assertIsNotNone(mol)
        # Find the central C (idx 2) and set a tetrahedral tag on it.
        mol.GetAtomWithIdx(2).SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
        tags_before = [a.GetChiralTag() for a in mol.GetAtoms()]
        self.assertTrue(any(t != Chem.ChiralType.CHI_UNSPECIFIED for t in tags_before))
        return mol

    def test_atomset_clean_stereo_strips_ghost_tags(self):
        from molito.core.atoms import AtomSet

        mol = self._ghost_chir_mol()
        # Without cleaning: molito sees the ghost tag.
        raw = AtomSet.from_rdkit(mol)
        self.assertGreater(int((raw.chirality != 0).sum()), 0)
        # With cleaning: all chirality should be zeroed (no CIP centre survives).
        cleaned = AtomSet.from_rdkit(mol, clean_stereo=True)
        self.assertEqual(int((cleaned.chirality != 0).sum()), 0)

    def test_graphmol_clean_stereo_strips_ghost_tags(self):
        mol = self._ghost_chir_mol()
        raw_g = GraphMol.from_rdkit(mol)
        clean_g = GraphMol.from_rdkit(mol, clean_stereo=True)
        self.assertGreater(int((raw_g.atoms.chirality != 0).sum()), 0)
        self.assertEqual(int((clean_g.atoms.chirality != 0).sum()), 0)

    def test_clean_stereo_preserves_real_chirality(self):
        # (R)-alanine: the central C IS a real CIP centre, should NOT be cleaned away.
        mol = Chem.MolFromSmiles("N[C@@H](C)C(=O)O")
        clean_g = GraphMol.from_rdkit(mol, clean_stereo=True)
        self.assertEqual(int((clean_g.atoms.chirality != 0).sum()), 1)

    def test_clean_stereo_does_not_mutate_input(self):
        mol = self._ghost_chir_mol()
        tags_before = [a.GetChiralTag() for a in mol.GetAtoms()]
        _ = GraphMol.from_rdkit(mol, clean_stereo=True)
        tags_after = [a.GetChiralTag() for a in mol.GetAtoms()]
        self.assertEqual(tags_before, tags_after)

    def test_clean_stereo_strips_ghost_bond_dir(self):
        # tert-butyl on a double bond — E/Z impossible because tBu has 3 identical Me.
        # Source SMILES `/` would set a BondDir but cleanIt should strip it.
        mol = Chem.MolFromSmiles(r"C/C=C(/C)C")  # ambiguous-stereo double bond
        if mol is None:
            self.skipTest("RDKit failed to parse the test SMILES")
        clean_g = GraphMol.from_rdkit(mol, clean_stereo=True)
        # No directional encodings should survive in the bond array.
        from molito.core.bonds import _DIRECTIONAL_ENCODING_MASK

        directional = _DIRECTIONAL_ENCODING_MASK[clean_g.bonds.types].any()
        self.assertFalse(bool(directional))
