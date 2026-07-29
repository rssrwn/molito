import unittest

import numpy as np

from molito.core.atoms import AtomSet
from molito.core.bonds import BondSet
from molito.core.confs import ConfSet
from molito.mol.complex import BindingComplex
from molito.mol.graph import GraphMol
from molito.mol.interactions import Interaction, InteractionSet
from molito.mol.protein import Protein

# *** Helper functions ***


def create_test_protein(n_atoms: int = 5, n_confs: int = 1) -> Protein:
    """Create a simple test protein."""

    # Cycle through element types and other properties for arbitrary n_atoms
    base_atomics = [6, 7, 8, 6, 7]
    base_charges = [0, 0, -1, 0, 1]
    base_res_names = ["ALA", "ALA", "ALA", "GLY", "GLY"]
    base_atom_names = ["CA", "N", "O", "CA", "N"]
    base_res_ids = [1, 1, 1, 2, 2]

    atomics = np.array([base_atomics[i % len(base_atomics)] for i in range(n_atoms)])
    charges = np.array([base_charges[i % len(base_charges)] for i in range(n_atoms)])
    res_names = np.array([base_res_names[i % len(base_res_names)] for i in range(n_atoms)])
    atom_names = np.array([base_atom_names[i % len(base_atom_names)] for i in range(n_atoms)])
    res_ids = np.array([base_res_ids[i % len(base_res_ids)] for i in range(n_atoms)])

    atoms = AtomSet(atomics, charges=charges, res_names=res_names, atom_names=atom_names, res_ids=res_ids)

    empty_bonds = np.array([]).reshape(0, 3).astype(np.int32)
    bonds_arr = np.array([[i, i + 1, 1] for i in range(n_atoms - 1)]) if n_atoms > 1 else empty_bonds
    bonds = BondSet(bonds_arr)

    coords = np.random.rand(n_confs, n_atoms, 3).astype(np.float32) * 10
    confs = ConfSet(coords)

    return Protein(atoms, bonds, confs)


def create_test_ligand(n_atoms: int = 4, n_confs: int = 1) -> GraphMol:
    """Create a simple test ligand."""

    base_atomics = [6, 6, 8, 7]
    base_charges = [0, -1, 0, 1]

    atomics = np.array([base_atomics[i % len(base_atomics)] for i in range(n_atoms)])
    charges = np.array([base_charges[i % len(base_charges)] for i in range(n_atoms)])
    atoms = AtomSet(atomics, charges=charges)

    empty_bonds = np.array([]).reshape(0, 3).astype(np.int32)
    bonds_arr = np.array([[i, i + 1, 1] for i in range(n_atoms - 1)]) if n_atoms > 1 else empty_bonds
    bonds = BondSet(bonds_arr)

    coords = np.random.rand(n_confs, n_atoms, 3).astype(np.float32) * 10
    confs = ConfSet(coords)

    return GraphMol(atoms, bonds, confs=confs)


def create_test_interactions(n_protein: int, n_ligand: int) -> InteractionSet:
    """Create a simple test InteractionSet."""

    interactions = [
        Interaction((0,), (0,), "Hydrophobic"),
        Interaction((1,), (1,), "HBAcceptor"),
    ]
    return InteractionSet(interactions, n_protein, n_ligand)


# ****************************
# *** BindingComplex Tests ***
# ****************************


class TestBindingComplexInit(unittest.TestCase):
    """Test BindingComplex initialization."""

    def test_basic_creation(self):
        protein = create_test_protein()
        ligand = create_test_ligand()

        complex = BindingComplex(protein, ligand)

        self.assertIsNotNone(complex.protein)
        self.assertIsNotNone(complex.ligand)
        self.assertIsNone(complex.interactions)
        self.assertEqual(complex.meta, {})

    def test_creation_with_interactions(self):
        protein = create_test_protein(n_atoms=5)
        ligand = create_test_ligand(n_atoms=4)
        interactions = create_test_interactions(5, 4)

        complex = BindingComplex(protein, ligand, interactions=interactions)

        self.assertIsNotNone(complex.interactions)
        self.assertEqual(len(complex.interactions), 2)

    def test_creation_with_meta(self):
        protein = create_test_protein()
        ligand = create_test_ligand()
        meta = {"system_id": "test_123", "source": "pdb"}

        complex = BindingComplex(protein, ligand, meta=meta)

        self.assertEqual(complex.meta["system_id"], "test_123")
        self.assertEqual(complex.meta["source"], "pdb")

    def test_invalid_protein_type_raises(self):
        ligand = create_test_ligand()

        with self.assertRaises(TypeError) as ctx:
            BindingComplex("not a protein", ligand)

        self.assertIn("protein", str(ctx.exception))

    def test_invalid_ligand_type_raises(self):
        protein = create_test_protein()

        with self.assertRaises(TypeError) as ctx:
            BindingComplex(protein, "not a ligand")

        self.assertIn("ligand", str(ctx.exception))

    def test_invalid_interactions_type_raises(self):
        protein = create_test_protein()
        ligand = create_test_ligand()

        with self.assertRaises(TypeError) as ctx:
            BindingComplex(protein, ligand, interactions="not an interaction set")

        self.assertIn("interactions", str(ctx.exception))

    def test_invalid_meta_type_raises(self):
        protein = create_test_protein()
        ligand = create_test_ligand()

        with self.assertRaises(TypeError) as ctx:
            BindingComplex(protein, ligand, meta="not a dict")

        self.assertIn("meta", str(ctx.exception))

    def test_interaction_protein_size_mismatch_raises(self):
        protein = create_test_protein(n_atoms=5)
        ligand = create_test_ligand(n_atoms=4)
        interactions = create_test_interactions(10, 4)  # Wrong protein size

        with self.assertRaises(ValueError) as ctx:
            BindingComplex(protein, ligand, interactions=interactions)

        self.assertIn("n_protein_atoms", str(ctx.exception))

    def test_interaction_ligand_size_mismatch_raises(self):
        protein = create_test_protein(n_atoms=5)
        ligand = create_test_ligand(n_atoms=4)
        interactions = create_test_interactions(5, 10)  # Wrong ligand size

        with self.assertRaises(ValueError) as ctx:
            BindingComplex(protein, ligand, interactions=interactions)

        self.assertIn("n_ligand_atoms", str(ctx.exception))


class TestBindingComplexProperties(unittest.TestCase):
    """Test BindingComplex property accessors."""

    def setUp(self):
        self.protein = create_test_protein(n_atoms=5, n_confs=1)
        self.ligand = create_test_ligand(n_atoms=4, n_confs=1)
        self.meta = {"system_id": "test_sys"}
        self.complex = BindingComplex(self.protein, self.ligand, meta=self.meta)

    def test_len(self):
        self.assertEqual(len(self.complex), 9)

    def test_seq_length(self):
        self.assertEqual(self.complex.seq_length, 9)

    def test_system_id(self):
        self.assertEqual(self.complex.system_id, "test_sys")

    def test_system_id_none_when_missing(self):
        complex = BindingComplex(self.protein, self.ligand)
        self.assertIsNone(complex.system_id)

    def test_coords_shape(self):
        coords = self.complex.coords

        # [n_ligand + n_protein, 3] - no conformer dimension
        self.assertEqual(coords.shape, (9, 3))

    def test_coords_concatenation_order(self):
        """Test that coords are ligand first, then protein."""

        coords = self.complex.coords

        # Check ligand coords are first (squeeze ligand 3D coords to 2D)
        np.testing.assert_array_almost_equal(coords[:4, :], self.ligand.coords[0])

        # Check protein coords are second
        np.testing.assert_array_almost_equal(coords[4:, :], self.protein.coords)

    def test_atomics(self):
        atomics = self.complex.atomics

        self.assertEqual(atomics.shape, (9,))
        # Ligand atomics first
        np.testing.assert_array_equal(atomics[:4], self.ligand.atomics)
        # Protein atomics second
        np.testing.assert_array_equal(atomics[4:], self.protein.atomics)

    def test_charges(self):
        charges = self.complex.charges

        self.assertEqual(charges.shape, (9,))
        np.testing.assert_array_equal(charges[:4], self.ligand.charges)
        np.testing.assert_array_equal(charges[4:], self.protein.charges)

    def test_res_names(self):
        res_names = self.complex.res_names

        self.assertEqual(res_names.shape, (9,))
        # Ligand atoms get "LIG" as residue name
        self.assertTrue(all(r == "LIG" for r in res_names[:4]))
        # Protein atoms keep their original residue names
        np.testing.assert_array_equal(res_names[4:], self.protein.res_names)

    def test_bonds_shape(self):
        bonds = self.complex.bonds

        # Total bonds = ligand bonds + protein bonds
        expected_n_bonds = self.ligand.n_bonds + self.protein.n_bonds
        self.assertEqual(bonds.shape, (expected_n_bonds, 3))

    def test_bonds_protein_indices_shifted(self):
        """Test that protein bond indices are shifted by ligand length."""

        bonds = self.complex.bonds
        n_ligand_bonds = self.ligand.n_bonds
        n_ligand_atoms = len(self.ligand)

        # Protein bonds should have indices shifted
        protein_bonds = bonds[n_ligand_bonds:]
        original_protein_bonds = self.protein.bonds.bonds

        # Check that indices are shifted by ligand length
        for i in range(len(protein_bonds)):
            self.assertEqual(protein_bonds[i, 0], original_protein_bonds[i, 0] + n_ligand_atoms)
            self.assertEqual(protein_bonds[i, 1], original_protein_bonds[i, 1] + n_ligand_atoms)
            # Bond type unchanged
            self.assertEqual(protein_bonds[i, 2], original_protein_bonds[i, 2])

    def test_bond_indices(self):
        bond_indices = self.complex.bond_indices

        self.assertEqual(bond_indices.shape[1], 2)

    def test_bond_types(self):
        bond_types = self.complex.bond_types

        expected_n_bonds = self.ligand.n_bonds + self.protein.n_bonds
        self.assertEqual(bond_types.shape, (expected_n_bonds,))

    def test_adjacency_shape(self):
        adj = self.complex.adjacency

        self.assertEqual(adj.shape, (9, 9))

    def test_adjacency_symmetry(self):
        adj = self.complex.adjacency

        np.testing.assert_array_equal(adj, adj.T)

    def test_ligand_mask(self):
        mask = self.complex.ligand_mask

        self.assertEqual(mask.shape, (9,))
        # First 4 atoms are ligand (value 1)
        np.testing.assert_array_equal(mask[:4], np.ones(4))
        # Last 5 atoms are protein (value 0)
        np.testing.assert_array_equal(mask[4:], np.zeros(5))

    def test_com_shape(self):
        com = self.complex.com

        self.assertEqual(com.shape, (3,))


class TestBindingComplexSerialization(unittest.TestCase):
    """Test BindingComplex serialization."""

    def test_to_bytes_from_bytes_roundtrip(self):
        protein = create_test_protein(n_atoms=5)
        ligand = create_test_ligand(n_atoms=4)
        meta = {"system_id": "test_123"}
        complex = BindingComplex(protein, ligand, meta=meta)

        data = complex.to_bytes()
        restored = BindingComplex.from_bytes(data)

        self.assertEqual(len(restored), len(complex))
        self.assertEqual(restored.meta, complex.meta)
        np.testing.assert_array_almost_equal(restored.coords, complex.coords)
        np.testing.assert_array_equal(restored.atomics, complex.atomics)
        np.testing.assert_array_equal(restored.charges, complex.charges)

    def test_serialization_with_interactions(self):
        protein = create_test_protein(n_atoms=5)
        ligand = create_test_ligand(n_atoms=4)
        interactions = create_test_interactions(5, 4)
        complex = BindingComplex(protein, ligand, interactions=interactions)

        data = complex.to_bytes()
        restored = BindingComplex.from_bytes(data)

        self.assertIsNotNone(restored.interactions)
        self.assertEqual(len(restored.interactions), len(complex.interactions))
        self.assertEqual(restored.interactions.n_protein_atoms, 5)
        self.assertEqual(restored.interactions.n_ligand_atoms, 4)

    def test_serialization_without_interactions(self):
        protein = create_test_protein()
        ligand = create_test_ligand()
        complex = BindingComplex(protein, ligand)

        data = complex.to_bytes()
        restored = BindingComplex.from_bytes(data)

        self.assertIsNone(restored.interactions)


class TestBindingComplexRemoveHs(unittest.TestCase):
    """Test BindingComplex.remove_hs method."""

    def _create_complex_with_hydrogens(self):
        """Create a complex where both protein and ligand have hydrogens."""

        # Protein with H atoms
        atomics = np.array([6, 1, 7, 1, 8])  # C, H, N, H, O
        charges = np.zeros(5, dtype=np.int16)
        res_names = np.array(["ALA", "ALA", "ALA", "ALA", "ALA"])
        atom_names = np.array(["CA", "H1", "N", "H2", "O"])
        res_ids = np.array([1, 1, 1, 1, 1])

        atoms = AtomSet(atomics, charges=charges, res_names=res_names, atom_names=atom_names, res_ids=res_ids)
        bonds = BondSet(np.array([[0, 2, 1], [2, 4, 1]]))
        coords = np.random.rand(1, 5, 3).astype(np.float32)
        confs = ConfSet(coords)

        protein = Protein(atoms, bonds, confs)

        # Ligand with H atoms
        lig_atomics = np.array([6, 1, 8, 1])  # C, H, O, H
        lig_charges = np.zeros(4, dtype=np.int16)
        lig_atoms = AtomSet(lig_atomics, charges=lig_charges)
        lig_bonds = BondSet(np.array([[0, 2, 1]]))
        lig_coords = np.random.rand(1, 4, 3).astype(np.float32)
        lig_confs = ConfSet(lig_coords)

        ligand = GraphMol(lig_atoms, lig_bonds, confs=lig_confs)

        return BindingComplex(protein, ligand)

    def test_remove_hs_removes_from_both(self):
        complex = self._create_complex_with_hydrogens()

        no_h = complex.remove_hs()

        # Protein: C, N, O (3 heavy atoms from 5)
        self.assertEqual(len(no_h.protein), 3)
        self.assertTrue(all(a != 1 for a in no_h.protein.atomics))

        # Ligand: C, O (2 heavy atoms from 4)
        self.assertEqual(len(no_h.ligand), 2)
        self.assertTrue(all(a != 1 for a in no_h.ligand.atomics))

    def test_remove_hs_ligand_only_false(self):
        complex = self._create_complex_with_hydrogens()

        no_h = complex.remove_hs(include_ligand=False)

        # Protein should have Hs removed
        self.assertEqual(len(no_h.protein), 3)

        # Ligand should keep all atoms
        self.assertEqual(len(no_h.ligand), 4)

    def test_remove_hs_preserves_meta(self):
        complex = self._create_complex_with_hydrogens()
        complex.meta["test_key"] = "test_value"

        no_h = complex.remove_hs()

        self.assertEqual(no_h.meta["test_key"], "test_value")

    def test_remove_hs_updates_interactions(self):
        """Test that interactions are remapped when atoms are removed."""

        complex = self._create_complex_with_hydrogens()

        # Add interactions - note indices for non-H atoms only
        # Protein: atoms 0 (C), 2 (N), 4 (O) -> become 0, 1, 2
        # Ligand: atoms 0 (C), 2 (O) -> become 0, 1
        interactions = InteractionSet(
            [Interaction((0,), (0,), "Hydrophobic")],  # C-C interaction
            n_protein_atoms=5,
            n_ligand_atoms=4,
        )
        complex_with_int = BindingComplex(complex.protein, complex.ligand, interactions=interactions)

        no_h = complex_with_int.remove_hs()

        self.assertIsNotNone(no_h.interactions)
        self.assertEqual(no_h.interactions.n_protein_atoms, 3)
        self.assertEqual(no_h.interactions.n_ligand_atoms, 2)


class TestBindingComplexZeroCom(unittest.TestCase):
    """Test BindingComplex.zero_com method."""

    def test_zero_com_centers_at_origin(self):
        protein = create_test_protein(n_atoms=5)
        ligand = create_test_ligand(n_atoms=4)
        complex = BindingComplex(protein, ligand)

        zeroed = complex.zero_com()

        # COM should be at origin (within floating point precision)
        np.testing.assert_array_almost_equal(zeroed.com, np.zeros(3), decimal=5)

    def test_zero_com_preserves_structure(self):
        protein = create_test_protein(n_atoms=5)
        ligand = create_test_ligand(n_atoms=4)
        complex = BindingComplex(protein, ligand)

        zeroed = complex.zero_com()

        # Relative distances should be preserved
        original_dists = complex.coords[0] - complex.coords[1]
        zeroed_dists = zeroed.coords[0] - zeroed.coords[1]
        np.testing.assert_array_almost_equal(original_dists, zeroed_dists)


class TestBindingComplexCopy(unittest.TestCase):
    """Test BindingComplex copy methods."""

    def setUp(self):
        self.protein = create_test_protein(n_atoms=5)
        self.ligand = create_test_ligand(n_atoms=4)
        self.complex = BindingComplex(self.protein, self.ligand, meta={"key": "value"})

    def test_copy_creates_independent_object(self):
        copied = self.complex.copy()

        # Modify original meta
        self.complex.meta["new_key"] = "new_value"

        # Copy should not be affected
        self.assertNotIn("new_key", copied.meta)

    def test_copy_preserves_all_data(self):
        copied = self.complex.copy()

        self.assertEqual(len(copied), len(self.complex))
        self.assertEqual(copied.meta["key"], "value")
        np.testing.assert_array_almost_equal(copied.coords, self.complex.coords)

    def test_copy_preserves_interactions(self):
        interactions = create_test_interactions(5, 4)
        complex_with_int = BindingComplex(self.protein, self.ligand, interactions=interactions)

        copied = complex_with_int.copy()

        self.assertIsNotNone(copied.interactions)
        self.assertEqual(len(copied.interactions), len(interactions))

    def test_copy_with_new_protein(self):
        new_protein = create_test_protein(n_atoms=3)

        copied = self.complex.copy_with(protein=new_protein)

        self.assertEqual(len(copied.protein), 3)
        self.assertEqual(len(copied.ligand), 4)  # Original ligand

    def test_copy_with_new_ligand(self):
        new_ligand = create_test_ligand(n_atoms=3)

        copied = self.complex.copy_with(ligand=new_ligand)

        self.assertEqual(len(copied.protein), 5)  # Original protein
        self.assertEqual(len(copied.ligand), 3)

    def test_copy_with_new_interactions(self):
        new_interactions = InteractionSet([Interaction((0,), (0,), "HBDonor")], n_protein_atoms=5, n_ligand_atoms=4)

        copied = self.complex.copy_with(interactions=new_interactions)

        self.assertEqual(len(copied.interactions), 1)
        self.assertEqual(copied.interactions.interaction_types, ["HBDonor"])

    def test_copy_with_preserves_meta(self):
        new_protein = create_test_protein(n_atoms=5)

        copied = self.complex.copy_with(protein=new_protein)

        self.assertEqual(copied.meta["key"], "value")


class TestBindingComplexEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    def test_single_atom_protein_and_ligand(self):
        # Single atom protein
        atomics = np.array([6])
        charges = np.array([0])
        res_names = np.array(["ALA"])
        atom_names = np.array(["CA"])
        res_ids = np.array([1])

        atoms = AtomSet(atomics, charges=charges, res_names=res_names, atom_names=atom_names, res_ids=res_ids)
        bonds = BondSet(np.array([]).reshape(0, 3).astype(np.int32))
        coords = np.random.rand(1, 1, 3).astype(np.float32)
        confs = ConfSet(coords)

        protein = Protein(atoms, bonds, confs)

        # Single atom ligand
        lig_atoms = AtomSet(np.array([8]), charges=np.array([0]))
        lig_bonds = BondSet(np.array([]).reshape(0, 3).astype(np.int32))
        lig_coords = np.random.rand(1, 1, 3).astype(np.float32)
        lig_confs = ConfSet(lig_coords)

        ligand = GraphMol(lig_atoms, lig_bonds, confs=lig_confs)

        complex = BindingComplex(protein, ligand)

        self.assertEqual(len(complex), 2)
        self.assertEqual(complex.seq_length, 2)

    def test_empty_interactions(self):
        protein = create_test_protein(n_atoms=5)
        ligand = create_test_ligand(n_atoms=4)
        interactions = InteractionSet([], n_protein_atoms=5, n_ligand_atoms=4)

        complex = BindingComplex(protein, ligand, interactions=interactions)

        self.assertIsNotNone(complex.interactions)
        self.assertEqual(len(complex.interactions), 0)

    def test_multiple_conformers_raises_error(self):
        """Test that multiple conformers in ligand raises an error."""

        protein = create_test_protein(n_atoms=5, n_confs=1)
        ligand = create_test_ligand(n_atoms=4, n_confs=10)

        complex = BindingComplex(protein, ligand)

        with self.assertRaises(ValueError):
            _ = complex.coords
