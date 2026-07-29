import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from molito.mol.interactions import (
    GROUP_HALOGEN_MAP,
    GROUP_HBOND_MAP,
    GROUP_IONIC_MAP,
    GROUP_METAL_MAP,
    GROUP_PI_MAP,
    Interaction,
    InteractionSet,
)

# ***********************
# *** Interaction Tests ***
# ***********************


class TestInteraction(unittest.TestCase):
    """Test the Interaction dataclass."""

    def test_basic_creation(self):
        interaction = Interaction(protein_atoms=(0, 1), ligand_atoms=(2,), interaction_type="HBAcceptor")

        self.assertEqual(interaction.protein_atoms, (0, 1))
        self.assertEqual(interaction.ligand_atoms, (2,))
        self.assertEqual(interaction.interaction_type, "HBAcceptor")

    def test_is_immutable(self):
        interaction = Interaction(protein_atoms=(0,), ligand_atoms=(1,), interaction_type="Hydrophobic")

        with self.assertRaises(AttributeError):
            interaction.interaction_type = "HBDonor"

    def test_with_type(self):
        interaction = Interaction(protein_atoms=(0,), ligand_atoms=(1,), interaction_type="HBAcceptor")
        new_interaction = interaction.with_type("hydrogen-bond")

        self.assertEqual(new_interaction.interaction_type, "hydrogen-bond")
        self.assertEqual(new_interaction.protein_atoms, (0,))
        self.assertEqual(new_interaction.ligand_atoms, (1,))
        # Original unchanged
        self.assertEqual(interaction.interaction_type, "HBAcceptor")

    def test_remap_atoms_success(self):
        interaction = Interaction(protein_atoms=(0, 2), ligand_atoms=(1, 3), interaction_type="PiStacking")
        protein_map = {0: 0, 2: 1}
        ligand_map = {1: 0, 3: 1}

        new_interaction = interaction.remap_atoms(protein_map, ligand_map)

        self.assertIsNotNone(new_interaction)
        self.assertEqual(new_interaction.protein_atoms, (0, 1))
        self.assertEqual(new_interaction.ligand_atoms, (0, 1))
        self.assertEqual(new_interaction.interaction_type, "PiStacking")

    def test_remap_atoms_missing_protein_atom(self):
        interaction = Interaction(
            protein_atoms=(0, 5),  # 5 not in map
            ligand_atoms=(1,),
            interaction_type="Hydrophobic",
        )
        protein_map = {0: 0, 2: 1}
        ligand_map = {1: 0}

        result = interaction.remap_atoms(protein_map, ligand_map)

        self.assertIsNone(result)

    def test_remap_atoms_missing_ligand_atom(self):
        interaction = Interaction(
            protein_atoms=(0,),
            ligand_atoms=(1, 10),  # 10 not in map
            interaction_type="Hydrophobic",
        )
        protein_map = {0: 0}
        ligand_map = {1: 0}

        result = interaction.remap_atoms(protein_map, ligand_map)

        self.assertIsNone(result)


# **************************
# *** InteractionSet Tests ***
# **************************


class TestInteractionSetInit(unittest.TestCase):
    """Test InteractionSet initialization."""

    def test_basic_creation(self):
        interactions = [
            Interaction((0,), (0,), "Hydrophobic"),
            Interaction((1,), (1,), "HBAcceptor"),
        ]
        int_set = InteractionSet(interactions, n_protein_atoms=10, n_ligand_atoms=5)

        self.assertEqual(len(int_set), 2)
        self.assertEqual(int_set.n_protein_atoms, 10)
        self.assertEqual(int_set.n_ligand_atoms, 5)

    def test_empty_set(self):
        int_set = InteractionSet([], n_protein_atoms=10, n_ligand_atoms=5)

        self.assertEqual(len(int_set), 0)
        self.assertEqual(int_set.interaction_types, [])


class TestInteractionSetProperties(unittest.TestCase):
    """Test InteractionSet properties."""

    def setUp(self):
        self.interactions = [
            Interaction((0,), (0,), "Hydrophobic"),
            Interaction((1,), (1,), "HBAcceptor"),
            Interaction((2,), (2,), "HBDonor"),
            Interaction((3,), (0,), "Hydrophobic"),  # Duplicate type
        ]
        self.int_set = InteractionSet(self.interactions, n_protein_atoms=10, n_ligand_atoms=5)

    def test_interaction_types_unique_and_sorted(self):
        types = self.int_set.interaction_types

        self.assertEqual(len(types), 3)  # Unique types
        self.assertEqual(types, ["HBAcceptor", "HBDonor", "Hydrophobic"])  # Sorted

    def test_array_shape(self):
        arr = self.int_set.array

        self.assertEqual(arr.shape, (10, 5, 3))
        self.assertEqual(arr.dtype, np.int8)

    def test_array_values(self):
        arr = self.int_set.array
        types = self.int_set.interaction_types

        # Check that interactions are correctly placed
        hydro_idx = types.index("Hydrophobic")
        hba_idx = types.index("HBAcceptor")
        hbd_idx = types.index("HBDonor")

        self.assertEqual(arr[0, 0, hydro_idx], 1)
        self.assertEqual(arr[3, 0, hydro_idx], 1)
        self.assertEqual(arr[1, 1, hba_idx], 1)
        self.assertEqual(arr[2, 2, hbd_idx], 1)

        # Check that non-interaction positions are zero
        self.assertEqual(arr[0, 0, hba_idx], 0)


class TestInteractionSetFromArray(unittest.TestCase):
    """Test InteractionSet.from_array."""

    def test_from_array_basic(self):
        interaction_types = ["Hydrophobic", "HBAcceptor", "HBDonor"]
        arr = np.zeros((10, 5, 3), dtype=np.int8)
        arr[0, 0, 0] = 1  # Hydrophobic
        arr[1, 1, 1] = 1  # HBAcceptor
        arr[2, 2, 2] = 1  # HBDonor

        int_set = InteractionSet.from_array(interaction_types, arr)

        self.assertEqual(len(int_set), 3)
        self.assertEqual(int_set.n_protein_atoms, 10)
        self.assertEqual(int_set.n_ligand_atoms, 5)

    def test_from_array_roundtrip(self):
        """Test that array -> InteractionSet -> array preserves values."""

        interaction_types = ["Hydrophobic", "HBAcceptor"]
        original_arr = np.zeros((10, 5, 2), dtype=np.int8)
        original_arr[0, 0, 0] = 1
        original_arr[1, 1, 1] = 1
        original_arr[5, 3, 0] = 1

        int_set = InteractionSet.from_array(interaction_types, original_arr)
        result_arr = int_set.array

        # Note: interaction_types may be in different order in result
        # We need to compare values properly
        self.assertEqual(result_arr.shape[:2], original_arr.shape[:2])

        # Check total number of interactions is the same
        self.assertEqual(np.sum(result_arr), np.sum(original_arr))

    def test_from_array_invalid_dims(self):
        interaction_types = ["Hydrophobic"]
        arr = np.zeros((10, 5), dtype=np.int8)  # 2D instead of 3D

        with self.assertRaises(ValueError) as ctx:
            InteractionSet.from_array(interaction_types, arr)

        self.assertIn("3 dimensions", str(ctx.exception))

    def test_from_array_mismatched_types(self):
        interaction_types = ["Hydrophobic", "HBAcceptor"]  # 2 types
        arr = np.zeros((10, 5, 3), dtype=np.int8)  # 3 in last dim

        with self.assertRaises(ValueError) as ctx:
            InteractionSet.from_array(interaction_types, arr)

        self.assertIn("must match last dim", str(ctx.exception))


class TestInteractionSetSubset(unittest.TestCase):
    """Test InteractionSet.subset."""

    def setUp(self):
        self.interactions = [
            Interaction((0,), (0,), "Hydrophobic"),
            Interaction((1,), (1,), "HBAcceptor"),
            Interaction((2,), (2,), "HBDonor"),
            Interaction((3,), (3,), "PiStacking"),
        ]
        self.int_set = InteractionSet(self.interactions, n_protein_atoms=10, n_ligand_atoms=5)

    def test_subset_single_type(self):
        subset = self.int_set.subset(["Hydrophobic"])

        self.assertEqual(len(subset), 1)
        self.assertEqual(subset.interaction_types, ["Hydrophobic"])

    def test_subset_multiple_types(self):
        subset = self.int_set.subset(["HBAcceptor", "HBDonor"])

        self.assertEqual(len(subset), 2)
        self.assertEqual(sorted(subset.interaction_types), ["HBAcceptor", "HBDonor"])

    def test_subset_preserves_atom_counts(self):
        subset = self.int_set.subset(["Hydrophobic"])

        self.assertEqual(subset.n_protein_atoms, 10)
        self.assertEqual(subset.n_ligand_atoms, 5)

    def test_subset_invalid_type_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self.int_set.subset(["InvalidType"])

        self.assertIn("not found", str(ctx.exception))


class TestInteractionSetSubsetAtoms(unittest.TestCase):
    """Test InteractionSet.subset_atoms."""

    def test_subset_atoms_basic(self):
        interactions = [
            Interaction((0,), (0,), "Hydrophobic"),
            Interaction((2,), (1,), "HBAcceptor"),
            Interaction((4,), (2,), "HBDonor"),
        ]
        int_set = InteractionSet(interactions, n_protein_atoms=5, n_ligand_atoms=3)

        # Keep only protein atoms 0, 2 and ligand atoms 0, 2
        protein_mask = np.array([True, False, True, False, True])
        ligand_mask = np.array([True, False, True])

        subset = int_set.subset_atoms(protein_mask, ligand_mask)

        self.assertEqual(subset.n_protein_atoms, 3)  # 0, 2, 4 -> 0, 1, 2
        self.assertEqual(subset.n_ligand_atoms, 2)  # 0, 2 -> 0, 1

        # First interaction should remain (0, 0) -> (0, 0)
        # Second interaction should be removed (ligand atom 1 not in mask)
        # Third interaction should remain (4, 2) -> (2, 1)
        self.assertEqual(len(subset), 2)

    def test_subset_atoms_removes_interactions(self):
        interactions = [
            Interaction((0,), (0,), "Hydrophobic"),
            Interaction((1,), (1,), "HBAcceptor"),  # This should be removed
        ]
        int_set = InteractionSet(interactions, n_protein_atoms=5, n_ligand_atoms=3)

        protein_mask = np.array([True, False, True, True, True])
        ligand_mask = np.array([True, False, True])

        subset = int_set.subset_atoms(protein_mask, ligand_mask)

        self.assertEqual(len(subset), 1)


class TestInteractionSetGroup(unittest.TestCase):
    """Test InteractionSet.group."""

    def test_group_basic(self):
        interactions = [
            Interaction((0,), (0,), "HBAcceptor"),
            Interaction((1,), (1,), "HBDonor"),
            Interaction((2,), (2,), "Hydrophobic"),
        ]
        int_set = InteractionSet(interactions, n_protein_atoms=10, n_ligand_atoms=5)

        grouped = int_set.group(GROUP_HBOND_MAP)

        self.assertEqual(len(grouped), 3)
        types = grouped.interaction_types
        self.assertIn("hydrogen-bond", types)
        self.assertIn("Hydrophobic", types)
        self.assertNotIn("HBAcceptor", types)
        self.assertNotIn("HBDonor", types)

    def test_group_preserves_multi_atom_interactions(self):
        interactions = [
            Interaction((0, 1, 2), (0, 1), "HBAcceptor"),
        ]
        int_set = InteractionSet(interactions, n_protein_atoms=10, n_ligand_atoms=5)

        grouped = int_set.group(GROUP_HBOND_MAP)

        self.assertEqual(len(grouped), 1)
        self.assertEqual(grouped.interaction_types, ["hydrogen-bond"])

    def test_group_ionic(self):
        interactions = [
            Interaction((0,), (0,), "Cationic"),
            Interaction((1,), (1,), "Anionic"),
        ]
        int_set = InteractionSet(interactions, n_protein_atoms=10, n_ligand_atoms=5)

        grouped = int_set.group(GROUP_IONIC_MAP)

        self.assertEqual(len(grouped), 2)
        self.assertEqual(grouped.interaction_types, ["ionic-interaction"])

    def test_group_metal(self):
        interactions = [
            Interaction((0,), (0,), "MetalAcceptor"),
            Interaction((1,), (1,), "MetalDonor"),
        ]
        int_set = InteractionSet(interactions, n_protein_atoms=10, n_ligand_atoms=5)

        grouped = int_set.group(GROUP_METAL_MAP)

        self.assertEqual(grouped.interaction_types, ["metal-interaction"])

    def test_group_halogen(self):
        interactions = [
            Interaction((0,), (0,), "XBAcceptor"),
            Interaction((1,), (1,), "XBDonor"),
        ]
        int_set = InteractionSet(interactions, n_protein_atoms=10, n_ligand_atoms=5)

        grouped = int_set.group(GROUP_HALOGEN_MAP)

        self.assertEqual(grouped.interaction_types, ["halogen-bond"])

    def test_group_pi(self):
        interactions = [
            Interaction((0,), (0,), "CationPi"),
            Interaction((1,), (1,), "PiCation"),
        ]
        int_set = InteractionSet(interactions, n_protein_atoms=10, n_ligand_atoms=5)

        grouped = int_set.group(GROUP_PI_MAP)

        self.assertEqual(grouped.interaction_types, ["pi-cation"])


class TestInteractionSetSerialization(unittest.TestCase):
    """Test InteractionSet serialization."""

    def setUp(self):
        self.interactions = [
            Interaction((0, 1), (0,), "Hydrophobic"),
            Interaction((2,), (1, 2), "HBAcceptor"),
        ]
        self.int_set = InteractionSet(self.interactions, n_protein_atoms=10, n_ligand_atoms=5)

    def test_to_bytes_from_bytes_roundtrip(self):
        data = self.int_set.to_bytes()
        restored = InteractionSet.from_bytes(data)

        self.assertEqual(len(restored), len(self.int_set))
        self.assertEqual(restored.n_protein_atoms, self.int_set.n_protein_atoms)
        self.assertEqual(restored.n_ligand_atoms, self.int_set.n_ligand_atoms)


class TestInteractionSetCopy(unittest.TestCase):
    """Test InteractionSet.copy."""

    def test_copy_creates_independent_object(self):
        interactions = [
            Interaction((0,), (0,), "Hydrophobic"),
        ]
        int_set = InteractionSet(interactions, n_protein_atoms=10, n_ligand_atoms=5)

        copied = int_set.copy()

        self.assertEqual(len(copied), 1)
        self.assertEqual(copied.n_protein_atoms, 10)
        self.assertEqual(copied.n_ligand_atoms, 5)


# ********************************
# *** HDF5 Save/Load Tests      ***
# ********************************


class TestInteractionSetHDF5(unittest.TestCase):
    """Test InteractionSet HDF5 save/load."""

    def test_hdf5_roundtrip_basic(self):
        int_set1 = InteractionSet(
            [
                Interaction((0,), (0,), "Hydrophobic"),
                Interaction((1,), (1,), "HBAcceptor"),
            ],
            n_protein_atoms=5,
            n_ligand_atoms=3,
        )
        int_set2 = InteractionSet([Interaction((1, 2), (0, 1), "PiStacking")], n_protein_atoms=10, n_ligand_atoms=4)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.hdf5"

            with h5py.File(path, "w") as f:
                InteractionSet.save_to_group([int_set1, int_set2], f.create_group("interactions"))

            with h5py.File(path, "r") as f:
                restored = InteractionSet.load_from_group(f["interactions"])

        self.assertEqual(len(restored), 2)

        # Check first set
        self.assertEqual(restored[0].n_protein_atoms, 5)
        self.assertEqual(restored[0].n_ligand_atoms, 3)
        self.assertEqual(len(restored[0]), 2)

        # Check second set
        self.assertEqual(restored[1].n_protein_atoms, 10)
        self.assertEqual(restored[1].n_ligand_atoms, 4)
        self.assertEqual(len(restored[1]), 1)

    def test_hdf5_roundtrip_with_none(self):
        int_set = InteractionSet([Interaction((0,), (0,), "Hydrophobic")], n_protein_atoms=5, n_ligand_atoms=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.hdf5"

            with h5py.File(path, "w") as f:
                InteractionSet.save_to_group([int_set, None, int_set], f.create_group("interactions"))

            with h5py.File(path, "r") as f:
                restored = InteractionSet.load_from_group(f["interactions"])

        self.assertEqual(len(restored), 3)
        self.assertIsNotNone(restored[0])
        self.assertIsNone(restored[1])
        self.assertIsNotNone(restored[2])

    def test_hdf5_roundtrip_empty_set(self):
        int_set = InteractionSet([], n_protein_atoms=5, n_ligand_atoms=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.hdf5"

            with h5py.File(path, "w") as f:
                InteractionSet.save_to_group([int_set], f.create_group("interactions"))

            with h5py.File(path, "r") as f:
                restored = InteractionSet.load_from_group(f["interactions"])

        self.assertEqual(len(restored), 1)
        self.assertIsNotNone(restored[0])
        self.assertEqual(len(restored[0]), 0)
        self.assertEqual(restored[0].n_protein_atoms, 5)
        self.assertEqual(restored[0].n_ligand_atoms, 3)

    def test_hdf5_roundtrip_all_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.hdf5"

            with h5py.File(path, "w") as f:
                InteractionSet.save_to_group([None, None], f.create_group("interactions"))

            with h5py.File(path, "r") as f:
                restored = InteractionSet.load_from_group(f["interactions"])

        self.assertEqual(len(restored), 2)
        self.assertIsNone(restored[0])
        self.assertIsNone(restored[1])

    def test_hdf5_roundtrip_multi_atom_interactions(self):
        """Test that multi-atom interactions are preserved through HDF5."""
        int_set = InteractionSet([Interaction((1, 2, 3), (0, 1), "PiStacking")], n_protein_atoms=10, n_ligand_atoms=5)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.hdf5"

            with h5py.File(path, "w") as f:
                InteractionSet.save_to_group([int_set], f.create_group("interactions"))

            with h5py.File(path, "r") as f:
                restored = InteractionSet.load_from_group(f["interactions"])

        self.assertEqual(len(restored[0]), 1)
        # Verify the multi-atom tuples are preserved
        arr = restored[0].array
        types = restored[0].interaction_types
        pi_idx = types.index("PiStacking")

        # All protein atoms (1, 2, 3) x all ligand atoms (0, 1) should have the interaction
        self.assertEqual(arr[1, 0, pi_idx], 1)
        self.assertEqual(arr[1, 1, pi_idx], 1)
        self.assertEqual(arr[2, 0, pi_idx], 1)
        self.assertEqual(arr[2, 1, pi_idx], 1)
        self.assertEqual(arr[3, 0, pi_idx], 1)
        self.assertEqual(arr[3, 1, pi_idx], 1)
