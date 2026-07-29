import unittest

import numpy as np

from molito.core.confs import ConfSet


class TestConfSet(unittest.TestCase):
    def setUp(self):
        # Create a simple conf set with 3 conformers, 4 atoms each
        self.coords = np.array(
            [
                [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]],
                [[0, 1, 0], [1, 1, 0], [2, 1, 0], [3, 1, 0]],
                [[0, 2, 0], [1, 2, 0], [2, 2, 0], [3, 2, 0]],
            ],
            dtype=np.float32,
        )
        self.weights = np.array([0.5, 0.3, 0.2], dtype=np.float32)
        self.confs = ConfSet(self.coords, weights=self.weights)

    def test_basic_properties(self):
        self.assertEqual(self.confs.n_atoms, 4)
        self.assertEqual(self.confs.n_conformers, 3)
        self.assertEqual(self.confs.seq_length, 3)
        self.assertEqual(len(self.confs), 3)

    def test_has_weights(self):
        self.assertTrue(self.confs.has_weights)

        confs_no_weights = ConfSet(self.coords)
        self.assertFalse(confs_no_weights.has_weights)

    def test_coords_property(self):
        np.testing.assert_array_equal(self.confs.coords, self.coords)

    def test_weights_property(self):
        np.testing.assert_array_equal(self.confs.weights, self.weights)

    def test_com(self):
        com = self.confs.com
        self.assertEqual(com.shape, (3, 1, 3))
        # COM for first conformer: mean of [[0,0,0], [1,0,0], [2,0,0], [3,0,0]] = [1.5, 0, 0]
        np.testing.assert_array_almost_equal(com[0], [[1.5, 0, 0]])

    def test_getitem_int_returns_array(self):
        conf = self.confs[0]
        self.assertEqual(conf.shape, (4, 3))
        np.testing.assert_array_equal(conf, self.coords[0])

    def test_getitem_array_returns_confset(self):
        indices = np.array([0, 2])
        subset = self.confs[indices]

        self.assertIsInstance(subset, ConfSet)
        self.assertEqual(len(subset), 2)
        np.testing.assert_array_equal(subset.coords[0], self.coords[0])
        np.testing.assert_array_equal(subset.coords[1], self.coords[2])

    def test_get_conformer(self):
        conf = self.confs.get_conformer(1)
        self.assertEqual(conf.shape, (4, 3))
        np.testing.assert_array_equal(conf, self.coords[1])

    def test_copy(self):
        copied = self.confs.copy()

        np.testing.assert_array_equal(copied.coords, self.confs.coords)
        np.testing.assert_array_equal(copied.weights, self.confs.weights)

        # Verify it's a deep copy
        copied.coords[0, 0, 0] = 999
        self.assertNotEqual(self.confs.coords[0, 0, 0], 999)

    def test_permute_atoms(self):
        indices = [3, 1, 0]
        permuted = self.confs.permute_atoms(indices)

        self.assertEqual(permuted.n_atoms, 3)
        self.assertEqual(permuted.n_conformers, 3)
        # First conformer, first atom should now be what was at index 3
        np.testing.assert_array_equal(permuted.coords[0, 0], self.coords[0, 3])

    def test_permute_atoms_duplicate_raises_error(self):
        with self.assertRaises(ValueError):
            self.confs.permute_atoms([0, 1, 1])

    def test_permute_atoms_out_of_bounds_raises_error(self):
        with self.assertRaises(ValueError):
            self.confs.permute_atoms([0, 1, 10])

    def test_permute_confs(self):
        indices = [2, 0]
        permuted = self.confs.permute_confs(indices)

        self.assertEqual(permuted.n_conformers, 2)
        np.testing.assert_array_equal(permuted.coords[0], self.coords[2])
        np.testing.assert_array_equal(permuted.coords[1], self.coords[0])
        # Weights should also be permuted
        np.testing.assert_array_almost_equal(permuted.weights, [0.2, 0.5])

    def test_pad_equal_size(self):
        padded = self.confs.pad(4)
        self.assertEqual(padded.n_atoms, 4)
        np.testing.assert_array_equal(padded.coords, self.confs.coords)

    def test_pad_smaller_raises_error(self):
        with self.assertRaises(ValueError):
            self.confs.pad(2)

    def test_pad_with_zeros(self):
        padded = self.confs.pad(6)

        self.assertEqual(padded.n_atoms, 6)
        # Original coords preserved
        np.testing.assert_array_equal(padded.coords[:, :4, :], self.coords)
        # Padded coords are zeros
        np.testing.assert_array_equal(padded.coords[:, 4:, :], np.zeros((3, 2, 3)))

    def test_zero_com(self):
        zeroed = self.confs.zero_com()
        com = zeroed.com
        np.testing.assert_array_almost_equal(com, np.zeros((3, 1, 3)), decimal=5)

    def test_shift(self):
        shift_vec = np.array([10, 20, 30])
        shifted = self.confs.shift(shift_vec)

        expected = self.coords + shift_vec
        np.testing.assert_array_almost_equal(shifted.coords, expected)

    def test_scale(self):
        scaled = self.confs.scale(2.0)
        expected = self.coords * 2.0
        np.testing.assert_array_almost_equal(scaled.coords, expected)

    def test_to_dict_from_dict(self):
        dict_repr = self.confs.to_dict()
        restored = ConfSet.from_dict(dict_repr)

        np.testing.assert_array_equal(restored.coords, self.confs.coords)
        np.testing.assert_array_equal(restored.weights, self.confs.weights)

    def test_to_dict_without_weights(self):
        confs = ConfSet(self.coords)
        dict_repr = confs.to_dict()

        self.assertIn("coords", dict_repr)
        self.assertNotIn("weights", dict_repr)

    def test_2d_coords_expanded(self):
        # 2D coords should be expanded to 3D with singleton conformer dim
        coords_2d = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
        confs = ConfSet(coords_2d)

        self.assertEqual(confs.n_conformers, 1)
        self.assertEqual(confs.n_atoms, 3)
        self.assertEqual(confs.coords.shape, (1, 3, 3))

    def test_weights_sum_zero_raises_error(self):
        with self.assertRaises(RuntimeError):
            ConfSet(self.coords, weights=np.zeros(3))


class TestConfSetSelection(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.coords = np.random.rand(5, 4, 3).astype(np.float32)
        self.weights = np.array([0.4, 0.3, 0.15, 0.1, 0.05], dtype=np.float32)
        self.confs = ConfSet(self.coords, weights=self.weights)

    def test_select_topk(self):
        top2 = self.confs.select_topk(2)

        self.assertEqual(len(top2), 2)
        # Should select conformers with highest weights (indices 0 and 1)
        np.testing.assert_array_equal(top2.coords[0], self.coords[0])
        np.testing.assert_array_equal(top2.coords[1], self.coords[1])

    def test_select_topk_default_is_1(self):
        top1 = self.confs.select_topk()
        self.assertEqual(len(top1), 1)

    def test_select_topk_requires_weights(self):
        confs_no_weights = ConfSet(self.coords)
        with self.assertRaises(RuntimeError):
            confs_no_weights.select_topk(2)

    def test_weighted_sample_without_weights_uses_uniform(self):
        confs_no_weights = ConfSet(self.coords)
        sampled = confs_no_weights.weighted_sample(3)
        self.assertEqual(len(sampled), 3)

    def test_uniform_sample(self):
        sampled = self.confs.uniform_sample(10)
        self.assertEqual(len(sampled), 10)
