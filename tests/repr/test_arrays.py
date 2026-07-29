import unittest

import numpy as np

from molito.arrays import adj_from_edges, one_hot_encode, pad_arrays


class TestPadArrays(unittest.TestCase):
    def test_equal_length_arrays(self):
        arrays = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        result = pad_arrays(arrays)
        self.assertEqual(result.shape, (2, 3))
        np.testing.assert_array_equal(result[0], [1, 2, 3])
        np.testing.assert_array_equal(result[1], [4, 5, 6])

    def test_different_length_arrays(self):
        arrays = [np.array([1, 2]), np.array([3, 4, 5])]
        result = pad_arrays(arrays)
        self.assertEqual(result.shape, (2, 3))
        np.testing.assert_array_equal(result[0], [1, 2, 0])
        np.testing.assert_array_equal(result[1], [3, 4, 5])

    def test_empty_list(self):
        result = pad_arrays([])
        self.assertEqual(result.shape, (0,))

    def test_single_array(self):
        arrays = [np.array([1, 2, 3])]
        result = pad_arrays(arrays)
        self.assertEqual(result.shape, (1, 3))

    def test_multidimensional(self):
        arrays = [np.ones((2, 3)), np.ones((4, 3))]
        result = pad_arrays(arrays)
        self.assertEqual(result.shape, (2, 4, 3))
        np.testing.assert_array_equal(result[0, 2:], 0)

    def test_preserves_dtype(self):
        arrays = [np.array([1, 2], dtype=np.float32), np.array([3, 4, 5], dtype=np.float32)]
        result = pad_arrays(arrays)
        self.assertEqual(result.dtype, np.float32)


class TestOneHotEncode(unittest.TestCase):
    def test_basic_encoding(self):
        indices = np.array([0, 2, 1])
        result = one_hot_encode(indices, 4)
        expected = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]])
        np.testing.assert_array_equal(result, expected)

    def test_shape(self):
        indices = np.array([0, 1, 2, 3])
        result = one_hot_encode(indices, 5)
        self.assertEqual(result.shape, (4, 5))

    def test_single_index(self):
        indices = np.array([3])
        result = one_hot_encode(indices, 5)
        self.assertEqual(result.shape, (1, 5))
        self.assertEqual(result[0, 3], 1)
        self.assertEqual(result.sum(), 1)


class TestAdjFromEdges(unittest.TestCase):
    def test_basic_adjacency(self):
        edge_indices = np.array([[0, 1], [1, 2]])
        edge_types = np.array([1, 2])
        adj = adj_from_edges(edge_indices, edge_types, 3)
        self.assertEqual(adj[0, 1], 1)
        self.assertEqual(adj[1, 2], 2)
        self.assertEqual(adj[1, 0], 0)

    def test_symmetric(self):
        edge_indices = np.array([[0, 1], [1, 2]])
        edge_types = np.array([1, 2])
        adj = adj_from_edges(edge_indices, edge_types, 3, symmetric=True)
        self.assertEqual(adj[0, 1], 1)
        self.assertEqual(adj[1, 0], 1)
        self.assertEqual(adj[1, 2], 2)
        self.assertEqual(adj[2, 1], 2)

    def test_empty_edges(self):
        edge_indices = np.zeros((0, 2), dtype=np.int64)
        edge_types = np.zeros(0, dtype=np.int64)
        adj = adj_from_edges(edge_indices, edge_types, 4)
        np.testing.assert_array_equal(adj, np.zeros((4, 4)))

    def test_shape(self):
        edge_indices = np.array([[0, 1]])
        edge_types = np.array([1])
        adj = adj_from_edges(edge_indices, edge_types, 5)
        self.assertEqual(adj.shape, (5, 5))

    def test_dtype_preserved(self):
        edge_indices = np.array([[0, 1]])
        edge_types = np.array([1], dtype=np.int16)
        adj = adj_from_edges(edge_indices, edge_types, 3)
        self.assertEqual(adj.dtype, np.int16)
