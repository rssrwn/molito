import tempfile
import unittest

import h5py
import numpy as np

from molito.core.lazydata import LazyData


class TestLazyData(unittest.TestCase):
    def setUp(self):
        # Create a temporary HDF5 file with test data
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
        self.temp_file.close()

        with h5py.File(self.temp_file.name, "w") as f:
            # 1D dataset
            self.data_1d = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32)
            f.create_dataset("data_1d", data=self.data_1d)

            # 2D dataset (flattened for storage, will be reshaped)
            self.data_2d = np.arange(24, dtype=np.int32)
            f.create_dataset("data_2d", data=self.data_2d)

        self.h5_file = h5py.File(self.temp_file.name, "r")

    def tearDown(self):
        self.h5_file.close()

    def test_shape_1d(self):
        lazy = LazyData(self.h5_file["data_1d"], start_idx=0, n_items=5)
        self.assertEqual(lazy.shape, (5,))

    def test_shape_multidim(self):
        lazy = LazyData(self.h5_file["data_2d"], start_idx=0, n_items=(4, 3))
        self.assertEqual(lazy.shape, (4, 3))

    def test_len(self):
        lazy = LazyData(self.h5_file["data_1d"], start_idx=0, n_items=5)
        self.assertEqual(len(lazy), 5)

    def test_dtype(self):
        lazy = LazyData(self.h5_file["data_1d"], start_idx=0, n_items=5)
        self.assertEqual(lazy.dtype, np.float32)

    def test_read_all_1d(self):
        lazy = LazyData(self.h5_file["data_1d"], start_idx=0, n_items=5)
        result = lazy.read()
        expected = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_read_all_1d_with_offset(self):
        lazy = LazyData(self.h5_file["data_1d"], start_idx=3, n_items=4)
        result = lazy.read()
        expected = np.array([4, 5, 6, 7], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_read_1d_then_index(self):
        lazy = LazyData(self.h5_file["data_1d"], start_idx=2, n_items=5)
        result = lazy.read()
        self.assertEqual(result[0], 3.0)

    def test_read_all_multidim(self):
        lazy = LazyData(self.h5_file["data_2d"], start_idx=0, n_items=(2, 3))
        result = lazy.read()
        expected = np.arange(6, dtype=np.int32).reshape(2, 3)
        np.testing.assert_array_equal(result, expected)

    def test_read_multidim_then_index(self):
        lazy = LazyData(self.h5_file["data_2d"], start_idx=0, n_items=(4, 3))
        result = lazy.read()
        expected = np.array([3, 4, 5], dtype=np.int32)
        np.testing.assert_array_equal(result[1], expected)
