from __future__ import annotations

import copy
from functools import reduce

import h5py
import numpy as np

from molito.core._checks import check_type

TArr = np.ndarray


class LazyData:
    """A thin wrapper for an h5py dataset.

    By allowing a shared h5py Dataset to be used and read from, this class abstracts reading from a segment of an h5py
    Dataset and allows end users to treat the data segment as a very simple np array with a read function for reading
    the data segment from h5py into a real np array.
    """

    __slots__ = ("_arr", "_start_idx", "_shape", "_n_items")

    def __init__(self, arr: h5py.Dataset, start_idx: int, n_items: int | tuple):
        check_type(start_idx, int, "start_idx")
        check_type(n_items, [int, tuple], "n_items")

        n_items = (n_items,) if isinstance(n_items, int) else copy.deepcopy(n_items)
        shape = self._calc_shape(arr.shape, n_items)
        n_items = reduce(lambda i, j: i * j, n_items)

        self._arr = arr
        self._start_idx = start_idx
        self._shape = shape
        self._n_items = n_items

    @classmethod
    def _load_unchecked(cls, arr: h5py.Dataset, start_idx: int, n_items: int | tuple) -> LazyData:
        """Fast-path factory for the HDF5 load path. Skips validation and shape normalisation.

        Shape invariant: `_shape` is always a tuple, matching the full-tensor layout (the
        leading slice dims followed by any trailing dims from the underlying array). `_n_items`
        is the flat element count for indexing into the h5py dataset.
        """

        obj = cls.__new__(cls)
        if isinstance(n_items, int):
            leading = (n_items,)
            n_flat = n_items
        else:
            leading = tuple(n_items)
            n_flat = reduce(lambda i, j: i * j, n_items)

        shape = leading + tuple(arr.shape[1:])

        obj._arr = arr
        obj._start_idx = start_idx
        obj._shape = shape
        obj._n_items = n_flat
        return obj

    @property
    def shape(self) -> tuple:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self._arr.dtype

    def __len__(self) -> int:
        return self.shape[0]

    def read(self) -> TArr:
        end_idx = self._start_idx + self._n_items
        data = self._arr[self._start_idx : end_idx]
        return np.array(data).reshape(self.shape)

    def _calc_shape(self, arr_shape, n_items):
        if len(arr_shape) == 1:
            return n_items

        return (*n_items, *arr_shape[1:])
