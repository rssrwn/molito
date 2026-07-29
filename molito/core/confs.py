from __future__ import annotations

from collections.abc import Sequence

import h5py
import numpy as np
from rdkit import Chem
from scipy.spatial.transform import Rotation

from molito.core._checks import check_dict_key, check_dim_shape, check_shape_len
from molito.core.lazydata import LazyData

TArr = np.ndarray


class ConfSet(Sequence):
    """Set of 3D conformers for a molecule.

    Stores conformer coordinates as a [n_confs, n_atoms, 3] float32 array, with optional
    Boltzmann weights per conformer. Supports geometric operations (rotation, shift, scale,
    centre-of-mass removal) and lazy loading from HDF5.

    A 2D array [n_atoms, 3] is automatically expanded to [1, n_atoms, 3].

    Args:
        coords: Conformer coordinates, shape [n_confs, n_atoms, 3] or [n_atoms, 3].
        weights: Optional conformer weights, shape [n_confs]. Must not sum to zero.
    """

    __slots__ = ("_coords", "_weights")

    def __init__(self, coords: TArr | LazyData, weights: TArr | LazyData | None = None):
        # Create a singleton conf dim if only one conf is provided (np array only, h5 should already have it)
        if isinstance(coords, np.ndarray):
            coords = np.expand_dims(coords, axis=0) if len(coords.shape) == 2 else coords
            coords = coords.astype(np.float32)

        check_shape_len(coords, 3, "coordinates")
        check_dim_shape(coords, 2, 3, "coordinates")

        if weights is not None:
            check_shape_len(weights, 1, "weights")
            check_dim_shape(weights, 0, coords.shape[0], "weights")

            if isinstance(weights, np.ndarray):
                weights = weights.astype(np.float32)
                if np.isclose(weights.sum().item(), 0.0, atol=1e-5):
                    raise RuntimeError("If conformer weights are provided they must not sum to 0")

        self._coords = coords
        self._weights = weights

    @classmethod
    def _load_unchecked(cls, coords, weights) -> ConfSet:
        """Fast-path factory for HDF5 load. Skips validation."""

        obj = cls.__new__(cls)
        obj._coords = coords
        obj._weights = weights
        return obj

    @classmethod
    def _iter_from_group(cls, group: h5py.Group):
        """Yield one ConfSet per mol from an HDF5 'confs' group.

        The 'sizes' dataset has shape [n_mols, 3] = [n_confs, n_atoms, n_weights] per row.
        Conformer weights are optional per mol -- n_weights=0 means 'no weights stored'.
        """

        sizes = np.asarray(group["sizes"][()])
        coords_ds = group["coords"]
        weights_ds = group["weights"]

        coord_off = 0
        weight_off = 0
        for n_confs, n_atoms, n_w in sizes.tolist():
            coords = LazyData._load_unchecked(coords_ds, coord_off, (n_confs, n_atoms))
            weights = LazyData._load_unchecked(weights_ds, weight_off, n_w) if n_w > 0 else None
            yield cls._load_unchecked(coords, weights)
            coord_off += n_confs * n_atoms
            weight_off += n_w

    # *** Publicly exposed properties ***

    @property
    def coords(self) -> TArr:
        """Returns conformer array of shape [n_confs, n_atoms, 3]"""

        if isinstance(self._coords, LazyData):
            return self._coords.read().astype(np.float32)

        return self._coords

    @property
    def weights(self) -> TArr | None:
        """Returns conf weights array of shape [n_confs,], or None if this set has no weights."""

        if self._weights is None:
            return None

        if isinstance(self._weights, LazyData):
            return self._weights.read().astype(np.float32)

        return self._weights

    @property
    def n_atoms(self) -> int:
        return self._coords.shape[1]

    @property
    def n_conformers(self) -> int:
        return len(self._coords)

    @property
    def seq_length(self) -> int:
        return len(self)

    @property
    def has_weights(self) -> bool:
        return self._weights is not None

    @property
    def com(self) -> TArr:
        """Returns a numpy array of shape [n_confs, 1, 3] representing the centre of mass of the confs."""
        return self.coords.mean(axis=1, keepdims=True)

    # *** Basic indexing and utility functions ***

    def __len__(self) -> int:
        """Returns the number of confomers in the set, not the number of atoms."""
        return self.n_conformers

    def __getitem__(self, index: int | TArr) -> ConfSet | TArr:
        """Index by int returns a single conformer array [N, 3]. Index by array returns a new ConfSet."""

        if isinstance(index, int):
            conf = self.coords[index].copy()
            return conf

        if isinstance(index, np.ndarray):
            confs = self.coords[index].copy()
            confs = ConfSet(confs)
            return confs

        raise TypeError("index must be either an int or an np array.")

    def read(self) -> ConfSet:
        """Force the data to read into memory if it's not already."""

        coords = self.coords
        weights = self.weights
        return ConfSet(coords, weights=weights)

    def get_conformer(self, conf_idx: int) -> TArr:
        return self.coords[conf_idx, :, :].copy()

    def copy_with(self, coords: TArr | None = None, weights: TArr | None = None) -> ConfSet:
        coords = self.coords.copy() if coords is None else coords.copy()

        if weights is not None:
            weights = weights.copy()
        elif self.has_weights:
            weights = self.weights.copy()

        confs = ConfSet(coords, weights=weights)
        return confs

    def copy(self) -> ConfSet:
        return self.copy_with()

    # *** Permutation functions ***

    def permute_atoms(self, indices: list[int] | TArr) -> ConfSet:
        """Used for permuting atom order. Can be used for reordering or taking a subset, but not for duplicating."""

        if len(set(indices)) != len(indices):
            raise ValueError("Indices list cannot contain duplicates.")

        if max(indices) >= self.n_atoms:
            raise ValueError(f"Index {max(indices)} is out of bounds for conf set with {self.n_atoms} atoms.")

        indices = np.array(indices)
        coords = self.coords[:, indices, :]

        confs = self.copy_with(coords=coords)
        return confs

    def permute_confs(self, indices: list[int] | TArr) -> ConfSet:
        """Used for permuting conformer order. Can reorder or take a subset, but not duplicate."""

        if len(set(indices)) != len(indices):
            raise ValueError("Indices list cannot contain duplicates.")

        if max(indices) >= len(self):
            raise ValueError(f"Index {max(indices)} is out of bounds for conf set with {len(self)} conformers.")

        indices = np.array(indices)
        coords = self.coords[indices, :, :]
        weights = self.weights[indices]

        confs = self.copy_with(coords=coords, weights=weights)
        return confs

    def weighted_sample(self, n_confs: int) -> ConfSet:
        """Sample conformers weighted by weights if available, otherwise uniform."""

        probs = self.weights / self.weights.sum() if self.has_weights else None
        sampled_idxs = np.random.choice(list(range(len(self))), size=n_confs, p=probs, replace=True)
        confs = np.stack([self.get_conformer(idx) for idx in sampled_idxs])
        return ConfSet(confs)

    def uniform_sample(self, n_confs: int) -> ConfSet:
        sampled_idxs = np.random.choice(list(range(len(self))), size=n_confs, p=None, replace=True)
        confs = np.stack([self.get_conformer(idx) for idx in sampled_idxs])
        return ConfSet(confs)

    def select_topk(self, k: int | None = None) -> ConfSet:
        k = 1 if k is None else k

        if not self.has_weights:
            raise RuntimeError("ConfSet must have weights to select top conformers.")

        if k > self.n_conformers:
            raise ValueError("k cannot be greater than the number of conformers.")

        indices = np.argsort(self.weights)[::-1][:k].tolist()
        confs = np.stack([self.get_conformer(idx) for idx in indices])
        return ConfSet(confs)

    def pad(self, n_atoms: int) -> ConfSet:
        """Pad the atoms to length n_atoms with zero coords for 'pad' atoms"""

        if n_atoms < self.n_atoms:
            raise ValueError("Cannot pad to fewer atoms than exist in the molecule.")

        if n_atoms == self.n_atoms:
            return self.copy()

        n_pad_atoms = n_atoms - self.n_atoms
        pad_coords = np.zeros((self.n_conformers, n_pad_atoms, 3))
        coords = np.concatenate((self.coords, pad_coords), axis=1)

        return self.copy_with(coords=coords)

    # *** Geometric specific functions ***

    def zero_com(self) -> ConfSet:
        shifted = self.coords - self.com
        return self.copy_with(coords=shifted)

    def rotate(self, rotation: Rotation | list[Rotation]) -> ConfSet:
        if isinstance(rotation, Rotation):
            rotated = [rotation.apply(conf) for conf in self._to_list()]

        elif isinstance(rotation, list):
            if len(rotation) != len(self):
                err = "If a list of rotations is provided, the length must match the number of conformers."
                raise RuntimeError(err)

            rotated = [rot.apply(conf) for rot, conf in zip(rotation, self._to_list(), strict=True)]

        else:
            raise TypeError("The rotation provided must be either a scipy Rotation or a list of Rotation objects.")

        rotated = np.stack(rotated)
        return self.copy_with(coords=rotated)

    def shift(self, shift: TArr | list[TArr]) -> ConfSet:
        if isinstance(shift, np.ndarray):
            shifted = self._apply_shift(self.coords, shift)

        elif isinstance(shift, list):
            if len(shift) != len(self):
                err = "If a list of shifts is provided, the length must match the number of conformers."
                raise RuntimeError(err)

            shifted = [self._apply_shift(sh, conf) for sh, conf in zip(shift, self._to_list(), strict=True)]
            shifted = np.stack(shifted)

        else:
            raise TypeError("The shift provided must be either a numpy array or a list of arrays.")

        return self.copy_with(coords=shifted)

    def _apply_shift(self, coords: TArr, shift: TArr) -> TArr:
        check_shape_len(shift, 1, "shift")
        check_dim_shape(shift, 0, 3, "shift")

        if len(coords.shape) == 2:
            shift = np.expand_dims(shift, 0)
        elif len(coords.shape) == 3:
            shift = np.expand_dims(shift, (0, 1))
        else:
            raise ValueError("coords array must be either 2 or 3 dimensional.")

        return coords + shift

    def scale(self, scale: float) -> ConfSet:
        scaled = self.coords * scale
        return self.copy_with(coords=scaled)

    # *** IO and conversion utility functions ***

    @staticmethod
    def from_rdkit(mol: Chem.rdchem.Mol) -> ConfSet:
        """Load a ConfSet from an RDKit mol. If conformers have a 'weight' property it will be used as the weight."""

        if mol.GetNumConformers() == 0:
            raise RuntimeError("mol must have at least one conformer.")

        coords = []
        weights = []

        for conf in mol.GetConformers():
            if not conf.Is3D():
                raise RuntimeError("All conformers within mol must have 3D coordinates.")

            conf_coords = np.array(conf.GetPositions())
            coords.append(conf_coords)

            try:
                weight = float(conf.GetProp("weight"))
            except (KeyError, ValueError):
                weight = None

            weights.append(weight)

        coord_arr = np.stack(coords, axis=0)
        weight_arr = np.array(weights) if all([w is not None for w in weights]) else None

        return ConfSet(coord_arr, weights=weight_arr)

    @staticmethod
    def from_dict(dict_repr) -> ConfSet:
        coords = dict_repr["coords"]
        weights = dict_repr.get("weights", None)
        return ConfSet(coords, weights=weights)

    def to_dict(self) -> dict[str, np.ndarray]:
        dict_repr = {"coords": self.coords}
        if self.weights is not None:
            dict_repr["weights"] = self.weights

        return dict_repr

    def _to_list(self) -> list[TArr]:
        return list(self.coords)

    @staticmethod
    def confs_from_arrays(array_map: dict[str, np.ndarray | h5py.Dataset]) -> list[ConfSet]:
        """Inverse function of arrays_from_confs, but can also be loaded from h5py Datasets."""

        check_dict_key(array_map, "coords", "conf array map")
        check_dict_key(array_map, "weights", "conf array map")
        check_dict_key(array_map, "sizes", "conf array map")

        sizes = np.array(array_map["sizes"][()]).tolist()

        cs_arr = array_map["coords"]
        ws_arr = array_map["weights"]

        cs_is_hdf5 = isinstance(cs_arr, h5py.Dataset)
        ws_is_hdf5 = isinstance(ws_arr, h5py.Dataset)

        if (not cs_is_hdf5) and (not isinstance(cs_arr, np.ndarray)):
            raise TypeError(f"Coords array must be either np.ndarray or h5py.Dataset, got {type(cs_arr)}")

        if (not ws_is_hdf5) and (not isinstance(ws_arr, np.ndarray)):
            raise TypeError(f"Weights array must be either np.ndarray or h5py.Dataset, got {type(ws_arr)}")

        cs_arr = cs_arr.copy() if not cs_is_hdf5 else cs_arr
        ws_arr = ws_arr.copy() if not ws_is_hdf5 else ws_arr

        conf_sets = []
        curr_coord_idx = 0
        curr_weight_idx = 0

        for n_confs, n_atoms, n_ws in sizes:
            if cs_is_hdf5:
                confs = LazyData(cs_arr, curr_coord_idx, (n_confs, n_atoms))
            else:
                confs = cs_arr[curr_coord_idx : curr_coord_idx + (n_confs * n_atoms)]

            if n_ws == 0:
                weights = None
            elif ws_is_hdf5:
                weights = LazyData(ws_arr, curr_weight_idx, n_ws)
            else:
                weights = ws_arr[curr_weight_idx : curr_weight_idx + n_ws]

            conf_set = ConfSet(confs, weights)
            conf_sets.append(conf_set)

            curr_coord_idx += n_confs * n_atoms
            curr_weight_idx += n_ws

        return conf_sets

    @staticmethod
    def arrays_from_confs(conf_sets: list[ConfSet]) -> dict[str, np.ndarray]:
        """Merge multiple conf sets into a dictionary of numpy arrays."""

        weights = [conf_set.weights for conf_set in conf_sets]
        coords = [conf_set.coords for conf_set in conf_sets]

        non_none_weights = [ws for ws in weights if ws is not None]
        weights_arr = np.concatenate(non_none_weights) if non_none_weights else np.array([], dtype=np.float32)
        coords_arr = np.concatenate([cs.reshape((cs.shape[0] * cs.shape[1], 3)) for cs in coords], axis=0)

        weight_sizes = np.array([0 if ws is None else len(ws) for ws in weights])
        coord_sizes = np.array([list(cs.shape[:2]) for cs in coords])
        sizes_arr = np.concatenate((coord_sizes, weight_sizes[:, None]), axis=1)

        arrays = {"coords": coords_arr, "weights": weights_arr, "sizes": sizes_arr}
        return arrays
