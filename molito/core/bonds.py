"""Bond storage primitives.

Stereochemistry design notes
----------------------------

Double bond E/Z is stored as bond *direction* tags (`BondDir.ENDUPRIGHT` / `ENDDOWNRIGHT`,
i.e. the SMILES `/` and `\\` tokens) on the **single bonds adjacent to the double bond** —
not as `BondStereo.STEREOE`/`STEREOZ` on the double bond itself.

These are *relative* tags, not absolute CIP descriptors. The analogy is:

    tetrahedral: CW / CCW  (relative parity tag) ↔ R / S  (absolute, CIP-derived)
    double bond: BondDir   (relative `/` / `\\`) ↔ E / Z  (absolute, CIP-derived)

We use the relative tag for both stereo types. RDKit's `Atom.SetChiralTag` accepts CW/CCW
but has no `SetCIPCode` equivalent for R/S, so for tetrahedral chirality the relative tag
is the only handle. For double bonds RDKit *does* expose absolute stereo via
`Bond.SetStereo` + `Bond.SetStereoAtoms`, but using it would require storing a pair of
CIP reference-atom indices per double bond and remapping them under permutation — so it
would just relocate the bookkeeping rather than eliminate it. Storing the relative tag
keeps both stereo types symmetric and bottoms out on a single invariant (below).

What survives permutation, and why
----------------------------------

`BondSet.permute_atoms` (see docstring there) relabels atom indices via `index_map` but
does NOT (a) reorder bond rows or (b) swap the two columns of any row. This is the load-
bearing invariant for both stereo types:

  * Tetrahedral CW/CCW depends on the cyclic order in which RDKit encounters a chiral
    atom's neighbours. `mol_from_atoms` iterates bonds in array-row order and calls
    `AddBond` in that order, so preserving row order keeps each chiral centre's neighbour
    cyclic order intact. See `molito/core/atoms.py` for the storage side.

  * Double bond stereo depends on `BondDir` being interpreted relative to the bond's
    begin/end atoms ("end atom is up-right of begin atom"). A row stored as
    `[a, b, BondDir.ENDUPRIGHT]` becomes `[p(a), p(b), BondDir.ENDUPRIGHT]` after
    permutation — never `[p(b), p(a), ...]`. So `mol.AddBond(start, end, ...)` in
    `mol_from_atoms` gives RDKit the same physical begin/end atoms as the original mol
    (just relabelled), and the geometric meaning of the direction tag is preserved.

What WOULD break stereo
-----------------------

Anything that breaks the row-order / column-order invariant inside `permute_atoms`:

  * Lex-sorting bond rows by `(start, end)` after the index remap. Flips chirality at any
    centre whose neighbour cyclic order changes.
  * Re-imposing `start < end` row-by-row after permutation (i.e. swapping the two index
    columns when `start > end`). Flips the begin/end role of the bond, which inverts the
    geometric meaning of `BondDir` and turns E into Z (and vice versa).
  * Grouping bond rows per atom, dropping/inserting rows, or any other reordering.

Anything that re-perceives stereo from incomplete / approximate context can also break
things — see `convert.py::mol_from_atoms` which deliberately skips
`AssignStereochemistryFrom3D` when the caller has supplied chirality tags, because 3D
coords from ML models or noisy sources can disagree with the intended stereochemistry.

Tests in `tests/repr/test_stereo.py` pin down both the round-trip behaviour and the
underlying row/column-order invariant directly (see `TestBondRowOrderInvariant`).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Union

import h5py
import numpy as np
from biotite.structure import BondList
from rdkit import Chem

from molito.arrays import adj_from_edges
from molito.core._checks import check_dict_key, check_dim_shape, check_shape_len, check_type
from molito.core.lazydata import LazyData

TArr = np.ndarray

_BondT = Union[Chem.BondType, str]
_DirT = Union[Chem.BondDir, str, None]


# ***************************************************
# ***** Bond Encoding (fixed, used for storage) *****
# ***************************************************


class BondEncoding:
    """Fixed encoding for bond storage. Maps bond properties <--> integer codes.

    This encoding is comprehensive and defines what integers mean when stored to disk.
    It should never change — otherwise old HDF5 files become unreadable.

    All methods are classmethods operating on class-level lookup tables.
    """

    _enum_bond_map: dict[int, _BondT] = {
        0: "NONE",
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
        -1: "MASK",
    }
    _bond_enum_map = {bond: idx for idx, bond in _enum_bond_map.items()}

    _dir_str_map = {
        Chem.BondDir.NONE: None,
        Chem.BondDir.ENDUPRIGHT: "U",
        Chem.BondDir.ENDDOWNRIGHT: "D",
    }
    _str_dir_map = {s: d for d, s in _dir_str_map.items()}

    _encoding_strs = [
        "0_F",  # 0:  none
        "1_F",  # 1:  single, non-aromatic
        "2_F",  # 2:  double, non-aromatic
        "3_F",  # 3:  triple, non-aromatic
        "1_T",  # 4:  single, aromatic
        "2_T",  # 5:  double, aromatic
        "3_T",  # 6:  triple, aromatic
        "1_F_U",  # 7:  single, non-aromatic, ENDUPRIGHT
        "1_F_D",  # 8:  single, non-aromatic, ENDDOWNRIGHT
        "1_T_U",  # 9:  single, aromatic, ENDUPRIGHT
        "1_T_D",  # 10: single, aromatic, ENDDOWNRIGHT
        "-1_F",  # 11: mask
    ]

    _token_to_idx = {token: idx for idx, token in enumerate(_encoding_strs)}
    _idx_to_token = {idx: token for idx, token in enumerate(_encoding_strs)}

    @classmethod
    def size(cls) -> int:
        """Return the number of encoding types (12)."""

        return len(cls._encoding_strs)

    @classmethod
    def encode(cls, bond: _BondT, is_aromatic: bool = False, direction: _DirT = None) -> int:
        """Encode bond properties into a storage index.

        Args:
            bond: Bond type (Chem.BondType.SINGLE, etc.) or "NONE"/"MASK" string.
            is_aromatic: Whether the bond is aromatic.
            direction: E/Z bond direction (Chem.BondDir.ENDUPRIGHT, etc.), or None.

        Returns:
            int: Encoding index (0-11).
        """

        bond_enum = cls._bond_enum_map[bond]
        enc_str = cls._make_str(bond_enum, is_aromatic, direction)
        return cls._token_to_idx[enc_str]

    @classmethod
    def decode(cls, index: int) -> tuple[_BondT, bool, _DirT]:
        """Decode a storage index back into bond properties.

        Args:
            index: Encoding index (0-11).

        Returns:
            Tuple of (bond_type, is_aromatic, direction).
        """

        token = cls._idx_to_token[index]
        parts = token.split("_")

        bond_type = cls._enum_bond_map[int(parts[0])]
        is_arom = parts[1] == "T"
        direction = cls._str_dir_map.get(parts[2]) if len(parts) == 3 else None

        return bond_type, is_arom, direction

    @classmethod
    def get_token(cls, index: int) -> str:
        """Return the string token for a given encoding index (e.g. '1_F' for single non-aromatic)."""

        return cls._idx_to_token[index]

    @classmethod
    def _make_str(cls, bond_enum: int, is_aromatic: bool, direction: _DirT) -> str:

        if isinstance(direction, Chem.BondDir):
            direction = cls._dir_str_map.get(direction)

        enc_str = f"{bond_enum}_{'T' if is_aromatic else 'F'}"
        if direction is not None:
            enc_str = f"{enc_str}_{direction.upper()}"

        return enc_str


# Per-encoding-index boolean: True iff the encoding carries a non-None direction tag.
# Used by BondSet.neighbour_ranks to identify atoms involved in directional bonds.
_DIRECTIONAL_ENCODING_MASK = np.array(
    [BondEncoding.decode(i)[2] is not None for i in range(BondEncoding.size())], dtype=bool
)


# *************************************************************
# ***** BondSet used for storing bonds for whole molecule *****
# *************************************************************


class BondSet(Sequence):
    """Set of bonds for a single molecule or protein chain.

    Bonds are stored as an [n_bonds, 3] int16 array where each row is [start_idx, end_idx, bond_encoding_index].
    Always upper triangular (start < end). Bond types are encoded via BondEncoding which preserves
    aromaticity and E/Z bond directions.

    Args:
        bonds: Bond array, shape [n_bonds, 3].
    """

    __slots__ = ("_bonds",)

    def __init__(self, bonds: TArr | LazyData):
        check_type(bonds, [np.ndarray, LazyData], "bonds")
        check_shape_len(bonds, 2, "bonds")
        check_dim_shape(bonds, 1, 3, "bonds")

        if isinstance(bonds, np.ndarray):
            bonds = bonds.astype(np.int16)

        self._bonds = bonds

    @classmethod
    def _load_unchecked(cls, bonds) -> BondSet:
        """Fast-path factory for HDF5 load. Skips validation and dtype casts."""

        obj = cls.__new__(cls)
        obj._bonds = bonds
        return obj

    @classmethod
    def _iter_from_group(cls, group: h5py.Group):
        """Yield one BondSet per mol from an HDF5 'bonds' group."""

        sizes = np.asarray(group["sizes"][()]).tolist()
        bonds_ds = group["bonds"]

        offset = 0
        for n in sizes:
            yield cls._load_unchecked(LazyData._load_unchecked(bonds_ds, offset, n))
            offset += n

    # *** Publicly exposed properties ***

    @property
    def bonds(self) -> TArr:
        """Returns array of shape [n_bonds, 3]"""

        if isinstance(self._bonds, LazyData):
            return self._bonds.read().astype(np.int16)

        return self._bonds

    @property
    def seq_length(self) -> int:
        return len(self)

    @property
    def indices(self) -> TArr:
        return self.bonds[:, :2]

    @property
    def types(self) -> TArr:
        return self.bonds[:, 2]

    @property
    def min_index(self) -> int | None:
        if len(self) == 0:
            return None

        index = self.indices.min().item()
        return index

    @property
    def max_index(self) -> int | None:
        if len(self) == 0:
            return None

        index = self.indices.max().item()
        return index

    # *** Basic indexing and utility functions ***

    def __len__(self) -> int:
        return len(self._bonds)

    def __getitem__(self, index: int | TArr) -> BondSet | tuple[int, int, int]:
        """Index by int returns (start, end, bond_type) tuple. Index by array returns a new BondSet."""

        if isinstance(index, int):
            bond = self.bonds[index]
            return tuple(bond.tolist())

        if isinstance(index, np.ndarray):
            bonds = self.bonds[index]
            bonds = self.copy_with(bonds=bonds)
            return bonds

        raise TypeError("index must be either an int or an np array.")

    def adj_matrix(self, n_atoms: int) -> TArr:
        if len(self) > 0 and self.indices.max().item() >= n_atoms:
            raise ValueError("The largest atom index is larger than the number of atoms requested.")

        adj = adj_from_edges(self.indices.astype(np.int64), self.types.astype(np.int64), n_atoms, symmetric=True)
        return adj

    def neighbour_ranks(self, n_atoms: int, chiral_mask: TArr | None = None) -> TArr:
        """[n_atoms, n_atoms] int matrix. rank[i, j] = rank of atom j in atom i's
        neighbour list, in bond row order (1-indexed). 0 means "no rank info" —
        either j isn't a neighbour of i, or atom i didn't need ranks (in
        stereo-only mode).

        This is intentionally asymmetric: rank[i, j] reflects i's neighbour history,
        rank[j, i] reflects j's. Exposing bond row order to the model is what makes
        chirality (CW/CCW interpreted relative to a neighbour cyclic order) and E/Z
        directions (interpreted relative to a bond's begin/end columns) recoverable.

        Args:
            n_atoms: total atom count, sets the output matrix shape.
            chiral_mask: optional [n_atoms] bool. When provided, only atoms that
                are chiral or involved in a directional bond get their ranks
                populated; all other atoms' rows stay 0. This keeps non-stereo
                atoms permutation-equivariant. When None, every neighbour pair
                is ranked.

        Returns:
            np.ndarray of shape [n_atoms, n_atoms], dtype int16.
        """

        if len(self) > 0 and self.indices.max().item() >= n_atoms:
            raise ValueError("The largest atom index is larger than the number of atoms requested.")

        rank_mat = np.zeros((n_atoms, n_atoms), dtype=np.int16)

        if len(self) == 0:
            return rank_mat

        indices = self.indices
        n_bonds = len(self)

        # Each bond row contributes two atom→neighbour appearances. The neighbour
        # count at atom A accumulates across all bond rows where A appears in
        # either column — that's the order in which RDKit encounters A's
        # neighbours during mol_from_atoms reconstruction.
        row_idxs = np.arange(n_bonds, dtype=np.int64)
        atoms = np.concatenate([indices[:, 0], indices[:, 1]]).astype(np.int64)
        neighs = np.concatenate([indices[:, 1], indices[:, 0]]).astype(np.int64)
        rows = np.concatenate([row_idxs, row_idxs])

        # Sort by (atom, row_idx) so each atom's appearances are contiguous and in
        # bond row order.
        order = np.lexsort((rows, atoms))
        sorted_atoms = atoms[order]
        sorted_neighs = neighs[order]

        # Rank within each contiguous atom group: 1, 2, 3, ...
        is_new_group = np.concatenate([[True], sorted_atoms[1:] != sorted_atoms[:-1]])
        group_start_positions = np.where(is_new_group)[0]
        group_lengths = np.diff(np.append(group_start_positions, len(sorted_atoms)))
        group_offsets = np.repeat(group_start_positions, group_lengths)
        rank_in_group = (np.arange(len(sorted_atoms)) - group_offsets + 1).astype(np.int16)

        if chiral_mask is None:
            rank_mat[sorted_atoms, sorted_neighs] = rank_in_group
            return rank_mat

        # Stereo-only mode: only populate ranks at atoms that need them.
        atoms_need_ranks = np.asarray(chiral_mask, dtype=bool).copy()
        directional_rows = _DIRECTIONAL_ENCODING_MASK[self.types]
        if directional_rows.any():
            directional_atom_idxs = indices[directional_rows].flatten()
            atoms_need_ranks[directional_atom_idxs] = True

        keep = atoms_need_ranks[sorted_atoms]
        rank_mat[sorted_atoms[keep], sorted_neighs[keep]] = rank_in_group[keep]
        return rank_mat

    def read(self) -> BondSet:
        return BondSet(self.bonds)

    def copy_with(
        self, bonds: TArr | None = None, bond_indices: TArr | None = None, bond_types: TArr | None = None
    ) -> BondSet:

        if bonds is not None and (bond_indices is not None or bond_types is not None):
            raise ValueError("BondSet copy_with cannot be provided with both bonds and bond indices or types")

        if bond_indices is not None and bond_types is not None and bond_indices.shape[0] != bond_types.shape[0]:
            raise ValueError("The length of bond_indices and bond_types must be the same.")

        if bond_indices is not None and bond_types is None and bond_indices.shape[0] != len(self):
            raise ValueError("The length of bond_indices must match the current bonds, if types are not provided.")

        if bond_indices is None and bond_types is not None and len(self) != bond_types.shape[0]:
            raise ValueError("The length of bond_type must match the current bonds, if indices are not provided.")

        if bonds is not None:
            check_shape_len(bonds, 2, "bonds")
            check_dim_shape(bonds, 1, 3, "bonds")

            bond_indices = bonds[:, :2]
            bond_types = bonds[:, 2]

        bond_indices = self.indices.copy() if bond_indices is None else bond_indices.copy()
        bond_types = self.types.copy() if bond_types is None else bond_types.copy()
        bond_types = np.expand_dims(bond_types, axis=1)

        bond_arr = np.concatenate((bond_indices, bond_types), axis=1)
        return BondSet(bond_arr)

    def copy(self) -> BondSet:
        return self.copy_with()

    def permute_atoms(self, indices: list[int] | TArr) -> BondSet:
        """Used for permuting atom order. The indices are the same as those given to the AtomSet permute function.

        STEREO-CRITICAL — read the module docstring before changing this function. Two
        invariants must hold for both tetrahedral CW/CCW and double-bond `BondDir` to
        round-trip correctly through arbitrary permutations:

          1. The relative row order of the bond array must be preserved (rows are filtered
             by mask, never reordered).
          2. The two index columns of each row must not be swapped (we relabel via
             `index_map`, we do NOT re-impose `start < end` row-by-row).

        Reordering rows breaks tetrahedral chirality. Swapping columns breaks double-bond
        E/Z. If you change this function, also update `tests/repr/test_stereo.py`
        (`TestBondRowOrderInvariant` plus the round-trip suites).
        """

        if len(set(indices)) != len(indices):
            raise ValueError("Indices list cannot contain duplicates.")

        indices = np.array(indices)

        # Remove any bonds which have an atom not in the index list
        mask = np.isin(self.indices, indices)
        mask = mask.all(axis=1)

        rem_bond_indices = self.indices[mask, :]
        bond_types = self.types[mask]

        # Map old atom indices to new sequential indices
        index_map = np.array([-1] * (indices.max().item() + 1))
        index_map[indices] = np.arange(indices.shape[0])

        bond_indices = index_map[rem_bond_indices]

        bonds = self.copy_with(bond_indices=bond_indices, bond_types=bond_types)
        return bonds

    # *** IO and conversion utility functions ***

    @staticmethod
    def from_rdkit(mol: Chem.rdchem.Mol) -> BondSet:
        # Kekulise so aromatic bonds are converted to single/double (aromatic property still stored)
        kekul_mol = Chem.Mol(mol)
        Chem.Kekulize(kekul_mol)

        bond_list = []

        for bond in kekul_mol.GetBonds():
            bond_start = bond.GetBeginAtomIdx()
            bond_end = bond.GetEndAtomIdx()

            # Keep upper tri only (start < end)
            if bond_start > bond_end:
                bond_start, bond_end = bond_end, bond_start

            bond_type = bond.GetBondType()
            is_arom = bond.GetIsAromatic()

            # Read bond direction from the original (non-kekulized) mol for E/Z stereo
            orig_bond = mol.GetBondBetweenAtoms(bond_start, bond_end)
            direction = orig_bond.GetBondDir() if orig_bond is not None else Chem.BondDir.NONE

            bond_index = BondEncoding.encode(bond_type, is_arom, direction=direction)
            bond_list.append([bond_start, bond_end, bond_index])

        bonds = BondSet(np.array(bond_list))
        return bonds

    @staticmethod
    def from_biotite(bond_list: BondList) -> BondSet:
        """Create a BondSet from a biotite BondList.

        Biotite BondType values:
            0: ANY, 1: SINGLE, 2: DOUBLE, 3: TRIPLE, 4: QUADRUPLE,
            5: AROMATIC_SINGLE, 6: AROMATIC_DOUBLE, 7: AROMATIC_TRIPLE,
            8: COORDINATION, 9: AROMATIC (generic)
        """

        if bond_list is None or len(bond_list.as_array()) == 0:
            return BondSet(np.zeros((0, 3), dtype=np.int16))

        bond_arr = bond_list.as_array()

        # Map biotite bond types to (RDKit BondType, is_aromatic)
        # Unsupported types (ANY=0, QUADRUPLE=4, COORDINATION=8) will raise an error
        biotite_to_rdkit = {
            1: (Chem.BondType.SINGLE, False),  # SINGLE
            2: (Chem.BondType.DOUBLE, False),  # DOUBLE
            3: (Chem.BondType.TRIPLE, False),  # TRIPLE
            5: (Chem.BondType.SINGLE, True),  # AROMATIC_SINGLE
            6: (Chem.BondType.DOUBLE, True),  # AROMATIC_DOUBLE
            7: (Chem.BondType.TRIPLE, True),  # AROMATIC_TRIPLE
            9: (Chem.BondType.SINGLE, True),  # AROMATIC (generic) -> treat as aromatic single
        }

        converted_bonds = []
        for bond in bond_arr:
            start, end, bond_type = bond[0], bond[1], bond[2]

            if start > end:
                start, end = end, start

            if bond_type not in biotite_to_rdkit:
                raise ValueError(f"Unsupported biotite bond type {bond_type}.")

            rdkit_type, is_aromatic = biotite_to_rdkit[bond_type]
            bond_index = BondEncoding.encode(rdkit_type, is_aromatic)
            converted_bonds.append([start, end, bond_index])

        return BondSet(np.array(converted_bonds, dtype=np.int16))

    @staticmethod
    def from_dict(dict_repr) -> BondSet:
        bonds = dict_repr["bonds"]
        return BondSet(bonds)

    def to_dict(self) -> dict[str, np.ndarray]:
        dict_repr = {"bonds": self.bonds}
        return dict_repr

    @staticmethod
    def bonds_from_arrays(array_map: dict[str, np.ndarray | h5py.Dataset]) -> list[BondSet]:
        """Convert the merged arrays from arrays_from_bonds back into BondSets."""

        check_dict_key(array_map, "sizes", "bond array map")
        check_dict_key(array_map, "bonds", "bond array map")

        bond_arr = array_map["bonds"]
        is_hdf5 = isinstance(bond_arr, h5py.Dataset)

        if (not is_hdf5) and (not isinstance(bond_arr, np.ndarray)):
            raise TypeError(f"Bond array must be either np.ndarray or h5py.Dataset, got {type(bond_arr)}")

        sizes = np.array(array_map["sizes"][()]).tolist()
        bond_arr = bond_arr.copy() if not is_hdf5 else bond_arr

        curr_idx = 0
        bond_sets = []

        for n_bonds in sizes:
            if is_hdf5:
                bonds = LazyData(bond_arr, curr_idx, n_bonds)
            else:
                bonds = bond_arr[curr_idx : curr_idx + n_bonds].copy()

            bond_set = BondSet(bonds)
            bond_sets.append(bond_set)
            curr_idx += n_bonds

        return bond_sets

    @staticmethod
    def arrays_from_bonds(bond_sets: list[BondSet]) -> dict[str, np.ndarray]:
        """Merge multiple bond sets into a dictionary of numpy arrays."""

        bonds = np.concatenate([bond_set.bonds for bond_set in bond_sets], axis=0)
        sizes = np.array([len(bond_set) for bond_set in bond_sets])

        arrays = {"bonds": bonds, "sizes": sizes}
        return arrays
