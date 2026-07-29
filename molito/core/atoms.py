from __future__ import annotations

from collections.abc import Sequence

import h5py
import numpy as np
from biotite.structure import AtomArray
from rdkit import Chem

from molito.core._checks import check_dict_key, check_shape_len, check_shapes_equal, check_type
from molito.core.lazydata import LazyData
from molito.core.pt import PT
from molito.core.vocab import CHIRAL_NONE, CHIRAL_SUFFIXES, RDKIT_CHIRAL_TO_INT

TArr = np.ndarray

# Maximum string lengths for fixed-length string storage
MAX_RES_NAME_LEN = 3
MAX_ATOM_NAME_LEN = 4
MAX_CHAIN_ID_LEN = 4


# *** Util functions specific to AtomSet ***


def _check_annotation_consistency(atom_sets: list[AtomSet], attr_name: str) -> bool:
    """Check consistency of optional annotations - all atom sets must either have or not have each annotation"""

    has_attr = [getattr(atoms, attr_name) is not None for atoms in atom_sets]
    if any(has_attr) and not all(has_attr):
        raise ValueError(f"Inconsistent {attr_name}: some atom sets have {attr_name} but others do not")
    return all(has_attr)


def _check_string_lengths(arr: np.ndarray | LazyData, max_len: int, name: str) -> None:
    """Validate that all strings in array are within max length. Skip validation for LazyData."""

    if isinstance(arr, LazyData):
        return

    for i, s in enumerate(arr):
        if len(s) > max_len:
            raise ValueError(f"{name}[{i}] = '{s}' exceeds max length {max_len}")


class AtomSet(Sequence):
    """Set of atoms for a single molecule or protein chain.

    Stores atomic numbers, formal charges, and chirality as compact numpy arrays (uint8/int8).
    Optionally stores protein residue annotations (res_names, atom_names, res_ids).
    Supports lazy loading from HDF5 via LazyData wrappers.

    Chirality is stored as RDKit's relative parity tag (CW/CCW), not as the absolute CIP
    descriptor (R/S). The tag's meaning depends on the cyclic order in which RDKit encounters
    the chiral atom's neighbours during mol reconstruction, which in turn depends on the bond
    array's row order in `BondSet`. See the `molito.core.bonds` module docstring for the full
    rationale and the invariants that make permutation-safe round-tripping work.

    Args:
        atomics: Atomic numbers, shape [n_atoms].
        charges: Formal charges, shape [n_atoms]. Defaults to zeros.
        chirality: Chirality tags (0=none, 1=CW, 2=CCW), shape [n_atoms]. Defaults to zeros.
        res_names: Residue names (e.g. "ALA"), shape [n_atoms]. Optional, for proteins.
        atom_names: Atom names (e.g. "CA"), shape [n_atoms]. Optional, for proteins.
        res_ids: Residue IDs, shape [n_atoms]. Optional, for proteins.
        chain_ids: Chain IDs (e.g. "A"), shape [n_atoms]. Optional, for proteins.
    """

    __slots__ = ("_atomics", "_charges", "_chirality", "_res_names", "_atom_names", "_res_ids", "_chain_ids")

    def __init__(
        self,
        atomics: TArr | LazyData,
        charges: TArr | LazyData | None = None,
        chirality: TArr | LazyData | None = None,
        res_names: TArr | LazyData | None = None,
        atom_names: TArr | LazyData | None = None,
        res_ids: TArr | LazyData | None = None,
        chain_ids: TArr | LazyData | None = None,
    ):
        check_type(atomics, [np.ndarray, LazyData], "atomics")
        check_shape_len(atomics, 1, "atomics")

        # Default charges and chirality to 0 if not provided
        charges = np.zeros(len(atomics), dtype=np.int8) if charges is None else charges
        chirality = np.zeros(len(atomics), dtype=np.int8) if chirality is None else chirality

        check_type(charges, [np.ndarray, LazyData], "charges")
        check_shape_len(charges, 1, "charges")
        check_shapes_equal(atomics, charges, 0)

        check_type(chirality, [np.ndarray, LazyData], "chirality")
        check_shape_len(chirality, 1, "chirality")

        # Validate shapes of optional residue annotations if provided
        if res_names is not None:
            check_shape_len(res_names, 1, "res names")
            check_shapes_equal(atomics, res_names, 0)
            _check_string_lengths(res_names, MAX_RES_NAME_LEN, "res_names")

        if atom_names is not None:
            check_shape_len(atom_names, 1, "atom names")
            check_shapes_equal(atomics, atom_names, 0)
            _check_string_lengths(atom_names, MAX_ATOM_NAME_LEN, "atom_names")

        if res_ids is not None:
            check_shape_len(res_ids, 1, "res ids")
            check_shapes_equal(atomics, res_ids, 0)

        if chain_ids is not None:
            check_shape_len(chain_ids, 1, "chain ids")
            check_shapes_equal(atomics, chain_ids, 0)
            _check_string_lengths(chain_ids, MAX_CHAIN_ID_LEN, "chain_ids")

        # Cast to compact dtypes if data is already in memory
        atomics = atomics.astype(np.uint8) if isinstance(atomics, np.ndarray) else atomics
        charges = charges.astype(np.int8) if isinstance(charges, np.ndarray) else charges
        chirality = chirality.astype(np.int8) if isinstance(chirality, np.ndarray) else chirality

        if res_ids is not None and isinstance(res_ids, np.ndarray):
            res_ids = res_ids.astype(np.int32)

        self._atomics = atomics
        self._charges = charges
        self._chirality = chirality

        # Optional residue-level annotations (used for proteins)
        self._res_names = res_names
        self._atom_names = atom_names
        self._res_ids = res_ids
        self._chain_ids = chain_ids

    @classmethod
    def _load_unchecked(
        cls, atomics, charges, chirality, res_names=None, atom_names=None, res_ids=None, chain_ids=None
    ) -> AtomSet:
        """Fast-path factory for HDF5 load. Skips validation and dtype casts; caller is
        responsible for passing correctly-shaped LazyData / ndarray objects."""

        obj = cls.__new__(cls)
        obj._atomics = atomics
        obj._charges = charges
        obj._chirality = chirality
        obj._res_names = res_names
        obj._atom_names = atom_names
        obj._res_ids = res_ids
        obj._chain_ids = chain_ids
        return obj

    @classmethod
    def _iter_from_group(cls, group: h5py.Group):
        """Yield one AtomSet per mol from an HDF5 'atoms' group.

        Opens all dataset refs once, then walks the 'sizes' array to slice out each mol's
        rows via LazyData wrappers. This is the hot loader used by GraphBatch / ProteinBatch.
        """

        sizes = np.asarray(group["sizes"][()]).tolist()
        atomics_ds = group["atomics"]
        charges_ds = group["charges"]
        chirality_ds = group.get("chirality")
        res_names_ds = group.get("res_names")
        atom_names_ds = group.get("atom_names")
        res_ids_ds = group.get("res_ids")
        chain_ids_ds = group.get("chain_ids")

        offset = 0
        for n in sizes:
            yield cls._load_unchecked(
                LazyData._load_unchecked(atomics_ds, offset, n),
                LazyData._load_unchecked(charges_ds, offset, n),
                LazyData._load_unchecked(chirality_ds, offset, n) if chirality_ds is not None else None,
                res_names=LazyData._load_unchecked(res_names_ds, offset, n) if res_names_ds is not None else None,
                atom_names=LazyData._load_unchecked(atom_names_ds, offset, n) if atom_names_ds is not None else None,
                res_ids=LazyData._load_unchecked(res_ids_ds, offset, n) if res_ids_ds is not None else None,
                chain_ids=LazyData._load_unchecked(chain_ids_ds, offset, n) if chain_ids_ds is not None else None,
            )
            offset += n

    # *** Publicly exposed properties ***

    @property
    def atomics(self) -> TArr:
        """Returns array of shape [n_atoms,]"""

        if isinstance(self._atomics, LazyData):
            return self._atomics.read().astype(np.uint8)

        return self._atomics

    @property
    def charges(self) -> TArr:
        """Returns array of shape [n_atoms,]"""

        if isinstance(self._charges, LazyData):
            return self._charges.read().astype(np.int8)

        return self._charges

    @property
    def chirality(self) -> TArr:
        """Returns array of shape [n_atoms,]"""

        if isinstance(self._chirality, LazyData):
            return self._chirality.read().astype(np.int8)

        return self._chirality

    @property
    def charged_symbols(self) -> list[str]:
        """Returns atom symbols with charge (e.g. 'C_0', 'N_-1'). Does not include chirality."""

        atoms = [PT.symbol_from_atomic(a) for a in self.atomics.tolist()]
        return [f"{atom}_{charge}" for atom, charge in zip(atoms, self.charges.tolist(), strict=True)]

    @property
    def tokens(self) -> list[str]:
        """Returns atom tokens encoding element, charge, and chirality (e.g. 'C_0', 'C_0_CW')."""

        symbols = [PT.symbol_from_atomic(a) for a in self.atomics.tolist()]
        charges = self.charges.tolist()
        chirals = self.chirality.tolist()

        result = []
        for sym, charge, chiral in zip(symbols, charges, chirals, strict=True):
            token = f"{sym}_{charge}"
            if chiral in CHIRAL_SUFFIXES:
                token = f"{token}_{CHIRAL_SUFFIXES[chiral]}"

            result.append(token)

        return result

    @property
    def res_names(self) -> TArr | None:
        """Returns array of residue names, shape [n_atoms,], or None if not set."""

        if self._res_names is None:
            return None

        if isinstance(self._res_names, LazyData):
            arr = self._res_names.read()
            return AtomSet._decode_bytes(arr)

        return self._res_names

    @property
    def atom_names(self) -> TArr | None:
        """Returns array of atom names, shape [n_atoms,], or None if not set."""

        if self._atom_names is None:
            return None

        if isinstance(self._atom_names, LazyData):
            arr = self._atom_names.read()
            return AtomSet._decode_bytes(arr)

        return self._atom_names

    @property
    def res_ids(self) -> TArr | None:
        """Returns array of residue IDs, shape [n_atoms,], or None if not set."""

        if self._res_ids is None:
            return None

        if isinstance(self._res_ids, LazyData):
            return self._res_ids.read().astype(np.int32)

        return self._res_ids

    @property
    def chain_ids(self) -> TArr | None:
        """Returns array of chain IDs, shape [n_atoms,], or None if not set."""

        if self._chain_ids is None:
            return None

        if isinstance(self._chain_ids, LazyData):
            arr = self._chain_ids.read()
            return AtomSet._decode_bytes(arr)

        return self._chain_ids

    @property
    def has_residue_annotations(self) -> bool:
        """Returns True if all residue-level annotations (res_names, atom_names, res_ids) are present."""

        return all([self._res_names is not None, self._atom_names is not None, self._res_ids is not None])

    @property
    def seq_length(self) -> int:
        return len(self)

    # *** Basic indexing and utility functions ***

    def __len__(self) -> int:
        return len(self._atomics)

    def __getitem__(self, index: int | TArr) -> AtomSet | tuple[int, int, int]:
        """Index by int returns (atomic, charge, chirality) tuple. Index by array returns a new AtomSet."""

        if isinstance(index, int):
            return self.atomics[index].item(), self.charges[index].item(), self.chirality[index].item()

        if isinstance(index, np.ndarray):
            res_names = self.res_names[index] if self.res_names is not None else None
            atom_names = self.atom_names[index] if self.atom_names is not None else None
            res_ids = self.res_ids[index] if self.res_ids is not None else None
            chain_ids = self.chain_ids[index] if self.chain_ids is not None else None

            atoms = self.copy_with(
                atomics=self.atomics[index],
                charges=self.charges[index],
                chirality=self.chirality[index],
                res_names=res_names,
                atom_names=atom_names,
                res_ids=res_ids,
                chain_ids=chain_ids,
            )
            return atoms

        raise TypeError("index must be either an int or an np array.")

    def read(self) -> AtomSet:
        """Force the data to read into memory if it's not already"""

        atoms = AtomSet(
            self.atomics,
            charges=self.charges,
            chirality=self.chirality,
            res_names=self.res_names,
            atom_names=self.atom_names,
            res_ids=self.res_ids,
            chain_ids=self.chain_ids,
        )
        return atoms

    def copy_with(
        self,
        atomics: TArr | None = None,
        charges: TArr | None = None,
        chirality: TArr | None = None,
        res_names: TArr | None = None,
        atom_names: TArr | None = None,
        res_ids: TArr | None = None,
        chain_ids: TArr | None = None,
    ) -> AtomSet:

        atomics = self.atomics.copy() if atomics is None else atomics.copy()
        charges = self.charges.copy() if charges is None else charges.copy()
        chirality = self.chirality.copy() if chirality is None else chirality.copy()

        if res_names is None and self.res_names is not None:
            res_names = self.res_names.copy()
        if atom_names is None and self.atom_names is not None:
            atom_names = self.atom_names.copy()
        if res_ids is None and self.res_ids is not None:
            res_ids = self.res_ids.copy()
        if chain_ids is None and self.chain_ids is not None:
            chain_ids = self.chain_ids.copy()

        atoms = AtomSet(
            atomics,
            charges=charges,
            chirality=chirality,
            res_names=res_names,
            atom_names=atom_names,
            res_ids=res_ids,
            chain_ids=chain_ids,
        )
        return atoms

    def copy(self) -> AtomSet:
        return self.copy_with()

    def permute_atoms(self, indices: list[int] | TArr) -> AtomSet:
        """Used for permuting atom order. Can be used for reordering or taking a subset, but not for duplicating."""

        indices = np.array(indices)

        if len(set(indices.tolist())) != len(indices.tolist()):
            raise ValueError("Indices list cannot contain duplicates.")

        if indices.min().item() < 0:
            raise ValueError("Indices cannot be negative.")

        if indices.max().item() >= self.seq_length:
            raise ValueError(f"Index {max(indices)} is out of bounds for atom set with {self.seq_length} atoms.")

        atoms = self[indices]
        return atoms

    def pad(
        self,
        n_atoms: int,
        pad_atomic: int = 0,
        pad_charge: int = 0,
        pad_res_name: str = "PAD",
        pad_atom_name: str = "PAD",
        pad_res_id: int = -1,
        pad_chain_id: str = "PAD",
    ) -> AtomSet:
        """Pad the atoms to length n_atoms"""

        if n_atoms < len(self):
            raise ValueError("Cannot pad to fewer atoms than exist in the molecule.")

        if n_atoms == len(self):
            return self.copy()

        n_pad = n_atoms - len(self)
        atomics = np.concatenate((self.atomics, np.full(n_pad, pad_atomic, dtype=np.uint8)))
        charges = np.concatenate((self.charges, np.full(n_pad, pad_charge, dtype=np.int8)))
        chirality = np.concatenate((self.chirality, np.zeros(n_pad, dtype=np.int8)))

        res_names = None
        atom_names = None
        res_ids = None
        chain_ids = None

        if self.res_names is not None:
            res_names = np.concatenate((self.res_names, np.array([pad_res_name] * n_pad)))
        if self.atom_names is not None:
            atom_names = np.concatenate((self.atom_names, np.array([pad_atom_name] * n_pad)))
        if self.res_ids is not None:
            res_ids = np.concatenate((self.res_ids, np.array([pad_res_id] * n_pad)))
        if self.chain_ids is not None:
            chain_ids = np.concatenate((self.chain_ids, np.array([pad_chain_id] * n_pad)))

        padded = self.copy_with(
            atomics=atomics,
            charges=charges,
            chirality=chirality,
            res_names=res_names,
            atom_names=atom_names,
            res_ids=res_ids,
            chain_ids=chain_ids,
        )
        return padded

    # *** IO and conversion utility functions ***

    @staticmethod
    def from_rdkit(mol: Chem.rdchem.Mol, clean_stereo: bool = False) -> AtomSet:
        """Build an AtomSet from an RDKit molecule.

        Args:
            mol: source RDKit molecule.
            clean_stereo: if True, copy `mol` and run
                `Chem.AssignStereochemistry(cleanIt=True, force=True)` on the copy
                before reading chiral tags. This strips `CHI_TETRAHEDRAL_CW/CCW`
                tags from atoms that don't actually carry CIP-resolvable stereo
                (equivalent neighbours, broken symmetry, draw-time artifacts in
                SMILES). Useful when ingesting datasets where the source SMILES
                or SDFs over-declare stereo: downstream consumers that key off
                `atoms.chirality` (e.g. inference-time chirality guidance) will
                otherwise fire on these "ghost" centres. Input mol is never
                mutated. Default False keeps tags verbatim.
        """

        if clean_stereo:
            mol = Chem.Mol(mol)
            Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

        atomics = []
        charges = []
        chirals = []

        for atom in mol.GetAtoms():
            atomics.append(atom.GetAtomicNum())
            charges.append(atom.GetFormalCharge())

            tag = atom.GetChiralTag()
            chirals.append(RDKIT_CHIRAL_TO_INT.get(tag, CHIRAL_NONE))

        atomics = np.array(atomics, dtype=np.uint8)
        charges = np.array(charges, dtype=np.int8)
        chirality = np.array(chirals, dtype=np.int8)

        atoms = AtomSet(atomics, charges=charges, chirality=chirality)
        return atoms

    @staticmethod
    def from_biotite(atom_array: AtomArray) -> AtomSet:
        # Biotite returns uppercase element strings from RCSB CIFs ("MG", "ZN", "SE", ...),
        # while RDKit's periodic table is case-sensitive and only accepts title-case for
        # multi-letter symbols. Normalise so any element parses, not just single-letter ones.
        elements = [el if len(el) == 1 else el.capitalize() for el in atom_array.element.tolist()]
        atomics = np.array([PT.atomic_from_symbol(el) for el in elements])

        charges = np.zeros(len(atomics), dtype=np.int8)
        if "charge" in atom_array.get_annotation_categories():
            charges = atom_array.charge.astype(np.int8)

        res_names = atom_array.res_name
        atom_names = atom_array.atom_name
        res_ids = atom_array.res_id
        chain_ids = atom_array.chain_id

        atoms = AtomSet(
            atomics, charges=charges, res_names=res_names, atom_names=atom_names, res_ids=res_ids, chain_ids=chain_ids
        )
        return atoms

    @staticmethod
    def from_dict(dict_repr) -> AtomSet:
        check_dict_key(dict_repr, "atomics")
        check_dict_key(dict_repr, "charges")

        atoms = AtomSet(
            dict_repr["atomics"],
            charges=dict_repr["charges"],
            chirality=dict_repr.get("chirality"),
            res_names=dict_repr.get("res_names"),
            atom_names=dict_repr.get("atom_names"),
            res_ids=dict_repr.get("res_ids"),
            chain_ids=dict_repr.get("chain_ids"),
        )
        return atoms

    def to_dict(self) -> dict[str, np.ndarray]:
        d = {"atomics": self.atomics, "charges": self.charges, "chirality": self.chirality}

        if self.res_names is not None:
            d["res_names"] = self.res_names
        if self.atom_names is not None:
            d["atom_names"] = self.atom_names
        if self.res_ids is not None:
            d["res_ids"] = self.res_ids
        if self.chain_ids is not None:
            d["chain_ids"] = self.chain_ids

        return d

    @staticmethod
    def atoms_from_arrays(array_map: dict[str, np.ndarray | h5py.Dataset]) -> list[AtomSet]:
        """Convert merged arrays back into AtomSets"""

        check_dict_key(array_map, "sizes", "atom array map")
        check_dict_key(array_map, "atomics", "atom array map")
        check_dict_key(array_map, "charges", "atom array map")

        sizes = np.array(array_map["sizes"][()]).tolist()
        split_arr_map = {name: AtomSet._split_array(arr, sizes) for name, arr in array_map.items()}

        atomics_arrs = split_arr_map["atomics"]
        charges_arrs = split_arr_map["charges"]
        chirality_arrs = split_arr_map.get("chirality")
        res_names_arrs = split_arr_map.get("res_names")
        atom_names_arrs = split_arr_map.get("atom_names")
        res_ids_arrs = split_arr_map.get("res_ids")
        chain_ids_arrs = split_arr_map.get("chain_ids")

        if res_names_arrs is not None:
            res_names_arrs = [AtomSet._decode_bytes(arr) for arr in res_names_arrs]
        if atom_names_arrs is not None:
            atom_names_arrs = [AtomSet._decode_bytes(arr) for arr in atom_names_arrs]
        if chain_ids_arrs is not None:
            chain_ids_arrs = [AtomSet._decode_bytes(arr) for arr in chain_ids_arrs]

        atom_sets = []
        for idx in range(len(sizes)):
            atom_sets.append(
                AtomSet(
                    atomics_arrs[idx],
                    charges=charges_arrs[idx],
                    chirality=chirality_arrs[idx] if chirality_arrs is not None else None,
                    res_names=res_names_arrs[idx] if res_names_arrs is not None else None,
                    atom_names=atom_names_arrs[idx] if atom_names_arrs is not None else None,
                    res_ids=res_ids_arrs[idx] if res_ids_arrs is not None else None,
                    chain_ids=chain_ids_arrs[idx] if chain_ids_arrs is not None else None,
                )
            )

        return atom_sets

    @staticmethod
    def arrays_from_atoms(atom_sets: list[AtomSet]) -> dict[str, np.ndarray]:
        """Merge multiple atom sets into a dictionary of arrays."""

        arrays = {
            "atomics": np.concatenate([a.atomics for a in atom_sets]).astype(np.uint8),
            "charges": np.concatenate([a.charges for a in atom_sets]).astype(np.int8),
            "chirality": np.concatenate([a.chirality for a in atom_sets]).astype(np.int8),
            "sizes": np.array([len(a) for a in atom_sets]),
        }

        all_have_res_names = _check_annotation_consistency(atom_sets, "res_names")
        all_have_atom_names = _check_annotation_consistency(atom_sets, "atom_names")
        all_have_res_ids = _check_annotation_consistency(atom_sets, "res_ids")
        all_have_chain_ids = _check_annotation_consistency(atom_sets, "chain_ids")

        if all_have_res_names:
            arrays["res_names"] = np.concatenate([a.res_names for a in atom_sets]).astype(f"S{MAX_RES_NAME_LEN}")
        if all_have_atom_names:
            arrays["atom_names"] = np.concatenate([a.atom_names for a in atom_sets]).astype(f"S{MAX_ATOM_NAME_LEN}")
        if all_have_res_ids:
            arrays["res_ids"] = np.concatenate([a.res_ids for a in atom_sets]).astype(np.int32)
        if all_have_chain_ids:
            arrays["chain_ids"] = np.concatenate([a.chain_ids for a in atom_sets]).astype(f"S{MAX_CHAIN_ID_LEN}")

        return arrays

    @staticmethod
    def _split_array(arr: np.ndarray | h5py.Dataset, sizes: list[int]) -> list[LazyData | np.ndarray]:
        is_hdf5 = isinstance(arr, h5py.Dataset)

        if (not is_hdf5) and (not isinstance(arr, np.ndarray)):
            raise TypeError(f"Each array must be either np.ndarray or h5py.Dataset, got {type(arr)}")

        arr = arr.copy() if not is_hdf5 else arr
        curr_idx = 0
        splits = []

        for n_atoms in sizes:
            if is_hdf5:
                splits.append(LazyData(arr, curr_idx, n_atoms))
            else:
                splits.append(arr[curr_idx : curr_idx + n_atoms])

            curr_idx += n_atoms

        return splits

    @staticmethod
    def _decode_bytes(arr: np.ndarray | LazyData | None) -> np.ndarray | LazyData | None:
        """Decode fixed-length byte string arrays to unicode strings. Returns None or LazyData unchanged."""

        if arr is None or isinstance(arr, LazyData):
            return arr

        if arr.dtype.kind == "S":
            return arr.astype("U")

        return arr
