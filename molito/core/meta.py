"""Meta storage helpers for GraphBatch / ProteinBatch / ComplexBatch.

Two on-disk formats are supported, dispatched via the meta.attrs["format"] marker:
    - "blob": one pickled list[dict] stored as a single attribute. Useful when batching mols with very
        different sets of meta keys.
    - "columnar": each meta key becomes its own gzip-compressed HDF5 dataset. Column dtype is
        auto-picked per column: all-int values are stored as int64, all-numeric as float64, anything
        else (mixed types, strings, or any missing rows) falls back to UTF-8 S-dtype. Useful for very
        large datasets where all members have the same set of keys in meta dicts.

Loaded metas are always read-only Mapping objects. This matches the immutability of
the other HDF5-backed components on a loaded mol (atomics, coords, bonds all return
fresh copies on each property access). Use `dict(mol.meta)` to get a mutable copy, or
reassign `mol.meta = {...}` outright.
"""

from __future__ import annotations

import pickle
from collections.abc import Iterable, Mapping
from types import MappingProxyType

import h5py
import numpy as np

from molito.core._checks import PICKLE_PROTOCOL

FORMAT_ATTR = "format"
BLOB_FORMAT = "blob"
COLUMNAR_FORMAT = "columnar"

_BLOB_ATTR = "metas"
_COLUMNAR_GROUP = "columns"


# ************************
# *** Saving functions ***
# ************************


def save_meta(parent: h5py.Group, metas: list[dict], columnar: bool) -> None:
    """Create a 'meta' subgroup under `parent` and persist `metas` in the chosen format."""

    meta_group = parent.create_group("meta", track_order=True)

    if columnar:
        meta_group.attrs[FORMAT_ATTR] = COLUMNAR_FORMAT
        _save_columnar(meta_group, metas)
    else:
        meta_group.attrs[FORMAT_ATTR] = BLOB_FORMAT
        _save_blob(meta_group, metas)


def _save_blob(meta_group: h5py.Group, metas: list[dict]) -> None:
    # Store as a single pickled list of dicts. Empty dict is used as sentinel for None.
    clean = [m if m is not None else {} for m in metas]
    meta_group.attrs[_BLOB_ATTR] = np.void(pickle.dumps(clean, protocol=PICKLE_PROTOCOL))


def _save_columnar(meta_group: h5py.Group, metas: list[dict]) -> None:
    clean = [m if m is not None else {} for m in metas]

    keys = _union_keys(clean)
    cols_group = meta_group.create_group(_COLUMNAR_GROUP, track_order=True)

    for key in keys:
        # Sentinel to track which mols had the key absent. Auto-promotion to numeric
        # dtypes is only allowed if every mol has the key set (no missing values);
        # otherwise we fall back to S-dtype with empty-string fill, matching the
        # original behaviour.
        raw_values = [m.get(key, _MISSING) for m in clean]
        arr = _column_to_array(raw_values)
        cols_group.create_dataset(key, data=arr, compression="gzip", compression_opts=4)


_MISSING = object()  # sentinel for absent meta keys (distinct from any user value)


def _column_to_array(values: list) -> np.ndarray:
    """Coerce a column of meta values to the narrowest numpy dtype that fits all rows.

    Promotion rules (only applied when no rows are missing the key):
    - All Python ints (excluding bool) → int64 dataset
    - All numeric (int or float, excluding bool) → float64 dataset
    - Anything else, or any missing values → S-dtype with UTF-8 encoding (fallback to
      the legacy string-of-everything behaviour, including empty-string fill for
      missing values).
    """

    has_missing = any(v is _MISSING for v in values)

    if not has_missing and values:
        if all(_is_int_like(v) for v in values):
            return np.array([int(v) for v in values], dtype=np.int64)

        if all(_is_float_like(v) for v in values):
            return np.array([float(v) for v in values], dtype=np.float64)

    # Fallback: encode everything as UTF-8 bytes. Non-ASCII chars (µ, Å) round-trip
    # correctly via UTF-8; the S-dtype is byte-oriented and would raise on raw str
    # with non-ASCII otherwise.
    str_values = [str(v if v is not _MISSING else "").encode("utf-8") for v in values]
    max_len = max((len(s) for s in str_values), default=0) or 1
    return np.array(str_values, dtype=f"S{max_len}")


def _is_int_like(v) -> bool:
    # bool is a subclass of int in Python; exclude it so True/False don't silently
    # collapse to 1/0 on disk.
    if isinstance(v, bool):
        return False
    return isinstance(v, (int, np.integer))


def _is_float_like(v) -> bool:
    if isinstance(v, bool):
        return False
    return isinstance(v, (int, float, np.integer, np.floating))


def _union_keys(metas: list[dict]) -> list[str]:
    """Return the stable-ordered union of keys across all metas.

    The first mol's keys come first (preserving its order), then any keys from later mols
    that weren't already seen are appended in first-seen order.
    """

    seen: dict[str, None] = {}
    for m in metas:
        for k in m:
            if k not in seen:
                seen[k] = None

    return list(seen)


# *************************
# *** Loading functions ***
# *************************


def load_meta(meta_group: h5py.Group, n_mols: int) -> list:
    """Return a list of `n_mols` read-only meta views. Format is selected by the
    group's 'format' attr.

    Both formats return read-only Mapping objects to match the immutability of other
    HDF5-backed components (atomics, coords, etc. return fresh copies per access).
    Callers that want to mutate should call `dict(mol.meta)` to get a mutable copy.

    - Blob format: each meta is a pre-loaded `types.MappingProxyType` over a dict.
    - Columnar format: each meta is a `_ColumnMetaView` that reads columns from HDF5
      lazily (only the keys actually accessed are materialised).
    """

    fmt = meta_group.attrs.get(FORMAT_ATTR)
    if isinstance(fmt, bytes):
        fmt = fmt.decode()

    if fmt == COLUMNAR_FORMAT:
        return _load_columnar(meta_group, n_mols)
    if fmt == BLOB_FORMAT or fmt is None:
        return _load_blob(meta_group, n_mols)

    raise ValueError(f"Unknown meta format {fmt!r} in HDF5 group {meta_group.name!r}")


def _load_blob(meta_group: h5py.Group, n_mols: int) -> list:
    blob = meta_group.attrs[_BLOB_ATTR].tobytes()
    metas = pickle.loads(blob)

    if len(metas) != n_mols:
        raise RuntimeError(f"Meta blob has {len(metas)} entries but expected {n_mols}")

    # Wrap each in a read-only view so loaded metas are immutable -- matches the
    # behaviour of the other HDF5-backed components (coords, atomics, etc. return fresh
    # copies per access). Callers who want to mutate should do `dict(mol.meta)` first.
    return [MappingProxyType(m if m is not None else {}) for m in metas]


def _load_columnar(meta_group: h5py.Group, n_mols: int) -> list:
    cols_group = meta_group[_COLUMNAR_GROUP]
    columns = {name: ds for name, ds in cols_group.items()}

    if not columns:
        return [{} for _ in range(n_mols)]

    # Sanity check that all columns agree on row count
    for name, ds in columns.items():
        if ds.shape[0] != n_mols:
            raise RuntimeError(f"Columnar meta column {name!r} has {ds.shape[0]} rows but expected {n_mols}")

    return [_ColumnMetaView(columns, i) for i in range(n_mols)]


def column_array(meta_group: h5py.Group, key: str) -> np.ndarray:
    """Read a single meta column as an ndarray.

    The dtype follows whatever the column was saved as: int64 / float64 columns come back
    with native numeric dtype, S-dtype columns are decoded from UTF-8 to a unicode ndarray.

    Only supported for columnar-format groups. Blob-format groups raise KeyError since there
    is no way to read one key cheaply without unpickling the whole blob.
    """

    fmt = meta_group.attrs.get(FORMAT_ATTR)
    if isinstance(fmt, bytes):
        fmt = fmt.decode()

    if fmt != COLUMNAR_FORMAT:
        raise KeyError(f"column_array requires columnar meta format; group has format {fmt!r}")

    ds = meta_group[_COLUMNAR_GROUP][key]
    raw = ds[:]

    # Fixed-length S arrays come back as bytes; decode as UTF-8 (matches _save_columnar).
    # Can't use `.astype("U")` here -- numpy treats S bytes as latin-1, which mangles multi-byte chars.
    if raw.dtype.kind == "S":
        return np.char.decode(raw, "utf-8")

    return raw


def materialise_meta(meta: Mapping | None) -> dict:
    """Convert any meta representation (dict / proxy / columnar view / None) to a
    fresh mutable dict. Empty/None -> {}.
    """

    if meta is None:
        return {}
    return dict(meta)


# ***********************
# *** Lazy view class ***
# ***********************


class _ColumnMetaView(Mapping):
    """A read-only dict-like view backed by columnar HDF5 datasets.

    Values are read lazily on first access and cached per-instance. Callers who want a
    mutable copy should do `dict(mol.meta)`. Matches the immutability of other
    HDF5-backed components on a loaded mol.
    """

    __slots__ = ("_columns", "_idx", "_cache")

    def __init__(self, columns: dict[str, h5py.Dataset], idx: int):
        self._columns = columns
        self._idx = idx
        self._cache: dict = {}

    def __getitem__(self, key: str):
        if key in self._cache:
            return self._cache[key]
        if key not in self._columns:
            raise KeyError(key)

        ds = self._columns[key]
        raw = ds[self._idx]

        # Dispatch on dataset dtype: numeric columns return native ints/floats; S
        # (fixed-length bytes) columns decode as UTF-8 to preserve non-ASCII chars.
        kind = ds.dtype.kind
        if kind == "S":
            v = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
        elif kind in ("i", "u"):
            v = int(raw)
        elif kind == "f":
            v = float(raw)
        else:
            v = raw

        self._cache[key] = v
        return v

    def __iter__(self) -> Iterable[str]:
        return iter(self._columns)

    def __len__(self) -> int:
        return len(self._columns)

    def __contains__(self, key) -> bool:
        return key in self._columns

    def __eq__(self, other) -> bool:
        if isinstance(other, Mapping):
            return dict(self) == dict(other)

        return NotImplemented

    def __repr__(self) -> str:
        return f"_ColumnMetaView(keys={list(self._columns)}, idx={self._idx})"
