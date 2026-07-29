"""Meta storage helpers for GraphBatch / ProteinBatch / ComplexBatch.

Three on-disk formats exist, dispatched via the meta.attrs["format"] marker:

    - "columnar" (default): each meta key becomes its own gzip-compressed HDF5 dataset. Column
        dtype is auto-picked per column: all-int values are stored as int64, all-numeric as
        float64, anything else (mixed types, strings, or any missing rows) falls back to UTF-8
        S-dtype. Best for large datasets whose metas share a set of scalar keys, and the only
        format that can read a single key without touching the rest.
    - "json": one gzip-compressed UTF-8 JSON document per shard. Handles metas that columnar
        cannot represent faithfully -- nested dicts, lists, differing key sets -- because
        columnar would silently stringify those. This is what a non-columnar save writes.
    - "blob" (legacy, read-only): one pickled list[dict] in an attribute. Written by molito
        before the JSON format existed.

Reading a "blob" shard executes pickle, so an HDF5 file from an untrusted source could run
arbitrary code on load. Loading one therefore requires an explicit `allow_pickle=True`; the
default refuses with an error naming the file. Nothing molito writes now contains pickle.

JSON is not a byte-exact replacement for pickle: numpy arrays and numpy scalars are written
as plain JSON lists and numbers, so they come back as lists and Python scalars. Everything
JSON models natively (str, int, float, bool, None, list, dict) round-trips unchanged.

Loaded metas are always read-only Mapping objects. This matches the immutability of
the other HDF5-backed components on a loaded mol (atomics, coords, bonds all return
fresh copies on each property access). Use `dict(mol.meta)` to get a mutable copy, or
reassign `mol.meta = {...}` outright.
"""

from __future__ import annotations

import json
import pickle
from collections.abc import Iterable, Mapping
from types import MappingProxyType

import h5py
import numpy as np

FORMAT_ATTR = "format"
BLOB_FORMAT = "blob"
COLUMNAR_FORMAT = "columnar"
JSON_FORMAT = "json"

_BLOB_ATTR = "metas"
_COLUMNAR_GROUP = "columns"
_JSON_DATASET = "json"


# ************************
# *** Saving functions ***
# ************************


def save_meta(parent: h5py.Group, metas: list[dict], columnar: bool) -> None:
    """Create a 'meta' subgroup under `parent` and persist `metas` in the chosen format.

    `columnar=True` writes the columnar format, `columnar=False` writes JSON. The legacy
    pickled blob format is never written -- see the module docstring.
    """

    meta_group = parent.create_group("meta", track_order=True)

    if columnar:
        meta_group.attrs[FORMAT_ATTR] = COLUMNAR_FORMAT
        _save_columnar(meta_group, metas)
    else:
        meta_group.attrs[FORMAT_ATTR] = JSON_FORMAT
        _save_json(meta_group, metas)


def _json_default(value):
    """Coerce the numpy types JSON cannot encode. Lossy for dtype -- see module docstring."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()

    raise TypeError(f"meta value of type {type(value).__name__!r} is not JSON serialisable")


def _save_json(meta_group: h5py.Group, metas: list[dict]) -> None:
    clean = [m if m is not None else {} for m in metas]
    payload = json.dumps(clean, default=_json_default).encode("utf-8")

    # A dataset rather than an attribute: attributes have size limits and no compression,
    # and shards can carry hundreds of thousands of metas.
    meta_group.create_dataset(
        _JSON_DATASET, data=np.frombuffer(payload, dtype=np.uint8), compression="gzip", compression_opts=4
    )


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


def load_meta(meta_group: h5py.Group, n_mols: int, allow_pickle: bool = False) -> list:
    """Return a list of `n_mols` read-only meta views. Format is selected by the
    group's 'format' attr.

    Every format returns read-only Mapping objects to match the immutability of other
    HDF5-backed components (atomics, coords, etc. return fresh copies per access).
    Callers that want to mutate should call `dict(mol.meta)` to get a mutable copy.

    - Columnar: each meta is a `_ColumnMetaView` reading columns from HDF5 lazily, so only
      the keys actually accessed are materialised.
    - JSON: the document is decoded once and each meta wrapped in a `MappingProxyType`.
    - Blob (legacy): unpickles, and so requires `allow_pickle=True`.

    Args:
        meta_group: The 'meta' group to read.
        n_mols: Expected number of entries, used to validate the stored data.
        allow_pickle: Permit loading a legacy pickled blob shard. Off by default because
            unpickling a file from an untrusted source can execute arbitrary code.

    Raises:
        ValueError: If the shard is blob format and `allow_pickle` is False, or if the
            format marker is unrecognised.
    """

    fmt = meta_group.attrs.get(FORMAT_ATTR)
    if isinstance(fmt, bytes):
        fmt = fmt.decode()

    if fmt == COLUMNAR_FORMAT:
        return _load_columnar(meta_group, n_mols)
    if fmt == JSON_FORMAT:
        return _load_json(meta_group, n_mols)
    if fmt == BLOB_FORMAT or fmt is None:
        return _load_blob(meta_group, n_mols, allow_pickle)

    raise ValueError(f"Unknown meta format {fmt!r} in HDF5 group {meta_group.name!r}")


def _load_json(meta_group: h5py.Group, n_mols: int) -> list:
    payload = meta_group[_JSON_DATASET][()].tobytes().decode("utf-8")
    metas = json.loads(payload)

    if len(metas) != n_mols:
        raise RuntimeError(f"Meta JSON has {len(metas)} entries but expected {n_mols}")

    return [MappingProxyType(m if m is not None else {}) for m in metas]


def _load_blob(meta_group: h5py.Group, n_mols: int, allow_pickle: bool) -> list:
    if not allow_pickle:
        raise ValueError(
            f"HDF5 group {meta_group.file.filename!r} stores metadata in the legacy pickled blob "
            f"format. Unpickling runs arbitrary code, so it is refused unless you trust the file. "
            f"Pass allow_pickle=True to load it, or re-save the dataset to write the JSON format."
        )

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
