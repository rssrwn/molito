from __future__ import annotations

from collections.abc import Sequence
from contextlib import ExitStack
from operator import index as integer_index
from pathlib import Path
from typing import TYPE_CHECKING, Generic, TypeVar, overload

import h5py
import numpy as np
from rdkit import Chem, rdBase

from molito.core.format import check_format, stamp_format
from molito.core.lazydata import LazyData
from molito.core.meta import column_array, load_meta, save_meta

from .base import ConversionError, MolRepr
from .rdkit import RDKitMol
from .string import SmilesMol, StringMol

if TYPE_CHECKING:
    from .graph import GraphBatch

TMol = TypeVar("TMol", bound=MolRepr)
TTarget = TypeVar("TTarget", bound=MolRepr)


class MolBatch(Sequence[TMol], Generic[TMol]):
    """Homogeneous native molecule collection with sharded HDF5 persistence.

    Supports StringMol subclasses and RDKitMol. Graph collections use GraphBatch,
    including when `.to(GraphMol)` is called. Loaded molecule payloads stay on disk until
    accessed. Keep the owning batch open, or call read() to detach its data into memory.
    Unknown string subclasses require `mol_type` at load.
    """

    def __init__(self, mols: Sequence[TMol], mol_type: type[TMol] | None = None):
        self._mols = list(mols)
        if mol_type is None:
            if not self._mols:
                raise ValueError("Empty batches require an explicit mol_type.")

            mol_type = type(self._mols[0])

        if not issubclass(mol_type, (StringMol, RDKitMol)):
            raise TypeError("MolBatch stores string or RDKit representations; use GraphBatch for graphs.")

        if any(type(mol) is not mol_type for mol in self._mols):
            raise TypeError("All batch entries must have the same representation type.")

        self.mol_type = mol_type
        self._open_fps = []

    def __len__(self) -> int:
        return len(self._mols)

    @overload
    def __getitem__(self, index: int) -> TMol: ...

    @overload
    def __getitem__(self, index: slice) -> list[TMol]: ...

    def __getitem__(self, index: int | slice) -> TMol | list[TMol]:
        return self._mols[index]

    def subset(self, idxs: Sequence[int]) -> MolBatch[TMol]:
        """Return borrowed molecule wrappers. Closing this subset leaves its source files open."""

        return MolBatch([self._mols[idx] for idx in idxs], self.mol_type)

    def meta_column(self, key: str) -> np.ndarray:
        return np.array([mol.meta.get(key, "") for mol in self._mols])

    def to(self, target: type[TTarget], strict: bool = False, **kwargs) -> MolBatch[TTarget] | GraphBatch:
        return self.convert_mols(self._mols, target, strict=strict, **kwargs)

    @staticmethod
    def convert_mols(
        mols: Sequence[MolRepr], target: type[TTarget], strict: bool = False, **kwargs
    ) -> MolBatch[TTarget] | GraphBatch:
        """Convert without skipping records; errors identify the failing batch index."""

        from .graph import GraphBatch, GraphMol

        if not isinstance(target, type) or not issubclass(target, MolRepr):
            raise TypeError("target must be a MolRepr subclass.")

        converted = []

        for idx, mol in enumerate(mols):
            try:
                converted.append(mol.to(target, strict=strict, **kwargs))
            except ConversionError as exc:
                raise ConversionError(f"Conversion failed at batch index {idx}: {exc}") from exc

        if target is GraphMol:
            return GraphBatch(converted)

        return MolBatch(converted, target)

    def save(self, save_path: str | Path, shard_size: int | None = None, columnar_meta: bool = False) -> None:
        save_path = Path(save_path)
        if save_path.exists() and (not save_path.is_dir() or any(save_path.iterdir())):
            raise RuntimeError("Save path must point to an empty or non-existing directory.")

        size = max(1, len(self)) if shard_size is None else shard_size
        if not isinstance(size, int) or size < 1:
            raise ValueError("shard_size must be a positive integer.")

        save_path.mkdir(parents=True, exist_ok=True)

        # An empty dataset still carries a representation type and format version.
        for idx, start in enumerate(range(0, max(1, len(self)), size)):
            batch = MolBatch(self._mols[start : start + size], self.mol_type)
            batch.save_hdf5_shard(save_path / f"{idx}.hdf5", columnar_meta=columnar_meta)

    def save_hdf5_shard(self, save_file: str | Path, columnar_meta: bool = False) -> None:
        save_file = Path(save_file)
        if save_file.suffix != ".hdf5":
            raise ValueError("save_file must end in .hdf5.")

        if issubclass(self.mol_type, StringMol):
            payloads = [mol.text.encode("utf-8") for mol in self._mols]
        else:
            flags = Chem.PropertyPickleOptions.AllProps | Chem.PropertyPickleOptions.CoordsAsDouble
            payloads = [mol.rdkit_mol.ToBinary(flags) for mol in self._mols]

        offsets = np.zeros(len(payloads) + 1, dtype=np.uint64)
        offsets[1:] = np.cumsum([len(payload) for payload in payloads], dtype=np.uint64)
        payload = np.frombuffer(b"".join(payloads), dtype=np.uint8)

        with h5py.File(save_file, "x") as f:
            stamp_format(f)

            f.attrs["representation"] = self.mol_type.__name__
            f.attrs["native_format_version"] = 1

            if issubclass(self.mol_type, RDKitMol):
                f.attrs["rdkit_version"] = rdBase.rdkitVersion

            f.create_dataset("offsets", data=offsets)
            f.create_dataset("payload", data=payload, compression="gzip")

            save_meta(f, [dict(mol.meta) for mol in self._mols], columnar=columnar_meta)

    def read(self) -> MolBatch[TMol]:
        """Return an independent in-memory batch, including all molecule metadata."""

        return MolBatch([mol.copy() for mol in self._mols], self.mol_type)

    def __enter__(self) -> MolBatch[TMol]:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close_hdf5()

    def close_hdf5(self) -> None:
        """Close owned files. Unread data in this batch and its borrowed views needs those files.

        Call read() before closing to keep an independent in-memory batch.
        """

        for fp in self._open_fps:
            fp.close()

    @staticmethod
    def load(save_path: str | Path, mol_type: type[TMol] | None = None, materialise: bool = True) -> MolBatch[TMol]:
        """Load native HDF5 shards, leaving molecule payloads on disk until accessed.

        materialise=True creates every molecule wrapper immediately. False also defers
        wrapper construction: each batch[i] returns a fresh wrapper, as in LazyGraphBatch.
        Offsets and JSON metadata are read up front; columnar metadata values remain lazy.
        Decoding an RDKit payload caches the owned molecule within that wrapper.

        Use a context manager to close the files and read() to detach data before closure.
        Loaded metadata is read-only, matching graph batches. Supply mol_type for custom
        StringMol subclasses; their _from_lazy factory must initialise any extra state.
        """

        paths = sorted(
            Path(save_path).glob("*.hdf5"),
            key=lambda path: (0, int(path.stem)) if path.stem.isdigit() else (1, path.stem),
        )
        if not paths:
            raise ValueError("No HDF5 shards found.")

        return MolBatch._load_shards(paths, mol_type, materialise)

    @staticmethod
    def load_hdf5_shard(
        save_file: str | Path, mol_type: type[TMol] | None = None, materialise: bool = True
    ) -> MolBatch[TMol]:
        """Load one shard with the same lazy payload and file-lifetime rules as load()."""

        return MolBatch._load_shards([Path(save_file)], mol_type, materialise)

    @staticmethod
    def _load_shards(paths: list[Path], mol_type: type[TMol] | None, materialise: bool) -> MolBatch[TMol]:
        with ExitStack() as stack:
            files = [stack.enter_context(h5py.File(path, "r")) for path in paths]
            batch = LazyMolBatch(files, mol_type)

            if materialise:
                batch = MolBatch(list(batch), batch.mol_type)
                batch._open_fps = files

            stack.pop_all()
            return batch


# ***************************************************************
# *************** Lazy (on-demand) Native Batch ******************
# ***************************************************************


class _LazyNativeMolList(Sequence[TMol]):
    """List-like access that constructs native molecule wrappers on demand."""

    def __init__(self, batch: LazyMolBatch[TMol]):
        self._batch = batch

    def __len__(self) -> int:
        return self._batch._total_mols

    @overload
    def __getitem__(self, index: int) -> TMol: ...

    @overload
    def __getitem__(self, index: slice) -> list[TMol]: ...

    def __getitem__(self, index: int | slice) -> TMol | list[TMol]:
        if isinstance(index, slice):
            return [self._batch._build_mol(i) for i in range(*index.indices(len(self)))]

        return self._batch._build_mol(integer_index(index))


class LazyMolBatch(MolBatch[TMol]):
    """Native batch with fresh molecule wrappers on each lookup and lazy payload reads.

    Returned by MolBatch.load(..., materialise=False). Hold a wrapper or take subset()
    before making in-memory edits; repeated indexing does not retain those edits.
    """

    def __init__(self, hdf5_files: list[h5py.File], mol_type: type[TMol] | None = None):
        self._open_fps = hdf5_files
        self._shards = []

        for f in hdf5_files:
            shard = self._build_shard_state(f, mol_type)
            mol_type = shard["mol_type"]
            self._shards.append(shard)

        self.mol_type = mol_type
        self._boundaries = np.concatenate([[0], np.cumsum([len(s["offsets"]) - 1 for s in self._shards])])
        self._total_mols = int(self._boundaries[-1])
        self._mols = _LazyNativeMolList(self)

    @staticmethod
    def _build_shard_state(f: h5py.File, mol_type: type[TMol] | None) -> dict:
        check_format(f, f.filename)

        if f.attrs.get("native_format_version") != 1:
            raise ValueError("Unsupported native molecule batch version.")

        kind = f.attrs.get("representation")
        known = {"SmilesMol": SmilesMol, "RDKitMol": RDKitMol}
        mol_type = known.get(kind) if mol_type is None else mol_type

        if mol_type is None or mol_type.__name__ != kind or not issubclass(mol_type, (StringMol, RDKitMol)):
            raise ValueError(f"Unknown or mismatched representation {kind!r}; supply the matching mol_type.")

        offsets = f["offsets"][()]
        payload = f["payload"]
        if (
            payload.ndim != 1
            or payload.dtype != np.dtype("uint8")
            or offsets.ndim != 1
            or len(offsets) == 0
            or offsets.dtype.kind not in "iu"
            or offsets[0] != 0
            or offsets[-1] != len(payload)
            or np.any(offsets[1:] < offsets[:-1])
        ):
            raise ValueError("Invalid molecule payload offsets or byte array.")

        return {
            "mol_type": mol_type,
            "offsets": offsets,
            "payload": payload,
            "meta_group": f["meta"],
            "metas": load_meta(f["meta"], len(offsets) - 1),
        }

    def _build_mol(self, index: int) -> TMol:
        if index < 0:
            index += len(self)

        if index < 0 or index >= len(self):
            raise IndexError("Molecule index out of range.")

        shard_idx = int(np.searchsorted(self._boundaries, index, side="right") - 1)
        shard = self._shards[shard_idx]
        local = index - int(self._boundaries[shard_idx])
        start, end = (int(value) for value in shard["offsets"][local : local + 2])
        payload = LazyData._load_unchecked(shard["payload"], start, end - start)
        return self.mol_type._from_lazy(payload, shard["metas"][local])

    def meta_column(self, key: str) -> np.ndarray:
        """Read metadata across shards without constructing molecule wrappers or decoding payloads."""

        columns = []
        for shard in self._shards:
            try:
                columns.append(column_array(shard["meta_group"], key))
            except KeyError:
                columns.append(np.array([meta.get(key, "") for meta in shard["metas"]]))

        return np.concatenate(columns)
