from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Generic, TypeVar, overload

import h5py
import numpy as np
from rdkit import Chem, rdBase

from molito.core.format import check_format, stamp_format
from molito.core.meta import load_meta, save_meta

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
    including when `.to(GraphMol)` is called. Native batches load eagerly and have no
    open-file lifetime requirements. Unknown string subclasses require `mol_type` at load.
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

    def __len__(self) -> int:
        return len(self._mols)

    @overload
    def __getitem__(self, index: int) -> TMol: ...

    @overload
    def __getitem__(self, index: slice) -> list[TMol]: ...

    def __getitem__(self, index: int | slice) -> TMol | list[TMol]:
        return self._mols[index]

    def subset(self, idxs: Sequence[int]) -> MolBatch[TMol]:
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

    @staticmethod
    def load(save_path: str | Path, mol_type: type[TMol] | None = None) -> MolBatch[TMol]:
        paths = sorted(Path(save_path).glob("*.hdf5"), key=lambda path: int(path.stem))
        if not paths:
            raise ValueError("No HDF5 shards found.")

        batches = [MolBatch.load_hdf5_shard(path, mol_type=mol_type) for path in paths]
        return MolBatch([mol for batch in batches for mol in batch], batches[0].mol_type)

    @staticmethod
    def load_hdf5_shard(save_file: str | Path, mol_type: type[TMol] | None = None) -> MolBatch[TMol]:
        with h5py.File(save_file, "r") as f:
            check_format(f, save_file)

            if f.attrs.get("native_format_version") != 1:
                raise ValueError("Unsupported native molecule batch version.")

            kind = f.attrs.get("representation")
            known = {"SmilesMol": SmilesMol, "RDKitMol": RDKitMol}

            mol_type = known.get(kind) if mol_type is None else mol_type
            if mol_type is None or mol_type.__name__ != kind or not issubclass(mol_type, (StringMol, RDKitMol)):
                raise ValueError(f"Unknown or mismatched representation {kind!r}; supply the matching mol_type.")

            offsets = f["offsets"][()]
            payload = f["payload"][()]
            if (
                offsets.ndim != 1
                or len(offsets) == 0
                or offsets.dtype.kind not in "iu"
                or offsets[0] != 0
                or offsets[-1] != len(payload)
                or np.any(offsets[1:] < offsets[:-1])
            ):
                raise ValueError("Invalid molecule payload offsets.")

            metas = load_meta(f["meta"], len(offsets) - 1)

            mols = []
            for idx, meta in enumerate(metas):
                data = payload[int(offsets[idx]) : int(offsets[idx + 1])].tobytes()
                if issubclass(mol_type, StringMol):
                    mol = mol_type(data.decode("utf-8"), meta=dict(meta))
                else:
                    mol = mol_type(Chem.Mol(data), meta=dict(meta))
                mols.append(mol)

        return MolBatch(mols, mol_type)
