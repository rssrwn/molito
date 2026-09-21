from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Self, TypeVar

import numpy as np
from rdkit import Chem

from molito.convert import smiles_from_mol

TMol = TypeVar("TMol", bound="MolRepr")


class ConversionError(ValueError):
    """A representation cannot be converted, or a strict conversion would lose data."""


class MolRepr(ABC):
    """Common interface for small-molecule representations.

    Representations may hold invalid chemistry. `validate` and conversion to chemical
    representations require sanitisation. Metadata is copied across conversions.
    """

    __slots__ = ()
    meta: Mapping[str, Any]

    @classmethod
    @abstractmethod
    def from_rdkit(cls, rdkit_mol: Chem.rdchem.Mol, **kwargs) -> Self:
        """Build this representation from an RDKit molecule."""

    @abstractmethod
    def to_rdkit(self, sanitise: bool = False) -> Chem.rdchem.Mol | None:
        """Return an independent RDKit molecule, or None for invalid chemistry.

        Errors reading lazy data propagate to the caller.
        """

    @abstractmethod
    def copy(self) -> Self:
        """Return an independent copy, including metadata."""

    @abstractmethod
    def to_bytes(self) -> bytes:
        """Serialise the native representation and metadata."""

    @classmethod
    @abstractmethod
    def from_bytes(cls, data: bytes) -> Self:
        """Restore the native representation and metadata."""

    def validate(self, connected: bool = False) -> None:
        """Raise ConversionError if sanitisation or the requested connectivity check fails.

        Disconnected molecules are allowed by default. Set connected=True to require
        exactly one connected component; this also rejects empty molecules. Validation
        does not modify the molecule or impose element, size, or charge constraints.
        """

        try:
            mol = self.to_rdkit(sanitise=True)
        except (ValueError, RuntimeError, KeyError) as exc:
            raise ConversionError(f"Cannot validate {type(self).__name__}: {exc}") from exc

        if mol is None:
            raise ConversionError(f"{type(self).__name__} does not represent a sanitisable molecule.")

        if connected:
            n_components = len(Chem.GetMolFrags(mol))
            if n_components != 1:
                raise ConversionError(f"Expected exactly one connected component; found {n_components}.")

    def to(self, target: type[TMol], strict: bool = False, **kwargs) -> TMol:
        """Convert through RDKit, preserving metadata in an independent mapping.

        Same-type copies with no options do not parse or sanitise. Other conversions
        require valid chemistry. Strict mode checks supported molecular state after
        conversion: chemical identity, atom order/state, properties, and conformers.
        It rejects known graph padding/masking loss too. It is not a guarantee for
        arbitrary toolkit extensions or Python attributes attached to an RDKit object.
        """

        if not isinstance(target, type) or not issubclass(target, MolRepr):
            raise TypeError("target must be a MolRepr subclass.")
        if type(self) is target and not kwargs:
            return self.copy()

        try:
            source = self.to_rdkit(sanitise=True)
            if source is None:
                raise ConversionError(f"Cannot convert invalid {type(self).__name__} to {target.__name__}.")

            if strict:
                from molito.mol.graph import GraphMol

                if isinstance(self, GraphMol) and (source.GetNumBonds() != self.n_bonds or (self.atomics == 0).any()):
                    raise ConversionError("Strict conversion cannot preserve padded atoms or omitted graph bonds.")

            result = target.from_rdkit(source, **kwargs)
            if strict:
                restored = result.to_rdkit(sanitise=True)
                self._check_conversion(source, restored)
        except ConversionError:
            raise
        except (ValueError, RuntimeError, KeyError) as exc:
            raise ConversionError(f"Cannot convert {type(self).__name__} to {target.__name__}: {exc}") from exc

        result.meta = copy.deepcopy(dict(self.meta)) if self.meta is not None else {}
        return result

    @staticmethod
    def _check_conversion(source: Chem.rdchem.Mol, result: Chem.rdchem.Mol | None) -> None:
        """Conservative checks for information lost by the built-in representations."""

        params = Chem.SmilesWriteParams()
        flags = Chem.CXSmilesFields.CX_ALL_BUT_COORDS
        if result is None or Chem.MolToCXSmiles(source, params, flags) != Chem.MolToCXSmiles(result, params, flags):
            raise ConversionError("Strict conversion would change molecular structure or annotations.")

        def atom_state(atom):
            return (
                atom.GetAtomicNum(),
                atom.GetIsotope(),
                atom.GetAtomMapNum(),
                atom.GetFormalCharge(),
                atom.GetNumRadicalElectrons(),
                atom.GetChiralTag(),
                atom.GetNoImplicit(),
                atom.GetNumExplicitHs(),
            )

        if [atom_state(a) for a in source.GetAtoms()] != [atom_state(a) for a in result.GetAtoms()]:
            raise ConversionError("Strict conversion would change atom order or atom state.")
        if source.GetNumConformers() != result.GetNumConformers():
            raise ConversionError("Strict conversion would discard conformers.")

        objects = [
            (source, result),
            *zip(source.GetAtoms(), result.GetAtoms(), strict=True),
            *zip(source.GetBonds(), result.GetBonds(), strict=True),
        ]
        for left, right in zip(source.GetConformers(), result.GetConformers(), strict=True):
            if left.GetId() != right.GetId() or left.Is3D() != right.Is3D():
                raise ConversionError("Strict conversion would change conformer IDs or dimensionality.")
            if not np.allclose(left.GetPositions(), right.GetPositions(), atol=1e-6, rtol=1e-6):
                raise ConversionError("Strict conversion would change conformer coordinates.")
            objects.append((left, right))

        for left, right in objects:
            names = left.GetPropNames(includePrivate=True, includeComputed=False)
            if any(not right.HasProp(name) or left.GetProp(name) != right.GetProp(name) for name in names):
                raise ConversionError("Strict conversion would discard or change RDKit properties.")

    def to_smiles(self, canonical: bool = True, explicit_hs: bool = False) -> str:
        mol = self.to_rdkit(sanitise=True)
        text = smiles_from_mol(mol, canonical=canonical, explicit_hs=explicit_hs)
        if text is None:
            raise ConversionError("Cannot generate SMILES from this representation.")
        return text

    def save(self, path: str | Path) -> None:
        """Save native bytes to a new file. Existing files are never overwritten.

        Serialisation finishes before the destination file is created.
        """

        data = self.to_bytes()
        with Path(path).open("xb") as stream:
            stream.write(data)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        return cls.from_bytes(Path(path).read_bytes())
