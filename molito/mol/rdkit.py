from __future__ import annotations

import base64
import copy
import json
from collections.abc import Mapping
from typing import Any, Self

import numpy as np
from rdkit import Chem, rdBase

from molito.convert import mol_from_smiles
from molito.core.meta import _json_default

from .base import ConversionError, MolRepr

TArr = np.ndarray


class RDKitMol(MolRepr):
    """An owned RDKit molecule with molito conversion, metadata, and persistence.

    Input molecules and `to_rdkit` results are copied. The `rdkit_mol` property exposes
    the owned mutable object explicitly for chemistry operations. Unsanitised objects
    can be stored; validation is explicit.
    """

    __slots__ = ("_mol", "meta")

    def __init__(self, rdkit_mol: Chem.rdchem.Mol, meta: Mapping[str, Any] | None = None):
        if not isinstance(rdkit_mol, Chem.rdchem.Mol):
            raise TypeError("rdkit_mol must be an RDKit molecule.")
        self._mol = Chem.Mol(rdkit_mol)
        self.meta = {} if meta is None else meta

    @property
    def rdkit_mol(self) -> Chem.rdchem.Mol:
        """The owned mutable molecule; mutations here update this representation."""

        return self._mol

    @property
    def n_atoms(self) -> int:
        return self._mol.GetNumAtoms()

    @property
    def n_heavy_atoms(self) -> int:
        return self._mol.GetNumHeavyAtoms()

    @property
    def n_bonds(self) -> int:
        return self._mol.GetNumBonds()

    @property
    def n_conformers(self) -> int:
        return self._mol.GetNumConformers()

    @property
    def coords(self) -> TArr | None:
        if self.n_conformers == 0:
            return None
        return np.stack([conf.GetPositions() for conf in self._mol.GetConformers()])

    def copy(self) -> Self:
        return type(self)(self._mol, meta=copy.deepcopy(dict(self.meta)))

    @classmethod
    def from_rdkit(cls, rdkit_mol: Chem.rdchem.Mol, **kwargs) -> Self:
        return cls(rdkit_mol, **kwargs)

    @classmethod
    def from_smiles(cls, smiles: str, explicit_hs: bool = False) -> Self:
        mol = mol_from_smiles(smiles, embed_hs=explicit_hs)
        if mol is None:
            raise ConversionError(f"Could not parse SMILES {smiles!r}.")
        return cls(mol)

    def to_rdkit(self, sanitise: bool = False) -> Chem.rdchem.Mol | None:
        mol = Chem.Mol(self._mol)
        if sanitise:
            try:
                Chem.SanitizeMol(mol)
            except (ValueError, RuntimeError):
                return None
        return mol

    def to_bytes(self) -> bytes:
        # Explicit flags preserve molecule/atom/bond/conformer properties and double
        # precision coordinates, independently of RDKit's process-global defaults.
        flags = Chem.PropertyPickleOptions.AllProps | Chem.PropertyPickleOptions.CoordsAsDouble
        payload = base64.b64encode(self._mol.ToBinary(flags)).decode("ascii")
        obj = {
            "version": 1,
            "kind": "RDKitMol",
            "rdkit_version": rdBase.rdkitVersion,
            "payload": payload,
            "meta": dict(self.meta),
        }
        return json.dumps(obj, default=_json_default).encode("utf-8")

    @classmethod
    def from_bytes(cls, data: bytes) -> Self:
        obj = json.loads(data)
        if obj["version"] != 1 or obj["kind"] != "RDKitMol":
            raise ValueError("RDKit molecule type or format version does not match.")
        mol = Chem.Mol(base64.b64decode(obj["payload"], validate=True))
        return cls(mol, meta=obj["meta"])
