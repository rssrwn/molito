from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from typing import Any, Self

from rdkit import Chem

from molito.convert import smiles_from_mol
from molito.core.lazydata import LazyData
from molito.core.meta import _json_default
from molito.tokenise import Tokeniser

from .base import ConversionError, MolRepr


class StringMol(MolRepr):
    """Abstract molecule representation storing exact text, including invalid text.

    Tokenisation is independent of chemical validity. Subclasses implement RDKit
    conversion; text storage, metadata, and tokenisation are shared here.
    """

    __slots__ = ("_text", "meta")

    def __init__(self, text: str, meta: Mapping[str, Any] | None = None):
        self.text = text
        self.meta = {} if meta is None else meta

    @classmethod
    def _from_lazy(cls, payload: LazyData, meta: Mapping[str, Any]) -> Self:
        """Build a wrapper without reading text. Subclasses with extra state can override this factory."""

        obj = cls.__new__(cls)
        obj._text = payload
        obj.meta = meta
        return obj

    @property
    def text(self) -> str:
        """Exact text, read from HDF5 on access when loaded as part of a batch."""

        if isinstance(self._text, LazyData):
            return self._text.read().tobytes().decode("utf-8")

        return self._text

    @text.setter
    def text(self, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError("text must be a string.")

        self._text = value

    def __str__(self) -> str:
        return self.text

    def copy(self) -> Self:
        return type(self)(self.text, meta=copy.deepcopy(dict(self.meta)))

    def read(self) -> Self:
        """Return an independent in-memory molecule, including its metadata."""

        return self.copy()

    def tokenise(self, tokeniser: Tokeniser) -> list[str]:
        return tokeniser.tokenise(self.text)

    def encode(self, tokeniser: Tokeniser, add_special_tokens: bool = False) -> list[int]:
        return tokeniser.encode(self.text, add_special_tokens=add_special_tokens)

    def to_bytes(self) -> bytes:
        obj = {"version": 1, "kind": type(self).__name__, "text": self.text, "meta": dict(self.meta)}
        return json.dumps(obj, ensure_ascii=False, default=_json_default).encode("utf-8")

    @classmethod
    def from_bytes(cls, data: bytes) -> Self:
        obj = json.loads(data)
        if obj["version"] != 1 or obj["kind"] != cls.__name__:
            raise ValueError("String molecule type or format version does not match.")

        return cls(obj["text"], meta=obj["meta"])


class SmilesMol(StringMol):
    """Exact SMILES text. Construction and storage do not require valid chemistry."""

    __slots__ = ()

    @classmethod
    def from_rdkit(cls, rdkit_mol: Chem.rdchem.Mol, canonical: bool = True, explicit_hs: bool = False) -> Self:
        if rdkit_mol is None:
            raise ConversionError("Cannot generate SMILES from None.")

        mol = Chem.Mol(rdkit_mol)
        try:
            Chem.SanitizeMol(mol)
            text = smiles_from_mol(mol, canonical=canonical, explicit_hs=explicit_hs)
        except (ValueError, RuntimeError) as exc:
            raise ConversionError("Cannot generate SMILES from an invalid RDKit molecule.") from exc

        if text is None:
            raise ConversionError("RDKit could not generate SMILES.")

        return cls(text)

    def to_rdkit(self, sanitise: bool = False) -> Chem.rdchem.Mol | None:
        """Parse without name/CXSMILES suffixes; sanitisation is optional.

        Explicit hydrogen atoms in the text are retained. Parsing failure returns None,
        as for GraphMol.to_rdkit; errors reading stored text propagate to the caller.
        `validate` and `.to(...)` request sanitisation.
        """

        params = Chem.SmilesParserParams()
        params.removeHs = False
        params.sanitize = sanitise
        params.parseName = False
        params.allowCXSMILES = False
        text = self.text
        try:
            return Chem.MolFromSmiles(text, params)
        except (ValueError, RuntimeError):
            return None
