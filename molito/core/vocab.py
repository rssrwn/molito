from __future__ import annotations

import pickle
from collections.abc import Iterator, Mapping
from typing import Generic, TypeVar

import numpy as np
from rdkit import Chem

from molito.core._checks import MASK_TOKEN, PAD_TOKEN, PICKLE_PROTOCOL, check_type_all, check_unique
from molito.core.presets import CHIRAL_ELIGIBLE, DRUG_LIKE_ATOMS

T = TypeVar("T")
TArr = np.ndarray


# Chirality constants
CHIRAL_NONE = 0
CHIRAL_CW = 1
CHIRAL_CCW = 2

RDKIT_CHIRAL_TO_INT = {
    Chem.ChiralType.CHI_UNSPECIFIED: CHIRAL_NONE,
    Chem.ChiralType.CHI_TETRAHEDRAL_CW: CHIRAL_CW,
    Chem.ChiralType.CHI_TETRAHEDRAL_CCW: CHIRAL_CCW,
}
INT_TO_RDKIT_CHIRAL = {v: k for k, v in RDKIT_CHIRAL_TO_INT.items()}

CHIRAL_SUFFIXES = {CHIRAL_CW: "CW", CHIRAL_CCW: "CCW"}


# ************************************
# ***** Generic Vocabulary class *****
# ************************************


class Vocabulary(Generic[T], Mapping):
    """Generic vocabulary class which maps tokens <--> indices."""

    def __init__(self, tokens: list[T]):
        check_unique(tokens, "tokens list")

        token_idx_map = {token: idx for idx, token in enumerate(tokens)}
        idx_token_map = {idx: token for idx, token in enumerate(tokens)}

        self.token_idx_map = token_idx_map
        self.idx_token_map = idx_token_map

    # *** Mapping Collection methods ***

    def __len__(self) -> int:
        return len(self.token_idx_map)

    def __getitem__(self, token: T) -> int:
        return self.get_index(token)

    def __contains__(self, token: T) -> bool:
        return token in self.token_idx_map

    def __iter__(self) -> Iterator[T]:
        return iter(self.token_idx_map)

    # *** Mapping functions ***

    def get_token(self, index: int) -> T:
        return self.idx_token_map[index]

    def get_index(self, token: T) -> int:
        return self.token_idx_map[token]

    def tokens_from_indices(self, indices: list[int]) -> list[T]:
        check_type_all(indices, int, "indices list")
        return [self.idx_token_map[idx] for idx in indices]

    def indices_from_tokens(self, tokens: list[T]) -> list[int]:
        return [self.token_idx_map[token] for token in tokens]

    # *** Check contents of vocab map ***

    def contains_token(self, token: T) -> bool:
        return token in self.token_idx_map

    def contains_index(self, index: int) -> bool:
        return index in self.idx_token_map

    # *** Iter functions ***

    def iter_tokens(self) -> Iterator[T]:
        return iter(self.token_idx_map)

    def iter_indices(self) -> Iterator[int]:
        return iter(self.idx_token_map)

    # *** Saving and loading functionality ***

    def to_bytes(self) -> bytes:
        tokens = list(self.token_idx_map.keys())
        return pickle.dumps(tokens, protocol=PICKLE_PROTOCOL)

    @staticmethod
    def from_bytes(data: bytes) -> Vocabulary:
        return Vocabulary(pickle.loads(data))


# **********************************
# ***** Domain-Specific Vocabs *****
# **********************************


class AtomVocab(Vocabulary[str]):
    """Atom vocabulary with optional chirality fallback."""

    @staticmethod
    def build(tokens: list[str] | None = None, chirality: bool = True) -> AtomVocab:
        """Build an atom vocabulary from base tokens, optionally expanding with chirality variants.

        Args:
            tokens: Base tokens in "element_charge" format (e.g. ["C_0", "N_1"]). Defaults to DRUG_LIKE_ATOMS.
            chirality: If True, expand eligible tokens (see CHIRAL_ELIGIBLE) with _CW and _CCW variants.
        """

        if tokens is None:
            tokens = DRUG_LIKE_ATOMS[:]

        expanded = []
        for token in tokens:
            expanded.append(token)
            if chirality and token in CHIRAL_ELIGIBLE:
                expanded.append(f"{token}_CW")
                expanded.append(f"{token}_CCW")

        return AtomVocab([PAD_TOKEN, *expanded, MASK_TOKEN])

    def resolve_token(self, token: str) -> int:
        """Look up index for a token, falling back to base form if chirality suffix not in vocab."""

        if token in self.token_idx_map:
            return self.token_idx_map[token]

        parts = token.rsplit("_", 1)
        if len(parts) == 2 and parts[1] in ("CW", "CCW"):
            base = parts[0]
            if base in self.token_idx_map:
                return self.token_idx_map[base]

        raise KeyError(f"Token '{token}' not found in vocabulary (also tried base form)")

    def resolve_tokens(self, tokens: list[str]) -> list[int]:
        return [self.resolve_token(t) for t in tokens]


class BondVocab(Vocabulary[str]):
    """Bond vocabulary for model training."""

    def __init__(self, tokens: list[str]):
        super().__init__(tokens)

        from molito.core.bonds import BondEncoding

        self._encoding = BondEncoding

        self._enc_to_model = {}
        for enc_idx in range(BondEncoding.size()):
            enc_token = BondEncoding.get_token(enc_idx)
            if enc_token in self.token_idx_map:
                self._enc_to_model[enc_idx] = self.token_idx_map[enc_token]
            else:
                base = "_".join(enc_token.split("_")[:2])
                if base in self.token_idx_map:
                    self._enc_to_model[enc_idx] = self.token_idx_map[base]

        # Vectorised form of _enc_to_model for resolve_types. -1 marks "no mapping" so
        # resolve_types can surface it as an error rather than silently aliasing to slot 0.
        self._enc_to_model_lut = np.full(BondEncoding.size(), -1, dtype=np.int64)
        for enc_idx, model_idx in self._enc_to_model.items():
            self._enc_to_model_lut[enc_idx] = model_idx

    @staticmethod
    def build(directions: bool = True) -> BondVocab:
        """Build a bond vocabulary for model training.

        Args:
            directions: If True, include E/Z direction tokens (1_F_U, 1_F_D, 1_T_U, 1_T_D).
                Direction bonds fall back to their base form when directions are disabled.
        """

        tokens = ["0_F", "1_F", "2_F", "3_F", "1_T", "2_T", "3_T"]

        if directions:
            tokens.extend(["1_F_U", "1_F_D", "1_T_U", "1_T_D"])

        tokens.append("-1_F")
        return BondVocab(tokens)

    def encode(self, bond, is_aromatic: bool = False, direction=None) -> int:
        """Look up model index from bond properties (type, aromaticity, direction)."""

        enc_idx = self._encoding.encode(bond, is_aromatic, direction)
        return self.encoding_to_model_index(enc_idx)

    def encoding_to_model_index(self, enc_idx: int) -> int:
        if enc_idx in self._enc_to_model:
            return self._enc_to_model[enc_idx]

        token = self._encoding.get_token(enc_idx)
        raise KeyError(f"Encoding index {enc_idx} ({token}) has no mapping in this bond vocab")

    def resolve_types(self, enc_arr: TArr) -> TArr:
        """Vectorised encoding_to_model_index: remap BondEncoding indices to model indices.

        Args:
            enc_arr: array of any shape containing BondEncoding indices (e.g. from
                BondSet.types or GraphMol.adjacency).

        Returns:
            array of the same shape with model indices for this vocab. Directional encodings
            fall back to their base bond when the vocab was built with directions=False.
        """

        out = self._enc_to_model_lut[enc_arr]
        if (out == -1).any():
            bad = np.unique(enc_arr[out == -1])
            bad_tokens = [self._encoding.get_token(int(i)) for i in bad]
            raise KeyError(f"Encoding indices {bad.tolist()} ({bad_tokens}) have no mapping in this bond vocab")

        return out

    def model_index_to_encoding_index(self, model_idx: int) -> int:
        token = self.get_token(model_idx)
        return self._encoding._token_to_idx[token]

    def get_bond_type(self, model_idx: int):
        token = self.get_token(model_idx)
        return self._encoding._enum_bond_map[int(token.split("_")[0])]

    def get_is_aromatic(self, model_idx: int) -> bool:
        return self.get_token(model_idx).split("_")[1] == "T"

    def get_mask_index(self) -> int:
        return self.token_idx_map["-1_F"]


# *******************************
# ***** Vocab Configuration *****
# *******************************


class VocabConfig:
    """Central configuration for molecular vocabularies."""

    _chirality: bool = True
    _directions: bool = True
    _atom_tokens: list[str] = None

    atoms: AtomVocab = AtomVocab.build(chirality=True)
    bonds: BondVocab = BondVocab.build(directions=True)

    @classmethod
    def set_chirality(cls, enabled: bool):
        cls._chirality = enabled
        cls.atoms = AtomVocab.build(tokens=cls._atom_tokens, chirality=enabled)

    @classmethod
    def set_directions(cls, enabled: bool):
        cls._directions = enabled
        cls.bonds = BondVocab.build(directions=enabled)

    @classmethod
    def set_atom_tokens(cls, tokens: list[str]):
        cls._atom_tokens = tokens
        cls.atoms = AtomVocab.build(tokens=tokens, chirality=cls._chirality)

    @classmethod
    def reset(cls):
        cls._chirality = True
        cls._directions = True
        cls._atom_tokens = None
        cls.atoms = AtomVocab.build(chirality=True)
        cls.bonds = BondVocab.build(directions=True)
