from __future__ import annotations

import json
import re
from collections.abc import Iterable
from importlib.resources import files

from molito.core._checks import check_type_all

from .base import Vocabulary

# The complete preset files define token indices. Derive these public constants
# from the character preset so the alphabet is only maintained in one place.
_CHAR_PRESET = json.loads(files("molito").joinpath("defs/smiles_char_vocab.json").read_text(encoding="utf-8"))
STRING_SPECIAL_TOKENS = tuple(_CHAR_PRESET["special_tokens"])
SMILES_CHARACTERS = tuple(token for token in _CHAR_PRESET["tokens"] if token not in STRING_SPECIAL_TOKENS)


class StringVocab(Vocabulary[str]):
    """An ordered string vocabulary with separate model-control tokens.

    Extension returns a new vocabulary and preserves all existing indices. Ordinary
    text never matches special tokens implicitly; encoding adds them explicitly.
    """

    def __init__(self, tokens: list[str], special_tokens: Iterable[str] = STRING_SPECIAL_TOKENS):
        check_type_all(tokens, str, "tokens")
        if any(not token for token in tokens):
            raise ValueError("Vocabulary tokens cannot be empty.")

        super().__init__(tokens)
        self.special_tokens = tuple(special_tokens)
        if len(set(self.special_tokens)) != len(self.special_tokens):
            raise ValueError("Special tokens must be unique.")
        if any(token not in self for token in self.special_tokens):
            raise ValueError("All special tokens must be present in the vocabulary.")

    @staticmethod
    def build(
        tokens: Iterable[str] = (),
        characters: Iterable[str] = SMILES_CHARACTERS,
        special_tokens: Iterable[str] = STRING_SPECIAL_TOKENS,
    ) -> StringVocab:
        """Build a vocabulary with fixed special/character prefixes and extra tokens."""

        characters = list(characters)
        specials = tuple(special_tokens)
        if any(not isinstance(char, str) or len(char) != 1 for char in characters):
            raise ValueError("Fallback characters must be single-character strings.")
        if set(characters).intersection(specials):
            raise ValueError("Fallback characters cannot also be special tokens.")

        ordered = list(dict.fromkeys([*specials, *characters, *tokens]))
        return StringVocab(ordered, special_tokens=specials)

    @staticmethod
    def characters() -> StringVocab:
        """Load the complete ordered SMILES character vocabulary, including special tokens."""

        resource = files("molito").joinpath("defs/smiles_char_vocab.json")
        return StringVocab.from_bytes(resource.read_bytes())

    @staticmethod
    def smiles(extra_tokens: Iterable[str] = ()) -> StringVocab:
        """Load the complete SMILES regex vocabulary, optionally appending extra tokens."""

        resource = files("molito").joinpath("defs/smiles_regex_vocab.json")
        return StringVocab.from_bytes(resource.read_bytes()).extend(extra_tokens)

    @staticmethod
    def is_core_token(token: str) -> bool:
        """Selection rule for fitting new vocabularies, never a preset-load or input-text filter.

        Keep syntax, common organic bracket atoms, and a small set of common ions.
        Atom maps, isotope labels, and unusual elements/charges use character fallback.
        """

        if token in SMILES_CHARACTERS or token in ("Cl", "Br"):
            return True
        if re.fullmatch(r"%[0-9]{2}", token):
            return True
        if token in ("[H]", "[H+]", "[Li+]", "[Na+]", "[K+]", "[Mg+2]", "[Ca+2]", "[Zn+2]"):
            return True

        pattern = r"\[(?:B|C|N|O|P|S|Si|b|c|n|o|p|s)(?:@@?)?(?:H[1-4]?)?(?:[+-][12]?)?\]"
        return re.fullmatch(pattern, token) is not None or token in ("[F-]", "[Cl-]", "[Br-]", "[I-]")

    def extend(self, tokens: Iterable[str]) -> StringVocab:
        """Return a vocabulary with additional tokens, retaining existing indices."""

        ordered = list(dict.fromkeys([*self, *tokens]))
        return StringVocab(ordered, self.special_tokens)

    def to_bytes(self) -> bytes:
        data = {"version": 1, "tokens": list(self), "special_tokens": self.special_tokens}
        return json.dumps(data, ensure_ascii=False).encode("utf-8")

    @staticmethod
    def from_bytes(data: bytes) -> StringVocab:
        obj = json.loads(data)
        if obj["version"] != 1:
            raise ValueError("Unsupported string vocabulary version.")
        return StringVocab(obj["tokens"], obj["special_tokens"])
