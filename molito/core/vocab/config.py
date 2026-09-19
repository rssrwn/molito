from __future__ import annotations

from .graph import AtomVocab, BondVocab

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
