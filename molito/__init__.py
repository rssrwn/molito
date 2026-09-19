"""molito: Molecular representation and processing toolkit.

Provides data structures for representing molecules as graphs with atoms, bonds, and 3D conformers,
with efficient HDF5 serialization. Supports chirality, E/Z stereo, protein structures, and
protein-ligand complexes.
"""

from importlib.metadata import PackageNotFoundError, version

from molito.core import PT, AtomSet, BondEncoding, BondSet, ConfSet, VocabConfig
from molito.mol import (
    BindingComplex,
    ComplexBatch,
    ConversionError,
    GraphBatch,
    GraphMol,
    MolBatch,
    MolRepr,
    Protein,
    ProteinBatch,
    RDKitMol,
    SmilesMol,
    StringMol,
)

try:
    __version__ = version("molito")
except PackageNotFoundError:
    # Package isn't installed (e.g. running from a source checkout without `pip install -e .`)
    __version__ = "0.0.0+unknown"

__all__ = [
    "PT",
    "AtomSet",
    "BindingComplex",
    "BondEncoding",
    "BondSet",
    "ComplexBatch",
    "ConfSet",
    "ConversionError",
    "GraphBatch",
    "GraphMol",
    "MolBatch",
    "MolRepr",
    "Protein",
    "ProteinBatch",
    "RDKitMol",
    "SmilesMol",
    "StringMol",
    "VocabConfig",
    "__version__",
]
