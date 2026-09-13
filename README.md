# *Molito* - Small Molecule Data Processing Utility

[![PyPI](https://img.shields.io/pypi/v/molito.svg)](https://pypi.org/project/molito/)
[![Python versions](https://img.shields.io/pypi/pyversions/molito.svg)](https://pypi.org/project/molito/)
[![Tests](https://github.com/rssrwn/molito/actions/workflows/tests.yml/badge.svg)](https://github.com/rssrwn/molito/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A Python toolkit for processing and storing small molecule and protein data into a training-ready format.

**[Documentation](https://rssrwn.github.io/molito/)** · **[Tutorial](https://rssrwn.github.io/molito/tutorial/)** · **[API reference](https://rssrwn.github.io/molito/api/)**

Molito provides compact, serialisable representations for molecular graphs with atoms, bonds, and 3D conformer ensembles. It's designed for machine learning workflows where you need to go from RDKit molecules to structured arrays efficiently, with full support for stereochemistry (E/Z bond directions, tetrahedral chirality) preserved through round-trips. It also supports protein structures and protein-ligand complexes for binding data preprocessing, and integrates directly with RDKit, biotite and numpy.

## Features

- **Compact storage** — atomic numbers (uint8), charges (int8), chirality (int8), bonds (int16)
- **Stereochemistry** — E/Z bond directions and chirality preserved through canonicalisation and atom reordering
- **HDF5 persistence** — sharded save/load, with array data read on demand rather than up front
- **Conformer ensembles** — multiple 3D conformers per molecule with optional Boltzmann weights
- **Protein support** — residue annotations, biotite integration, prolif interaction detection
- **Configurable vocabularies** — toggle chirality and E/Z directions without rewriting your dataset
- **Lightweight** — core package needs only numpy, rdkit, scipy, h5py, biotite

## Installation

```bash
pip install molito
```

Requires Python >= 3.11.

Optional extras, installed as `pip install "molito[interactions]"`:

| extra | adds | enables |
|---|---|---|
| `interactions` | prolif, MDAnalysis | `Protein.to_prolif()`, `InteractionSet.from_system()` |
| `dev`          | matplotlib, jupyter, ruff, mypy | development tooling |
| `docs`         | mkdocs-material, mkdocstrings | building the documentation site |

`optimise_mol_xtb()` additionally requires `xtb-python`, which only installs reliably from
conda-forge:

```bash
mamba install -c conda-forge xtb-python
```

## Quick Start

### Reading molecules

```python
from molito.mol import GraphBatch, GraphMol

mol = GraphMol.from_smiles("C/C=C/C")

# A whole SDF at once. Tags on each record become mol.meta, and 3D coordinates
# are kept. Records holding only 2D depiction coordinates load without conformers.
batch = GraphBatch.from_sdf("ligands.sdf", remove_hs=True)
```

RDKit molecules go in directly, which is also where the atom-ordering choice lives:

```python
from rdkit import Chem

mol = GraphMol.from_rdkit(Chem.MolFromSmiles("C/C=C/C"))

# Opt in to canonical atom ordering when ingesting the same compounds from
# different formats, so the stored representation matches. Off by default.
mol = GraphMol.from_rdkit(rdkit_mol, canonicalise=True)
```

### Accessing properties

```python
mol.atomics          # uint8 array of atomic numbers
mol.charges          # int8 array of formal charges
mol.tokens           # ['C_0', 'C_0', 'C_0', 'C_0'] — includes chirality if present
mol.charged_symbols  # ['C_0', 'C_0', 'C_0', 'C_0'] — without chirality
mol.bond_indices     # [n_bonds, 2] array
mol.bond_types       # bond encoding indices
mol.coords           # [n_confs, n_atoms, 3] float32, or None
```

### Writing molecules back out

```python
mol.to_smiles()                 # stereochemistry preserved
rdkit_mol = mol.to_rdkit()      # None if the graph cannot be sanitised
batch.to_sdf("out.sdf")         # meta entries are written as SDF tags
```

### Saving and loading datasets

```python
batch = GraphBatch([mol1, mol2, mol3])
batch.save("my_dataset/", shard_size=1000, columnar_meta=True)

loaded = GraphBatch.load("my_dataset/")
mol = loaded[0]
mol.atomics               # array data is read from HDF5 here, not at load time
loaded.close_hdf5()       # loaded mols stop working after this - see below
```

For datasets beyond a million or so molecules, `materialise=False` skips building the
Python objects until each molecule is asked for, and `meta_column` scans a metadata key
without building any at all:

```python
import numpy as np

loaded = GraphBatch.load("my_dataset/", materialise=False)

ids = loaded.meta_column("mol_id")                 # one HDF5 column read
train = loaded.subset(np.where(ids != "")[0])      # builds only the selection
```

Molecules from a loaded batch read their arrays from the open file, so they stop working
once you call `close_hdf5()`. Call `mol.read()` to detach the ones you want to keep.

### Vocabularies for model training

```python
from molito.core import VocabConfig

# Defaults: chirality and E/Z directions enabled
n_atom_types = len(VocabConfig.atoms)   # includes PAD, MASK, CW/CCW variants
n_bond_types = len(VocabConfig.bonds)

# Convert tokens to model indices (with chirality fallback)
indices = VocabConfig.atoms.resolve_tokens(mol.tokens)

# Disable features you don't need. This changes the model index space only —
# the stored dataset is untouched.
VocabConfig.set_chirality(False)
VocabConfig.set_directions(False)
```

### Conformer generation and geometry

These take RDKit molecules that already have hydrogens and a conformer; they return `None`
rather than raising when the underlying force field cannot run.

```python
from rdkit import Chem
from rdkit.Chem import AllChem
from molito.geometry import calc_energy_mmff, optimise_mol_mmff, sample_ensemble

rdkit_mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
AllChem.EmbedMolecule(rdkit_mol)

energy = calc_energy_mmff(rdkit_mol)
opt_mol = optimise_mol_mmff(rdkit_mol, max_iters=500)

# A Boltzmann-weighted ensemble: deduplicated, strain-filtered conformers plus weights
final_mol, weights, e_min = sample_ensemble(rdkit_mol, max_confs=128)
```

### Proteins and complexes

```python
import biotite.structure.io as strucio
from molito.mol import BindingComplex, Protein

atom_array = strucio.load_structure("protein.pdb")
protein = Protein.from_biotite(atom_array)

system = BindingComplex(protein, ligand_mol)
system.atomics      # ligand atoms first, then protein
system.coords       # [n_complex_atoms, 3]
```

## Documentation

Full documentation lives at **[rssrwn.github.io/molito](https://rssrwn.github.io/molito/)**.

- [Tutorial](https://rssrwn.github.io/molito/tutorial/) — an SDF through to a padded training batch
- [Concepts](https://rssrwn.github.io/molito/concepts/) — bond encodings vs vocabulary indices, atom ordering, deferred loading, metadata
- [Stereochemistry](https://rssrwn.github.io/molito/stereochemistry/) — how stereo survives reordering, and what would break it
- [API reference](https://rssrwn.github.io/molito/api/) — every public class and function

Build the site locally with `mkdocs serve` after installing the `docs` extra.

## Package Structure

```
molito/
    core/           # AtomSet, BondSet, ConfSet, vocab, metadata, on-disk format
    mol/            # GraphMol, Protein, BindingComplex, InteractionSet
    geometry/       # conformer sampling, alignment, MMFF, xTB
    convert.py      # RDKit <-> molito conversion
    arrays.py       # numpy array ops (padding, one-hot, adjacency)
```

## Development

Clone and install editable:

```bash
git clone https://github.com/rssrwn/molito
cd molito
pip install -e ".[interactions,dev]"
```

Run the tests:

```bash
python -m unittest discover tests/ -v
```

CI enforces lint, formatting and type checking, so run these before opening a PR:

```bash
ruff format . && ruff check . && mypy
```

## Contributing

Issues and pull requests are welcome. CI runs the test suite on Python 3.11-3.13, checks that
the package still imports without the optional dependencies, and enforces ruff, mypy and a
strict docs build — so run the commands above before opening a PR.

Notable changes are recorded in the [changelog](CHANGELOG.md), which flags on-disk format
changes separately from code changes.

## License

MIT — see [LICENSE](LICENSE).
