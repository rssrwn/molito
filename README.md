# molito

A Python toolkit for storing and processing small molecules into a training-ready format.

molito provides compact, serialisable representations for molecular graphs with atoms, bonds, and 3D conformer ensembles. It's designed for machine learning workflows where you need to go from RDKit molecules to structured arrays efficiently, with full support for stereochemistry (E/Z bond directions, tetrahedral chirality) preserved through round-trips.

It also supports protein structures and protein-ligand complexes for binding data preprocessing. Dataset-specific preprocessing (BindingNet, SAIR, PLINDER) is deliberately kept out of this package and lives in a separate repository.

## Why molito

The obvious way to keep a molecular dataset around is a pickled list of RDKit molecules. That
works until the dataset stops fitting comfortably in memory. Here are 100,000 BindingNet
ligands stored four ways, measured on the same machine:

| format | on disk | load time | peak memory |
|---|---|---|---|
| pickled RDKit mols | 44.8 MB | 0.96 s | 1537 MB |
| pickled + gzip | **8.4 MB** | 1.07 s | 1540 MB |
| SDF | 258.7 MB | 9.67 s | 2351 MB |
| molito HDF5 | 30.1 MB | 0.37 s | **228 MB** |
| molito, `materialise=False` | 30.1 MB | **0.02 s** | **159 MB** |

**Memory is the real difference** — 228 MB against 1537 MB for the same molecules, because
molito keeps typed arrays rather than a graph of Python and C++ objects. That is what decides
whether a dataset fits in a dataloader worker, and the gap widens with dataset size.

Being straight about the rest: **gzipped pickle is smaller on disk than molito**, and if
minimum bytes is all you want, it wins. What it costs you is random access — you must
decompress and unpickle the entire file to look at one molecule, which is why its load time and
memory are unchanged from raw pickle. molito reads one molecule, or one metadata column, without
touching the rest.

What you also get, which none of the alternatives give you: stereochemistry that provably
survives atom reordering, a frozen on-disk encoding with a version stamp, and vocabularies that
let you switch features like chirality on and off without rewriting the dataset.

## Features

- **Compact storage** — atomic numbers (uint8), charges (int8), chirality (int8), bonds (int16)
- **Stereochemistry** — E/Z bond directions and chirality preserved through canonicalisation and atom reordering
- **HDF5 persistence** — sharded save/load with lazy on-demand reading for large datasets
- **Conformer ensembles** — multiple 3D conformers per molecule with optional Boltzmann weights
- **Protein support** — residue annotations, biotite integration, prolif interaction detection
- **Configurable vocabularies** — toggle chirality and E/Z directions for model training
- **Lightweight** — core package needs only numpy, rdkit, scipy, h5py, biotite

## Installation

> Not yet on PyPI. For now, install editable from a local clone (see [Development](#development)).
> Once published the install will be `pip install molito`.

Optional extras:

| extra | adds | enables |
|---|---|---|
| `interactions` | prolif, MDAnalysis | `Protein.to_prolif()`, `InteractionSet.from_system()` |
| `dev`          | matplotlib, jupyter, ipykernel, py3Dmol | notebook/visualisation tools |
| `docs`         | mkdocs-material, mkdocstrings | building the documentation site |

`optimise_mol_xtb()` additionally requires `xtb-python`, which only installs reliably from
conda-forge:

```bash
mamba install -c conda-forge xtb-python
```

Requires Python >= 3.11.

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

Lint and format (both are enforced in CI):

```bash
ruff format . && ruff check .
```

## Quick Start

### Loading molecules

```python
from rdkit import Chem
from molito.mol import GraphMol, GraphBatch

# From SMILES — atom order is preserved as-is (canonicalise=False by default)
mol = GraphMol.from_rdkit(Chem.MolFromSmiles("C/C=C/C"))

# From an SDF with 3D coordinates (Hs preserved)
supplier = Chem.SDMolSupplier("molecule.sdf", removeHs=False)
mol = GraphMol.from_rdkit(next(supplier))

# Opt in to canonical atom ordering for deterministic storage across input formats
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

### Round-tripping to RDKit

```python
rdkit_mol = mol.to_rdkit()
smiles = Chem.MolToSmiles(rdkit_mol)  # stereochemistry preserved
```

### Saving and loading datasets

```python
batch = GraphBatch([mol1, mol2, mol3])
batch.save("my_dataset/", shard_size=1000)

loaded = GraphBatch.load("my_dataset/")
mol = loaded[0]           # lazy — HDF5 read happens on property access
mol.atomics               # triggers read
```

### Vocabularies for model training

```python
from molito.core import VocabConfig

# Defaults: chirality and E/Z directions enabled
n_atom_types = len(VocabConfig.atoms)   # includes PAD, MASK, CW/CCW variants
n_bond_types = len(VocabConfig.bonds)

# Convert tokens to model indices (with chirality fallback)
indices = VocabConfig.atoms.resolve_tokens(mol.tokens)

# Disable features you don't need
VocabConfig.set_chirality(False)
VocabConfig.set_directions(False)
```

### Conformer generation and geometry

```python
from molito.geometry import sample_ensemble, calc_energy_mmff, optimise_mol_mmff

# Generate a Boltzmann-weighted conformer ensemble
result = sample_ensemble(rdkit_mol, max_confs=128)
final_mol, weights, e_min = result

# MMFF energy and optimisation
energy = calc_energy_mmff(rdkit_mol)
opt_mol = optimise_mol_mmff(rdkit_mol, max_iters=500)
```

### Proteins and complexes

```python
from molito.mol import Protein, BindingComplex
import biotite.structure.io as strucio

atom_array = strucio.load_structure("protein.pdb")
protein = Protein.from_biotite(atom_array)

complex = BindingComplex(protein, ligand_mol)
```

## Package Structure

```
molito/
    core/           # AtomSet, BondSet, ConfSet, vocab, PT, pharmacophore
    mol/            # GraphMol, Protein, BindingComplex, InteractionSet
    geometry/       # conformer alignment, MMFF, xTB, sampling
    convert.py      # RDKit <-> molito conversion
    arrays.py       # numpy array ops (padding, one-hot, adjacency)
```

## License

MIT — see [LICENSE](LICENSE).
