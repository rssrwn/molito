# Getting Started

## Installation

```bash
pip install molito
```

The core package requires: numpy, rdkit, scipy, h5py, biotite, more-itertools.

### Optional Dependencies

| Extra | Packages | Used by |
|-------|----------|---------|
| `interactions` | prolif, MDAnalysis | `Protein.to_prolif()`, `InteractionSet.from_system()` |
| `dev` | matplotlib, jupyter, ipykernel, py3Dmol | notebook and visualisation tooling |
| `docs` | mkdocs-material, mkdocstrings | building this documentation |

`optimise_mol_xtb()` needs xtb-python, which is not available as a working pip wheel and must be
installed from conda-forge:

```bash
mamba install -c conda-forge xtb-python
```

## Core Concepts

### GraphMol

The primary molecule class. Combines an `AtomSet`, `BondSet`, and optional `ConfSet`:

```python
from molito.mol import GraphMol
from rdkit import Chem

# From SMILES (via RDKit)
mol = GraphMol.from_rdkit(Chem.MolFromSmiles("c1ccccc1"))

# From an SDF file with 3D coordinates
supplier = Chem.SDMolSupplier("molecule.sdf", removeHs=False)
mol = GraphMol.from_rdkit(next(supplier))

# Opt in to canonical atom ordering for deterministic storage across input formats.
# Default is canonicalise=False (original atom ordering is preserved).
mol = GraphMol.from_rdkit(rdkit_mol, canonicalise=True)
```

### Saving and Loading

```python
from molito.mol import GraphBatch

# Save to sharded HDF5
batch = GraphBatch(molecules)
batch.save("dataset/", shard_size=1000)

# Load back (data is read on demand from HDF5)
loaded = GraphBatch.load("dataset/")
mol = loaded[0]
print(mol.atomics)  # triggers HDF5 read

# For very large datasets, defer building the GraphMol objects until they're asked for.
# Array data is read on demand either way -- this only skips object construction.
lazy = GraphBatch.load("dataset/", materialise=False)
```

### Vocabularies

Map atom/bond tokens to model indices for training:

```python
from molito.core import VocabConfig

# Defaults: chirality=True, directions=True
n_atom_types = len(VocabConfig.atoms)
n_bond_types = len(VocabConfig.bonds)

# Convert molecule tokens to indices
indices = VocabConfig.atoms.resolve_tokens(mol.tokens)

# Toggle features
VocabConfig.set_chirality(False)   # drops CW/CCW variants
VocabConfig.set_directions(False)  # drops E/Z direction bonds
```

### Proteins and Complexes

```python
from molito.mol import Protein, BindingComplex, InteractionSet
import biotite.structure.io as strucio

# From a biotite AtomArray
atom_array = strucio.load_structure("protein.pdb")
protein = Protein.from_biotite(atom_array)

# Combine with a ligand
complex = BindingComplex(protein, ligand_mol)

# Interaction detection (requires the `interactions` extra)
interactions = InteractionSet.from_system(complex)
```
