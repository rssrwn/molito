# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**molito** is a Python package for storing and processing small molecules into a training-ready format. It provides data structures and utilities for representing molecules as graphs with atoms, bonds, and 3D conformers, with efficient serialization to HDF5 format. It also supports protein and protein-ligand complex representations for preprocessing binding data.

## Repository Structure

```
molito/                  # pip-installable package
    core/                # primitive data containers (AtomSet, BondSet, ConfSet, vocab, PT, pharmacophore)
    mol/                 # molecular structures (GraphMol, Protein, BindingComplex, InteractionSet)
    geometry/            # conformer alignment, energy, optimisation, sampling
    convert.py           # RDKit ↔ molito conversion (mol_from_atoms, mol_from_smiles, etc.)
    arrays.py            # numpy array ops (pad_arrays, one_hot_encode, adj_from_edges)
docs/                    # mkdocs site (built with --strict in CI)
tests/                   # test suite (outside package)
    repr/                # tests for the molito package
```

Preprocessing scripts (BindingNet / SAIR / PLINDER → ComplexBatch HDF5) live in a separate
repository and are deliberately not part of this package.

## Installation

```bash
pip install -e .                                  # core package
pip install -e ".[interactions]"                   # + prolif/MDAnalysis for interaction detection
```

For `optimise_mol_xtb()`, additionally install xtb from conda-forge (no working pip wheel):

```bash
mamba install -c conda-forge xtb-python
```

## Architecture

### Core Data Structures (`molito/core/`)

- **`AtomSet`** (`atoms.py`): Atomic numbers (uint8), formal charges (int8), chirality (int8), and optional protein annotations (res_names, atom_names, res_ids).
- **`BondSet`** (`bonds.py`): Bond indices and types encoded via `BondEncoding` (int16). Supports kekulized bonds with aromatic flags and E/Z bond directions.
- **`ConfSet`** (`confs.py`): 3D conformer coordinates `[n_confs, n_atoms, 3]` with optional Boltzmann weights.
- **`LazyData`** (`lazydata.py`): Wrapper for HDF5 datasets enabling on-demand reading.
- **`PT`** (`pt.py`): Periodic table singleton for atomic number ↔ symbol conversion.
- **`PharmacophoreFinder`** (`pharmacophore.py`): Pharmacophore feature detection on RDKit molecules.

### Vocabularies (`molito/core/vocab.py`)

- **`AtomVocab`**: Configurable vocab mapping atom tokens (e.g., "C_0", "C_0_CW") to indices, with chirality fallback.
- **`BondVocab`**: Maps bond encoding tokens to model indices, with E/Z direction support.
- **`VocabConfig`**: Central configuration for toggling chirality and bond directions.
- **`BondEncoding`** (`bonds.py`): Fixed storage encoding for bonds (12 types). Never changes — ensures HDF5 backward compat.

### Molecular Structures (`molito/mol/`)

- **`GraphMol`** / **`GraphBatch`** (`graph.py`): Primary molecule class with RDKit conversion, canonical ordering, geometric transforms, HDF5 serialization.
- **`Protein`** / **`ProteinBatch`** (`protein.py`): Protein chain representation with residue annotations. `to_prolif()` requires optional prolif/MDAnalysis deps.
- **`BindingComplex`** / **`ComplexBatch`** (`complex.py`): Protein-ligand complex representation.
- **`InteractionSet`** (`interactions.py`): Sparse protein-ligand interactions via prolif (optional dep).

### Geometry (`molito/geometry/`)

- **`common.py`**: Shared utilities — `possibly_add_hs`, `sample_conformers`, `sample_ensemble`.
- **`mmff.py`**: MMFF force field — `calc_energy_mmff`, `optimise_mol_mmff`.
- **`xtb.py`**: xTB semi-empirical — `optimise_mol_xtb`. Requires optional xtb-python dep.
- **`align.py`**: Conformer alignment via rdShapeAlign.

### Other Modules

- **`convert.py`**: RDKit ↔ molito conversion functions.
- **`arrays.py`**: NumPy array operations (padding, one-hot encoding, adjacency matrices).

### Optional Dependencies

- **prolif + MDAnalysis**: Required for `Protein.to_prolif()` and `InteractionSet.from_system()`.
- **xtb-python**: Required for `optimise_mol_xtb()` in geometry module.

## Key Patterns

### Lazy Loading
Data loaded from HDF5 uses `LazyData` wrappers. Properties read data on-demand; call `.read()` to force loading into memory.

### Immutable Transformations
`GraphMol` methods like `permute()`, `rotate()`, `remove_hs()` return new instances via `copy_with()`.

### Bond Representation
Bonds are stored as `[n_bonds, 3]` arrays: `[start_idx, end_idx, bond_encoding_index]`. Always upper triangular (start < end). E/Z stereo is encoded via bond direction (ENDUPRIGHT/ENDDOWNRIGHT), not BondStereo.

### Canonicalization
`GraphMol.from_rdkit(canonicalise=True)` reorders atoms to RDKit's canonical ordering (requires the mol to be sanitisable). Default is `canonicalise=False` — atom order is preserved as-is. E/Z directions and chirality are preserved regardless of this flag.

## Code Formatting

Layout is handled by `ruff format` (Black-compatible, 120 cols) — do not hand-format.
Run both before committing:

```bash
ruff format . && ruff check --fix .
```

This settles line width, argument wrapping, trailing commas, blank lines and import
order automatically, so there is nothing to argue about there. `ruff check` must be
clean; config lives in `pyproject.toml`.

Conventions ruff does not enforce, which still apply:

- **Type aliases**: Define at top of file (e.g., `TArr = np.ndarray`)
- **Section headers**: Use `# *** Section Name ***` style for grouping methods in classes
- **Imports**: `from __future__ import annotations` at the start of a file if future annotations are needed
- **Static methods**: Prefer `@staticmethod` over module-level functions for class-related utilities
- **Docstrings**: Leave an empty line after docstrings before the function body, unless the body is a single line
- **Comprehensions**: Prefer a list comprehension over an indented for-loop. Chaining sequential
  comps is fine; past ~3, use a loop instead
- **Functions**: Prefer one function with a flag parameter over two near-identical functions. Don't
  add helpers or abstractions for one-time operations
- **Deletions**: When something is unused, delete it — no compatibility shims, no `# removed` comments

## Testing

Run all tests with unittest:

```bash
python -m unittest discover tests/ -v
```

## Agent Usage

Use specialized agents to improve efficiency:

- **test-runner**: Run after writing or modifying significant parts of the code to verify tests pass. Always use this agent rather than running tests manually via Bash. If you are unsure whether to run the full test suite, ask the user.
- **Explore**: Use for open-ended codebase exploration, finding files, or understanding how features work. Prefer this over multiple manual Glob/Grep calls.
- **Plan**: Use for designing implementation approaches before making significant changes.

## Dependencies

Core: numpy, rdkit, scipy, h5py, biotite, more-itertools

Optional: prolif + MDAnalysis (interactions), xtb-python (semi-empirical optimisation)
