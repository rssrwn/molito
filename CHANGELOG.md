# Changelog

All notable changes to molito are recorded here, so you can tell what an upgrade will do
before you take it.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versioning
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Changes to the **on-disk HDF5 layout** are called out explicitly, since those affect data you
already have rather than just code you can update. Every shard records the format version that
wrote it, and readers refuse shards from a newer molito rather than misreading them.

## [Unreleased]

## [0.1.0] - 2026-07-30

First public release.

### Added

- **Molecular graphs** — `GraphMol` and `GraphBatch`, storing atomic numbers (uint8), formal
  charges (int8), chirality (int8) and bonds (int16).
- **Stereochemistry that survives reordering** — E/Z bond directions and tetrahedral chirality
  are preserved through canonicalisation and arbitrary atom permutation. See
  [Stereochemistry](docs/stereochemistry.md) for the invariant this rests on.
- **Conformer ensembles** — `ConfSet` holds `[n_confs, n_atoms, 3]` coordinates with optional
  Boltzmann weights.
- **Proteins and complexes** — `Protein`/`ProteinBatch` with residue and chain annotations, and
  `BindingComplex`/`ComplexBatch` for protein-ligand systems, both with padded array accessors.
- **Sparse interactions** — `InteractionSet`, via prolif with the `interactions` extra.
- **HDF5 persistence** — sharded save and load. Array data is read on property access rather
  than at load time, and `materialise=False` additionally defers building the Python objects,
  which matters past roughly a million molecules.
- **Metadata storage** — a columnar format with one compressed dataset per key, allowing a
  single key to be scanned via `meta_column` without constructing molecules, plus a JSON format
  for nested or ragged metadata.
- **File IO** — `GraphMol.from_smiles`/`to_smiles`, and `GraphBatch.from_smiles`/`from_sdf`/
  `to_sdf`. SDF tags are carried into `mol.meta`.
- **Vocabularies** — `AtomVocab` and `BondVocab` map storage encodings to compact model indices,
  with chirality and E/Z directions toggleable without rewriting a dataset.
- **Geometry** — conformer sampling and Boltzmann-weighted ensembles, MMFF energies and
  optimisation, xTB optimisation (with `xtb-python`), and shape alignment.
- **Typed** — the package ships `py.typed`, and annotations are checked in CI.

### On-disk format

- Introduces **format version 1**. Every shard carries `molito_format_version` and the package
  version that wrote it.
- Shards written by pre-release molito carry no version attribute, load as version 0, and remain
  readable — the layout did not change when the stamp was added.
- Metadata written before the JSON format used Python `pickle`, which executes arbitrary code on
  load. Those shards now require an explicit `allow_pickle=True`. Columnar shards were never
  affected, and nothing molito writes now contains pickle.

[Unreleased]: https://github.com/rssrwn/molito/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/rssrwn/molito/releases/tag/v0.1.0
