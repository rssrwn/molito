# Changelog

All notable changes to molito are recorded here, so you can tell what an upgrade will do
before you take it.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versioning
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Changes to the **on-disk HDF5 layout** are called out explicitly, since those affect data you
already have rather than just code you can update. Every shard records the format version that
wrote it, and readers refuse shards from a newer molito rather than misreading them.

## [Unreleased]

### Fixed

- Conformer array indexing and `select_topk` retain the selected weights without renormalising.
  Small positive weights are no longer treated as zero. Random sampling and single-conformer
  molecule extraction continue to omit ensemble weights.
- Graph, protein and complex batch subsets and `from_batches` results borrow their source data;
  closing them no longer closes the source files. Keep source batches open while using these views.
  Batch `.read()` detaches data into memory, and context managers close owned files on exit.
- Atom, charge, chirality, residue ID and bond arrays reject fractional, nonfinite or overflowing
  values before compact integer conversion. Integral floating-point inputs remain accepted.
  Combined complex bond offsets also reject overflow instead of wrapping to negative indices.
- xTB optimisation adds all missing hydrogens without requiring MMFF parameters, including
  partially hydrogenated inputs. Returned molecules preserve the original atoms and explicit Hs.
  Initial and final energies describe the complete with-Hs calculation; initial energies and
  optimisation paths may differ from 0.2.0 because the MMFF preparation step is removed.
- Conformer alignment and RMSD deduplication accept noncontiguous RDKit conformer IDs.
- Array padding promotes mixed dtypes instead of truncating values and rejects mismatched
  trailing shapes instead of broadcasting. Matching input dtypes remain unchanged.
- Adjacency construction rejects negative/out-of-range endpoints and mismatched edge arrays.

### On-disk format

- No changes to HDF5 layouts, format versions, storage dtypes, bond encodings or vocabulary indices.
  Existing datasets remain readable and are not rewritten. Previously overflowed values cannot
  be recovered by upgrading; regenerate affected data from the original inputs.

## [0.2.0] - 2026-09-19

### Added

- `calc_energy_xtb` for single-point energies in kcal/mol, including conformer ensembles,
  per-atom normalisation, solvent and spin settings, with live native-backend tests.
- `MolRepr`, abstract `StringMol`, `SmilesMol`, and `RDKitMol`, with metadata-preserving
  `.to(TargetClass)` conversion, optional strict loss checks, and native persistence.
- Invalid string storage and tokenisation, plus homogeneous string/RDKit `MolBatch`
  collections with sharded HDF5 storage. Graph batches can convert to other representations.
- Character and custom-regex tokenisers, a filtered MolBART/Chemformer SMILES vocabulary,
  lossless character fallback, deterministic vocabulary fitting, explicit custom tokens,
  `<PAD>`/`<MASK>`/`<BOS>`/`<EOS>`, and versioned JSON tokeniser artifacts.

### Changed

- Both xTB functions return kcal/mol by default, matching MMFF. Use `units="hartree"` to
  retain native units, including the pre-0.2.0 optimiser behaviour. Both initial and final
  optimiser energies are converted; optimisation itself still runs in atomic units.
  Existing stored energies are not modified.
- Split `molito.core.vocab` into a package; existing imports and graph vocabulary indices
  remain unchanged. New native HDF5 shards have separate representation/version markers;
  graph/protein shards retain format 1; complex and native shards use format 2 (see below).

### Fixed

- RDKit-to-graph conversion handles molecules with no bonds, including single atoms and salts.
- Failed graph sanitisation returns `None` even when conformer weights are present.
- Per-conformer shifts, unweighted conformer permutation, and NumPy ensemble reconstruction.
- Padded graph batch bonds, in-memory batch subsets, and protein shard limits that omitted the final shard.
- Default chirality for older atom datasets, and file handle cleanup when shard loading fails.
- Validation of chirality lengths, conformer weights, array indices, temperatures and shard sizes.
- MMFF energies for noncontiguous conformer IDs, rejection of unsupported forcefield parameters,
  and handling of failed/nonfinite ensemble energies.
- xTB now receives molecular charge and unpaired electrons, resolves solvent names to the API enum,
  and rejects failed optimisation. An optional `uhf` argument permits explicit spin selection.
  Iteration-limited results remain available by default; `allow_unconverged=False` requires convergence.
- Regex tokenisers accept global inline flags alongside literal extra tokens.
- Release tags now run the full validation workflow before building and publishing. Wheel smoke
  tests use isolated imports and check the bundled SMILES vocabularies.

### On-disk format

- **Format version 2** writes complex interactions as JSON. Earlier interaction payloads used
  pickle even when metadata was JSON or columnar; loading them now requires `allow_pickle=True`.
- Version 0/1 files remain readable, with explicit opt-in for any legacy pickle payloads.
  Older molito versions reject new complex/native version 2 shards. Graph/protein writers
  retain format 1 and remain readable by 0.1.1. Array layouts are unchanged.
- Public graph/protein/complex `to_bytes`/`from_bytes` APIs continue to use pickle and require
  trusted inputs. The no-pickle guarantee applies to newly written HDF5 shards.

## [0.1.1] - 2026-09-13

### Fixed

- RDKit import preserves bond endpoint order instead of forcing `start < end`,
  preventing loss or inversion of E/Z stereochemistry when directional bonds
  run from a higher atom index to a lower one. Existing files remain readable;
  affected graphs previously written with reversed endpoints must be regenerated
  from their original molecules to restore the intended stereochemistry.
- RDKit import now normalises leftover directional tags on aromatic bonds before
  kekulisation, avoiding `KeyError: '2_T_D'` / `'2_T_U'` while preserving perceived
  alkene stereochemistry. This also applies when `clean_stereo=False`, without
  changing the input molecule or its atom chirality tags. Existing bond storage
  codes and the HDF5 format are unchanged.

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
  affected by the metadata change. Complex interaction payloads remained pickled until 0.2.0.

[Unreleased]: https://github.com/rssrwn/molito/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/rssrwn/molito/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/rssrwn/molito/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/rssrwn/molito/releases/tag/v0.1.0
