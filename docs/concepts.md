# Concepts

The parts of molito that are not obvious from the API surface. If something in the library
surprises you, the explanation is probably here.

## Two levels of bond indexing

This is the single most confusing thing in molito, and it exists for a good reason.

Bonds are stored on disk as **`BondEncoding` indices** — a fixed table of 12 entries that
*never changes*, because changing it would make every existing HDF5 file unreadable. Models,
though, want a compact index space that reflects the features they actually use. Those are
**vocabulary indices**, and they move whenever you reconfigure the vocabulary.

```python
from rdkit import Chem
from molito.core import BondEncoding, VocabConfig
from molito.mol import GraphMol

mol = GraphMol.from_rdkit(Chem.MolFromSmiles("C/C=C/C"))

mol.bond_types                                   # [7, 2, 7]  <- storage indices
[BondEncoding.get_token(int(i)) for i in mol.bond_types]
                                                 # ['1_F_U', '2_F', '1_F_U']
```

The token format is `bondorder_aromatic[_direction]`: `1_F_U` is a single, non-aromatic bond
carrying the `ENDUPRIGHT` direction tag, and `2_F` is a plain double bond.

To get model indices, resolve them through the vocabulary:

```python
VocabConfig.bonds.resolve_types(mol.bond_types)  # [7, 2, 7], out of len(...) == 12
```

They happen to coincide here. Now disable E/Z directions — the *stored* data does not change,
but what the model sees does:

```python
VocabConfig.set_directions(False)
VocabConfig.bonds.resolve_types(mol.bond_types)  # [1, 2, 1], out of len(...) == 8
```

The directional bonds fell back to their plain single-bond form and the index space shrank
from 12 to 8. **This is the whole point of the split**: you can train a model that ignores
E/Z without rewriting a byte of your dataset.

The same applies to atoms, where chirality is the feature that folds away:

```python
mol = GraphMol.from_rdkit(Chem.MolFromSmiles("N[C@@H](C)C(=O)O"))

mol.tokens            # ['N_0', 'C_0_CW', 'C_0', 'C_0', 'O_0', 'O_0']
mol.charged_symbols   # ['N_0', 'C_0',    'C_0', 'C_0', 'O_0', 'O_0']  <- chirality dropped

VocabConfig.atoms.resolve_tokens(mol.tokens)   # [23, 34, 33, 33, 5, 5]
```

`resolve_tokens` falls back automatically: if the vocabulary was built without chirality,
`C_0_CW` resolves to whatever `C_0` maps to rather than raising.

!!! warning "VocabConfig is process-global"

    `VocabConfig.set_chirality()` and `set_directions()` mutate class-level state, so they
    change behaviour for everything in the process. You cannot currently have two models with
    different vocabularies in one script. Call `VocabConfig.reset()` to restore defaults.

    Nothing on disk depends on this — HDF5 stores `BondEncoding` indices and raw
    atomics/charges/chirality, never vocabulary indices — so a misconfigured vocabulary can
    never corrupt a dataset.

## Atom ordering and `canonicalise`

`GraphMol.from_rdkit(mol)` preserves the atom order it was given. That is the default because
it is the least surprising behaviour: what you put in is what you get back.

```python
mol = GraphMol.from_rdkit(rdkit_mol)                      # order preserved
mol = GraphMol.from_rdkit(rdkit_mol, canonicalise=True)   # RDKit canonical order
```

Reach for `canonicalise=True` when you are ingesting the same molecules from different sources
— an SDF and a SMILES string of one compound will have different atom orders, and canonical
ordering makes the stored representation identical. It requires the molecule to be sanitisable
and raises `ValueError` if it is not.

Stereochemistry survives either way. That is not free, and [Stereochemistry](stereochemistry.md)
explains the invariant that makes it work.

## Materialised and deferred loading

`GraphBatch.load(path)` builds every `GraphMol` — and its `AtomSet`, `BondSet` and `ConfSet` —
up front. Passing `materialise=False` defers that construction until each molecule is accessed.

```python
batch = GraphBatch.load("dataset/")                       # builds all the objects now
batch = GraphBatch.load("dataset/", materialise=False)    # builds them on access
```

**What this does not change is when array data is read.** Coordinates, atomics and bonds go
through `LazyData` in both modes and are read from HDF5 on property access either way. Metadata
is handled identically in both modes too. The only thing deferred is Python object
construction, so the saving scales with the *number* of molecules rather than their size:

| molecules | `materialise=True` | `materialise=False` |
|---|---|---|
| 100k | 0.41 s, 237 MB | 0.03 s, 164 MB |
| 500k | 3.11 s, 596 MB | 0.16 s, 252 MB |
| 2M | 14.36 s, 1922 MB | 0.74 s, 565 MB |

Below roughly 100k molecules it is not worth thinking about. Above a million it is the
difference between opening a dataset instantly and waiting fifteen seconds for it.

Two behavioural differences to know about with `materialise=False`:

- Each `batch[i]` returns a **fresh wrapper**, so `batch[i] is batch[i]` is `False`. You cannot
  hold a reference and expect `mol.atoms = ...` to persist across lookups.
- `subset(idxs)` returns a normal materialised `GraphBatch` containing exactly the molecules you
  selected — which is what you want for assembling a training batch.

For filtering large datasets, `meta_column` reads one metadata key across every shard without
constructing any molecules at all:

```python
batch = GraphBatch.load("dataset/", materialise=False)
ids = batch.meta_column("id")                  # no GraphMols built
train = batch.subset(np.where(ids != "")[0])   # materialises only the selection
```

## Loaded molecules depend on the open file

Because array access always goes through `LazyData`, every molecule from a loaded batch holds a
reference to its HDF5 file. Closing the batch invalidates them:

```python
batch = GraphBatch.load("dataset/")
train = batch.subset([0, 1, 2])

batch.close_hdf5()
train.atomics          # raises - the file is gone
```

This applies to `materialise=True` loads and to `subset` results, despite `subset` being
described as materialising: it materialises the *objects*, not the arrays behind them. Close the
batch once you have finished with the data, not once you have finished selecting it.

To keep molecules beyond the file's lifetime, `read()` pulls their arrays into memory:

```python
train = GraphBatch([mol.read() for mol in batch.subset([0, 1, 2])])
batch.close_hdf5()
train.atomics          # fine
```

## Metadata is read-only after loading

`mol.meta` on a loaded molecule is a read-only mapping. This matches every other HDF5-backed
attribute — `mol.atomics` and friends hand back fresh copies rather than a live view — and it
stops accidental writes that would silently fail to reach disk.

```python
mol.meta["k"] = "v"        # TypeError

meta = dict(mol.meta)      # mutable copy
meta["k"] = "v"
mol.meta = meta            # or reassign wholesale
```

### On-disk formats

`columnar_meta=True` writes one gzip-compressed dataset per key. It is markedly smaller, lets
you read a single key without touching the rest, and makes `meta_column` a single dataset read:

```python
batch.save("dataset/", columnar_meta=True)
```

It suits metadata that is a flat set of scalars. Missing keys are filled with empty strings, and
anything it cannot represent as a column — a nested dict, a list — is stringified, which loses
the structure.

`columnar_meta=False` writes one gzip-compressed JSON document per shard instead. Use it when
your metadata is nested or ragged. JSON has no array type, so numpy arrays come back as lists
and numpy scalars as Python scalars; everything JSON models natively round-trips unchanged.

### Reading older files

Before the JSON format existed, a non-columnar save wrote a **pickled** blob. Unpickling runs
arbitrary code, so loading a shard from an untrusted source could execute anything. Those shards
are therefore refused unless you opt in:

```python
GraphBatch.load("old_dataset/")                      # raises, naming the file
GraphBatch.load("old_dataset/", allow_pickle=True)   # loads it
```

Columnar shards have never contained pickle and are unaffected. Nothing molito writes now
contains pickle at all.

## Bond storage layout

Bonds live in an `[n_bonds, 3]` array of `[start, end, encoding]`, always with `start < end`
— each bond appears once, not twice.

Two invariants hold and are load-bearing for stereochemistry: **row order is never changed**,
and **the two index columns of a row are never swapped**. Sorting bond rows or re-imposing
`start < end` after a permutation would silently invert stereochemistry. If you are tempted to
reorder bonds, read [Stereochemistry](stereochemistry.md) first.

## On-disk format versioning

Every shard carries a `molito_format_version` attribute at its root, plus the package version
that wrote it. Readers refuse shards written by a newer molito rather than misreading datasets
whose layout has moved.

Shards written before versioning existed have no attribute and load as version 0. They remain
readable — the layout did not change when the stamp was introduced.

Adding an *optional* dataset does not require a version bump, because readers treat a missing
dataset as "not stored". That is how per-atom chain IDs were added without breaking existing
files.
