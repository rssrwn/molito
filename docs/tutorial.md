# Tutorial: an SDF to a training batch

A complete pass through what molito is for: take a file of molecules, store it, filter it
without loading it, and get padded arrays and model indices out the other end.

The outputs below are real, from 200 BindingNet ligands with 3D conformers and two SDF tags
each (`chembl_id` and `pchembl`). Any SDF with properties will behave the same way.

## 1. Read the file

```python
from molito.mol import GraphBatch

batch = GraphBatch.from_sdf("ligands.sdf", remove_hs=True)
len(batch)                  # 200
```

SDF tags land in `mol.meta`, so the property columns come along for free:

```python
mol = batch[0]

mol.n_atoms, mol.n_bonds, mol.n_conformers   # (16, 17, 1)
dict(mol.meta)              # {'chembl_id': 'CHEMBL857', 'pchembl': 14.0}
```

Note `pchembl` came back as a `float`, not a string — numeric tags keep their type.

Stereochemistry survived the trip, which is the property most easily lost when molecules pass
through an intermediate representation:

```python
mol.to_smiles()             # 'O=C(O)CCCC[C@@H]1SC[C@@H]2NC(=O)N[C@@H]21'
```

## 2. Store it

```python
batch.save("dataset/", shard_size=64, columnar_meta=True)
# dataset/0.hdf5  dataset/1.hdf5  dataset/2.hdf5  dataset/3.hdf5
```

`shard_size` splits the output so you can load part of a dataset, and `columnar_meta=True`
writes one compressed dataset per metadata key rather than a single blob — which is what makes
the next step cheap.

## 3. Filter without building molecules

```python
import numpy as np

loaded = GraphBatch.load("dataset/", materialise=False)
len(loaded)                 # 200
```

`materialise=False` means no `GraphMol` objects are constructed yet. That matters at scale — at
two million molecules it is the difference between 0.7s and 14s — and it lets you scan metadata
without paying for objects you are about to discard:

```python
pchembl = loaded.meta_column("pchembl").astype(float)
pchembl.min(), pchembl.max()          # (1.82, 14.0)

keep = np.where(pchembl >= 7.0)[0]
len(keep)                             # 62
```

That read one HDF5 column. No molecule was built.

```python
train = loaded.subset(keep[:8].tolist())
len(train)                            # 8
```

`subset` hands back a normal materialised batch containing exactly the molecules you asked for.

!!! warning "Keep the batch open while you use it"

    `subset` materialises the *objects*, but their arrays are still read from the open HDF5
    file on access. Calling `loaded.close_hdf5()` invalidates every molecule that came out of
    it, subsets included, and a later `train.atomics` will raise. This applies to
    `materialise=True` loads too.

    Either close the batch once you have finished with the data rather than once you have
    finished selecting it, or detach the molecules you want to keep:

    ```python
    train = GraphBatch([mol.read() for mol in loaded.subset(keep[:8].tolist())])
    loaded.close_hdf5()   # train is now independent of the file
    ```

## 4. Get arrays

```python
train.atomics.shape         # (8, 35)         padded to the largest molecule
train.coords.shape          # (8, 1, 35, 3)   [batch, conformers, atoms, xyz]
train.adjacency.shape       # (8, 35, 35)
train.mask.shape            # (8, 35)         1 for real atoms, 0 for padding
```

Everything is a plain numpy array, so handing it to a model is whatever your framework wants —
`torch.from_numpy(train.atomics)` and so on. molito has no opinion and no tensor dependency.

## 5. Map to model indices

Raw atomic numbers and bond codes are not what a model embeds. Vocabularies map them into a
compact index space:

```python
from molito.core import VocabConfig

VocabConfig.atoms.resolve_tokens(train[0].tokens)   # e.g. [23, 34, 33, ...] out of 40
VocabConfig.bonds.resolve_types(train.adjacency)    # (8, 35, 35) out of 12
```

The useful part is that this is a *view* on the stored data, not a property of it. Turn off E/Z
directions and the same dataset resolves into a smaller index space, with no rewrite:

```python
VocabConfig.set_directions(False)
VocabConfig.bonds.resolve_types(train.adjacency)    # same shape, now out of 8
VocabConfig.reset()
```

See [Concepts](concepts.md#two-levels-of-bond-indexing) for why storage and model indices are
deliberately separate.

## The whole thing

```python
import numpy as np
from molito.core import VocabConfig
from molito.mol import GraphBatch

GraphBatch.from_sdf("ligands.sdf", remove_hs=True).save("dataset/", shard_size=64, columnar_meta=True)

loaded = GraphBatch.load("dataset/", materialise=False)
potent = np.where(loaded.meta_column("pchembl").astype(float) >= 7.0)[0]

train = loaded.subset(potent[:8].tolist())
atom_indices = VocabConfig.atoms.resolve_tokens(train[0].tokens)
bond_indices = VocabConfig.bonds.resolve_types(train.adjacency)

# ... use the arrays, then close. Closing earlier would invalidate `train`.
loaded.close_hdf5()
```

## Where to go next

- [Concepts](concepts.md) — the two-level bond indexing, canonicalisation, deferred loading
- [Stereochemistry](stereochemistry.md) — how stereo survives reordering, and what would break it
- [API Reference](api/index.md) — everything else
