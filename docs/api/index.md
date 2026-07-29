# API Reference

## Core Data Structures (`molito.core`)

| Class | Description |
|-------|-------------|
| [AtomSet](atoms.md#molito.core.atoms.AtomSet) | Atomic numbers, charges, chirality, residue annotations |
| [BondSet](bonds.md#molito.core.bonds.BondSet) | Bond indices and encoded types |
| [ConfSet](confs.md#molito.core.confs.ConfSet) | 3D conformer coordinates with optional weights |
| [PT](util.md#core-utilities) | Periodic table singleton |
| [PharmacophoreFinder](util.md#core-utilities) | Pharmacophore feature detection |

## Molecular Structures (`molito.mol`)

| Class | Description |
|-------|-------------|
| [GraphMol](mol.md#molito.mol.graph.GraphMol) | Molecular graph with atoms, bonds, and conformers |
| [GraphBatch](mol.md#molito.mol.graph.GraphBatch) | Batched molecules with HDF5 persistence |
| [Protein](protein.md#molito.mol.protein.Protein) | Protein chain with residue annotations |
| [BindingComplex](complex.md#molito.mol.complex.BindingComplex) | Protein-ligand complex |
| [InteractionSet](interactions.md#molito.mol.interactions.InteractionSet) | Sparse protein-ligand interactions |

## Vocabularies (`molito.core.vocab`)

| Class | Description |
|-------|-------------|
| [AtomVocab](vocab.md#molito.core.vocab.AtomVocab) | Atom token vocabulary with chirality support |
| [BondVocab](vocab.md#molito.core.vocab.BondVocab) | Bond token vocabulary for model training |
| [VocabConfig](vocab.md#molito.core.vocab.VocabConfig) | Central vocabulary configuration |

## Utilities

| Module | Description |
|--------|-------------|
| [arrays](util.md#array-operations) | Padding, one-hot encoding, adjacency matrices |
| [convert](util.md#conversion) | RDKit ↔ molito conversion |
| [geometry](util.md#geometry) | Conformer alignment, energy, optimisation, sampling |
