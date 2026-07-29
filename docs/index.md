# molito

**Molecular representation and processing toolkit.**

molito provides data structures for representing molecules as graphs with atoms, bonds, and 3D conformers, with efficient HDF5 serialization. It supports chirality, E/Z stereo, protein structures, and protein-ligand complexes.

## Key Features

- **Compact storage** — atomic numbers (uint8), charges (int8), chirality (int8), bonds (int16)
- **Stereochemistry** — E/Z bond directions and tetrahedral chirality preserved through round-trips
- **HDF5 persistence** — sharded save/load with lazy on-demand reading
- **Conformer ensembles** — multiple 3D conformers with optional Boltzmann weights
- **Protein support** — residue annotations, biotite integration, prolif interaction detection
- **Configurable vocabularies** — toggle chirality and E/Z directions for model training

## Quick Example

```python
from rdkit import Chem
from molito.mol import GraphMol, GraphBatch

# Load from RDKit — atom order is preserved as-is (canonicalise=False by default)
mol = GraphMol.from_rdkit(Chem.MolFromSmiles("C/C=C/C"))

# Access properties
print(mol.tokens)           # ['C_0', 'C_0', 'C_0', 'C_0']
print(mol.n_atoms)          # 4
print(mol.bond_types)       # bond encoding indices

# Round-trip back to RDKit (E/Z preserved)
rdkit_mol = mol.to_rdkit()
print(Chem.MolToSmiles(rdkit_mol))  # C/C=C/C

# Save a batch to HDF5
batch = GraphBatch([mol, mol])
batch.save("my_molecules/")
```

## Installation

```bash
pip install molito
```

Optional dependencies for specific features:

```bash
pip install "molito[interactions]"   # prolif + MDAnalysis for interaction detection
```

`optimise_mol_xtb()` additionally needs xtb-python, which has no working pip wheel and must
come from conda-forge:

```bash
mamba install -c conda-forge xtb-python
```
