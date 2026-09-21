# Utilities

## Array Operations

::: molito.arrays.pad_arrays

::: molito.arrays.one_hot_encode

::: molito.arrays.adj_from_edges

## Conversion

::: molito.convert.mol_is_valid

::: molito.convert.smiles_from_mol

::: molito.convert.mol_from_smiles

::: molito.convert.mol_from_atoms

## Core Utilities

::: molito.core.pt.PT
    options:
      show_root_heading: true

::: molito.core.pharmacophore.PharmacophoreFinder
    options:
      show_root_heading: true

## Geometry

`calc_energy_xtb` evaluates each conformer without optimisation. It returns energies in
**kcal/mol**, matching MMFF and `optimise_mol_xtb`. Both xTB functions accept
`units="hartree"` for native energies (1 Hartree is approximately 627.509474 kcal/mol). One conformer
returns a float, multiple conformers return a list in iteration order, and failed conformers
return `None`. Missing hydrogens get RDKit-generated coordinates without minimisation;
provide all hydrogens explicitly when evaluating a specific geometry.

```python
from molito.geometry import calc_energy_xtb

energy = calc_energy_xtb(rdkit_mol)
solvated_energy = calc_energy_xtb(rdkit_mol, solvent="water")
native_energy = calc_energy_xtb(rdkit_mol, units="hartree")
```

The xTB optimiser always runs in native atomic units internally, so selecting output units
changes neither its geometry nor convergence tolerances. Its final and initial energies use
the same requested units. Before 0.2.0, it returned Hartree; existing callers that expect
those values should pass `units="hartree"`. Previously saved energy values are not converted.

Since 0.2.1, optimisation also adds every missing hydrogen without MMFF preparation. This permits
xTB-supported molecules lacking MMFF parameters and inputs with only some hydrogens explicit.
The returned molecule retains exactly the input atoms; energies refer to the full with-Hs
geometry used for the calculation. Supply all hydrogens explicitly if you need to retain that
complete optimised geometry or reproduce its final energy with another single-point call.

::: molito.geometry.sample_conformers

::: molito.geometry.sample_ensemble

::: molito.geometry.calc_energy_mmff

::: molito.geometry.optimise_mol_mmff

::: molito.geometry.calc_energy_xtb

::: molito.geometry.optimise_mol_xtb

::: molito.geometry.align_conf

::: molito.geometry.align_best_conf
