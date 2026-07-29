# Stereochemistry

molito preserves tetrahedral chirality and double-bond E/Z through storage, atom reordering and
arbitrary permutation. This page explains how, because the mechanism constrains what you can
safely do to a `BondSet` — and because silently flipping a stereocentre changes the molecule
without changing anything that looks wrong.

## Relative tags, not absolute descriptors

Both stereo types are stored as **relative** tags rather than absolute CIP descriptors:

| | relative tag (stored) | absolute descriptor (derived) |
|---|---|---|
| tetrahedral | `CW` / `CCW` | `R` / `S` |
| double bond | `BondDir` (`/` and `\` in SMILES) | `E` / `Z` |

Double-bond stereo is therefore recorded as a direction tag on the **single bonds adjacent to
the double bond**, not as `STEREOE`/`STEREOZ` on the double bond itself.

That choice is partly forced and partly deliberate. For tetrahedral centres it is forced:
RDKit's `Atom.SetChiralTag` accepts CW/CCW and there is no `SetCIPCode` equivalent, so the
relative tag is the only handle available. For double bonds RDKit *does* expose absolute stereo
via `Bond.SetStereo` plus `Bond.SetStereoAtoms`, but using it would mean storing a pair of CIP
reference-atom indices per double bond and remapping them under every permutation. That
relocates the bookkeeping rather than removing it. Storing the relative tag keeps both stereo
types symmetric and reduces the whole problem to one invariant.

## The invariant

`BondSet.permute_atoms` relabels atom indices through the permutation map, and deliberately
does **not**:

1. reorder bond rows, or
2. swap the two index columns within a row.

Everything else follows from that.

**Tetrahedral chirality** depends on the cyclic order in which RDKit encounters a chiral atom's
neighbours. `mol_from_atoms` iterates bond rows in array order and calls `AddBond` in that
order, so preserving row order preserves each centre's neighbour ordering, and CW stays CW.

**Double-bond stereo** depends on `BondDir` being read relative to the bond's begin and end
atoms — "the end atom is up-and-right of the begin atom". A row stored as
`[a, b, ENDUPRIGHT]` becomes `[p(a), p(b), ENDUPRIGHT]` after permutation, never
`[p(b), p(a), ...]`. RDKit therefore receives the same physical begin and end atoms, just
relabelled, and the geometric meaning of the tag is unchanged.

## What breaks it

Anything that violates either half of the invariant:

- **Lex-sorting bond rows** by `(start, end)` after remapping indices. This flips chirality at
  any centre whose neighbour cyclic order changes.
- **Re-imposing `start < end` row-by-row** after permutation, i.e. swapping the two index
  columns when `start > end`. This exchanges the begin/end roles of the bond, inverting the
  meaning of the direction tag and turning E into Z.
- **Grouping bond rows per atom**, or dropping, inserting or otherwise reordering rows.

There is a third hazard that has nothing to do with permutation: **re-perceiving stereo from
incomplete or approximate 3D coordinates**. `mol_from_atoms` deliberately skips
`AssignStereochemistryFrom3D` when the caller has supplied chirality tags, because coordinates
produced by an ML model or from a noisy source can disagree with the intended stereochemistry,
and silently overwriting explicit tags with a guess is worse than not perceiving at all.

## Cleaning over-declared stereo

Some sources over-declare stereochemistry — SMILES or SDFs carrying `CHI_TETRAHEDRAL` tags on
atoms that are not genuine, CIP-resolvable stereocentres. Downstream consumers that key off
`atoms.chirality` will then fire on tags that mean nothing.

```python
mol = GraphMol.from_rdkit(rdkit_mol, clean_stereo=True)
```

This copies the input and runs `Chem.AssignStereochemistry(cleanIt=True, force=True)` before
reading stereo, dropping tags from atoms that cannot carry them. The input molecule is never
mutated. The default is `False`, which stores exactly what it was given.

## Testing

`tests/repr/test_stereo.py` pins down both the observable behaviour and the underlying
invariant. Alongside round-trip tests for E, Z, CW and CCW through canonicalisation, it
exhaustively checks **all 24 permutations** of four-atom stereo molecules, and asserts the
row-order and column-order invariant directly in `TestBondRowOrderInvariant`.

If you are changing anything in `BondSet`, that file is the specification.
