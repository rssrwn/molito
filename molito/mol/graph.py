from __future__ import annotations

import copy
import pickle
from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np
from more_itertools import grouper
from rdkit import Chem
from scipy.spatial.transform import Rotation

from molito.arrays import pad_arrays
from molito.convert import mol_from_atoms, mol_from_smiles, smiles_from_mol
from molito.core._checks import PICKLE_PROTOCOL, check_dict_key, check_type
from molito.core.atoms import AtomSet
from molito.core.bonds import BondSet
from molito.core.confs import ConfSet
from molito.core.format import check_format, stamp_format
from molito.core.lazydata import LazyData
from molito.core.meta import column_array, load_meta, save_meta

# Type aliases
TArr = np.ndarray


class GraphMol:
    """A molecular graph combining atoms, bonds, and optional 3D conformers.

    The primary molecule class in molito. Supports RDKit conversion with chirality and E/Z stereo
    preservation, geometric transformations, canonical atom ordering, HDF5 serialization, and
    byte-level pickling.

    Args:
        atoms: AtomSet with atomic numbers, charges, and optional chirality/residue annotations.
        bonds: BondSet with bond indices and types.
        confs: Optional ConfSet with 3D conformer coordinates.
        meta: Optional metadata dictionary (serialized alongside the molecule).
    """

    __slots__ = ("atoms", "bonds", "confs", "meta")

    def __init__(
        self, atoms: AtomSet, bonds: BondSet, confs: ConfSet | None = None, meta: dict[str, str] | None = None
    ):
        self._check_mol(atoms, bonds, confs=confs)

        meta = {} if meta is None else meta

        self.atoms = atoms
        self.bonds = bonds
        self.confs = confs
        self.meta = meta

    @classmethod
    def _load_unchecked(cls, atoms: AtomSet, bonds: BondSet, confs: ConfSet | None, meta) -> GraphMol:
        """Fast loading which skips validation checks."""

        obj = cls.__new__(cls)
        obj.atoms = atoms
        obj.bonds = bonds
        obj.confs = confs
        obj.meta = meta if meta is not None else {}
        return obj

    # *** Publicly exposed properties ***

    @property
    def n_atoms(self) -> int:
        return len(self)

    @property
    def n_heavy_atoms(self) -> int:
        atom_mask = self.atomics != 1
        return atom_mask.sum().item()

    @property
    def n_bonds(self) -> int:
        return len(self.bonds)

    @property
    def n_conformers(self) -> int:
        return 0 if self.confs is None else len(self.confs)

    @property
    def atomics(self) -> TArr:
        return self.atoms.atomics

    @property
    def charges(self) -> TArr:
        return self.atoms.charges

    @property
    def charged_symbols(self) -> list[str]:
        return self.atoms.charged_symbols

    @property
    def tokens(self) -> list[str]:
        return self.atoms.tokens

    @property
    def coords(self) -> TArr | None:
        return self.confs.coords if self.confs is not None else None

    @property
    def conf_weights(self) -> TArr | None:
        return self.confs.weights if self.confs is not None else None

    @property
    def bond_indices(self) -> TArr:
        return self.bonds.indices

    @property
    def bond_types(self) -> TArr:
        return self.bonds.types

    @property
    def adjacency(self) -> TArr:
        return self.bonds.adj_matrix(len(self))

    @property
    def seq_length(self) -> int:
        return len(self)

    # *** Basic indexing and utility functions ***

    def __len__(self) -> int:
        return len(self.atoms)

    def __str__(self) -> str:
        if self.meta is not None and "str_id" in self.meta:
            return self.meta["str_id"]

        return super().__str__()

    def get_conformer(self, idx: int) -> TArr:
        return self.confs.get_conformer(idx)

    def neighbour_ranks(self, stereo_only: bool = False) -> TArr:
        """Per-atom neighbour rank matrix. See BondSet.neighbour_ranks for details.

        Args:
            stereo_only: if True, only chiral atoms (chirality != 0) and atoms
                involved in directional bonds have their ranks populated; all
                other rows stay 0. Preserves permutation equivariance at
                non-stereo atoms.
        """

        chiral_mask = (self.atoms.chirality != 0) if stereo_only else None
        return self.bonds.neighbour_ranks(len(self), chiral_mask=chiral_mask)

    def mol_with_conformer(self, idx: int) -> GraphMol:
        """Return a copy of this mol carrying only the conformer at `idx`.

        Weights are intentionally not preserved — a per-conformer weight only carries
        information *relative to the other conformers in the ensemble*, so it's
        meaningless on a singleton. Pulling out a conf with weight 0 (e.g. extreme-
        energy entries that underflowed in float32) would also fail the ConfSet's
        non-zero-sum check.
        """

        if self.confs is None:
            raise ValueError("Cannot select a conformer from a mol with no confs.")

        coords = self.confs.get_conformer(idx)  # [n_atoms, 3]; ConfSet ctor adds the conf dim
        confs = ConfSet(coords)
        return self.copy_with(confs=confs)

    def read(self) -> GraphMol:
        """Force the data to be read into memory if it isn't already.

        Molecules from a loaded batch read their arrays from HDF5 on access, so they stop
        working once the batch is closed. Call this to get a molecule that outlives the file:

            train = GraphBatch([mol.read() for mol in loaded.subset(idxs)])
            loaded.close_hdf5()

        Mirrors `Protein.read`.
        """

        atoms = self.atoms.read()
        bonds = self.bonds.read()
        confs = self.confs.read() if self.confs is not None else None

        # Materialise: meta may be a live HDF5-backed view that would break with the file.
        meta = dict(self.meta) if self.meta is not None else None
        return GraphMol(atoms, bonds, confs=confs, meta=meta)

    def copy_with(
        self, atoms: AtomSet | None = None, bonds: BondSet | None = None, confs: ConfSet | None = None
    ) -> GraphMol:

        atoms = self.atoms.copy() if atoms is None else atoms.copy()
        bonds = self.bonds.copy() if bonds is None else bonds.copy()

        if confs is not None:
            confs = confs.copy()
        elif self.confs is not None:
            confs = self.confs.copy()

        # Materialise first: meta may be a live HDF5-backed _ColumnMetaView which holds
        # h5py.Dataset handles that can't be pickled/deepcopied.
        meta = copy.deepcopy(dict(self.meta)) if self.meta is not None else None
        return GraphMol(atoms, bonds, confs=confs, meta=meta)

    def copy(self) -> GraphMol:
        return self.copy_with()

    def permute(self, indices: list[int] | TArr) -> GraphMol:
        atoms = self.atoms.permute_atoms(indices)
        bonds = self.bonds.permute_atoms(indices)
        confs = self.confs.permute_atoms(indices) if self.confs is not None else None
        mol = self.copy_with(atoms=atoms, bonds=bonds, confs=confs)
        return mol

    def remove_hs(self) -> GraphMol:
        indices = np.arange(len(self.atomics))
        non_h_idxs = indices[self.atomics != 1]
        return self.permute(non_h_idxs)

    def drop_3d(self) -> GraphMol:
        """Returns a copy of this molecule without confs"""

        new_mol = self.copy()
        new_mol.confs = None
        return new_mol

    def order_by_bonds(self, canonical: bool = True) -> GraphMol:
        """Returns a permuted version of the molecule where the ordering of atoms is defined by RDKit.

        The molecule must be sanitisable, otherwise a ValueError is raised.
        """

        rdkit_mol = self.to_rdkit(sanitise=True)
        if rdkit_mol is None:
            raise ValueError("Cannot canonicalise a molecule that fails RDKit sanitisation.")

        _ = Chem.MolToSmiles(rdkit_mol, canonical=canonical, doRandom=not canonical)
        atom_order = rdkit_mol.GetPropsAsDict(True, True)["_smilesAtomOutputOrder"]
        return self.permute(list(atom_order))

    def pad(self, n_atoms: int, pad_atomic: int = 0, pad_charge: int = 0) -> GraphMol:
        """Pad the mol to length n_atoms"""

        atoms = self.atoms.pad(n_atoms, pad_atomic=pad_atomic, pad_charge=pad_charge)
        confs = self.confs.pad(n_atoms) if self.confs is not None else None

        # Note we don't add any bonds — 'pad' atoms are disconnected from the rest of the graph
        padded_mol = self.copy_with(atoms=atoms, confs=confs)
        return padded_mol

    def _check_mol(self, atoms, bonds, confs=None):
        check_type(atoms, AtomSet, "atoms")
        check_type(bonds, BondSet, "bonds")

        if confs is not None:
            check_type(confs, ConfSet, "confs")

        if len(bonds) > 0 and bonds.min_index < 0:
            raise RuntimeError("Bond indices within a molecule cannot be negative.")
        if len(bonds) > 0 and bonds.max_index >= len(atoms):
            raise RuntimeError("The maximum atom index in bonds cannot be larger the largest atom index.")
        if confs is not None and confs.n_atoms != len(atoms):
            raise RuntimeError("The number of atoms in the atom set and conf set must be the same.")

    # *** Geometric specific functions ***

    def zero_com(self) -> GraphMol:
        confs = self.confs.zero_com()
        return self.copy_with(confs=confs)

    def rotate(self, rotation: Rotation | list[Rotation]) -> GraphMol:
        confs = self.confs.rotate(rotation)
        return self.copy_with(confs=confs)

    def shift(self, shift: TArr | list[TArr]) -> GraphMol:
        confs = self.confs.shift(shift)
        return self.copy_with(confs=confs)

    def scale(self, scale: float) -> GraphMol:
        confs = self.confs.scale(scale)
        return self.copy_with(confs=confs)

    # *** IO and conversion utility functions ***

    @staticmethod
    def from_rdkit(rdkit_mol: Chem.rdchem.Mol, canonicalise: bool = False, clean_stereo: bool = False) -> GraphMol:
        """Create a GraphMol from an RDKit molecule.

        Args:
            rdkit_mol: RDKit molecule object.
            canonicalise: If True, reorder atoms to RDKit's canonical ordering. Useful when ingesting
                from a source with non-canonical atom order (SDF, CIF) and you want the stored
                representation to be deterministic across input formats. Requires the molecule to be
                sanitisable (raises ValueError otherwise). Default False -- atom order is preserved
                as-is. E/Z bond directions and chirality are preserved regardless of this flag.
            clean_stereo: If True, copy `rdkit_mol` and run
                `Chem.AssignStereochemistry(cleanIt=True, force=True)` on the copy before
                reading atom + bond stereo. Drops CHI_TETRAHEDRAL tags from atoms that aren't
                CIP-resolvable stereocentres (and BondDir tags from double bonds that can't
                actually carry E/Z). Useful when ingesting datasets where source SMILES /
                SDFs over-declare stereo — downstream consumers keying off `atoms.chirality`
                or bond direction (e.g. inference-time stereo guidance) otherwise fire on
                these "ghost" tags. Input mol is never mutated. Default False is verbatim.
        """

        if clean_stereo:
            rdkit_mol = Chem.Mol(rdkit_mol)
            Chem.AssignStereochemistry(rdkit_mol, cleanIt=True, force=True)

        atoms = AtomSet.from_rdkit(rdkit_mol)
        bonds = BondSet.from_rdkit(rdkit_mol)

        confs = None
        if rdkit_mol.GetNumConformers() > 0:
            confs = ConfSet.from_rdkit(rdkit_mol)

        mol = GraphMol(atoms, bonds, confs=confs)

        if canonicalise:
            mol = mol.order_by_bonds(canonical=True)

        return mol

    @staticmethod
    def from_smiles(
        smiles: str, canonicalise: bool = False, clean_stereo: bool = False, explicit_hs: bool = False
    ) -> GraphMol:
        """Create a GraphMol from a SMILES string.

        The result has no conformers -- use `molito.geometry.sample_conformers` to generate them.

        Args:
            smiles: SMILES string. Explicit hydrogens written in the SMILES are preserved.
            canonicalise: See `from_rdkit`.
            clean_stereo: See `from_rdkit`.
            explicit_hs: If True, add all implicit hydrogens as explicit atoms. This changes
                the atom count.

        Raises:
            ValueError: If the SMILES cannot be parsed.
        """

        rdkit_mol = mol_from_smiles(smiles, embed_hs=explicit_hs)

        if rdkit_mol is None:
            raise ValueError(f"Could not parse SMILES {smiles!r}")

        return GraphMol.from_rdkit(rdkit_mol, canonicalise=canonicalise, clean_stereo=clean_stereo)

    def to_smiles(self, canonical: bool = True, explicit_hs: bool = False) -> str:
        """Return a SMILES string for this molecule, with stereochemistry preserved.

        Args:
            canonical: Whether to produce RDKit's canonical SMILES.
            explicit_hs: Whether to write hydrogens explicitly.

        Raises:
            ValueError: If the molecule cannot be sanitised into a valid RDKit mol.
        """

        rdkit_mol = self.to_rdkit(sanitise=True)

        if rdkit_mol is None:
            raise ValueError("Cannot write SMILES for a molecule that fails RDKit sanitisation.")

        return smiles_from_mol(rdkit_mol, canonical=canonical, explicit_hs=explicit_hs)

    @staticmethod
    def _from_core_repr(dict_repr: dict[str, dict[str, TArr]]) -> GraphMol:
        check_type(dict_repr, dict, "unpickled object")
        check_dict_key(dict_repr, "atoms")
        check_dict_key(dict_repr, "bonds")

        atoms = AtomSet.from_dict(dict_repr["atoms"])
        bonds = BondSet.from_dict(dict_repr["bonds"])

        confs = dict_repr.get("confs")
        confs = ConfSet.from_dict(confs) if confs is not None else None

        meta = dict_repr.get("meta")

        mol = GraphMol(atoms, bonds, confs=confs, meta=meta)
        return mol

    @staticmethod
    def from_bytes(data: bytes) -> GraphMol:
        obj = pickle.loads(data)
        return GraphMol._from_core_repr(obj)

    def to_rdkit(self, sanitise: bool = False) -> Chem.rdchem.Mol:
        rdkit_mol = mol_from_atoms(
            self.atomics,
            self.bonds.bonds,
            coords=self.coords,
            charges=self.charges,
            chirality=self.atoms.chirality,
            sanitise=sanitise,
        )

        # Set conformer weights within the mol if they exist
        if self.confs is not None and self.confs.weights is not None:
            assert rdkit_mol.GetNumConformers() == self.n_conformers
            for idx, conf in enumerate(rdkit_mol.GetConformers()):
                conf.SetProp("weight", str(self.confs.weights[idx].item()))

        return rdkit_mol

    def _to_core_repr(self) -> dict[str, dict[str, TArr]]:
        """A representation of the molecule using only built-in types and numpy arrays."""

        dict_repr = {"atoms": self.atoms.to_dict(), "bonds": self.bonds.to_dict()}

        if self.confs is not None:
            dict_repr["confs"] = self.confs.to_dict()

        if self.meta is not None:
            # Materialise: meta may be a _ColumnMetaView holding h5py handles
            # that aren't picklable.
            dict_repr["meta"] = dict(self.meta)

        return dict_repr

    def to_bytes(self) -> bytes:
        dict_repr = self._to_core_repr()
        byte_obj = pickle.dumps(dict_repr, protocol=PICKLE_PROTOCOL)
        return byte_obj


# ***************************************************************
# ***************** Batched Representations *********************
# ***************************************************************


class GraphBatch(Sequence):
    """A collection of GraphMol objects with batched property access and HDF5 persistence.

    Provides padded batch views of atomic properties, coordinates, and bonds. Supports
    sharded HDF5 save/load for large datasets.
    """

    def __init__(self, mols: list[GraphMol], hdf5_file: h5py.File | list[h5py.File] | None = None):
        for mol in mols:
            check_type(mol, GraphMol, "molecule object")

        open_fps = []
        if hdf5_file is not None:
            if isinstance(hdf5_file, h5py.File):
                open_fps = [hdf5_file]
            elif isinstance(hdf5_file, list):
                check_type(hdf5_file[0], h5py.File, "hdf5 file list item")
                open_fps = hdf5_file
            else:
                raise TypeError("hdf5_file must be either an h5py.File or a list of h5py.File objects.")

        self._mols = mols
        self._open_fps = open_fps

    # *** Publicly exposed properties ***

    @property
    def lengths(self) -> list[int]:
        return [len(mol) for mol in self._mols]

    @property
    def mask(self) -> TArr:
        return pad_arrays([np.ones(mol.seq_length) for mol in self._mols])

    @property
    def atomics(self) -> TArr:
        return pad_arrays([mol.atomics for mol in self._mols])

    @property
    def charges(self) -> TArr:
        return pad_arrays([mol.charges for mol in self._mols])

    @property
    def coords(self) -> TArr | None:
        n_confs = [mol.n_conformers for mol in self._mols]
        if any([n in [None, 0] for n in n_confs]) or any([n != n_confs[0] for n in n_confs]):
            raise RuntimeError("All mols in the batch must have the same number of conformers.")

        # Transpose conf and atom dims for padding, then transpose back
        coords = [mol.coords.transpose((1, 0, 2)) for mol in self._mols]
        padded = pad_arrays(coords).transpose((0, 2, 1, 3))
        return padded

    @property
    def bonds(self) -> TArr:
        return pad_arrays([mol.bonds for mol in self._mols])

    @property
    def bond_indices(self) -> TArr:
        return pad_arrays([mol.bond_indices for mol in self._mols])

    @property
    def bond_types(self) -> TArr:
        return pad_arrays([mol.bond_types for mol in self._mols])

    @property
    def adjacency(self) -> TArr:
        max_length = max(self.lengths)
        adjs = [mol.bonds.adj_matrix(max_length) for mol in self._mols]
        return np.stack(adjs, axis=0)

    def neighbour_ranks(self, stereo_only: bool = False) -> TArr:
        """[B, max_atoms, max_atoms] neighbour rank matrix, padded per batch.

        See GraphMol.neighbour_ranks / BondSet.neighbour_ranks for the per-mol shape.
        """

        max_length = max(self.lengths)

        ranks = []
        for mol in self._mols:
            if stereo_only:
                chiral_mask = np.zeros(max_length, dtype=bool)
                chiral_mask[: len(mol)] = mol.atoms.chirality != 0
            else:
                chiral_mask = None

            ranks.append(mol.bonds.neighbour_ranks(max_length, chiral_mask=chiral_mask))

        return np.stack(ranks, axis=0)

    # *** Basic indexing and utility functions ***

    def __len__(self) -> int:
        return len(self._mols)

    def __getitem__(self, index: int) -> GraphMol:
        return self._mols[index]

    def subset(self, idxs: list[int]) -> GraphBatch:
        subset_mols = [self._mols[idx] for idx in idxs]
        batch = GraphBatch(subset_mols, self._open_fps)
        return batch

    def meta_column(self, key: str) -> TArr:
        """Return values of a meta key across all mols in the batch as a string ndarray.

        For eager batches this just reads from the cached per-mol dicts; for lazy batches
        (see `LazyGraphBatch`) it reads the HDF5 column dataset directly when the shard
        was saved with columnar meta, avoiding any GraphMol construction.
        """

        return np.array([mol.meta.get(key, "") for mol in self._mols])

    # *** IO and conversion utility functions ***

    @staticmethod
    def _from_core_repr(obj: list[dict[str, dict[str, TArr]]]) -> GraphBatch:
        mols = [GraphMol._from_core_repr(mol_data) for mol_data in obj]
        return GraphBatch(mols)

    @staticmethod
    def from_bytes(data: bytes) -> GraphBatch:
        obj = pickle.loads(data)
        return GraphBatch._from_core_repr(obj)

    @staticmethod
    def from_smiles(
        smiles: Sequence[str], canonicalise: bool = False, clean_stereo: bool = False, skip_invalid: bool = False
    ) -> GraphBatch:
        """Create a batch from an iterable of SMILES strings.

        Args:
            smiles: SMILES strings.
            canonicalise: See `GraphMol.from_rdkit`.
            clean_stereo: See `GraphMol.from_rdkit`.
            skip_invalid: If True, silently drop SMILES that cannot be parsed. Default False
                raises on the first failure, naming the offending string.
        """

        mols = []
        for smi in smiles:
            try:
                mols.append(GraphMol.from_smiles(smi, canonicalise=canonicalise, clean_stereo=clean_stereo))
            except ValueError:
                if not skip_invalid:
                    raise

        return GraphBatch(mols)

    @staticmethod
    def from_sdf(
        sdf_path: str | Path,
        canonicalise: bool = False,
        clean_stereo: bool = False,
        remove_hs: bool = False,
        skip_invalid: bool = False,
        read_props: bool = True,
    ) -> GraphBatch:
        """Create a batch from an SDF file, preserving 3D coordinates where present.

        Args:
            sdf_path: Path to the .sdf file.
            canonicalise: See `GraphMol.from_rdkit`. Useful here, since SDF atom order is
                whatever the writing tool chose.
            clean_stereo: See `GraphMol.from_rdkit`.
            remove_hs: If True, strip explicit hydrogens on read. Default False keeps them,
                since an SDF with 3D coordinates usually has them for a reason.
            skip_invalid: If True, drop records RDKit cannot parse. Default False raises,
                naming the record index.
            read_props: If True, copy each record's SDF tags into `mol.meta`. Numeric tags
                keep their type when saved with `columnar_meta=True`.

        Note:
            Records carrying 2D depiction coordinates rather than real 3D geometry (common in
            SDFs exported from databases) are read as graphs with no conformers. A flat
            depiction is not a conformer, and storing it as one would produce a collapsed
            structure that looks valid. Check `mol.n_conformers` if you are unsure what a
            file contained.
        """

        sdf_path = Path(sdf_path)

        if not sdf_path.is_file():
            raise FileNotFoundError(f"No SDF file at {sdf_path}")

        mols = []
        supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=remove_hs)

        for idx, rdkit_mol in enumerate(supplier):
            if rdkit_mol is None:
                if skip_invalid:
                    continue

                raise ValueError(f"RDKit could not parse record {idx} of {sdf_path}")

            # A 2D depiction is not a conformer -- see the note in the docstring.
            if rdkit_mol.GetNumConformers() > 0 and not rdkit_mol.GetConformer().Is3D():
                rdkit_mol = Chem.Mol(rdkit_mol)
                rdkit_mol.RemoveAllConformers()

            mol = GraphMol.from_rdkit(rdkit_mol, canonicalise=canonicalise, clean_stereo=clean_stereo)

            if read_props:
                mol.meta = rdkit_mol.GetPropsAsDict()

            mols.append(mol)

        return GraphBatch(mols)

    def to_sdf(self, sdf_path: str | Path, write_meta: bool = True) -> None:
        """Write the batch to an SDF file, one record per molecule.

        Molecules without conformers are written with no coordinates, which is valid SDF but
        of limited use -- generate conformers first if you need geometry.

        Args:
            sdf_path: Output path. Overwritten if it exists.
            write_meta: If True, write each molecule's meta entries as SDF tags.
        """

        sdf_path = Path(sdf_path)
        writer = Chem.SDWriter(str(sdf_path))

        try:
            for mol in self._mols:
                rdkit_mol = mol.to_rdkit()

                if write_meta and mol.meta is not None:
                    for key, value in dict(mol.meta).items():
                        rdkit_mol.SetProp(str(key), str(value))

                writer.write(rdkit_mol)
        finally:
            writer.close()

    @staticmethod
    def from_batches(batches: list[GraphBatch]) -> GraphBatch:
        """Accumulate a list of GraphBatch objects into one batch."""

        mols = [mol for batch in batches for mol in batch]
        open_fps = [fp for batch in batches for fp in batch._open_fps]
        batch = GraphBatch(mols, hdf5_file=open_fps)
        return batch

    @staticmethod
    def load(
        save_path: str | Path,
        n_shards: int | None = None,
        materialise: bool = True,
        allow_pickle: bool = False,
    ) -> GraphBatch:
        """Load a sharded directory of HDF5 files into a batch.

        Note what `materialise` does *not* control: the underlying arrays (coords, atomics,
        bonds) are read from HDF5 on property access either way, via `LazyData`. Nothing
        reads bulk array data at load time regardless of this flag, and `mol.meta` is handled
        identically in both modes.

        Args:
            save_path: Directory produced by `GraphBatch.save`.
            n_shards: If set, only the first `n_shards` shards are loaded.
            materialise: If True (default), build every GraphMol -- and its AtomSet, BondSet
                  and ConfSet -- up front. If False, return a `LazyGraphBatch` that constructs
                  those wrapper objects on demand instead. Only the Python object construction
                  is deferred, which is what makes the difference on large datasets: at 2M
                  molecules, materialising costs roughly 14s and 1.9GB against 0.7s and 0.6GB.
                  Below ~100k the difference is not worth thinking about.

                  With materialise=False each `batch[i]` call returns a fresh GraphMol wrapper
                  (so `batch[i] is batch[i]` is False), meaning you cannot hold a reference and
                  expect reassignments like `mol.atoms = ...` to persist across lookups. For all
                  loaded batches, `mol.meta` is a read-only Mapping view -- see
                  `molito.core.meta` for details.
            allow_pickle: Permit loading a shard whose metadata is in the legacy pickled-blob
                format. Off by default because unpickling a file from an untrusted source can
                execute arbitrary code. Nothing molito writes now contains pickle.
        """

        save_path = Path(save_path)

        if not (save_path.exists() and save_path.is_dir()):
            raise RuntimeError(f"The folder was not found at path {save_path!s}")

        shard_paths = [path for path in save_path.iterdir() if path.suffix == ".hdf5"]
        sorted_paths = list(sorted(shard_paths, key=lambda p: int(p.stem)))

        if n_shards is not None:
            sorted_paths = sorted_paths[:n_shards]

        if not materialise:
            files = [h5py.File(p, "r") for p in sorted_paths]
            return LazyGraphBatch(files, allow_pickle=allow_pickle)

        shards = [GraphBatch.load_hdf5_shard(p, allow_pickle=allow_pickle) for p in sorted_paths]
        batch = GraphBatch.from_batches(shards)
        return batch

    @staticmethod
    def load_hdf5_shard(save_file: str | Path, allow_pickle: bool = False) -> GraphBatch:
        save_file = Path(save_file)

        if save_file.suffix != ".hdf5":
            raise RuntimeError("Save file must have an hdf5 suffix.")

        hdf5_file = h5py.File(save_file, "r")
        check_format(hdf5_file, save_file)
        mols = GraphBatch._load_from_group(hdf5_file, allow_pickle=allow_pickle)
        return GraphBatch(mols, hdf5_file=hdf5_file)

    def _to_core_repr(self) -> list[dict[str, dict[str, TArr]]]:
        dict_list = [mol._to_core_repr() for mol in self._mols]
        return dict_list

    def to_bytes(self) -> bytes:
        dict_repr = self._to_core_repr()
        byte_obj = pickle.dumps(dict_repr, protocol=PICKLE_PROTOCOL)
        return byte_obj

    def save(self, save_path: str | Path, shard_size: int | None = None, columnar_meta: bool = False) -> None:
        """Save the batch to a directory of HDF5 shards.

        Args:
            save_path: Output directory. Must be empty or non-existing.
            shard_size: Number of mols per shard. Default: all mols in one shard.
            columnar_meta: If True, store meta as one gzip-compressed HDF5 dataset per key (faster filter-scan,
                much smaller on disk, memory proportional to accessed columns). Requires metas to share a set of keys.
                Missing keys are filled with empty strings.
                If False (default), meta is stored as one gzip-compressed JSON document per
                shard, which handles nested or ragged metadata that columnar would stringify.
        """

        save_path = Path(save_path)

        if save_path.exists():
            if not (save_path.is_dir() and len(list(save_path.iterdir())) == 0):
                raise RuntimeError("Save path must point to an empty or non-existing directory.")

        save_path.mkdir(exist_ok=True, parents=True)

        shard_size = len(self) if shard_size is None else shard_size
        mol_shards = [[mol for mol in mols if mol is not None] for mols in grouper(self, shard_size)]

        for idx, shard in enumerate(mol_shards):
            shard_batch = GraphBatch(shard)
            save_file = save_path / f"{idx}.hdf5"
            shard_batch.save_hdf5_shard(save_file, columnar_meta=columnar_meta)

    def save_hdf5_shard(self, save_file: str | Path, columnar_meta: bool = False) -> None:
        hdf5_path = Path(save_file)

        if save_file.exists():
            raise RuntimeError(f"File {save_file!s} already exists.")

        if hdf5_path.suffix != ".hdf5":
            raise ValueError(f"save_file must end in .hdf5, got {save_file}")

        with h5py.File(hdf5_path, "x") as f:
            stamp_format(f)
            self._save_to_group(f, columnar_meta=columnar_meta)

    @staticmethod
    def _load_from_group(group: h5py.Group, allow_pickle: bool = False) -> list[GraphMol]:
        """Load molecules from an HDF5 group using the fast-path factories.

        Iterates AtomSet / BondSet / ConfSet slices in lockstep with the meta list and
        constructs GraphMols via `_load_unchecked` (no re-validation). Shard-level size
        agreement between groups is checked upfront before any iteration.
        """

        n = len(group["atoms"]["sizes"])
        if len(group["bonds"]["sizes"]) != n:
            raise RuntimeError(f"atom/bond shard size mismatch: {n} vs {len(group['bonds']['sizes'])}")

        has_confs = "confs" in group
        if has_confs and len(group["confs"]["sizes"]) != n:
            raise RuntimeError(f"atom/conf shard size mismatch: {n} vs {len(group['confs']['sizes'])}")

        atom_iter = AtomSet._iter_from_group(group["atoms"])
        bond_iter = BondSet._iter_from_group(group["bonds"])
        conf_iter = ConfSet._iter_from_group(group["confs"]) if has_confs else iter([None] * n)
        metas = load_meta(group["meta"], n, allow_pickle=allow_pickle)

        zipped = zip(atom_iter, bond_iter, conf_iter, metas, strict=True)
        mols = [GraphMol._load_unchecked(atoms, bonds, confs, meta) for atoms, bonds, confs, meta in zipped]
        return mols

    def _save_to_group(self, group: h5py.Group, columnar_meta: bool = False) -> None:
        """Save molecule data to an HDF5 group."""

        atoms = [mol.atoms for mol in self._mols]
        bonds = [mol.bonds for mol in self._mols]
        confs = [mol.confs for mol in self._mols if mol.confs is not None]

        if 0 < len(confs) < len(self._mols):
            raise RuntimeError(
                "Mixed conformer presence: all mols in a shard must have confs or none. "
                f"Got {len(confs)} with confs out of {len(self._mols)} total."
            )

        atom_arrays = AtomSet.arrays_from_atoms(atoms)
        bond_arrays = BondSet.arrays_from_bonds(bonds)
        conf_arrays = ConfSet.arrays_from_confs(confs) if len(confs) > 0 else None

        atom_group = group.create_group("atoms")
        for name, arr in atom_arrays.items():
            atom_group.create_dataset(name, data=arr)

        bond_group = group.create_group("bonds")
        for name, arr in bond_arrays.items():
            bond_group.create_dataset(name, data=arr)

        if conf_arrays is not None:
            conf_group = group.create_group("confs")
            for name, arr in conf_arrays.items():
                conf_group.create_dataset(name, data=arr)

        # Materialise each meta in case any are live HDF5-backed views — both save formats
        # serialise the list (blob via pickle, columnar via .get() on each entry).
        metas = [dict(mol.meta) if mol.meta is not None else {} for mol in self._mols]
        save_meta(group, metas, columnar=columnar_meta)

    def close_hdf5(self) -> None:
        if self._open_fps is not None:
            for fp in self._open_fps:
                fp.close() if fp is not None else None


# ***************************************************************
# *************** Lazy (on-demand) Graph Batch ******************
# ***************************************************************


class _LazyMolList:
    """Sequence stand-in for GraphBatch._mols that builds GraphMols on demand.

    Exposes a list-like API (length, indexing, iteration) without storing materialised GraphMols.
    Each access constructs a fresh GraphMol from the underlying HDF5 data.
    """

    __slots__ = ("_batch",)

    def __init__(self, batch: LazyGraphBatch):
        self._batch = batch

    def __len__(self) -> int:
        return self._batch._total_mols

    def __getitem__(self, i: int) -> GraphMol:
        return self._batch._build_mol(i)

    def __iter__(self):
        for i in range(self._batch._total_mols):
            yield self._batch._build_mol(i)


class LazyGraphBatch(GraphBatch):
    """A GraphBatch that constructs GraphMols on demand from HDF5 shards.

    Returned by `GraphBatch.load(..., materialise=False)`. Drop-in compatible with GraphBatch — same public API,
    same property semantics — but no mols are constructed at load time. Handles one or many shards transparently
    via an internal boundary lookup.

    What this saves is Python object construction, not data reading: array access goes through `LazyData` in both
    modes, so nothing reads bulk coords or atomics at load time either way. The saving is therefore proportional to
    the number of molecules rather than their size — around 14s and 1.9GB at 2M mols, against 0.7s and 0.6GB here.
    Below ~100k mols it isn't worth reaching for. Also suits filter-scan workflows, which can use `meta_column` to
    avoid mol construction entirely.

    Semantics vs a materialised GraphBatch:
        - Each `batch[i]` call builds a fresh GraphMol wrapper, so identity isn't preserved
            (`batch[i] is batch[i]` is False). The underlying data (atomics, coords, bonds)
            still round-trips correctly because it's re-read from HDF5.
        - `mol.meta` is a read-only Mapping view on loaded mols -- same policy as other HDF5-backed attributes, which
            return fresh copies on access. To mutate, call `dict(mol.meta)` for a mutable copy,
            or reassign `mol.meta = {...}`.
        - `subset(idxs)` returns an eager GraphBatch with exactly the selected mols materialised --
            convenient for training batches.
    """

    def __init__(self, hdf5_files: list[h5py.File], allow_pickle: bool = False):
        self._open_fps = hdf5_files
        self._shards = [self._build_shard_state(f, allow_pickle=allow_pickle) for f in hdf5_files]

        shard_n = np.array([s["n"] for s in self._shards], dtype=np.int64)
        self._shard_boundaries = np.concatenate([[0], np.cumsum(shard_n)])
        self._total_mols = int(self._shard_boundaries[-1])

        self._mols = _LazyMolList(self)

    @staticmethod
    def _build_shard_state(f: h5py.File, allow_pickle: bool = False) -> dict:
        """Precompute per-shard dataset refs, sizes, and cumulative offsets."""

        check_format(f, f.filename)

        atom_g = f["atoms"]
        atom_sizes = np.asarray(atom_g["sizes"][()], dtype=np.int64)
        atom_offsets = np.concatenate([[0], np.cumsum(atom_sizes[:-1])])

        bond_g = f["bonds"]
        bond_sizes = np.asarray(bond_g["sizes"][()], dtype=np.int64)
        bond_offsets = np.concatenate([[0], np.cumsum(bond_sizes[:-1])])

        n = len(atom_sizes)
        if len(bond_sizes) != n:
            raise RuntimeError(f"atom/bond shard size mismatch: {n} vs {len(bond_sizes)}")

        conf_state = None
        if "confs" in f:
            conf_g = f["confs"]
            conf_sizes = np.asarray(conf_g["sizes"][()], dtype=np.int64)

            if len(conf_sizes) != n:
                raise RuntimeError(f"atom/conf shard size mismatch: {n} vs {len(conf_sizes)}")

            coord_counts = conf_sizes[:, 0] * conf_sizes[:, 1]
            weight_counts = conf_sizes[:, 2]

            conf_state = {
                "coords_ds": conf_g["coords"],
                "weights_ds": conf_g["weights"],
                "sizes": conf_sizes,
                "coord_offsets": np.concatenate([[0], np.cumsum(coord_counts[:-1])]),
                "weight_offsets": np.concatenate([[0], np.cumsum(weight_counts[:-1])]),
            }

        shard_state = {
            "n": n,
            "meta_group": f["meta"],
            "metas": load_meta(f["meta"], n, allow_pickle=allow_pickle),
            "atoms": {
                "atomics": atom_g["atomics"],
                "charges": atom_g["charges"],
                "chirality": atom_g.get("chirality"),
                "res_names": atom_g.get("res_names"),
                "atom_names": atom_g.get("atom_names"),
                "res_ids": atom_g.get("res_ids"),
                "chain_ids": atom_g.get("chain_ids"),
                "sizes": atom_sizes,
                "offsets": atom_offsets,
            },
            "bonds": {"bonds": bond_g["bonds"], "sizes": bond_sizes, "offsets": bond_offsets},
            "conf": conf_state,
        }
        return shard_state

    def _locate(self, i: int) -> tuple[int, int]:
        """Map a global mol index to (shard_idx, local_idx)."""

        if i < 0 or i >= self._total_mols:
            raise IndexError(f"Index {i} out of range for batch of size {self._total_mols}")

        s = int(np.searchsorted(self._shard_boundaries, i, side="right") - 1)
        return s, i - int(self._shard_boundaries[s])

    def _build_mol(self, i: int) -> GraphMol:
        s, local = self._locate(i)
        shard = self._shards[s]

        atoms = self._build_atoms(shard["atoms"], local)
        bonds = self._build_bonds(shard["bonds"], local)
        confs = self._build_confs(shard["conf"], local) if shard["conf"] is not None else None

        return GraphMol._load_unchecked(atoms, bonds, confs, shard["metas"][local])

    @staticmethod
    def _build_atoms(state: dict, local: int) -> AtomSet:
        off = int(state["offsets"][local])
        n = int(state["sizes"][local])

        chir_ds = state["chirality"]
        rn_ds = state["res_names"]
        an_ds = state["atom_names"]
        ri_ds = state["res_ids"]
        ci_ds = state["chain_ids"]

        atoms = AtomSet._load_unchecked(
            LazyData._load_unchecked(state["atomics"], off, n),
            LazyData._load_unchecked(state["charges"], off, n),
            LazyData._load_unchecked(chir_ds, off, n) if chir_ds is not None else None,
            res_names=LazyData._load_unchecked(rn_ds, off, n) if rn_ds is not None else None,
            atom_names=LazyData._load_unchecked(an_ds, off, n) if an_ds is not None else None,
            res_ids=LazyData._load_unchecked(ri_ds, off, n) if ri_ds is not None else None,
            chain_ids=LazyData._load_unchecked(ci_ds, off, n) if ci_ds is not None else None,
        )
        return atoms

    @staticmethod
    def _build_bonds(state: dict, local: int) -> BondSet:
        off = int(state["offsets"][local])
        n = int(state["sizes"][local])
        return BondSet._load_unchecked(LazyData._load_unchecked(state["bonds"], off, n))

    @staticmethod
    def _build_confs(state: dict, local: int) -> ConfSet:
        n_confs, n_atoms, n_w = state["sizes"][local].tolist()
        co = int(state["coord_offsets"][local])
        wo = int(state["weight_offsets"][local])

        coords = LazyData._load_unchecked(state["coords_ds"], co, (n_confs, n_atoms))
        weights = LazyData._load_unchecked(state["weights_ds"], wo, n_w) if n_w > 0 else None
        return ConfSet._load_unchecked(coords, weights)

    def meta_column(self, key: str) -> TArr:
        """Read one meta key across all shards as a concatenated string ndarray.

        Fast path: for columnar-format shards this reads a single HDF5 dataset per shard
        (no GraphMol construction). For blob-format shards it falls back to pulling the
        key out of each cached meta dict. Missing keys come back as empty strings.
        """

        cols = []
        for shard in self._shards:
            try:
                cols.append(column_array(shard["meta_group"], key))
            except KeyError:
                cols.append(np.array([m.get(key, "") for m in shard["metas"]]))

        return np.concatenate(cols)
