from __future__ import annotations

import copy
import pickle
from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np
from more_itertools import grouper

from molito.arrays import adj_from_edges, pad_arrays
from molito.core._checks import PICKLE_PROTOCOL, check_dict_key, check_type
from molito.core.format import check_format, stamp_format
from molito.core.meta import load_meta, save_meta
from molito.mol.graph import GraphBatch, GraphMol
from molito.mol.protein import Protein, ProteinBatch, _pad_string_lists

TArr = np.ndarray


class BindingComplex:
    """A protein-ligand binding complex with optional interaction data."""

    __slots__ = ("protein", "ligand", "interactions", "meta")

    def __init__(
        self, protein: Protein, ligand: GraphMol, interactions: InteractionSet | None = None, meta: dict | None = None
    ):
        meta = {} if meta is None else meta

        check_type(protein, Protein, name="protein")
        check_type(ligand, GraphMol, name="ligand")

        if interactions is not None:
            from molito.mol.interactions import InteractionSet

            check_type(interactions, InteractionSet, name="interactions")
            self._check_interactions(interactions, protein, ligand)

        if meta is not None:
            check_type(meta, dict, name="meta")

        self.protein = protein
        self.ligand = ligand
        self.interactions = interactions
        self.meta = meta

    @classmethod
    def _load_unchecked(cls, protein: Protein, ligand: GraphMol, interactions, meta) -> BindingComplex:
        """Fast loading which skips validation checks."""

        obj = cls.__new__(cls)
        obj.protein = protein
        obj.ligand = ligand
        obj.interactions = interactions
        obj.meta = meta if meta is not None else {}
        return obj

    # *** Properties ***

    def __len__(self) -> int:
        return self.seq_length

    @property
    def seq_length(self) -> int:
        return len(self.ligand) + len(self.protein)

    @property
    def system_id(self) -> str | None:
        return self.meta.get("system_id")

    @property
    def coords(self) -> TArr:
        """Returns concatenation of ligand and protein coords. Shape [n_complex_atoms, 3]"""

        lig_coords = self.ligand.coords

        # Handle 3D ligand coords (squeeze first conformer)
        if len(lig_coords.shape) == 3:
            if lig_coords.shape[0] != 1:
                raise ValueError("Ligands with multiple conformers are not supported in complexes.")
            lig_coords = lig_coords[0]

        return np.concatenate((lig_coords, self.protein.coords), axis=0)

    @property
    def atomics(self) -> TArr:
        """Returns concatenation of ligand and protein atomics."""
        return np.concatenate((self.ligand.atomics, self.protein.atomics), axis=0)

    @property
    def charges(self) -> TArr:
        """Returns concatenation of ligand and protein charges."""
        return np.concatenate((self.ligand.charges, self.protein.charges), axis=0)

    @property
    def res_names(self) -> TArr:
        """Returns concatenation of protein and ligand residue names ('LIG' for all ligand atoms)."""

        ligand_res_names = ["LIG"] * len(self.ligand)
        complex_res_names = ligand_res_names + self.protein.res_names
        return np.array(complex_res_names)

    @property
    def bonds(self) -> TArr:
        """Returns concatenation of ligand and protein bonds with indices adjusted."""

        # Shift the protein bond indices to account for ligand atoms coming first
        protein_bonds = self.protein.bonds.bonds.copy()
        protein_bonds[:, :2] += len(self.ligand)

        return np.concatenate((self.ligand.bonds.bonds, protein_bonds), axis=0)

    @property
    def bond_indices(self) -> TArr:
        return self.bonds[:, :2]

    @property
    def bond_types(self) -> TArr:
        return self.bonds[:, 2]

    @property
    def adjacency(self) -> TArr:
        adj = adj_from_edges(
            self.bond_indices.astype(np.int64), self.bond_types.astype(np.int64), self.seq_length, symmetric=True
        )
        return adj

    @property
    def ligand_mask(self) -> TArr:
        """Returns a 1D mask, 1 for ligand atoms, 0 for protein atoms."""
        mask = np.array([1] * len(self.ligand) + [0] * len(self.protein))
        return mask

    @property
    def com(self) -> TArr:
        """Centre of mass of the complex."""
        return self.coords.mean(axis=0)

    # *** IO and conversion functions ***

    def to_bytes(self) -> bytes:
        dict_repr = {
            "protein": self.protein.to_bytes(),
            "ligand": self.ligand.to_bytes(),
            # Materialise: meta may be a _ColumnMetaView holding h5py handles
            # that aren't picklable.
            "meta": dict(self.meta) if self.meta is not None else None,
        }

        if self.interactions is not None:
            dict_repr["interactions"] = self.interactions.to_bytes()

        byte_obj = pickle.dumps(dict_repr, protocol=PICKLE_PROTOCOL)
        return byte_obj

    @staticmethod
    def from_bytes(data: bytes) -> BindingComplex:
        from molito.mol.interactions import InteractionSet

        obj = pickle.loads(data)

        check_dict_key(obj, "protein")
        check_dict_key(obj, "ligand")
        check_dict_key(obj, "meta")

        protein = Protein.from_bytes(obj["protein"])
        ligand = GraphMol.from_bytes(obj["ligand"])

        interactions = None
        if obj.get("interactions") is not None:
            interactions = InteractionSet.from_bytes(obj["interactions"])

        system = BindingComplex(protein, ligand, interactions=interactions, meta=obj["meta"])
        return system

    # *** Subset and transformation functions ***

    def remove_hs(self, include_ligand: bool = True) -> BindingComplex:
        """Remove hydrogen atoms from the complex.

        Args:
            include_ligand: If True, also remove Hs from ligand. Default True.

        Returns:
            New BindingComplex with hydrogens removed.
        """

        protein_atom_mask = self.protein.atomics != 1
        protein_non_h_idxs = np.arange(len(self.protein))[protein_atom_mask]
        protein_subset = self.protein.permute(protein_non_h_idxs)

        ligand_subset = self.ligand
        ligand_atom_mask = np.ones(len(self.ligand), dtype=bool)

        if include_ligand:
            ligand_atom_mask = self.ligand.atomics != 1
            ligand_subset = self.ligand.remove_hs()

        interactions_subset = None
        if self.interactions is not None:
            interactions_subset = self.interactions.subset_atoms(protein_atom_mask, ligand_atom_mask)

        subset = BindingComplex(
            protein_subset,
            ligand_subset,
            interactions=interactions_subset,
            meta=copy.deepcopy(dict(self.meta)) if self.meta is not None else None,
        )
        return subset

    def zero_com(self) -> BindingComplex:
        """Returns a copy with centre of mass at origin."""

        system_shift = -self.com
        shifted_ligand = self.ligand.shift(system_shift)
        shifted_protein = self.protein.shift(system_shift)
        return self.copy_with(protein=shifted_protein, ligand=shifted_ligand)

    def copy_with(
        self, protein: Protein | None = None, ligand: GraphMol | None = None, interactions: InteractionSet | None = None
    ) -> BindingComplex:

        protein_copy = self.protein.copy() if protein is None else protein
        ligand_copy = self.ligand.copy() if ligand is None else ligand

        if interactions is None:
            interactions_copy = self.interactions.copy() if self.interactions is not None else None
        else:
            interactions_copy = interactions

        # Materialise first: meta may be a live HDF5-backed _ColumnMetaView which holds
        # h5py.Dataset handles that can't be pickled/deepcopied.
        meta_copy = copy.deepcopy(dict(self.meta)) if self.meta is not None else None

        complex = BindingComplex(protein_copy, ligand_copy, interactions=interactions_copy, meta=meta_copy)
        return complex

    def copy(self) -> BindingComplex:
        return self.copy_with()

    # *** Validation ***

    @staticmethod
    def _check_interactions(interactions, protein, ligand):
        if interactions.n_protein_atoms != len(protein):
            err = "InteractionSet n_protein_atoms must match protein length, got "
            raise ValueError(f"{err}{interactions.n_protein_atoms} and {len(protein)}")

        if interactions.n_ligand_atoms != len(ligand):
            err = "InteractionSet n_ligand_atoms must match ligand length, got "
            raise ValueError(f"{err}{interactions.n_ligand_atoms} and {len(ligand)}")


# *****************************************************************************
# ************************ Batched Complex Representation *********************
# *****************************************************************************


class ComplexBatch(Sequence):
    """Utility class for loading, saving and batching BindingComplex objects."""

    def __init__(self, complexes: list[BindingComplex], hdf5_file: h5py.File | list[h5py.File] | None = None):
        for cx in complexes:
            check_type(cx, BindingComplex, "complex object")

        open_fps = []
        if hdf5_file is not None:
            if isinstance(hdf5_file, h5py.File):
                open_fps = [hdf5_file]
            elif isinstance(hdf5_file, list):
                if len(hdf5_file) > 0:
                    check_type(hdf5_file[0], h5py.File, "hdf5 file list item")
                open_fps = hdf5_file
            else:
                raise TypeError("hdf5_file must be either an h5py.File or a list of h5py.File objects.")

        self._complexes = complexes
        self._open_fps = open_fps

    # *** Publicly exposed properties ***

    @property
    def lengths(self) -> list[int]:
        return [len(cx) for cx in self._complexes]

    @property
    def protein_lengths(self) -> list[int]:
        return [len(cx.protein) for cx in self._complexes]

    @property
    def ligand_lengths(self) -> list[int]:
        return [len(cx.ligand) for cx in self._complexes]

    @property
    def mask(self) -> TArr:
        return pad_arrays([np.ones(cx.seq_length) for cx in self._complexes])

    @property
    def protein_mask(self) -> TArr:
        return pad_arrays([np.ones(len(cx.protein)) for cx in self._complexes])

    @property
    def ligand_mask(self) -> TArr:
        return pad_arrays([np.ones(len(cx.ligand)) for cx in self._complexes])

    @property
    def atomics(self) -> TArr:
        """[B, max_atoms] atomic numbers, ligand atoms first then protein, zero-padded."""
        return pad_arrays([cx.atomics for cx in self._complexes])

    @property
    def charges(self) -> TArr:
        """[B, max_atoms] formal charges, in the same order as `atomics`."""
        return pad_arrays([cx.charges for cx in self._complexes])

    @property
    def coords(self) -> TArr:
        """[B, max_atoms, 3] coordinates, in the same order as `atomics`.

        Raises RuntimeError if any ligand carries more than one conformer, since there
        would be no single geometry to batch.
        """

        return pad_arrays([cx.coords for cx in self._complexes])

    @property
    def res_names(self) -> TArr:
        """[B, max_atoms] residue names, with 'LIG' for every ligand atom."""
        return _pad_string_lists([cx.res_names.tolist() for cx in self._complexes])

    @property
    def bonds(self) -> TArr:
        """[B, max_bonds, 3] bond rows, with protein indices shifted past the ligand atoms."""
        return pad_arrays([cx.bonds for cx in self._complexes])

    @property
    def bond_indices(self) -> TArr:
        return pad_arrays([cx.bond_indices for cx in self._complexes])

    @property
    def bond_types(self) -> TArr:
        return pad_arrays([cx.bond_types for cx in self._complexes])

    @property
    def adjacency(self) -> TArr:
        """[B, max_atoms, max_atoms] symmetric adjacency over the combined complex."""

        max_length = max(self.lengths)
        adjs = [
            adj_from_edges(cx.bond_indices.astype(np.int64), cx.bond_types.astype(np.int64), max_length, symmetric=True)
            for cx in self._complexes
        ]
        return np.stack(adjs, axis=0)

    def meta_column(self, key: str) -> TArr:
        """Return values of a meta key across the batch as a string ndarray."""
        return np.array([cx.meta.get(key, "") for cx in self._complexes])

    # *** Basic indexing and utility functions ***

    def __len__(self) -> int:
        return len(self._complexes)

    def __getitem__(self, index: int) -> BindingComplex:
        return self._complexes[index]

    def subset(self, idxs: list[int]) -> ComplexBatch:
        subset_complexes = [self._complexes[idx] for idx in idxs]
        batch = ComplexBatch(subset_complexes, self._open_fps)
        return batch

    # *** IO and conversion utility functions ***

    def to_bytes(self) -> bytes:
        byte_list = [cx.to_bytes() for cx in self._complexes]
        return pickle.dumps(byte_list, protocol=PICKLE_PROTOCOL)

    @staticmethod
    def from_bytes(data: bytes) -> ComplexBatch:
        byte_list = pickle.loads(data)
        return ComplexBatch([BindingComplex.from_bytes(b) for b in byte_list])

    @staticmethod
    def from_batches(batches: list[ComplexBatch]) -> ComplexBatch:
        """Accumulate a list of ComplexBatch objects into one batch."""

        complexes = [cx for batch in batches for cx in batch]
        open_fps = [fp for batch in batches for fp in batch._open_fps]
        batch = ComplexBatch(complexes, hdf5_file=open_fps)
        return batch

    @staticmethod
    def load(save_path: str | Path, n_shards: int | None = None) -> ComplexBatch:
        """Load data from a folder that was saved using the save function.

        If n_shards is provided, only the first <n_shards> shards will be loaded.
        """

        save_path = Path(save_path)

        if not (save_path.exists() and save_path.is_dir()):
            raise RuntimeError(f"The folder was not found at path {save_path!s}")

        shard_paths = [path for path in save_path.iterdir() if path.suffix == ".hdf5"]
        sorted_paths = list(sorted(shard_paths, key=lambda p: (0, int(p.stem)) if p.stem.isdigit() else (1, p.stem)))

        if n_shards is not None:
            n_shards = min(len(sorted_paths), n_shards)
            sorted_paths = sorted_paths[:n_shards]

        shards = [ComplexBatch.load_hdf5_shard(shard_path) for shard_path in sorted_paths]
        batch = ComplexBatch.from_batches(shards)
        return batch

    @staticmethod
    def load_hdf5_shard(save_file: str | Path) -> ComplexBatch:
        save_file = Path(save_file)

        if save_file.suffix != ".hdf5":
            raise RuntimeError("Save file must have an hdf5 suffix.")

        hdf5_file = h5py.File(save_file, "r")
        check_format(hdf5_file, save_file)

        proteins = ProteinBatch._load_from_group(hdf5_file["proteins"])
        ligands = GraphBatch._load_from_group(hdf5_file["ligands"])
        interaction_sets = InteractionSet.load_from_group(hdf5_file["interactions"])

        n = len(proteins)
        if len(ligands) != n or len(interaction_sets) != n:
            raise RuntimeError(
                f"Complex shard size mismatch: {n} proteins, {len(ligands)} ligands, "
                f"{len(interaction_sets)} interaction sets."
            )

        complexes = []
        complex_metas = load_meta(hdf5_file["meta"], n)

        for protein, ligand, ints, meta in zip(proteins, ligands, interaction_sets, complex_metas, strict=True):
            complexes.append(BindingComplex._load_unchecked(protein, ligand, ints, meta))

        return ComplexBatch(complexes, hdf5_file=hdf5_file)

    def save(self, save_path: str | Path, shard_size: int | None = None, columnar_meta: bool = False) -> None:
        """Save the batch of data under the directory given by save_path.

        Args:
            save_path: Output directory. Must be empty or non-existing.
            shard_size: Number of complexes per shard. Default: all in one shard.
            columnar_meta: If True, store meta as one gzip-compressed HDF5 dataset per key (faster filter-scan,
                much smaller on disk, memory proportional to accessed columns). Requires metas to share a set of keys.
                Missing keys are filled with empty strings.
                If False (default), meta is stored as a single pickle blob per shard.
        """

        save_path = Path(save_path)

        if save_path.exists():
            if not (save_path.is_dir() and len(list(save_path.iterdir())) == 0):
                raise RuntimeError("Save path must point to an empty or non-existing directory.")

        save_path.mkdir(exist_ok=True, parents=True)

        shard_size = len(self) if shard_size is None else shard_size
        complex_shards = [[cx for cx in cxs if cx is not None] for cxs in grouper(self, shard_size)]

        for idx, shard in enumerate(complex_shards):
            shard_batch = ComplexBatch(shard)
            save_file = save_path / f"{idx}.hdf5"
            shard_batch.save_hdf5_shard(save_file, columnar_meta=columnar_meta)

    def save_hdf5_shard(self, save_file: str | Path, columnar_meta: bool = False) -> None:
        hdf5_path = Path(save_file)

        if hdf5_path.exists():
            raise RuntimeError(f"File {save_file!s} already exists.")

        if hdf5_path.suffix != ".hdf5":
            raise ValueError(f"save_file must end in .hdf5, got {save_file}")

        proteins = [cx.protein for cx in self._complexes]
        ligands = [cx.ligand for cx in self._complexes]
        interactions = [cx.interactions for cx in self._complexes]

        # Materialise each meta in case any are live HDF5-backed views.
        complex_metas = [dict(cx.meta) if cx.meta is not None else {} for cx in self._complexes]

        with h5py.File(hdf5_path, "x") as f:
            stamp_format(f)
            ProteinBatch(proteins)._save_to_group(f.create_group("proteins"), columnar_meta=columnar_meta)
            GraphBatch(ligands)._save_to_group(f.create_group("ligands"), columnar_meta=columnar_meta)
            InteractionSet.save_to_group(interactions, f.create_group("interactions"))
            save_meta(f, complex_metas, columnar=columnar_meta)

    def close_hdf5(self) -> None:
        """Closes any HDF5 files associated with this batch."""

        if self._open_fps is not None:
            for fp in self._open_fps:
                fp.close() if fp is not None else None


# Import at end to avoid circular import issues
from molito.mol.interactions import InteractionSet  # noqa: E402
