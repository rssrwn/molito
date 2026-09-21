import tempfile
import unittest
from pathlib import Path

import numpy as np
from biotite.structure import AtomArray, BondList
from rdkit import Chem
from rdkit.Chem import AllChem

from molito.arrays import adj_from_edges, pad_arrays
from molito.core import AtomSet, BondSet, ConfSet
from molito.geometry.align import align_best_conf
from molito.geometry.common import _dedup_conformers
from molito.mol import BindingComplex, ComplexBatch, GraphBatch, GraphMol, Protein, ProteinBatch


class TestConformerSelection(unittest.TestCase):
    def test_selection_retains_weights_and_copies_arrays(self):
        coords = np.arange(27).reshape(3, 3, 3)
        confs = ConfSet(coords, weights=np.array([0.2, 0.5, 0.3]))
        for index in [np.array([2, 0]), np.array([True, False, True]), np.array([1, 1])]:
            with self.subTest(index=index):
                selected = confs[index]
                np.testing.assert_array_equal(selected.coords, confs.coords[index])
                np.testing.assert_array_equal(selected.weights, confs.weights[index])
                selected.weights[:] = 1
                selected.coords[:] = -1
                np.testing.assert_array_equal(confs.coords, coords)
                np.testing.assert_allclose(confs.weights, [0.2, 0.5, 0.3])

        unweighted = ConfSet(coords)[np.array([2, 0])]
        self.assertIsNone(unweighted.weights)

    def test_topk_retains_selected_weights_without_renormalising(self):
        confs = ConfSet(np.arange(27).reshape(3, 3, 3), weights=np.array([0.2, 0.5, 0.3]))
        selected = confs.select_topk(np.int64(2))
        np.testing.assert_array_equal(selected.coords, confs.coords[[1, 2]])
        np.testing.assert_array_equal(selected.weights, confs.weights[[1, 2]])
        for k in [0, -1, 4, 1.5]:
            with self.subTest(k=k), self.assertRaises(ValueError):
                confs.select_topk(k)

    def test_small_positive_weights_remain_selectable(self):
        confs = ConfSet(np.zeros((2, 1, 3)), weights=np.array([1.0, 1e-8]))
        selected = confs[np.array([1])]
        np.testing.assert_array_equal(selected.weights, confs.weights[[1]])
        np.testing.assert_array_equal(selected.select_topk().weights, selected.weights)


class TestIntegerStorage(unittest.TestCase):
    def test_atom_fields_reject_overflow_and_truncation(self):
        for field, values in [
            ("atomics", [-1, 256, 6.5, np.nan, np.inf]),
            ("charges", [-129, 128, 0.5]),
            ("chirality", [-129, 128, 0.5]),
            ("res_ids", [-(2**31) - 1, 2**31, 1.5]),
        ]:
            for value in values:
                kwargs = {"atomics": np.array([6]), field: np.array([value])}
                with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, field):
                    AtomSet(**kwargs)

    def test_valid_storage_bounds_and_integral_floats(self):
        atoms = AtomSet(
            np.array([0.0, 255.0]),
            charges=np.array([-128, 127]),
            chirality=np.array([-128, 127]),
            res_ids=np.array([-(2**31), 2**31 - 1]),
        )
        for field, dtype in [("atomics", "uint8"), ("charges", "int8"), ("chirality", "int8"), ("res_ids", "int32")]:
            self.assertEqual(getattr(atoms, field).dtype, np.dtype(dtype))
        np.testing.assert_array_equal(atoms.atomics, [0, 255])
        self.assertEqual(BondSet(np.array([[0, 32767, 1]])).indices[0, 1], 32767)
        self.assertEqual(BondSet(np.empty((0, 3))).bonds.dtype, np.dtype("int16"))

    def test_float_storage_boundaries_are_not_rounded_up(self):
        with self.assertRaisesRegex(ValueError, "res_ids"):
            AtomSet(np.array([6]), res_ids=np.array([2**31], dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "bonds"):
            BondSet(np.array([[0, 32768, 1]], dtype=np.float16))
        with self.assertRaisesRegex(ValueError, "atomics"):
            AtomSet(np.array([2**64 - 1], dtype=np.uint64))

    def test_bonds_and_importers_validate_before_casting(self):
        for value in [32768, 40000, -32769, 1.5, np.inf, np.nan]:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "bonds"):
                BondSet(np.array([[0, value, 1]]))
        with self.assertRaisesRegex(ValueError, "bonds"):
            BondSet.from_biotite(BondList(40001, np.array([[0, 40000, 1]])))
        atoms = AtomArray(1)
        atoms.element = np.array(["C"])
        atoms.set_annotation("charge", np.array([128]))
        with self.assertRaisesRegex(ValueError, "charges"):
            AtomSet.from_biotite(atoms)

    def test_complex_bond_offsets_do_not_wrap(self):
        complex = TestBatchFileOwnership._batches()[-1][0]
        # Each component fits int16; their combined atom indices may not.
        complex.protein = complex.protein.copy_with(
            atoms=complex.protein.atoms.pad(32768),
            bonds=BondSet(np.array([[0, 32767, 1]])),
            confs=complex.protein.confs.pad(32768),
        )
        with self.assertRaisesRegex(ValueError, "bonds"):
            _ = complex.bonds


class TestArrayAndGeometryAudit(unittest.TestCase):
    def test_padding_preserves_values_across_dtypes(self):
        for arrays in [
            [np.array([1]), np.array([1.5])],
            [np.array(["C"]), np.array(["Cl"])],
            [np.array([1], dtype=np.int8), np.array([1000], dtype=np.int16)],
        ]:
            with self.subTest(arrays=arrays):
                padded = pad_arrays(arrays)
                for idx, arr in enumerate(arrays):
                    np.testing.assert_array_equal(padded[idx, : len(arr)], arr)

    def test_padding_rejects_broadcastable_shape_mismatch(self):
        for shapes in [[(2, 3), (1, 1)], [(2, 3), (1,)], [(), (1,)]]:
            with self.subTest(shapes=shapes), self.assertRaises(ValueError):
                pad_arrays([np.ones(shape) for shape in shapes])

    def test_adjacency_rejects_invalid_edges_and_broadcasting(self):
        for indices, types in [
            (np.array([[-1, 0]]), np.array([1])),
            (np.array([[0, 2]]), np.array([1])),
            (np.array([[0.5, 1]]), np.array([1])),
            (np.array([[0, 1], [1, 0]]), np.array([1])),
        ]:
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                adj_from_edges(indices, types, 2)

    def test_alignment_and_deduplication_accept_noncontiguous_ids(self):
        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        AllChem.EmbedMultipleConfs(mol, numConfs=2, randomSeed=42)
        reference = Chem.Mol(mol)
        expected = align_best_conf(mol, reference)
        expected_ids = _dedup_conformers(mol)
        ids = [7, 19]
        for conf, conf_id in zip(mol.GetConformers(), ids, strict=True):
            conf.SetId(conf_id)
        original_coords = [conf.GetPositions().copy() for conf in mol.GetConformers()]
        actual = align_best_conf(mol, reference)
        np.testing.assert_allclose(actual[1:], expected[1:])
        np.testing.assert_allclose(actual[0].GetConformer().GetPositions(), expected[0].GetConformer().GetPositions())
        self.assertEqual(_dedup_conformers(mol), [ids[idx] for idx in expected_ids])
        for conf, coords in zip(mol.GetConformers(), original_coords, strict=True):
            np.testing.assert_array_equal(conf.GetPositions(), coords)

    def test_alignment_rejects_missing_conformers_and_nan_weight(self):
        mol = Chem.MolFromSmiles("CCO")
        with self.assertRaisesRegex(ValueError, "conformer"):
            align_best_conf(mol, mol)
        with self.assertRaisesRegex(ValueError, "align_weight"):
            align_best_conf(mol, mol, align_weight=float("nan"))


class TestBatchFileOwnership(unittest.TestCase):
    @staticmethod
    def _batches():
        atoms = AtomSet(
            np.array([6, 7]),
            res_names=np.array(["ALA", "ALA"]),
            atom_names=np.array(["CA", "N"]),
            res_ids=np.array([1, 1]),
        )
        protein = Protein(atoms, BondSet(np.array([[0, 1, 1]])), ConfSet(np.zeros((1, 2, 3))))
        ligand = GraphMol.from_smiles("CO")
        ligand.confs = ConfSet(np.ones((1, 2, 3)), weights=np.array([1.0]))
        ligand.meta = {"label": 2}
        complex = BindingComplex(protein, ligand, meta={"system_id": "test"})
        return [GraphBatch([ligand, ligand]), ProteinBatch([protein, protein]), ComplexBatch([complex, complex])]

    def test_views_do_not_close_owners_and_read_detaches(self):
        for original in self._batches():
            modes = [True, False] if isinstance(original, GraphBatch) else [None]
            for materialise in modes:
                with self.subTest(batch=type(original), materialise=materialise), tempfile.TemporaryDirectory() as tmp:
                    original.save(tmp, shard_size=1, columnar_meta=True)
                    kwargs = {} if materialise is None else {"materialise": materialise}
                    with type(original).load(tmp, **kwargs) as loaded:
                        expected = loaded[0].atomics.copy()
                        with loaded.subset([0]) as subset:
                            detached = subset.read()
                            nested = subset.subset([0])
                            nested.close_hdf5()
                        combined = type(original).from_batches([loaded, subset])
                        combined.close_hdf5()
                        np.testing.assert_array_equal(loaded[0].atomics, expected)
                        np.testing.assert_array_equal(subset[0].atomics, expected)
                        np.testing.assert_array_equal(combined[0].atomics, expected)
                        files = list(loaded._open_fps)
                    self.assertTrue(all(not fp.id.valid for fp in files))
                    np.testing.assert_array_equal(detached[0].atomics, expected)
                    # Saving after closure also exercises detached metadata and conformers.
                    detached.save(Path(tmp) / "detached")
                    loaded.close_hdf5()

    def test_context_closes_files_on_exception(self):
        with tempfile.TemporaryDirectory() as tmp:
            GraphBatch.from_smiles(["C"]).save(tmp)
            with self.assertRaisesRegex(RuntimeError, "test"):
                with GraphBatch.load(tmp) as loaded:
                    files = list(loaded._open_fps)
                    raise RuntimeError("test")
            self.assertTrue(all(not fp.id.valid for fp in files))
