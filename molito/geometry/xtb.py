import numpy as np
from rdkit import Chem

from molito.geometry.common import possibly_add_hs

BOHR_PER_ANGSTROM = 1.8897259886


def _get_xtb_method_map():
    """Lazy-load xTB Param enum and return method map. Requires xtb-python."""

    try:
        from xtb.interface import Param
    except ImportError as e:
        raise ImportError(
            "optimise_mol_xtb requires xtb-python, which has no working pip wheel. "
            "Install from conda-forge: `mamba install -c conda-forge xtb-python`"
        ) from e

    return {"GFN1-xTB": Param.GFN1xTB, "GFN2-xTB": Param.GFN2xTB, "GFN-FF": Param.GFNFF}


def _make_xtb_calculator(atomic_nums, positions_bohr, method, accuracy, electronic_temperature, solvent):
    """Create a muted xTB calculator with the given settings. Requires xtb-python."""

    from xtb.interface import Calculator
    from xtb.libxtb import VERBOSITY_MUTED

    method_map = _get_xtb_method_map()
    calc = Calculator(method_map[method], atomic_nums, positions_bohr)
    calc.set_verbosity(VERBOSITY_MUTED)
    calc.set_accuracy(accuracy)
    calc.set_electronic_temperature(electronic_temperature)

    if solvent is not None:
        calc.set_solvent(solvent)

    return calc


def _check_positions(positions):
    """Return True if positions are valid for xTB (no NaN/Inf, no overlapping atoms)."""

    if not np.all(np.isfinite(positions)):
        return False

    n_atoms = len(positions)
    if n_atoms < 2:
        return True

    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < 0.1:
                return False

    return True


def _xtb_energy_and_gradient(calc, coords_flat, n_atoms):
    """Run an xTB singlepoint and return (energy, flat gradient)."""

    coords = coords_flat.reshape(n_atoms, 3)

    if not _check_positions(coords):
        raise ValueError("Invalid positions: NaN/Inf or overlapping atoms")

    calc.update(coords)
    res = calc.singlepoint()
    return res.get_energy(), res.get_gradient().flatten()


def _set_conf_positions(mol, positions):
    """Return a copy of mol with a single conformer set to the given positions."""

    mol_out = Chem.Mol(mol)
    mol_out.RemoveAllConformers()
    conf = Chem.Conformer(mol_out.GetNumAtoms())

    for i, pos in enumerate(positions):
        conf.SetAtomPosition(i, pos.tolist())

    mol_out.AddConformer(conf, assignId=True)
    return mol_out


def optimise_mol_xtb(
    mol: Chem.rdchem.Mol,
    conf_idx: int = 0,
    max_iters: int = 200,
    method: str = "GFN2-xTB",
    accuracy: float = 1.0,
    electronic_temperature: float = 300.0,
    solvent: str | None = None,
):
    """Optimise a conformer using the xTB semi-empirical method and return the minimised mol and energy.

    Uses xtb-python for energy/gradient evaluation and scipy L-BFGS-B for geometry optimisation.
    The molecule is copied so the original is not modified. Hs are added if not already present since xTB
    requires all atoms. The returned molecule will have the same H-atom status as the input.

    Requires xtb-python (conda-forge only): `mamba install -c conda-forge xtb-python`.

    Args:
        mol (Chem.Mol): RDKit molecule with at least one conformer
        conf_idx (int): Index of the conformer to optimise, default 0
        max_iters (int): Maximum number of geometry optimisation steps
        method (str): xTB method to use, one of "GFN1-xTB", "GFN2-xTB", "GFN-FF"
        accuracy (float): Numerical accuracy for the calculation (lower = tighter, 1.0 is default)
        electronic_temperature (float): Electronic temperature in Kelvin for Fermi smearing
        solvent (str, optional): ALPB solvent name (e.g. "water", "methanol"), None for gas phase

    Returns:
        (Chem.Mol, float, float): Tuple of (optimised molecule, final energy in Hartree, initial energy in Hartree),
            or None if optimisation fails
    """

    import os

    from scipy.optimize import minimize

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    _XTB_METHOD_MAP = _get_xtb_method_map()
    if method not in _XTB_METHOD_MAP:
        raise ValueError(f"Unknown xTB method '{method}', must be one of {list(_XTB_METHOD_MAP.keys())}")

    mol_copy = Chem.Mol(mol)

    try:
        Chem.SanitizeMol(mol_copy)
    except Exception:
        return None

    contains_hs = mol_copy.GetNumAtoms() != mol_copy.GetNumHeavyAtoms()
    mol_copy = possibly_add_hs(mol_copy, max_iters=50) if not contains_hs else mol_copy

    if mol_copy is None:
        return None

    positions = np.array(mol_copy.GetConformer(conf_idx).GetPositions())
    atomics = np.array([atom.GetAtomicNum() for atom in mol_copy.GetAtoms()])
    n_atoms = len(atomics)

    positions_bohr = positions * BOHR_PER_ANGSTROM
    if not _check_positions(positions_bohr):
        return None

    try:
        calc = _make_xtb_calculator(atomics, positions_bohr, method, accuracy, electronic_temperature, solvent)

        x0 = positions_bohr.flatten()
        initial_energy, _ = _xtb_energy_and_gradient(calc, x0, n_atoms)

        _cache = {}

        def cached_eval(coords_flat):
            key = coords_flat.tobytes()
            if key not in _cache:
                _cache.clear()
                _cache[key] = _xtb_energy_and_gradient(calc, coords_flat, n_atoms)
            return _cache[key]

        result = minimize(
            fun=lambda x: cached_eval(x)[0],
            x0=x0,
            jac=lambda x: cached_eval(x)[1],
            method="L-BFGS-B",
            options={"maxiter": max_iters, "gtol": 1e-5},
        )

    except Exception as e:
        print(f"[optimise_mol_xtb] failed: {e}")
        return None

    opt_positions = result.x.reshape(n_atoms, 3) / BOHR_PER_ANGSTROM
    opt_mol = _set_conf_positions(mol_copy, opt_positions)
    opt_mol = opt_mol if contains_hs else Chem.RemoveAllHs(opt_mol)
    return opt_mol, result.fun, initial_energy
