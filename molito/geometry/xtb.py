import os

import numpy as np
from rdkit import Chem
from scipy.constants import Avogadro, calorie, physical_constants

from molito.geometry.common import possibly_add_hs

BOHR_PER_ANGSTROM = 1.8897259886
KCAL_MOL_PER_HARTREE = physical_constants["Hartree energy"][0] * Avogadro / (1000 * calorie)


def _xtb_energy_factor(units: str) -> float:
    """Scale native Hartree energies to the requested output units."""

    if units == "kcal/mol":
        return KCAL_MOL_PER_HARTREE
    if units == "hartree":
        return 1.0
    raise ValueError("units must be 'kcal/mol' or 'hartree'.")


def _get_xtb_method_map():
    """Lazy-load xTB Param enum and return method map. Requires xtb-python."""

    try:
        from xtb.interface import Param
    except ImportError as e:
        raise ImportError(
            "xTB calculations require xtb-python, which has no working pip wheel. "
            "Install from conda-forge: `mamba install -c conda-forge xtb-python`"
        ) from e

    return {"GFN1-xTB": Param.GFN1xTB, "GFN2-xTB": Param.GFN2xTB, "GFN-FF": Param.GFNFF}


def _make_xtb_calculator(atomic_nums, positions_bohr, method, accuracy, electronic_temperature, solvent, charge, uhf):
    """Create a muted xTB calculator with the given settings. Requires xtb-python."""

    from xtb.interface import Calculator
    from xtb.libxtb import VERBOSITY_MUTED
    from xtb.utils import get_solvent

    method_map = _get_xtb_method_map()
    calc = Calculator(method_map[method], atomic_nums, positions_bohr, charge=charge, uhf=uhf)
    calc.set_verbosity(VERBOSITY_MUTED)
    calc.set_accuracy(accuracy)
    calc.set_electronic_temperature(electronic_temperature)

    if solvent is not None:
        solvent_enum = get_solvent(solvent)
        if solvent_enum is None:
            raise ValueError(f"Unknown xTB solvent '{solvent}'.")
        calc.set_solvent(solvent_enum)

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


def calc_energy_xtb(
    mol: Chem.rdchem.Mol,
    per_atom: bool = False,
    method: str = "GFN2-xTB",
    accuracy: float = 1.0,
    electronic_temperature: float = 300.0,
    solvent: str | None = None,
    uhf: int | None = None,
    units: str = "kcal/mol",
) -> float | list[float | None] | None:
    """Calculate xTB single-point energies in kcal/mol without optimising the geometry.

    The molecule is copied and missing Hs are added with RDKit-generated coordinates. Existing
    coordinates are preserved; no MMFF or xTB minimisation is performed. For energies at an exact
    geometry, supply all hydrogens explicitly. Multiple conformers are evaluated independently
    in iteration order, regardless of their IDs. Requires optional xtb-python.

    Args:
        mol (Chem.Mol): RDKit molecule with coordinates.
        per_atom (bool): Divide by the number of atoms including added hydrogens, default False.
        method (str): One of "GFN1-xTB", "GFN2-xTB", "GFN-FF".
        accuracy (float): Numerical accuracy (lower = tighter, 1.0 is default).
        electronic_temperature (float): Electronic temperature in Kelvin for Fermi smearing.
        solvent (str, optional): Solvent name (e.g. "water"), None for gas phase.
        uhf (int, optional): Unpaired electrons; defaults to the sum of RDKit radical electrons.
            Supply explicitly when a different spin coupling is required.
        units (str): Output energy units, "kcal/mol" (default) or "hartree".

    Returns:
        A float for one conformer, or a list in conformer iteration order. Failed conformers
        return None entries. Returns None if the molecule cannot be prepared or has no conformers.

    Raises:
        ImportError: If xtb-python is unavailable.
        ValueError: If method, uhf or units is invalid.
    """

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    energy_factor = _xtb_energy_factor(units)
    method_map = _get_xtb_method_map()
    if method not in method_map:
        raise ValueError(f"Unknown xTB method '{method}', must be one of {list(method_map)}")
    if uhf is not None and (not isinstance(uhf, int) or uhf < 0):
        raise ValueError("uhf must be a nonnegative integer.")

    try:
        mol_copy = Chem.Mol(mol)
        Chem.SanitizeMol(mol_copy)
        mol_copy = Chem.AddHs(mol_copy, addCoords=True)
    except Exception:
        return None

    n_atoms = mol_copy.GetNumAtoms()
    if n_atoms == 0 or mol_copy.GetNumConformers() == 0:
        return None

    atomics = np.array([atom.GetAtomicNum() for atom in mol_copy.GetAtoms()])
    charge = Chem.GetFormalCharge(mol_copy)
    uhf = sum(atom.GetNumRadicalElectrons() for atom in mol_copy.GetAtoms()) if uhf is None else uhf

    energies = []
    for conf in mol_copy.GetConformers():
        positions_bohr = conf.GetPositions() * BOHR_PER_ANGSTROM
        energy = None
        if _check_positions(positions_bohr):
            try:
                calc = _make_xtb_calculator(
                    atomics, positions_bohr, method, accuracy, electronic_temperature, solvent, charge, uhf
                )
                value = float(calc.singlepoint().get_energy())
                if np.isfinite(value):
                    energy = value * energy_factor / n_atoms if per_atom else value * energy_factor
            except Exception:
                pass
        energies.append(energy)

    return energies[0] if len(energies) == 1 else energies


def optimise_mol_xtb(
    mol: Chem.rdchem.Mol,
    conf_idx: int = 0,
    max_iters: int = 200,
    method: str = "GFN2-xTB",
    accuracy: float = 1.0,
    electronic_temperature: float = 300.0,
    solvent: str | None = None,
    uhf: int | None = None,
    allow_unconverged: bool = True,
    units: str = "kcal/mol",
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
        uhf (int, optional): Number of unpaired electrons; defaults to the sum of RDKit radical electrons.
            Supply explicitly when a different spin coupling is required.
        allow_unconverged (bool): Return the last finite geometry if the iteration limit is reached.
            Set False to require convergence. Other optimiser failures return None.
        units (str): Output energy units, "kcal/mol" (default) or "hartree". Optimisation
            and its convergence tolerances always use native atomic units internally.

    Returns:
        (Chem.Mol, float, float): Tuple of (optimised molecule, final energy, initial energy),
            with both energies in the requested units, or None if optimisation fails.
    """

    from scipy.optimize import minimize

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    energy_factor = _xtb_energy_factor(units)
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

    try:
        positions = np.array(mol_copy.GetConformer(conf_idx).GetPositions())
    except ValueError:
        return None
    atomics = np.array([atom.GetAtomicNum() for atom in mol_copy.GetAtoms()])
    n_atoms = len(atomics)
    charge = Chem.GetFormalCharge(mol_copy)
    uhf = sum(atom.GetNumRadicalElectrons() for atom in mol_copy.GetAtoms()) if uhf is None else uhf
    if not isinstance(uhf, int) or uhf < 0:
        raise ValueError("uhf must be a nonnegative integer.")

    positions_bohr = positions * BOHR_PER_ANGSTROM
    if not _check_positions(positions_bohr):
        return None

    try:
        calc = _make_xtb_calculator(
            atomics, positions_bohr, method, accuracy, electronic_temperature, solvent, charge, uhf
        )

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

    if not result.success and (not allow_unconverged or result.status != 1):
        return None
    if not np.isfinite(result.fun) or not _check_positions(result.x.reshape(n_atoms, 3)):
        return None

    opt_positions = result.x.reshape(n_atoms, 3) / BOHR_PER_ANGSTROM
    opt_mol = _set_conf_positions(mol_copy, opt_positions)
    opt_mol = opt_mol if contains_hs else Chem.RemoveAllHs(opt_mol)
    return opt_mol, result.fun * energy_factor, initial_energy * energy_factor
