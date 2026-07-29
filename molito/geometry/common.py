import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign


def possibly_add_hs(mol: Chem.Mol, max_iters: int = 100) -> Chem.Mol | None:
    """Adds Hs (and applies a small minimisation to them) to the molecule if they are needed.

    The function will only add Hs if there are none in the molecule, otherwise none will be added.

    If the molecule has at least one conformer, coords are added to the Hs and a small minimisation is applied with
    the MMFF forcefield, keeping all heavy atoms fixed.

    Args:
        mol (Chem.Mol): Molecule to have Hs added. The molecule is not modified.
        max_iters (int): Maximum number of optimisation steps applied to H atoms.

    Returns:
        Chem.Mol: A copy of mol with Hs added
    """

    mol_hs = Chem.Mol(mol)

    add_hs = mol_hs.GetNumAtoms() == mol_hs.GetNumHeavyAtoms()

    try:
        mol_hs = Chem.AddHs(mol_hs, addCoords=True) if add_hs else mol_hs
    except Exception:
        return None

    if not add_hs:
        return mol_hs

    heavy_idxs = [idx for idx, atom in enumerate(mol_hs.GetAtoms()) if atom.GetAtomicNum() != 1]

    try:
        ff_props = AllChem.MMFFGetMoleculeProperties(mol_hs)
        ff = AllChem.MMFFGetMoleculeForceField(mol_hs, ff_props)

        for idx in heavy_idxs:
            ff.MMFFAddPositionConstraint(idx, 0, 10000.0)

        AllChem.OptimizeMoleculeConfs(mol_hs, ff, maxIters=max_iters)
    except Exception:
        return None

    return mol_hs


def _dedup_conformers(mol, rmsd_threshold=0.5):
    mol_copy = Chem.Mol(mol)

    n_confs = mol_copy.GetNumConformers()
    if n_confs <= 1:
        return list(range(n_confs))

    curr_indices = [0]

    for i in range(1, n_confs):
        is_unique = True

        for j in curr_indices:
            rmsd = rdMolAlign.AlignMol(mol_copy, mol_copy, i, j)
            if rmsd < rmsd_threshold:
                is_unique = False
                break

        if is_unique:
            curr_indices.append(i)

    return curr_indices


def _calc_weights(energies, temp=300):
    """Energies in kcal/mol"""

    kT = 0.001987 * temp

    energies = np.array(energies)
    relative_energies = energies - np.min(energies)

    boltzmann_factors = np.exp(-relative_energies / kT)
    weights = boltzmann_factors / np.sum(boltzmann_factors)
    return weights


def sample_conformers(
    mol: Chem.rdchem.Mol,
    n_confs: int = 1,
    max_attempts: int = 10,
    fast_conf: bool = False,
    opt_iters: int | None = None,
    n_threads: int = 1,
) -> Chem.rdchem.Mol:
    """Create a (set of) conformer(s) for a molecule using the RDKit ETKDGv3 method.

    The molecule is copied and the input is not modified.

    NOTE if the input molecule contains at least one H atom, no further Hs will be added before the conf gen.
    If there are no Hs, Hs will be added with RDkit.

    Args:
        mol (Chem.Mol): RDKit molecule (existing conformers will be ignored)
        n_confs (int): The number of conformers to generate, default 1
        max_attempts (int): Max num of attempts per conformer, default 1
        fast_conf (bool): Whether to use a faster version of conf gen with lower quality results, default False
        opt_iters (int, optional): Optional apply some number of MMFF optimisation iters to each conf
        n_threads (int): Number of threads to ask RDKit to use (0 means use all available processors)

    Returns:
        Chem.Mol: Copied molecule with conformers added (or None on failure)
    """

    from molito.geometry.mmff import optimise_mol_mmff

    mol_copy = Chem.Mol(mol)

    try:
        Chem.SanitizeMol(mol_copy)
    except Exception:
        return None

    # Hs are added and will be returned in the sampled conformers
    contains_hs = mol_copy.GetNumAtoms() != mol_copy.GetNumHeavyAtoms()
    mol_copy = Chem.AddHs(mol_copy) if not contains_hs else mol_copy

    params = AllChem.ETKDGv3()
    params.maxIterations = max_attempts
    params.numThreads = n_threads

    if fast_conf:
        params.optimizerForceTol = 0.002
        params.useBasicKnowledge = False
        params.useExpTorsionAnglePrefs = False

    try:
        outs = AllChem.EmbedMultipleConfs(mol_copy, n_confs, params)
    except Exception:
        return None

    if len(outs) != n_confs:
        return None

    if opt_iters not in [None, 0]:
        mol_copy = optimise_mol_mmff(mol_copy, max_iters=opt_iters, n_threads=n_threads, allow_unconverged=True)
        if mol_copy is None:
            return None

    return mol_copy


def sample_ensemble(
    mol,
    max_confs=128,
    max_conf_attempts=100,
    max_opt_iters=1000,
    dedup_rmsd_threshold=0.5,
    strain_filter=6.0,
    temp=300.0,
    n_threads=1,
):
    """Generate a full set of conformers and weights for a given molecule.

    Samples conformers for a molecular graph using RDKit ETKDG, applies MMFF optimisation to each and applies
    RMSD-based deduplication before calculating boltzmann weights. This will return a copy of the mol with new confs.

    Args:
        mol (Chem.Mol): RDKit molecule for an ensemble to be sampled
        max_confs (int): Max number of conformers to attempt to sample
        max_opt_iters (int): Maximum number of MMFF optimisation steps to apply
        strain_filter (float): Filter out conformers with global strain higher than this
        dedup_rmsd_threshold (float): RMSD threshold for removing deduplicate conformers
        temp (float): Temperature for boltzmann weight calculation
        n_threads (int): Number of threads to ask RDKit to use (0 means use all available processors)

    Returns:
        (Chem.Mol, np.ndarray, float): Molecule with embedded confs, conformer weights, and minimum energy
    """

    from molito.geometry.mmff import calc_energy_mmff

    embedded = sample_conformers(
        mol, n_confs=max_confs, max_attempts=max_conf_attempts, opt_iters=max_opt_iters, n_threads=n_threads
    )

    if embedded is None:
        return None

    if dedup_rmsd_threshold not in [None, 0.0]:
        conf_indices = _dedup_conformers(embedded, rmsd_threshold=dedup_rmsd_threshold)
    else:
        conf_indices = list(range(embedded.GetNumConformers()))

    confs = [embedded.GetConformer(idx) for idx in conf_indices]

    if len(confs) == 0:
        return None

    emb_dedup = Chem.Mol(embedded)
    emb_dedup.RemoveAllConformers()

    for conf in confs:
        emb_dedup.AddConformer(conf, assignId=True)

    energies = calc_energy_mmff(emb_dedup)
    energies = [energies] if not isinstance(energies, list) else energies

    assert len(energies) == len(confs)

    is_valid = [energy is not None for energy in energies]
    confs = [conf for valid, conf in zip(is_valid, confs, strict=True) if valid]
    energies = [energy for valid, energy in zip(is_valid, energies, strict=True) if valid]

    if len(energies) == 0:
        return None

    global_strains = np.array(energies) - min(energies)

    is_valid = (global_strains <= strain_filter).tolist()
    confs = [conf for valid, conf in zip(is_valid, confs, strict=True) if valid]
    energies = [energy for valid, energy in zip(is_valid, energies, strict=True) if valid]

    assert len(energies) == len(confs)

    final_mol = Chem.Mol(emb_dedup)
    final_mol.RemoveAllConformers()

    for conf in confs:
        final_mol.AddConformer(conf, assignId=True)

    weights = _calc_weights(energies, temp=temp)
    e_min = min(energies)
    return final_mol, weights, e_min
