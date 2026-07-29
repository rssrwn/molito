from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures

RDKIT_SMARTS_PATH = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
SMARTS_PATH = os.path.join(os.path.dirname(__file__), "../defs/pharmacophore.fdef")


TArr = np.ndarray


@dataclass
class PharmacophoreFeature:
    type: int
    atom_ids: tuple[int, ...]
    position: TArr = None
    direction: TArr = None


class PharmacophoreFinder:
    """Pharmacophore feature detection on RDKit molecules.

    Uses SMARTS-based feature definitions to identify pharmacophore features (donors, acceptors,
    aromatics, etc.). Works with or without 3D conformers — positions are populated when a
    conformer is available, and directions can be computed for donors (requires Hs) and aromatics.

    All methods are classmethods operating on shared state (the feature factory). Use
    set_default_features / set_rdkit_features / set_custom_features to change the SMARTS definitions.
    """

    _feature_factory = ChemicalFeatures.BuildFeatureFactory(SMARTS_PATH)

    @classmethod
    def set_default_features(cls):
        """Reset the feature factory to the default molito SMARTS definitions."""

        cls._feature_factory = ChemicalFeatures.BuildFeatureFactory(SMARTS_PATH)

    @classmethod
    def set_rdkit_features(cls):
        """Switch the feature factory to RDKit's built-in BaseFeatures definitions."""

        cls._feature_factory = ChemicalFeatures.BuildFeatureFactory(RDKIT_SMARTS_PATH)

    @classmethod
    def set_custom_features(cls, path: str):
        """Load a custom feature definition file (.fdef) into the feature factory.

        Args:
            path: Path to a .fdef file containing SMARTS-based feature definitions.
        """

        cls._feature_factory = ChemicalFeatures.BuildFeatureFactory(path)

    @classmethod
    def get_vocab_size(cls) -> int:
        """Return the number of feature families in the current feature factory."""

        return len(cls.get_feature_vocab())

    @classmethod
    def get_feature_vocab(cls) -> list[str]:
        """Return the list of feature family names (e.g. ['Donor', 'Acceptor', 'Aromatic', ...])."""

        return list(cls._feature_factory.GetFeatureFamilies())

    @classmethod
    def get_feature_index(cls, feature: str) -> int:
        """Return the index of a feature family name in the current vocab."""

        feat_idx_map = {feat: idx for idx, feat in enumerate(cls.get_feature_vocab())}
        return feat_idx_map[feature]

    @classmethod
    def get_feature_name(cls, feat_idx: int) -> str:
        """Return the feature family name for a given index."""

        features = cls.get_feature_vocab()
        if feat_idx >= len(features):
            raise RuntimeError(f"Tried to access feature index {feat_idx} with only {len(features)} features.")

        return features[feat_idx]

    @classmethod
    def run_mol(
        cls, mol: Chem.rdchem.Mol, conf_idx: int | None = None, directions: bool = False
    ) -> list[PharmacophoreFeature]:
        """Detect pharmacophore features on a molecule.

        SMARTS matching works with or without 3D coordinates. Positions are populated when the
        molecule has at least one conformer. Direction vectors (donor H-bond directions and
        aromatic ring normals) are computed when directions=True.

        Args:
            mol: RDKit molecule.
            conf_idx: Conformer index to use for positions/directions. None uses the default (-1).
            directions: If True, compute direction vectors for donor and aromatic features.
                Requires a conformer and explicit Hs in the molecule (raises ValueError otherwise).
        """

        conf_idx = -1 if conf_idx is None else conf_idx
        has_conf = mol.GetNumConformers() > 0

        if directions and not has_conf:
            raise ValueError("directions=True requires a molecule with at least one conformer.")

        if directions and mol.GetNumAtoms() == mol.GetNumHeavyAtoms():
            raise ValueError("directions=True requires explicit Hs in the molecule for donor direction vectors.")

        feats = cls._feature_factory.GetFeaturesForMol(mol, confId=conf_idx)

        results = []
        for feature in feats:
            position = None
            if has_conf:
                pos = feature.GetPos()
                position = np.array([pos.x, pos.y, pos.z])

            results.append(
                PharmacophoreFeature(
                    type=cls.get_feature_index(feature.GetFamily()),
                    atom_ids=tuple(feature.GetAtomIds()),
                    position=position,
                )
            )

        if directions:
            results = cls._expand_donor_directions(mol, results, conf_idx)
            results = cls._expand_aromatic_directions(mol, results, conf_idx)

        return results

    @classmethod
    def _expand_donor_directions(
        cls, mol: Chem.rdchem.Mol, features: list[PharmacophoreFeature], conf_idx: int = -1
    ) -> list[PharmacophoreFeature]:
        """Expand each donor feature into one feature per bonded H, with direction vectors.

        For donors: direction = normalised(H_pos - heavy_atom_pos) for each bonded H.
        Non-donors pass through unchanged. If a donor has no bonded Hs, it passes through unchanged.
        """

        donor_type_idx = cls.get_feature_index("Donor")
        conf = mol.GetConformer(conf_idx)
        expanded = []

        for feat in features:
            if feat.type != donor_type_idx:
                expanded.append(feat)
                continue

            h_atoms = []
            for atom_idx in feat.atom_ids:
                atom = mol.GetAtomWithIdx(atom_idx)
                for neighbor in atom.GetNeighbors():
                    if neighbor.GetAtomicNum() == 1:
                        h_atoms.append((atom_idx, neighbor.GetIdx()))

            if not h_atoms:
                expanded.append(feat)
                continue

            for heavy_idx, h_idx in h_atoms:
                heavy_pos = np.array(conf.GetAtomPosition(heavy_idx))
                h_pos = np.array(conf.GetAtomPosition(h_idx))
                direction = h_pos - heavy_pos
                norm = np.linalg.norm(direction)
                if norm > 1e-8:
                    direction = direction / norm

                expanded.append(
                    PharmacophoreFeature(
                        type=feat.type,
                        atom_ids=feat.atom_ids,
                        position=feat.position.copy() if feat.position is not None else None,
                        direction=direction,
                    )
                )

        return expanded

    @classmethod
    def _expand_aromatic_directions(
        cls, mol: Chem.rdchem.Mol, features: list[PharmacophoreFeature], conf_idx: int = -1
    ) -> list[PharmacophoreFeature]:
        """Set direction vectors for aromatic features to the ring normal.

        The ring normal is computed from the cross product of two in-plane vectors
        formed by the ring atom positions. Non-aromatic features pass through unchanged.
        """

        aromatic_type_idx = cls.get_feature_index("Aromatic")
        conf = mol.GetConformer(conf_idx)
        expanded = []

        for feat in features:
            if feat.type != aromatic_type_idx or len(feat.atom_ids) < 3:
                expanded.append(feat)
                continue

            positions = np.array([list(conf.GetAtomPosition(idx)) for idx in feat.atom_ids])
            v1 = positions[1] - positions[0]
            v2 = positions[2] - positions[0]
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm > 1e-8:
                normal = normal / norm

            expanded.append(
                PharmacophoreFeature(
                    type=feat.type,
                    atom_ids=feat.atom_ids,
                    position=feat.position.copy() if feat.position is not None else None,
                    direction=normal,
                )
            )

        return expanded
