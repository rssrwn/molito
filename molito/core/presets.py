"""Pre-made atom token lists for common chemistry domains.

Each list contains base tokens in element_charge format (e.g. "C_0", "N_1").
Chirality expansion (CW/CCW suffixes) is handled by AtomVocab.build() based on
the chirality flag and CHIRAL_ELIGIBLE, not by the preset itself.

Presets are designed to compose. Use the additive groups (SELENIUM_ATOMS,
COMMON_METAL_IONS) to extend DRUG_LIKE_ATOMS for richer chemistry, or pick the
BIO_COMPLEX_ATOMS superset for protein-ligand training data that may contain
selenocysteine and bound metal ions.
"""

# Covers the most common atom types in drug-like molecules
DRUG_LIKE_ATOMS = [
    "Br_1",
    "S_1",
    "S_0",
    "F_-1",
    "O_0",
    "I_0",
    "Bi_2",
    "N_1",
    "F_0",
    "O_-1",
    "C_1",
    "C_-1",
    "Bi_0",
    "P_0",
    "H_0",
    "Cl_1",
    "Cl_0",
    "Br_0",
    "Cl_-1",
    "S_2",
    "I_1",
    "B_-1",
    "N_0",
    "N_-1",
    "Si_0",
    "P_-1",
    "B_0",
    "N_-2",
    "H_1",
    "I_2",
    "P_1",
    "O_1",
    "C_0",
    "Si_1",
    "S_-1",
    "S_3",
]

# Selenium variants. Se_0 covers selenocysteine and most selenium-containing
# ligands (ebselen, selenomethionine analogues, etc.).
SELENIUM_ATOMS = [
    "Se_0",
    "Se_-1",
    "Se_1",
]

# Common biologically-bound metal ions. RCSB CIFs almost always report these
# with formal charge 0, so the charge-0 tokens are what biotite hands us; we
# also expose the formal-charge variants for callers that infer charges from
# residue identity (Zn²⁺, Mg²⁺, Ca²⁺ etc.).
COMMON_METAL_IONS = [
    "Zn_0",
    "Zn_2",
    "Mg_0",
    "Mg_2",
    "Ca_0",
    "Ca_2",
    "Mn_0",
    "Mn_2",
    "Fe_0",
    "Fe_2",
    "Fe_3",
    "Cu_0",
    "Cu_1",
    "Cu_2",
    "Ni_0",
    "Ni_2",
    "Co_0",
    "Co_2",
    "Co_3",
    "Na_0",
    "Na_1",
    "K_0",
    "K_1",
]

# Convenience superset for protein-ligand complexes that may contain
# selenocysteine residues and bound metal cofactors.
BIO_COMPLEX_ATOMS = DRUG_LIKE_ATOMS + SELENIUM_ATOMS + COMMON_METAL_IONS

# Base tokens eligible for chirality expansion (CW/CCW suffixes added by AtomVocab.build)
CHIRAL_ELIGIBLE = {"C_0"}
