"""
castep/config.py  —  CASTEP engine configuration.

All CASTEP-specific defaults in one place.
Global settings (VEC thresholds, elastic strain) live in the top-level config.py.

Tuning guide
────────────
ELASTIC_STRAIN_N_MAGNITUDES  — Number of positive strain amplitudes per pattern.
    The minimum needed for a reliable OLS fit is 3 (one on each side of zero
    plus zero itself is implicit).  Increasing to 5-7 improves R² and catches
    non-linearity but multiplies compute by 5/3 or 7/3.
    Default: 3  (gives 6 SinglePoint runs for a cubic cell, ~2-3 min on 6 cores)

ELASTIC_USE_ONLY_THREE_POINTS — When True, only ±1 and the central (zero-strain)
    stress from the GeomOpt output are used for OLS, giving the minimum 3 data
    points and halving the number of CASTEP runs vs the default 3-magnitude setting.
    Useful for quick screening; set False for publication-quality data.
    Default: False
"""

# ── Binary discovery ──────────────────────────────────────────────────────────
BINARY_DEFAULT:  str = "castep.mpi"
BINARY_FALLBACKS: list[str] = [
    "~/Applications/CASTEP*/bin/linux*/castep.mpi",
    "~/castep*/bin/linux*/castep.mpi",
    "/opt/castep*/bin/castep.mpi",
    "/usr/local/bin/castep.mpi",
    "~/bin/castep.mpi",
]
SERIAL_NAMES: list[str] = ["castep.serial", "castep"]

# ── Output suffix ─────────────────────────────────────────────────────────────
OUTPUT_SUFFIX: str = ".castep"

# ── Default DFT settings ──────────────────────────────────────────────────────
XC_DEFAULT:     str = "PBE"
CUTOFF_HARD:    int = 700    # eV — mandatory for C, N, O, F, B, H
CUTOFF_SOFT:    int = 500    # eV — pure metals
CUTOFF_NCP:     int = 900    # eV — norm-conserving PSP

# ── SCF convergence ───────────────────────────────────────────────────────────
MAX_SCF:        int   = 300
ELEC_TOL_GEOM:  str   = "1.0e-5 eV"
ELEC_TOL_SP:    str   = "1.0e-7 eV"  # tighter for elastic Cij
METALS_METHOD:  str   = "dm"
MIXING_SCHEME:  str   = "Pulay"
MIX_AMP_GEOM:   float = 0.2
MIX_AMP_SP:     float = 0.1           # conservative for cold strained cells
FINITE_BASIS:   int   = 1             # one correction pass (~60% cost of mode 2)

# ── Geometry optimisation ─────────────────────────────────────────────────────
GEOM_MAX_ITER:  int   = 150
GEOM_E_TOL:     str   = "1.0e-6 eV"
GEOM_F_TOL:     str   = "0.005 eV/ang"
GEOM_S_TOL:     str   = "0.03 GPa"
GEOM_D_TOL:     str   = "0.0005 ang"

# ── Elastic finite-strain workflow ────────────────────────────────────────────
# Number of positive strain magnitudes per pattern (default 3 → 6 SinglePoints
# per cubic cell pattern).  Minimum for a valid OLS fit is 3.
# Raise to 5-7 for higher R² at the cost of proportionally more compute.
ELASTIC_STRAIN_N_MAGNITUDES: int = 3

# When True the workflow reuses the zero-strain stress from the GeomOpt output
# and runs only ±1 × max_strain CASTEP SinglePoints (2 instead of 6 for cubic).
# Minimum valid mode — useful for quick screening, not for publication.
ELASTIC_USE_ONLY_THREE_POINTS: bool = False

# ── Files to delete after a successful step (unless --keep-all) ───────────────
CLEANUP_GLOBS: tuple[str, ...] = (
    "*.castep_bin", "*.check", "*.cst_esp",
    "*.bands", "*.bib", "*.usp", "*.upf", "*.oepr",
)

# ── Hard / magnetic element sets ─────────────────────────────────────────────
HARD_ELEMENTS:     frozenset[str] = frozenset({"C", "N", "O", "F", "B", "H"})
MAGNETIC_ELEMENTS: frozenset[str] = frozenset({"Fe", "Co", "Ni", "Mn", "Cr", "Gd", "V"})
D_BLOCK:           frozenset[str] = frozenset({
    "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
})
METALS: frozenset[str] = frozenset({
    "Li", "Na", "K",  "Rb", "Cs", "Be", "Mg", "Ca", "Sr", "Ba", "Ra",
    "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Al", "Ga", "In", "Sn", "Tl", "Pb", "Bi",
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu", "Th", "U",  "Pu",
})
NONMETALS: frozenset[str] = frozenset({
    "B",  "C",  "N",  "O",  "F",  "Si", "P",  "S",
    "Cl", "As", "Se", "Br", "Te", "I",  "At", "H",
})
