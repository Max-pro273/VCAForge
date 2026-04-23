"""
config.py  —  VCAForge global configuration (Single Source of Truth).
═════════════════════════════════════════════════════════════════════
All engine-agnostic and engine-specific constants live here.
Unified element database for atomic numbers, valences, radii, and flags.
"""

from typing import Any

# ── Project & IO Directory Names ─────────────────────────────────────────────
VERSION: str = "2.0"
STATE_FILE: str = "vca_state.json"
CSV_FILE: str = "vca_results.csv"
LOG_FILE: str = "run.log"
ELASTIC_DIR: str = "elastic"

CASTEP_SUBDIR: str = "CASTEP"
VASP_SUBDIR: str = "VASP"

# ── VCA Sweep Defaults ───────────────────────────────────────────────────────
SWEEP_C_START: float = 0.0
SWEEP_C_END: float = 1.0
SWEEP_N_DEFAULT: int = 8

# ── VEC Stability Thresholds (Ti-Nb-C heuristic) ─────────────────────────────
VEC_YELLOW: float = 8.2  # Marginal zone — extra SCF iterations often needed
VEC_RED: float = 8.4     # Born instability zone — C44 likely negative

# ── Elastic Workflow ─────────────────────────────────────────────────────────
ELASTIC_MAX_STRAIN: float = 0.003  # Increased to override SCF noise
ELASTIC_N_STEPS: int = 3
ELASTIC_NEXTRA_PURE: int = 10
ELASTIC_NEXTRA_BASE: int = 15

# ── Smearing (eV) ────────────────────────────────────────────────────────────
SMEARING_VCA: float = 0.20     # Broadened for fractional Z
SMEARING_SINGLE: float = 0.10  # Standard for ordered structures

# ── Unified Element Database ─────────────────────────────────────────────────
# Structure: { Symbol: {"Z": int, "val": int, "rad": float, "hard": bool, "mag": bool, "nonmetal": bool} }
# rad = Atomic radius in Angstroms
ELEMENTS: dict[str, dict[str, Any]] = {
    # Non-Metals
    "H":  {"Z": 1,  "val": 1, "rad": 0.53, "hard": True,  "mag": False, "nonmetal": True},
    "B":  {"Z": 5,  "val": 3, "rad": 0.87, "hard": True,  "mag": False, "nonmetal": True},
    "C":  {"Z": 6,  "val": 4, "rad": 0.77, "hard": True,  "mag": False, "nonmetal": True},
    "N":  {"Z": 7,  "val": 5, "rad": 0.75, "hard": True,  "mag": False, "nonmetal": True},
    "O":  {"Z": 8,  "val": 6, "rad": 0.73, "hard": True,  "mag": False, "nonmetal": True},
    "F":  {"Z": 9,  "val": 7, "rad": 0.60, "hard": True,  "mag": False, "nonmetal": True},
    "Si": {"Z": 14, "val": 4, "rad": 1.17, "hard": False, "mag": False, "nonmetal": True},
    "P":  {"Z": 15, "val": 5, "rad": 1.06, "hard": False, "mag": False, "nonmetal": True},
    "S":  {"Z": 16, "val": 6, "rad": 1.02, "hard": False, "mag": False, "nonmetal": True},

    # 3d Transition Metals
    "Sc": {"Z": 21, "val": 3, "rad": 1.62, "hard": False, "mag": False, "nonmetal": False},
    "Ti": {"Z": 22, "val": 4, "rad": 1.47, "hard": False, "mag": False, "nonmetal": False},
    "V":  {"Z": 23, "val": 5, "rad": 1.34, "hard": False, "mag": False, "nonmetal": False},
    "Cr": {"Z": 24, "val": 6, "rad": 1.28, "hard": False, "mag": True,  "nonmetal": False},
    "Mn": {"Z": 25, "val": 7, "rad": 1.32, "hard": False, "mag": True,  "nonmetal": False},
    "Fe": {"Z": 26, "val": 8, "rad": 1.26, "hard": False, "mag": True,  "nonmetal": False},
    "Co": {"Z": 27, "val": 9, "rad": 1.25, "hard": False, "mag": True,  "nonmetal": False},
    "Ni": {"Z": 28, "val": 10,"rad": 1.24, "hard": False, "mag": True,  "nonmetal": False},
    "Cu": {"Z": 29, "val": 11,"rad": 1.28, "hard": False, "mag": False, "nonmetal": False},
    "Zn": {"Z": 30, "val": 12,"rad": 1.34, "hard": False, "mag": False, "nonmetal": False},

    # 4d Transition Metals
    "Y":  {"Z": 39, "val": 3, "rad": 1.80, "hard": False, "mag": False, "nonmetal": False},
    "Zr": {"Z": 40, "val": 4, "rad": 1.60, "hard": False, "mag": False, "nonmetal": False},
    "Nb": {"Z": 41, "val": 5, "rad": 1.46, "hard": False, "mag": False, "nonmetal": False},
    "Mo": {"Z": 42, "val": 6, "rad": 1.39, "hard": False, "mag": False, "nonmetal": False},
    "Tc": {"Z": 43, "val": 7, "rad": 1.36, "hard": False, "mag": False, "nonmetal": False},
    "Ru": {"Z": 44, "val": 8, "rad": 1.34, "hard": False, "mag": False, "nonmetal": False},
    "Rh": {"Z": 45, "val": 9, "rad": 1.34, "hard": False, "mag": False, "nonmetal": False},
    "Pd": {"Z": 46, "val": 10,"rad": 1.37, "hard": False, "mag": False, "nonmetal": False},
    "Ag": {"Z": 47, "val": 11,"rad": 1.44, "hard": False, "mag": False, "nonmetal": False},
    "Cd": {"Z": 48, "val": 12,"rad": 1.51, "hard": False, "mag": False, "nonmetal": False},

    # 5d Transition Metals & Others
    "Hf": {"Z": 72, "val": 4, "rad": 1.59, "hard": False, "mag": False, "nonmetal": False},
    "Ta": {"Z": 73, "val": 5, "rad": 1.46, "hard": False, "mag": False, "nonmetal": False},
    "W":  {"Z": 74, "val": 6, "rad": 1.39, "hard": False, "mag": False, "nonmetal": False},
    "Re": {"Z": 75, "val": 7, "rad": 1.37, "hard": False, "mag": False, "nonmetal": False},
    "Os": {"Z": 76, "val": 8, "rad": 1.35, "hard": False, "mag": False, "nonmetal": False},
    "Ir": {"Z": 77, "val": 9, "rad": 1.36, "hard": False, "mag": False, "nonmetal": False},
    "Pt": {"Z": 78, "val": 10,"rad": 1.39, "hard": False, "mag": False, "nonmetal": False},
    "Au": {"Z": 79, "val": 11,"rad": 1.44, "hard": False, "mag": False, "nonmetal": False},
    "Al": {"Z": 13, "val": 3, "rad": 1.43, "hard": False, "mag": False, "nonmetal": False},
    "Pb": {"Z": 82, "val": 4, "rad": 1.75, "hard": False, "mag": False, "nonmetal": False},
}

# ── CASTEP Specific ──────────────────────────────────────────────────────────
TASKS_FULL: list[str] = ["GeometryOptimization", "SinglePoint", "ElasticConstants"]
TASKS_VCA: list[str] = ["GeometryOptimization", "SinglePoint"]

XC_LIST: list[str] = ["PBE", "PBESOL", "LDA"]
XC_DEFAULT: str = "PBE"

ENCUT_SOFT: int = 500
ENCUT_HARD: int = 700

MAX_SCF: int = 150
METALS_METHOD: str = "dm"
MIXING_SCHEME: str = "Pulay"
MIX_AMP_GEOM: float = 0.50
MIX_AMP_SP: float = 0.20
ELEC_TOL_GEOM: str = "1.0e-6 eV"
ELEC_TOL_SP: str = "1.0e-7 eV"

GEOM_MAX_ITER: int = 150
GEOM_E_TOL: str = "1.0e-6 eV"
GEOM_F_TOL: str = "0.005 eV/ang"
GEOM_S_TOL: str = "0.03 GPa"
GEOM_D_TOL: str = "0.0005 ang"
FINITE_BASIS: int = 0

CASTEP_CLEANUP_GLOBS: list[str] = ["*.check", "*.bib", "*.bands", "*.cst_esp", "*.err", "*.usp", "*.cst_esp"]
CASTEP_SEARCH_PATHS: list[str] = [
    "~/Applications/CASTEP*/bin/*/castep.mpi",
    "~/Applications/CASTEP*/bin/*/castep",
    "/opt/CASTEP*/bin/*/castep.mpi",
    "/opt/CASTEP*/bin/*/castep",
    "/usr/local/bin/castep.mpi",
    "castep.mpi",
    "castep"
]

# ── VASP Specific ────────────────────────────────────────────────────────────

# INCAR Parameters
KSPACING: float = 0.04
EDIFF_IBRION6: str = "1E-7"
EDIFF_GEOM: str = "1E-5"
NSW_MAX_VASP: int = 300
EDIFFG_VASP: str = "-0.01"
IBRION_GEOM: int = 2
ISIF: int = 3
ISMEAR: int = 1

# VASP Auto-Discovery Paths
VASP_SEARCH_PATHS: list[str] = [
    "~/Applications/vasp*/bin/vasp_std",
    "~/Applications/vasp*/vasp*/bin/vasp_std",
    "~/vasp*/bin/vasp_std",
    "/opt/vasp*/bin/vasp_std",
    "/usr/local/bin/vasp_std",
    "vasp_std"
]

VASPKIT_SEARCH_PATHS: list[str] = [
    "~/Applications/vaspkit*/vaspkit",
    "~/vaspkit/vaspkit",
    "/opt/vaspkit/vaspkit",
]

POTCAR_SEARCH_PATHS: list[str] = [
    "~/Applications/vasp.*/pseudopotentials/PAW_PBE",
    "~/vasp/pp/PAW_PBE",
    "/opt/vasp/potcars/PAW_PBE",
    "/usr/local/vasp/pp/PAW_PBE",
]

POTCAR_PREFERRED: dict[str, str] = {
    "Ti": "Ti_sv", "Zr": "Zr_sv", "Hf": "Hf_sv", "V": "V_pv", "Nb": "Nb_pv", "Ta": "Ta_pv",
    "Cr": "Cr_pv", "Mo": "Mo_pv", "W": "W", "Mn": "Mn_pv", "Re": "Re_pv",
    "Fe": "Fe", "Co": "Co", "Ni": "Ni", "Cu": "Cu", "Ru": "Ru_pv", "Os": "Os_pv",
    "Rh": "Rh_pv", "Ir": "Ir", "Pd": "Pd", "Pt": "Pt", "Au": "Au", "Ag": "Ag",
}

VASP_CLEANUP_GLOBS: list[str] = ["WAVECAR", "CHG", "CHGCAR", "PROCAR", "DOSCAR", "PCDAT", "XDATCAR", "IBZKPT"]

# ── Process Watchdog Limits ──────────────────────────────────────────────────
STEP_TIMEOUT_S: int = 18000
SMAX_KILL_GPa: float = 50.0
SMAX_STALL_ITERS: int = 15
