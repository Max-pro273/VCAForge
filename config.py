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
    "H": {"Z": 1,  "val": 1, "rad": 0.53, "en": 2.2, "hard": True,  "mag": False, "nonmetal": True},
    "B": {"Z": 5,  "val": 3, "rad": 0.87, "en": 2.04, "hard": True,  "mag": False, "nonmetal": True},
    "C": {"Z": 6,  "val": 4, "rad": 0.77, "en": 2.55, "hard": True,  "mag": False, "nonmetal": True},
    "N": {"Z": 7,  "val": 5, "rad": 0.75, "en": 3.04, "hard": True,  "mag": False, "nonmetal": True},
    "O": {"Z": 8,  "val": 6, "rad": 0.73, "en": 3.44, "hard": True,  "mag": False, "nonmetal": True},
    "F": {"Z": 9,  "val": 7, "rad": 0.60, "en": 3.98, "hard": True,  "mag": False, "nonmetal": True},
    "Si": {"Z": 14, "val": 4, "rad": 1.17, "en": 1.9, "hard": False, "mag": False, "nonmetal": True},
    "P": {"Z": 15, "val": 5, "rad": 1.06, "en": 2.19, "hard": False, "mag": False, "nonmetal": True},
    "S": {"Z": 16, "val": 6, "rad": 1.02, "en": 2.58, "hard": False, "mag": False, "nonmetal": True},

    # 3d Transition Metals
    "Sc": {"Z": 21, "val": 3, "rad": 1.62, "en": 1.36, "hard": False, "mag": False, "nonmetal": False},
    "Ti": {"Z": 22, "val": 4, "rad": 1.47, "en": 1.54, "hard": False, "mag": False, "nonmetal": False},
    "V": {"Z": 23, "val": 5, "rad": 1.34, "en": 1.63, "hard": False, "mag": False, "nonmetal": False},
    "Cr": {"Z": 24, "val": 6, "rad": 1.28, "en": 1.66, "hard": False, "mag": True,  "nonmetal": False},
    "Mn": {"Z": 25, "val": 7, "rad": 1.32, "en": 1.55, "hard": False, "mag": True,  "nonmetal": False},
    "Fe": {"Z": 26, "val": 8, "rad": 1.26, "en": 1.83, "hard": False, "mag": True,  "nonmetal": False},
    "Co": {"Z": 27, "val": 9, "rad": 1.25, "en": 1.88, "hard": False, "mag": True,  "nonmetal": False},
    "Ni": {"Z": 28, "val": 10,"rad": 1.24, "en": 1.91, "hard": False, "mag": True,  "nonmetal": False},
    "Cu": {"Z": 29, "val": 11,"rad": 1.28, "en": 1.9, "hard": False, "mag": False, "nonmetal": False},
    "Zn": {"Z": 30, "val": 12,"rad": 1.34, "en": 1.65, "hard": False, "mag": False, "nonmetal": False},

    # 4d Transition Metals
    "Y": {"Z": 39, "val": 3, "rad": 1.80, "en": 1.22, "hard": False, "mag": False, "nonmetal": False},
    "Zr": {"Z": 40, "val": 4, "rad": 1.60, "en": 1.33, "hard": False, "mag": False, "nonmetal": False},
    "Nb": {"Z": 41, "val": 5, "rad": 1.46, "en": 1.6, "hard": False, "mag": False, "nonmetal": False},
    "Mo": {"Z": 42, "val": 6, "rad": 1.39, "en": 2.16, "hard": False, "mag": False, "nonmetal": False},
    "Tc": {"Z": 43, "val": 7, "rad": 1.36, "en": 1.9, "hard": False, "mag": False, "nonmetal": False},
    "Ru": {"Z": 44, "val": 8, "rad": 1.34, "en": 2.2, "hard": False, "mag": False, "nonmetal": False},
    "Rh": {"Z": 45, "val": 9, "rad": 1.34, "en": 2.28, "hard": False, "mag": False, "nonmetal": False},
    "Pd": {"Z": 46, "val": 10,"rad": 1.37, "en": 2.2, "hard": False, "mag": False, "nonmetal": False},
    "Ag": {"Z": 47, "val": 11,"rad": 1.44, "en": 1.93, "hard": False, "mag": False, "nonmetal": False},
    "Cd": {"Z": 48, "val": 12,"rad": 1.51, "en": 1.69, "hard": False, "mag": False, "nonmetal": False},

    # 5d Transition Metals & Others
    "Hf": {"Z": 72, "val": 4, "rad": 1.59, "en": 1.3, "hard": False, "mag": False, "nonmetal": False},
    "Ta": {"Z": 73, "val": 5, "rad": 1.46, "en": 1.5, "hard": False, "mag": False, "nonmetal": False},
    "W": {"Z": 74, "val": 6, "rad": 1.39, "en": 2.36, "hard": False, "mag": False, "nonmetal": False},
    "Re": {"Z": 75, "val": 7, "rad": 1.37, "en": 1.9, "hard": False, "mag": False, "nonmetal": False},
    "Os": {"Z": 76, "val": 8, "rad": 1.35, "en": 2.2, "hard": False, "mag": False, "nonmetal": False},
    "Ir": {"Z": 77, "val": 9, "rad": 1.36, "en": 2.2, "hard": False, "mag": False, "nonmetal": False},
    "Pt": {"Z": 78, "val": 10,"rad": 1.39, "en": 2.28, "hard": False, "mag": False, "nonmetal": False},
    "Au": {"Z": 79, "val": 11,"rad": 1.44, "en": 2.54, "hard": False, "mag": False, "nonmetal": False},
    "Al": {"Z": 13, "val": 3, "rad": 1.43, "en": 1.61, "hard": False, "mag": False, "nonmetal": False},
    "Pb": {"Z": 82, "val": 4, "rad": 1.75, "en": 2.33, "hard": False, "mag": False, "nonmetal": False},
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

# ── Navigator (Bayesian Optimization) ────────────────────────────────────────
NAVIGATOR_TARGET: str = "H_Vickers_GPa"
NAVIGATOR_MODE: str = "maximize"          # "maximize" | "minimize"
NAVIGATOR_ACQUISITION: str = "CEI"        # "CEI" | "EI" | "UCB" | "MaxVar"
NAVIGATOR_N_STARTS: int = 20
NAVIGATOR_MIN_POINTS: int = 5
NAVIGATOR_UCB_KAPPA: float = 2.0
NAVIGATOR_EI_THRESHOLD: float = 0.01
NAVIGATOR_KERNEL: str = "auto"            # "auto" | "matern" | "rbf"

# Descriptor space — when False, navigator runs in legacy 1D-on-x mode.
# When True (default), GP fits on physics descriptors enabling
# transfer learning between systems.
NAVIGATOR_USE_DESCRIPTORS: bool = True

# Descriptor mode (N-component support):
#   "full"          — [x_1..x_{N-1}, VEC, r_avg, Z_avg, EN_avg, δr, δEN, nm_VEC]
#                     Recommended. Both composition AND physics. Preserves
#                     identifiability (no two distinct compositions map to same d).
#   "physics_only"  — [VEC, r_avg, Z_avg, EN_avg, δr, δEN, nm_VEC] only.
#                     Useful for studying pure transfer learning across systems
#                     of different N. WARNING: for fixed system, the GP cannot
#                     distinguish nearby x-vectors; use only with care.
NAVIGATOR_DESCRIPTOR_MODE: str = "full"

# Active-learning loop
NAVIGATOR_LOOP_MAX_ITER: int = 10         # safety cap on auto-loop iterations
NAVIGATOR_LOOP_DEDUPE_TOL: float = 1e-2   # ‖x_new - x_existing‖ < tol → duplicate.
# Note: must be ≥ 2e-3 in practice. VCAForge rounds 'concentration' to 4 decimals
# and the bracket-fraction reconstruction (concentration × inner_frac) means the
# composition stored in the CSV can differ from what the navigator suggested by
# up to ~5e-3. Below 1e-2 the loop will falsely re-suggest already-evaluated points.
NAVIGATOR_LOOP_MPI_PROCS: int = 6         # 0 = ask interactively (passes -np to VCAForge if its CLI accepts it)
NAVIGATOR_LOOP_RUN_ELASTIC: bool = True   # default for the elastic prompt
NAVIGATOR_LOOP_SCAN_PARENT: bool = True   # auto-scan structure_file.parent if no --dir given

# Aggregator
NAVIGATOR_MASTER_CSV_SUFFIX: str = "_master.csv"
NAVIGATOR_AGGREGATE_ON_ITER: bool = True  # update master CSV after each loop iteration

# N-component simplex
NAVIGATOR_SIMPLEX_ALPHA: float = 1.0      # Dirichlet α (1.0 = uniform on simplex)
NAVIGATOR_MIN_X_COMPONENT: float = 0.0    # minimum fraction per metal (0 = pure end-members allowed)

# Convergence / completeness criteria for "is the study done?"
NAVIGATOR_DONE_MIN_POINTS: int = 8        # need at least this many stable points
NAVIGATOR_DONE_PEAK_SIGMA_MAX: float = 1.0  # σ at predicted optimum < this → confident
NAVIGATOR_DONE_DEDUPE_REPEATS: int = 3    # N consecutive duplicate suggestions → done

# Multi-fidelity (XC-functional ladder)
# Use FidelityLevel objects from navigator.py if you want a custom ladder.
# Set to None to use DEFAULT_FIDELITY_LADDER (LDA → PBESOL → PBE).
NAVIGATOR_FIDELITY_LEVELS = None

# ── MLIP Engine ──────────────────────────────────────────────────────────────
MLIP_DEFAULT_BACKEND: str = "mace"
MLIP_FMAX: float = 0.01            # eV/Å convergence for ASE BFGS
MLIP_RELAX_STEPS: int = 300        # max ionic steps
MLIP_ELASTIC_DELTA: float = 0.003  # Voigt strain delta (matches ELASTIC_MAX_STRAIN)
MLIP_DEVICE: str = "cpu"           # "cpu" | "cuda" | "mps"
MLIP_CACHE_PATHS: list[str] = [    # auto-discovery of model files
    "~/.cache/mace/*",
    "~/.cache/mace/*",
    "~/.local/share/mace/*",
]
MLIP_SUBDIR: str = "MLIP"
