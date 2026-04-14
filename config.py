"""
config.py  —  VCAForge global configuration.

Edit this file to tune any parameter without touching program logic.
CASTEP-specific settings live in castep/config.py.
"""

# ── Version ───────────────────────────────────────────────────────────────────
VERSION: str = "6.0"
# ── Output layout ─────────────────────────────────────────────────────────────
# Sub-directory inside each project folder where CASTEP job folders live.
#   MyRun_Apr12/CASTEP/x0.2500/TiC.castep
CASTEP_SUBDIR: str = "CASTEP"

STATE_FILE: str = "vca_state.json"
CSV_FILE: str = "vca_results.csv"
LOG_FILE: str = "run.log"

# ── VCA sweep ─────────────────────────────────────────────────────────────────
SWEEP_C_START: float = 0.0
SWEEP_C_END: float = 1.0
SWEEP_N_DEFAULT: int = 8  # default number of intervals

# ── VEC stability thresholds (Mei et al. 2014, Ti(1-x)Nb(x)C) ────────────────
VEC_YELLOW: float = 8.2  # marginal zone — extra SCF iterations often needed
VEC_RED: float = 8.4  # Born instability zone — C44 likely negative

# ── Elastic finite-strain workflow ────────────────────────────────────────────
ELASTIC_MAX_STRAIN: float = 0.003  # ±0.3 % — linear regime for TMC/TMN
ELASTIC_N_STEPS: int = 3  # positive magnitudes per pattern
ELASTIC_NEXTRA_PURE: int = 10  # nextra_bands for pure endpoints (x=0,1)
ELASTIC_NEXTRA_BASE: int = 15  # nextra_bands base for VCA intermediates

# ── Smearing widths (eV) ──────────────────────────────────────────────────────
# Empirical optimum from R² benchmarking, Ti-Nb carbide/nitride series.
SMEARING_VCA: float = 0.20  # fractional nuclear charge broadens bands
SMEARING_SINGLE: float = 0.10  # pure compound: sharp Fermi edge

# ── CIF → primitive cell reduction ───────────────────────────────────────────
CELL_AUTO_REDUCE: bool = True
CELL_LENGTH_TOL: float = 1e-3  # Angstrom
CELL_COORD_TOL: float = 1e-3  # fractional
CELL_NIGGLI_EPS: float = 1e-5
