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

# ── Elastic finite-strain workflow ────────────────────────────────────────────
ELASTIC_MAX_STRAIN: float = 0.003  # ±0.3 % — linear regime for TMC/TMN
ELASTIC_N_STEPS: int = 3  # positive magnitudes per pattern
ELASTIC_NEXTRA_PURE: int = 10  # nextra_bands for pure endpoints (x=0,1)
ELASTIC_NEXTRA_BASE: int = 15  # nextra_bands base for VCA intermediates

# ── Smearing widths (eV) ──────────────────────────────────────────────────────
# Empirical optimum from R² benchmarking, Ti-Nb carbide/nitride series.
SMEARING_VCA: float = 0.20  # fractional nuclear charge broadens bands
SMEARING_SINGLE: float = 0.10  # pure compound: sharp Fermi edge

# ── Watchdog: step timeout ────────────────────────────────────────────────────
# Hard wall-clock limit per CASTEP step (seconds).
# A fast GeomOpt on 6 cores takes ~1 min; elastic SinglePoints ~25 s each.
# 15 minutes is generous for well-behaved systems and catches the pathological
# case shown in TiC_base.castep (ran for 3600 s without converging).
STEP_TIMEOUT_S: float = 900.0  # 15 minutes

# ── Watchdog: Smax stall detection ───────────────────────────────────────────
# If the max-stress component (Smax, GPa) stays above SMAX_KILL_GPa for
# SMAX_STALL_ITERS consecutive completed LBFGS iterations, the process is
# killed.  This catches the stress-oscillation pattern where Smax bounces
# between 10–270 GPa for 30+ iterations (see TiC_base.castep).
#
# Calibration:
#   SMAX_KILL_GPa:   50 GPa — well above the 0.03 GPa convergence tolerance
#                    but below the ~14 GPa initial stress that is recoverable
#                    in a few LBFGS steps with a good geometry.
#   SMAX_STALL_ITERS: 8 — eight consecutive failures (≈ 5–10 min) before kill.
SMAX_KILL_GPa: float = 50.0
SMAX_STALL_ITERS: int = 4

# ── CIF → primitive cell reduction ───────────────────────────────────────────
CELL_AUTO_REDUCE: bool = True
CELL_LENGTH_TOL: float = 1e-3  # Angstrom
CELL_COORD_TOL: float = 1e-3  # fractional
CELL_NIGGLI_EPS: float = 1e-5
