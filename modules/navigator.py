"""
navigator.py  —  Bayesian Optimization for VCAForge with N-component support.
═════════════════════════════════════════════════════════════════════════════
Reads completed VCA sweeps, fits twin Gaussian Processes (regressor +
classifier) over a physics-aware descriptor space, and proposes the next
composition to evaluate. Handles binary, ternary, quaternary, and beyond
through a single unified architecture.

Architecture
────────────
SystemSpec         → list of metals + nonmetal. Generalizes from binary.
DescriptorBuilder  → maps (system, x-vector) → descriptor in R^d.
                     "full" mode: [x_1..x_{N-1}, VEC, r_avg, Z_avg, EN_avg,
                                   δr, δEN, nm_VEC] — preserves identifiability.
                     "physics_only" mode: drops the x-prefix (transfer-learning
                                   research; use with care for fixed systems).
DataIngestor       → loads CSVs, parses 'metals=' or '# System :' header,
                     reads either {x_Ti, x_Nb, ...} columns OR legacy
                     'concentration' column, classifies stable/unstable.
SurrogateModel     → twin GPs in descriptor space.
AcquisitionFunction→ EI / CEI / UCB / MaxVar.
AcquisitionOptimizer → multi-start SLSQP on the (N-1)-simplex
                     {x ∈ R^N : x_i ≥ 0, Σ x_i = 1}, Dirichlet seeding.
NavigatorOrchestrator → public façade for CLI and main.py.

Backward compatibility
──────────────────────
Existing binary CSV files (concentration column, '# System : Ti(1-x)Nb(x)C'
header) work unchanged. Existing CLI commands (--system Ti-Nb-C, etc.)
work unchanged. Candidate.concentration is preserved for binary; for N≥3
the new field Candidate.composition (dict[str, float]) holds the full
composition vector.

Critical design choices
───────────────────────
1.  "full" descriptor mode includes [x_1..x_{N-1}] in front of physics
    fields. This guarantees identifiability — two distinct compositions
    cannot map to the same descriptor (which would corrupt the GP).
    The physics fields enable transfer learning between systems while
    the x-prefix discriminates within a system.

2.  Vegard-interpolated points NEVER train the GP (DataIngestor filter).

3.  Born-stability safety net catches orchestrator bugs (C12 > C11 etc.).

4.  Optimization happens on the simplex of the TARGET system only. The
    GP is fitted on data from ALL systems (transfer learning), but the
    acquisition optimizer fixes the metal set and varies only x.

Dependencies: numpy, pandas, scipy, scikit-learn.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    Matern,
    WhiteKernel,
)

_root_dir = Path(__file__).resolve().parent.parent
if str(_root_dir) not in sys.path:
    sys.path.insert(0, str(_root_dir))
import config

log = logging.getLogger("vcaforge.navigator")

# ─────────────────────────────────────────────────────────────────────────────
# Config accessors  (getattr with fallback — backward compatible)
# ─────────────────────────────────────────────────────────────────────────────

def _cfg(name: str, default: Any) -> Any:
    return getattr(config, name, default)


def _navigator_target() -> str:        return _cfg("NAVIGATOR_TARGET", "H_Vickers_GPa")
def _navigator_mode() -> str:          return _cfg("NAVIGATOR_MODE", "maximize")
def _navigator_acquisition() -> str:   return _cfg("NAVIGATOR_ACQUISITION", "CEI")
def _navigator_n_starts() -> int:      return int(_cfg("NAVIGATOR_N_STARTS", 20))
def _navigator_min_points() -> int:    return int(_cfg("NAVIGATOR_MIN_POINTS", 5))
def _navigator_ucb_kappa() -> float:   return float(_cfg("NAVIGATOR_UCB_KAPPA", 2.0))
def _navigator_ei_threshold() -> float: return float(_cfg("NAVIGATOR_EI_THRESHOLD", 0.01))
def _navigator_kernel() -> str:        return _cfg("NAVIGATOR_KERNEL", "auto")
def _use_descriptors() -> bool:        return bool(_cfg("NAVIGATOR_USE_DESCRIPTORS", True))
def _descriptor_mode() -> str:         return _cfg("NAVIGATOR_DESCRIPTOR_MODE", "full")
def _loop_max_iter() -> int:           return int(_cfg("NAVIGATOR_LOOP_MAX_ITER", 10))
def _loop_dedupe_tol() -> float:       return float(_cfg("NAVIGATOR_LOOP_DEDUPE_TOL", 1e-3))
def _simplex_alpha() -> float:         return float(_cfg("NAVIGATOR_SIMPLEX_ALPHA", 1.0))
def _min_x_component() -> float:       return float(_cfg("NAVIGATOR_MIN_X_COMPONENT", 0.0))
def _master_suffix() -> str:           return _cfg("NAVIGATOR_MASTER_CSV_SUFFIX", "_master.csv")
def _done_min_points() -> int:         return int(_cfg("NAVIGATOR_DONE_MIN_POINTS", 8))
def _done_peak_sigma_max() -> float:   return float(_cfg("NAVIGATOR_DONE_PEAK_SIGMA_MAX", 1.0))
def _done_dedupe_repeats() -> int:     return int(_cfg("NAVIGATOR_DONE_DEDUPE_REPEATS", 3))
def _csv_filename() -> str:            return _cfg("CSV_FILE", "vca_results.csv")
def _elements() -> dict[str, dict[str, Any]]: return _cfg("ELEMENTS", {})


# ─────────────────────────────────────────────────────────────────────────────
# SystemSpec — generalised to N metals
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SystemSpec:
    """Defines an N-metal alloy + optional nonmetal: M_1(x_1) M_2(x_2)... [N].

    For binary (N=2), `metals` has 2 entries and the legacy `metal_a`/`metal_b`
    properties are computed from `metals`. For ternary, len(metals) == 3, etc.
    The set is order-sensitive: x-vectors are interpreted in the same order.

    Frozen → usable as a dict key for grouping data by system.
    """
    metals: tuple[str, ...]
    nonmetal: str = ""

    def __post_init__(self) -> None:
        if len(self.metals) < 1:
            raise ValueError(
                f"SystemSpec needs at least 1 metal, got {len(self.metals)}: {self.metals}"
            )

    @property
    def n_metals(self) -> int:
        return len(self.metals)

    @property
    def n_dims(self) -> int:
        """Free dimensions on the simplex (N-1 because Σx=1)."""
        return max(0, self.n_metals - 1)

    @property
    def metal_a(self) -> str:
        """Backward-compat: first metal (binary's M_a)."""
        return self.metals[0]

    @property
    def metal_b(self) -> str:
        """Backward-compat: second metal (binary's M_b)."""
        if self.n_metals < 2:
            return ""
        return self.metals[1]

    def label(self) -> str:
        """Compact label like 'Zr-Nb-C' or 'Ti-Nb-V-Zr-C'."""
        if self.nonmetal:
            return "-".join(self.metals) + f"-{self.nonmetal}"
        return "-".join(self.metals)

    def cli_species_args(self) -> list[str]:
        """For binary VCAForge (legacy): ['Ti', 'Nb']."""
        return list(self.metals)

    def is_binary(self) -> bool:
        return self.n_metals == 2

    def column_names(self) -> list[str]:
        """Per-metal concentration column names: ['x_Ti', 'x_Nb', 'x_Zr']."""
        return [f"x_{m}" for m in self.metals]


# ─────────────────────────────────────────────────────────────────────────────
# Candidate / NavigatorReport
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Candidate:
    """A single proposed evaluation point.

    For binary alloys, `concentration` (float) holds x_2 (fraction of M_b)
    — preserves the legacy API. For N≥3, `composition` (dict) is the
    authoritative field; `concentration` carries x_2 for backward-compat
    display but should not be relied on alone.
    """
    concentration: float                                 # legacy: x_2 for binary
    predicted_mean: float
    predicted_std: float
    p_feasible: float | None
    acq_value: float | None
    converged: bool = False
    source: str = "GP-CEI"
    system: SystemSpec | None = None
    descriptor_vector: tuple[float, ...] | None = None
    composition: dict[str, float] | None = None          # NEW: full vector for N≥2

    def composition_str(self, precision: int = 4) -> str:
        """Human-readable composition like 'Ti=0.6000 Nb=0.3000 V=0.1000'."""
        if self.composition:
            return " ".join(f"{m}={x:.{precision}f}" for m, x in self.composition.items())
        return f"x={self.concentration:.{precision}f}"


@dataclass
class NavigatorReport:
    """Full diagnostic report from a single suggest() call."""
    n_stable: int
    n_failed: int
    n_total: int
    current_best_value: float
    current_best_x: float                                # legacy: best x_2 for binary
    selected_kernel: str
    candidate: Candidate
    target: str = ""
    mode: str = ""
    acquisition: str = ""
    target_system: SystemSpec | None = None
    systems_in_data: list[str] = field(default_factory=list)
    descriptor_mode: bool = False
    current_best_composition: dict[str, float] | None = None  # NEW: for N≥3


# ─────────────────────────────────────────────────────────────────────────────
# DescriptorBuilder — N-component physics descriptor
# ─────────────────────────────────────────────────────────────────────────────

# Physics-only descriptor names (always present, regardless of mode/system size)
PHYSICS_DESC_NAMES: tuple[str, ...] = (
    "VEC", "r_avg", "Z_avg", "EN_avg", "delta_r", "delta_EN", "nm_VEC",
)
N_PHYSICS_DESC = len(PHYSICS_DESC_NAMES)

# Normalisation scales (chosen to keep each component in roughly [0, 1])
_SCALE_VEC      = 12.0   # max metal valence in transition rows
_SCALE_R        = 2.0    # max ~1.8 Å (Y)
_SCALE_Z        = 80.0   # 1..79 covers H..Au
_SCALE_EN       = 4.0    # Pauling 0..4
_SCALE_DELTA_R  = 0.3    # δr typically 0..0.2 even for highly mismatched sets
_SCALE_DELTA_EN = 1.5    # δEN rarely > 1.0 for transition metals
_SCALE_NM       = 7.0    # F = 7


class DescriptorBuilder:
    """Maps (system, x-vector) to a physics-aware descriptor in R^d.

    Modes
    ─────
    "full" (default, recommended):
        d = [x_1, x_2, ..., x_{N-1},  VEC, r_avg, Z_avg, EN_avg,
             δr, δEN, nm_VEC]
        Length = (N-1) + 7. The (N-1)-prefix preserves identifiability:
        no two distinct compositions on the simplex map to the same
        descriptor. The physics suffix is shared across systems of any N
        and enables transfer learning.

    "physics_only":
        d = [VEC, r_avg, Z_avg, EN_avg, δr, δEN, nm_VEC]   (length 7)
        WARNING: for any fixed system, Σ x_i = 1 means the physics fields
        often vary slowly and several distinct x-vectors can land near the
        same d. The GP will treat them as noise. Use only for cross-system
        transfer-learning experiments.

    Physics field definitions (over the metal sublattice)
    ─────────────────────────────────────────────────────
        VEC       = Σ x_i · valence(M_i)
        r_avg     = Σ x_i · radius(M_i)        (Vegard's law)
        Z_avg     = Σ x_i · Z(M_i)
        EN_avg    = Σ x_i · EN(M_i)            (Pauling)
        δr        = √( Σ x_i · (1 − r(M_i)/r_avg)² )      (atomic-size mismatch)
        δEN       = √( Σ x_i · (EN(M_i) − EN_avg)² )      (electronegativity mismatch)
        nm_VEC    = valence(nonmetal)          (system constant; 0 if no nonmetal)
    """

    def __init__(self, elements: dict[str, dict[str, Any]] | None = None) -> None:
        self.elements = elements if elements is not None else _elements()

    # ── Public API ─────────────────────────────────────────────────────────

    def build(self, system: SystemSpec, x_vector: np.ndarray | list[float]) -> np.ndarray:
        """Returns the descriptor vector for (system, x_vector).

        x_vector : array-like of shape (N,) where N = system.n_metals.
                   Must satisfy x_i ≥ 0 and Σ x_i = 1 (within 1e-6 tolerance).
        """
        x = np.asarray(x_vector, dtype=float).ravel()
        self._validate_x(system, x)
        physics = self._physics_features(system, x)
        if _descriptor_mode() == "physics_only":
            return physics
        # "full" mode: prepend free composition coords (all but last)
        free_x = x[:-1]   # x_N is the dependent coordinate
        return np.concatenate([free_x, physics])

    def descriptor_length(self, system: SystemSpec) -> int:
        """Length of descriptor vector for this system in current mode."""
        if _descriptor_mode() == "physics_only":
            return N_PHYSICS_DESC
        return (system.n_metals - 1) + N_PHYSICS_DESC

    def descriptor_names(self, system: SystemSpec) -> list[str]:
        """Human-readable names matching descriptor_length()."""
        physics = list(PHYSICS_DESC_NAMES)
        if _descriptor_mode() == "physics_only":
            return physics
        x_names = [f"x_{m}" for m in system.metals[:-1]]
        return x_names + physics

    # ── Internals ─────────────────────────────────────────────────────────

    def _validate_x(self, system: SystemSpec, x: np.ndarray) -> None:
        if x.shape != (system.n_metals,):
            raise ValueError(
                f"x_vector has shape {x.shape}, expected ({system.n_metals},) "
                f"for system {system.label()}"
            )
        if np.any(x < -1e-9):
            raise ValueError(f"x_vector has negative components: {x}")
        s = float(x.sum())
        if not (0.99 < s < 1.01):
            raise ValueError(
                f"x_vector must sum to 1.0 (within 1e-2), got Σ={s:.6f}: {x}"
            )

    def _physics_features(self, system: SystemSpec, x: np.ndarray) -> np.ndarray:
        # Pull element data for each metal
        vals  = np.array([self._lookup(m).get("val", 0)  for m in system.metals], dtype=float)
        rads  = np.array([self._lookup(m).get("rad", 0.0) for m in system.metals], dtype=float)
        zs    = np.array([self._lookup(m).get("Z", 0)   for m in system.metals], dtype=float)
        ens   = np.array([self._lookup(m).get("en", 0.0) for m in system.metals], dtype=float)

        VEC    = float(np.sum(x * vals))
        r_avg  = float(np.sum(x * rads))
        Z_avg  = float(np.sum(x * zs))
        EN_avg = float(np.sum(x * ens))

        # Atomic size mismatch (Yang-Zhang δ parameter, dimensionless)
        if r_avg > 1e-9:
            delta_r = float(np.sqrt(np.sum(x * (1.0 - rads / r_avg) ** 2)))
        else:
            delta_r = 0.0
        # Electronegativity mismatch (analogous to Yang-Zhang)
        delta_EN = float(np.sqrt(np.sum(x * (ens - EN_avg) ** 2)))

        nm_val = 0.0
        if system.nonmetal:
            nm_val = float(self._lookup(system.nonmetal).get("val", 0))

        return np.array([
            VEC      / _SCALE_VEC,
            r_avg    / _SCALE_R,
            Z_avg    / _SCALE_Z,
            EN_avg   / _SCALE_EN,
            delta_r  / _SCALE_DELTA_R,
            delta_EN / _SCALE_DELTA_EN,
            nm_val   / _SCALE_NM,
        ], dtype=float)

    def _lookup(self, sym: str) -> dict[str, Any]:
        if sym not in self.elements:
            raise ValueError(
                f"Element '{sym}' not in config.ELEMENTS. "
                f"Add it (with Z, val, rad, en) before using this system."
            )
        return self.elements[sym]


# ─────────────────────────────────────────────────────────────────────────────
# System parsing — handles both binary and N-component CSV headers
# ─────────────────────────────────────────────────────────────────────────────

# Binary VCAForge format (existing, keep working):
#   "# System  : Zr(1-x)Nb(x)C"
# N-component proposed format (new):
#   "# System  : Ti-Nb-Zr-C  metals=Ti,Nb,Zr  nonmetal=C"
#   OR just "# System  : Ti-Nb-Zr-C" with metals inferred from dash-split.
# Multi-component (skip — VCA approximation, not VCAForge):
#   "# System  : Ti(1-x)[Nb0.50Zr0.50](x)C"
_SYSTEM_LINE_RE   = re.compile(r"^\s*#\s*System\s*:\s*(.+?)\s*$")
_BINARY_VCA_RE    = re.compile(r"^([A-Z][a-z]?)\(1-x\)([A-Z][a-z]?)\(x\)([A-Z][a-z]?)?\s*$")
_METALS_KV_RE     = re.compile(r"metals\s*=\s*([A-Za-z][A-Za-z,]*)", re.IGNORECASE)
_NONMETAL_KV_RE   = re.compile(r"nonmetal\s*=\s*([A-Z][a-z]?)", re.IGNORECASE)
_DASH_SYSTEM_RE   = re.compile(r"^([A-Z][a-z]?)((?:-[A-Z][a-z]?)+)\s*$")

# Bracketed VCA mix form (used by VCAForge for ternary+ alloys with a single
# `concentration` column):  "Ti(1-x)[Zr0.33Nb0.67](x)C"
# Group 1: pure metal at (1-x)
# Group 2: bracket contents (e.g. "Zr0.33Nb0.67") — parsed below into fractions
# Group 3: optional nonmetal
_BRACKETED_VCA_RE = re.compile(
    r"^([A-Z][a-z]?)\(1-x\)\[([^\]]+)\]\(x\)([A-Z][a-z]?)?\s*$"
)
# Inside the bracket: pairs like "Zr0.33Nb0.67" → [("Zr",0.33),("Nb",0.67)]
_BRACKET_PAIR_RE  = re.compile(r"([A-Z][a-z]?)([0-9]*\.?[0-9]+)")


def _parse_bracket_pairs(content: str) -> list[tuple[str, float]] | None:
    """Parse 'Zr0.33Nb0.67' → [('Zr', 0.33), ('Nb', 0.67)]. Returns None on failure."""
    matches = _BRACKET_PAIR_RE.findall(content)
    if not matches:
        return None
    try:
        pairs = [(sym, float(frac)) for sym, frac in matches]
    except ValueError:
        return None
    if not pairs:
        return None
    return pairs


def parse_system_from_csv(csv_path: Path) -> SystemSpec | None:
    """Parse the '# System :' header line of a VCAForge CSV.

    Tries (in order):
      1. 'metals=Ti,Nb,Zr  nonmetal=C'  key-value form
      2. 'A(1-x)B(x)C'  legacy binary form
      3. 'A(1-x)[B0.33C0.67](x)D'  bracketed VCA mix form (ternary+ in one CSV)
      4. 'Ti-Nb-Zr-C'  dash-separated form
    Returns None for unparseable / multi-component / missing-element cases.
    """
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            for _ in range(20):
                line = f.readline()
                if not line:
                    break
                if not line.startswith("#"):
                    break
                m = _SYSTEM_LINE_RE.match(line)
                if not m:
                    continue
                # 1. Key-value form (most explicit)
                full = line
                metals_match = _METALS_KV_RE.search(full)
                if metals_match:
                    metals_str = metals_match.group(1).strip()
                    metals = tuple(s.strip() for s in metals_str.split(",") if s.strip())
                    if len(metals) >= 1:
                        nm_match = _NONMETAL_KV_RE.search(full)
                        nm = nm_match.group(1).strip() if nm_match else ""
                        return _safe_make_system(metals, nm)
                # 2. Binary VCA form
                label = m.group(1).strip()
                bm = _BINARY_VCA_RE.match(label)
                if bm:
                    a, b, n = bm.group(1), bm.group(2), bm.group(3) or ""
                    return _safe_make_system((a, b), n)
                # 3. Bracketed VCA mix form: A(1-x)[B0.33C0.67](x)D
                br = _BRACKETED_VCA_RE.match(label)
                if br:
                    metal_a = br.group(1)
                    inner = br.group(2)
                    nm = br.group(3) or ""
                    pairs = _parse_bracket_pairs(inner)
                    if pairs:
                        # Canonical metal ordering: A first, then bracket metals
                        # sorted alphabetically (so different runs with same set
                        # produce the SAME SystemSpec regardless of fraction order)
                        bracket_metals = sorted(p[0] for p in pairs)
                        metals = tuple(sorted(set([metal_a] + bracket_metals)))
                        return _safe_make_system(metals, nm)
                    log.debug("CSV %s has unparseable bracket content '%s'", csv_path.name, inner)
                    return None
                # 4. Dash-separated form
                if _DASH_SYSTEM_RE.match(label):
                    return _parse_dash_system(label)

                # 5. Single compound fallback (e.g. "BW2" seed -> maybe pure W-B)
                # This helps discover baseline points for aggregator.
                try:
                    return parse_system_from_string(label)
                except ValueError:
                    pass

                log.debug("CSV %s has unrecognised system label '%s'", csv_path.name, label)
                return None
    except (OSError, UnicodeDecodeError) as exc:
        log.warning("Failed to read header of %s: %s", csv_path, exc)
    return None


def parse_bracket_header_with_fractions(
    csv_path: Path,
) -> tuple[SystemSpec, dict[str, float], str] | None: # <-- Додано str для metal_a
    """For bracketed-VCA CSVs, return (system, {metal: inner_fraction}, metal_a)."""
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            for _ in range(20):
                line = f.readline()
                if not line or not line.startswith("#"):
                    break
                m = _SYSTEM_LINE_RE.match(line)
                if not m:
                    continue
                label = m.group(1).strip()
                br = _BRACKETED_VCA_RE.match(label)
                if not br:
                    return None

                metal_a = br.group(1)
                pairs = _parse_bracket_pairs(br.group(2))
                nm = br.group(3) or ""
                if not pairs:
                    return None

                bracket_metals = sorted(p[0] for p in pairs)
                metals = tuple(sorted(set([metal_a] + bracket_metals)))
                sys_spec = _safe_make_system(metals, nm)
                if sys_spec is None:
                    return None

                fracs = {}
                for sym, frac in pairs:
                    fracs[sym] = frac
                return sys_spec, fracs, metal_a # <-- Явно повертаємо metal_a
    except (OSError, UnicodeDecodeError) as exc:
        log.warning("Failed to read header of %s: %s", csv_path, exc)
    return None


def parse_system_from_string(s: str) -> SystemSpec:
    """Parse a CLI-provided system spec.

    Supported formats:
      'Ti-Nb-C'              → binary metals + C nonmetal
      'Ti-Nb'                → binary metals, no nonmetal
      'Ti-Nb-Zr-C'           → ternary metals + C
      'Ti-Nb-Zr-Hf-V-C'      → 5 metals + C
      'Ti(1-x)Nb(x)C'        → legacy binary VCA form
      'W-B' or 'WB'          → unary metal + nonmetal
      'W'                    → unary metal

    Whether the last token is a metal or nonmetal is decided by looking it
    up in config.ELEMENTS — if the entry has 'nonmetal': True (e.g. C, N, B),
    it's treated as the nonmetal, otherwise it's the last metal.
    """
    s = s.strip()
    if not s:
        raise ValueError("Empty system spec.")

    # 1. Legacy binary VCA form
    m = _BINARY_VCA_RE.match(s)
    if m:
        return _safe_make_system((m.group(1), m.group(2)), m.group(3) or "")

    # 2. Dash-separated form
    if "-" in s:
        return _parse_dash_system(s)

    # 3. Compact form or single metal (e.g. "WB" or "Ti")
    import re
    # Match element pairs (e.g. "TiNb", "WB")
    tokens = re.findall(r"([A-Z][a-z]?)", s)
    if tokens:
        elements = _elements()
        metals = []
        nonmetal = ""
        for t in tokens:
            if t in elements:
                if elements[t].get("nonmetal"):
                    nonmetal = t
                else:
                    metals.append(t)
        if metals:
            return _safe_make_system(tuple(metals), nonmetal)

    raise ValueError(
        f"Cannot parse system '{s}'. Expected formats: "
        f"'Ti-Nb-C', 'Ti-Nb-Zr-C', 'W-B', 'WB', or 'Ti'."
    )


def _parse_dash_system(s: str) -> SystemSpec:
    """Split on dashes; last token is nonmetal iff config.ELEMENTS marks it so.

    Metal order is canonicalised alphabetically so that 'Ti-Zr-Nb-C' and
    'Ti-Nb-Zr-C' produce the SAME SystemSpec. This matches the bracket-VCA
    header parser (which also sorts), so a CLI --system flag and a CSV header
    that describe the same chemistry compare as equal.
    """
    parts = [p.strip() for p in s.split("-") if p.strip()]
    if len(parts) < 2:
        raise ValueError(f"System '{s}' needs at least 2 elements.")
    elements = _elements()
    last = parts[-1]
    last_is_nonmetal = (
        last in elements
        and bool(elements[last].get("nonmetal", False))
    )
    if last_is_nonmetal and len(parts) >= 3:
        raw_metals = parts[:-1]
        nonmetal = last
    else:
        raw_metals = parts
        nonmetal = ""
    if len(raw_metals) < 2:
        raise ValueError(
            f"System '{s}' resolved to <2 metals (parsed as {tuple(raw_metals)}). "
            f"For pure-nonmetal alloys (rare), specify metals explicitly."
        )
    # Canonicalise: alphabetical order
    metals = tuple(sorted(raw_metals))
    # Hard validation: every element must exist in config.ELEMENTS
    for m in metals:
        if m not in elements:
            raise ValueError(
                f"Metal '{m}' from system '{s}' not in config.ELEMENTS. "
                f"Add it to config.ELEMENTS (with Z, val, rad, en) before use."
            )
    if nonmetal and nonmetal not in elements:
        raise ValueError(
            f"Nonmetal '{nonmetal}' from system '{s}' not in config.ELEMENTS."
        )
    return SystemSpec(metals=metals, nonmetal=nonmetal)


def _safe_make_system(metals: tuple[str, ...], nonmetal: str) -> SystemSpec | None:
    """Construct SystemSpec, returning None instead of raising for soft failures.

    Metals are canonicalised to alphabetical order so that all parsers
    (CLI, bracket VCA, key-value) produce the SAME SystemSpec for the same
    chemistry, regardless of the order they appeared in the input string.
    """
    elements = _elements()
    for m in metals:
        if m not in elements:
            log.warning("Element '%s' missing from config.ELEMENTS — system skipped.", m)
            return None
    if nonmetal and nonmetal not in elements:
        log.warning("Nonmetal '%s' missing from config.ELEMENTS — system skipped.", nonmetal)
        return None
    try:
        return SystemSpec(metals=tuple(sorted(metals)), nonmetal=nonmetal)
    except ValueError as exc:
        log.warning("Cannot build SystemSpec(metals=%s, nonmetal=%s): %s", metals, nonmetal, exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Born stability check (cubic) — safety net for orchestrator bugs
# ─────────────────────────────────────────────────────────────────────────────

def _born_cubic_stable(c11: float, c12: float, c44: float) -> bool:
    """Cubic Born conditions. Returns True if unknown (any NaN) — defer to other rules."""
    if not (np.isfinite(c11) and np.isfinite(c12) and np.isfinite(c44)):
        return True
    return (
        c11 > 0
        and c44 > 0
        and (c11 - c12) > 0
        and (c11 + 2 * c12) > 0
    )


# ─────────────────────────────────────────────────────────────────────────────
# DataIngestor — N-component aware
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _LoadedRow:
    """Internal: one fully-parsed CSV row in simplex coords + descriptor."""
    system: SystemSpec
    x_vector: np.ndarray              # shape (n_metals,)
    target_value: float
    label: int                        # 0 unstable, 1 stable
    descriptor: np.ndarray            # shape (descriptor_length,)
    source_csv: str

    @property
    def x_legacy(self) -> float:
        """Backward-compat: for binary, the second metal's fraction."""
        if len(self.x_vector) == 2:
            return float(self.x_vector[1])
        return float("nan")

    @property
    def composition(self) -> dict[str, float]:
        return {m: float(x) for m, x in zip(self.system.metals, self.x_vector)}


class DataIngestor:
    """Loads vca_results.csv files, parses system + composition, classifies rows.

    Composition column resolution
    ─────────────────────────────
    For a system with metals = (M_1, ..., M_N), the loader looks for, in order:
      1. Per-metal columns 'x_M_1', 'x_M_2', ..., 'x_M_N'.
         If ALL N are present, the row's x-vector is read directly from them
         (clamped to ≥ 0 and renormalised so Σ = 1 if rounding errors caused drift).
      2. Otherwise, the legacy binary 'concentration' column. Only valid when
         N = 2: x-vector becomes [1 - c, c] (binary VCAForge convention).
      3. Otherwise the row is skipped with a warning.

    Classification rules (unchanged from binary version)
    ────────────────────────────────────────────────────
    Class 1 (stable): status=done AND target finite AND born_stable≠'no'
                      AND elastic_source≠'Vegard_interpolation' AND Born OK.
    Class 0 (unstable): failed status, born_stable='no', kill reason ∈ {scf,
                        smax, timeout}, C44 < 0, or Born conditions fail.
    Ignored: pending/skipped, Vegard rows, unparseable systems.
    """

    _UNSTABLE_KILL_REASONS: frozenset[str] = frozenset(
        {"scf_nosconv", "smax_stall", "timeout"}
    )

    def __init__(
        self,
        csv_path: Path | None = None,
        base_dir: Path | None = None,
        target: str | None = None,
        descriptor_builder: DescriptorBuilder | None = None,
        explicit_system: SystemSpec | None = None,
        bracket_fracs: dict[str, float] | None = None,
        bracket_metal_a: str | None = None,
    ) -> None:
        self.csv_path = csv_path
        self.base_dir = base_dir
        self.target = target or _navigator_target()
        self.descriptor_builder = descriptor_builder or DescriptorBuilder()
        self.explicit_system = explicit_system
        self.bracket_fracs = bracket_fracs
        self.bracket_metal_a = bracket_metal_a

        self._loaded_rows: list[_LoadedRow] = []
        self._systems_seen: list[SystemSpec] = []

    # ── Public API ─────────────────────────────────────────────────────────

    def load(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns (X_stable, y_stable, X_all, labels_all) in DESCRIPTOR space.

        Note: rows from systems with different N may produce descriptor
        vectors of different LENGTH (in 'full' mode). When merging across
        systems, this would normally crash GP fitting. To avoid that:
          - In "physics_only" mode all descriptors have length 7 → safe.
          - In "full" mode, we PAD shorter descriptors with zeros on the
            x-prefix portion when mixing systems of different N. This is
            a deliberate compromise: it preserves identifiability within
            each N while still allowing transfer learning across N. The
            zero-padded entries become "this system has no x_5" markers.
        """
        rows = self._load_all_rows()
        self._loaded_rows = rows

        if not rows:
            return (
                np.empty((0, 0), dtype=float),
                np.empty((0,), dtype=float),
                np.empty((0, 0), dtype=float),
                np.empty((0,), dtype=int),
            )

        # Determine padded descriptor length (max across all systems)
        max_desc_len = max(len(r.descriptor) for r in rows)
        padded = [self._pad(r.descriptor, max_desc_len) for r in rows]

        stable = [(p, r) for p, r in zip(padded, rows) if r.label == 1]
        unstable = [(p, r) for p, r in zip(padded, rows) if r.label == 0]

        X_stable = (
            np.array([p for p, _ in stable], dtype=float)
            if stable else np.empty((0, max_desc_len), dtype=float)
        )
        y_stable = np.array([r.target_value for _, r in stable], dtype=float)

        all_pairs = stable + unstable
        X_all = (
            np.array([p for p, _ in all_pairs], dtype=float)
            if all_pairs else np.empty((0, max_desc_len), dtype=float)
        )
        labels_all = np.array([r.label for _, r in all_pairs], dtype=int)

        return X_stable, y_stable, X_all, labels_all

    def loaded_rows(self) -> list[_LoadedRow]:
        return list(self._loaded_rows)

    def systems_seen(self) -> list[SystemSpec]:
        return list(self._systems_seen)

    @staticmethod
    def _pad(vec: np.ndarray, target_len: int) -> np.ndarray:
        if len(vec) == target_len:
            return vec
        if len(vec) > target_len:
            return vec[:target_len]   # shouldn't happen but be safe
        pad = np.zeros(target_len - len(vec), dtype=float)
        return np.concatenate([vec, pad])

    # ── Internal: file discovery ───────────────────────────────────────────

    def _discover_csv_files(self) -> list[Path]:
        if self.csv_path is not None:
            csv_path = Path(self.csv_path).expanduser().resolve()
            if not csv_path.exists():
                raise FileNotFoundError(f"CSV not found: {csv_path}")
            return [csv_path]

        if self.base_dir is None:
            raise ValueError("DataIngestor needs either csv_path or base_dir.")

        base_dir = Path(self.base_dir).expanduser().resolve()
        if not base_dir.exists():
            raise FileNotFoundError(f"Scan directory does not exist: {base_dir}")

        csv_name = _csv_filename()
        # Discover both per-run and master CSVs
        files = sorted(set(base_dir.rglob(csv_name)) | set(base_dir.rglob("*" + _master_suffix())))
        if not files:
            log.info(f"No '{csv_name}' or '*{_master_suffix()}' files found under {base_dir} (recursive).")
            return []
        return files

    # ── Internal: row loading ──────────────────────────────────────────────

    def _load_all_rows(self) -> list[_LoadedRow]:
        files = self._discover_csv_files()
        all_rows: list[_LoadedRow] = []
        seen_systems: dict[str, SystemSpec] = {}
        parse_errors: list[str] = []

        for f in files:
            bracket_info = parse_bracket_header_with_fractions(f)
            if bracket_info is not None:
                sys_spec, bracket_fracs, bracket_metal_a = bracket_info # <-- Приймаємо metal_a
            else:
                sys_spec = parse_system_from_csv(f) or self.explicit_system
                bracket_fracs = None
                bracket_metal_a = None

            if sys_spec is None:
                parse_errors.append(f"{f}: no parseable system header")
                continue

            try:
                df = pd.read_csv(f, comment="#")
            except Exception as exc:
                parse_errors.append(f"{f}: parse failed: {exc}")
                continue

            if self.target not in df.columns:
                parse_errors.append(f"{f}: missing target '{self.target}'")
                continue

            seen_systems.setdefault(sys_spec.label(), sys_spec)
            # Передаємо bracket_metal_a у класифікатор
            file_rows = self._classify_dataframe(df, sys_spec, str(f), bracket_fracs, bracket_metal_a)
            if not file_rows:
                parse_errors.append(f"{f}: dataframe classification failed (columns/format mismatch for system {sys_spec.label()})")
            all_rows.extend(file_rows)

        if not all_rows:
            detail = "\n  ".join(parse_errors[:5]) or "(no specific errors recorded)"
            extra = f"\n  ... and {len(parse_errors) - 5} more" if len(parse_errors) > 5 else ""
            if files:
                if any(self.target in e for e in parse_errors):
                    avail = self._collect_available_columns(files)
                    log.warning(
                        f"Target '{self.target}' not found in some CSVs, or no usable rows.\n"
                        f"  Numeric columns seen: {avail}\n"
                        f"  Errors:\n  {detail}{extra}"
                    )
                else:
                    log.warning(
                        f"No usable rows from {len(files)} CSV file(s).\n"
                        f"  Errors:\n  {detail}{extra}"
                    )

        self._systems_seen = list(seen_systems.values())
        return all_rows

    def _classify_dataframe(
        self,
        df: pd.DataFrame,
        sys_spec: SystemSpec,
        source: str,
        bracket_fracs: dict[str, float] | None = None,
        bracket_metal_a: str | None = None, # <-- Додано аргумент
    ) -> list[_LoadedRow]:
        df = df.copy()

        col_names = sys_spec.column_names()
        if all(c in df.columns for c in col_names):
            x_matrix = df[col_names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        elif bracket_fracs is not None and bracket_metal_a is not None and "concentration" in df.columns:
            c = pd.to_numeric(df["concentration"], errors="coerce").to_numpy(dtype=float)
            cols: list[np.ndarray] = []
            for m in sys_spec.metals:
                frac = float(bracket_fracs.get(m, 0.0))
                if m == bracket_metal_a: # <-- ВИПРАВЛЕНО: Додаємо c * frac якщо метал є і в дужках
                    cols.append((1.0 - c) + c * frac)
                else:
                    cols.append(c * frac)
            x_matrix = np.column_stack(cols)
        elif sys_spec.is_binary() and "concentration" in df.columns:
            c = pd.to_numeric(df["concentration"], errors="coerce").to_numpy(dtype=float)
            x_matrix = np.column_stack([1.0 - c, c])
        else:
            return []

        # Drop rows with any NaN composition
        valid_comp = ~np.any(np.isnan(x_matrix), axis=1)

        # Normalise rows where Σ has small drift; reject if drift is large
        sums = np.nansum(x_matrix, axis=1)
        normalisable = valid_comp & (sums > 0.5) & (sums < 1.5)
        with np.errstate(invalid="ignore", divide="ignore"):
            x_matrix_norm = np.where(
                normalisable[:, None] & (sums[:, None] > 0),
                x_matrix / sums[:, None],
                np.nan,
            )

        # ── Classification flags (vectorised on the dataframe) ────────────
        status         = df.get("status",         pd.Series([""] * len(df))).fillna("").astype(str).str.lower()
        born_stable    = df.get("born_stable",    pd.Series([""] * len(df))).fillna("").astype(str).str.lower()
        elastic_source = df.get("elastic_source", pd.Series([""] * len(df))).fillna("").astype(str)
        kill_reason    = df.get("kill_reason",    pd.Series([""] * len(df))).fillna("").astype(str).str.lower()

        c11 = pd.to_numeric(df.get("C11", pd.Series([np.nan] * len(df))), errors="coerce")
        c12 = pd.to_numeric(df.get("C12", pd.Series([np.nan] * len(df))), errors="coerce")
        c44 = pd.to_numeric(df.get("C44", pd.Series([np.nan] * len(df))), errors="coerce")
        target_vals = pd.to_numeric(df[self.target], errors="coerce")

        is_pending = status.isin(["pending", "skipped"])
        is_vegard  = elastic_source.eq("Vegard_interpolation")

        # Born safety net (only when all three constants finite)
        all_three_finite = c11.notna() & c12.notna() & c44.notna()
        born_pass = pd.Series([True] * len(df), index=df.index)
        born_fail = pd.Series([False] * len(df), index=df.index)
        for i in df.index:
            if all_three_finite[i]:
                ok = _born_cubic_stable(float(c11[i]), float(c12[i]), float(c44[i]))
                born_pass[i] = ok
                born_fail[i] = not ok

        # Class 1: stable
        is_done = status.eq("done")
        target_ok = target_vals.notna()
        born_ok   = ~born_stable.eq("no")
        not_vegard = ~is_vegard
        stable_mask = is_done & target_ok & born_ok & not_vegard & born_pass

        # Class 0: unstable
        is_failed = status.eq("failed")
        born_no   = born_stable.eq("no")
        kill_unstable = kill_reason.isin(self._UNSTABLE_KILL_REASONS)
        c44_neg = c44.notna() & (c44 < 0)
        unstable_mask = (
            (is_failed | born_no | kill_unstable | c44_neg | born_fail)
            & ~is_vegard & ~is_pending
        )
        stable_mask = stable_mask & ~unstable_mask

        # ── Build per-row records ─────────────────────────────────────────
        rows: list[_LoadedRow] = []
        for i in df.index:
            if not normalisable[i]:
                continue
            x_vec = x_matrix_norm[i]
            if np.any(np.isnan(x_vec)):
                continue
            if stable_mask[i]:
                label = 1
                tv = float(target_vals[i])
            elif unstable_mask[i]:
                label = 0
                tv = float("nan")
            else:
                continue

            try:
                desc = self.descriptor_builder.build(sys_spec, x_vec)
            except ValueError as exc:
                log.debug("Skip row x=%s in %s: %s", x_vec, source, exc)
                continue
            rows.append(_LoadedRow(
                system=sys_spec, x_vector=np.asarray(x_vec, dtype=float),
                target_value=tv, label=label, descriptor=desc, source_csv=source,
            ))
        return rows

    def _collect_available_columns(self, files: list[Path]) -> list[str]:
        cols: set[str] = set()
        for f in files[:5]:
            try:
                df = pd.read_csv(f, comment="#", nrows=1)
                cols.update(df.columns)
            except Exception:  # noqa: BLE001
                continue
        return sorted(cols)


# ─────────────────────────────────────────────────────────────────────────────
# SurrogateModel — operates on descriptor vectors of any length
# ─────────────────────────────────────────────────────────────────────────────

class SurrogateModel:
    """Twin GPs in descriptor space. Length-agnostic — works for any N."""

    _KERNEL_NAMES = {
        "matern25": "Matern-5/2 + White",
        "rbf":      "RBF + White",
        "matern15": "Matern-3/2 + White",
    }

    def __init__(self) -> None:
        self.gpr: GaussianProcessRegressor | None = None
        self.gpc: GaussianProcessClassifier | None = None
        self.fitted: bool = False
        self.selected_kernel: str = ""
        self.n_stable: int = 0
        self.n_total: int = 0
        self.descriptor_length: int = 0

    def fit(
        self,
        X_stable: np.ndarray,
        y_stable: np.ndarray,
        X_all: np.ndarray,
        labels_all: np.ndarray,
    ) -> "SurrogateModel":
        self.n_stable = len(X_stable)
        self.n_total = len(X_all)
        self.descriptor_length = X_stable.shape[1] if len(X_stable) else 0
        min_pts = _navigator_min_points()

        if self.n_stable < min_pts:
            log.info("Surrogate not fitted: n_stable=%d < min=%d.", self.n_stable, min_pts)
            self.fitted = False
            return self

        try:
            self._fit_regressor(X_stable, y_stable)
            self._fit_classifier(X_all, labels_all)
            self.fitted = True
        except Exception as exc:  # noqa: BLE001
            log.warning("GP fit failed (%s) — fallback to LHS.", exc)
            self.fitted = False
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.fitted or self.gpr is None:
            raise RuntimeError("predict() before successful fit().")
        X = np.atleast_2d(np.asarray(X, dtype=float))
        # Pad if caller's descriptor is shorter than what we fitted
        X = self._pad_to_fit(X)
        mu, sigma = self.gpr.predict(X, return_std=True)
        return np.asarray(mu, dtype=float), np.asarray(sigma, dtype=float)

    def p_feasible(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        X = self._pad_to_fit(X)
        if self.gpc is None:
            return np.ones(len(X), dtype=float)
        try:
            proba = self.gpc.predict_proba(X)
        except Exception as exc:  # noqa: BLE001
            log.warning("GPC predict_proba failed (%s) → P=1.0", exc)
            return np.ones(len(X), dtype=float)
        classes = list(getattr(self.gpc, "classes_", [0, 1]))
        col = classes.index(1) if 1 in classes else proba.shape[1] - 1
        return np.asarray(proba[:, col], dtype=float)

    # ── Internals ─────────────────────────────────────────────────────────

    def _pad_to_fit(self, X: np.ndarray) -> np.ndarray:
        if X.shape[1] == self.descriptor_length:
            return X
        if X.shape[1] > self.descriptor_length:
            return X[:, :self.descriptor_length]
        pad = np.zeros((X.shape[0], self.descriptor_length - X.shape[1]), dtype=float)
        return np.concatenate([X, pad], axis=1)

    def _fit_regressor(self, X: np.ndarray, y: np.ndarray) -> None:
        kernel_choice = _navigator_kernel().lower()
        if kernel_choice == "matern":
            candidates = [("matern25", self._k_matern25())]
        elif kernel_choice == "rbf":
            candidates = [("rbf", self._k_rbf())]
        else:
            if kernel_choice != "auto":
                log.warning("Unknown kernel '%s' → auto.", kernel_choice)
            candidates = [
                ("matern25", self._k_matern25()),
                ("rbf",      self._k_rbf()),
                ("matern15", self._k_matern15()),
            ]

        best_lml = -np.inf
        best_gpr: GaussianProcessRegressor | None = None
        best_name = ""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for name, kernel in candidates:
                gpr = GaussianProcessRegressor(
                    kernel=kernel, n_restarts_optimizer=5,
                    normalize_y=True, random_state=0,
                )
                try:
                    gpr.fit(X, y)
                except Exception as exc:  # noqa: BLE001
                    log.debug("GPR %s failed: %s", name, exc)
                    continue
                lml = float(gpr.log_marginal_likelihood_value_)
                if lml > best_lml:
                    best_lml, best_gpr, best_name = lml, gpr, name

        if best_gpr is None:
            raise RuntimeError("All GPR kernels failed.")
        self.gpr = best_gpr
        self.selected_kernel = self._KERNEL_NAMES.get(best_name, best_name)

    def _fit_classifier(self, X: np.ndarray, labels: np.ndarray) -> None:
        if len(np.unique(labels)) < 2:
            log.info("Single-class feasibility data — GPC skipped.")
            self.gpc = None
            return
        kernel = ConstantKernel(1.0) * RBF(length_scale_bounds=(1e-3, 10.0))
        gpc = GaussianProcessClassifier(kernel=kernel, n_restarts_optimizer=3, random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gpc.fit(X, labels)
        self.gpc = gpc

    @staticmethod
    def _k_matern25() -> Any:
        return Matern(nu=2.5, length_scale_bounds=(1e-3, 10.0)) + WhiteKernel()

    @staticmethod
    def _k_matern15() -> Any:
        return Matern(nu=1.5, length_scale_bounds=(1e-3, 10.0)) + WhiteKernel()

    @staticmethod
    def _k_rbf() -> Any:
        return RBF(length_scale_bounds=(1e-3, 10.0)) + WhiteKernel()


# ─────────────────────────────────────────────────────────────────────────────
# AcquisitionFunction — unchanged from binary version (works on any descriptor)
# ─────────────────────────────────────────────────────────────────────────────

class AcquisitionFunction:
    """EI / CEI / UCB / MaxVar."""

    _ALLOWED = frozenset({"EI", "CEI", "UCB", "MaxVar"})

    def __init__(self, name: str | None = None, mode: str | None = None) -> None:
        n = (name or _navigator_acquisition()).strip()
        if n not in self._ALLOWED:
            log.warning("Unknown acquisition '%s' → CEI.", n)
            n = "CEI"
        self.name = n

        m = (mode or _navigator_mode()).strip().lower()
        if m not in {"maximize", "minimize"}:
            log.warning("Unknown mode '%s' → maximize.", m)
            m = "maximize"
        self.mode = m
        self.kappa = _navigator_ucb_kappa()

    def compute(self, descriptor: np.ndarray, surrogate: SurrogateModel, y_best: float) -> float:
        x_arr = np.atleast_2d(np.asarray(descriptor, dtype=float))
        if not surrogate.fitted:
            return 0.0

        mu, sigma = surrogate.predict(x_arr)
        mu_s = float(mu[0])
        sigma_s = float(max(sigma[0], 0.0))

        if self.name == "MaxVar":
            return sigma_s

        if self.name == "UCB":
            if self.mode == "maximize":
                return mu_s + self.kappa * sigma_s
            return -(mu_s - self.kappa * sigma_s)

        ei = self._expected_improvement(mu_s, sigma_s, y_best)
        if self.name == "EI":
            return ei
        p_f = float(surrogate.p_feasible(x_arr)[0])
        return ei * p_f

    def _expected_improvement(self, mu: float, sigma: float, y_best: float) -> float:
        if sigma <= 1e-12:
            improvement = (mu - y_best) if self.mode == "maximize" else (y_best - mu)
            return max(improvement, 0.0)
        improvement = (mu - y_best) if self.mode == "maximize" else (y_best - mu)
        z = improvement / sigma
        ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
        return float(max(ei, 0.0))


# ─────────────────────────────────────────────────────────────────────────────
# AcquisitionOptimizer — N-component simplex search
# ─────────────────────────────────────────────────────────────────────────────

class AcquisitionOptimizer:
    """Multi-start SLSQP for x* = argmax acq(x; system) on the (N-1)-simplex.

    For N=2: bounds [0,1] suffice (Σ=1 enforced by 1D parameterization).
    For N≥3: bounds [0,1]^N + equality constraint Σx_i = 1 + Dirichlet starts.
    """

    def __init__(
        self,
        surrogate: SurrogateModel,
        acq_fn: AcquisitionFunction,
        descriptor_builder: DescriptorBuilder,
        target_system: SystemSpec,
        n_starts: int | None = None,
        ei_threshold: float | None = None,
        rng_seed: int | None = None,
        sqs_sites: int | None = None,
    ) -> None:
        self.surrogate = surrogate
        self.acq_fn = acq_fn
        self.db = descriptor_builder
        self.system = target_system
        self.n_starts = int(n_starts if n_starts is not None else _navigator_n_starts())
        self.ei_threshold = float(
            ei_threshold if ei_threshold is not None else _navigator_ei_threshold()
        )
        self._rng = np.random.default_rng(rng_seed)
        self.sqs_sites = sqs_sites

    @staticmethod
    def _quantize_simplex(x: np.ndarray, n_sites: int) -> np.ndarray:
        """Snaps continuous fractions to discrete multiples of 1/n_sites using the Largest Remainder Method."""
        x_scaled = x * n_sites
        floored = np.floor(x_scaled)
        remainder = int(round(n_sites - floored.sum()))
        if remainder > 0:
            diff = x_scaled - floored
            idx = np.argsort(diff)[::-1]
            for i in range(remainder):
                floored[idx[i]] += 1
        return floored / n_sites

    def find_next_point(self, y_best: float) -> Candidate:
        starts = self._generate_starts()    # shape (n_starts, N)
        N = self.system.n_metals

        # Evaluate at each start (for fallback if all SLSQP fails)
        start_acqs = np.array(
            [self._neg_acq_x(s, y_best) for s in starts], dtype=float,
        )

        best_x_vec: np.ndarray | None = None
        best_neg = np.inf
        any_success = False

        # Build constraints
        if N == 2:
            # Parameterize as scalar x_2 ∈ [0,1]; x_1 = 1 - x_2.
            # This avoids SLSQP's headache with a 1D equality constraint.
            for s in starts:
                x0_scalar = float(s[1])  # x_2
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        res = minimize(
                            fun=lambda xv: self._neg_acq_x(np.array([1.0 - xv[0], xv[0]]), y_best),
                            x0=[x0_scalar],
                            method="SLSQP",
                            bounds=[(_min_x_component(), 1.0 - _min_x_component())],
                            options={"maxiter": 100, "ftol": 1e-8},
                        )
                except Exception as exc:  # noqa: BLE001
                    log.debug("SLSQP failed (binary): %s", exc)
                    continue
                if not res.success:
                    continue
                x2 = float(np.clip(res.x[0], 0.0, 1.0))
                x_vec = np.array([1.0 - x2, x2])
                f_val = float(self._neg_acq_x(x_vec, y_best))
                if f_val < best_neg:
                    best_neg, best_x_vec, any_success = f_val, x_vec, True
        else:
            # Full N-D simplex with equality constraint
            min_x = _min_x_component()
            bounds = [(min_x, 1.0)] * N
            constraints = [{
                "type": "eq",
                "fun": lambda x: float(np.sum(x) - 1.0),
            }]
            for s in starts:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        res = minimize(
                            fun=lambda xv: self._neg_acq_x(np.asarray(xv, dtype=float), y_best),
                            x0=s,
                            method="SLSQP",
                            bounds=bounds,
                            constraints=constraints,
                            options={"maxiter": 200, "ftol": 1e-8},
                        )
                except Exception as exc:  # noqa: BLE001
                    log.debug("SLSQP failed (N=%d): %s", N, exc)
                    continue
                if not res.success:
                    continue
                x_vec = self._project_simplex(np.asarray(res.x, dtype=float))
                f_val = float(self._neg_acq_x(x_vec, y_best))
                if f_val < best_neg:
                    best_neg, best_x_vec, any_success = f_val, x_vec, True

        if not any_success or best_x_vec is None:
            log.warning("All %d SLSQP starts failed — using best Dirichlet sample.", self.n_starts)
            idx = int(np.argmin(start_acqs))
            best_x_vec = self._project_simplex(starts[idx])
            best_neg = float(start_acqs[idx])

        best_acq = -best_neg
        peak_acq = max(
            float(-np.min(start_acqs)) if np.isfinite(np.min(start_acqs)) else 0.0,
            best_acq,
        )
        return self._build_candidate(best_x_vec, best_acq, peak_acq, y_best)

    # ── Internals ─────────────────────────────────────────────────────────

    def _generate_starts(self) -> np.ndarray:
        N = self.system.n_metals
        n = max(self.n_starts, 2)
        if N == 2:
            # Linspace covers the line edge-to-edge — better than random for 1D
            xs2 = np.linspace(0.0, 1.0, n)
            return np.column_stack([1.0 - xs2, xs2])
        # Dirichlet(α=1,…,1) is uniform over the simplex
        alpha = np.full(N, _simplex_alpha())
        return self._rng.dirichlet(alpha, size=n)

    @staticmethod
    def _project_simplex(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, 0.0, 1.0)
        s = float(x.sum())
        if s > 1e-9:
            return x / s
        return np.full_like(x, 1.0 / len(x))

    def _neg_acq_x(self, x_vec: np.ndarray, y_best: float) -> float:
        x_vec = np.asarray(x_vec, dtype=float).ravel()
        # Renormalise softly — SLSQP's iterate may drift a bit off Σ=1
        s = float(x_vec.sum())
        if 0.5 < s < 1.5 and s > 0:
            x_vec = x_vec / s
        try:
            desc = self.db.build(self.system, x_vec)
            val = self.acq_fn.compute(desc, self.surrogate, y_best)
        except Exception as exc:  # noqa: BLE001
            log.debug("acq.compute crashed at %s: %s", x_vec, exc)
            return 1e12
        if not np.isfinite(val):
            return 1e12
        return -float(val)

    def _build_candidate(self, x_vec: np.ndarray, acq_value: float, peak_acq: float, y_best: float) -> Candidate:
        x_vec = self._project_simplex(np.asarray(x_vec, dtype=float))

        if self.sqs_sites and self.sqs_sites > 0:
            x_vec = self._quantize_simplex(x_vec, self.sqs_sites)
            # Re-evaluate the true acquisition value at this new discrete point
            acq_value = -self._neg_acq_x(x_vec, y_best)

        desc = self.db.build(self.system, x_vec).reshape(1, -1)

        if self.surrogate.fitted:
            mu_arr, sigma_arr = self.surrogate.predict(desc)
            mu, sigma = float(mu_arr[0]), float(sigma_arr[0])
            p_f = float(self.surrogate.p_feasible(desc)[0])
        else:
            mu, sigma = float("nan"), float("nan")
            p_f = None

        converged = self.acq_fn.name in {"EI", "CEI"} and peak_acq < self.ei_threshold

        source = {
            "CEI": "GP-CEI", "EI": "GP-EI",
            "UCB": "GP-UCB", "MaxVar": "GP-MaxVar",
        }.get(self.acq_fn.name, f"GP-{self.acq_fn.name}")

        composition = {m: float(x) for m, x in zip(self.system.metals, x_vec)}
        # Legacy concentration: x_2 for binary, NaN for N>=3 (caller can use composition)
        concentration_legacy = float(x_vec[1]) if self.system.is_binary() else float(x_vec[1])

        return Candidate(
            concentration=concentration_legacy,
            predicted_mean=mu,
            predicted_std=sigma,
            p_feasible=p_f,
            acq_value=float(acq_value),
            converged=bool(converged),
            source=source,
            system=self.system,
            descriptor_vector=tuple(desc[0].tolist()),
            composition=composition,
        )


# ─────────────────────────────────────────────────────────────────────────────
# NavigatorOrchestrator — top-level façade
# ─────────────────────────────────────────────────────────────────────────────

class NavigatorOrchestrator:
    """Public API: load → fit → suggest. Used by CLI and main.py."""

    def __init__(
        self,
        csv_path: Path | None = None,
        base_dir: Path | None = None,
        target: str | None = None,
        mode: str | None = None,
        acquisition: str | None = None,
        target_system: SystemSpec | str | None = None,
        sqs_sites: int | None = None,
        bracket_fracs: dict[str, float] | None = None,
        bracket_metal_a: str | None = None,
    ) -> None:
        if csv_path is None and base_dir is None:
            base_dir = Path.cwd()

        self.target = target or _navigator_target()
        self.mode = (mode or _navigator_mode()).lower()
        self.acquisition = acquisition or _navigator_acquisition()
        self.sqs_sites = sqs_sites

        if isinstance(target_system, str):
            target_system = parse_system_from_string(target_system)
        self.requested_system: SystemSpec | None = target_system

        self.descriptor_builder = DescriptorBuilder()
        self.ingestor = DataIngestor(
            csv_path=csv_path, base_dir=base_dir, target=self.target,
            descriptor_builder=self.descriptor_builder,
            explicit_system=target_system,
            bracket_fracs=bracket_fracs,
            bracket_metal_a=bracket_metal_a,
        )
        self.surrogate = SurrogateModel()
        self.acq_fn = AcquisitionFunction(name=self.acquisition, mode=self.mode)
        self.optimizer: AcquisitionOptimizer | None = None

        self._last_X_stable: np.ndarray | None = None
        self._last_y_stable: np.ndarray | None = None
        self._last_X_all: np.ndarray | None = None
        self._last_labels_all: np.ndarray | None = None
        self._last_candidate: Candidate | None = None
        self._resolved_system: SystemSpec | None = None

    def suggest(self) -> Candidate:
        X_stable_all, y_stable_all, X_all_all, labels_all_all = self.ingestor.load()
        self._last_X_stable = X_stable_all
        self._last_y_stable = y_stable_all
        self._last_X_all = X_all_all
        self._last_labels_all = labels_all_all

        target_sys = self._resolve_target_system()
        self._resolved_system = target_sys

        # CRITICAL: filter training data to ONLY rows from systems whose descriptor
        # is directly comparable to the target. Mixing systems with different metal
        # sets (e.g. Ni-B-Mn-C training data → Ti-Zr-Nb-C recommendation) was
        # producing nonsensical recommendations because the descriptor's x-prefix
        # gets zero-padded for missing components, which is not a real point on
        # any simplex. Strict same-metals filtering is the safe default.
        rows_target = [
            r for r in self.ingestor.loaded_rows()
            if r.system == target_sys
        ]
        if rows_target:
            X_stable = np.array(
                [r.descriptor for r in rows_target if r.label == 1],
                dtype=float,
            ) if any(r.label == 1 for r in rows_target) else np.empty(
                (0, self.descriptor_builder.descriptor_length(target_sys)), dtype=float,
            )
            y_stable = np.array(
                [r.target_value for r in rows_target if r.label == 1],
                dtype=float,
            )
            X_all = np.array([r.descriptor for r in rows_target], dtype=float)
            labels_all = np.array([r.label for r in rows_target], dtype=int)
        else:
            # No data for this exact system → empty arrays, will trigger LHS fallback
            d_len = self.descriptor_builder.descriptor_length(target_sys)
            X_stable = np.empty((0, d_len), dtype=float)
            y_stable = np.empty((0,), dtype=float)
            X_all = np.empty((0, d_len), dtype=float)
            labels_all = np.empty((0,), dtype=int)

        # Cache the FILTERED data (not the raw ingested) so report() and other
        # consumers see only target-system points.
        self._last_X_stable = X_stable
        self._last_y_stable = y_stable
        self._last_X_all = X_all
        self._last_labels_all = labels_all

        y_best, _x_best = self._compute_y_best(target_sys)

        if len(X_stable) < _navigator_min_points():
            cand = self._lhs_fallback(target_sys, X_stable)
            self._last_candidate = cand
            return cand

        self.surrogate.fit(X_stable, y_stable, X_all, labels_all)
        if not self.surrogate.fitted:
            cand = self._lhs_fallback(target_sys, X_stable)
            self._last_candidate = cand
            return cand

        self.optimizer = AcquisitionOptimizer(
            surrogate=self.surrogate, acq_fn=self.acq_fn,
            descriptor_builder=self.descriptor_builder,
            target_system=target_sys,
            sqs_sites=self.sqs_sites,
        )
        cand = self.optimizer.find_next_point(y_best)
        self._last_candidate = cand
        return cand

    def report(self) -> NavigatorReport:
        if self._last_candidate is None:
            self.suggest()

        target_sys = self._resolved_system
        n_stable, best_val, best_x, best_comp = self._best_for_target(target_sys)
        n_failed = self._n_failed_for_target(target_sys)

        return NavigatorReport(
            n_stable=n_stable,
            n_failed=n_failed,
            n_total=n_stable + n_failed,
            current_best_value=best_val,
            current_best_x=best_x,
            selected_kernel=self.surrogate.selected_kernel or "(not fitted)",
            candidate=self._last_candidate or Candidate(
                concentration=float("nan"),
                predicted_mean=float("nan"),
                predicted_std=float("nan"),
                p_feasible=None, acq_value=None, source="LHS-fallback",
            ),
            target=self.target,
            mode=self.mode,
            acquisition=self.acquisition,
            target_system=target_sys,
            systems_in_data=[s.label() for s in self.ingestor.systems_seen()],
            descriptor_mode=_use_descriptors(),
            current_best_composition=best_comp,
        )

    # ── Internals ─────────────────────────────────────────────────────────

    def _resolve_target_system(self) -> SystemSpec:
        if self.requested_system is not None:
            return self.requested_system
        seen = self.ingestor.systems_seen()
        if not seen:
            raise ValueError("No systems found in data and no --system specified.")
        if len(seen) == 1:
            return seen[0]
        # Pick the system with most stable data
        rows = self.ingestor.loaded_rows()
        counts: dict[str, int] = {}
        for r in rows:
            if r.label == 1:
                counts[r.system.label()] = counts.get(r.system.label(), 0) + 1
        if not counts:
            return seen[0]
        best_label = max(counts, key=lambda k: counts[k])
        chosen = next(s for s in seen if s.label() == best_label)
        log.info("Multi-system data: auto-selected '%s' (%d stable points). Use --system to override.",
                 chosen.label(), counts[best_label])
        return chosen

    def _compute_y_best(self, target_sys: SystemSpec) -> tuple[float, float]:
        rows_t = [r for r in self.ingestor.loaded_rows()
                  if r.label == 1 and r.system == target_sys]
        if not rows_t:
            ys = self._last_y_stable if self._last_y_stable is not None else np.array([])
            if len(ys) == 0:
                return float("-inf") if self.mode == "maximize" else float("inf"), float("nan")
            idx = int(np.argmax(ys) if self.mode == "maximize" else np.argmin(ys))
            return float(ys[idx]), float("nan")
        if self.mode == "maximize":
            best = max(rows_t, key=lambda r: r.target_value)
        else:
            best = min(rows_t, key=lambda r: r.target_value)
        return float(best.target_value), float(best.x_legacy)

    def _best_for_target(
        self, target_sys: SystemSpec | None,
    ) -> tuple[int, float, float, dict[str, float] | None]:
        """Returns (n_stable, best_value, best_x_legacy, best_composition)."""
        if target_sys is None:
            return 0, float("nan"), float("nan"), None
        rows_t = [r for r in self.ingestor.loaded_rows()
                  if r.label == 1 and r.system == target_sys]
        if not rows_t:
            return 0, float("nan"), float("nan"), None
        if self.mode == "maximize":
            best = max(rows_t, key=lambda r: r.target_value)
        else:
            best = min(rows_t, key=lambda r: r.target_value)
        return len(rows_t), float(best.target_value), float(best.x_legacy), best.composition

    def _n_failed_for_target(self, target_sys: SystemSpec | None) -> int:
        if target_sys is None:
            return 0
        return sum(1 for r in self.ingestor.loaded_rows()
                   if r.label == 0 and r.system == target_sys)

    def _lhs_fallback(self, target_sys: SystemSpec, X_existing: np.ndarray) -> Candidate:
        N = target_sys.n_metals
        n_grid = max(_navigator_n_starts(), 50)
        rng = np.random.default_rng()

        if N == 2:
            edges = np.linspace(0.0, 1.0, n_grid + 1)
            grid_x2 = edges[:-1] + (edges[1:] - edges[:-1]) * rng.random(n_grid)
            grid = np.column_stack([1.0 - grid_x2, grid_x2])
        else:
            grid = rng.dirichlet(np.full(N, _simplex_alpha()), size=n_grid)

        # Pick the grid point furthest (in Euclidean simplex distance)
        # from any existing target-system point.
        target_xs = np.array(
            [r.x_vector for r in self.ingestor.loaded_rows() if r.system == target_sys],
            dtype=float,
        )
        if len(target_xs) == 0:
            x_pick = grid[len(grid) // 2]
        else:
            # min-distance from each grid point to nearest existing point
            min_d = np.array([
                float(np.min(np.linalg.norm(target_xs - g, axis=1))) for g in grid
            ])
            x_pick = grid[int(np.argmax(min_d))]

        desc = self.descriptor_builder.build(target_sys, x_pick)
        composition = {m: float(x) for m, x in zip(target_sys.metals, x_pick)}

        return Candidate(
            concentration=float(x_pick[1]) if target_sys.is_binary() else float(x_pick[1]),
            predicted_mean=float("nan"),
            predicted_std=float("nan"),
            p_feasible=None,
            acq_value=None,
            converged=False,
            source="LHS-fallback",
            system=target_sys,
            descriptor_vector=tuple(desc.tolist()),
            composition=composition,
        )




# ─────────────────────────────────────────────────────────────────────────────
# Command generation — VCAForge CLI builder
# ─────────────────────────────────────────────────────────────────────────────

def build_vcaforge_command(
    structure_file: str | Path,
    candidate: Candidate,
    engine: str = "castep",
    n_steps: int = 0,
    run_elastic: bool = True,
    mpi_procs: int | None = None,
    crystal_mode: str | None = None,
    template: str | None = None,
) -> str:
    """Render the VCAForge command line for `candidate`."""
    if candidate.system is None or candidate.composition is None:
        raise ValueError("Candidate invalid — cannot build command.")

    sys_spec = candidate.system
    comp = candidate.composition

    species_specs = " ".join(f"{m}:{comp[m]:.4f}" for m in sys_spec.metals)

    cmd = (
        f"python main.py '{structure_file}' "
        f"--composition {species_specs} "
        f"--x 1.0 "
        f"--engine {engine} "
        f"-n {mpi_procs}"
    )
    if run_elastic:
        cmd += " --elastic"
    if crystal_mode:
        cmd += f" --mode {crystal_mode}"
    if template:
        cmd += f" --template {template}"
    return cmd


# ─────────────────────────────────────────────────────────────────────────────
# Study completion criterion — "is the exploration done?"
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CompletionAssessment:
    """Whether the BO study has converged (or should keep going)."""
    is_done: bool
    reason: str
    n_stable: int
    peak_predicted_value: float
    peak_predicted_sigma: float
    peak_composition: dict[str, float] | None
    last_acq_value: float | None
    confidence_pct: int           # 0..100, how confident we are the peak is found

    def short_summary(self) -> str:
        if self.is_done:
            return f"DONE ({self.confidence_pct}% confident): {self.reason}"
        return f"CONTINUE: {self.reason}"


def assess_completion(
    nav: NavigatorOrchestrator,
    consecutive_duplicate_suggestions: int = 0,
) -> CompletionAssessment:
    """Returns whether the BO study should stop. Multi-criterion check.

    A study is "done" when:
      (a) candidate.converged is True (acquisition < threshold), OR
      (b) σ at predicted optimum < NAVIGATOR_DONE_PEAK_SIGMA_MAX AND
          n_stable ≥ NAVIGATOR_DONE_MIN_POINTS, OR
      (c) The last ≥ NAVIGATOR_DONE_DEDUPE_REPEATS suggestions all duplicated
          existing points (loop is suggesting refinements only).
    """
    if nav._last_candidate is None:
        nav.suggest()
    cand = nav._last_candidate
    if cand is None:
        return CompletionAssessment(
            is_done=False, reason="No suggestion yet",
            n_stable=0, peak_predicted_value=float("nan"),
            peak_predicted_sigma=float("nan"), peak_composition=None,
            last_acq_value=None, confidence_pct=0,
        )

    rep = nav.report()
    n_stable = rep.n_stable

    # LHS-fallback or zero stable points → study has barely started, never "done".
    # This guards against the absurd case of declaring 95% confidence after
    # 0 evaluations (e.g. when the user supplies a brand-new --system that has
    # no data yet — every recommendation will be LHS-fallback).
    if cand.source == "LHS-fallback" or n_stable < _navigator_min_points():
        return CompletionAssessment(
            is_done=False,
            reason=(f"only {n_stable} stable points "
                    f"(need ≥ {_navigator_min_points()} before any 'done' verdict)"
                    if n_stable < _navigator_min_points()
                    else "still in LHS exploration phase — GP not fitted yet"),
            n_stable=n_stable,
            peak_predicted_value=cand.predicted_mean,
            peak_predicted_sigma=cand.predicted_std,
            peak_composition=cand.composition,
            last_acq_value=cand.acq_value,
            confidence_pct=int(min(100, n_stable / max(_navigator_min_points(), 1) * 50)),
        )

    # (a) hard convergence from acquisition
    if cand.converged:
        return CompletionAssessment(
            is_done=True,
            reason=f"acquisition ({rep.acquisition}={cand.acq_value:.4g}) below threshold",
            n_stable=n_stable,
            peak_predicted_value=cand.predicted_mean,
            peak_predicted_sigma=cand.predicted_std,
            peak_composition=cand.composition,
            last_acq_value=cand.acq_value,
            confidence_pct=95,
        )

    # (c) stuck-loop check
    if consecutive_duplicate_suggestions >= _done_dedupe_repeats():
        return CompletionAssessment(
            is_done=True,
            reason=f"{consecutive_duplicate_suggestions} consecutive duplicate suggestions — loop saturated",
            n_stable=n_stable,
            peak_predicted_value=cand.predicted_mean,
            peak_predicted_sigma=cand.predicted_std,
            peak_composition=cand.composition,
            last_acq_value=cand.acq_value,
            confidence_pct=85,
        )

    # (b) sigma-at-peak check
    sigma_thresh = _done_peak_sigma_max()
    min_pts = _done_min_points()
    if (n_stable >= min_pts
        and np.isfinite(cand.predicted_std)
        and cand.predicted_std < sigma_thresh
    ):
        return CompletionAssessment(
            is_done=True,
            reason=f"σ at predicted optimum ({cand.predicted_std:.3f}) < {sigma_thresh}, "
                   f"with {n_stable} stable points",
            n_stable=n_stable,
            peak_predicted_value=cand.predicted_mean,
            peak_predicted_sigma=cand.predicted_std,
            peak_composition=cand.composition,
            last_acq_value=cand.acq_value,
            confidence_pct=80,
        )

    # Not done — provide useful "why not"
    if n_stable < min_pts:
        reason = f"need ≥ {min_pts} stable points (have {n_stable})"
    else:
        reason = (f"σ at optimum still {cand.predicted_std:.3f} > {sigma_thresh}; "
                  f"acquisition {cand.acq_value:.4g} > threshold")
    return CompletionAssessment(
        is_done=False, reason=reason,
        n_stable=n_stable,
        peak_predicted_value=cand.predicted_mean if cand else float("nan"),
        peak_predicted_sigma=cand.predicted_std if cand else float("nan"),
        peak_composition=cand.composition if cand else None,
        last_acq_value=cand.acq_value if cand else None,
        confidence_pct=int(min(100, n_stable / max(min_pts, 1) * 50)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _format_report(
    report: NavigatorReport,
    structure_file: Path | None = None,
    engine: str = "castep",
    completion: CompletionAssessment | None = None,
) -> str:
    cand = report.candidate
    sys_label = report.target_system.label() if report.target_system else "?"
    N = report.target_system.n_metals if report.target_system else 0
    lines: list[str] = []

    lines.append("  ── Navigator ──────────────────────────────────────────────")
    lines.append(f"    System     : {sys_label}  (N={N})")
    if len(report.systems_in_data) > 1:
        others = [s for s in report.systems_in_data if s != sys_label]
        lines.append(f"    Also in data: {', '.join(others)}  (transfer learning)")
    lines.append(
        f"    Data       : {report.n_stable} stable | "
        f"{report.n_failed} failed (this system)"
    )
    lines.append(f"    Target     : {report.target}  ({report.mode})")
    lines.append(f"    Kernel     : {report.selected_kernel}")
    lines.append(f"    Acquisition: {report.acquisition}")

    # Best so far
    if np.isfinite(report.current_best_value):
        if report.current_best_composition and len(report.current_best_composition) > 2:
            comp_str = " ".join(f"{m}={x:.3f}" for m, x in report.current_best_composition.items())
            lines.append(f"    Current best: {report.current_best_value:.3f}  at {comp_str}")
        else:
            lines.append(
                f"    Current best: {report.current_best_value:.3f}  "
                f"at x = {report.current_best_x:.4f}"
            )
    else:
        lines.append("    Current best: (no data yet for this system)")

    lines.append("")
    lines.append("  ── Recommendation ─────────────────────────────────────────")

    if cand.composition and len(cand.composition) > 2:
        # N-component: show full composition
        lines.append(f"    Next       : {cand.composition_str()}")
    else:
        lines.append(f"    Next x     : {cand.concentration:.4f}")

    if np.isfinite(cand.predicted_mean) and np.isfinite(cand.predicted_std):
        lines.append(f"    Predicted  : {cand.predicted_mean:.3f} ± {cand.predicted_std:.3f}")
    else:
        lines.append("    Predicted  : (LHS — surrogate not fitted)")

    if cand.p_feasible is not None and np.isfinite(cand.p_feasible):
        lines.append(f"    P(stable)  : {int(round(cand.p_feasible * 100))}%")
    else:
        lines.append("    P(stable)  : (n/a)")

    if cand.acq_value is not None and np.isfinite(cand.acq_value):
        label = report.acquisition if report.acquisition in {"EI", "CEI", "UCB"} else "Acq"
        lines.append(f"    {label} score  : {cand.acq_value:.4f}")
    lines.append(f"    Source     : {cand.source}")

    if cand.converged:
        lines.append("")
        lines.append(f"    !! Convergence: {report.acquisition} below threshold "
                     f"({_navigator_ei_threshold():g}). Optimum likely found.")
    if cand.source == "LHS-fallback":
        lines.append("")
        lines.append(f"    !! Fallback: only {report.n_stable} stable points "
                     f"in this system — GP not fitted. Using LHS.")

    if completion is not None:
        lines.append("")
        lines.append("  ── Study Completion ───────────────────────────────────────")
        lines.append(f"    Status     : {completion.short_summary()}")
        lines.append(f"    Confidence : {completion.confidence_pct}%")

    if structure_file is not None and cand.system is not None:
        lines.append("")
        lines.append("  ── Run this in VCAForge ───────────────────────────────────")
        lines.append("    " + build_vcaforge_command(structure_file, cand, engine=engine))

    return "\n".join(lines)


def _build_cli_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="navigator",
        description="VCAForge Bayesian Optimization Navigator (N-component, descriptor-space)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python navigator.py results.csv\n"
            "  python navigator.py ../analysis/ --system Zr-Nb-C\n"
            "  python navigator.py ../analysis/ --system Ti-Nb-V-C \\\n"
            "       --generate-cmd ../structures/TiC_base.cif\n"
            "  python navigator.py --aggregate ../analysis/ --system Ti-Nb-C\n"
            "  python navigator.py --loop ../structures/TiC_base.cif \\\n"
            "       --system Ti-Nb-V-C --max-iter 8\n"
        ),
    )
    p.add_argument("path", type=Path, nargs="?", default=None,
                   help="Path to a CSV file OR a directory to scan (auto-detected).")
    p.add_argument("--dir", type=Path, default=None)
    p.add_argument("--csv", type=Path, default=None)
    p.add_argument("--target", type=str, default=None)
    p.add_argument("--mode", choices=["maximize", "minimize"], default=None)
    p.add_argument("--acq", type=str, default=None, choices=["CEI", "EI", "UCB", "MaxVar"])
    p.add_argument("--system", type=str, default=None,
                   help="Target system (e.g. 'Ti-Nb-C' or 'Ti-Nb-V-C'). "
                        "Filters recommendation; GP still trains on all systems.")
    p.add_argument("--list-systems", action="store_true",
                   help="List unique systems found and exit.")
    p.add_argument("--list-columns", action="store_true",
                   help="Print available columns and exit.")
    p.add_argument("--generate-cmd", type=Path, default=None, metavar="STRUCTURE",
                   help="Print a ready-to-paste VCAForge command using STRUCTURE.")
    p.add_argument("--engine", type=str, default="castep")
    p.add_argument("--loop", type=Path, default=None, metavar="STRUCTURE",
                   help="Active-learning loop: run VCAForge, ingest result, suggest next, repeat.")
    p.add_argument("--max-iter", type=int, default=None,
                   help=f"Loop iteration cap (default: {_loop_max_iter()}).")
    p.add_argument("--aggregate", type=Path, default=None, metavar="DIR",
                   help="Aggregate all CSVs under DIR for --system into a master CSV. "
                        "If --system is omitted, one master CSV is created per system found.")
    p.add_argument("--master-out", type=Path, default=None,
                   help="Output path for --aggregate (default: <DIR>/<s>_master.csv).")
    p.add_argument("--dedup-policy", choices=["best", "mean", "all"], default="best",
                   help="How to merge repeated compositions in aggregate: "
                        "'best' keeps highest target, 'mean' averages numeric "
                        "columns and adds n_repeats, 'all' keeps every row "
                        "(default: best).")
    p.add_argument("--check-done", action="store_true",
                   help="Run completion assessment and print whether study is done.")
    p.add_argument("--supercell", type=int, default=None,
                   help="Supercell dimension (e.g. 3 for 3x3x3). Quantizes recommendation to integer atoms.")
    p.add_argument("--crystal-mode", type=str, choices=["vca", "sqs", "direct"], default=None,
                   help="Crystal preparation mode for main.py (e.g. sqs).")
    p.add_argument("--template", type=str, default=None,
                   help="Template element for main.py substitution (e.g. W).")
    p.add_argument("--verbose", action="store_true")
    return p


def _resolve_input_paths(args: argparse.Namespace) -> tuple[Path | None, Path | None]:
    csv_path: Path | None = args.csv
    base_dir: Path | None = args.dir
    if csv_path is None and base_dir is None and args.path is not None:
        p = args.path.expanduser()
        if p.is_file():
            csv_path = p
        elif p.is_dir():
            base_dir = p
        else:
            if str(p).endswith(".csv"):
                csv_path = p
            else:
                base_dir = p
    if csv_path is None and base_dir is None:
        base_dir = Path.cwd()
    return csv_path, base_dir


def _cli_main(argv: list[str] | None = None) -> int:
    args = _build_cli_parser().parse_args(argv)

    if args.verbose:
        logging.basicConfig(level=logging.INFO,
                            format="  [%(levelname)s] %(message)s",
                            stream=sys.stderr)

    csv_path, base_dir = _resolve_input_paths(args)

    # ── SQS sites calculation ─────────────────────────────────────────────
    sqs_sites = None
    if args.supercell:
        struct_file = args.loop or args.generate_cmd
        base_sites = 1
        if struct_file and struct_file.exists():
            from core_physics import load_crystal
            try:
                crystal = load_crystal(struct_file)
                # To accurately count metal sites, we need the system metals
                temp_sys = None
                if args.system:
                    temp_sys = parse_system_from_string(args.system)
                elif csv_path:
                    temp_sys = parse_system_from_csv(csv_path)

                if temp_sys:
                    base_sites = sum(1 for s in crystal.species if s in temp_sys.metals)
            except Exception:
                pass
        sqs_sites = base_sites * (args.supercell ** 3)

    # ── --aggregate (delegate to aggregator module) ───────────────────────
    if args.aggregate is not None:
        try:
            from aggregator import ResultAggregator
        except ImportError as exc:
            print(f"  ERROR: --aggregate needs aggregator.py: {exc}", file=sys.stderr)
            return 2
        try:
            agg = ResultAggregator(
                target=args.target, mode=args.mode or "maximize",
                dedupe_policy=args.dedup_policy,
            )
            if args.system is None:
                # Aggregate every discovered system, one master CSV each
                results = agg.aggregate_all_systems(args.aggregate)
                if not results:
                    print(f"  ERROR: No parseable systems under {args.aggregate}",
                          file=sys.stderr)
                    return 2
                print(f"  Aggregated {len(results)} system(s):")
                for label, out in sorted(results.items()):
                    print(f"    {label:<20} → {out}")
                return 0
            sys_spec = parse_system_from_string(args.system)
            out = agg.aggregate(args.aggregate, sys_spec, output_path=args.master_out)
            print(f"  Aggregated master CSV → {out}")
            return 0
        except (FileNotFoundError, ValueError) as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            return 2

    # ── --list-columns ────────────────────────────────────────────────────
    if args.list_columns:
        try:
            ing = DataIngestor(csv_path=csv_path, base_dir=base_dir, target="concentration")
            files = ing._discover_csv_files()
            cols = ing._collect_available_columns(files)
        except (FileNotFoundError, ValueError) as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            return 2
        print(f"  Columns across {len(files)} CSV file(s):")
        for c in cols:
            print(f"    - {c}")
        return 0

    # ── --list-systems ────────────────────────────────────────────────────
    if args.list_systems:
        try:
            ing = DataIngestor(csv_path=csv_path, base_dir=base_dir,
                               target=args.target or _navigator_target())
            files = ing._discover_csv_files()
        except (FileNotFoundError, ValueError) as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            return 2
        seen: dict[str, int] = {}
        for f in files:
            sys_spec = parse_system_from_csv(f)
            if sys_spec is not None:
                seen[sys_spec.label()] = seen.get(sys_spec.label(), 0) + 1
        if not seen:
            print(f"  No parseable systems in {len(files)} file(s).")
            return 1
        print(f"  Systems found in {len(files)} CSV file(s):")
        for label, count in sorted(seen.items(), key=lambda kv: -kv[1]):
            print(f"    {label:<20} ({count} file{'s' if count != 1 else ''})")
        return 0

    # ── --loop (delegate to navigator_loop) ───────────────────────────────
    if args.loop is not None:
        try:
            from navigator_loop import run_loop
        except ImportError as exc:
            print(f"  ERROR: --loop needs navigator_loop.py: {exc}", file=sys.stderr)
            return 2
        return run_loop(
            structure_file=args.loop,
            csv_path=csv_path, base_dir=base_dir,
            target=args.target, mode=args.mode, acquisition=args.acq,
            target_system=args.system, engine=args.engine,
            max_iter=args.max_iter or _loop_max_iter(),
            verbose=args.verbose,
            sqs_sites=sqs_sites,
            crystal_mode=args.crystal_mode,
            template=args.template,
        )

    # ── Standard one-shot recommendation ──────────────────────────────────
    try:
        nav = NavigatorOrchestrator(
            csv_path=csv_path, base_dir=base_dir,
            target=args.target, mode=args.mode, acquisition=args.acq,
            target_system=args.system,
            sqs_sites=sqs_sites,
        )
        nav.suggest()
        report = nav.report()
        completion = assess_completion(nav) if args.check_done else None
    except FileNotFoundError as exc:
        print(f"  ERROR: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"  ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"  ERROR: unexpected: {exc}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    print(_format_report(report, structure_file=args.generate_cmd,
                         engine=args.engine, completion=completion))
    return 0


if __name__ == "__main__":
    sys.exit(_cli_main())


# ─────────────────────────────────────────────────────────────────────────────
# Multi-fidelity support — Hierarchical kriging (simplified Kennedy-O'Hagan)
# ─────────────────────────────────────────────────────────────────────────────
#
# Idea
# ────
# DFT runs at different XC functionals give different costs and accuracies:
#     LDA      ~30% faster, ~5-10% systematic bias on lattice/elastic constants
#     PBE      reference (your current default)
#     PBESOL   between LDA and PBE
# Doing all-PBE is expensive. Doing all-LDA is biased. Doing a mix lets you
# explore broadly with cheap LDA and refine with PBE.
#
# Hierarchical kriging (simplified):
#     y_high(x) = ρ · y_low(x) + δ(x) + ε
# where ρ is a scalar learned from points evaluated at BOTH fidelities, and
# δ(x) is a GP on the (limited) high-fidelity data minus ρ·μ_low.
#
# When no overlap exists (no point evaluated at both fidelities), we fall
# back to two independent GPs and bias-correct using the global mean shift.

@dataclass
class FidelityLevel:
    """Encapsulates one DFT fidelity (XC functional)."""
    name: str                    # "lda" | "pbe" | "pbesol" | ...
    cost_factor: float = 1.0     # relative compute cost (PBE = 1.0)


# Default fidelity ladder. Users can override via config.NAVIGATOR_FIDELITY_LEVELS.
DEFAULT_FIDELITY_LADDER: tuple[FidelityLevel, ...] = (
    FidelityLevel(name="lda",    cost_factor=0.7),
    FidelityLevel(name="pbesol", cost_factor=0.9),
    FidelityLevel(name="pbe",    cost_factor=1.0),    # reference / "high" fidelity
)


def _fidelity_ladder() -> tuple[FidelityLevel, ...]:
    custom = _cfg("NAVIGATOR_FIDELITY_LEVELS", None)
    if custom and isinstance(custom, (list, tuple)) and all(isinstance(c, FidelityLevel) for c in custom):
        return tuple(custom)
    return DEFAULT_FIDELITY_LADDER


def _high_fidelity() -> str:
    """Returns the highest-fidelity name (last in ladder by convention)."""
    return _fidelity_ladder()[-1].name


class MultiFidelitySurrogate:
    """Lightweight hierarchical-kriging surrogate over (descriptor, fidelity).

    Fits one GP per fidelity (independent) and learns a scalar correction
    factor ρ from points that were evaluated at multiple fidelities. When
    asked to predict at the highest fidelity:
        μ_hf(x) = ρ · μ_lf(x) + δ(x)    if a low-fidelity GP exists nearby
        μ_hf(x) = μ_hf(x) [own GP]      otherwise
    Uncertainty propagates via standard variance addition.

    This is NOT a full Kennedy-O'Hagan MFGP. It's a pragmatic 100-line
    approximation that captures the main benefit (use cheap data to
    inform expensive predictions) without a multi-week BoTorch port.
    """

    def __init__(self) -> None:
        self.gps: dict[str, GaussianProcessRegressor] = {}     # fidelity → GPR
        self.gpc: GaussianProcessClassifier | None = None       # one shared classifier
        self.descriptor_length: int = 0
        self.rho: dict[str, float] = {}                         # low-fidelity → ρ
        self.fitted: bool = False
        self.selected_kernel: str = ""
        self.n_per_fidelity: dict[str, int] = {}

    def fit(
        self,
        X_per_fid: dict[str, np.ndarray],
        y_per_fid: dict[str, np.ndarray],
        X_all: np.ndarray,
        labels_all: np.ndarray,
    ) -> "MultiFidelitySurrogate":
        """Fit one GP per fidelity. X_per_fid keys are fidelity names."""
        min_pts = _navigator_min_points()
        self.descriptor_length = max(
            (X.shape[1] for X in X_per_fid.values() if len(X)), default=0,
        )
        self.n_per_fidelity = {f: len(X) for f, X in X_per_fid.items()}

        # Total data across all fidelities for the min-points check
        total = sum(len(X) for X in X_per_fid.values())
        if total < min_pts:
            log.info("MFGP not fitted: total %d < %d.", total, min_pts)
            self.fitted = False
            return self

        try:
            # Fit one GP per fidelity that has enough points (use min/2 as the per-level threshold)
            per_level_min = max(2, min_pts // 2)
            kernel_names: dict[str, str] = {}
            for fid, X in X_per_fid.items():
                if len(X) < per_level_min:
                    log.info("Fidelity '%s' has only %d points (< %d); skipping its GP.",
                             fid, len(X), per_level_min)
                    continue
                gpr, kname = _fit_best_gpr(X, y_per_fid[fid])
                self.gps[fid] = gpr
                kernel_names[fid] = kname

            if not self.gps:
                raise RuntimeError("No fidelity has enough points for a GP.")

            # Learn ρ between each low fidelity and the high fidelity using the
            # high-fidelity data: ρ = mean(y_hf / μ_lf(x_hf)) where μ_lf is the
            # low-fidelity GP's prediction at the high-fidelity points.
            hf = _high_fidelity()
            if hf in self.gps and len(X_per_fid.get(hf, [])) >= per_level_min:
                X_hf, y_hf = X_per_fid[hf], y_per_fid[hf]
                for fid, gpr in self.gps.items():
                    if fid == hf:
                        continue
                    try:
                        mu_lf, _ = gpr.predict(self._pad(X_hf, gpr), return_std=True)
                        # Robust ρ via least-squares: y_hf ≈ ρ · μ_lf
                        denom = float(np.dot(mu_lf, mu_lf))
                        if denom > 1e-9:
                            self.rho[fid] = float(np.dot(mu_lf, y_hf) / denom)
                        else:
                            self.rho[fid] = 1.0
                    except Exception as exc:  # noqa: BLE001
                        log.debug("ρ for %s failed: %s", fid, exc)
                        self.rho[fid] = 1.0

            # Shared feasibility classifier across ALL fidelities
            self._fit_classifier(X_all, labels_all)

            self.selected_kernel = " | ".join(f"{f}:{k}" for f, k in kernel_names.items())
            self.fitted = True
        except Exception as exc:  # noqa: BLE001
            log.warning("MFGP fit failed (%s) → fallback to LHS.", exc)
            self.fitted = False
        return self

    def predict(
        self,
        X: np.ndarray,
        fidelity: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict at the requested fidelity (default: highest available).

        If predicting at the highest fidelity but its GP has very little data
        AND a low-fidelity GP is available, blend with bias correction.
        """
        if not self.fitted:
            raise RuntimeError("predict() before successful fit().")
        fid = fidelity or _high_fidelity()
        X = np.atleast_2d(np.asarray(X, dtype=float))

        # Direct prediction at the requested fidelity
        if fid in self.gps:
            X_p = self._pad(X, self.gps[fid])
            mu, sigma = self.gps[fid].predict(X_p, return_std=True)
            mu = np.asarray(mu, dtype=float)
            sigma = np.asarray(sigma, dtype=float)
            # If fidelity has few points, blend with low-fidelity prediction
            if self.n_per_fidelity.get(fid, 0) < _navigator_min_points():
                mu, sigma = self._blend_with_lower(X, fid, mu, sigma)
            return mu, sigma

        # Fidelity not directly available — try blending from a lower one
        for fid_low in [fl.name for fl in _fidelity_ladder() if fl.name != fid]:
            if fid_low in self.gps:
                X_p = self._pad(X, self.gps[fid_low])
                mu_lf, sigma_lf = self.gps[fid_low].predict(X_p, return_std=True)
                rho = self.rho.get(fid_low, 1.0)
                return rho * np.asarray(mu_lf), np.abs(rho) * np.asarray(sigma_lf)
        raise RuntimeError(f"No GP available for fidelity '{fid}' or any fallback.")

    def p_feasible(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if self.gpc is None:
            return np.ones(len(X), dtype=float)
        # Pad to the classifier's training-time descriptor length
        n_train = getattr(self.gpc, "n_features_in_", None) or self.descriptor_length
        if X.shape[1] < n_train:
            pad = np.zeros((X.shape[0], n_train - X.shape[1]), dtype=float)
            X = np.concatenate([X, pad], axis=1)
        elif X.shape[1] > n_train:
            X = X[:, :n_train]
        try:
            proba = self.gpc.predict_proba(X)
        except Exception as exc:  # noqa: BLE001
            log.warning("MFGP classifier failed (%s) → P=1.0", exc)
            return np.ones(len(X), dtype=float)
        classes = list(getattr(self.gpc, "classes_", [0, 1]))
        col = classes.index(1) if 1 in classes else proba.shape[1] - 1
        return np.asarray(proba[:, col], dtype=float)

    # ── Internals ─────────────────────────────────────────────────────────

    def _pad(self, X: np.ndarray, gpr: GaussianProcessRegressor) -> np.ndarray:
        n_train = getattr(gpr, "n_features_in_", X.shape[1])
        if X.shape[1] == n_train:
            return X
        if X.shape[1] > n_train:
            return X[:, :n_train]
        pad = np.zeros((X.shape[0], n_train - X.shape[1]), dtype=float)
        return np.concatenate([X, pad], axis=1)

    def _blend_with_lower(
        self, X: np.ndarray, fid: str, mu_hf: np.ndarray, sigma_hf: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Blend high-fidelity prediction with bias-corrected low-fidelity."""
        for fid_low in [fl.name for fl in _fidelity_ladder() if fl.name != fid and fl.name in self.gps]:
            X_p = self._pad(X, self.gps[fid_low])
            mu_lf, sigma_lf = self.gps[fid_low].predict(X_p, return_std=True)
            rho = self.rho.get(fid_low, 1.0)
            # Weighted average inversely by variance
            v_hf = sigma_hf ** 2 + 1e-9
            v_lf = (np.abs(rho) * sigma_lf) ** 2 + 1e-9
            w_hf = 1.0 / v_hf
            w_lf = 1.0 / v_lf
            w_total = w_hf + w_lf
            mu = (w_hf * mu_hf + w_lf * (rho * np.asarray(mu_lf))) / w_total
            sigma = np.sqrt(1.0 / w_total)
            return mu, sigma
        return mu_hf, sigma_hf

    def _fit_classifier(self, X_all: np.ndarray, labels_all: np.ndarray) -> None:
        if len(np.unique(labels_all)) < 2:
            log.info("Single-class feasibility data — GPC skipped.")
            self.gpc = None
            return
        kernel = ConstantKernel(1.0) * RBF(length_scale_bounds=(1e-3, 10.0))
        gpc = GaussianProcessClassifier(kernel=kernel, n_restarts_optimizer=3, random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gpc.fit(X_all, labels_all)
        self.gpc = gpc


def _fit_best_gpr(X: np.ndarray, y: np.ndarray) -> tuple[GaussianProcessRegressor, str]:
    """Helper: fit several kernels, return the best by log-marginal likelihood."""
    kernel_choice = _navigator_kernel().lower()
    if kernel_choice == "matern":
        candidates = [("matern25", Matern(nu=2.5, length_scale_bounds=(1e-3, 10.0)) + WhiteKernel())]
    elif kernel_choice == "rbf":
        candidates = [("rbf", RBF(length_scale_bounds=(1e-3, 10.0)) + WhiteKernel())]
    else:
        candidates = [
            ("matern25", Matern(nu=2.5, length_scale_bounds=(1e-3, 10.0)) + WhiteKernel()),
            ("rbf",      RBF(length_scale_bounds=(1e-3, 10.0)) + WhiteKernel()),
            ("matern15", Matern(nu=1.5, length_scale_bounds=(1e-3, 10.0)) + WhiteKernel()),
        ]
    best_lml = -np.inf
    best_gpr: GaussianProcessRegressor | None = None
    best_name = ""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, kernel in candidates:
            gpr = GaussianProcessRegressor(
                kernel=kernel, n_restarts_optimizer=5,
                normalize_y=True, random_state=0,
            )
            try:
                gpr.fit(X, y)
            except Exception as exc:  # noqa: BLE001
                log.debug("GPR %s failed: %s", name, exc)
                continue
            lml = float(gpr.log_marginal_likelihood_value_)
            if lml > best_lml:
                best_lml, best_gpr, best_name = lml, gpr, name
    if best_gpr is None:
        raise RuntimeError("All GPR kernels failed.")
    pretty = {"matern25": "Matern-5/2 + White", "rbf": "RBF + White",
              "matern15": "Matern-3/2 + White"}.get(best_name, best_name)
    return best_gpr, pretty

if not log.handlers:
    log.addHandler(logging.NullHandler())


# ─────────────────────────────────────────────────────────────────────────────
# Config accessors  (getattr with fallback — backward compatible)
# ─────────────────────────────────────────────────────────────────────────────
