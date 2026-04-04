"""
core/elasticity.py  —  Pure-Python elastic constants engine.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Replaces the entire elastic-constants-master repository.
No subprocess calls.  No scipy dependency.  All logic is
invoked as ordinary Python functions from elastic_workflow.py.

Public API
──────────
  VEC_VALENCES            : dict[str, int]   element → valence electrons
  vec_for_concentration   : float            VEC at concentration x
  nextra_bands_for        : int              adaptive nextra_bands
  vec_stability_band      : str              "green" | "yellow" | "red"

  StrainPattern           : dataclass        strain patterns for a lattice type
  get_strain_patterns     : list[StrainPattern]   patterns for symmetry code
  generate_strain_cells   : list[tuple[str, np.ndarray]]   (name, strain_voigt)
  write_cijdat            : None             write .cijdat file

  read_stress             : np.ndarray|None  parse stress from .castep
  fit_cij_from_stress     : dict             C11/C12/C44 + B/G/E/ν or {"error": …}
  cubic_derived           : dict             polycrystalline properties
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Valence electron table  (outer d + s electrons only, not core)
# ─────────────────────────────────────────────────────────────────────────────

VEC_VALENCES: dict[str, int] = {
    # 3d transition metals
    "Sc": 3, "Ti": 4, "V": 5, "Cr": 6, "Mn": 7,
    "Fe": 8, "Co": 9, "Ni": 10, "Cu": 11, "Zn": 12,
    # 4d transition metals
    "Y": 3, "Zr": 4, "Nb": 5, "Mo": 6, "Tc": 7,
    "Ru": 8, "Rh": 9, "Pd": 10, "Ag": 11, "Cd": 12,
    # 5d
    "Hf": 4, "Ta": 5, "W": 6, "Re": 7,
    # Non-metals (sp electrons)
    "B": 3, "C": 4, "N": 5, "O": 6, "F": 7,
    "Al": 3, "Si": 4, "P": 5, "S": 6,
}

# VEC stability thresholds (from literature, e.g. Mei et al. 2014 Ti(1-x)NbxC)
VEC_YELLOW   = 8.2   # start of marginal zone
VEC_RED      = 8.4   # Born instability likely


def vec_for_concentration(
    species_a: str,
    species_b: str,
    x: float,
    nonmetal: str | None = None,
) -> float:
    """
    Valence Electron Count for (A_{1-x} B_x) nonmetal system.

    VEC = (1-x)·val(A) + x·val(B) + val(nonmetal)
    For a pure compound (no nonmetal), the nonmetal term is omitted.
    """
    val_a = VEC_VALENCES.get(species_a.capitalize(), 4)
    val_b = VEC_VALENCES.get(species_b.capitalize(), 5)
    val_nm = VEC_VALENCES.get(nonmetal.capitalize(), 4) if nonmetal else 0
    return (1.0 - x) * val_a + x * val_b + val_nm


def vec_stability_band(vec: float) -> str:
    """Return 'green', 'yellow', or 'red' stability category."""
    if vec <= VEC_YELLOW:
        return "green"
    if vec <= VEC_RED:
        return "yellow"
    return "red"


def nextra_bands_for(x: float, vec: float) -> int:
    """
    Adaptive nextra_bands based on concentration and VEC.

    Physics rationale
    ─────────────────
    Pure endpoints (x=0, x=1) have integer nuclear charges and
    a well-defined Fermi level → fewer extra bands needed.

    VCA intermediates have fractional charges — the Fermi surface
    is smeared out in proportion to |VEC − 8|.  The further VEC
    deviates from the ideal 8 electrons/cell, the more empty bands
    are needed to correctly represent the density matrix.

    Formula
    ───────
      endpoints    → 10
      VCA          → 15 + int(|VEC − 8.0| × 20)

    Examples: VEC=8.0 → 15, VEC=8.5 → 25, VEC=8.2 → 19
    """
    if x < 1e-5 or x > 1.0 - 1e-5:
        return 10
    return 15 + int(abs(vec - 8.0) * 20)


# ─────────────────────────────────────────────────────────────────────────────
# Strain pattern generation  (replaces generate_strain.py logic)
# ─────────────────────────────────────────────────────────────────────────────

# Strain patterns by lattice symmetry code (same convention as generate_strain.py)
# Each row is [e1, e2, e3, e4, e5, e6] in IRE Voigt notation.
_STRAIN_PATTERNS: dict[int, list[list[float]]] = {
    1: [  # Triclinic — all 6 independent strains
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
    ],
    2: [  # Monoclinic
        [1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 1],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
    ],
    3: [  # Orthorhombic
        [1, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1],
    ],
    4: [  # Tetragonal
        [1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 1],
    ],
    5: [  # Cubic — single pattern e1+e4
        [1, 0, 0, 1, 0, 0],
    ],
    6: [  # Trigonal-Low
        [1, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0],
    ],
    7: [  # Trigonal-High / Hexagonal
        [0, 0, 1, 0, 0, 0],
        [1, 0, 0, 1, 0, 0],
    ],
}


def pointgroup_to_lattice_code(pg: int) -> int:
    """Convert CASTEP point group index → lattice symmetry code."""
    if pg <= 0:
        raise ValueError(f"Invalid point group: {pg}")
    if pg <= 2:
        return 1  # Triclinic
    if pg <= 5:
        return 2  # Monoclinic
    if pg <= 8:
        return 3  # Orthorhombic
    if pg <= 15:
        return 4  # Tetragonal
    if pg <= 17:
        return 6  # Trigonal-Low
    if pg <= 27:
        return 7  # Trigonal-High / Hexagonal
    if pg <= 32:
        return 5  # Cubic
    raise ValueError(f"Point group {pg} out of range")


@dataclass
class StrainStep:
    """One strained cell: its name tag and Voigt strain vector."""
    pattern_idx: int        # 1-based pattern index
    step_idx:    int        # 1-based step within the pattern (1…n_steps*2)
    magnitude:   float      # signed strain magnitude
    pattern_vec: list[float]                   # un-scaled [0/1] pattern
    strain_voigt: np.ndarray = field(repr=False)  # final scaled Voigt vector

    @property
    def name(self) -> str:
        """Name fragment _cij__p__s (matches generate_strain.py convention)."""
        return f"_cij__{self.pattern_idx}__{self.step_idx}"


def generate_strain_steps(
    lattice_vecs: np.ndarray,     # shape (3, 3), rows = vectors in Å
    lattice_code: int = 5,
    max_strain: float = 0.003,
    n_steps: int = 3,
) -> list[StrainStep]:
    """
    Generate all StrainStep objects for the given lattice symmetry.

    Each pattern is applied at n_steps positive and n_steps negative
    magnitudes (2*n_steps total per pattern), matching the original
    generate_strain.py behaviour exactly.

    lattice_vecs rows:  [[a_x, a_y, a_z], [b_x, b_y, b_z], [c_x, c_y, c_z]]
    """
    if lattice_code not in _STRAIN_PATTERNS:
        raise ValueError(f"Unsupported lattice code: {lattice_code}")

    patterns = _STRAIN_PATTERNS[lattice_code]
    a_len = float(np.linalg.norm(lattice_vecs[0]))
    b_len = float(np.linalg.norm(lattice_vecs[1]))
    c_len = float(np.linalg.norm(lattice_vecs[2]))
    lens = [a_len, b_len, c_len, b_len, a_len, a_len]  # per IRE Voigt index

    steps: list[StrainStep] = []
    for patt_idx, pattern in enumerate(patterns, 1):
        step_counter = 0
        for a in range(n_steps):
            for neg in (False, True):
                step_counter += 1
                frac = (a + 1) / n_steps
                mag = frac * max_strain * (-1 if neg else 1)
                # Scale each component by its lattice vector length ratio
                # (diagonal → displacement/length, off-diag → half that)
                voigt = np.zeros(6)
                for i, p in enumerate(pattern):
                    if p == 0:
                        continue
                    disp = p * mag
                    if i < 3:
                        voigt[i] = disp / lens[i]
                    else:
                        voigt[i] = 0.5 * disp / lens[i]
                steps.append(
                    StrainStep(
                        pattern_idx=patt_idx,
                        step_idx=step_counter,
                        magnitude=mag,
                        pattern_vec=list(pattern),
                        strain_voigt=voigt,
                    )
                )
    return steps


def write_cijdat(
    path: Path,
    lattice_code: int,
    n_steps: int,
    max_strain: float,
    seed: str,
    strain_steps: list[StrainStep],
) -> None:
    """
    Write a .cijdat file in the format expected by elastic_analysis.read_cijdat.
    Fully compatible replacement for generate_strain.py's cijdat output.
    """
    lines: list[str] = []
    lines.append(f"{lattice_code} {n_steps * 2} 0 0")
    lines.append(str(max_strain))

    for step in strain_steps:
        sv = step.strain_voigt
        # IRE convention for the 3×3 matrix in .cijdat:
        # row0: e11  e12  e13   (e12 = voigt[5]/2, e13 = voigt[4]/2)
        # row1: e12  e22  e23   (e23 = voigt[3]/2)
        # row2: e13  e23  e33
        e11, e22, e33 = sv[0], sv[1], sv[2]
        e23, e13, e12 = sv[3] / 2, sv[4] / 2, sv[5] / 2
        lines.append(f"{seed}{step.name}")
        lines.append(f"{e11} {e12} {e13}")
        lines.append(f"{e12} {e22} {e23}")
        lines.append(f"{e13} {e23} {e33}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def apply_strain_to_cell(
    source_cell: Path,
    dest_cell: Path,
    strain_voigt: np.ndarray,
) -> None:
    """
    Write a strained copy of source_cell preserving all MIXTURE / VCA syntax.

    Only LATTICE_CART is modified — POSITIONS_FRAC (with MIXTURE tags) is
    invariant under homogeneous strain because fractional coordinates are
    defined relative to the lattice vectors.

    Deformation gradient:  F = I + ε
    New lattice:           L_new = L_old @ F.T   (rows = vectors)
    """
    text = source_cell.read_text(encoding="utf-8", errors="replace")

    e11, e22, e33 = strain_voigt[0], strain_voigt[1], strain_voigt[2]
    e23, e13, e12 = strain_voigt[3] / 2, strain_voigt[4] / 2, strain_voigt[5] / 2
    F = np.array([
        [1.0 + e11, e12,       e13      ],
        [e12,       1.0 + e22, e23      ],
        [e13,       e23,       1.0 + e33],
    ])

    _RE_LAT = re.compile(
        r"(%BLOCK\s+LATTICE_CART\s*\n)(.*?)(%ENDBLOCK\s+LATTICE_CART)",
        re.DOTALL | re.I,
    )
    match = _RE_LAT.search(text)
    if match is None:
        dest_cell.write_text(text, encoding="utf-8")
        return

    header, body, footer = match.group(1), match.group(2), match.group(3)

    rows: list[str | None] = []
    lattice_vecs: list[np.ndarray] = []
    for line in body.splitlines(keepends=True):
        stripped = line.strip()
        if not stripped or stripped.lower() in {"ang", "bohr", "a.u.", "angstrom"}:
            rows.append(line)
            continue
        parts = stripped.split()
        if len(parts) >= 3:
            try:
                rows.append(None)
                lattice_vecs.append(np.array([float(p) for p in parts[:3]]))
                continue
            except ValueError:
                pass
        rows.append(line)

    if len(lattice_vecs) != 3:
        dest_cell.write_text(text, encoding="utf-8")
        return

    L = np.array(lattice_vecs)
    L_new = L @ F.T

    vec_iter = iter(L_new)
    new_body = "".join(
        f"  {next(vec_iter)[0]:22.15f} {0:.15f} {0:.15f}\n".replace(
            "  0.000000000000000", f"  {next(iter([0])):.15f}"
        )
        if line is None
        else line
        for line in rows
    )
    # Rebuild properly
    new_body_lines: list[str] = []
    vec_iter2 = iter(L_new)
    for line in rows:
        if line is None:
            v = next(vec_iter2)
            new_body_lines.append(f"  {v[0]:22.15f} {v[1]:22.15f} {v[2]:22.15f}\n")
        else:
            new_body_lines.append(line)

    new_lat_block = header + "".join(new_body_lines) + footer
    new_text = _RE_LAT.sub(new_lat_block, text)

    # Enforce FIX_ALL_CELL (constant volume at the strained geometry)
    new_text = re.sub(r"^\s*FIX_ALL_CELL\s*:.*$\n?", "", new_text, flags=re.M | re.I)
    new_text = re.sub(r"^\s*FIX_COM\s*:.*$\n?",     "", new_text, flags=re.M | re.I)
    new_text = re.sub(r"^\s*FIX_ALL_IONS\s*:.*$\n?","", new_text, flags=re.M | re.I)
    new_text = new_text.rstrip() + "\nFIX_ALL_CELL : true\n"

    dest_cell.write_text(new_text, encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Stress tensor parser  (identical logic to elastic_analysis.read_stress)
# ─────────────────────────────────────────────────────────────────────────────

_RE_STRESS_NEW = re.compile(
    r"^\s*([-+]?\d+\.\d+[Ee][+-]?\d+)"
    r"\s+([-+]?\d+\.\d+[Ee][+-]?\d+)"
    r"\s+([-+]?\d+\.\d+[Ee][+-]?\d+)\s+<-- S",
    re.MULTILINE,
)
_RE_STRESS_OLD = re.compile(
    r"\*\s+[xyz]\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s*\*"
)


def read_stress(castep_path: Path) -> np.ndarray | None:
    """
    Parse the FINAL symmetrised stress tensor from a CASTEP output file.
    Returns 6-element Voigt array [s11, s22, s33, s23, s13, s12] in GPa.
    Returns None on failure (empty file, no stress block).
    """
    if not castep_path.exists():
        return None
    try:
        text = castep_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if not text.strip():
        return None

    rows = _RE_STRESS_NEW.findall(text)
    if len(rows) >= 3:
        try:
            mat = np.array([[float(v) for v in r] for r in rows[-3:]])
            return np.array([mat[0,0], mat[1,1], mat[2,2], mat[1,2], mat[0,2], mat[0,1]])
        except (ValueError, IndexError):
            pass

    rows = _RE_STRESS_OLD.findall(text)
    if len(rows) >= 3:
        try:
            mat = np.array([[float(v) for v in r] for r in rows[-3:]])
            return np.array([mat[0,0], mat[1,1], mat[2,2], mat[1,2], mat[0,2], mat[0,1]])
        except (ValueError, IndexError):
            pass

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Cubic derived properties (Voigt-Reuss-Hill)
# ─────────────────────────────────────────────────────────────────────────────

def cubic_derived(c11: float, c12: float, c44: float) -> dict[str, float]:
    """
    Polycrystalline elastic properties from cubic Cij.

    Born stability for cubic: C11 > 0, C44 > 0, C11 > |C12|, C11 + 2*C12 > 0.
    Returns empty dict if the crystal is mechanically unstable.
    """
    if c11 <= 0 or c44 <= 0:
        return {}
    if c11 <= abs(c12):
        return {}
    if c11 + 2 * c12 <= 0:
        return {}

    b_v = (c11 + 2 * c12) / 3.0
    g_v = (c11 - c12 + 3 * c44) / 5.0

    denom = (c11 + 2 * c12) * (c11 - c12)
    s11 =  (c11 + c12) / denom
    s12 =  -c12 / denom
    s44 =  1.0 / c44

    b_r = 1.0 / (3 * (s11 + 2 * s12))
    g_r = 5.0 / (4 * (s11 - s12) + 3 * s44)

    b_h = (b_v + b_r) / 2
    g_h = (g_v + g_r) / 2
    e_h = 9 * b_h * g_h / (3 * b_h + g_h)
    nu  = (3 * b_h - 2 * g_h) / (2 * (3 * b_h + g_h))

    return {
        "C11": c11, "C12": c12, "C44": c44,
        "B_Voigt_GPa": b_v, "B_Reuss_GPa": b_r, "B_Hill_GPa": b_h,
        "G_Voigt_GPa": g_v, "G_Reuss_GPa": g_r, "G_Hill_GPa": g_h,
        "E_GPa":   e_h,
        "nu":      nu,
        "Zener_A": 2 * c44 / (c11 - c12),
        "Pugh_ratio": g_h / b_h,
        "Cauchy_pressure_GPa": c12 - c44,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Cij fitting  (OLS with intercept — handles non-zero residual stress)
# ─────────────────────────────────────────────────────────────────────────────

def fit_cij_from_stress(
    stress_list: list[np.ndarray],
    strain_list: list[np.ndarray],
    lattice_code: int = 5,
) -> dict[str, Any]:
    """
    Fit elastic constants from stress vs strain data.

    Parameters
    ──────────
    stress_list   List of Voigt stress vectors [s11,s22,s33,s23,s13,s12] (GPa).
    strain_list   Corresponding Voigt strain vectors [e11,e22,e33,2e23,2e13,2e12].
    lattice_code  Crystal symmetry (5 = cubic only currently supported).

    Returns
    ───────
    dict with str keys formatted to 4 decimal places:
      C11, C12, C44, B_Hill_GPa, G_Hill_GPa, E_GPa, nu, ...
    OR {"error": "<message>"} on failure — no exceptions propagate.

    OLS with intercept
    ──────────────────
    σ = C·ε + σ₀  (σ₀ is residual stress after GeomOpt, should be small).
    Using the intercept gives the correct slope regardless of σ₀, and
    gives a meaningful R² = 1 - SS_res/SS_tot instead of the R²→0 that
    a through-origin fit produces when σ₀ ≠ 0.
    """
    if lattice_code != 5:
        return {"error": f"lattice code {lattice_code}: only cubic (5) implemented"}

    n = len(stress_list)
    if n < 3:
        return {"error": f"only {n} stress tensors — need ≥ 3"}

    stress_arr = np.array(stress_list)   # (N, 6)
    strain_arr = np.array(strain_list)   # (N, 6)

    def _ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """OLS y = slope*x + intercept.  Returns (slope, R²)."""
        n_pts = len(x)
        sx, sy = float(x.sum()), float(y.sum())
        sxx = float(np.dot(x, x))
        sxy = float(np.dot(x, y))
        denom = n_pts * sxx - sx * sx
        if abs(denom) < 1e-30:
            return float("nan"), 0.0
        slope = (n_pts * sxy - sx * sy) / denom
        intercept = (sy - slope * sx) / n_pts
        y_pred = slope * x + intercept
        ss_res = float(np.sum((y - y_pred) ** 2))
        y_mean = sy / n_pts
        ss_tot = float(np.sum((y - y_mean) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
        return slope, r2

    # Cubic pattern e1+e4:
    #   C11 = ∂σ₁₁/∂ε₁₁   C12 = ∂σ₂₂/∂ε₁₁   C44 = ∂σ₂₃/∂(2ε₂₃)
    c11, r2_c11 = _ols(strain_arr[:, 0], stress_arr[:, 0])
    c12, r2_c12 = _ols(strain_arr[:, 0], stress_arr[:, 1])
    c44, r2_c44 = _ols(strain_arr[:, 3], stress_arr[:, 3])

    if any(np.isnan(v) for v in (c11, c12, c44)):
        return {"error": "NaN in OLS regression — strain data is degenerate"}

    r2_min = min(r2_c11, r2_c12, r2_c44)

    props = cubic_derived(c11, c12, c44)
    if not props:
        return {
            "error": (
                f"Born stability violated: "
                f"C11={c11:.1f}  C12={c12:.1f}  C44={c44:.1f} GPa"
            )
        }

    result: dict[str, Any] = {k: f"{v:.4f}" for k, v in props.items()}
    result["elastic_n_points"] = str(n)
    result["elastic_R2_min"]   = f"{r2_min:.4f}"
    if r2_min < 0.99:
        result["elastic_quality_note"] = (
            f"low R² ({r2_min:.3f}) — check SCF convergence or increase nextra_bands"
        )
    return result
