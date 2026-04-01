"""
elastic_analysis.py — Stress parser and Cij fitter for CASTEP 25.
───────────────────────────────────────────────────────────────────
Replaces castep.py + elastics.py from elastic-constants-master.
Supports both CASTEP 25.12 format (<-- S) and older ASCII-box format.
No scipy or external dependencies — pure numpy.

Public API
──────────
  read_stress(castep_path) → np.ndarray shape (6,) in GPa  [Voigt]
  count_atoms_from_castep(castep_path) → int
  read_cijdat(cijdat_path) → CijDat | None
  fit_cij(cijdat, seed_dir) → dict[str, str]
  delta_h_mix_note() → str   (physics warning for VCA)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Stress parser — supports CASTEP 25.12 (<-- S) and older ASCII-box
# ─────────────────────────────────────────────────────────────────────────────

# CASTEP 25.12:  three rows each ending with  <-- S
#   -6.21684372E-006   3.03062374E-005   3.23291890E-005 <-- S
_RE_STRESS_NEW = re.compile(
    r"^\s*([-+]?\d+\.\d+[Ee][+-]?\d+)"
    r"\s+([-+]?\d+\.\d+[Ee][+-]?\d+)"
    r"\s+([-+]?\d+\.\d+[Ee][+-]?\d+)\s+<-- S",
    re.MULTILINE,
)

# Old ASCII-box:  *  x   -9.999  0.000  0.000  *
_RE_STRESS_OLD = re.compile(
    r"\*\s+[xyz]\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s*\*"
)

# Total number of ions line (used for n_atoms)
_RE_N_ATOMS = re.compile(r"Total number of ions in cell\s*=\s*(\d+)", re.I)


def read_stress(castep_path: Path) -> np.ndarray | None:
    """
    Read the FINAL symmetrised stress tensor from a CASTEP output file.

    Returns a 6-element Voigt array (GPa):
        [sigma_11, sigma_22, sigma_33, sigma_23, sigma_13, sigma_12]

    Returns None if no stress block is found (empty or crashed .castep).

    Reads from the BOTTOM of the file so the final geometry is returned,
    not an intermediate iteration.
    """
    if not castep_path.exists():
        return None

    try:
        text = castep_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    if not text.strip():
        return None

    # CASTEP 25.12 format: three consecutive  <-- S  rows
    rows_new = _RE_STRESS_NEW.findall(text)
    if len(rows_new) >= 3:
        last_three = rows_new[-3:]
        try:
            matrix = np.array([[float(v) for v in row] for row in last_three])
            return np.array([
                matrix[0, 0], matrix[1, 1], matrix[2, 2],
                matrix[1, 2], matrix[0, 2], matrix[0, 1],
            ])
        except (ValueError, IndexError):
            pass

    # Old ASCII-box format
    rows_old = _RE_STRESS_OLD.findall(text)
    if len(rows_old) >= 3:
        last_three = rows_old[-3:]
        try:
            matrix = np.array([[float(v) for v in row] for row in last_three])
            return np.array([
                matrix[0, 0], matrix[1, 1], matrix[2, 2],
                matrix[1, 2], matrix[0, 2], matrix[0, 1],
            ])
        except (ValueError, IndexError):
            pass

    return None


def count_atoms_from_castep(castep_path: Path) -> int:
    """
    Read the number of atoms in the cell from a .castep file.

    Looks for:  "Total number of ions in cell =    N"

    Returns 1 as fallback (safe for per-atom normalisation).
    """
    if not castep_path.exists():
        return 1
    try:
        text = castep_path.read_text(encoding="utf-8", errors="replace")
        match = _RE_N_ATOMS.search(text)
        if match:
            return int(match.group(1))
    except OSError:
        pass
    return 1


# ─────────────────────────────────────────────────────────────────────────────
# .cijdat reader  (produced by generate_strain.py)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class CijDat:
    """Contents of a generate_strain.py .cijdat file."""

    lattice_code: int          # 5=Cubic, 4=Tetragonal, 7=Hexagonal, etc.
    n_steps: int               # deformation steps per pattern (= numsteps*2)
    max_strain: float          # maximum strain magnitude
    patterns: list[str] = field(default_factory=list)
    strain_tensors: list[np.ndarray] = field(default_factory=list)


def read_cijdat(cijdat_path: Path) -> CijDat | None:
    """
    Parse a .cijdat file produced by generate_strain.py.

    File format (generate_strain.py writes it as):
      line 1:  lattice_code  n_steps  0  0
      line 2:  max_strain
      repeating blocks of 4 lines:
        seedname_cij__p__s
        e11  e12  e13        (row 0 of strain tensor)
        e12  e22  e23        (row 1)
        e13  e23  e33        (row 2)

    The strain tensor uses IRE convention (off-diagonals are half the
    engineering shear, i.e. this_strain[3] = e23, NOT 2*e23).

    Output Voigt vector: [e11, e22, e33, 2*e23, 2*e13, 2*e12]
    so e4 = 2*e23 = 2*row1[2], consistent with sigma_4 = C44 * e4.
    """
    if not cijdat_path.exists():
        return None

    lines = cijdat_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 2:
        return None

    try:
        header = lines[0].split()
        lattice_code = int(header[0])
        n_steps = int(header[1])
        max_strain = float(lines[1].strip())
    except (ValueError, IndexError):
        return None

    dat = CijDat(lattice_code=lattice_code, n_steps=n_steps, max_strain=max_strain)

    i = 2
    while i < len(lines):
        name_line = lines[i].strip()
        if not name_line:
            i += 1
            continue
        dat.patterns.append(name_line)
        try:
            row0 = [float(v) for v in lines[i + 1].split()]
            row1 = [float(v) for v in lines[i + 2].split()]
            row2 = [float(v) for v in lines[i + 3].split()]
            # IRE → Voigt: off-diagonals doubled (engineering shear)
            strain_vec = np.array([
                row0[0],            # e11
                row1[1],            # e22
                row2[2],            # e33
                row1[2] + row2[1],  # 2*e23  (IRE: row1[2]=e23, row2[1]=e23)
                row0[2] + row2[0],  # 2*e13
                row0[1] + row1[0],  # 2*e12
            ])
            dat.strain_tensors.append(strain_vec)
            i += 4
        except (ValueError, IndexError):
            i += 1

    return dat


# ─────────────────────────────────────────────────────────────────────────────
# Derived elastic properties (cubic, Voigt-Reuss-Hill)
# ─────────────────────────────────────────────────────────────────────────────


def _cubic_derived(c11: float, c12: float, c44: float) -> dict[str, float]:
    """
    Compute polycrystalline elastic properties for a cubic crystal.

    Uses Voigt-Reuss-Hill averaging.  Stability criteria:
        C11 > 0,  C44 > 0,  C11 > |C12|,  C11 + 2*C12 > 0

    Returns empty dict if the matrix is mechanically unstable.
    """
    # Born stability conditions for cubic crystals
    if c11 <= 0 or c44 <= 0:
        return {}
    if c11 <= abs(c12):
        return {}
    if c11 + 2 * c12 <= 0:
        return {}

    # Voigt bounds (upper, iso. average assuming uniform strain)
    b_voigt = (c11 + 2.0 * c12) / 3.0
    g_voigt = (c11 - c12 + 3.0 * c44) / 5.0

    # Reuss bounds (lower, iso. average assuming uniform stress)
    # Compliance for cubic: s11=(c11+c12)/[(c11-c12)(c11+2c12)]
    #                        s12=-c12/[(c11-c12)(c11+2c12)]
    #                        s44=1/c44
    denom = (c11 + 2.0 * c12) * (c11 - c12)
    s11 = (c11 + c12) / denom
    s12 = -c12 / denom
    s44 = 1.0 / c44

    b_reuss = 1.0 / (3.0 * (s11 + 2.0 * s12))
    g_reuss = 5.0 / (4.0 * (s11 - s12) + 3.0 * s44)

    # Hill averages
    b_hill = (b_voigt + b_reuss) / 2.0
    g_hill = (g_voigt + g_reuss) / 2.0

    # Young's modulus and Poisson ratio (from Hill averages)
    e_hill = 9.0 * b_hill * g_hill / (3.0 * b_hill + g_hill)
    nu_hill = (3.0 * b_hill - 2.0 * g_hill) / (2.0 * (3.0 * b_hill + g_hill))

    # Zener anisotropy index: A = 2*C44 / (C11 - C12)
    # A=1 → isotropic, A≠1 → anisotropic
    zener = 2.0 * c44 / (c11 - c12)

    # Pugh ratio G/B: > 0.57 → brittle, < 0.57 → ductile
    pugh = g_hill / b_hill

    # Cauchy pressure (C12 - C44): positive → metallic, negative → covalent
    cauchy = c12 - c44

    return {
        "C11": c11,
        "C12": c12,
        "C44": c44,
        "B_Voigt_GPa": b_voigt,
        "B_Reuss_GPa": b_reuss,
        "B_Hill_GPa": b_hill,
        "G_Voigt_GPa": g_voigt,
        "G_Reuss_GPa": g_reuss,
        "G_Hill_GPa": g_hill,
        "E_GPa": e_hill,
        "nu": nu_hill,
        "Zener_A": zener,
        "Pugh_ratio": pugh,
        "Cauchy_pressure_GPa": cauchy,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Cij fitting — full least-squares, no masks
# ─────────────────────────────────────────────────────────────────────────────


def fit_cij(cijdat: CijDat, seed_dir: Path) -> dict[str, Any]:
    """
    Fit elastic constants from stress vs strain data (cubic symmetry).

    Physics
    ───────
    For cubic crystal, three independent constants suffice: C11, C12, C44.

    generate_strain.py uses the combined e1+e4 pattern (Voigt) for cubic:
    each deformed cell simultaneously has non-zero axial strain (e1=epsilon_11)
    and shear strain (e4=2*epsilon_23).  Both components have identical
    magnitude in each cell (ratio e4/e1 = 1.0 for cubic because |a|=|b|).

    The linear regressions are independent because they use DIFFERENT
    stress components as the dependent variable:

        C11 = ∂σ₁₁/∂ε₁₁  =  slope(stress[:,0], strain[:,0])
        C12 = ∂σ₂₂/∂ε₁₁  =  slope(stress[:,1], strain[:,0])
        C44 = ∂σ₂₃/∂(2ε₂₃) = slope(stress[:,3], strain[:,3])

    All three use ALL data points (no masking needed — e1 and e4 are
    both nonzero in every row, but the stress component axes are orthogonal).

    Regression
    ──────────
    Through-origin least squares (σ = C·ε, no intercept) via:
        C = (εᵀ · σ) / (εᵀ · ε)

    The R² value is computed to flag poor convergence or nonlinearity.

    Returns
    ───────
    dict[str, str] with all keys formatted to 4 decimal places, ready
    for CSV insertion.  Empty dict if data is insufficient or unstable.
    """
    if cijdat.lattice_code != 5:
        # Only cubic is implemented.  Tetragonal (4), hexagonal (7)
        # require multi-pattern fitting — extend here if needed.
        return {"_elastic_note": "non-cubic system: Cij fit not implemented"}

    # ── Collect stress / strain pairs ─────────────────────────────────────────
    stresses: list[np.ndarray] = []
    strains: list[np.ndarray] = []
    missing: list[str] = []

    for pattern_name, strain_vec in zip(cijdat.patterns, cijdat.strain_tensors):
        castep_file = seed_dir / f"{pattern_name}.castep"
        stress_vec = read_stress(castep_file)
        if stress_vec is None:
            missing.append(pattern_name)
            continue
        stresses.append(stress_vec)
        strains.append(strain_vec)

    if missing:
        # Return diagnostic so the user knows which runs failed
        note = f"missing stress from: {', '.join(missing[:4])}"
        if len(missing) > 4:
            note += f" (+{len(missing)-4} more)"
        if not stresses:
            return {"_elastic_error": note}

    n_points = len(stresses)
    if n_points < 3:
        return {"_elastic_error": f"only {n_points} stress tensors — need ≥ 3"}

    stress_arr = np.array(stresses)   # shape (N, 6)  [s11,s22,s33,s23,s13,s12]
    strain_arr = np.array(strains)    # shape (N, 6)  [e11,e22,e33,2e23,2e13,2e12]

    # ── Least-squares with intercept ─────────────────────────────────────────
    # sigma = C * epsilon + sigma_0
    #
    # sigma_0 is the residual stress of the reference cell (should be near
    # zero after GeomOpt, but non-zero for VCA cells where Vegard scaling
    # leaves a small residual pressure). Including the intercept:
    #   - Gives the correct slope C regardless of sigma_0
    #   - Gives a proper R² based on variance around the fit line
    #     (not around zero), so R²=1.0 for a perfect linear response
    #
    # Through-origin fit (old approach) gave R²→0 when sigma_0 ≠ 0
    # because SS_tot was computed as sum(y²) instead of sum((y-mean)²).

    def _slope_with_intercept(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
        """
        Fit y = C*x + sigma_0 by OLS.
        Returns (slope C, intercept sigma_0, R²).
        R² is standard: 1 - SS_res / SS_tot where SS_tot = sum((y - mean(y))²).
        """
        n = len(x)
        if n < 2:
            return float("nan"), float("nan"), 0.0
        # Normal equations for [C, sigma_0]
        sx = float(np.sum(x))
        sy = float(np.sum(y))
        sxx = float(np.dot(x, x))
        sxy = float(np.dot(x, y))
        denom = n * sxx - sx * sx
        if abs(denom) < 1e-30:
            return float("nan"), float("nan"), 0.0
        slope = (n * sxy - sx * sy) / denom
        intercept = (sy - slope * sx) / n
        y_pred = slope * x + intercept
        ss_res = float(np.sum((y - y_pred) ** 2))
        y_mean = sy / n
        ss_tot = float(np.sum((y - y_mean) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
        return slope, intercept, r2

    e11 = strain_arr[:, 0]   # axial strain (Voigt e1)
    e4 = strain_arr[:, 3]    # shear strain 2*e23 (Voigt e4)

    s11 = stress_arr[:, 0]   # sigma_11
    s22 = stress_arr[:, 1]   # sigma_22
    s23 = stress_arr[:, 3]   # sigma_23  (Voigt s4)

    c11, s0_11, r2_c11 = _slope_with_intercept(e11, s11)
    c12, s0_12, r2_c12 = _slope_with_intercept(e11, s22)
    c44, s0_44, r2_c44 = _slope_with_intercept(e4,  s23)

    # ── Quality check ─────────────────────────────────────────────────────────
    r2_min = min(r2_c11, r2_c12, r2_c44)
    r2_note = ""
    if r2_min < 0.99:
        r2_note = (
            f"low R² ({r2_min:.3f}) — check SCF convergence or increase nextra_bands"
        )

    if any(np.isnan(v) for v in (c11, c12, c44)):
        return {"_elastic_error": "NaN in regression — strain data degenerate"}

    # ── Derived properties ────────────────────────────────────────────────────
    props = _cubic_derived(c11, c12, c44)
    if not props:
        return {
            "_elastic_error": (
                f"Born stability violated: C11={c11:.1f} C12={c12:.1f} C44={c44:.1f} GPa"
            )
        }

    result: dict[str, Any] = {k: f"{v:.4f}" for k, v in props.items()}
    result["elastic_n_points"] = str(n_points)
    result["elastic_R2_min"] = f"{r2_min:.4f}"
    if r2_note:
        result["elastic_quality_note"] = r2_note

    return result


# ─────────────────────────────────────────────────────────────────────────────
# ΔH_mix physics note
# ─────────────────────────────────────────────────────────────────────────────


def delta_h_mix_note() -> str:
    """
    Return a one-line warning to display alongside VCA ΔH_mix values.

    VCA ΔH_mix from DFT total energies is dominated by pseudopotential
    energy differences between species (e.g. Ti vs Nb: ~70 eV absolute
    difference) rather than true chemical mixing energy.  The values are
    therefore not physically meaningful for thermodynamic stability analysis.

    What IS valid:
      - Lattice parameter vs x (Vegard law verification)
      - Elastic constants vs x (linear trend in Cij)
      - Relative enthalpy vs x within the SAME element pair
        (deviation from Vegard line, qualitative trend only)

    For quantitative ΔH_mix use SQS + total energy differences,
    or experimental calorimetry.
    """
    return (
        "⚠  VCA ΔH_mix: values dominated by pseudopotential energy offsets "
        "(ΔE_psp ≫ ΔE_mixing). Use as qualitative trend only, not for "
        "thermodynamic stability. For accurate ΔH_mix: use SQS supercells."
    )
