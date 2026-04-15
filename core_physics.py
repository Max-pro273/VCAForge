"""
core_physics.py  —  Pure crystallographic and elastic mathematics.
══════════════════════════════════════════════════════════════════
No file I/O, no subprocess calls, no user interaction.
All functions are deterministic given the same numeric inputs.

Primitive-cell reduction is delegated to spglib (industry standard).
The former hand-rolled Křivý-Gruber and find_primitive_cell
implementations have been removed (~150 lines of fragile custom code).

Public API
──────────
  VEC
    VEC_VALENCES              dict[str, int]
    vec_for_system(species, nonmetal)  -> float
    vec_stability(vec)                 -> "green" | "yellow" | "red"
    nextra_bands_for(x, vec)           -> int

  Lattice helpers
    cell_volume(L)            -> float
    lattice_to_abc(L)         -> (a, b, c, alpha, beta, gamma)
    abc_to_lattice(…)         -> np.ndarray
    standardize_cubic(L)      -> np.ndarray

  Strain
    StrainStep                dataclass
    generate_strain_steps(L, code, strain, n) -> list[StrainStep]
    apply_strain(L, voigt)    -> np.ndarray
    pointgroup_to_lattice_code(pg)            -> int

  Elastic fitting
    fit_cij_cubic(stresses, strains, …) -> dict | {"error": str}
    cubic_vrh(c11, c12, c44, …)         -> dict[str, float]

  Vegard interpolation
    vegard_interpolate(x, d0, d1) -> dict
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import config
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Valence electron count table
# ─────────────────────────────────────────────────────────────────────────────

VEC_VALENCES: dict[str, int] = {
    "Sc": 3,
    "Ti": 4,
    "V": 5,
    "Cr": 6,
    "Mn": 7,
    "Fe": 8,
    "Co": 9,
    "Ni": 10,
    "Cu": 11,
    "Zn": 12,
    "Y": 3,
    "Zr": 4,
    "Nb": 5,
    "Mo": 6,
    "Tc": 7,
    "Ru": 8,
    "Rh": 9,
    "Pd": 10,
    "Ag": 11,
    "Cd": 12,
    "Hf": 4,
    "Ta": 5,
    "W": 6,
    "Re": 7,
    "B": 3,
    "C": 4,
    "N": 5,
    "O": 6,
    "F": 7,
    "Al": 3,
    "Si": 4,
    "P": 5,
    "S": 6,
}


def vec_for_system(
    species: list[tuple[str, float]],
    nonmetal: str | None = None,
) -> float:
    """Compute the valence electron count for a multi-component system.

    Args:
        species:  List of ``(element, mole_fraction)`` pairs.
                  Fractions must sum to 1.
        nonmetal: Element on the anion sublattice (e.g. ``"C"``, ``"N"``).

    Returns:
        VEC as a float.

    Example::

        vec_for_system([("Ti", 0.6), ("Nb", 0.4)], "C")
        # (0.6*4 + 0.4*5) + 4 = 6.8
    """
    metal_vec = sum(
        frac * VEC_VALENCES.get(elem.capitalize(), 0) for elem, frac in species
    )
    nm_vec = VEC_VALENCES.get(nonmetal.capitalize(), 0) if nonmetal else 0.0
    return metal_vec + nm_vec


def nextra_bands_for(x: float, vec: float) -> int:
    """Return adaptive ``nextra_bands`` for an elastic SinglePoint run.

    Pure endpoints (x ≈ 0 or x ≈ 1) use :data:`config.ELASTIC_NEXTRA_PURE`.
    VCA intermediates use
    :data:`config.ELASTIC_NEXTRA_BASE` + ⌊|VEC − 8| × 20⌋.

    Args:
        x:   VCA concentration parameter in [0, 1].
        vec: Valence electron count.
    """
    if x < 1e-5 or x > 1.0 - 1e-5:
        return config.ELASTIC_NEXTRA_PURE
    return config.ELASTIC_NEXTRA_BASE + int(abs(vec - 8.0) * 20)


# ─────────────────────────────────────────────────────────────────────────────
# Lattice helpers
# ─────────────────────────────────────────────────────────────────────────────


def cell_volume(L: np.ndarray) -> float:
    """Return the scalar cell volume from a 3x3 lattice matrix (rows = vectors)."""
    return abs(float(np.dot(L[0], np.cross(L[1], L[2]))))


def lattice_to_abc(
    L: np.ndarray,
) -> tuple[float, float, float, float, float, float]:
    """Convert a 3x3 lattice matrix to conventional cell parameters.

    Args:
        L: 3x3 array whose rows are lattice vectors (Angstrom).

    Returns:
        ``(a, b, c, alpha_deg, beta_deg, gamma_deg)``
    """
    a = float(np.linalg.norm(L[0]))
    b = float(np.linalg.norm(L[1]))
    c = float(np.linalg.norm(L[2]))
    ca = float(np.clip(np.dot(L[1], L[2]) / (b * c), -1.0, 1.0))
    cb = float(np.clip(np.dot(L[0], L[2]) / (a * c), -1.0, 1.0))
    cg = float(np.clip(np.dot(L[0], L[1]) / (a * b), -1.0, 1.0))
    return (
        a,
        b,
        c,
        float(np.degrees(np.arccos(ca))),
        float(np.degrees(np.arccos(cb))),
        float(np.degrees(np.arccos(cg))),
    )


def abc_to_lattice(
    a: float,
    b: float,
    c: float,
    alpha: float,
    beta: float,
    gamma: float,
) -> np.ndarray:
    """Build an upper-triangular 3x3 lattice from conventional parameters.

    Args:
        a, b, c:            Lattice lengths in Angstrom.
        alpha, beta, gamma: Angles in degrees.

    Returns:
        3x3 numpy array (rows = lattice vectors).
    """
    ar = np.radians(alpha)
    br = np.radians(beta)
    gr = np.radians(gamma)
    sg = max(float(np.sin(gr)), 1e-15)
    cx = c * float(np.cos(br))
    cy = c * (float(np.cos(ar)) - float(np.cos(br)) * float(np.cos(gr))) / sg
    cz = float(np.sqrt(max(c**2 - cx**2 - cy**2, 0.0)))
    return np.array(
        [
            [a, 0.0, 0.0],
            [b * float(np.cos(gr)), b * sg, 0.0],
            [cx, cy, cz],
        ]
    )


def standardize_cubic(L: np.ndarray) -> np.ndarray:
    """Convert a primitive FCC/BCC cell to an orthogonal conventional cell.

    Required before computing Voigt strain magnitudes so that the
    C11 = dsigma11/depsilon11 relation uses aligned axes.
    Returns *L* unchanged for non-cubic or already-orthogonal cells.

    Args:
        L: 3x3 lattice matrix (rows = vectors, Angstrom).
    """
    a, b, c, al, be, ga = lattice_to_abc(L)
    avg_len = (a + b + c) / 3.0
    if max(abs(a - avg_len), abs(b - avg_len), abs(c - avg_len)) / avg_len > 0.005:
        return L  # Not equal-length -> not cubic.
    if max(abs(al - 90), abs(be - 90), abs(ga - 90)) < 1.5:
        return L  # Already orthogonal.
    avg_ang = (al + be + ga) / 3.0
    vol = cell_volume(L)
    if abs(avg_ang - 60.0) < 1.5:  # FCC primitive
        a_conv = (4.0 * vol) ** (1.0 / 3.0)
    elif abs(avg_ang - 109.47) < 1.5:  # BCC primitive
        a_conv = (2.0 * vol) ** (1.0 / 3.0)
    else:
        return L
    return np.diag([a_conv, a_conv, a_conv])


# ─────────────────────────────────────────────────────────────────────────────
# Strain patterns  (indexed by lattice symmetry code)
# ─────────────────────────────────────────────────────────────────────────────

_STRAIN_PATTERNS: dict[int, list[list[float]]] = {
    1: [  # triclinic — all 6 independent
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
    ],
    2: [  # monoclinic
        [1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 1],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
    ],
    3: [  # orthorhombic
        [1, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1],
    ],
    4: [[1, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 1]],  # tetragonal
    5: [[1, 0, 0, 1, 0, 0]],  # cubic
    6: [[1, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0]],  # trigonal
    7: [[0, 0, 1, 0, 0, 0], [1, 0, 0, 1, 0, 0]],  # hexagonal
}


@dataclass
class StrainStep:
    """One strained cell in the finite-strain elastic workflow.

    Attributes:
        pattern_idx:  1-based index of the strain pattern used.
        step_idx:     1-based counter within the pattern (alternating +/-).
        magnitude:    Signed strain magnitude applied.
        strain_voigt: 6-component Voigt strain vector.
    """

    pattern_idx: int
    step_idx: int
    magnitude: float
    strain_voigt: np.ndarray = field(repr=False)

    @property
    def name(self) -> str:
        """Unique filename suffix, e.g. ``_cij__1__3``."""
        return f"_cij__{self.pattern_idx}__{self.step_idx}"


def generate_strain_steps(
    L: np.ndarray,
    lattice_code: int = 5,
    max_strain: float = config.ELASTIC_MAX_STRAIN,
    n_steps: int = config.ELASTIC_N_STEPS,
) -> list[StrainStep]:
    """Generate all StrainStep objects for a given lattice symmetry.

    Args:
        L:            3x3 lattice matrix used to set physical strain magnitudes.
        lattice_code: Symmetry code 1-7 (default 5 = cubic).
        max_strain:   Maximum strain magnitude (default +/-0.3 %).
        n_steps:      Positive magnitudes per pattern.

    Returns:
        List of StrainStep objects, alternating +/- per pattern.
    """
    patterns = _STRAIN_PATTERNS[lattice_code]
    a = float(np.linalg.norm(L[0]))
    b = float(np.linalg.norm(L[1]))
    lens = [a, b, b, b, a, a]  # Per IRE Voigt index order.

    steps: list[StrainStep] = []
    for pi, pattern in enumerate(patterns, 1):
        sc = 0
        for k in range(n_steps):
            for neg in (False, True):
                sc += 1
                frac = (k + 1) / n_steps
                mag = frac * max_strain * (-1 if neg else 1)
                v = np.zeros(6)
                for i, p in enumerate(pattern):
                    if p:
                        v[i] = p * mag / lens[i] if i < 3 else 0.5 * p * mag / lens[i]
                steps.append(StrainStep(pi, sc, mag, v))
    return steps


def apply_strain(L: np.ndarray, voigt: np.ndarray) -> np.ndarray:
    """Return L_new = L @ F.T where F = I + epsilon (Voigt -> 3x3).

    Args:
        L:     3x3 reference lattice matrix.
        voigt: 6-component Voigt strain ``[e11, e22, e33, e23, e13, e12]``.
    """
    e11, e22, e33 = voigt[0], voigt[1], voigt[2]
    e23, e13, e12 = voigt[3] / 2, voigt[4] / 2, voigt[5] / 2
    F = np.array(
        [
            [1 + e11, e12, e13],
            [e12, 1 + e22, e23],
            [e13, e23, 1 + e33],
        ]
    )
    return L @ F.T


def pointgroup_to_lattice_code(pg: int) -> int:
    """Map a CASTEP point-group index to a strain-pattern key (1-7).

    Args:
        pg: Integer point-group index from the CASTEP output file.

    Returns:
        Lattice symmetry code 1-7 (defaults to 5 = cubic for pg >= 28).
    """
    if pg <= 2:
        return 1  # triclinic
    if pg <= 5:
        return 2  # monoclinic
    if pg <= 8:
        return 3  # orthorhombic
    if pg <= 15:
        return 4  # tetragonal
    if pg <= 17:
        return 6  # trigonal
    if pg <= 27:
        return 7  # hexagonal
    return 5  # cubic


# ─────────────────────────────────────────────────────────────────────────────
# Elastic fitting  (OLS, cubic symmetry)
# ─────────────────────────────────────────────────────────────────────────────


def _ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Ordinary-least-squares fit y = slope*x + intercept.

    Args:
        x: Predictor array (1-D).
        y: Response array (1-D).

    Returns:
        ``(slope, R_squared)``
    """
    n = len(x)
    sx = float(x.sum())
    sy = float(y.sum())
    sxx = float(np.dot(x, x))
    sxy = float(np.dot(x, y))
    d = n * sxx - sx * sx
    if abs(d) < 1e-30:
        return float("nan"), 0.0
    slope = (n * sxy - sx * sy) / d
    ic = (sy - slope * sx) / n
    res = float(np.sum((y - slope * x - ic) ** 2))
    tot = float(np.sum((y - sy / n) ** 2))
    r2 = 1.0 - res / tot if tot > 1e-12 else 1.0
    return slope, r2


def fit_cij_cubic(
    stresses: list[np.ndarray],
    strains: list[np.ndarray],
    density_gcm3: float | None = None,
    n_atoms: int | None = None,
    volume_ang3: float | None = None,
) -> dict[str, Any]:
    """Fit C11, C12, C44 from stress-strain data (cubic symmetry).

    Pattern ``[1, 0, 0, 1, 0, 0]`` gives::

        C11 = d(sigma11)/d(epsilon11)
        C12 = d(sigma22)/d(epsilon11)
        C44 = d(sigma23)/d(2*epsilon23)

    Args:
        stresses:     Voigt stress vectors ``[s11, s22, s33, s23, s13, s12]``
                      in GPa.
        strains:      Corresponding Voigt strain vectors.
        density_gcm3: Density in g/cm^3 (enables sound speeds, Debye temp).
        n_atoms:      Atoms per unit cell (needed for Debye temperature).
        volume_ang3:  Cell volume in Angstrom^3.

    Returns:
        Flat dict of VRH moduli (values as ``str`` for CSV compatibility), or
        ``{"error": description}`` on failure.  Never raises.
    """
    if len(stresses) < 3:
        return {"error": f"only {len(stresses)} stress tensors — need >= 3"}

    sa = np.array(stresses)  # (N, 6)
    ea = np.array(strains)  # (N, 6)

    c11, r2_11 = _ols(ea[:, 0], sa[:, 0])
    c12, r2_12 = _ols(ea[:, 0], sa[:, 1])
    c44, r2_44 = _ols(ea[:, 3], sa[:, 3])

    if any(np.isnan(v) for v in (c11, c12, c44)):
        return {"error": "NaN in OLS — strain data degenerate"}

    r2_min = min(r2_11, r2_12, r2_44)
    props = cubic_vrh(
        c11,
        c12,
        c44,
        density_gcm3=density_gcm3,
        n_atoms=n_atoms,
        volume_ang3=volume_ang3,
    )
    if not props:
        return {
            "error": (
                f"Born stability violated: "
                f"C11={c11:.1f}  C12={c12:.1f}  C44={c44:.1f} GPa"
            )
        }

    result: dict[str, Any] = {k: f"{v:.4f}" for k, v in props.items()}
    result["elastic_n_points"] = str(len(stresses))
    result["elastic_R2_min"] = f"{r2_min:.4f}"
    if r2_min < 0.99:
        result["elastic_quality_note"] = (
            f"low R2 ({r2_min:.3f}) — check SCF convergence or increase nextra_bands"
        )
    return result


def cubic_vrh(
    c11: float,
    c12: float,
    c44: float,
    density_gcm3: float | None = None,
    n_atoms: int | None = None,
    volume_ang3: float | None = None,
) -> dict[str, float]:
    """Compute polycrystalline elastic properties from cubic Cij.

    Applies the Born mechanical-stability check:
    C11 > 0, C44 > 0, C11 > |C12|, C11 + 2*C12 > 0.

    Args:
        c11, c12, c44: Cubic elastic constants in GPa.
        density_gcm3:  Density (enables sound speeds / Debye temperature).
        n_atoms:       Atoms per cell (enables Debye temperature).
        volume_ang3:   Cell volume in Angstrom^3.

    Returns:
        Dict of physical properties, or ``{}`` if Born stability is violated.
    """
    if c11 <= 0 or c44 <= 0 or c11 <= abs(c12) or c11 + 2 * c12 <= 0:
        return {}

    bv = (c11 + 2 * c12) / 3
    gv = (c11 - c12 + 3 * c44) / 5
    den = (c11 + c12) * (c11 - c12)
    s11 = (c11 + c12) / den
    s12 = -c12 / den
    s44 = 1.0 / c44
    br = 1.0 / (3 * (s11 + 2 * s12))
    gr = 5.0 / (4 * (s11 - s12) + 3 * s44)
    bh = (bv + br) / 2
    gh = (gv + gr) / 2
    eh = 9 * bh * gh / (3 * bh + gh)
    nu = (3 * bh - 2 * gh) / (2 * (3 * bh + gh))
    za = 2 * c44 / (c11 - c12)
    pugh = gh / bh
    cauchy = c12 - c44
    cp = (c11 - c12) / 2
    klein = (c11 + 8 * c12) / (7 * c11 + 2 * c12)
    lam = bh - 2 * gh / 3
    ga_a = 1.5 * (1 + nu) / (2 - 3 * nu)

    p: dict[str, float] = {
        "C11": c11,
        "C12": c12,
        "C44": c44,
        "B_Voigt_GPa": bv,
        "B_Reuss_GPa": br,
        "B_Hill_GPa": bh,
        "G_Voigt_GPa": gv,
        "G_Reuss_GPa": gr,
        "G_Hill_GPa": gh,
        "E_GPa": eh,
        "nu": nu,
        "Zener_A": za,
        "Pugh_ratio": pugh,
        "Cauchy_pressure_GPa": cauchy,
        "C_prime_GPa": cp,
        "Kleinman_zeta": klein,
        "lambda_Lame_GPa": lam,
        "mu_Lame_GPa": gh,
        "acoustic_Gruneisen": ga_a,
        "H_Vickers_GPa": max(0.0, 2 * (pugh**2 * gh) ** 0.585 - 3),
    }

    if density_gcm3 and density_gcm3 > 0:
        rho = density_gcm3 * 1e3
        vl = ((bh + 4 * gh / 3) * 1e9 / rho) ** 0.5
        vs = (gh * 1e9 / rho) ** 0.5
        vm = (1 / 3 * (2 / vs**3 + 1 / vl**3)) ** (-1 / 3)
        p["v_longitudinal_ms"] = vl
        p["v_transverse_ms"] = vs
        p["v_mean_ms"] = vm

        if n_atoms and volume_ang3 and volume_ang3 > 0:
            hbar = 1.054571817e-34
            kb = 1.380649e-23
            n_vol = n_atoms / (volume_ang3 * 1e-30)
            p["T_Debye_K"] = (hbar / kb) * vm * (6 * np.pi**2 * n_vol) ** (1 / 3)

    return p


# ─────────────────────────────────────────────────────────────────────────────
# Vegard interpolation
# ─────────────────────────────────────────────────────────────────────────────

_VEGARD_KEYS: tuple[str, ...] = (
    "C11",
    "C12",
    "C44",
    "B_Hill_GPa",
    "G_Hill_GPa",
    "E_GPa",
    "nu",
    "Zener_A",
    "Pugh_ratio",
    "Cauchy_pressure_GPa",
)


def vegard_interpolate(
    x: float,
    d0: dict[str, Any],
    d1: dict[str, Any],
) -> dict[str, Any]:
    """Linear interpolation of elastic constants (Vegard's law).

    Typical error < 5 % for transition-metal carbide/nitride series.

    Args:
        x:  VCA concentration parameter in [0, 1].
        d0: Parsed results dict for the x = 0 endpoint.
        d1: Parsed results dict for the x = 1 endpoint.

    Returns:
        Dict of interpolated constants, or ``{}`` if endpoints are missing.
    """
    if not d0 or not d1:
        return {}
    r: dict[str, Any] = {}
    for k in _VEGARD_KEYS:
        v0, v1 = d0.get(k, ""), d1.get(k, "")
        if not v0 or not v1:
            continue
        try:
            r[k] = f"{(1 - x) * float(v0) + x * float(v1):.4f}"
        except ValueError:
            pass
    if r:
        r["elastic_source"] = "Vegard_interpolation"
        r["elastic_n_points"] = "0"
        r["elastic_R2_min"] = "N/A"
    return r
