"""
core_physics.py  —  Core physical representations and crystallographic math.
══════════════════════════════════════════════════════════════════════════
Contains the universal Crystal object with self-aware symmetry,
strain generators, and cubic elastic tensor fitting logic.
No silent exceptions.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import spglib

import config

# ─────────────────────────────────────────────────────────────────────────────
# Mathematical Utilities
# ─────────────────────────────────────────────────────────────────────────────

def lattice_to_abc(L: np.ndarray) -> tuple[float, float, float, float, float, float]:
    """Convert 3x3 matrix to (a, b, c, alpha, beta, gamma) in degrees."""
    a, b, c = np.linalg.norm(L, axis=1)
    ca = np.clip(np.dot(L[1], L[2]) / (b * c), -1.0, 1.0)
    cb = np.clip(np.dot(L[0], L[2]) / (a * c), -1.0, 1.0)
    cg = np.clip(np.dot(L[0], L[1]) / (a * b), -1.0, 1.0)
    return a, b, c, float(np.degrees(np.arccos(ca))), float(np.degrees(np.arccos(cb))), float(np.degrees(np.arccos(cg)))

def abc_to_lattice(a: float, b: float, c: float, alpha: float, beta: float, gamma: float) -> np.ndarray:
    """Convert (a, b, c, alpha, beta, gamma) to upper-triangular 3x3 matrix."""
    ar, br, gr = np.radians(alpha), np.radians(beta), np.radians(gamma)
    sg = max(float(np.sin(gr)), 1e-15)
    cx = c * float(np.cos(br))
    cy = c * (float(np.cos(ar)) - float(np.cos(br)) * float(np.cos(gr))) / sg
    cz = float(np.sqrt(max(c**2 - cx**2 - cy**2, 0.0)))
    return np.array([[a, 0.0, 0.0], [b * float(np.cos(gr)), b * sg, 0.0], [cx, cy, cz]])

def nextra_bands_for(x: float, vec: float) -> int:
    if x < 1e-5 or x > 1.0 - 1e-5:
        return getattr(config, "ELASTIC_NEXTRA_PURE", 10)
    return getattr(config, "ELASTIC_NEXTRA_BASE", 15) + int(abs(vec - 8.0) * 20)

def vec_for_system(species: list[tuple[str, float]], nonmetal: str | None = None) -> float:
    """Compute VEC (Valence Electron Concentration)."""
    metal_vec = sum(frac * config.ELEMENTS.get(elem.capitalize(), {}).get("val", 0) for elem, frac in species)
    nm_vec = config.ELEMENTS.get(nonmetal.capitalize(), {}).get("val", 0) if nonmetal else 0
    return metal_vec + nm_vec

# ─────────────────────────────────────────────────────────────────────────────
# Core Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Crystal:
    """Universal representation of a crystal lattice. Self-aware of its symmetry."""
    lattice: np.ndarray
    frac_coords: np.ndarray
    species: list[str]

    _sym_dataset: Any = field(default=None, repr=False, init=False)

    @property
    def num_atoms(self) -> int:
        return len(self.species)

    @property
    def volume(self) -> float:
        return abs(float(np.dot(self.lattice[0], np.cross(self.lattice[1], self.lattice[2]))))

    def _get_spglib_dataset(self, symprec: float = 1e-5) -> Any: # <-- Changed to 1e-5
        if self._sym_dataset is None:
            nums = [config.ELEMENTS.get(s.capitalize(), {}).get("Z", 1) for s in self.species]
            cell = (self.lattice, self.frac_coords, nums)
            dataset = spglib.get_symmetry_dataset(cell, symprec=symprec)
            if dataset is None:
                raise ValueError(f"spglib failed to find symmetry (symprec={symprec}). Atoms may be overlapping or cell is heavily distorted.")
            self._sym_dataset = dataset
        return self._sym_dataset

    @property
    def spacegroup_number(self) -> int:
        return self._get_spglib_dataset().number

    @property
    def spacegroup_symbol(self) -> str:
        return self._get_spglib_dataset().international

    @property
    def lattice_type(self) -> str:
        sg = self.spacegroup_number
        if sg >= 195: return "cubic"
        if sg >= 168: return "hexagonal"
        if sg >= 143: return "trigonal"
        if sg >= 75:  return "tetragonal"
        if sg >= 16:  return "orthorhombic"
        if sg >= 3:   return "monoclinic"
        return "triclinic"

    @property
    def strain_pattern_code(self) -> int:
        sg = self.spacegroup_number
        if sg >= 195: return 5  # cubic
        if sg >= 168: return 7  # hexagonal
        if sg >= 143: return 6  # trigonal
        if sg >= 75:  return 4  # tetragonal
        if sg >= 16:  return 3  # orthorhombic
        if sg >= 3:   return 2  # monoclinic
        return 1                # triclinic

    def clear_cache(self) -> None:
        """Clears cached spglib dataset. Must be called if lattice is mutated."""
        self._sym_dataset = None

    def get_symmetry_operations(self, cartesian_rotations: bool = False, symprec: float = 1e-5) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns symmetry operations.
        If cartesian_rotations=True, transforms spglib's native fractional
        rotation matrices into Cartesian coordinates via R_c = L^T * R_f * (L^T)^-1.
        Translations are always returned in fractional coordinates.
        """
        dataset = self._get_spglib_dataset(symprec)
        rotations = np.array(dataset.rotations)
        translations = np.array(dataset.translations)

        if cartesian_rotations:
            # spglib uses column vectors for position math (x' = R*x + t)
            # Our lattice rows are basis vectors, so the basis matrix is L.T
            L_T = self.lattice.T
            L_T_inv = np.linalg.inv(L_T)

            cart_rots = []
            for R_frac in rotations:
                R_c = L_T @ R_frac @ L_T_inv
                # Snap FP noise to exact integers for standard point group operations
                cart_rots.append(np.round(R_c).astype(int))
            rotations = np.array(cart_rots)

        # Snap translation float noise to strict 0.0
        translations[np.abs(translations) < 1e-12] = 0.0

        return rotations, translations
# ─────────────────────────────────────────────────────────────────────────────
# Initialization & Standardization
# ─────────────────────────────────────────────────────────────────────────────

def load_crystal(file_path: Path) -> Crystal:
    """Load any geometry file and return a canonical primitive Crystal."""
    raw = _read_raw(file_path)
    return standardize_crystal(raw)

def _read_raw(file_path: Path) -> Crystal:
    """Parse geometry file into a Crystal WITHOUT standardisation."""
    ext = file_path.suffix.lower()
    text = file_path.read_text(encoding="utf-8", errors="replace")

    if ext == ".cif":
        return _read_cif_raw(text)
    elif ext == ".cell":
        return _read_castep_cell(text)
    elif ext in (".poscar", "") or file_path.name.startswith("POSCAR") or file_path.name.startswith("CONTCAR"):
        return _read_vasp_poscar(text)
    raise ValueError(f"Unsupported geometry format: {file_path.name}")

def standardize_crystal(crystal: Crystal, symprec: float = 1e-5) -> Crystal:
    """Always standardizes to the primitive cell for maximum speed."""
    nums = [config.ELEMENTS.get(s.capitalize(), {}).get("Z", 1) for s in crystal.species]
    cell = (crystal.lattice, crystal.frac_coords, nums)
    std = spglib.standardize_cell(cell, to_primitive=True, symprec=symprec)
    if std is None:
        raise ValueError("spglib failed to standardize crystal.")

    L, f_coords, n = std
    # Reverse lookup atomic number to Symbol from config
    z_to_sym = {v["Z"]: k for k, v in config.ELEMENTS.items()}
    species = [z_to_sym.get(z, "X") for z in n]

    return Crystal(lattice=L, frac_coords=f_coords, species=species)

def _read_cif_raw(text: str) -> Crystal:
    params = {}
    for k in ("_cell_length_a", "_cell_length_b", "_cell_length_c",
              "_cell_angle_alpha", "_cell_angle_beta", "_cell_angle_gamma"):
        m = re.search(rf"^{re.escape(k)}\s+([\d.]+(?:\(\d+\))?)", text, re.M | re.I)
        if m: params[k] = float(re.sub(r"\(\d+\)$", "", m.group(1)))

    L = abc_to_lattice(
        params.get("_cell_length_a", 1.0), params.get("_cell_length_b", 1.0), params.get("_cell_length_c", 1.0),
        params.get("_cell_angle_alpha", 90.0), params.get("_cell_angle_beta", 90.0), params.get("_cell_angle_gamma", 90.0)
    )

    sp, fc_list = [], []
    loop_pat = re.compile(r"loop_\s+((?:_atom_site_\S+\s+)+)((?:(?!loop_|_\S+\s+).*\n?)+)", re.M)
    for lm in loop_pat.finditer(text):
        cols = re.findall(r"(_atom_site_\S+)", lm.group(1))
        if "_atom_site_fract_x" not in cols: continue
        t_idx = next((i for i, c in enumerate(cols) if c in ("_atom_site_type_symbol", "_atom_site_label")), 0)
        xi, yi, zi = cols.index("_atom_site_fract_x"), cols.index("_atom_site_fract_y"), cols.index("_atom_site_fract_z")

        for line in lm.group(2).splitlines():
            parts = line.split()
            if len(parts) < len(cols): continue
            try:
                x, y, z = (float(re.sub(r"\(\d+\)$", "", parts[i])) for i in (xi, yi, zi))
                sym_match = re.match(r"([A-Za-z]+)", parts[t_idx])
                if sym_match:
                    sp.append(sym_match.group(1).capitalize())
                    fc_list.append([x, y, z])
            except ValueError:
                pass
    return Crystal(lattice=L, frac_coords=np.array(fc_list), species=sp)

def _read_castep_cell(text: str) -> Crystal:
    m_lat = re.search(r"%BLOCK\s+LATTICE_CART\s*\n(.*?)%ENDBLOCK\s+LATTICE_CART", text, re.DOTALL | re.I)
    vecs = []
    for line in m_lat.group(1).splitlines():
        s = line.strip().lower()
        if not s or s in {"ang", "bohr", "a.u.", "angstrom"}: continue
        parts = s.split()
        if len(parts) >= 3:
            try: vecs.append(np.array([float(p) for p in parts[:3]]))
            except ValueError: pass

    m_pos = re.search(r"%BLOCK\s+POSITIONS_FRAC\s*\n(.*?)%ENDBLOCK\s+POSITIONS_FRAC", text, re.DOTALL | re.I)
    sp, fc = [], []
    for line in m_pos.group(1).splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[0].isalpha():
            sp.append(parts[0].capitalize())
            fc.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return Crystal(lattice=np.array(vecs), frac_coords=np.array(fc), species=sp)

def _read_vasp_poscar(text: str) -> Crystal:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    scale = float(lines[1].split()[0])
    vecs = [[float(x) * scale for x in line.split()[:3]] for line in lines[2:5]]

    species_names = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    sp = []
    for name, count in zip(species_names, counts):
        sp.extend([name.capitalize()] * count)

    start_idx = 8 if lines[7].lower().startswith("d") or lines[7].lower().startswith("c") else 7
    if lines[7].lower().startswith("s"): start_idx += 1

    fc = []
    for line in lines[start_idx:start_idx + sum(counts)]:
        fc.append([float(x) for x in line.split()[:3]])

    if lines[start_idx-1].lower().startswith("c") or lines[start_idx-1].lower().startswith("k"):
        fc = np.array(fc) @ np.linalg.inv(np.array(vecs))

    return Crystal(lattice=np.array(vecs), frac_coords=np.array(fc), species=sp)

# ─────────────────────────────────────────────────────────────────────────────
# Strain & Elasticity (Finite-strain logic)
# ─────────────────────────────────────────────────────────────────────────────

_STRAIN_PATTERNS: dict[int, list[list[float]]] = {
    1: [[1,0,0,0,0,0], [0,1,0,0,0,0], [0,0,1,0,0,0], [0,0,0,1,0,0], [0,0,0,0,1,0], [0,0,0,0,0,1]],
    2: [[1,0,0,1,0,0], [0,0,1,0,0,1], [0,1,0,0,0,0], [0,0,0,0,1,0]],
    3: [[1,0,0,1,0,0], [0,1,0,0,1,0], [0,0,1,0,0,1]],
    4: [[1,0,0,1,0,0], [0,0,1,0,0,1]],
    5: [[1,0,0,1,0,0]], # Cubic - minimal patterns
    6: [[1,0,0,0,0,0], [0,0,1,1,0,0]],
    7: [[0,0,1,0,0,0], [1,0,0,1,0,0]],
}

@dataclass
class StrainStep:
    pattern_idx: int
    step_idx: int
    magnitude: float
    strain_voigt: np.ndarray
    @property
    def name(self) -> str: return f"_cij__{self.pattern_idx}__{self.step_idx}"

def generate_strain_steps(crystal: Crystal, max_strain: float = 0.003, n_steps: int = 3) -> list[StrainStep]:
    """Generates optimal strain steps based on the crystal's exact symmetry."""
    pattern_code = crystal.strain_pattern_code
    patterns = _STRAIN_PATTERNS[pattern_code]

    L = crystal.lattice
    if crystal.lattice_type == "cubic":
        vol = crystal.volume
        a_conv = (4.0 * vol) ** (1.0 / 3.0) if crystal.spacegroup_symbol.startswith("F") else (2.0 * vol) ** (1.0 / 3.0)
        L = np.diag([a_conv, a_conv, a_conv])

    a, b, c = np.linalg.norm(L, axis=1)
    lens = [a, b, c, c, b, a]

    steps: list[StrainStep] = []
    for pi, pattern in enumerate(patterns, 1):
        sc = 0
        for k in range(n_steps):
            for neg in (False, True):
                sc += 1
                mag = ((k + 1) / n_steps) * max_strain * (-1 if neg else 1)
                v = np.zeros(6)
                for i, p in enumerate(pattern):
                    if p: v[i] = p * mag / lens[i] if i < 3 else 0.5 * p * mag / lens[i]
                steps.append(StrainStep(pi, sc, mag, v))
    return steps


def fit_cij_cubic(stresses: list[np.ndarray], strains: list[np.ndarray], density_gcm3: float | None = None, n_atoms: int | None = None, volume_ang3: float | None = None) -> dict[str, Any]:
    if len(stresses) < 3: return {"error": f"Need >= 3 stress tensors, got {len(stresses)}"}
    sa, ea = np.array(stresses), np.array(strains)

    def _ols(x, y):
        n = len(x); d = n * np.dot(x, x) - x.sum()**2
        if abs(d) < 1e-30: return float("nan"), 0.0
        slope = (n * np.dot(x, y) - x.sum() * y.sum()) / d
        ic = (y.sum() - slope * x.sum()) / n
        tot = float(np.sum((y - y.sum() / n) ** 2))
        r2 = 1.0 - float(np.sum((y - slope * x - ic) ** 2)) / tot if tot > 1e-12 else 1.0
        return slope, r2

    c11, r2_11 = _ols(ea[:, 0], sa[:, 0])
    c12, r2_12 = _ols(ea[:, 0], sa[:, 1])
    c44, r2_44 = _ols(ea[:, 3], sa[:, 3])

    if any(np.isnan(v) for v in (c11, c12, c44)):
        return {"error": "NaN in OLS fit"}

    props = cubic_vrh(c11, c12, c44, density_gcm3, n_atoms, volume_ang3)
    if not props:
        return {"error": f"Born stability violated: C11={c11:.1f} C12={c12:.1f} C44={c44:.1f}"}

    result: dict[str, Any] = {k: f"{v:.4f}" for k, v in props.items()}
    result.update({"elastic_n_points": str(len(stresses)), "elastic_R2_min": f"{min(r2_11, r2_12, r2_44):.4f}"})
    if min(r2_11, r2_12, r2_44) < 0.99: result["elastic_quality_note"] = "Low R2 — check SCF convergence"
    return result

def cubic_vrh(c11: float, c12: float, c44: float, density: float | None = None, n_atoms: int | None = None, vol: float | None = None) -> dict[str, float]:
    if c11 <= 0 or c44 <= 0 or c11 <= abs(c12) or c11 + 2 * c12 <= 0: return {}
    bv, gv = (c11 + 2*c12)/3, (c11 - c12 + 3*c44)/5
    den = (c11 + c12) * (c11 - c12)
    s11, s12, s44 = (c11 + c12)/den, -c12/den, 1.0/c44
    br, gr = 1.0/(3*(s11 + 2*s12)), 5.0/(4*(s11 - s12) + 3*s44)
    bh, gh = (bv + br)/2, (gv + gr)/2
    pugh = gh / bh
    props = {
        "C11": c11, "C12": c12, "C44": c44,
        "B_Voigt_GPa": bv, "B_Reuss_GPa": br, "B_Hill_GPa": bh,
        "G_Voigt_GPa": gv, "G_Reuss_GPa": gr, "G_Hill_GPa": gh,
        "E_GPa": 9*bh*gh / (3*bh + gh), "nu": (3*bh - 2*gh) / (2*(3*bh + gh)),
        "Zener_A": 2*c44 / (c11 - c12), "Pugh_ratio": pugh,
        "Cauchy_pressure_GPa": c12 - c44, "C_prime_GPa": (c11 - c12)/2,
        "H_Vickers_GPa": max(0.0, 2 * (pugh**2 * gh) ** 0.585 - 3),
    }
    if density and density > 0:
        rho = density * 1e3
        vl, vs = ((bh + 4*gh/3)*1e9 / rho)**0.5, (gh*1e9 / rho)**0.5
        vm = (1/3 * (2/vs**3 + 1/vl**3))**(-1/3)
        props.update({"v_longitudinal_ms": vl, "v_transverse_ms": vs, "v_mean_ms": vm})
        if n_atoms and vol and vol > 0:
            props["T_Debye_K"] = (1.05457e-34 / 1.3806e-23) * vm * (6 * np.pi**2 * (n_atoms / (vol * 1e-30)))**(1/3)
    return props
