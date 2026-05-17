"""
core_physics.py  —  Crystal representation, symmetry, strain, and elastic fitting.
No silent exceptions. Engine-agnostic.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import ase
    from runstate import RunState, Step


import config
import numpy as np
import spglib

# ─────────────────────────────────────────────────────────────────────────────
# Mathematical Utilities
# ─────────────────────────────────────────────────────────────────────────────


def lattice_to_abc(L: np.ndarray) -> tuple[float, float, float, float, float, float]:
    """Convert 3x3 matrix to (a, b, c, alpha, beta, gamma) in degrees."""
    a, b, c = np.linalg.norm(L, axis=1)
    ca = np.clip(np.dot(L[1], L[2]) / (b * c), -1.0, 1.0)
    cb = np.clip(np.dot(L[0], L[2]) / (a * c), -1.0, 1.0)
    cg = np.clip(np.dot(L[0], L[1]) / (a * b), -1.0, 1.0)
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
    """Convert (a, b, c, alpha, beta, gamma) to upper-triangular 3x3 matrix."""
    ar, br, gr = math.radians(alpha), math.radians(beta), math.radians(gamma)
    sg = max(math.sin(gr), 1e-15)
    cx = c * math.cos(br)
    cy = c * (math.cos(ar) - math.cos(br) * math.cos(gr)) / sg
    cz = math.sqrt(max(c**2 - cx**2 - cy**2, 0.0))
    return np.array([[a, 0.0, 0.0], [b * math.cos(gr), b * sg, 0.0], [cx, cy, cz]])


# ─────────────────────────────────────────────────────────────────────────────
# Core Data Structures
# ─────────────────────────────────────────────────────────────────────────────

# Shared lookup for lattice_type and strain_pattern_code.
# Each row: (min_spacegroup_number, lattice_name, strain_pattern_code).
# Rows are tested in descending order; first match wins.
_SG_THRESHOLDS: list[tuple[int, str, int]] = [
    (195, "cubic",        5),
    (168, "hexagonal",    7),
    (143, "trigonal",     6),
    (75,  "tetragonal",   4),
    (16,  "orthorhombic", 3),
    (3,   "monoclinic",   2),
]


@dataclass
class Crystal:
    """Universal crystal lattice with self-aware symmetry via spglib."""

    lattice: np.ndarray
    frac_coords: np.ndarray
    sites: list[dict[str, float]]

    @property
    def num_atoms(self) -> int:
        return len(self.sites)

    @property
    def species(self) -> list[str]:
        """Returns a list of unique element symbols present in the crystal."""
        all_species = set()
        for site in self.sites:
            all_species.update(site.keys())
        return sorted(list(all_species))


    @property
    def volume(self) -> float:
        return abs(
            float(np.dot(self.lattice[0], np.cross(self.lattice[1], self.lattice[2])))
        )

    # Manual cache — functools.cached_property is incompatible with @dataclass:
    # the dataclass machinery introspects descriptors at class-creation time and
    # treats them as field definitions, corrupting __init__ and __eq__.
    _sym_dataset: Any = field(default=None, repr=False, init=False, compare=False)

    def _get_spglib_dataset(self) -> Any:
        if self._sym_dataset is None:
            # For symmetry analysis, represent each site by its dominant species.
            # This is a simplification for VCA crystals. For SQS/direct, there's only one species per site.
            temp_species = []
            for site in self.sites:
                if not site:
                    # Handle empty site dict if it can happen
                    temp_species.append("X") # Placeholder
                else:
                    # Get species with highest occupancy
                    dominant_species = max(site, key=site.get)
                    temp_species.append(dominant_species)

            nums = [
                config.ELEMENTS.get(s.capitalize(), {}).get("Z", 1)
                for s in temp_species
            ]
            cell = (self.lattice, self.frac_coords, nums)
            # symprec=1e-5 is the default; callers needing a different tolerance
            # should call spglib.get_symmetry_dataset directly.
            dataset = spglib.get_symmetry_dataset(cell, symprec=1e-5)
            if dataset is None:
                raise ValueError(
                    "spglib failed to find symmetry (symprec=1e-5). "
                    "Atoms may be overlapping or cell is heavily distorted."
                )
            self._sym_dataset = dataset
        return self._sym_dataset

    def clear_cache(self) -> None:
        """Evict cached spglib dataset — call after mutating lattice or coords."""
        self._sym_dataset = None

    @property
    def spacegroup_number(self) -> int:
        return self._get_spglib_dataset().number

    @property
    def spacegroup_symbol(self) -> str:
        return self._get_spglib_dataset().international

    @property
    def lattice_type(self) -> str:
        sg = self.spacegroup_number
        for threshold, name, _ in _SG_THRESHOLDS:
            if sg >= threshold:
                return name
        return "triclinic"

    @property
    def strain_pattern_code(self) -> int:
        sg = self.spacegroup_number
        for threshold, _, code in _SG_THRESHOLDS:
            if sg >= threshold:
                return code
        return 1

    def get_symmetry_operations(
        self,
        cartesian_rotations: bool = False,
        symprec: float = 1e-5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (rotations, translations).
        If cartesian_rotations=True, transforms fractional rotation matrices to
        Cartesian via R_c = L^T * R_f * (L^T)^-1. Translations remain fractional.
        """
        dataset = self._get_spglib_dataset()
        rotations = np.array(dataset.rotations)
        translations = np.array(dataset.translations)

        if cartesian_rotations:
            L_T = self.lattice.T
            L_T_inv = np.linalg.inv(L_T)
            rotations = np.array(
                [np.round(L_T @ R_frac @ L_T_inv).astype(int) for R_frac in rotations]
            )

        translations[np.abs(translations) < 1e-12] = 0.0
        return rotations, translations

    # ── New Crystal methods ───────────────────────────────────────────────────

    def to_ase(self) -> "ase.Atoms":
        """Convert to ASE Atoms.

        NOTE: For sites with fractional occupancy, this method simplifies by
        choosing the species with the highest occupancy. This is a limitation
        of the ASE Atoms object, which does not natively support VCA.
        """
        try:
            import ase
            import ase.data
        except ImportError:
            raise ImportError("pip install ase")

        cart_positions = self.frac_coords @ self.lattice

        symbols = []
        for site in self.sites:
            if site:
                dominant_species = max(site, key=site.get)
                symbols.append(dominant_species)
            else:
                symbols.append("X")  # Placeholder for an empty site

        return ase.Atoms(
            symbols=symbols,
            positions=cart_positions,
            cell=self.lattice,
            pbc=True,
        )

    @classmethod
    def from_ase(cls, atoms: "ase.Atoms") -> "Crystal":
        """Construct Crystal from ASE Atoms without standardization."""
        try:
            import ase.data
        except ImportError:
            raise ImportError("pip install ase")

        species = [ase.data.chemical_symbols[z] for z in atoms.numbers]
        sites = [{s: 1.0} for s in species]
        frac_coords = atoms.get_scaled_positions()
        lattice = np.array(atoms.get_cell())
        return cls(lattice=lattice, frac_coords=frac_coords, sites=sites)

    def vec(self) -> float:
        """Computes the average Valence Electron Concentration (VEC) per site."""
        total_electrons = 0.0
        num_sites = len(self.sites)

        if num_sites == 0:
            return 0.0

        for site in self.sites:
            for element, fraction in site.items():
                total_electrons += fraction * config.ELEMENTS.get(
                    element.capitalize(), {}
                ).get("val", 0)

        return total_electrons / num_sites


# ─────────────────────────────────────────────────────────────────────────────
# Initialization & Standardization
# ─────────────────────────────────────────────────────────────────────────────


def load_crystal(file_path: Path) -> Crystal:
    """Load any geometry file and return a canonical primitive Crystal."""
    raw = _read_raw(file_path)
    return standardize_crystal(raw, symprec=1e-5)


def read_geometry(file_path: Path) -> Crystal:
    """Read geometry file without standardization. Public API for engines."""
    return _read_raw(file_path)


def _read_raw(file_path: Path) -> Crystal:
    """Parse geometry file into Crystal. Uses native fast parsers, falls back to ASE."""
    ext = file_path.suffix.lower()
    text = file_path.read_text(encoding="utf-8", errors="replace")

    # 1. Швидкі нативні парсери
    if ext == ".cif":
        return _read_cif_raw(text)
    elif ext == ".cell":
        return _read_castep_cell(text)
    elif (
        ext in (".poscar", ".vasp", "")
        or file_path.name.startswith("POSCAR")
        or file_path.name.startswith("CONTCAR")
    ):
        return _read_vasp_poscar(text)

    # 2. Універсальний Fallback через ASE для всіх інших форматів (rndstr.in, .xyz, .xsf, etc.)
    try:
        import warnings

        import ase.io

        # ASE іноді сипле UserWarning про нестандартні теги — глушимо їх
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            atoms = ase.io.read(file_path)

        return Crystal.from_ase(atoms)

    except ImportError:
        raise ValueError(
            f"Unsupported geometry format: '{file_path.name}'. "
            f"  Hint: Install ASE (`pip install ase`) to automatically read this and 100+ other formats."
        )
    except Exception as e:
        raise ValueError(f"Failed to read '{file_path.name}' via ASE: {e}")


def standardize_crystal(crystal: Crystal, *, symprec: float) -> Crystal:
    """Standardize to the primitive cell. symprec is required — differs by use-case."""
    # For standardization, represent each site by its dominant species.
    temp_species = []
    for site in crystal.sites:
        if site:
            dominant_species = max(site, key=site.get)
            temp_species.append(dominant_species)
        else:
            temp_species.append("X")

    nums = [
        config.ELEMENTS.get(s.capitalize(), {}).get("Z", 1) for s in temp_species
    ]
    cell = (crystal.lattice, crystal.frac_coords, nums)
    std = spglib.standardize_cell(cell, to_primitive=True, symprec=symprec)
    if std is None:
        raise ValueError("spglib failed to standardize crystal.")

    L, f_coords, n = std
    z_to_sym = {v["Z"]: k for k, v in config.ELEMENTS.items()}
    new_species = [z_to_sym.get(z, "X") for z in n]
    new_sites = [{s: 1.0} for s in new_species]
    return Crystal(lattice=L, frac_coords=f_coords, sites=new_sites)


def _read_cif_raw(text: str) -> Crystal:
    params = {}
    for k in (
        "_cell_length_a",
        "_cell_length_b",
        "_cell_length_c",
        "_cell_angle_alpha",
        "_cell_angle_beta",
        "_cell_angle_gamma",
    ):
        m = re.search(rf"^{re.escape(k)}\s+([\d.]+(?:\(\d+\))?)", text, re.M | re.I)
        if m:
            params[k] = float(re.sub(r"\(\d+\)$", "", m.group(1)))

    L = abc_to_lattice(
        params.get("_cell_length_a", 1.0),
        params.get("_cell_length_b", 1.0),
        params.get("_cell_length_c", 1.0),
        params.get("_cell_angle_alpha", 90.0),
        params.get("_cell_angle_beta", 90.0),
        params.get("_cell_angle_gamma", 90.0),
    )

    sp, fc_list = [], []
    loop_pat = re.compile(
            r"loop_\s+((?:_atom_site_\S+\s+)+)((?:(?!loop_|_\S+\s+).*\n?)+)", re.M
    )
    for lm in loop_pat.finditer(text):
        cols = re.findall(r"(_atom_site_\S+)", lm.group(1))
        if "_atom_site_fract_x" not in cols:
            continue
        t_idx = next(
            (
                i
                for i, c in enumerate(cols)
                if c in ("_atom_site_type_symbol", "_atom_site_label")
            ),
            0,
        )
        xi = cols.index("_atom_site_fract_x")
        yi = cols.index("_atom_site_fract_y")
        zi = cols.index("_atom_site_fract_z")

        for line in lm.group(2).splitlines():
            parts = line.split()
            if len(parts) < len(cols):
                continue
            try:
                x, y, z = (
                    float(re.sub(r"\(\d+\)$", "", parts[i])) for i in (xi, yi, zi)
                )
                sym_match = re.match(r"([A-Za-z]+)", parts[t_idx])
                if sym_match:
                    sp.append(sym_match.group(1).capitalize())
                    fc_list.append([x, y, z])
            except ValueError:
                continue

    sites = [{s: 1.0} for s in sp]
    return Crystal(lattice=L, frac_coords=np.array(fc_list), sites=sites)


def _read_castep_cell(text: str) -> Crystal:
    m_lat = re.search(r"%BLOCK\s+LATTICE_CART\s*(.*?)%ENDBLOCK\s+LATTICE_CART",
        text,
        re.DOTALL | re.I,
    )
    vecs = []
    for line in m_lat.group(1).splitlines():
        s = line.strip().lower()
        if not s or s in {"ang", "bohr", "a.u.", "angstrom"}:
            continue
        parts = s.split()
        if len(parts) >= 3:
            try:
                vecs.append(np.array([float(p) for p in parts[:3]]))
            except ValueError:
                continue

    m_pos = re.search(r"%BLOCK\s+POSITIONS_FRAC\s*(.*?)%ENDBLOCK\s+POSITIONS_FRAC",
        text,
        re.DOTALL | re.I,
    )

    # In VCA, multiple lines can refer to the same site. We need to group them.
    # We use a dictionary where keys are coordinate tuples.
    site_map = {}

    for line in m_pos.group(1).splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[0].isalpha():
            species = parts[0].capitalize()
            coords = tuple(float(p) for p in parts[1:4])

            occupancy = 1.0
            mixture_match = re.search(r"MIXTURE:\(\s*\d+\s+([\d\.]+)\s*\)", line, re.IGNORECASE)
            if mixture_match:
                occupancy = float(mixture_match.group(1))

            if coords not in site_map:
                site_map[coords] = {}
            site_map[coords][species] = occupancy

    # Now, build the final lists for the Crystal object
    fc_list = list(site_map.keys())
    sites_list = [site_map[coords] for coords in fc_list]

    return Crystal(lattice=np.array(vecs), frac_coords=np.array(fc_list), sites=sites_list)


def _read_vasp_poscar(text: str) -> Crystal:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    scale = float(lines[1].split()[0])
    vecs = [[float(x) * scale for x in line.split()[:3]] for line in lines[2:5]]

    species_names = lines[5].split()
    counts = [int(x) for x in lines[6].split()]

    raw_sp = []
    for name, count in zip(species_names, counts):
        raw_sp.extend([name.capitalize()] * count)

    start_idx = 8 if lines[7].lower().startswith(("d", "c")) else 7
    if lines[7].lower().startswith("s"):
        start_idx += 1

    raw_fc = [
        [float(x) for x in line.split()[:3]]
        for line in lines[start_idx : start_idx + sum(counts)]
    ]

    if lines[start_idx - 1].lower().startswith(("c", "k")):
        raw_fc = (np.array(raw_fc) @ np.linalg.inv(np.array(vecs))).tolist()

    sites = [{s: 1.0} for s in raw_sp]
    return Crystal(lattice=np.array(vecs), frac_coords=np.array(raw_fc), sites=sites)


# ─────────────────────────────────────────────────────────────────────────────
# Strain & Elasticity
# ─────────────────────────────────────────────────────────────────────────────

_STRAIN_PATTERNS: dict[int, list[list[float]]] = {
    1: [
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
    ],
    2: [[1, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 1], [0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0]],
    3: [[1, 0, 0, 1, 0, 0], [0, 1, 0, 0, 1, 0], [0, 0, 1, 0, 0, 1]],
    4: [[1, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 1]],
    5: [[1, 0, 0, 1, 0, 0]],  # cubic — minimal
    6: [[1, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0]],
    7: [[0, 0, 1, 0, 0, 0], [1, 0, 0, 1, 0, 0]],
}


@dataclass
class StrainStep:
    pattern_idx: int
    step_idx: int
    magnitude: float
    strain_voigt: np.ndarray

    @property
    def name(self) -> str:
        return f"_cij__{self.pattern_idx}__{self.step_idx}"


def generate_strain_steps(
    crystal: Crystal, max_strain: float = 0.003, n_steps: int = 3
) -> list[StrainStep]:
    """Generate universal strain steps for ANY crystal symmetry.
    Always uses the full 6-dimensional strain basis to ensure the Cij matrix
    is never singular, enabling true universal fitting.
    Returns engineering Voigt strains (mag, not 0.5*mag for shear).
    """
    patterns = [
        [1, 0, 0, 0, 0, 0],  # C11, C12, C13
        [0, 1, 0, 0, 0, 0],  # C22, C23
        [0, 0, 1, 0, 0, 0],  # C33
        [0, 0, 0, 1, 0, 0],  # C44
        [0, 0, 0, 0, 1, 0],  # C55
        [0, 0, 0, 0, 0, 1],  # C66
    ]

    steps: list[StrainStep] = []
    for pi, pattern in enumerate(patterns, 1):
        sc = 0
        for k in range(n_steps):
            for neg in (False, True):
                sc += 1
                mag = ((k + 1) / n_steps) * max_strain * (-1 if neg else 1)
                v = np.zeros(6)
                for i, p in enumerate(pattern):
                    if p:
                        # Engineering Voigt strain: ε1, ε2, ε3, γ4, γ5, γ6
                        # No 0.5 factor here; that belongs in the deformation gradient building.
                        # No division by lattice vector lengths: strain must be uniform.
                        v[i] = p * mag
                steps.append(StrainStep(pi, sc, mag, v))
    return steps

def fit_cij_universal(
    stresses: list[np.ndarray],
    strains: list[np.ndarray],
    density_gcm3: float | None = None,
    n_atoms: int | None = None,
    volume_ang3: float | None = None,
) -> dict[str, Any]:
    """Універсальний фіттер матриці Cij 6x6 для БУДЬ-ЯКОЇ сингонії з урахуванням залишкового стресу."""
    if len(stresses) < 3:
        return {"error": f"Need >= 3 stress tensors, got {len(stresses)}"}

    E = np.vstack(strains)
    S = np.vstack(stresses)

    # МАГІЯ ТУТ: Додаємо колонку одиниць для фітування вільного члена (залишкового стресу S0)
    E_with_intercept = np.hstack([E, np.ones((E.shape[0], 1))])

    try:
        # Розв'язуємо E * C_T + 1 * S0 = S
        sol, residuals, rank, s_vals = np.linalg.lstsq(E_with_intercept, S, rcond=None)
        C_T = sol[:-1, :]  # Матриця пружності (6x6)
        S0 = sol[-1, :]    # Залишковий стрес (6,)
        C = C_T.T
    except np.linalg.LinAlgError:
        return {"error": "Linear algebra solver failed to fit Cij matrix."}

    # Термодинамічна вимога: матриця Cij має бути симетричною
    C = (C + C.T) / 2.0

    # Розрахунок якості фіту (R^2)
    S_pred = E_with_intercept @ sol
    S_mean = np.mean(S, axis=0)
    SS_tot = np.sum((S - S_mean)**2, axis=0)
    SS_res = np.sum((S - S_pred)**2, axis=0)

    # Розрахуємо глобальний R2 (зважений за дисперсією компонент)
    total_SS_tot = np.sum(SS_tot)
    total_SS_res = np.sum(SS_res)
    r2_global = 1.0 - (total_SS_res / total_SS_tot) if total_SS_tot > 1e-12 else 0.0

    # Мінімальний R2 серед компонент з суттєвою дисперсією (> 1% від макс)
    max_ss_tot = np.max(SS_tot)
    r2_vals = []
    for i in range(6):
        if SS_tot[i] > 0.01 * max_ss_tot:
            r2_vals.append(1.0 - (SS_res[i] / SS_tot[i]))
    
    r2_min = float(np.min(r2_vals)) if r2_vals else r2_global

    props = universal_vrh(C, density_gcm3, n_atoms, volume_ang3)

    result = {
        "born_stable": "yes" if props.get("born_stable") else "no",
        "elastic_n_points": str(len(stresses)),
        "elastic_R2_min": f"{r2_min:.4f}",
        "elastic_R2_global": f"{r2_global:.4f}",
        "residual_pressure_GPa": f"{np.mean(S0[:3]):.4f}", 
    }

    # Extract all Voigt components: Cij (i,j in 1..6)
    # Voigt index mapping: 0=xx, 1=yy, 2=zz, 3=yz, 4=xz, 5=xy
    for i in range(6):
        for j in range(i, 6):
            key = f"C{i+1}{j+1}"
            result[key] = f"{C[i,j]:.4f}"

    if r2_min < 0.90:
        result["elastic_quality_note"] = "Poor Fit (Non-linear/Unstable)"
    elif r2_min < 0.98:
        result["elastic_quality_note"] = "Acceptable Fit"

    for k, v in props.items():
        if k not in result:
            result[k] = f"{v:.4f}" if isinstance(v, float) else v

    return result

def universal_vrh(
    C: np.ndarray,
    density: float | None = None,
    n_atoms: int | None = None,
    vol: float | None = None,
) -> dict[str, Any]:
    """
    Універсальний розрахунок Voigt-Reuss-Hill.
    Захищено від розрахунку твердості для нестабільних ґраток.
    """
    if C.shape != (6, 6):
        return {"error": f"Matrix must be 6x6, got {C.shape}"}

    try:
        eigenvalues = np.linalg.eigvals(C)
        born_stable = bool(np.all(eigenvalues > 0))
    except np.linalg.LinAlgError:
        return {"error": "Cij matrix is invalid (failed eigenvalue calculation)."}

    try:
        S = np.linalg.inv(C)
    except np.linalg.LinAlgError:
        return {"error": "Cij matrix is singular (non-invertible).", "born_stable": False}

    # Наближення Фойгта (Voigt Bounds)
    K_V = (C[0, 0] + C[1, 1] + C[2, 2]) + 2 * (C[0, 1] + C[0, 2] + C[1, 2])
    K_V /= 9.0

    G_V = (C[0, 0] + C[1, 1] + C[2, 2]) - (C[0, 1] + C[0, 2] + C[1, 2]) + 3 * (C[3, 3] + C[4, 4] + C[5, 5])
    G_V /= 15.0

    # Наближення Ройсса (Reuss Bounds)
    K_R_inv = (S[0, 0] + S[1, 1] + S[2, 2]) + 2 * (S[0, 1] + S[0, 2] + S[1, 2])
    K_R = 1.0 / K_R_inv if abs(K_R_inv) > 1e-12 else float('nan')

    G_R_inv = 4 * (S[0, 0] + S[1, 1] + S[2, 2]) - 4 * (S[0, 1] + S[0, 2] + S[1, 2]) + 3 * (S[3, 3] + S[4, 4] + S[5, 5])
    G_R = 15.0 / G_R_inv if abs(G_R_inv) > 1e-12 else float('nan')

    K_H = (K_V + K_R) / 2.0
    G_H = (G_V + G_R) / 2.0

    props: dict[str, Any] = {
        "born_stable": born_stable,
        "B_Voigt_GPa": K_V, "B_Reuss_GPa": K_R, "B_Hill_GPa": K_H,
        "G_Voigt_GPa": G_V, "G_Reuss_GPa": G_R, "G_Hill_GPa": G_H,
    }

    # РАХУЄМО ТВЕРДІСТЬ ТІЛЬКИ ЯКЩО КРИСТАЛ ФІЗИЧНО СТАБІЛЬНИЙ
    if born_stable and K_H > 0 and G_H > 0:
        E = 9 * K_H * G_H / (3 * K_H + G_H)
        nu = (3 * K_H - 2 * G_H) / (2 * (3 * K_H + G_H))
        pugh = G_H / K_H
        props.update({
            "E_GPa": E, "nu": nu,
            "Pugh_ratio": pugh,
            "Cauchy_pressure_GPa": K_H - (2 * G_H / 3)
        })

        try:
            props["H_Vickers_Chen_GPa"] = max(0.0, float(2 * (pugh**2 * G_H) ** 0.585 - 3))
            props["H_Vickers_Tian_GPa"] = max(0.0, float(0.92 * (pugh**1.137) * (G_H**0.708)))
        except Exception:
            pass

    if density and density > 0 and K_H > 0 and G_H > 0:
        rho = density * 1e3
        vl = ((K_H + 4 * G_H / 3) * 1e9 / rho) ** 0.5
        vs = (G_H * 1e9 / rho) ** 0.5
        vm = (1 / 3 * (2 / vs**3 + 1 / vl**3)) ** (-1 / 3)
        props.update({"v_longitudinal_ms": vl, "v_transverse_ms": vs, "v_mean_ms": vm})
        if n_atoms and vol and vol > 0:
            props["T_Debye_K"] = (1.05457e-34 / 1.3806e-23) * vm * (6 * np.pi**2 * (n_atoms / (vol * 1e-30))) ** (1 / 3)

    return props

def vec_for_system(
    species_mix: list[tuple[str, float]],
    nonmetal: str | None = None,
) -> float:
    """Valence-electron concentration for a composition spec.

    Args:
        species_mix: list of (element, fraction) on the substituted sublattice.
        nonmetal:    symbol of the non-metal sublattice species (e.g. 'C', 'N'),
                     or None for pure-metal systems.
    """
    metal_vec = sum(
        frac * config.ELEMENTS.get(elem.capitalize(), {}).get("val", 0)
        for elem, frac in species_mix
    )
    nm_vec = (
        config.ELEMENTS.get(nonmetal.capitalize(), {}).get("val", 0)
        if nonmetal else 0
    )
    return metal_vec + nm_vec


def _maybe_decorate_dh_mix(state: "RunState") -> None:
    """Add dH_mix_meV_per_fu where computable. Only for binary VCA."""
    if state.single_mode or len(state.species) > 2:
        return
    dh_data = mixing_enthalpy(state.steps)
    for x, _, dh in dh_data:
        for s in state.steps:
            if s.status == "done" and abs(s.concentration - x) < 1e-9:
                s.parsed.setdefault("dH_mix_meV_per_fu", f"{dh:.3f}")


def mixing_enthalpy(
    steps: list["Step"],
) -> list[tuple[float, float, float]]:
    def _h(s: "Step") -> float | None:
        try:
            return float(s.parsed["enthalpy_eV"])
        except (KeyError, TypeError, ValueError):
            return None

    done = [s for s in steps if s.status == "done" and _h(s) is not None]
    try:
        h0 = next(_h(s) for s in done if abs(s.concentration) < 1e-4)
        h1 = next(_h(s) for s in done if abs(s.concentration - 1) < 1e-4)
    except StopIteration:
        return []

    out: list[tuple[float, float, float]] = []
    for s in done:
        h = _h(s)
        if h is None:
            continue
        dh = (h - ((1 - s.concentration) * h0 + s.concentration * h1)) * 1000
        out.append((s.concentration, h, dh))
    return sorted(out, key=lambda t: t[0])
