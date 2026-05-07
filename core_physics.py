"""
core_physics.py  —  Crystal representation, symmetry, strain, and elastic fitting.
No silent exceptions. Engine-agnostic.
"""

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    ar, br, gr = np.radians(alpha), np.radians(beta), np.radians(gamma)
    sg = max(float(np.sin(gr)), 1e-15)
    cx = c * float(np.cos(br))
    cy = c * (float(np.cos(ar)) - float(np.cos(br)) * float(np.cos(gr))) / sg
    cz = float(np.sqrt(max(c**2 - cx**2 - cy**2, 0.0)))
    return np.array([[a, 0.0, 0.0], [b * float(np.cos(gr)), b * sg, 0.0], [cx, cy, cz]])


# ─────────────────────────────────────────────────────────────────────────────
# Core Data Structures
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Crystal:
    """Universal crystal lattice with self-aware symmetry via spglib."""

    lattice: np.ndarray
    frac_coords: np.ndarray
    species: list[str]

    @property
    def num_atoms(self) -> int:
        return len(self.species)

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
            nums = [
                config.ELEMENTS.get(s.capitalize(), {}).get("Z", 1)
                for s in self.species
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
        if sg >= 195:
            return "cubic"
        if sg >= 168:
            return "hexagonal"
        if sg >= 143:
            return "trigonal"
        if sg >= 75:
            return "tetragonal"
        if sg >= 16:
            return "orthorhombic"
        if sg >= 3:
            return "monoclinic"
        return "triclinic"

    @property
    def strain_pattern_code(self) -> int:
        sg = self.spacegroup_number
        if sg >= 195:
            return 5  # cubic
        if sg >= 168:
            return 7  # hexagonal
        if sg >= 143:
            return 6  # trigonal
        if sg >= 75:
            return 4  # tetragonal
        if sg >= 16:
            return 3  # orthorhombic
        if sg >= 3:
            return 2  # monoclinic
        return 1  # triclinic

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

    def with_vegard(
        self, template_element: str, target_mix: dict[str, float]
    ) -> "Crystal":
        """Return a new Crystal with lattice uniformly scaled by Vegard law.

        k = r_mix / r_template, where r_mix weights the alloy atomic radii.
        Returns self unchanged when k ≈ 1 or when radii are unavailable.
        """
        eps = 1e-9
        nonzero_mix = {e: f for e, f in target_mix.items() if f > eps}
        if not nonzero_mix:
            return self

        r_template = config.ELEMENTS.get(template_element.capitalize(), {}).get(
            "rad", 0.0
        )
        if r_template < 1e-6:
            return self

        total_new_frac = sum(nonzero_mix.values())
        template_remaining = max(0.0, 1.0 - total_new_frac)
        # Vegard: r_mix = r_tmpl*(1-Σf) + Σ(r_i * f_i)
        r_mix = r_template * template_remaining + sum(
            config.ELEMENTS.get(e.capitalize(), {}).get("rad", 0.0) * f
            for e, f in nonzero_mix.items()
        )

        if r_mix < 1e-6 or abs(r_mix / r_template - 1.0) < 1e-9:
            return self

        return Crystal(
            lattice=self.lattice * (r_mix / r_template),
            frac_coords=self.frac_coords.copy(),
            species=list(self.species),
        )

    def to_ase(self, mix: dict[str, float] | None = None) -> "ase.Atoms":
        """Convert to ASE Atoms. If mix is provided, VCA-average atomic numbers/masses."""
        try:
            import ase
            import ase.data
        except ImportError:
            raise ImportError("pip install ase")

        cart_positions = self.frac_coords @ self.lattice

        if mix is None:
            numbers = [
                config.ELEMENTS.get(s.capitalize(), {}).get("Z", 1)
                for s in self.species
            ]
            return ase.Atoms(
                numbers=numbers,
                positions=cart_positions,
                cell=self.lattice,
                pbc=True,
            )

        # VCA averaging: compute a single effective Z and mass for the mix
        averaged_Z = sum(
            config.ELEMENTS[e.capitalize()]["Z"] * f for e, f in mix.items()
        )
        averaged_mass = sum(
            ase.data.atomic_masses[config.ELEMENTS[e.capitalize()]["Z"]] * f
            for e, f in mix.items()
        )
        template_syms = set(mix.keys())

        numbers = []
        masses = []
        for s in self.species:
            if s.capitalize() in {t.capitalize() for t in template_syms}:
                numbers.append(int(round(averaged_Z)))
                masses.append(averaged_mass)
            else:
                z = config.ELEMENTS.get(s.capitalize(), {}).get("Z", 1)
                numbers.append(z)
                masses.append(ase.data.atomic_masses[z])

        atoms = ase.Atoms(
            numbers=numbers,
            positions=cart_positions,
            cell=self.lattice,
            pbc=True,
        )
        atoms.set_masses(masses)
        return atoms

    @classmethod
    def from_ase(cls, atoms: "ase.Atoms") -> "Crystal":
        """Construct Crystal from ASE Atoms without standardization."""
        try:
            import ase.data
        except ImportError:
            raise ImportError("pip install ase")

        species = [ase.data.chemical_symbols[z] for z in atoms.numbers]
        frac_coords = atoms.get_scaled_positions()
        lattice = np.array(atoms.get_cell())
        return cls(lattice=lattice, frac_coords=frac_coords, species=species)

    def vec(
        self,
        species_mix: list[tuple[str, float]],
        nonmetal: str | None = None,
    ) -> float:
        """Valence Electron Concentration for a given composition species_mix."""
        metal_vec = sum(
            frac * config.ELEMENTS.get(elem.capitalize(), {}).get("val", 0)
            for elem, frac in species_mix
        )
        nm_vec = (
            config.ELEMENTS.get(nonmetal.capitalize(), {}).get("val", 0)
            if nonmetal
            else 0
        )
        return metal_vec + nm_vec


# ─────────────────────────────────────────────────────────────────────────────
# Initialization & Standardization
# ─────────────────────────────────────────────────────────────────────────────


def load_crystal(file_path: Path) -> Crystal:
    """Load any geometry file and return a canonical primitive Crystal."""
    raw = _read_raw(file_path)
    return standardize_crystal(raw, symprec=1e-5)


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
            f"Unsupported geometry format: '{file_path.name}'. \n"
            f"  Hint: Install ASE (`pip install ase`) to automatically read this and 100+ other formats."
        )
    except Exception as e:
        raise ValueError(f"Failed to read '{file_path.name}' via ASE: {e}")


def standardize_crystal(crystal: Crystal, *, symprec: float) -> Crystal:
    """Standardize to the primitive cell. symprec is required — differs by use-case."""
    nums = [
        config.ELEMENTS.get(s.capitalize(), {}).get("Z", 1) for s in crystal.species
    ]
    cell = (crystal.lattice, crystal.frac_coords, nums)
    std = spglib.standardize_cell(cell, to_primitive=True, symprec=symprec)
    if std is None:
        raise ValueError("spglib failed to standardize crystal.")

    L, f_coords, n = std
    z_to_sym = {v["Z"]: k for k, v in config.ELEMENTS.items()}
    species = [z_to_sym.get(z, "X") for z in n]
    return Crystal(lattice=L, frac_coords=f_coords, species=species)


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

    return Crystal(lattice=L, frac_coords=np.array(fc_list), species=sp)


def _read_castep_cell(text: str) -> Crystal:
    m_lat = re.search(
        r"%BLOCK\s+LATTICE_CART\s*\n(.*?)%ENDBLOCK\s+LATTICE_CART",
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

    m_pos = re.search(
        r"%BLOCK\s+POSITIONS_FRAC\s*\n(.*?)%ENDBLOCK\s+POSITIONS_FRAC",
        text,
        re.DOTALL | re.I,
    )
    raw_pairs: dict[tuple, str] = {}
    for line in m_pos.group(1).splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[0].isalpha():
            key = tuple(
                np.round([float(parts[1]), float(parts[2]), float(parts[3])], 4)
            )
            if key not in raw_pairs:
                raw_pairs[key] = parts[0].capitalize()

    sp = list(raw_pairs.values())
    fc = [list(k) for k in raw_pairs.keys()]
    return Crystal(lattice=np.array(vecs), frac_coords=np.array(fc), species=sp)


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

    raw_pairs: dict[tuple, str] = {}
    for s, c in zip(raw_sp, raw_fc):
        key = tuple(np.round(c, 4))
        if key not in raw_pairs:
            raw_pairs[key] = s

    sp = list(raw_pairs.values())
    fc = [list(k) for k in raw_pairs.keys()]
    return Crystal(lattice=np.array(vecs), frac_coords=np.array(fc), species=sp)


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
    """Generate optimal strain steps based on crystal symmetry."""
    pattern_code = crystal.strain_pattern_code
    patterns = _STRAIN_PATTERNS[pattern_code]

    L = crystal.lattice
    if crystal.lattice_type == "cubic":
        vol = crystal.volume
        a_conv = (
            (4.0 * vol) ** (1.0 / 3.0)
            if crystal.spacegroup_symbol.startswith("F")
            else (2.0 * vol) ** (1.0 / 3.0)
        )
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
                    if p:
                        v[i] = p * mag / lens[i] if i < 3 else 0.5 * p * mag / lens[i]
                steps.append(StrainStep(pi, sc, mag, v))
    return steps


def fit_cij_cubic(
    stresses: list[np.ndarray],
    strains: list[np.ndarray],
    density_gcm3: float | None = None,
    n_atoms: int | None = None,
    volume_ang3: float | None = None,
) -> dict[str, Any]:
    if len(stresses) < 3:
        return {"error": f"Need >= 3 stress tensors, got {len(stresses)}"}
    sa, ea = np.array(stresses), np.array(strains)

    def _ols(x, y):
        n = len(x)
        d = n * np.dot(x, x) - x.sum() ** 2
        if abs(d) < 1e-30:
            return float("nan"), 0.0
        slope = (n * np.dot(x, y) - x.sum() * y.sum()) / d
        ic = (y.sum() - slope * x.sum()) / n
        tot = float(np.sum((y - y.sum() / n) ** 2))
        r2 = (
            1.0 - float(np.sum((y - slope * x - ic) ** 2)) / tot if tot > 1e-12 else 1.0
        )
        return slope, r2

    c11, r2_11 = _ols(ea[:, 0], sa[:, 0])
    c12, r2_12 = _ols(ea[:, 0], sa[:, 1])
    c44, r2_44 = _ols(ea[:, 3], sa[:, 3])

    if any(np.isnan(v) for v in (c11, c12, c44)):
        return {"error": "NaN in OLS fit"}

    props = cubic_vrh(c11, c12, c44, density_gcm3, n_atoms, volume_ang3)

    # Always record raw elastic constants even when Born stability is violated
    result = {
        "C11": f"{c11:.4f}",
        "C12": f"{c12:.4f}",
        "C44": f"{c44:.4f}",
        "born_stable": "yes" if props.get("born_stable") else "no",
        "elastic_n_points": str(len(stresses)),
        "elastic_R2_min": f"{min(r2_11, r2_12, r2_44):.4f}",
    }
    if min(r2_11, r2_12, r2_44) < 0.99:
        result["elastic_quality_note"] = "Low R2"

    # Overlay VRH-derived quantities when available
    for k, v in props.items():
        if k not in result:
            result[k] = f"{v:.4f}" if isinstance(v, float) else v

    return result


def cubic_vrh(
    c11: float,
    c12: float,
    c44: float,
    density: float | None = None,
    n_atoms: int | None = None,
    vol: float | None = None,
) -> dict[str, Any]:
    born_stable = c11 > 0 and c44 > 0 and c11 > abs(c12) and c11 + 2 * c12 > 0
    props: dict[str, Any] = {
        "C11": c11,
        "C12": c12,
        "C44": c44,
        "born_stable": born_stable,
    }
    if not born_stable:
        return props  # Return raw constants; caller decides how to handle

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
    pugh = gh / bh

    props.update(
        {
            "B_Voigt_GPa": bv,
            "B_Reuss_GPa": br,
            "B_Hill_GPa": bh,
            "G_Voigt_GPa": gv,
            "G_Reuss_GPa": gr,
            "G_Hill_GPa": gh,
            "E_GPa": 9 * bh * gh / (3 * bh + gh),
            "nu": (3 * bh - 2 * gh) / (2 * (3 * bh + gh)),
            "Zener_A": 2 * c44 / (c11 - c12),
            "Pugh_ratio": pugh,
            "Cauchy_pressure_GPa": c12 - c44,
            "C_prime_GPa": (c11 - c12) / 2,
            # Chen 2011 and Tian 2012 empirical hardness models
            "H_Vickers_Chen_GPa": max(0.0, 2 * (pugh**2 * gh) ** 0.585 - 3),
            "H_Vickers_Tian_GPa": 0.92 * (pugh**1.137) * (gh**0.708),
        }
    )

    if density and density > 0:
        rho = density * 1e3  # g/cm³ → kg/m³
        vl = ((bh + 4 * gh / 3) * 1e9 / rho) ** 0.5
        vs = (gh * 1e9 / rho) ** 0.5
        vm = (1 / 3 * (2 / vs**3 + 1 / vl**3)) ** (-1 / 3)
        props.update({"v_longitudinal_ms": vl, "v_transverse_ms": vs, "v_mean_ms": vm})
        if n_atoms and vol and vol > 0:
            # Debye temperature: ħ/k_B * v_m * (6π² n/V)^(1/3)
            props["T_Debye_K"] = (
                (1.05457e-34 / 1.3806e-23)
                * vm
                * (6 * np.pi**2 * (n_atoms / (vol * 1e-30))) ** (1 / 3)
            )

    return props
