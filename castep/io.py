"""
castep/io.py  —  Cell I/O, .param generation, and output parsing.
══════════════════════════════════════════════════════════════════
No subprocess calls, no user interaction.

All cell reduction is delegated to spglib, which guarantees the
smallest primitive cell and canonical symmetry — removing ~150 lines
of fragile hand-rolled Niggli/primitive-cell code.

Public API
──────────
  Cell helpers
    read_species(cell)              -> list[str]
    atom_count(cell)                -> int
    is_conventional_cell(cell)      -> bool
    write_vca_cell(dest, src, template_element, target_mix)
    inject_ncp(cell)
    write_primitive_cell(dest, r)

  Param helpers
    smart_defaults(species, is_vca) -> dict
    write_geomopt_param(path, ...)
    write_singlepoint_param(path, ...)
    patch_nextra(path, n)
    nextra_for_step(A, B, x)        -> int
    sp_param_content(param_path, x, vec) -> str

  Output parsing
    parse_output(path)              -> EngineResult
    parse_elastic_file(path)        -> dict
    read_stress(path)               -> np.ndarray | None
    count_atoms(path)               -> int

  CIF reduction (spglib-backed)
    CifResult                       NamedTuple
    reduce_cif(path)                -> CifResult
"""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import spglib

import config as _top_cfg
import core_physics as _phys
from castep import config as _cfg
from engine import EngineResult

# ─────────────────────────────────────────────────────────────────────────────
# Vegard radii table (Angstrom)
# ─────────────────────────────────────────────────────────────────────────────

_RADIUS: dict[str, float] = {
    "Li": 1.52, "Na": 1.86, "K":  2.27, "Mg": 1.60, "Ca": 1.97,
    "Sr": 2.15, "Ba": 2.22, "Sc": 1.62, "Y":  1.80, "Ti": 1.47,
    "Zr": 1.60, "Hf": 1.59, "V":  1.34, "Nb": 1.46, "Ta": 1.46,
    "Cr": 1.28, "Mo": 1.39, "W":  1.39, "Mn": 1.32, "Re": 1.37,
    "Fe": 1.26, "Ru": 1.34, "Os": 1.35, "Co": 1.25, "Rh": 1.34,
    "Ir": 1.36, "Ni": 1.24, "Pd": 1.37, "Pt": 1.39, "Cu": 1.28,
    "Ag": 1.44, "Au": 1.44, "Zn": 1.22, "Cd": 1.49, "Al": 1.43,
    "Ga": 1.22, "In": 1.63, "Sn": 1.41, "B":  0.87, "C":  0.77,
    "Si": 1.17, "Ge": 1.22, "N":  0.75, "P":  1.10, "As": 1.21,
    "Sb": 1.41, "La": 1.87, "Ce": 1.82, "Gd": 1.80, "Lu": 1.74,
}

# Periodic table: atomic number by symbol (subset sufficient for TMC/TMN).
_ATOMIC_NUMBER: dict[str, int] = {
    "H":  1,  "He": 2,  "Li": 3,  "Be": 4,  "B":  5,
    "C":  6,  "N":  7,  "O":  8,  "F":  9,  "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P":  15,
    "S":  16, "Cl": 17, "Ar": 18, "K":  19, "Ca": 20,
    "Sc": 21, "Ti": 22, "V":  23, "Cr": 24, "Mn": 25,
    "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30,
    "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35,
    "Y":  39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43,
    "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48,
    "In": 49, "Sn": 50, "Sb": 51, "La": 57, "Ce": 58,
    "Gd": 64, "Lu": 71, "Hf": 72, "Ta": 73, "W":  74,
    "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79,
    "Hg": 80, "Pb": 82, "Bi": 83, "U":  92,
}

# ─────────────────────────────────────────────────────────────────────────────
# Regex patterns (module-level — compiled once)
# ─────────────────────────────────────────────────────────────────────────────

_RE_COORD = re.compile(r"([-\d.Ee+]+)\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)")
_RE_LAT = re.compile(
    r"(%BLOCK\s+LATTICE_CART\s*\n)(.*?)(%ENDBLOCK\s+LATTICE_CART)",
    re.DOTALL | re.I,
)
_RE_LAT_BLOCK = re.compile(
    r"(%BLOCK LATTICE_CART.*?%ENDBLOCK LATTICE_CART)", re.DOTALL | re.I
)
_RE_POS_BLOCK = re.compile(
    r"%BLOCK POSITIONS_FRAC.*?%ENDBLOCK POSITIONS_FRAC",
    re.DOTALL | re.I,
)
_RE_STRESS_NEW = re.compile(
    r"^\s*([-+]?\d+\.\d+[Ee][+-]?\d+)\s+([-+]?\d+\.\d+[Ee][+-]?\d+)"
    r"\s+([-+]?\d+\.\d+[Ee][+-]?\d+)\s+<-- S",
    re.M,
)
_RE_STRESS_OLD = re.compile(
    r"\*\s+[xyz]\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s*\*"
)

# ─────────────────────────────────────────────────────────────────────────────
# Cell parsing helpers
# ─────────────────────────────────────────────────────────────────────────────


def read_species(cell_path: Path) -> list[str]:
    """Return unique element labels from POSITIONS_FRAC, in order of appearance.

    Args:
        cell_path: Path to a CASTEP ``.cell`` file.
    """
    seen: list[str] = []
    inside = False
    for line in cell_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if re.search(r"%BLOCK POSITIONS_FRAC", line, re.I):
            inside = True
            continue
        if re.search(r"%ENDBLOCK POSITIONS_FRAC", line, re.I):
            break
        if not inside or line.strip().startswith("#"):
            continue
        parts = line.split()
        if parts and re.match(r"^[A-Za-z]+$", parts[0]) and _RE_COORD.search(line):
            label = parts[0].capitalize()
            if label not in seen:
                seen.append(label)
    return seen


def atom_count(cell_path: Path) -> int:
    """Count atoms in the POSITIONS_FRAC block."""
    count, inside = 0, False
    for line in cell_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if re.search(r"%BLOCK POSITIONS_FRAC", line, re.I):
            inside = True
            continue
        if re.search(r"%ENDBLOCK POSITIONS_FRAC", line, re.I):
            break
        if inside and not line.strip().startswith("#") and _RE_COORD.search(line):
            count += 1
    return count


def _parse_positions(
    text: str,
) -> tuple[dict[str, list[str]], list[str]]:
    """Parse POSITIONS_FRAC into ``{label: [coord_str, ...]}``.

    Returns:
        ``(coords_by_label, ordered_label_list)``
    """
    coords: dict[str, list[str]] = {}
    order: list[str] = []
    inside = False
    for line in text.splitlines():
        if re.search(r"%BLOCK POSITIONS_FRAC", line, re.I):
            inside = True
            continue
        if re.search(r"%ENDBLOCK POSITIONS_FRAC", line, re.I):
            break
        if not inside or line.strip().startswith("#"):
            continue
        parts = line.split()
        if not parts:
            continue
        m = _RE_COORD.search(line)
        if not m:
            continue
        label = parts[0]
        key = label.lower()
        coord = f"{m.group(1)}   {m.group(2)}   {m.group(3)}"
        if key not in coords:
            coords[key] = []
            order.append(label)
        if coord not in coords[key]:
            coords[key].append(coord)
    return coords, order


def _parse_lattice(cell_path: Path) -> np.ndarray | None:
    """Parse LATTICE_CART into a 3x3 numpy array (rows = vectors, Angstrom).

    Returns ``None`` if the block is absent or malformed.
    """
    text = cell_path.read_text(encoding="utf-8", errors="replace")
    match = _RE_LAT.search(text)
    if not match:
        return None
    vecs: list[np.ndarray] = []
    for line in match.group(2).splitlines():
        parts = line.strip().split()
        if len(parts) >= 3:
            try:
                vecs.append(np.array([float(p) for p in parts[:3]]))
            except ValueError:
                pass
    return np.array(vecs) if len(vecs) == 3 else None


def is_conventional_cell(cell_path: Path) -> bool:
    """Heuristic: detect FCC/BCC conventional cells from half-lattice translations."""
    sc: dict[str, list[tuple[float, float, float]]] = {}
    inside = False
    for line in cell_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if re.search(r"%BLOCK POSITIONS_FRAC", line, re.I):
            inside = True
            continue
        if re.search(r"%ENDBLOCK POSITIONS_FRAC", line, re.I):
            break
        if not inside or line.strip().startswith("#"):
            continue
        m = _RE_COORD.search(line)
        if not m:
            continue
        label = line.split()[0].capitalize() if line.split() else ""
        coord = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
        sc.setdefault(label, []).append(coord)
    for coords in sc.values():
        if len(coords) < 2:
            continue
        for i, ca in enumerate(coords):
            for cb in coords[i + 1:]:
                diffs = [abs(ca[k] - cb[k]) for k in range(3)]
                if sum(
                    1 for d in diffs if abs(d - 0.5) < 0.01 or abs(d) < 0.01
                ) == 3:
                    return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# VCA cell writing
# ─────────────────────────────────────────────────────────────────────────────


def _scale_lattice(block: str, scale: float) -> str:
    """Scale all numeric vectors in a LATTICE_CART text block by *scale*."""
    lines = []
    for line in block.splitlines():
        s = line.strip().lower()
        if not s or s in {"ang", "bohr", "a.u.", "angstrom"} or s.startswith("%"):
            lines.append(line)
            continue
        m = _RE_COORD.search(line)
        if m:
            v = [float(m.group(i + 1)) * scale for i in range(3)]
            lines.append(
                _RE_COORD.sub(
                    f"{v[0]:20.15f}   {v[1]:20.15f}   {v[2]:20.15f}", line, 1
                )
            )
        else:
            lines.append(line)
    return "\n".join(lines)


def _vegard_scale_lattice(
    text: str,
    target_mix: dict[str, float],
    template_element: str,
) -> str:
    """Apply Vegard-law lattice scaling based on the mixed-element radii.

    Args:
        text:             Full .cell file text.
        target_mix:       ``{element: fraction}`` for the substituted sublattice.
        template_element: The original element being replaced.
    """
    r_template = _RADIUS.get(template_element.capitalize(), 0.0)
    r_mix = sum(
        _RADIUS.get(e.capitalize(), 0.0) * f for e, f in target_mix.items()
    )
    if r_template < 1e-6 or r_mix < 1e-6:
        return text
    scale = r_mix / r_template
    if abs(scale - 1.0) < 1e-6:
        return text
    return _RE_LAT_BLOCK.sub(lambda m: _scale_lattice(m.group(1), scale), text)


def write_vca_cell(
    dest: Path,
    src: Path,
    template_element: str,
    target_mix: dict[str, float],
    *,
    occ: float = 1.0,
    vegard: bool = True,
) -> None:
    """Write a CASTEP .cell file with MIXTURE syntax.

    The template element is identified by label in POSITIONS_FRAC.  Every
    site that originally belongs to *template_element* is replaced by the
    weighted mix in *target_mix*.  All other sublattices (e.g. C in TiC)
    remain untouched.

    This decouples the "template topology" from the "chemical identity":
    loading a TiC geometry and passing ``template_element="Ti"``
    ``target_mix={"Nb": 0.6, "V": 0.4}`` produces Nb0.6V0.4C
    without editing any source file.

    Args:
        dest:             Destination path.
        src:              Source primitive .cell template.
        template_element: Element label in *src* to be substituted.
        target_mix:       ``{element: fraction}`` for the new sublattice.
                          Fractions must sum to 1.
        occ:              Nonmetal site occupancy (<1 creates vacancies).
        vegard:           Apply Vegard-law lattice scaling when ``True``.
    """
    text = src.read_text(encoding="utf-8", errors="replace")
    sp_coords, sp_order = _parse_positions(text)
    tmpl_key = template_element.lower()

    # Normalise fractions to sum exactly to 1.
    total = sum(target_mix.values())
    if abs(total - 1.0) > 1e-6 and total > 1e-9:
        target_mix = {e: f / total for e, f in target_mix.items()}

    eps = 1e-9
    nonzero = {e: f for e, f in target_mix.items() if f > eps}

    def _site_lines(coord: str) -> list[str]:
        """Generate POSITIONS_FRAC line(s) for one VCA site coordinate."""
        if len(nonzero) == 1:
            elem = next(iter(nonzero))
            return [f"{elem}   {coord}"]
        return [
            f"{elem}   {coord}  MIXTURE:( 1 {frac:.8f})"
            for elem, frac in nonzero.items()
        ]

    rows = ["%BLOCK POSITIONS_FRAC"]
    for label in sp_order:
        key = label.lower()
        if key == tmpl_key:
            # Replace every site of this label with the VCA mix.
            for coord in sp_coords[key]:
                rows.extend(_site_lines(coord))
        else:
            # Preserve all other sublattices unchanged.
            for coord in sp_coords[key]:
                if occ < 1.0 - 1e-6 and label.capitalize() in _cfg.NONMETALS:
                    rows.append(f"{label}   {coord}  MIXTURE:( 1 {occ:.8f})")
                else:
                    rows.append(f"{label}   {coord}")
    rows.append("%ENDBLOCK POSITIONS_FRAC")

    new_text = _RE_POS_BLOCK.sub("\n".join(rows), text)
    if vegard:
        new_text = _vegard_scale_lattice(new_text, target_mix, template_element)

    mix_label = "  ".join(f"{e}:{f:.4f}" for e, f in target_mix.items())
    header = f"# VCAForge v{_top_cfg.VERSION}  mix=[{mix_label}]  tmpl={template_element}\n"
    dest.write_text(header + new_text, encoding="utf-8")


def inject_ncp(cell_path: Path) -> None:
    """Inject a SPECIES_POT NCP block (required for ElasticConstants/Phonon tasks)."""
    content = cell_path.read_text(encoding="utf-8", errors="replace")
    species = read_species(cell_path) or ["NCP"]
    sp_lines = "\n".join(f"{s}  NCP" for s in species)
    content = re.sub(
        r"%BLOCK\s+SPECIES_POT.*?%ENDBLOCK\s+SPECIES_POT\s*",
        "",
        content,
        flags=re.DOTALL | re.I,
    )
    cell_path.write_text(
        content.rstrip()
        + f"\n%BLOCK SPECIES_POT\n{sp_lines}\n%ENDBLOCK SPECIES_POT\n",
        encoding="utf-8",
    )


# ─────────────────────────────────────────────────────────────────────────────
# CIF reduction  (spglib)
# ─────────────────────────────────────────────────────────────────────────────


class CifResult(NamedTuple):
    """Primitive-cell data returned by :func:`reduce_cif`."""

    lattice:     np.ndarray   # 3x3, rows = vectors (Angstrom)
    frac:        np.ndarray   # (N_prim, 3) fractional coordinates
    species:     list[str]    # element symbol per atom
    n_original:  int          # atoms in the input (conventional) cell
    n_primitive: int          # atoms in the primitive cell


def _parse_cif_cell(text: str) -> dict[str, float]:
    """Extract unit-cell parameters from CIF text."""
    r: dict[str, float] = {}
    for k in (
        "_cell_length_a", "_cell_length_b", "_cell_length_c",
        "_cell_angle_alpha", "_cell_angle_beta", "_cell_angle_gamma",
    ):
        m = re.search(rf"^{re.escape(k)}\s+([\d.]+(?:\(\d+\))?)", text, re.M | re.I)
        if m:
            try:
                r[k] = float(re.sub(r"\(\d+\)$", "", m.group(1)))
            except ValueError:
                pass
    return r


def _parse_cif_atoms(text: str) -> tuple[list[str], list[np.ndarray]]:
    """Extract species and fractional coordinates from a CIF loop_ block."""
    sp, fc = [], []
    loop_pat = re.compile(
        r"loop_\s+((?:_atom_site_\S+\s+)+)((?:(?!loop_|_\S+\s+).*\n?)+)", re.M
    )
    for lm in loop_pat.finditer(text):
        cols = re.findall(r"(_atom_site_\S+)", lm.group(1))
        if "_atom_site_fract_x" not in cols:
            continue
        t_idx = next(
            (
                i for i, c in enumerate(cols)
                if c in ("_atom_site_type_symbol", "_atom_site_label")
            ),
            0,
        )
        xi = cols.index("_atom_site_fract_x")
        yi = cols.index("_atom_site_fract_y")
        zi = cols.index("_atom_site_fract_z")
        nc = len(cols)
        for line in lm.group(2).splitlines():
            parts = line.split()
            if len(parts) < nc:
                continue
            try:
                x_ = float(re.sub(r"\(\d+\)$", "", parts[xi]))
                y_ = float(re.sub(r"\(\d+\)$", "", parts[yi]))
                z_ = float(re.sub(r"\(\d+\)$", "", parts[zi]))
            except (ValueError, IndexError):
                continue
            e = re.match(r"([A-Za-z]+)", parts[t_idx])
            if e:
                sp.append(e.group(1).capitalize())
                fc.append(np.array([x_, y_, z_]))
    return sp, fc


def reduce_cif(cif_path: Path) -> CifResult:
    """Read a CIF file and return the primitive cell via spglib.

    Uses ``spglib.standardize_cell(to_primitive=True, symprec=1e-3)`` which
    guarantees the minimal primitive cell with correct symmetry — replacing
    the former hand-rolled Niggli + find_primitive_cell implementation.

    Args:
        cif_path: Path to the CIF file.

    Returns:
        :class:`CifResult` containing the primitive lattice, fractional
        coordinates, species list, and both atom counts.

    Raises:
        ValueError: If required cell parameters or atom sites are missing,
                    or if spglib fails to standardise the cell.
    """
    text = cif_path.read_text(encoding="utf-8", errors="replace")
    params = _parse_cif_cell(text)
    missing = [
        k for k in (
            "_cell_length_a", "_cell_length_b", "_cell_length_c",
            "_cell_angle_alpha", "_cell_angle_beta", "_cell_angle_gamma",
        )
        if k not in params
    ]
    if missing:
        raise ValueError(f"CIF missing parameters: {', '.join(missing)}")

    L = _phys.abc_to_lattice(
        params["_cell_length_a"], params["_cell_length_b"], params["_cell_length_c"],
        params["_cell_angle_alpha"], params["_cell_angle_beta"], params["_cell_angle_gamma"],
    )
    sp, fc_list = _parse_cif_atoms(text)
    if not sp:
        raise ValueError(f"No atom sites found in {cif_path.name}")

    n_orig = len(sp)
    frac   = np.array(fc_list, dtype=float)

    # Build spglib cell tuple: (lattice, scaled_positions, atomic_numbers).
    atomic_nums = [_ATOMIC_NUMBER.get(s, 0) for s in sp]
    spg_cell    = (L, frac, atomic_nums)

    result = spglib.standardize_cell(spg_cell, to_primitive=True, symprec=1e-3)
    if result is None:
        raise ValueError(
            f"spglib could not standardise {cif_path.name} "
            "(check for overlapping atoms or unphysical geometry)"
        )

    prim_L, prim_frac, prim_nums = result

    # Reconstruct element symbols from atomic numbers.
    _NUM_TO_SYM = {v: k for k, v in _ATOMIC_NUMBER.items()}
    prim_sp = [_NUM_TO_SYM.get(int(n), f"Z{n}") for n in prim_nums]

    return CifResult(prim_L, prim_frac, prim_sp, n_orig, len(prim_sp))


def write_primitive_cell(dest: Path, r: CifResult) -> None:
    """Write a CASTEP-format .cell file from a :class:`CifResult`.

    Args:
        dest: Destination path for the ``.cell`` file.
        r:    Primitive-cell result returned by :func:`reduce_cif`.
    """
    lines: list[str] = [
        f"# VCAForge v{_top_cfg.VERSION}  primitive cell\n",
        "%BLOCK LATTICE_CART\nANG\n",
    ]
    for vec in r.lattice:
        lines.append(f"  {vec[0]:20.15f}  {vec[1]:20.15f}  {vec[2]:20.15f}\n")
    lines.append("%ENDBLOCK LATTICE_CART\n\n%BLOCK POSITIONS_FRAC\n")
    for sym, fc in zip(r.species, r.frac):
        lines.append(f"  {sym}   {fc[0]:.10f}   {fc[1]:.10f}   {fc[2]:.10f}\n")
    lines.append("%ENDBLOCK POSITIONS_FRAC\n")
    dest.write_text("".join(lines), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# .param helpers
# ─────────────────────────────────────────────────────────────────────────────


def smart_defaults(species: list[str], is_vca: bool) -> dict[str, Any]:
    """Return recommended DFT defaults for the given species list.

    Args:
        species:  List of element symbols present in the cell.
        is_vca:   ``True`` for a VCA sweep, ``False`` for a single compound.
    """
    has_hard = any(s.capitalize() in _cfg.HARD_ELEMENTS for s in species)
    has_mag  = any(s.capitalize() in _cfg.MAGNETIC_ELEMENTS for s in species)
    has_d    = any(s.capitalize() in _cfg.D_BLOCK for s in species)
    return {
        "cut_off_energy": _cfg.CUTOFF_HARD if has_hard else _cfg.CUTOFF_SOFT,
        "spin_polarized": has_mag,
        "nextra_bands": (20 if has_d else 15) if is_vca else (10 if has_d else 4),
        "hard_detected": sorted(
            s for s in species if s.capitalize() in _cfg.HARD_ELEMENTS
        ),
        "magnetic_detected": sorted(
            s for s in species if s.capitalize() in _cfg.MAGNETIC_ELEMENTS
        ),
    }


def _scf_block(
    xc: str,
    cutoff: int,
    spin: bool,
    nextra: int,
    smearing: float,
    mix_amp: float,
    *,
    ncp: bool = False,
) -> str:
    """Return the shared SCF settings block for any .param file.

    The caller supplies exactly one *mix_amp* value, which prevents the
    ``"mix_charge_amp multiply defined"`` CASTEP fatal error that arose
    when both the SCF block and write_singlepoint_param wrote the key.

    Args:
        xc:       XC functional string (e.g. ``"PBE"``).
        cutoff:   Plane-wave cut-off energy in eV.
        spin:     Whether spin polarisation is enabled.
        nextra:   Number of extra empty bands.
        smearing: Fermi smearing width in eV.
        mix_amp:  Charge-mixing amplitude (task-specific, written exactly once).
        ncp:      Append a NCP cut-off note when ``True``.
    """
    ncp_note = "  # NCP: raise cutoff >= 900 eV for C/N/O" if ncp else ""
    return (
        f"# VCAForge v{_top_cfg.VERSION}\n"
        f"xc_functional       : {xc}\n"
        f"cut_off_energy      : {cutoff} eV{ncp_note}\n"
        f"spin_polarized      : {'true' if spin else 'false'}\n\n"
        f"max_scf_cycles      : {_cfg.MAX_SCF}\n"
        f"metals_method       : {_cfg.METALS_METHOD}\n"
        f"mixing_scheme       : {_cfg.MIXING_SCHEME}\n"
        f"smearing_width      : {smearing:.2f} eV\n"
        f"mix_charge_amp      : {mix_amp}\n"
        f"nextra_bands        : {nextra}\n"
    )


def write_geomopt_param(
    path: Path,
    xc: str,
    cutoff: int,
    spin: bool,
    nextra: int,
    smearing: float = _top_cfg.SMEARING_SINGLE,
    *,
    ncp: bool = False,
) -> None:
    """Write a GeometryOptimization .param file.

    Args:
        path:     Destination path.
        xc:       XC functional string.
        cutoff:   Plane-wave cut-off energy in eV.
        spin:     Enable spin polarisation.
        nextra:   Extra empty bands.
        smearing: Fermi smearing width in eV.
        ncp:      Append NCP note when ``True``.
    """
    body = (
        _scf_block(xc, cutoff, spin, nextra, smearing, _cfg.MIX_AMP_GEOM, ncp=ncp)
        + f"elec_energy_tol     : {_cfg.ELEC_TOL_GEOM}\n\n"
        f"task                : GeometryOptimization\n"
        f"calculate_stress    : true\n"
        f"geom_method         : LBFGS\n"
        f"geom_max_iter       : {_cfg.GEOM_MAX_ITER}\n"
        f"geom_energy_tol     : {_cfg.GEOM_E_TOL}\n"
        f"geom_force_tol      : {_cfg.GEOM_F_TOL}\n"
        f"geom_stress_tol     : {_cfg.GEOM_S_TOL}\n"
        f"geom_disp_tol       : {_cfg.GEOM_D_TOL}\n\n"
        f"opt_strategy        : speed\n"
        f"write_checkpoint    : none\n"
        f"num_dump_cycles     : 0\n"
        f"write_cell_structure: true\n"
    )
    path.write_text(body, encoding="utf-8")


def write_singlepoint_param(
    path: Path,
    xc: str,
    cutoff: int,
    spin: bool,
    nextra: int,
    *,
    ncp: bool = False,
) -> None:
    """Write a SinglePoint (elastic strain) .param file.

    Uses MIX_AMP_SP and a tighter SCF tolerance than the GeomOpt param to
    ensure reliable stress tensors from strained cells.

    Args:
        path:   Destination path.
        xc:     XC functional string.
        cutoff: Plane-wave cut-off energy in eV.
        spin:   Enable spin polarisation.
        nextra: Extra empty bands.
        ncp:    Append NCP note when ``True``.
    """
    body = (
        _scf_block(
            xc, cutoff, spin, nextra,
            smearing=_top_cfg.SMEARING_SINGLE,
            mix_amp=_cfg.MIX_AMP_SP,
            ncp=ncp,
        )
        + f"elec_energy_tol     : {_cfg.ELEC_TOL_SP}\n"
        f"finite_basis_corr   : {_cfg.FINITE_BASIS}\n\n"
        f"task                : SinglePoint\n"
        f"calculate_stress    : true\n\n"
        f"opt_strategy        : speed\n"
        f"write_checkpoint    : none\n"
        f"num_dump_cycles     : 0\n"
        f"write_cell_structure: false\n"
    )
    path.write_text(body, encoding="utf-8")


def patch_nextra(param_path: Path, nextra: int) -> None:
    """Replace the ``nextra_bands`` line in an existing .param file in-place."""
    lines, replaced = [], False
    for line in param_path.read_text(encoding="utf-8").splitlines():
        if line.strip().lower().startswith("nextra_bands") and ":" in line:
            lines.append(f"nextra_bands        : {nextra}")
            replaced = True
        else:
            lines.append(line)
    if not replaced:
        lines.append(f"nextra_bands        : {nextra}")
    param_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def nextra_for_step(species_a: str, species_b: str, x: float) -> int:
    """Return adaptive ``nextra_bands`` for one VCA GeomOpt step.

    Linearly interpolates between the d-block base counts of the two
    endpoint species, then adds 10 extra bands for VCA intermediates.

    Args:
        species_a: Element at x = 0.
        species_b: Element at x = 1.
        x:         VCA concentration.
    """
    def _base(e: str) -> int:
        return 10 if e.capitalize() in _cfg.D_BLOCK else 4

    base = round(_base(species_a) * (1 - x) + _base(species_b) * x)
    return base + 10 if 0.01 < x < 0.99 else base


def sp_param_content(param_path: Path, x: float, vec: float) -> str:
    """Build SinglePoint .param text for a strained cell.

    Reads ``xc_functional`` and ``cut_off_energy`` from an existing GeomOpt
    .param so that the strained calculation inherits the same DFT settings.

    Args:
        param_path: Path to the GeomOpt .param file (may be absent).
        x:          VCA concentration — determines ``nextra_bands``.
        vec:        Valence electron count — determines ``nextra_bands``.

    Returns:
        Full text of a SinglePoint .param file.
    """
    xc, cutoff = _cfg.XC_DEFAULT, _cfg.CUTOFF_HARD
    if param_path.exists():
        for line in param_path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            kv = line.split(":", 1)
            if len(kv) != 2:
                continue
            k = kv[0].strip().lower()
            v = kv[1].strip().split()[0] if kv[1].strip() else ""
            if k == "xc_functional":
                xc = v
            elif k == "cut_off_energy":
                try:
                    cutoff = int(float(v))
                except ValueError:
                    pass

    with tempfile.NamedTemporaryFile(
        suffix=".param", delete=False, mode="w", encoding="utf-8"
    ) as tmp:
        tp = Path(tmp.name)
    write_singlepoint_param(
        tp, xc, cutoff, spin=False, nextra=_phys.nextra_bands_for(x, vec)
    )
    content = tp.read_text(encoding="utf-8")
    tp.unlink(missing_ok=True)
    return content


# ─────────────────────────────────────────────────────────────────────────────
# Output parsing
# ─────────────────────────────────────────────────────────────────────────────


def _try_float(s: str) -> float | None:
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def parse_output(castep_path: Path) -> EngineResult:
    """Parse a CASTEP output file.  Single reverse pass — last value wins.

    Handles both GeometryOptimization and SinglePoint output formats.

    Args:
        castep_path: Path to the ``.castep`` output file.

    Returns:
        :class:`EngineResult` populated from the output file.
    """
    r = EngineResult()
    if not castep_path.exists():
        r.warnings.append("output file not found")
        return r
    try:
        lines = castep_path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
    except OSError as exc:
        r.warnings.append(str(exc))
        return r

    la, lb, lc = None, None, None
    no_empty = False

    for line in reversed(lines):
        s = line.strip()

        if not r.task_type and "type of calculation" in s.lower():
            parts = s.split(":")
            if len(parts) >= 2:
                r.task_type = parts[-1].strip()

        if (
            not r.geom_converged
            and "Geometry optimization completed successfully" in line
        ):
            r.geom_converged = True

        if r.enthalpy_eV is None and "Final Enthalpy" in line and "Pseudo" not in line:
            p = line.split("=")
            if len(p) >= 2:
                r.enthalpy_eV = _try_float(p[-1].strip().split()[0])

        if r.enthalpy_eV is None and "Final energy, E" in line and "Pseudo" not in line:
            p = line.split("=")
            if len(p) >= 2:
                r.enthalpy_eV = _try_float(p[-1].strip().split()[0])

        if la is None and s.startswith("a ="):
            la = _try_float(s.split("=")[1].strip().split()[0])
        if lb is None and s.startswith("b ="):
            lb = _try_float(s.split("=")[1].strip().split()[0])
        if lc is None and s.startswith("c ="):
            lc = _try_float(s.split("=")[1].strip().split()[0])
        if la is not None and lb is not None and lc is not None and r.a_opt_ang is None:
            r.a_opt_ang = la
            r.b_opt_ang = lb
            r.c_opt_ang = lc

        if r.volume_ang3 is None and "Current cell volume" in line and "=" in line:
            p = line.split("=")
            if len(p) >= 2:
                r.volume_ang3 = _try_float(p[-1].strip().split()[0])

        if (
            r.density_gcm3 is None
            and "g/cm" in line
            and "=" in line
            and "AMU" not in line
        ):
            p = line.split("=")
            if len(p) >= 2:
                v = _try_float(p[-1].strip().split()[0])
                if v and v > 0:
                    r.density_gcm3 = v

        if r.bulk_modulus_GPa is None and "Final bulk modulus" in line and "=" in line:
            p = line.split("=")
            if len(p) >= 2:
                r.bulk_modulus_GPa = _try_float(p[-1].strip().split()[0])

        if r.wall_time_s is None and "Total time" in line and "=" in line:
            p = line.split("=")
            if len(p) >= 2:
                r.wall_time_s = _try_float(p[-1].strip().split()[0])

        if not no_empty and "no empty bands" in s.lower():
            r.warnings.append("no empty bands — increase nextra_bands in .param")
            no_empty = True

        if s.lower().startswith("warning") and len(r.warnings) < 5:
            p = s.split(":", 1)
            if len(p) == 2:
                msg = p[1].strip()
                if msg and msg not in r.warnings:
                    r.warnings.append(msg)

    return r


def read_stress(castep_path: Path) -> np.ndarray | None:
    """Parse the final symmetrised stress tensor from a .castep file.

    Tries both the newer ``<-- S`` format and the older ``* x * y * z *``
    format.

    Returns:
        Voigt 6-vector ``[s11, s22, s33, s23, s13, s12]`` in GPa,
        or ``None`` if the stress block is absent or unreadable.
    """
    if not castep_path.exists():
        return None
    try:
        text = castep_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if not text.strip():
        return None
    for pat in (_RE_STRESS_NEW, _RE_STRESS_OLD):
        rows = pat.findall(text)
        if len(rows) >= 3:
            try:
                m = np.array([[float(v) for v in r] for r in rows[-3:]])
                return np.array(
                    [m[0, 0], m[1, 1], m[2, 2], m[1, 2], m[0, 2], m[0, 1]]
                )
            except (ValueError, IndexError):
                pass
    return None


def count_atoms(castep_path: Path) -> int:
    """Count atoms per unit cell from a .castep output file."""
    if not castep_path.exists():
        return 0
    for line in castep_path.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        if "Total number of ions in cell" in line and "=" in line:
            try:
                return int(line.split("=")[-1].strip())
            except ValueError:
                pass
    return 0


def parse_elastic_file(path: Path) -> dict[str, Any]:
    """Parse a CASTEP .elastic file (ElasticConstants task output).

    Uses a simple line-reader: look for the ``Elastic Stiffness Constants``
    header, then collect the next six non-empty, non-separator data rows.
    This is ~10x faster than the previous regex approach and cannot break
    on minor output-format changes.

    Args:
        path: Path to the ``.elastic`` file.

    Returns:
        Dict with keys such as ``C11``, ``B_Hill_GPa``, etc.  Empty dict
        if the file does not exist or contains no parsable data.
    """
    if not path.exists():
        return {}
    r: dict[str, Any] = {}
    in_cij = False
    cij_rows: list[list[float]] = []

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()

        # --- Stiffness matrix block ---
        if "Elastic Stiffness Constants" in line:
            in_cij = True
            cij_rows = []
            continue

        if in_cij:
            if not s or s.startswith("=") or s.startswith("-"):
                # Skip separators; stop once we have all 6 rows.
                if len(cij_rows) == 6:
                    in_cij = False
                continue
            parts = s.split()
            if len(parts) >= 6:
                try:
                    cij_rows.append([float(v) for v in parts[:6]])
                except ValueError:
                    pass
            if len(cij_rows) == 6:
                in_cij = False
                for key, i, j in [
                    ("C11", 0, 0), ("C12", 0, 1), ("C13", 0, 2),
                    ("C22", 1, 1), ("C23", 1, 2), ("C33", 2, 2),
                    ("C44", 3, 3), ("C55", 4, 4), ("C66", 5, 5),
                ]:
                    r[key] = f"{cij_rows[i][j]:.4f}"

        # --- Scalar moduli ---
        for label, col in [
            ("Voigt bulk modulus",  "B_Voigt_GPa"),
            ("Reuss bulk modulus",  "B_Reuss_GPa"),
            ("Hill bulk modulus",   "B_Hill_GPa"),
            ("Voigt shear modulus", "G_Voigt_GPa"),
            ("Reuss shear modulus", "G_Reuss_GPa"),
            ("Hill shear modulus",  "G_Hill_GPa"),
            ("Young modulus",       "E_GPa"),
            ("Poisson ratio",       "nu"),
            ("Debye temperature",   "T_Debye_K"),
            ("Vickers hardness",    "H_Vickers_GPa"),
        ]:
            if label in line and "=" in line:
                p = line.split("=")
                if len(p) >= 2:
                    v = _try_float(p[-1].strip().split()[0])
                    if v is not None:
                        r[col] = f"{v:.4f}"

    return r
