"""
castep_io.py — File I/O and generation for CASTEP VCA calculations.
────────────────────────────────────────────────────────────────────
Responsibilities:
  • Read / parse .cell files (species, positions, lattice)
  • Validate VCA element pairs (metal/nonmetal check)
  • Detect conventional vs primitive cells (FCC/BCC heuristic)
  • Write VCA .cell files with MIXTURE syntax + Vegard scaling
  • Generate smart .param files (adaptive nextra_bands, cutoff, spin)
  • Parse .castep output files (reverse-read, no fragile regex)

No subprocess calls, no user prompts, no state management here.
All functions are pure or pure-I/O (read/write files only).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

VERSION = "5.1"

# ─────────────────────────────────────────────────────────────────────────────
# Engine configuration
# ─────────────────────────────────────────────────────────────────────────────
# Each DFT engine has its own EngineConfig instance.
# ui.py and workflow.py import the active engine's config — they never
# hard-code binary paths or command templates.
#
# To add a new engine (e.g. Quantum ESPRESSO):
#   1. Create an EngineConfig instance below (QE_ENGINE).
#   2. Write qe_io.py with the same public API:
#        write_input(), write_params(), parse_log() → dict[str, Any]
#   3. In main.py select the engine via --engine flag and pass its config to ui.py.
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class EngineConfig:
    """
    All engine-specific constants in one place.

    Attributes:
        name:           Human-readable engine label shown in the UI.
        default_bin:    Default binary path (supports ~ expansion).
                        Used only when the user has not provided --castep-cmd.
        cmd_template:   Full MPI command template.
                        Placeholders: {bin} = binary path, {ncores} = MPI count,
                        {seed} = job name (substituted at runtime by workflow.py).
        input_suffix:   Input file extension this engine reads (.cell, .in, ...).
        output_suffix:  Primary output file this engine writes (.castep, .out, ...).
    """

    name: str
    default_bin: str
    cmd_template: str
    input_suffix: str
    output_suffix: str


# ── CASTEP (default) ──────────────────────────────────────────────────────────
# default_bin is resolved at runtime by find_castep_binary() in ui.py.
# We store only a sentinel here; the wizard always validates the actual path.
CASTEP_ENGINE = EngineConfig(
    name="CASTEP",
    default_bin="castep.mpi",  # sentinel — wizard probes PATH + common dirs
    cmd_template="mpirun -n {ncores} {bin} {seed}",
    input_suffix=".cell",
    output_suffix=".castep",
)

# ── Example: Quantum ESPRESSO stub (uncomment + implement qe_io.py to use) ───
# QE_ENGINE = EngineConfig(
#     name="Quantum ESPRESSO",
#     default_bin="~/qe-7.3/bin/pw.x",
#     cmd_template="mpirun -n {ncores} {bin} -in {seed}.in",
#     input_suffix=".in",
#     output_suffix=".out",
# )

# ── Active engine — change this line to switch engines project-wide ───────────
ACTIVE_ENGINE: EngineConfig = CASTEP_ENGINE


# ─────────────────────────────────────────────────────────────────────────────
# Element classification
# ─────────────────────────────────────────────────────────────────────────────

# All s-, d-, f-block metals (VCA-compatible within this set)
_METALS: frozenset[str] = frozenset(
    {
        # s-block
        "Li",
        "Na",
        "K",
        "Rb",
        "Cs",
        "Be",
        "Mg",
        "Ca",
        "Sr",
        "Ba",
        "Ra",
        # d-block (transition metals)
        "Sc",
        "Ti",
        "V",
        "Cr",
        "Mn",
        "Fe",
        "Co",
        "Ni",
        "Cu",
        "Zn",
        "Y",
        "Zr",
        "Nb",
        "Mo",
        "Tc",
        "Ru",
        "Rh",
        "Pd",
        "Ag",
        "Cd",
        "Hf",
        "Ta",
        "W",
        "Re",
        "Os",
        "Ir",
        "Pt",
        "Au",
        "Hg",
        # p-block metals
        "Al",
        "Ga",
        "In",
        "Sn",
        "Tl",
        "Pb",
        "Bi",
        # f-block
        "La",
        "Ce",
        "Pr",
        "Nd",
        "Pm",
        "Sm",
        "Eu",
        "Gd",
        "Tb",
        "Dy",
        "Ho",
        "Er",
        "Tm",
        "Yb",
        "Lu",
        "Th",
        "U",
        "Pu",
    }
)

_NONMETALS: frozenset[str] = frozenset(
    {
        "B",
        "C",
        "N",
        "O",
        "F",
        "Si",
        "P",
        "S",
        "Cl",
        "As",
        "Se",
        "Br",
        "Te",
        "I",
        "At",
        "H",
    }
)

# Magnetic elements → spin_polarized : true
_MAGNETIC: frozenset[str] = frozenset({"Fe", "Co", "Ni", "Mn", "Cr", "Gd", "V"})

# Hard elements → minimum 700 eV cutoff
_HARD: frozenset[str] = frozenset({"C", "N", "O", "F", "B", "H"})

# d-block elements → need nextra_bands : 20 for VCA
_D_BLOCK: frozenset[str] = frozenset(
    {
        "Sc",
        "Ti",
        "V",
        "Cr",
        "Mn",
        "Fe",
        "Co",
        "Ni",
        "Cu",
        "Zn",
        "Y",
        "Zr",
        "Nb",
        "Mo",
        "Tc",
        "Ru",
        "Rh",
        "Pd",
        "Ag",
        "Cd",
        "Hf",
        "Ta",
        "W",
        "Re",
        "Os",
        "Ir",
        "Pt",
        "Au",
        "Hg",
    }
)

# Valence electron count per element (s+p+d, standard oxidation state in carbides/nitrides)
# Sources: periodic table group number / CASTEP VEC convention for TMC/TMN
VALENCE: dict[str, int] = {
    # s-block
    "Li": 1,
    "Na": 1,
    "K": 1,
    "Be": 2,
    "Mg": 2,
    "Ca": 2,
    "Sr": 2,
    "Ba": 2,
    # d-block (transition metals — group number = VEC contribution)
    "Sc": 3,
    "Y": 3,
    "La": 3,
    "Ti": 4,
    "Zr": 4,
    "Hf": 4,
    "V": 5,
    "Nb": 5,
    "Ta": 5,
    "Cr": 6,
    "Mo": 6,
    "W": 6,
    "Mn": 7,
    "Re": 7,
    "Fe": 8,
    "Ru": 8,
    "Os": 8,
    "Co": 9,
    "Rh": 9,
    "Ir": 9,
    "Ni": 10,
    "Pd": 10,
    "Pt": 10,
    "Cu": 11,
    "Ag": 11,
    "Au": 11,
    "Zn": 12,
    "Cd": 12,
    # p-block nonmetals (as in transition metal carbides/nitrides context)
    "B": 3,
    "Al": 3,
    "Ga": 3,
    "In": 3,
    "C": 4,
    "Si": 4,
    "Ge": 4,
    "Sn": 4,
    "N": 5,
    "P": 5,
    "As": 5,
    "O": 6,
    "S": 6,
}


def vec_for_system(
    species_a: str,
    species_b: str,
    x: float,
    nonmetal: str,
) -> float:
    """
    Compute the Valence Electron Concentration (VEC) for a VCA carbide/nitride.

    VEC = (1-x)*val(A) + x*val(B) + val(nonmetal)
    per formula unit (one metal + one nonmetal).

    Used to predict elastic (in)stability:
      VEC < 8.3  → elastically stable cubic (positive C44)
      VEC > 8.4  → risk of C44 → 0 or negative (Born stability violated)
      VEC ≈ 8.0  → optimal for hardness and stability (TiC = 8.0)

    Reference: Music et al., Appl. Phys. Lett. 86 (2005); Zhang et al. (2010).
    """
    val_a = VALENCE.get(species_a.capitalize(), 0)
    val_b = VALENCE.get(species_b.capitalize(), 0)
    val_nm = VALENCE.get(nonmetal.capitalize(), 0)
    return (1.0 - x) * val_a + x * val_b + val_nm


def vacancy_fraction_for_vec(
    species_a: str,
    species_b: str,
    x: float,
    nonmetal: str,
    target_vec: float = 8.0,
) -> float:
    """
    Compute the nonmetal site occupancy needed to reach target_vec.

    For Ti(1-x)Nb(x)C with VEC > target:
      VEC(c) = [(1-x)*val(A) + x*val(B)] + c * val(nonmetal)
      c = (target_vec - metal_vec) / val(nonmetal)

    Returns a float in [0, 1]:
      1.0 → full occupancy (no vacancies needed)
      <1  → partial occupancy (vacancy concentration = 1 - c)
      0.0 → degenerate case (clamped)
    """
    val_a = VALENCE.get(species_a.capitalize(), 0)
    val_b = VALENCE.get(species_b.capitalize(), 0)
    val_nm = VALENCE.get(nonmetal.capitalize(), 1)  # avoid div-by-zero
    metal_vec = (1.0 - x) * val_a + x * val_b
    c = (target_vec - metal_vec) / val_nm
    return max(0.0, min(1.0, c))


# Covalent/metallic radii in Å for Vegard law scaling
_RADIUS: dict[str, float] = {
    "Li": 1.52,
    "Na": 1.86,
    "K": 2.27,
    "Mg": 1.60,
    "Ca": 1.97,
    "Sr": 2.15,
    "Ba": 2.22,
    "Sc": 1.62,
    "Y": 1.80,
    "Ti": 1.47,
    "Zr": 1.60,
    "Hf": 1.59,
    "V": 1.34,
    "Nb": 1.46,
    "Ta": 1.46,
    "Cr": 1.28,
    "Mo": 1.39,
    "W": 1.39,
    "Mn": 1.32,
    "Re": 1.37,
    "Fe": 1.26,
    "Ru": 1.34,
    "Os": 1.35,
    "Co": 1.25,
    "Rh": 1.34,
    "Ir": 1.36,
    "Ni": 1.24,
    "Pd": 1.37,
    "Pt": 1.39,
    "Cu": 1.28,
    "Ag": 1.44,
    "Au": 1.44,
    "Zn": 1.22,
    "Cd": 1.49,
    "Al": 1.43,
    "Ga": 1.22,
    "In": 1.63,
    "Sn": 1.41,
    "B": 0.87,
    "C": 0.77,
    "Si": 1.17,
    "Ge": 1.22,
    "N": 0.75,
    "P": 1.10,
    "As": 1.21,
    "Sb": 1.41,
    "La": 1.87,
    "Ce": 1.82,
    "Gd": 1.80,
    "Lu": 1.74,
}


# ─────────────────────────────────────────────────────────────────────────────
# VCA pair validation
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class VCAResult:
    """Result of validate_vca_pair()."""

    ok: bool
    error: bool  # True = hard error, False = warning or ok
    message: str


def validate_vca_pair(species_a: str, species_b: str) -> VCAResult:
    """
    Check whether mixing species_a and species_b in VCA is physically valid.

    Rule: metals and nonmetals sit on different crystallographic sublattices.
    Mixing them creates a virtual atom with nonsensical core charge → ERROR.
    Same-sublattice mixing (metal+metal or nonmetal+nonmetal) → OK.
    """
    elem_a = species_a.capitalize()
    elem_b = species_b.capitalize()

    a_is_metal = elem_a in _METALS
    b_is_metal = elem_b in _METALS
    a_is_nonmetal = elem_a in _NONMETALS
    b_is_nonmetal = elem_b in _NONMETALS

    if (a_is_metal and b_is_nonmetal) or (a_is_nonmetal and b_is_metal):
        metal = elem_a if a_is_metal else elem_b
        nonmetal = elem_b if a_is_metal else elem_a
        return VCAResult(
            ok=False,
            error=True,
            message=(
                f"Cannot mix metal ({metal}) with non-metal ({nonmetal}) in VCA. "
                "They occupy different crystallographic sublattices. "
                "Choose elements from the same sublattice "
                f"(e.g. Ti→Zr for the metal site in TiC, not Ti→C)."
            ),
        )

    a_magnetic = elem_a in _MAGNETIC
    b_magnetic = elem_b in _MAGNETIC
    if a_magnetic or b_magnetic:
        mag = elem_a if a_magnetic else elem_b
        return VCAResult(
            ok=True,
            error=False,
            message=(
                f"{mag} is magnetic — VCA averages spin potentials. "
                "spin_polarized : true will be set in .param. "
                "Check SCF convergence carefully."
            ),
        )

    return VCAResult(ok=True, error=False, message="")


def is_single_compound_mode(species_list: list[str]) -> bool:
    """
    Return True when the cell contains a metal+nonmetal mix (e.g. TiC, TiN).
    In this case no VCA sweep makes sense — offer Single Compound mode instead.
    """
    has_metal = any(s.capitalize() in _METALS for s in species_list)
    has_nonmetal = any(s.capitalize() in _NONMETALS for s in species_list)
    return has_metal and has_nonmetal


# ─────────────────────────────────────────────────────────────────────────────
# .cell file reading
# ─────────────────────────────────────────────────────────────────────────────

# Matches three floats on a coordinate line (used for POSITIONS_FRAC parsing)
_RE_COORD = re.compile(r"([-\d.Ee+]+)\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)")


def read_species(cell_path: Path) -> list[str]:
    """Return unique element labels from POSITIONS_FRAC, in order of appearance."""
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
    """Count total atoms in POSITIONS_FRAC block."""
    count = 0
    inside = False
    for line in cell_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if re.search(r"%BLOCK POSITIONS_FRAC", line, re.I):
            inside = True
            continue
        if re.search(r"%ENDBLOCK POSITIONS_FRAC", line, re.I):
            break
        if inside and not line.strip().startswith("#") and _RE_COORD.search(line):
            count += 1
    return count


def is_conventional_cell(cell_path: Path) -> bool:
    """
    Heuristic: detect conventional (non-primitive) FCC/BCC cells.

    If the same element appears at positions that differ by exactly 0.5
    along one or more axes (e.g. 0,0,0 and 0.5,0.5,0), it is almost certainly
    a conventional cubic cell — 4× or 2× larger than needed.
    """
    species_coords: dict[str, list[tuple[float, float, float]]] = {}
    inside = False

    for line in cell_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if re.search(r"%BLOCK POSITIONS_FRAC", line, re.I):
            inside = True
            continue
        if re.search(r"%ENDBLOCK POSITIONS_FRAC", line, re.I):
            break
        if not inside or line.strip().startswith("#"):
            continue
        match = _RE_COORD.search(line)
        if not match:
            continue
        parts = line.split()
        if not parts:
            continue
        label = parts[0].capitalize()
        coord = (float(match.group(1)), float(match.group(2)), float(match.group(3)))
        species_coords.setdefault(label, []).append(coord)

    for coords in species_coords.values():
        if len(coords) < 2:
            continue
        for i, coord_a in enumerate(coords):
            for coord_b in coords[i + 1 :]:
                diffs = [abs(coord_a[k] - coord_b[k]) for k in range(3)]
                half_count = sum(
                    1 for d in diffs if abs(d - 0.5) < 0.01 or abs(d) < 0.01
                )
                if half_count == 3:
                    return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# POSITIONS_FRAC parser (internal)
# ─────────────────────────────────────────────────────────────────────────────


def _parse_positions(
    text: str,
) -> tuple[dict[str, list[str]], list[str]]:
    """
    Parse POSITIONS_FRAC block into a structured form.

    Returns:
        species_coords: mapping from lowercased element label to list of
                        coordinate strings (deduplicated)
        species_order:  original-case labels in first-appearance order
    """
    species_coords: dict[str, list[str]] = {}
    species_order: list[str] = []
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
        match = _RE_COORD.search(line)
        if not match:
            continue
        label = parts[0]
        label_lower = label.lower()
        coord = f"{match.group(1)}   {match.group(2)}   {match.group(3)}"
        if label_lower not in species_coords:
            species_coords[label_lower] = []
            species_order.append(label)
        if coord not in species_coords[label_lower]:
            species_coords[label_lower].append(coord)

    return species_coords, species_order


# ─────────────────────────────────────────────────────────────────────────────
# Vegard law lattice scaling
# ─────────────────────────────────────────────────────────────────────────────

_RE_LATTICE_BLOCK = re.compile(
    r"(%BLOCK LATTICE_CART.*?%ENDBLOCK LATTICE_CART)", re.DOTALL | re.I
)


def _scale_lattice_block(block_text: str, scale: float) -> str:
    """Scale all numeric lattice vector lines in a LATTICE_CART block."""
    result_lines: list[str] = []
    for line in block_text.splitlines():
        stripped = line.strip().lower()
        if not stripped or stripped in {"ang", "bohr", "a.u.", "angstrom"}:
            result_lines.append(line)
            continue
        if stripped.startswith("%"):
            result_lines.append(line)
            continue
        match = _RE_COORD.search(line)
        if match:
            v0 = float(match.group(1)) * scale
            v1 = float(match.group(2)) * scale
            v2 = float(match.group(3)) * scale
            scaled_line = _RE_COORD.sub(
                f"{v0:20.15f}   {v1:20.15f}   {v2:20.15f}", line, count=1
            )
            result_lines.append(scaled_line)
        else:
            result_lines.append(line)
    return "\n".join(result_lines)


def vegard_scale(cell_text: str, species_a: str, species_b: str, x: float) -> str:
    """
    Scale LATTICE_CART vectors using Vegard's law for concentration x.

    a_VCA(x) = (1-x) * r_A + x * r_B
    scale     = a_VCA(x) / r_A

    No-op if x ≈ 0, or if either element radius is unknown.
    """
    if x < 1e-9:
        return cell_text

    radius_a = _RADIUS.get(species_a.capitalize())
    radius_b = _RADIUS.get(species_b.capitalize())
    if radius_a is None or radius_b is None:
        return cell_text

    scale = ((1.0 - x) * radius_a + x * radius_b) / radius_a
    if abs(scale - 1.0) < 1e-6:
        return cell_text

    def _replace_block(match: re.Match) -> str:  # type: ignore[type-arg]
        return _scale_lattice_block(match.group(1), scale)

    return _RE_LATTICE_BLOCK.sub(_replace_block, cell_text)


# ─────────────────────────────────────────────────────────────────────────────
# VCA .cell writer
# ─────────────────────────────────────────────────────────────────────────────

_RE_POSITIONS_BLOCK = re.compile(
    r"%BLOCK POSITIONS_FRAC.*?%ENDBLOCK POSITIONS_FRAC", re.DOTALL | re.I
)


def write_vca_cell(
    dest: Path,
    src: Path,
    species_a: str,
    species_b: str,
    x: float,
    *,
    vegard: bool = True,
    nonmetal_occupancy: float = 1.0,
) -> None:
    """
    Write a CASTEP .cell file with VCA MIXTURE syntax for concentration x.

    Case A — species_b absent in template (e.g. TiC.cell, only Ti present):
        Every species_a row → MIXTURE(species_a, species_b).
        Other species (C, N …) are copied verbatim.

    Case B — species_b already present (e.g. NbMo.cell, both Nb+Mo rows):
        All species_a and species_b rows are pooled (deduplicated) and each
        position gets MIXTURE(species_a, species_b).

    At x ≈ 0 or x ≈ 1 the MIXTURE tag is omitted and the pure element is
    written directly (avoids symmetry issues).
    Lattice is pre-scaled by Vegard law unless vegard=False.

    nonmetal_occupancy (VEC stabilizer)
    ────────────────────────────────────
    When nonmetal_occupancy < 1.0, all non-metal (C, N …) sites are written
    with a MIXTURE tag that creates a partial vacancy:

        C   0.5 0.5 0.5   MIXTURE:( 1 {nonmetal_occupancy:.8f})

    CASTEP interprets the missing fraction as a structural vacancy — the
    site is occupied with probability nonmetal_occupancy and empty otherwise.
    This reduces the effective VEC towards the target (default 8.0) and
    stabilises the C44 elastic constant for Nb/Mo-rich compositions.
    Use vacancy_fraction_for_vec() from castep_io to compute this value.
    """
    fraction_a = round(1.0 - x, 10)
    fraction_b = round(x, 10)
    src_text = src.read_text(encoding="utf-8", errors="replace")
    species_a_lower = species_a.lower()
    species_b_lower = species_b.lower()

    species_coords, species_order = _parse_positions(src_text)

    if species_a_lower not in species_coords:
        found_labels = ", ".join(species_order)
        raise ValueError(
            f"Species '{species_a}' not found in {src.name} POSITIONS_FRAC.\n"
            f"  Found: {found_labels}\n"
            f"  Hint: --species labels must match element symbols in the .cell file."
        )

    species_b_in_template = species_b_lower in species_coords

    def _build_site_lines(coord: str) -> list[str]:
        """Return one or two POSITIONS_FRAC lines for a single mixed site."""
        if x < 1e-9:
            return [f"{species_a}   {coord}"]
        if x > 1.0 - 1e-9:
            return [f"{species_b}   {coord}"]
        return [
            f"{species_a}   {coord}  MIXTURE:( 1 {fraction_a:.8f})",
            f"{species_b}   {coord}  MIXTURE:( 1 {fraction_b:.8f})",
        ]

    new_rows: list[str] = ["%BLOCK POSITIONS_FRAC"]

    if species_b_in_template:
        # Case B: pool both species → one VCA sublattice
        pooled_coords: list[str] = []
        seen_coords: set[str] = set()
        for key in (species_a_lower, species_b_lower):
            for coord in species_coords.get(key, []):
                if coord not in seen_coords:
                    seen_coords.add(coord)
                    pooled_coords.append(coord)
        for coord in pooled_coords:
            new_rows.extend(_build_site_lines(coord))
        for label in species_order:
            if label.lower() not in (species_a_lower, species_b_lower):
                for coord in species_coords[label.lower()]:
                    new_rows.append(f"{label}   {coord}")
    else:
        # Case A: only species_a rows are substituted
        for label in species_order:
            for coord in species_coords[label.lower()]:
                if label.lower() == species_a_lower:
                    new_rows.extend(_build_site_lines(coord))
                else:
                    new_rows.append(f"{label}   {coord}")

    new_rows.append("%ENDBLOCK POSITIONS_FRAC")

    # ── Apply nonmetal vacancy (VEC stabilizer) ───────────────────────────────
    # If nonmetal_occupancy < 1.0, rewrite all non-metal site lines to include
    # a MIXTURE tag. Nonmetals are identified as elements NOT equal to species_a
    # or species_b that also appear in _NONMETALS. Only applied for intermediate
    # x (not pure endpoints) since pure phases don't need VEC correction.
    if nonmetal_occupancy < 1.0 - 1e-6 and 1e-6 < x < 1.0 - 1e-6:
        occ = round(nonmetal_occupancy, 8)
        patched_rows: list[str] = []
        for row in new_rows:
            stripped = row.strip()
            if stripped.startswith("%"):
                patched_rows.append(row)
                continue
            parts = stripped.split()
            if parts and parts[0].capitalize() in _NONMETALS:
                # Rebuild with vacancy MIXTURE
                label_nm = parts[0]
                coords_nm = (
                    "   ".join(parts[1:4]) if len(parts) >= 4 else " ".join(parts[1:])
                )
                patched_rows.append(f"{label_nm}   {coords_nm}  MIXTURE:( 1 {occ:.8f})")
            else:
                patched_rows.append(row)
        new_rows = patched_rows
    new_block = "\n".join(new_rows)
    new_text = _RE_POSITIONS_BLOCK.sub(new_block, src_text)

    if vegard:
        new_text = vegard_scale(new_text, species_a, species_b, x)

    header = (
        f"# VCA  {species_a}(1-x){species_b}(x)  x={x:.6f}  — vca_tool v{VERSION}\n"
    )
    dest.write_text(header + new_text, encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Smart .param generator
# ─────────────────────────────────────────────────────────────────────────────


def nextra_for_element(elem: str) -> int:
    """
    Base nextra_bands for a PURE element (used at x=0 and x=1 endpoints).

    d-block metals have complex Fermi surfaces → need more empty bands.
    Hard nonmetals (C, N …) are insulators/semiconductors in pure form → few bands.
    """
    element = elem.capitalize()
    if element in _D_BLOCK:
        return 10  # pure d-metal: moderate Fermi surface
    if element in _HARD:
        return 4  # pure nonmetal: typically gapped
    return 6  # s/p metal


def nextra_for_step(species_a: str, species_b: str, x: float) -> int:
    """
    Per-step adaptive nextra_bands using linear interpolation + VCA overhead.

    At x=0: pure A → nextra_for_element(A)
    At x=1: pure B → nextra_for_element(B)
    At 0 < x < 1: interpolated base + 10 (VCA fractional charge creates
                  extra band complexity not present in pure compounds)

    Example Ti→Nb sweep:
      x=0.0 → 10  (pure Ti)
      x=0.33→ 20  (Ti₀.₆₇Nb₀.₃₃ VCA)
      x=0.67→ 20  (Ti₀.₃₃Nb₀.₆₇ VCA)
      x=1.0 → 10  (pure Nb)
    """
    n_a = nextra_for_element(species_a)
    n_b = nextra_for_element(species_b)
    base = round(n_a * (1.0 - x) + n_b * x)
    if 0.01 < x < 0.99:
        return base + 10  # VCA fractional charge overhead
    return base


def _choose_nextra_bands(species_list: list[str], is_vca: bool) -> int:
    """
    Single-compound (non-VCA) nextra_bands recommendation.
    Used by wizard_param() for the template .param file.
    """
    has_d = any(s.capitalize() in _D_BLOCK for s in species_list)
    if is_vca:
        # Template value shown during wizard — actual per-step values come from
        # nextra_for_step() at execution time.
        return 20 if has_d else 15
    return 10 if has_d else 4


def _choose_cutoff(species_list: list[str]) -> int:
    """Return recommended cut-off energy in eV. Hard elements require ≥700 eV."""
    has_hard = any(s.capitalize() in _HARD for s in species_list)
    return 700 if has_hard else 500


def _choose_spin(species_list: list[str]) -> bool:
    """Return True if any magnetic element is present."""
    return any(s.capitalize() in _MAGNETIC for s in species_list)


def param_smart_defaults(species_list: list[str], is_vca: bool) -> dict[str, Any]:
    """
    Return a dict of recommended .param values for the given species set.
    Called by ui.py to show the user what the program recommends.
    """
    hard_detected = sorted(s for s in species_list if s.capitalize() in _HARD)
    magnetic_detected = sorted(s for s in species_list if s.capitalize() in _MAGNETIC)

    return {
        "nextra_bands": _choose_nextra_bands(species_list, is_vca),
        "cut_off_energy": _choose_cutoff(species_list),
        "spin_polarized": _choose_spin(species_list),
        "hard_detected": hard_detected,
        "magnetic_detected": magnetic_detected,
    }


def write_param(
    param_path: Path,
    task: str,
    xc_functional: str,
    cut_off_energy: int,
    spin_polarized: bool,
    nextra_bands: int,
    *,
    ncp: bool = False,
) -> None:
    """
    Write a CASTEP 25 .param file with VCA-optimised settings.

    ncp=True is required for ElasticConstants and Phonons (DFPT) tasks —
    those tasks cannot use the default ultrasoft pseudopotentials.
    When ncp=True the caller must also call inject_species_pot_ncp() on
    the corresponding .cell file, and use a higher cut_off_energy
    (≥900 eV for systems with C/N/O, ≥700 eV for pure metals).
    """
    # FiniteStrainElastic is a vca_tool marker task, not a native CASTEP task.
    # It is stored in the .param file so main.py knows to run the 2-phase
    # workflow. The actual CASTEP runs use GeometryOptimization (phase 1)
    # and SinglePoint (phase 2) with tighter tolerances for accurate Cij.
    is_geom = task in {"GeometryOptimization", "FiniteStrainElastic"}
    is_elastic = task == "ElasticConstants"

    if is_geom:
        # FiniteStrainElastic needs tighter tolerances than normal GeomOpt
        # so we apply the elastic-grade tolerances for both.
        geom_block = (
            "calculate_stress    : true\n"
            "geom_method         : LBFGS\n"
            "geom_max_iter       : 150\n"
            "geom_energy_tol     : 1.0e-6 eV\n"
            "geom_force_tol      : 0.005 eV/ang\n"
            "geom_stress_tol     : 0.03 GPa\n"
            "geom_disp_tol       : 0.0005 ang\n"
        )
    elif is_elastic:
        # ElasticConstants needs tighter SCF convergence for accurate shear moduli.
        # CASTEP internally runs geometry optimisation for each strained structure
        # (ions only, fixed cell) — these tolerances apply to those sub-runs.
        geom_block = (
            "calculate_stress    : true\n"
            "geom_method         : LBFGS\n"
            "geom_max_iter       : 100\n"
            "geom_energy_tol     : 1.0e-6 eV\n"
            "geom_force_tol      : 0.005 eV/ang\n"
            "geom_disp_tol       : 0.0005 ang\n"
        )
    else:
        geom_block = "calculate_stress    : false\n"

    spin_str = "true" if spin_polarized else "false"
    ncp_note = (
        "  # NCP required for this task — raise cutoff to ≥900 eV for C/N/O"
        if ncp
        else ""
    )
    content = (
        f"# CASTEP 25 parameter file — generated by vca_tool v{VERSION}\n"
        f"# ========================================================\n"
        f"# BASIC SETUP\n"
        f"# ========================================================\n"
        f"task                : {task}\n"
        f"xc_functional       : {xc_functional}\n"
        f"cut_off_energy      : {cut_off_energy} eV{ncp_note}\n"
        f"spin_polarized      : {spin_str}\n"
        f"\n"
        f"# ========================================================\n"
        f"# SCF & ELECTRONIC MINIMIZATION  (VCA optimised)\n"
        f"# ========================================================\n"
        f"max_scf_cycles      : 300\n"
        f"metals_method       : dm\n"
        f"mixing_scheme       : Pulay\n"
        f"smearing_width      : 0.10 eV\n"
        f"mix_charge_amp      : 0.2\n"
        f"nextra_bands        : {nextra_bands}\n"
        f"elec_energy_tol     : 1.0e-5 eV  # GeomOpt tolerance; elastic uses 1e-7\n"
        f"\n"
        f"# ========================================================\n"
        f"# GEOMETRY / TASK SETTINGS\n"
        f"# ========================================================\n"
        f"{geom_block}"
        f"\n"
        f"# ========================================================\n"
        f"# I/O AND PERFORMANCE\n"
        f"# ========================================================\n"
        f"opt_strategy        : speed\n"
        f"write_checkpoint    : none\n"
        f"num_dump_cycles     : 0\n"
        f"write_cell_structure: true\n"
    )
    param_path.write_text(content, encoding="utf-8")


def patch_nextra_bands(param_path: Path, nextra: int) -> None:
    """
    In-place replacement of the nextra_bands line in an existing .param file.

    Reads the file, replaces only the nextra_bands line, writes it back.
    All other settings (task, xc_functional, cut_off_energy …) are preserved.
    This is how per-step param files are created from the template without
    touching the original in the user's working directory.
    """
    text = param_path.read_text(encoding="utf-8")
    new_lines: list[str] = []
    replaced = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("nextra_bands") and ":" in stripped:
            new_lines.append(f"nextra_bands        : {nextra}")
            replaced = True
        else:
            new_lines.append(line)
    if not replaced:
        # nextra_bands line absent — append it before the last section
        new_lines.append(f"nextra_bands        : {nextra}")
    param_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# .castep output parser
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class CastepResult:
    """Physical results extracted from a .castep output file."""

    enthalpy_eV: float | None = None
    a_opt_ang: float | None = None
    b_opt_ang: float | None = None
    c_opt_ang: float | None = None
    volume_ang3: float | None = None
    density_gcm3: float | None = None
    bulk_modulus_GPa: float | None = None
    wall_time_s: float | None = None
    geom_converged: bool = False
    task_type: str = ""  # "GeometryOptimization" or "SinglePoint" etc.
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Export all non-None fields as a flat dict for CSV/JSON serialisation."""
        result: dict[str, Any] = {}
        result["enthalpy_eV"] = (
            f"{self.enthalpy_eV:.6f}" if self.enthalpy_eV is not None else ""
        )
        result["a_opt_ang"] = (
            f"{self.a_opt_ang:.5f}" if self.a_opt_ang is not None else ""
        )
        result["b_opt_ang"] = (
            f"{self.b_opt_ang:.5f}" if self.b_opt_ang is not None else ""
        )
        result["c_opt_ang"] = (
            f"{self.c_opt_ang:.5f}" if self.c_opt_ang is not None else ""
        )
        result["volume_ang3"] = (
            f"{self.volume_ang3:.4f}" if self.volume_ang3 is not None else ""
        )
        result["density_gcm3"] = (
            f"{self.density_gcm3:.4f}" if self.density_gcm3 is not None else ""
        )
        result["bulk_modulus_GPa"] = (
            f"{self.bulk_modulus_GPa:.2f}" if self.bulk_modulus_GPa is not None else ""
        )
        result["wall_time_s"] = (
            f"{self.wall_time_s:.1f}" if self.wall_time_s is not None else ""
        )
        result["geom_converged"] = "yes" if self.geom_converged else "no"
        result["warnings"] = "; ".join(self.warnings[:3])
        return result


def inject_species_pot_ncp(cell_path: Path) -> None:
    """
    Ensure every species in the .cell file uses norm-conserving pseudopotentials.

    ElasticConstants and Phonon tasks cannot use ultrasoft pseudopotentials —
    including for VCA MIXTURE atoms. The only reliable way to force NCP for
    all species (including on-the-fly VCA virtual atoms) is to list each
    element explicitly in SPECIES_POT with the "NCP" keyword:

        %BLOCK SPECIES_POT
        Ti  NCP
        V   NCP
        C   NCP
        %ENDBLOCK SPECIES_POT

    A bare "NCP" without element names is ignored for MIXTURE/OTFG atoms.

    This function:
      1. Reads the species present in POSITIONS_FRAC.
      2. Removes any existing SPECIES_POT block.
      3. Writes a fresh block with every species explicitly set to NCP.

    Safe to call multiple times — re-generates the block each time to
    ensure completeness if species change between calls.
    NCP requires a higher cutoff than USP: ≥900 eV for C/N/O, ≥700 for metals.
    """
    content = cell_path.read_text(encoding="utf-8", errors="replace")
    species = read_species(cell_path)
    if not species:
        # Fallback: generic NCP block if we cannot read species
        species_lines = "NCP"
    else:
        species_lines = "\n".join(f"{s}  NCP" for s in species)

    # Remove any existing SPECIES_POT block before re-writing
    content = re.sub(
        r"%BLOCK\s+SPECIES_POT.*?%ENDBLOCK\s+SPECIES_POT\s*",
        "",
        content,
        flags=re.DOTALL | re.I,
    )
    ncp_block = f"\n%BLOCK SPECIES_POT\n{species_lines}\n%ENDBLOCK SPECIES_POT\n"
    cell_path.write_text(content.rstrip() + ncp_block, encoding="utf-8")


def _try_float(value: str) -> float | None:
    """Safe float conversion; returns None on failure."""
    try:
        return float(value)
    except ValueError:
        return None


def parse_castep_log(castep_path: Path) -> CastepResult:
    """
    Parse a CASTEP 25 output file and extract physical results.

    Single reverse pass (bottom-to-top) so the FINAL values are found first
    and earlier intermediate values are automatically ignored.

    Handles both:
      GeometryOptimization → "LBFGS: Final Enthalpy     = -1.90301018E+003 eV"
      SinglePoint          → "Final energy, E             =  -1902.769822248     eV"

    Real CASTEP 25.12 line formats (no fragile regex):
      a =      3.065962          alpha =   60.000000
      Current cell volume =            20.379108       A**3
                          =            12.449217     g/cm^3   (density line, no AMU)
      Total time          =     48.95 s
    """
    result = CastepResult()

    if not castep_path.exists():
        result.warnings.append("output file not found")
        return result

    try:
        lines = castep_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        result.warnings.append(str(exc))
        return result

    no_empty_bands_seen = False

    # Accumulate a/b/c separately; assign to result once all three are found
    # (first complete triplet in reverse order = last/final geometry block)
    lat_a: float | None = None
    lat_b: float | None = None
    lat_c: float | None = None

    for line in reversed(lines):
        stripped = line.strip()

        # ── Task type ────────────────────────────────────────────────────────
        if not result.task_type and "type of calculation" in stripped.lower():
            colon_parts = stripped.split(":")
            if len(colon_parts) >= 2:
                result.task_type = colon_parts[-1].strip()

        # ── Geometry convergence ─────────────────────────────────────────────
        if (
            not result.geom_converged
            and "Geometry optimization completed successfully" in line
        ):
            result.geom_converged = True

        # ── Enthalpy: "LBFGS: Final Enthalpy     = -1.90301018E+003 eV" ──────
        if (
            result.enthalpy_eV is None
            and "Final Enthalpy" in line
            and "Pseudo" not in line
        ):
            eq_parts = line.split("=")
            if len(eq_parts) >= 2:
                result.enthalpy_eV = _try_float(eq_parts[-1].strip().split()[0])

        # ── Energy fallback: "Final energy, E             =  -1902.769 eV" ───
        if (
            result.enthalpy_eV is None
            and "Final energy, E" in line
            and "Pseudo" not in line
        ):
            eq_parts = line.split("=")
            if len(eq_parts) >= 2:
                result.enthalpy_eV = _try_float(eq_parts[-1].strip().split()[0])

        # ── Lattice: "   a =      3.065962          alpha =   60.000000" ──────
        if lat_a is None and stripped.startswith("a ="):
            lat_a = _try_float(stripped.split("=")[1].strip().split()[0])
        if lat_b is None and stripped.startswith("b ="):
            lat_b = _try_float(stripped.split("=")[1].strip().split()[0])
        if lat_c is None and stripped.startswith("c ="):
            lat_c = _try_float(stripped.split("=")[1].strip().split()[0])

        # Once all three are found for the first time (= last geom in file), save
        if lat_a is not None and lat_b is not None and lat_c is not None:
            if result.a_opt_ang is None:
                result.a_opt_ang = lat_a
                result.b_opt_ang = lat_b
                result.c_opt_ang = lat_c

        # ── Volume: "Current cell volume =            20.379108       A**3" ───
        if result.volume_ang3 is None and "Current cell volume" in line and "=" in line:
            eq_parts = line.split("=")
            if len(eq_parts) >= 2:
                result.volume_ang3 = _try_float(eq_parts[-1].strip().split()[0])

        # ── Density: "                    =            12.449217     g/cm^3" ──
        # Must exclude "AMU/A**3" line that appears just above it
        if (
            result.density_gcm3 is None
            and "g/cm" in line
            and "=" in line
            and "AMU" not in line
        ):
            eq_parts = line.split("=")
            if len(eq_parts) >= 2:
                value = _try_float(eq_parts[-1].strip().split()[0])
                if value is not None and value > 0:
                    result.density_gcm3 = value

        # ── Bulk modulus: "LBFGS: Final bulk modulus = …" ────────────────────
        if (
            result.bulk_modulus_GPa is None
            and "Final bulk modulus" in line
            and "=" in line
        ):
            eq_parts = line.split("=")
            if len(eq_parts) >= 2:
                result.bulk_modulus_GPa = _try_float(eq_parts[-1].strip().split()[0])

        # ── Wall time: "Total time          =     48.95 s" ────────────────────
        if result.wall_time_s is None and "Total time" in line and "=" in line:
            eq_parts = line.split("=")
            if len(eq_parts) >= 2:
                result.wall_time_s = _try_float(eq_parts[-1].strip().split()[0])

        # ── Warnings ─────────────────────────────────────────────────────────
        if not no_empty_bands_seen and "no empty bands" in stripped.lower():
            result.warnings.append("no empty bands — increase nextra_bands in .param")
            no_empty_bands_seen = True

        if stripped.lower().startswith("warning") and len(result.warnings) < 5:
            colon_parts = stripped.split(":", 1)
            if len(colon_parts) == 2:
                msg = colon_parts[1].strip()
                if msg and msg not in result.warnings:
                    result.warnings.append(msg)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# .elastic file parser
# ─────────────────────────────────────────────────────────────────────────────


def parse_elastic_file(elastic_path: Path) -> dict[str, Any]:
    """
    Parse a CASTEP .elastic output file and extract the full elastic tensor
    and derived mechanical properties.

    CASTEP writes the .elastic file when task : ElasticConstants completes.
    It contains the 6×6 Cij tensor, Sij compliances, and derived properties
    (bulk modulus, shear modulus, Young's modulus, Poisson ratio, Debye temp,
    Vickers hardness) using Voigt, Reuss, and Hill averaging schemes.

    Returns a flat dict[str, Any] suitable for direct insertion into Step.parsed
    so all columns appear automatically in vca_results.csv.

    Real .elastic file format (CASTEP 25):
        ===================================== Elastic Stiffness Constants Cij (GPa) ================
        775.33017  163.75610  163.75610  0.00000  0.00000  0.00000
        ...
        Voigt bulk modulus    =   367.613
        Reuss bulk modulus    =   367.613
        Hill bulk modulus     =   367.613
        Voigt shear modulus   =   ...
        Young modulus ...
        Poisson ratio ...
        Debye temperature     =   ...
        Vickers hardness      =   ...
    """
    result: dict[str, Any] = {}

    if not elastic_path.exists():
        return result

    try:
        lines = elastic_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return result

    in_cij = False
    cij_rows: list[list[float]] = []

    for line in lines:
        stripped = line.strip()

        # ── Cij 6×6 matrix ───────────────────────────────────────────────────
        if "Elastic Stiffness Constants Cij" in line:
            in_cij = True
            cij_rows = []
            continue
        if in_cij:
            if stripped.startswith("=") or not stripped:
                if len(cij_rows) == 6:
                    in_cij = False
                continue
            parts = stripped.split()
            if len(parts) == 6:
                try:
                    cij_rows.append([float(v) for v in parts])
                except ValueError:
                    pass
            if len(cij_rows) == 6:
                in_cij = False
                # Store individual Cij components (uppercase C11, C12, …)
                labels = [
                    ("C11", 0, 0),
                    ("C12", 0, 1),
                    ("C13", 0, 2),
                    ("C22", 1, 1),
                    ("C23", 1, 2),
                    ("C33", 2, 2),
                    ("C44", 3, 3),
                    ("C55", 4, 4),
                    ("C66", 5, 5),
                ]
                for key, i, j in labels:
                    result[key] = f"{cij_rows[i][j]:.4f}"

        # ── Derived properties — "Label = value" format ──────────────────────
        _PROPS = {
            "Voigt bulk modulus": "B_Voigt_GPa",
            "Reuss bulk modulus": "B_Reuss_GPa",
            "Hill bulk modulus": "B_Hill_GPa",
            "Voigt shear modulus": "G_Voigt_GPa",
            "Reuss shear modulus": "G_Reuss_GPa",
            "Hill shear modulus": "G_Hill_GPa",
            "Young modulus": "E_GPa",
            "Poisson ratio": "nu",
            "Debye temperature": "T_Debye_K",
            "Vickers hardness": "H_Vickers_GPa",
        }
        for label, col_name in _PROPS.items():
            if label in line and "=" in line:
                eq_parts = line.split("=")
                if len(eq_parts) >= 2:
                    value = _try_float(eq_parts[-1].strip().split()[0])
                    if value is not None:
                        result[col_name] = f"{value:.4f}"

    return result
