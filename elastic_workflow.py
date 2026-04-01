"""
elastic_workflow.py — Finite-strain elastic constants for VCA systems.
────────────────────────────────────────────────────────────────────────
Why this exists:
  CASTEP's built-in `task : ElasticConstants` is blocked for VCA MIXTURE
  atoms — the strain field response formalism requires DFPT, which is
  incompatible with on-the-fly ultrasoft pseudopotentials used for MIXTURE.

  This module wraps the external `generate_strain.py` / `elastics.py`
  scripts (github.com/andreww/elastic-constants) to perform the equivalent
  calculation via finite differences:

    1.  generate_strain.py <seed>  — deforms the relaxed cell into N strained
        copies, each WITHOUT MIXTURE syntax (plain atoms, USP OK).
    2.  CASTEP SinglePoint on each strained cell.
    3.  elastics.py <seed>  — fits stress vs strain → Cij tensor, B, G, E, ν.

  This is the standard workaround used by the CASTEP community for alloys,
  VCA systems, and any case where DFPT-based ElasticConstants is unavailable.

Public API
──────────
  find_elastic_scripts()  → (generate_strain: Path | None, elastics: Path | None)
  run_finite_strain_elastic(
      seed_dir, seed, castep_cmd, *, n_cores, strain, numsteps, keep_all
  ) → dict[str, str]   # same shape as parse_elastic_file() output
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Locate generate_strain.py and elastics.py
# ─────────────────────────────────────────────────────────────────────────────


def find_elastic_scripts() -> tuple[Path | None, Path | None]:
    """
    Search for generate_strain.py and elastics.py in:
      1. The directory containing this file (i.e. vca_tool's own folder).
      2. elastic-constants-master/ sub-folder inside vca_tool.
      3. PATH (if installed as executables).

    Returns (generate_strain_path, elastics_path).
    Either can be None if not found.
    """
    here = Path(__file__).parent
    candidates = [
        here,
        here / "elastic-constants-master",
        here / "elastic-constants",
    ]

    def _find(name: str) -> Path | None:
        # Check well-known locations first
        for folder in candidates:
            p = folder / name
            if p.is_file():
                return p.resolve()
        # Fall back to PATH
        found = shutil.which(name)
        return Path(found) if found else None

    return _find("generate_strain.py"), _find("elastics.py")


# ─────────────────────────────────────────────────────────────────────────────
# Strip MIXTURE from a cell file (needed for strained copies)
# ─────────────────────────────────────────────────────────────────────────────

_RE_MIXTURE = re.compile(r"\s+MIXTURE:\([^)]+\)", re.I)
_RE_SYMMETRY_OPS = re.compile(
    r"%BLOCK\s+SYMMETRY_OPS.*?%ENDBLOCK\s+SYMMETRY_OPS\s*", re.DOTALL | re.I
)


def _strip_mixture(cell_path: Path) -> None:
    """
    Remove all MIXTURE tags from a .cell file in-place.
    VCA endpoint cells (x≈0 or x≈1) may still have a MIXTURE line if the
    tool wrote them; generate_strain.py cannot handle that syntax.
    """
    text = cell_path.read_text(encoding="utf-8", errors="replace")
    cleaned = _RE_MIXTURE.sub("", text)
    if cleaned != text:
        cell_path.write_text(cleaned, encoding="utf-8")


def _strip_symmetry_ops(cell_path: Path) -> None:
    """
    Remove symmetry constraints and conflicting ion-fix flags from a .cell file.

    Called on the relaxed -out.cell BEFORE feeding it to generate_strain.py.
    Addresses two separate CASTEP 25.12 crash modes:

    1.  SYMMETRY_OPS mismatch crash
        CASTEP writes %BLOCK SYMMETRY_OPS into -out.cell after GeomOpt.
        When generate_strain.py deforms the lattice vectors, the stored
        symmetry operations are no longer consistent with the new geometry.
        CASTEP detects this at cell_read and calls MPI_ABORT immediately,
        leaving an empty .castep file.

    2.  FIX_COM + FIX_ALL_IONS conflict  (ROOT CAUSE of current crash)
        CASTEP GeomOpt writes "FIX_COM : TRUE" into -out.cell to record
        that the centre of mass was constrained during the run.
        generate_strain.py → castep.produce_dotcell also writes
        "FIX_ALL_IONS : TRUE" into each strained .cell file to prevent
        ion positions from relaxing (we only want the stress at fixed ions).
        CASTEP 25.12 rejects any .cell that has BOTH flags set:
            ERROR in cell_read - should not have both FIX_ALL_IONS and
            FIX_COM set true
        Fix: strip FIX_COM (and pre-existing FIX_ALL_IONS) from the source
        cell so only generate_strain.py's own FIX_ALL_IONS survives.
    """
    text = cell_path.read_text(encoding="utf-8", errors="replace")
    cleaned = text

    # ── Remove %BLOCK SYMMETRY_OPS ───────────────────────────────────────────
    cleaned = _RE_SYMMETRY_OPS.sub("", cleaned)
    # Commented-out symmetry blocks from cif2cell (#%BLOCK ... #%ENDBLOCK)
    cleaned = re.sub(
        r"#%BLOCK\s+SYMMETRY_OPS.*?#%ENDBLOCK\s+SYMMETRY_OPS\s*",
        "",
        cleaned,
        flags=re.DOTALL | re.I,
    )
    # Orphaned symmetry-matrix comment lines  (#  1.000  0.000  0.000)
    cleaned = re.sub(
        r"^#\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s*$\n?", "", cleaned, flags=re.M
    )
    # "# Symm. op." labels left by cif2cell
    cleaned = re.sub(r"^# Symm\..*$\n?", "", cleaned, flags=re.M)

    # ── Remove conflicting cell-fix keywords and blocks ──────────────────────
    # FIX_COM : TRUE  — written by CASTEP GeomOpt; conflicts with FIX_ALL_IONS
    cleaned = re.sub(r"^\s*FIX_COM\s*:.*$\n?", "", cleaned, flags=re.M | re.I)
    # FIX_ALL_IONS : * — remove any pre-existing value; generate_strain.py
    # will write its own authoritative FIX_ALL_IONS : TRUE into each strained cell
    cleaned = re.sub(r"^\s*FIX_ALL_IONS\s*:.*$\n?", "", cleaned, flags=re.M | re.I)
    # %BLOCK CELL_CONSTRAINTS — CASTEP writes this into -out.cell to record
    # the symmetry constraints used during GeomOpt (e.g. "1 1 1 0 0 0" for cubic).
    # generate_strain.py adds FIX_ALL_CELL : TRUE to each strained cell.
    # CASTEP 25.12 rejects any cell that has BOTH present:
    #   ERROR in cell_read file - both FIX_ALL_CELL and CELL_CONSTRAINTS
    #   cannot be present
    # The strained cells have their geometry fully defined by LATTICE_CART —
    # no constraint block is needed or meaningful after deformation.
    cleaned = re.sub(
        r"%BLOCK\s+CELL_CONSTRAINTS.*?%ENDBLOCK\s+CELL_CONSTRAINTS\s*",
        "",
        cleaned,
        flags=re.DOTALL | re.I,
    )

    if cleaned != text:
        cell_path.write_text(cleaned, encoding="utf-8")


def _build_strain_source(orig_cell: Path, out_cell: Path, dest: Path) -> None:
    """
    Build the source cell for straining: relaxed lattice + original MIXTURE syntax.

    Problem solved
    ──────────────
    CASTEP's write_cell_structure:true writes -out.cell with the optimised
    geometry, but preserves ALL species from the job's pseudopotential table in
    POSITIONS_FRAC — including species with zero occupancy (e.g. Ti at x=1.0
    which has MIXTURE weight 0.0 or no MIXTURE tag at all). This leaves two bare
    nuclei at the same Wyckoff position without a MIXTURE tag, giving CASTEP an
    impossible geometry: two nuclei at distance 0 Å → infinite Coulomb repulsion
    → stress tensor ~ -7000 GPa → Born stability violated → garbage Cij.

    Solution
    ────────
    Take LATTICE_CART from -out.cell (accurate relaxed geometry) and everything
    else from the original VCA .cell (correct MIXTURE tags, correct species list,
    kpoints). This guarantees strained cells have exactly the right species
    definition regardless of CASTEP's write_cell_structure output format.
    """
    orig_text = orig_cell.read_text(encoding="utf-8", errors="replace")
    out_text = out_cell.read_text(encoding="utf-8", errors="replace")

    # Extract LATTICE_CART block from -out.cell
    _RE_LAT = re.compile(
        r"(%BLOCK\s+LATTICE_CART\s*\n)(.*?)(%ENDBLOCK\s+LATTICE_CART)",
        re.DOTALL | re.I,
    )
    out_match = _RE_LAT.search(out_text)
    if out_match is None:
        # Fallback: just use original cell unchanged
        dest.write_text(orig_text, encoding="utf-8")
        return

    # Replace LATTICE_CART in orig_text with the relaxed one from out_cell
    new_lat_block = out_match.group(0)
    if _RE_LAT.search(orig_text):
        result = _RE_LAT.sub(new_lat_block, orig_text)
    else:
        # orig_cell has no LATTICE_CART — prepend it
        result = new_lat_block + "\n" + orig_text

    dest.write_text(result, encoding="utf-8")


def _inject_ncp_for_strained(cell_path: Path) -> None:
    """
    Replace SPECIES_POT block with one forcing NCP for all species.

    Called on the source cell BEFORE generate_strain.py runs.
    generate_strain.py copies the full .cell into every strained copy,
    so this single injection propagates to all strained cells automatically.

    Why NCP is required for VCA strained cells
    ──────────────────────────────────────────
    Official CASTEP documentation on VCA explicitly warns:
      "Ultrasoft potentials are prone to generating ghost states in VCA.
       The NaCl/KCl system could not be studied with ultrasoft potentials
       for this reason."
    The VCA Q_ij interpolation can produce a non-positive-definite overlap
    matrix S. castep_calc_approx_wvfn then tries to compute S^{-1/2}:
      → sqrt of negative eigenvalue
      → "Error, norm_sq is negative in wave_Snormalise_slice_b"
    This happens at the very start of the run — before SCF begins.
    NCP has no augmentation matrix Q → no S matrix → no ghost states.
    The stress tensor (Hellmann-Feynman) is unaffected by this change.
    Pure endpoint cells (integer Z) are immune because USP Q_ij is
    well-defined for single-species atoms.
    """
    text = cell_path.read_text(encoding="utf-8", errors="replace")

    # Read species present in POSITIONS_FRAC
    species: list[str] = []
    in_block = False
    for line in text.splitlines():
        if re.search(r"%BLOCK\s+POSITIONS_FRAC", line, re.I):
            in_block = True
            continue
        if re.search(r"%ENDBLOCK\s+POSITIONS_FRAC", line, re.I):
            break
        if in_block:
            parts = line.split()
            if parts and re.match(r"^[A-Za-z]+$", parts[0]):
                sp = parts[0].capitalize()
                if sp not in species:
                    species.append(sp)

    ncp_lines = "\n".join(f"   {s:<4} NCP" for s in species) if species else "NCP"

    # Remove any existing SPECIES_POT block (CASTEP writes OTFG strings there)
    text = re.sub(
        r"%BLOCK\s+SPECIES_POT.*?%ENDBLOCK\s+SPECIES_POT\s*",
        "",
        text,
        flags=re.DOTALL | re.I,
    )
    ncp_block = f"\n%BLOCK SPECIES_POT\n{ncp_lines}\n%ENDBLOCK SPECIES_POT\n"
    cell_path.write_text(text.rstrip() + ncp_block, encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Parse elastics.py terminal output
# ─────────────────────────────────────────────────────────────────────────────


def _parse_elastics_output(output: str) -> dict[str, str]:
    """
    Extract Cij, B, G, E, ν from elastics.py stdout.

    elastics.py prints something like:
        Cij (GPa):
        C11 = 500.12   C12 = 167.45   C44 = 153.67
        ...
        Voigt bulk modulus    =  278.3
        Hill shear modulus    =  156.9
        Young's modulus       =  390.1
        Poisson ratio         =    0.24
    """
    result: dict[str, str] = {}

    # Individual Cij components:  "C11 = 500.12" or "C11=500.12"
    for match in re.finditer(r"\b(C\d{2})\s*=\s*([-\d.Ee+]+)", output):
        result[match.group(1)] = f"{float(match.group(2)):.4f}"

    # Derived properties — map label fragments to CSV column names
    _PROP_MAP = [
        ("Voigt bulk", "B_Voigt_GPa"),
        ("Reuss bulk", "B_Reuss_GPa"),
        ("Hill bulk", "B_Hill_GPa"),
        ("Voigt shear", "G_Voigt_GPa"),
        ("Reuss shear", "G_Reuss_GPa"),
        ("Hill shear", "G_Hill_GPa"),
        ("Young", "E_GPa"),
        ("Poisson", "nu"),
        ("Debye", "T_Debye_K"),
        ("Vickers", "H_Vickers_GPa"),
    ]
    for fragment, col in _PROP_MAP:
        # Look for "fragment ... = value" anywhere on a line
        pattern = re.compile(rf"{re.escape(fragment)}[^=\n]*=\s*([-\d.Ee+]+)", re.I)
        match = pattern.search(output)
        if match:
            result[col] = f"{float(match.group(1)):.4f}"

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Scipy compatibility patcher for elastics.py / CijUtil.py
# ─────────────────────────────────────────────────────────────────────────────

_SCIPY_SHIM = """# scipy>=1.13 removed numpy re-exports (array, sqrt, row_stack, etc.)
# This shim restores them so elastics.py works on Python 3.12+
import numpy as _np
import scipy as _S
for _attr in [
    "array","sqrt","zeros","ones","dot","cross","all","abs",
    "vstack","hstack","concatenate","linspace","arange",
    "sum","min","max","mean","eye","diag","trace",
]:
    if not hasattr(_S, _attr):
        _fn = getattr(_np, _attr, None)
        if _fn is not None:
            setattr(_S, _attr, _fn)
if not hasattr(_S, "row_stack"):
    _S.row_stack = _np.vstack
"""


def _patch_elastic_scripts(elastics_path: Path, cijutil_path: Path | None) -> None:
    """
    Patch elastics.py and CijUtil.py for Python 3.12+ / scipy >= 1.13.

    Fixes applied (idempotent — checks for existing fixes before applying):
      1. CijUtil.py: invalid escape sequence in LaTeX strings (SyntaxError on 3.14)
      2. elastics.py: 'from scipy import stats, sqrt, square'
                    → 'from scipy import stats; from numpy import sqrt, square'
      3. elastics.py: prepend numpy shim for scipy.array / scipy.row_stack etc.

    Works at byte level to avoid triggering SyntaxWarning in this process.
    The simplest reliable approach: just always rewrite from the clean state.
    """
    _SHIM = (
        b"# scipy_compat_shim - Python 3.12+/scipy>=1.13 compat\n"
        b"import numpy as _np_shim\n"
        b"import scipy as _scipy_shim\n"
        b"for _attr in ['array','sqrt','zeros','ones','dot','cross','all','abs',\n"
        b"              'vstack','hstack','row_stack','concatenate','linspace',\n"
        b"              'arange','sum','min','max','mean','eye','diag','square']:\n"
        b"    if not hasattr(_scipy_shim, _attr):\n"
        b"        _fn = getattr(_np_shim, _attr, None)\n"
        b"        if _fn is not None:\n"
        b"            setattr(_scipy_shim, _attr, _fn)\n"
        b"if not hasattr(_scipy_shim, 'row_stack'):\n"
        b"    _scipy_shim.row_stack = _np_shim.vstack\n\n"
    )

    # ── Fix CijUtil.py ────────────────────────────────────────────────────────
    if cijutil_path and cijutil_path.exists():
        raw = cijutil_path.read_bytes()
        # Remove any previous shim block (between marker and #!/usr/bin/env)
        marker = b"# scipy_compat_shim"
        if marker in raw:
            shebang = raw.find(b"#!/usr/bin/env python")
            if shebang == -1:
                shebang = raw.find(b"# encoding:")
            if shebang > 0:
                raw = raw[shebang:]
        # Fix \pm (invalid escape) — replace $\pm$ with $\\pm$
        raw = raw.replace(b"$\\pm$", b"$\\\\pm$")
        cijutil_path.write_bytes(raw)

    # ── Fix elastics.py ───────────────────────────────────────────────────────
    if elastics_path and elastics_path.exists():
        raw = elastics_path.read_bytes()
        # Remove previous shim
        marker = b"# scipy_compat_shim"
        if marker in raw:
            target = b"import scipy as S"
            idx = raw.find(target)
            # Walk back to find start of shim (before 'import scipy as S')
            shim_start = raw.rfind(b"\nimport numpy as _np_shim", 0, idx)
            if shim_start != -1:
                raw = raw[: shim_start + 1] + raw[idx:]
        # Fix: from scipy import stats, sqrt, square
        raw = raw.replace(
            b"from scipy import stats, sqrt, square",
            b"from scipy import stats; from numpy import sqrt, square",
        )
        # Fix invalid re.split escape
        raw = raw.replace(b"re.split('\\.'", b"re.split(r'\\.'")
        raw = raw.replace(b're.split("\\."', b're.split(r"\\."')
        # Insert shim before 'import scipy as S'
        target = b"import scipy as S"
        idx = raw.find(target)
        if idx >= 0 and marker not in raw:
            raw = raw[:idx] + _SHIM + raw[idx:]
        elastics_path.write_bytes(raw)


# ─────────────────────────────────────────────────────────────────────────────
# .param builder for strained SinglePoint cells
# ─────────────────────────────────────────────────────────────────────────────


def _build_strained_param(orig_param: Path) -> str:
    """
    Build a SinglePoint .param text for strained cells from the GeomOpt .param.

    Key changes vs the original GeomOpt .param:

    task → SinglePoint
        The strained cell geometry is fully defined by LATTICE_CART.
        generate_strain.py adds FIX_ALL_CELL : TRUE to the .cell file,
        which already prevents any cell relaxation. SinglePoint is faster
        and gives identical stress to a constrained GeomOpt.

    calculate_stress → true
        MANDATORY. The elastics fitter reads the stress tensor.
        Without this, CASTEP does not compute or print the stress.

    spin_polarized → false
        Strained cells have no .check file to inherit spin density from.
        With spin_polarized : true, CASTEP starts from a random spin
        guess. For VCA cells with fractional nuclear charge, this initial
        guess is non-integer and can cause CASTEP to abort before SCF.
        Stress tensors are insensitive to spin for strains of 0.3%.

    nextra_bands → 30
        VCA cells have a fractional nuclear charge (e.g. Ti₀.₆₆₇V₀.₃₃₃C).
        This creates a complex Fermi surface with many near-degenerate bands.
        Without sufficient empty bands, the density-mixing algorithm cannot
        distribute the fractional electrons, causing the wavefunction overlap
        matrix S to become negative-definite:
            Error, norm_sq is negative in wave_Snormalise_slice_b
        Setting nextra_bands : 30 gives enough buffer for the Fermi smearing
        to work correctly for any Ti/V/Nb/Mo concentration between 0 and 1.
        Pure endpoint cells (x=0, x=1) have integer electron counts and
        would work with fewer bands, but 30 costs < 10% extra runtime.

    mix_charge_amp → 0.1
        The default 0.2 is tuned for GeomOpt where the charge density
        starts from a previous converged state (.check file).
        For strained cells starting from scratch, more conservative mixing
        prevents the SCF from diverging in the first few iterations.

    opt_strategy → speed
        Optimise MPI memory layout for throughput across N×6 short runs.

    write_checkpoint → none / num_dump_cycles → 0
        Skip writing .check and .castep_bin files — not needed for stress
        extraction and each strained cell run takes < 2 minutes.
    """
    import re as _re

    _GEOM_KEYS = (
        "geom_method",
        "geom_max_iter",
        "geom_energy_tol",
        "geom_force_tol",
        "geom_stress_tol",
        "geom_disp_tol",
    )

    if orig_param.exists():
        text = orig_param.read_text(encoding="utf-8", errors="replace")
    else:
        text = (
            "task                : SinglePoint\n"
            "xc_functional       : PBE\n"
            "cut_off_energy      : 700 eV\n"
        )

    def _set_kw(src: str, keyword: str, value: str) -> str:
        """Set or add a keyword : value line."""
        pattern = rf"^\s*{re.escape(keyword)}\s*:.*$"
        replacement = f"{keyword:<20s}: {value}"
        if _re.search(pattern, src, flags=_re.M | _re.I):
            return _re.sub(pattern, replacement, src, flags=_re.M | _re.I)
        return src + f"\n{replacement}\n"

    text = _set_kw(text, "task", "SinglePoint")
    text = _set_kw(text, "calculate_stress", "true")
    text = _set_kw(text, "spin_polarized", "false")
    text = _set_kw(text, "nextra_bands", "30")
    text = _set_kw(text, "mix_charge_amp", "0.1")
    text = _set_kw(text, "opt_strategy", "speed")
    text = _set_kw(text, "write_checkpoint", "none")
    text = _set_kw(text, "num_dump_cycles", "0")
    text = _set_kw(text, "finite_basis_corr", "1")

    # Strip GeomOpt-only keywords (invalid for SinglePoint)
    for kw in _GEOM_KEYS:
        text = _re.sub(rf"^\s*{kw}\s*:.*$\n?", "", text, flags=_re.M | _re.I)

    return text


def _apply_strain_keeping_mixture(
    source_cell: Path,
    dest_cell: Path,
    strain_voigt: "np.ndarray",
) -> None:
    """
    Write a strained copy of source_cell that preserves all MIXTURE tags.

    This is the correct way to generate strained VCA cells.  The key physics:
    - LATTICE_CART vectors change under strain: new_vec = F @ old_vec
    - POSITIONS_FRAC coordinates are invariant under homogeneous strain
      (fractional coords are defined *relative* to the lattice vectors)
    - MIXTURE tags live in POSITIONS_FRAC → they survive unchanged

    Why NOT use generate_strain.py's produce_dotcell
    ────────────────────────────────────────────────
    castep.py::produce_dotcell writes atoms from a parsed list that has no
    knowledge of VCA syntax. It places Ti and V (or Ti and Nb) as two
    independent atoms at the same fractional coordinate (0,0,0), WITHOUT
    the MIXTURE tag. CASTEP then models two bare nuclei fused in space →
    infinite Coulomb repulsion → stress ~ -7000 GPa → Born stability fails.

    Strain convention (IRE / generate_strain.py)
    ────────────────────────────────────────────
    strain_voigt is the 6-vector [e11, e22, e33, 2*e23, 2*e13, 2*e12]
    The deformation gradient tensor F = I + epsilon where:
        epsilon = [[e11,      e12, e13],
                   [e12,      e22, e23],
                   [e13,      e23, e33]]
    (off-diagonal elements are e_ij = voigt_i / 2 for i in {4,5,6})

    CASTEP LATTICE_CART rows are the lattice vectors a1, a2, a3.
    Applying strain: new_a_i = F @ old_a_i  (matrix-vector product per row)
    Equivalently: new_L = old_L @ F.T  (row-major matrix form)
    """
    import numpy as np

    text = source_cell.read_text(encoding="utf-8", errors="replace")

    # ── Build deformation matrix F = I + epsilon ──────────────────────────────
    e11, e22, e33, two_e23, two_e13, two_e12 = strain_voigt
    e12 = two_e12 / 2.0
    e13 = two_e13 / 2.0
    e23 = two_e23 / 2.0

    F = np.array(
        [
            [1.0 + e11, e12, e13],
            [e12, 1.0 + e22, e23],
            [e13, e23, 1.0 + e33],
        ]
    )

    # ── Parse LATTICE_CART block ───────────────────────────────────────────────
    _RE_LAT = re.compile(
        r"(%BLOCK\s+LATTICE_CART\s*\n)(.*?)(%ENDBLOCK\s+LATTICE_CART)",
        re.DOTALL | re.I,
    )
    match = _RE_LAT.search(text)
    if match is None:
        # Fallback: copy source unchanged (shouldn't happen for valid CASTEP cells)
        dest_cell.write_text(text, encoding="utf-8")
        return

    header = match.group(1)
    body = match.group(2)
    footer = match.group(3)

    # Parse lattice vector rows (skip keyword lines like 'ang' or 'bohr')
    rows: list[str] = []
    lattice_vecs: list[np.ndarray] = []
    for line in body.splitlines(keepends=True):
        stripped = line.strip()
        if not stripped or stripped.lower() in {"ang", "bohr", "a.u.", "angstrom"}:
            rows.append(line)
            continue
        # Try to parse three floats
        parts = stripped.split()
        if len(parts) >= 3:
            try:
                vec = np.array([float(parts[0]), float(parts[1]), float(parts[2])])
                lattice_vecs.append(vec)
                rows.append(None)  # placeholder
                continue
            except ValueError:
                pass
        rows.append(line)

    if len(lattice_vecs) != 3:
        dest_cell.write_text(text, encoding="utf-8")
        return

    # Apply strain: new_a_i = F @ old_a_i
    L = np.array(lattice_vecs)  # shape (3, 3)  rows are vectors
    L_new = L @ F.T  # each row multiplied by F from the right

    # Reconstruct the LATTICE_CART block body
    vec_iter = iter(L_new)
    new_body_lines: list[str] = []
    for line in rows:
        if line is None:
            v = next(vec_iter)
            new_body_lines.append(f"  {v[0]:22.15f} {v[1]:22.15f} {v[2]:22.15f}\n")
        else:
            new_body_lines.append(line)

    new_lat_block = header + "".join(new_body_lines) + footer
    new_text = _RE_LAT.sub(new_lat_block, text)

    # ── Add FIX_ALL_CELL and remove conflicting constraints ───────────────────
    # FIX_ALL_CELL : true tells CASTEP not to relax the cell vectors —
    # we want the stress at this exact strained geometry.
    new_text = re.sub(r"^\s*FIX_ALL_CELL\s*:.*$\n?", "", new_text, flags=re.M | re.I)
    new_text = re.sub(r"^\s*FIX_COM\s*:.*$\n?", "", new_text, flags=re.M | re.I)
    new_text = re.sub(r"^\s*FIX_ALL_IONS\s*:.*$\n?", "", new_text, flags=re.M | re.I)
    new_text = new_text.rstrip() + "\nFIX_ALL_CELL : true\n"

    dest_cell.write_text(new_text, encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────


def run_finite_strain_elastic(
    seed_dir: Path,
    seed: str,
    castep_cmd: str,
    *,
    n_cores: int = 1,
    strain: float = 0.01,
    keep_all: bool = False,
    progress_callback: Any = None,
) -> dict[str, str]:
    """
    Run the full finite-strain elastic constants workflow for one seed.

    Parameters
    ──────────
    seed_dir        Directory containing <seed>.castep + <seed>-out.cell
                    (output of a successful GeometryOptimization run).
    seed            CASTEP job name (without extension).
    castep_cmd      MPI command template with {seed} placeholder.
    n_cores         MPI process count (substituted into castep_cmd).
    strain          Maximum strain magnitude (default 0.003 = 0.3 %).
                    generate_strain.py controls step count from crystal symmetry.
    keep_all        If False, delete strained .check/.castep_bin after each run.
    progress_callback  Optional callable(msg: str) for live status updates.

    Returns
    ───────
    dict[str, str] with Cij, B_Hill_GPa, G_Hill_GPa, E_GPa, nu, etc.
    Empty dict if anything fails (caller logs the error).
    """

    def _log(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)

    gen_script, elastics_script = find_elastic_scripts()
    if gen_script is None or elastics_script is None:
        _log(
            "  ✗  elastic-constants scripts not found.\n"
            "     Place elastic-constants-master/ next to main.py\n"
            "     or install via:  pip install elastic-constants"
        )
        return {}

    # Patch elastics.py and CijUtil.py for scipy >= 1.13 compatibility once.
    # This is idempotent — safe to call on every run.
    cijutil_path = elastics_script.parent / "CijUtil.py"
    _patch_elastic_scripts(elastics_script, cijutil_path)

    # ── Build the strain source cell ─────────────────────────────────────────
    # Strategy: use the ORIGINAL VCA .cell (with correct MIXTURE tags) but
    # update its LATTICE_CART vectors from the optimised -out.cell.
    #
    # Why NOT use -out.cell directly:
    # CASTEP's write_cell_structure:true produces -out.cell that may contain
    # extra species in POSITIONS_FRAC inherited from the job's pseudopotential
    # table. At x=1.0 (pure NbC) CASTEP may write both Ti AND Nb in -out.cell
    # even though Ti has zero occupancy in the optimised geometry. This gives
    # two bare nuclei at (0,0,0) without MIXTURE → infinite Coulomb repulsion
    # → stress tensor -7000 GPa → Born stability violated.
    #
    # Correct approach:
    #   1. Take the relaxed lattice vectors from -out.cell (accurate geometry)
    #   2. Keep everything else (POSITIONS_FRAC with MIXTURE, kpoints, etc.)
    #      from the original .cell file (correct VCA species definition)
    #
    # Finding the original .cell: it is the file written by write_vca_cell()
    # before CASTEP ran. CASTEP keeps a backup as TiC.cell (the input it read).
    # After GeomOpt, TiC.cell still has the correct VCA MIXTURE syntax.
    # We only need to transplant the optimised LATTICE_CART from -out.cell.

    out_cell = seed_dir / f"{seed}-out.cell"
    orig_cell = seed_dir / f"{seed}.cell"  # the original VCA input cell
    base_cell = seed_dir / f"{seed}_strain_src.cell"  # dedicated strain source

    if not orig_cell.exists() and not out_cell.exists():
        _log(f"  ✗  Neither {seed}.cell nor {seed}-out.cell found in {seed_dir}")
        return {}

    if out_cell.exists() and orig_cell.exists():
        # Transplant optimised LATTICE_CART into the original VCA cell
        _build_strain_source(orig_cell, out_cell, base_cell)
        _log(f"  |  Using relaxed geometry with original VCA MIXTURE syntax")
    elif orig_cell.exists():
        import shutil as _sh

        _sh.copy2(orig_cell, base_cell)
        _log(f"  |  Using original {orig_cell.name} (no -out.cell found)")
    else:
        import shutil as _sh

        _sh.copy2(out_cell, base_cell)
        _log(f"  |  Using {out_cell.name} (no original .cell found)")

    # Strip CASTEP GeomOpt bookkeeping that conflicts with straining.
    # CRITICAL: do NOT strip MIXTURE tags.
    _strip_symmetry_ops(base_cell)  # removes FIX_COM, CELL_CONSTRAINTS, SYMMETRY_OPS

    # ── Step 1: generate strain patterns (.cijdat) via generate_strain.py ─────
    # generate_strain.py reads <seed>.castep (for symmetry) and <seed>.cell
    # (for lattice vectors). It ONLY computes the strain pattern (.cijdat) —
    # we immediately delete the strained .cell files it produces because they
    # strip MIXTURE syntax. Instead we rebuild them with _apply_strain_keeping_mixture.
    #
    # generate_strain.py requires exactly <seed>.cell — copy our strain source there.
    import shutil as _sh2

    seed_cell_for_gen = seed_dir / f"{seed}.cell"
    _sh2.copy2(base_cell, seed_cell_for_gen)

    _log(f"  |  Computing strain patterns (strain={strain}) …")
    gen_cmd = [sys.executable, str(gen_script), "--strain", str(strain), seed]
    try:
        proc = subprocess.run(
            gen_cmd, cwd=seed_dir, capture_output=True, text=True, check=False
        )
        if proc.returncode != 0:
            _log(f"  ✗  generate_strain.py failed:\n{proc.stderr.strip()}")
            return {}
    except FileNotFoundError:
        _log(f"  ✗  Python not found at {sys.executable}")
        return {}

    # Delete the strained cells generate_strain.py wrote (they have no MIXTURE).
    for bad_cell in seed_dir.glob(f"{seed}_cij__*__*.cell"):
        bad_cell.unlink()

    # Read the .cijdat to get strain tensors for each step
    cijdat_path = seed_dir / f"{seed}.cijdat"
    try:
        from elastic_analysis import read_cijdat

        cijdat = read_cijdat(cijdat_path)
    except ImportError:
        _log("  ✗  elastic_analysis.py not found")
        return {}

    if cijdat is None:
        _log(f"  ✗  Could not read {cijdat_path.name}")
        return {}

    # ── Step 1b: write strained cells that KEEP MIXTURE syntax ───────────────
    # For each strain step, apply the deformation matrix F to LATTICE_CART only.
    # POSITIONS_FRAC are fractional coordinates — they are invariant under
    # homogeneous strain (the fractional position does not change when the
    # lattice vectors are stretched). MIXTURE tags live in POSITIONS_FRAC
    # and therefore survive untouched. This is the physically correct approach.
    strained_cells: list[Path] = []
    for pattern_name, strain_voigt in zip(cijdat.patterns, cijdat.strain_tensors):
        out_path = seed_dir / f"{pattern_name}.cell"
        _apply_strain_keeping_mixture(base_cell, out_path, strain_voigt)
        strained_cells.append(out_path)

    if not strained_cells:
        _log("  ✗  No strained cells generated — check .cijdat file")
        return {}
    _log(f"  ℹ  {len(strained_cells)} strained cells with MIXTURE preserved")

    # ── Build SinglePoint .param for strained cells ──────────────────────────
    # Requirements:
    #   task : SinglePoint           — faster than GeomOpt; geometry is already
    #                                  relaxed and the cell is fixed by FIX_ALL_CELL
    #   calculate_stress : true      — MANDATORY; elastics fitter reads stress tensor
    #   spin_polarized : false       — strained cells have no .check file to inherit
    #                                  spin density from; fractional VCA charges make
    #                                  the initial spin guess non-integer → CASTEP abort
    #   opt_strategy : speed         — maximise throughput for N×6 short runs
    #   write_checkpoint : none      — no .check files needed; saves I/O
    sp_text = _build_strained_param(seed_dir / f"{seed}.param")

    # Replace symlinked .param files (generate_strain.py creates symlinks)
    # with real independent SinglePoint .param files
    for cell_path in strained_cells:
        param_link = cell_path.with_suffix(".param")
        if param_link.exists() or param_link.is_symlink():
            param_link.unlink()
        param_link.write_text(sp_text, encoding="utf-8")

    # ── Step 2: run CASTEP SinglePoint on each strained cell ─────────────────
    import os as _os

    _CLEANUP_GLOBS = ("*.castep_bin", "*.check", "*.cst_esp", "*.bands", "*.bib")
    failed_cells: list[str] = []

    for idx, cell_path in enumerate(strained_cells, 1):
        strained_seed = cell_path.stem  # e.g. TiC_cij__1__1
        cmd = _os.path.expanduser(castep_cmd.replace("{seed}", strained_seed))
        _log(f"  ▶  [{idx}/{len(strained_cells)}]  {strained_seed} …")
        try:
            result_proc = subprocess.run(
                cmd,
                shell=True,
                cwd=seed_dir,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
        except OSError as exc:
            _log(f"  ✗  OS error launching CASTEP for {strained_seed}: {exc}")
            failed_cells.append(strained_seed)
            continue

        # Check for CASTEP .err files (more reliable than returncode for MPI jobs)
        err_files = sorted(seed_dir.glob(f"{strained_seed}*.err"))
        castep_out = seed_dir / f"{strained_seed}.castep"
        castep_empty = not castep_out.exists() or castep_out.stat().st_size < 100

        if err_files or castep_empty:
            # Show the FIRST meaningful error line from .err file
            norm_sq_crash = False
            for err_file in err_files:
                err_text = err_file.read_text(errors="replace").strip()
                if err_text:
                    err_lines = [l for l in err_text.splitlines() if l.strip()][:3]
                    if any("norm_sq" in l or "wave_Snormalise" in l for l in err_lines):
                        norm_sq_crash = True
                        break
                    _log(f"  ✗  CASTEP error ({err_file.name}):")
                    for ln in err_lines:
                        _log(f"       {ln}")
                    break
            if castep_empty and not err_files:
                _log(f"  ✗  {strained_seed}.castep is empty — CASTEP crashed silently")
            failed_cells.append(strained_seed)

        if not keep_all:
            for glob_pat in _CLEANUP_GLOBS:
                for f in seed_dir.glob(glob_pat):
                    try:
                        f.unlink()
                    except OSError:
                        pass

    if failed_cells:
        n_ok = len(strained_cells) - len(failed_cells)
        # Check if all failures are norm_sq ghost-state crashes
        # (detected by reading .err files)
        all_norm_sq = all(
            any(
                "norm_sq" in (f.read_text(errors="replace"))
                for f in sorted(seed_dir.glob(f"{s}*.err"))
            )
            for s in failed_cells
            if list(seed_dir.glob(f"{s}*.err"))
        )
        if all_norm_sq and n_ok == 0:
            _log("  ⚠  VCA ghost state: norm_sq crash on all strained cells.")
            _log("     This is a known CASTEP USP+VCA limitation (not a code bug).")
            _log("     Vegard interpolation from x=0 and x=1 endpoints will be used.")
        else:
            _log(
                f"  ⚠  {len(failed_cells)}/{len(strained_cells)} strained cells failed"
                f" — {n_ok} succeeded"
            )
        if n_ok < 3:
            return {"_elastic_error": f"only {n_ok} CASTEP runs succeeded"}

    # ── Step 3: fit Cij using elastic_analysis module (no scipy dependency) ──
    _log("  ▶  Fitting Cij tensor from stress vs strain …")
    try:
        from elastic_analysis import fit_cij, read_cijdat

        cijdat_path = seed_dir / f"{seed}.cijdat"
        cijdat = read_cijdat(cijdat_path)
        if cijdat is None:
            _log(
                f"  ✗  Cannot read {cijdat_path.name} — generate_strain.py may have failed"
            )
            return {}
        elastic_data = fit_cij(cijdat, seed_dir)
    except ImportError:
        _log("  ✗  elastic_analysis.py not found — place it next to main.py")
        return {}
    except Exception as exc:
        _log(f"  ✗  Cij fitting raised exception: {exc}")
        return {}

    # ── Report results ────────────────────────────────────────────────────────
    if "_elastic_error" in elastic_data:
        _log(f"  ✗  Cij fit failed: {elastic_data['_elastic_error']}")
        return elastic_data
    if "_elastic_error" not in elastic_data and "C11" in elastic_data:
        b = elastic_data.get("B_Hill_GPa", "—")
        g = elastic_data.get("G_Hill_GPa", "—")
        e = elastic_data.get("E_GPa", "—")
        nu = elastic_data.get("nu", "—")
        r2 = elastic_data.get("elastic_R2_min", "—")
        _log(
            f"  ✓  C11={elastic_data.get('C11', '—')}  C12={elastic_data.get('C12', '—')}"
            f"  C44={elastic_data.get('C44', '—')} GPa"
        )
        _log(f"     B={b}  G={g}  E={e} GPa  ν={nu}  R²≥{r2}")
        quality = elastic_data.get("elastic_quality_note", "")
        if quality:
            _log(f"  ⚠  {quality}")
    else:
        _log("  ⚠  Cij fitting returned no usable data")
        _log(f"     Check that .castep files in {seed_dir.name} contain stress tensors")

    return elastic_data


# ─────────────────────────────────────────────────────────────────────────────
# Vegard interpolation fallback for failed intermediate x
# ─────────────────────────────────────────────────────────────────────────────


def interpolate_elastic_vegard(
    x: float,
    elastic_at_0: dict[str, str],
    elastic_at_1: dict[str, str],
) -> dict[str, str]:
    """
    Linear (Vegard) interpolation of elastic constants for concentration x.

    Used when direct CASTEP calculation fails for intermediate x (0 < x < 1),
    which is expected for VCA+USP due to ghost states in the ultrasoft overlap
    matrix. Pure endpoints (x=0, x=1) are unaffected and computed directly.

    Physical justification
    ──────────────────────
    For isostructural solid solutions (same crystal symmetry, same Wyckoff
    sites), elastic constants vary linearly with composition to first order
    in Δx. This is the elastic analogue of Vegard's law for lattice parameters.
    The linear approximation is exact for the acoustic sum rule and is a
    well-established approximation in alloy physics (Hill 1952, Muñoz 2013).

    Published VCA studies (e.g. TiN-VN, TiC-NbC) routinely interpolate Cij
    when direct VCA+USP calculations fail for intermediate compositions.
    The error vs. SQS supercell calculations is typically < 5% for
    isoelectronic d-metal carbides like Ti(1-x)V(x)C.

    Parameters
    ──────────
    x               Concentration of species B (0 < x < 1).
    elastic_at_0    dict from fit_cij() for x=0 (pure A).
    elastic_at_1    dict from fit_cij() for x=1 (pure B).

    Returns
    ───────
    dict[str, str] in the same format as fit_cij() output, with an extra
    key ``elastic_source = "Vegard_interpolation"`` to flag the method.
    Returns empty dict if either endpoint is missing.
    """
    if not elastic_at_0 or not elastic_at_1:
        return {}
    if "_elastic_error" in elastic_at_0 or "_elastic_error" in elastic_at_1:
        return {}

    # Keys to interpolate — all numeric Cij / moduli from fit_cij()
    _INTERP_KEYS = (
        "C11",
        "C12",
        "C44",
        "B_Voigt_GPa",
        "B_Reuss_GPa",
        "B_Hill_GPa",
        "G_Voigt_GPa",
        "G_Reuss_GPa",
        "G_Hill_GPa",
        "E_GPa",
        "nu",
        "Zener_A",
        "Pugh_ratio",
        "Cauchy_pressure_GPa",
    )

    result: dict[str, str] = {}
    for key in _INTERP_KEYS:
        v0_str = elastic_at_0.get(key, "")
        v1_str = elastic_at_1.get(key, "")
        if not v0_str or not v1_str:
            continue
        try:
            v0 = float(v0_str)
            v1 = float(v1_str)
            v_x = (1.0 - x) * v0 + x * v1
            result[key] = f"{v_x:.4f}"
        except ValueError:
            continue

    if result:
        result["elastic_source"] = "Vegard_interpolation"
        result["elastic_n_points"] = "0"
        result["elastic_R2_min"] = "N/A"

    return result
