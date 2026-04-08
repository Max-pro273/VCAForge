"""
elastic_workflow.py  —  Finite-strain elastic constants for VCA systems.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Why this exists
───────────────
CASTEP's built-in `task : ElasticConstants` is blocked for VCA MIXTURE
atoms — DFPT-based strain response is incompatible with on-the-fly
ultrasoft pseudopotentials used for MIXTURE.

This module performs the equivalent via finite differences:
  1.  Generate strained copies of the relaxed cell (pure Python, no subprocess).
  2.  CASTEP SinglePoint × N_strains on each strained cell.
  3.  Fit C11 / C12 / C44 from stress vs strain by OLS.

All elastic math lives in core/elasticity.py.  No calls to external
generate_strain.py or elastics.py scripts.

Public API
──────────
  run_finite_strain_elastic(seed_dir, seed, castep_cmd, …) → dict
  interpolate_elastic_vegard(x, elastic_at_0, elastic_at_1)  → dict
"""

from __future__ import annotations

import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from castep_io import atom_count
from elasticity import (
    apply_strain_to_cell,
    fit_cij_from_stress,
    generate_strain_steps,
    nextra_bands_for,
    pointgroup_to_lattice_code,
    read_stress,
    standardize_cubic_cell,
    vec_for_concentration,
    write_cijdat,
)

# ─────────────────────────────────────────────────────────────────────────────
# Cell preprocessing helpers  (unchanged — CASTEP-specific fixes)
# ─────────────────────────────────────────────────────────────────────────────

_RE_MIXTURE = re.compile(r"\s+MIXTURE:\([^)]+\)", re.I)
_RE_SYMMETRY_OPS = re.compile(
    r"%BLOCK\s+SYMMETRY_OPS.*?%ENDBLOCK\s+SYMMETRY_OPS\s*", re.DOTALL | re.I
)


def _strip_geomopt_bookkeeping(cell_path: Path) -> None:
    """
    Remove CASTEP GeomOpt artefacts that conflict with strained SinglePoint runs:
      • SYMMETRY_OPS block (strained geometry violates stored ops → MPI_ABORT)
      • FIX_COM (conflicts with FIX_ALL_CELL written by apply_strain_to_cell)
      • FIX_ALL_IONS (same conflict)
      • CELL_CONSTRAINTS (conflicts with FIX_ALL_CELL)
    Does NOT touch MIXTURE tags.
    """
    text = cell_path.read_text(encoding="utf-8", errors="replace")
    cleaned = text

    cleaned = _RE_SYMMETRY_OPS.sub("", cleaned)
    cleaned = re.sub(
        r"#%BLOCK\s+SYMMETRY_OPS.*?#%ENDBLOCK\s+SYMMETRY_OPS\s*",
        "",
        cleaned,
        flags=re.DOTALL | re.I,
    )
    cleaned = re.sub(
        r"^#\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s*$\n?", "", cleaned, flags=re.M
    )
    cleaned = re.sub(r"^# Symm\..*$\n?", "", cleaned, flags=re.M)
    cleaned = re.sub(r"^\s*FIX_COM\s*:.*$\n?", "", cleaned, flags=re.M | re.I)
    cleaned = re.sub(r"^\s*FIX_ALL_IONS\s*:.*$\n?", "", cleaned, flags=re.M | re.I)
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
    Take LATTICE_CART from the relaxed -out.cell (optimised geometry) and
    everything else (POSITIONS_FRAC with MIXTURE, kpoints) from the original
    VCA .cell.  This avoids the ghost-species problem in CASTEP's -out.cell.
    """
    orig_text = orig_cell.read_text(encoding="utf-8", errors="replace")
    out_text = out_cell.read_text(encoding="utf-8", errors="replace")

    _RE_LAT = re.compile(
        r"(%BLOCK\s+LATTICE_CART\s*\n)(.*?)(%ENDBLOCK\s+LATTICE_CART)",
        re.DOTALL | re.I,
    )
    out_match = _RE_LAT.search(out_text)
    if out_match is None:
        dest.write_text(orig_text, encoding="utf-8")
        return

    new_lat = out_match.group(0)
    result = (
        _RE_LAT.sub(new_lat, orig_text)
        if _RE_LAT.search(orig_text)
        else new_lat + "\n" + orig_text
    )
    dest.write_text(result, encoding="utf-8")


def _parse_lattice_from_cell(cell_path: Path) -> np.ndarray | None:
    """
    Extract LATTICE_CART block from a .cell file.
    Returns (3, 3) array with rows = lattice vectors, or None on failure.
    """
    text = cell_path.read_text(encoding="utf-8", errors="replace")
    _RE_LAT = re.compile(
        r"%BLOCK\s+LATTICE_CART\s*\n(.*?)%ENDBLOCK\s+LATTICE_CART",
        re.DOTALL | re.I,
    )
    match = _RE_LAT.search(text)
    if not match:
        return None
    vecs: list[np.ndarray] = []
    for line in match.group(1).splitlines():
        parts = line.strip().split()
        if len(parts) >= 3:
            try:
                vecs.append(np.array([float(p) for p in parts[:3]]))
            except ValueError:
                continue
    return np.array(vecs) if len(vecs) == 3 else None


def _parse_lattice_code_from_castep(castep_path: Path) -> int | None:
    """Read point group index from a .castep output file; return lattice code or None."""
    if not castep_path.exists():
        return None
    try:
        text = castep_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    m = re.search(r"Point group of crystal\s*=\s*(\d+)", text, re.I)
    if m:
        try:
            return pointgroup_to_lattice_code(int(m.group(1)))
        except (ValueError, KeyError):
            pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# .param builder — adaptive nextra_bands, tight elastic tolerances
# ─────────────────────────────────────────────────────────────────────────────


def _build_strained_param(
    orig_param: Path,
    x: float = 0.5,
    vec: float = 8.5,
) -> str:
    """
    Build a SinglePoint .param for a strained cell.

    Key settings vs the original GeomOpt .param:

      task                 → SinglePoint
      calculate_stress     → true       (MANDATORY — elastics reads this)
      elec_energy_tol      → 1.0e-7 eV (tight; elastic constants sensitive to SCF)
      finite_basis_corr    → 1          (one correction pass; ~60% faster vs mode 2)
      nextra_bands         → adaptive   (core/elasticity.nextra_bands_for)
      spin_polarized       → false      (no .check to inherit; fractional VCA charges
                                         make initial spin non-integer → CASTEP abort)
      mix_charge_amp       → 0.1        (conservative; strained cells start cold)
      opt_strategy         → speed      (throughput for N×6 short runs)
      write_checkpoint     → none       (saves I/O; not needed for stress extraction)
      num_dump_cycles      → 0
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

    def _set(src: str, kw: str, val: str) -> str:
        pat = rf"^\s*{re.escape(kw)}\s*:.*$"
        rep = f"{kw:<24}: {val}"
        if _re.search(pat, src, flags=_re.M | _re.I):
            return _re.sub(pat, rep, src, flags=_re.M | _re.I)
        return src + f"\n{rep}\n"

    nextra = nextra_bands_for(x, vec)

    text = _set(text, "task", "SinglePoint")
    text = _set(text, "calculate_stress", "true")
    text = _set(text, "elec_energy_tol", "1.0e-7 eV")
    text = _set(text, "finite_basis_corr", "1")
    text = _set(text, "nextra_bands", str(nextra))
    text = _set(text, "spin_polarized", "false")
    text = _set(text, "mix_charge_amp", "0.1")
    text = _set(text, "opt_strategy", "speed")
    text = _set(text, "write_checkpoint", "none")
    text = _set(text, "num_dump_cycles", "0")
    text = _set(text, "write_cell_structure", "false")  # suppress -out.cell

    for kw in _GEOM_KEYS:
        text = _re.sub(rf"^\s*{kw}\s*:.*$\n?", "", text, flags=_re.M | _re.I)

    return text


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────

_CLEANUP_GLOBS = ("*.castep_bin", "*.check", "*.cst_esp", "*.bands", "*.bib")


def run_finite_strain_elastic(
    seed_dir: Path,
    seed: str,
    castep_cmd: str,
    *,
    x: float = 0.5,
    species_a: str = "Ti",
    species_b: str = "Nb",
    nonmetal: str | None = "C",
    strain: float = 0.003,
    n_steps: int = 3,
    keep_all: bool = False,
    progress_callback: Callable[[str], None] | None = None,
    density_gcm3: float | None = None,
    volume_ang3: float | None = None,
) -> dict[str, Any]:
    """
    Run the full finite-strain elastic constants workflow for one seed.

    Parameters
    ──────────
    seed_dir         Directory with <seed>.castep + <seed>-out.cell from GeomOpt.
    seed             CASTEP job name (no extension).
    castep_cmd       Command template with {seed} placeholder.
    x                VCA concentration — used for adaptive nextra_bands.
    species_a/b      Metal species — used for VEC-based nextra_bands.
    nonmetal         Non-metal (e.g. "C") — used for VEC.
    strain           Maximum strain magnitude (default 0.003 = 0.3 %).
    n_steps          Number of positive strain magnitudes per pattern.
    keep_all         If False, strained .check/.bands files are deleted after fitting.
    progress_callback  Optional callable for live status messages.

    Returns
    ───────
    dict with C11, C12, C44, B_Hill_GPa, G_Hill_GPa, E_GPa, nu, …
    or {"_elastic_error": "<message>"} on failure — never raises.
    """
    t_start = time.monotonic()

    def _log(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)

    # ── VEC for adaptive nextra_bands ─────────────────────────────────────────
    vec = vec_for_concentration(species_a, species_b, x, nonmetal)

    # ── Build strain source cell ──────────────────────────────────────────────
    out_cell = seed_dir / f"{seed}-out.cell"
    orig_cell = seed_dir / f"{seed}.cell"
    base_cell = seed_dir / f"{seed}_strain_src.cell"

    if not orig_cell.exists() and not out_cell.exists():
        return {
            "_elastic_error": f"no {seed}.cell or {seed}-out.cell in {seed_dir.name}"
        }

    if out_cell.exists() and orig_cell.exists():
        _build_strain_source(orig_cell, out_cell, base_cell)
    elif orig_cell.exists():
        shutil.copy2(orig_cell, base_cell)
    else:
        shutil.copy2(out_cell, base_cell)

    _strip_geomopt_bookkeeping(base_cell)

    # ── Parse lattice vectors + lattice symmetry code ────────────────────────
    lattice_vecs = _parse_lattice_from_cell(base_cell)
    if lattice_vecs is None:
        return {"_elastic_error": "cannot parse LATTICE_CART from strain source cell"}

    castep_out = seed_dir / f"{seed}.castep"
    lattice_code = _parse_lattice_code_from_castep(castep_out) or 5  # default cubic

    # ── Generate strain steps (pure Python) ──────────────────────────────────
    # CRITICAL: standardize primitive FCC/BCC cells to orthogonal form
    # before computing strain magnitudes.  Primitive cells (e.g. from cif2cell)
    # have non-orthogonal vectors — straining them along Voigt axes gives
    # physically wrong stress-strain relationships → garbage Cij values.
    lattice_vecs_orth = standardize_cubic_cell(lattice_vecs)
    strain_steps = generate_strain_steps(
        lattice_vecs=lattice_vecs_orth,
        lattice_code=lattice_code,
        max_strain=strain,
        n_steps=n_steps,
    )

    # Write .cijdat (for diagnostics / external analysis)
    cijdat_path = seed_dir / f"{seed}.cijdat"
    write_cijdat(cijdat_path, lattice_code, n_steps, strain, seed, strain_steps)

    # ── Write strained .cell files (VCA MIXTURE preserved) ───────────────────
    strained_seeds: list[str] = []
    for step in strain_steps:
        cell_out = seed_dir / f"{seed}{step.name}.cell"
        apply_strain_to_cell(base_cell, cell_out, step.strain_voigt)
        strained_seeds.append(f"{seed}{step.name}")

    # ── Write .param files (adaptive nextra_bands, tight tolerance) ──────────
    sp_text = _build_strained_param(seed_dir / f"{seed}.param", x=x, vec=vec)
    for ss in strained_seeds:
        p = seed_dir / f"{ss}.param"
        if p.exists() or p.is_symlink():
            p.unlink()
        p.write_text(sp_text, encoding="utf-8")

    # ── Run CASTEP SinglePoint on each strained cell ─────────────────────────
    failed: list[str] = []

    for idx, ss in enumerate(strained_seeds, 1):
        cmd = castep_cmd.replace("{seed}", ss)
        try:
            subprocess.run(
                cmd,
                shell=True,
                cwd=seed_dir,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
        except OSError as exc:
            _log(f"  ✗  OS error running CASTEP for {ss}: {exc}")
            failed.append(ss)
            continue

        castep_log = seed_dir / f"{ss}.castep"
        err_files = sorted(seed_dir.glob(f"{ss}*.err"))
        empty = not castep_log.exists() or castep_log.stat().st_size < 100

        if err_files or empty:
            for ef in err_files:
                err_lines = [
                    l for l in ef.read_text(errors="replace").splitlines() if l.strip()
                ][:2]
                for ln in err_lines:
                    _log(f"  ✗  [{ss}] {ln}")
            if empty and not err_files:
                _log(f"  ✗  {ss}.castep is empty — CASTEP crashed silently")
            failed.append(ss)

        if not keep_all:
            for pat in _CLEANUP_GLOBS:
                prefix = ss
                for f in seed_dir.glob(f"{prefix}{pat.lstrip('*')}"):
                    try:
                        f.unlink()
                    except OSError:
                        pass

    n_ok = len(strained_seeds) - len(failed)
    if n_ok < 3:
        return {
            "_elastic_error": f"only {n_ok}/{len(strained_seeds)} CASTEP runs succeeded"
        }

    # Read n_atoms from the GeomOpt .castep (not strained) for Debye calculation
    n_atoms = atom_count(castep_out)

    # ── Collect stress / strain pairs ─────────────────────────────────────────
    stresses: list[np.ndarray] = []
    strains: list[np.ndarray] = []
    missing: list[str] = []

    for ss, step in zip(strained_seeds, strain_steps):
        sv = read_stress(seed_dir / f"{ss}.castep")
        if sv is None:
            missing.append(ss)
        else:
            stresses.append(sv)
            strains.append(step.strain_voigt)

    if missing:
        n_miss = len(missing)
        _log(
            f"  ⚠  stress missing from {n_miss} cell(s): "
            + ", ".join(missing[:3])
            + (f" (+{n_miss - 3} more)" if n_miss > 3 else "")
        )

    # ── Fit Cij ───────────────────────────────────────────────────────────────
    elastic_data = fit_cij_from_stress(
        stresses,
        strains,
        lattice_code,
        density_gcm3=density_gcm3,
        n_atoms_per_cell=n_atoms if n_atoms > 0 else None,
        volume_ang3=volume_ang3,
    )

    if "error" in elastic_data:
        return {"_elastic_error": elastic_data["error"]}

    # ── Post-fit cleanup: remove ALL intermediate elastic files ─────────────
    # Kept after fitting (diagnostics):  <seed>.cijdat, <seed>_cij__*.castep
    # Removed always:  strained .cell, .param, -out.cell, _strain_src.cell
    # Removed unless --keep-all:  *.geom (trajectory)
    if not keep_all:
        for ss in strained_seeds:
            for ext in (".cell", ".param", "-out.cell"):
                try:
                    (seed_dir / f"{ss}{ext}").unlink(missing_ok=True)
                except OSError:
                    pass
        # strain_src cell — internal working copy, not needed after fitting
        try:
            base_cell.unlink(missing_ok=True)
        except OSError:
            pass
        # .geom files written by CASTEP SinglePoint (not useful for stress calc)
        for f in seed_dir.glob("*.geom"):
            try:
                f.unlink()
            except OSError:
                pass

    t_elapsed = time.monotonic() - t_start
    elastic_data["elastic_wall_time_s"] = f"{t_elapsed:.0f}"
    return elastic_data


# ─────────────────────────────────────────────────────────────────────────────
# Vegard interpolation fallback
# ─────────────────────────────────────────────────────────────────────────────

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


def interpolate_elastic_vegard(
    x: float,
    elastic_at_0: dict[str, Any],
    elastic_at_1: dict[str, Any],
) -> dict[str, Any]:
    """
    Linear (Vegard) interpolation of elastic constants for concentration x.

    Used when direct CASTEP calculation fails for intermediate x (0 < x < 1),
    which is expected for VCA+USP due to ghost states.  Pure endpoints are
    computed directly and are reliable.

    Physical justification
    ──────────────────────
    For isostructural solid solutions (same crystal symmetry), elastic constants
    vary linearly with composition to first order.  Published VCA studies
    (TiN-VN, TiC-NbC) routinely use this approximation (error < 5%).
    """
    if not elastic_at_0 or not elastic_at_1:
        return {}
    if "_elastic_error" in elastic_at_0 or "_elastic_error" in elastic_at_1:
        return {}

    result: dict[str, Any] = {}
    for key in _INTERP_KEYS:
        v0, v1 = elastic_at_0.get(key, ""), elastic_at_1.get(key, "")
        if not v0 or not v1:
            continue
        try:
            result[key] = f"{(1.0 - x) * float(v0) + x * float(v1):.4f}"
        except ValueError:
            continue

    if result:
        result["elastic_source"] = "Vegard_interpolation"
        result["elastic_n_points"] = "0"
        result["elastic_R2_min"] = "N/A"

    return result
