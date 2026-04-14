"""
castep/elastic.py  —  Finite-strain elastic-constants workflow.

Public API
──────────
  run_elastic(seed_dir, seed, castep_cmd, *, ...) -> dict
"""

from __future__ import annotations

import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

import config as _top_cfg
import core_physics as _phys
from castep import config as _cfg
from castep.io import (
    _parse_lattice,   # private — same package
    _RE_LAT,
    count_atoms,
    read_stress,
    sp_param_content,
)

# ─────────────────────────────────────────────────────────────────────────────
# Cell preparation helpers
# ─────────────────────────────────────────────────────────────────────────────

_RE_SYMM_OPS = re.compile(
    r"%BLOCK\s+SYMMETRY_OPS.*?%ENDBLOCK\s+SYMMETRY_OPS\s*",
    re.DOTALL | re.I,
)


def _prep_strain_source(orig_cell: Path, out_cell: Path, dest: Path) -> None:
    """Merge relaxed geometry into the VCA cell template.

    Takes ``LATTICE_CART`` from *out_cell* (relaxed geometry) and
    ``POSITIONS_FRAC`` + ``MIXTURE`` tags from *orig_cell* (the VCA cell
    written by :func:`~castep.io.write_vca_cell`).  This avoids CASTEP
    ghost-species errors that appear when strained cells are built from
    the bare ``-out.cell``.

    Args:
        orig_cell: Original VCA ``.cell`` with MIXTURE syntax.
        out_cell:  Geometry-optimised ``-out.cell`` with updated lattice.
        dest:      Destination path for the merged strain-source cell.
    """
    orig_t = orig_cell.read_text(encoding="utf-8", errors="replace")
    out_t = out_cell.read_text(encoding="utf-8", errors="replace")
    m = _RE_LAT.search(out_t)
    if m is None:
        dest.write_text(orig_t, encoding="utf-8")
        return
    new_lat = m.group(0)
    result = (
        _RE_LAT.sub(new_lat, orig_t)
        if _RE_LAT.search(orig_t)
        else new_lat + "\n" + orig_t
    )
    dest.write_text(result, encoding="utf-8")


def _strip_geomopt_tags(cell_path: Path) -> None:
    """Remove GeomOpt-only blocks/keywords from a cell file in-place.

    Strips ``SYMMETRY_OPS``, ``CELL_CONSTRAINTS``, ``FIX_COM``, and
    ``FIX_ALL_IONS`` so that the resulting file is valid for SinglePoint.
    """
    text = cell_path.read_text(encoding="utf-8", errors="replace")
    cleaned = _RE_SYMM_OPS.sub("", text)
    for pat in (
        r"#%BLOCK\s+SYMMETRY_OPS.*?#%ENDBLOCK\s+SYMMETRY_OPS\s*",
        r"^#\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s*$\n?",
        r"^# Symm\..*$\n?",
        r"^\s*FIX_COM\s*:.*$\n?",
        r"^\s*FIX_ALL_IONS\s*:.*$\n?",
    ):
        cleaned = re.sub(pat, "", cleaned, flags=re.M | re.I | re.DOTALL)
    cleaned = re.sub(
        r"%BLOCK\s+CELL_CONSTRAINTS.*?%ENDBLOCK\s+CELL_CONSTRAINTS\s*",
        "",
        cleaned,
        flags=re.DOTALL | re.I,
    )
    if cleaned != text:
        cell_path.write_text(cleaned, encoding="utf-8")


def _write_strained_cell(base: Path, dest: Path, voigt: np.ndarray) -> None:
    """Write a strained copy of *base* by applying the Voigt strain tensor.

    Only ``LATTICE_CART`` is modified — ``POSITIONS_FRAC`` (including any
    ``MIXTURE`` tags) is invariant under homogeneous strain.

    Args:
        base:   Source cell file.
        dest:   Destination path.
        voigt:  6-component Voigt strain ``[e11, e22, e33, e23, e13, e12]``.
    """
    text = base.read_text(encoding="utf-8", errors="replace")
    match = _RE_LAT.search(text)
    if match is None:
        dest.write_text(text, encoding="utf-8")
        return

    e11, e22, e33 = voigt[0], voigt[1], voigt[2]
    e23, e13, e12 = voigt[3] / 2, voigt[4] / 2, voigt[5] / 2
    F = np.array(
        [[1 + e11, e12, e13], [e12, 1 + e22, e23], [e13, e23, 1 + e33]]
    )

    header, body, footer = match.group(1), match.group(2), match.group(3)
    rows: list[str | None] = []
    vecs: list[np.ndarray] = []
    for line in body.splitlines(keepends=True):
        s = line.strip()
        if not s or s.lower() in {"ang", "bohr", "a.u.", "angstrom"}:
            rows.append(line)
            continue
        parts = s.split()
        if len(parts) >= 3:
            try:
                rows.append(None)
                vecs.append(np.array([float(p) for p in parts[:3]]))
                continue
            except ValueError:
                pass
        rows.append(line)

    if len(vecs) != 3:
        dest.write_text(text, encoding="utf-8")
        return

    L_new = np.array(vecs) @ F.T
    vi = iter(L_new)
    new_body_lines = []
    for line in rows:
        if line is None:
            v = next(vi)
            new_body_lines.append(f"  {v[0]:22.15f} {v[1]:22.15f} {v[2]:22.15f}\n")
        else:
            new_body_lines.append(line)

    new_block = header + "".join(new_body_lines) + footer
    new_text = _RE_LAT.sub(new_block, text)
    new_text = re.sub(r"^\s*FIX_ALL_CELL\s*:.*$\n?", "", new_text, flags=re.M | re.I)
    new_text = re.sub(r"^\s*FIX_COM\s*:.*$\n?", "", new_text, flags=re.M | re.I)
    new_text = new_text.rstrip() + "\nFIX_ALL_CELL : true\n"
    dest.write_text(new_text, encoding="utf-8")


def _lattice_code_from_geometry(L: np.ndarray) -> int:
    """Determine the strain-pattern code purely from lattice geometry.

    VCA calculations run under P1 symmetry (CASTEP reports point group = 1),
    so CASTEP's own point-group index is useless for choosing how many
    independent elastic strain patterns are needed.  Instead, we inspect the
    lattice metric directly.

    Recognition rules (by lattice parameters):
      - All sides equal, all angles ≈ 60° (FCC primitive)   → cubic (5)
      - All sides equal, all angles ≈ 109.47° (BCC prim.)   → cubic (5)
      - All sides equal, all angles equal ≈ 90°              → cubic (5)
      - Two sides equal, two angles equal ≈ 90°, one ≠ 90°   → tetragonal (4)
      - All angles ≈ 90°, sides not all equal                → orthorhombic (3)
      - All sides equal, angles equal but not 60/90/109.47°  → trigonal (6)
      - Default                                              → triclinic (1)

    Args:
        L: 3×3 lattice matrix (rows = vectors, Angstrom).

    Returns:
        Lattice symmetry code 1-7 matching ``_STRAIN_PATTERNS``.
    """
    a, b, c, al, be, ga = _phys.lattice_to_abc(L)

    def _eq(x: float, y: float, tol: float = 0.5) -> bool:
        return abs(x - y) < tol

    # ── Cubic: equal sides, characteristic angle ──────────────────────────
    sides_equal = _eq(a, b) and _eq(b, c)
    ang_avg = (al + be + ga) / 3.0
    if sides_equal:
        if _eq(ang_avg, 60.0,  1.5): return 5   # FCC primitive
        if _eq(ang_avg, 90.0,  1.5): return 5   # simple cubic / conventional
        if _eq(ang_avg, 109.47, 1.5): return 5  # BCC primitive
        # Rhombohedral / trigonal primitive (e.g. R-3m)
        angs_equal = _eq(al, be) and _eq(be, ga)
        if angs_equal:                 return 6  # trigonal

    # ── All angles ≈ 90° → orthogonal family ──────────────────────────────
    ortho = _eq(al, 90.0) and _eq(be, 90.0) and _eq(ga, 90.0)
    if ortho:
        if _eq(a, b) or _eq(b, c) or _eq(a, c):
            return 4   # tetragonal (two sides equal)
        return 3       # orthorhombic (all sides different)

    # ── Hexagonal: a=b≠c, α=β=90°, γ=120° ───────────────────────────────
    if _eq(a, b) and _eq(al, 90.0) and _eq(be, 90.0) and _eq(ga, 120.0, 1.5):
        return 7

    # ── Monoclinic: one angle ≠ 90°, other two ≈ 90° ─────────────────────
    n_right = sum(_eq(x, 90.0) for x in (al, be, ga))
    if n_right == 2:
        return 2

    return 1   # triclinic — most general


def _pg_to_lattice_code(castep_path: Path, L: np.ndarray | None = None) -> int:
    """Return the strain-pattern lattice code for this CASTEP run.

    Strategy:
    1. Parse the point-group index from ``castep_path``.
    2. If it is 1 (C1 / P1 — which CASTEP always reports for VCA cells because
       the MIXTURE syntax breaks crystal symmetry), fall back to geometric
       detection using the optimised lattice matrix *L*.
    3. If *L* is also unavailable, return 5 (cubic) as a safe default for
       the TMC / TMN carbide / nitride systems VCAForge is designed for.

    Args:
        castep_path: Path to the ``.castep`` output file.
        L:           Optimised lattice matrix (3×3, Angstrom) for geometric
                     fallback — usually read from ``{seed}-out.cell`` or
                     ``{seed}_strain_src.cell``.

    Returns:
        Lattice symmetry code 1-7.
    """
    if not castep_path.exists():
        return _lattice_code_from_geometry(L) if L is not None else 5

    try:
        text = castep_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return _lattice_code_from_geometry(L) if L is not None else 5

    m = re.search(r"Point group of crystal\s*=\s*(\d+)", text, re.I)
    if m:
        try:
            pg_code = int(m.group(1))
            # pg_code == 1 means C1 / P1 — almost always a VCA artefact.
            # The physical crystal is still cubic; use geometry instead.
            if pg_code > 1:
                return _phys.pointgroup_to_lattice_code(pg_code)
        except (ValueError, KeyError):
            pass

    # pg == 1 or unreadable → geometric fallback
    if L is not None:
        return _lattice_code_from_geometry(L)
    return 5   # safe default for TMC/TMN systems


# ─────────────────────────────────────────────────────────────────────────────
# Elastic progress monitor (background thread)
# ─────────────────────────────────────────────────────────────────────────────


def _elastic_progress_monitor(
    seed_dir: Path,
    strained: list[str],
    stop: threading.Event,
    progress_cb: Callable[[str], None] | None,
) -> None:
    """Background thread: display a live progress bar for elastic SinglePoint runs.

    Two pieces of information are shown on a single overwriting line:

    * **Outer** — strained cells fully completed (non-empty ``.castep``,
      no ``.err`` file, ``"Total time"`` present as a completion sentinel).
    * **Inner** — SCF cycle number of the *currently running* cell.

    Output format::

        │  [████████████░░░░░░░░░░]  4/6  scf 7  ⏱ 02:15

    Args:
        seed_dir:    Directory containing the strained CASTEP files.
        strained:    Ordered list of seed names to track.
        stop:        Event set by the caller when all runs are complete.
        progress_cb: Callback for non-interactive callers (receives the line).
    """
    _BAR    = 22
    n_total = len(strained)
    t0      = time.time()

    while not stop.is_set():
        # Completed = has a non-empty .castep with "Total time" (CASTEP sentinel)
        # and no .err file.
        done = 0
        current_scf = 0
        for ss in strained:
            castep_f = seed_dir / f"{ss}.castep"
            err_f    = list(seed_dir.glob(f"{ss}*.err"))
            if not castep_f.exists() or err_f:
                # This is either not started or failed — try to read SCF from it.
                if castep_f.exists() and not err_f:
                    try:
                        text = castep_f.read_text(errors="replace")
                        for ln in text.splitlines():
                            if "<-- SCF" in ln:
                                s = ln.lstrip()
                                if s and s[0].isdigit():
                                    try:
                                        current_scf = int(s.split()[0])
                                    except ValueError:
                                        pass
                    except OSError:
                        pass
                continue
            try:
                text = castep_f.read_text(errors="replace")
            except OSError:
                continue
            if "Total time" in text and castep_f.stat().st_size > 200:
                done += 1
            else:
                # Currently running — extract latest SCF cycle.
                for ln in text.splitlines():
                    if "<-- SCF" in ln:
                        s = ln.lstrip()
                        if s and s[0].isdigit():
                            try:
                                current_scf = int(s.split()[0])
                            except ValueError:
                                pass

        pct    = min(100, done * 100 // max(n_total, 1))
        filled = pct * _BAR // 100
        bar    = "█" * filled + "░" * (_BAR - filled)
        el     = int(time.time() - t0)
        mm, ss = divmod(el, 60)
        scf_tag = f"  scf {current_scf}" if current_scf > 0 else ""
        line = f"  │  [{bar}] {done}/{n_total}{scf_tag}  ⏱ {mm:02d}:{ss:02d}"

        if progress_cb:
            progress_cb(line)
        else:
            print(line + "    ", end="\r", flush=True)

        stop.wait(0.8)

    if not progress_cb:
        print(" " * 80, end="\r")


# ─────────────────────────────────────────────────────────────────────────────
# Public workflow entry point
# ─────────────────────────────────────────────────────────────────────────────


def run_elastic(
    seed_dir: Path,
    seed: str,
    castep_cmd: str,
    *,
    x: float = 0.5,
    species: list[tuple[str, float]] | None = None,
    nonmetal: str | None = None,
    strain: float = _top_cfg.ELASTIC_MAX_STRAIN,
    n_steps: int = _top_cfg.ELASTIC_N_STEPS,
    keep_all: bool = False,
    progress_cb: Callable[[str], None] | None = None,
    density_gcm3: float | None = None,
    volume_ang3: float | None = None,
) -> dict[str, Any]:
    """Run the finite-strain elastic-constants workflow for one seed directory.

    Workflow:
      1. Build the strain-source cell from ``{seed}-out.cell`` + ``{seed}.cell``.
      2. Generate all :class:`~core_physics.StrainStep` objects.
      3. Write strained ``.cell`` and ``.param`` files.
      4. Run CASTEP SinglePoint on each strained cell (with live progress bar).
      5. Collect stress tensors and fit Cij via OLS.

    Args:
        seed_dir:     Directory containing the GeomOpt results.
        seed:         CASTEP seed name (filename stem).
        castep_cmd:   Shell command template, with ``{seed}`` placeholder.
        x:            VCA concentration — determines ``nextra_bands``.
        species:      ``[(element, fraction), …]`` for VEC calculation.
        nonmetal:     Anion element (e.g. ``"C"``), or ``None``.
        strain:       Maximum strain magnitude (default ±0.3 %).
        n_steps:      Number of positive strain magnitudes per pattern.
        keep_all:     When ``False``, intermediate files are deleted on success.
        progress_cb:  Optional callback receiving each progress-bar string.
        density_gcm3: Density in g/cm³ (from GeomOpt, for Debye/sound speed).
        volume_ang3:  Cell volume in Å³ (from GeomOpt, for acoustic Grüneisen).

    Returns:
        Dict with ``C11``/``C12``/``C44`` and all VRH moduli, or
        ``{"_elastic_error": "…"}`` on failure.  Never raises.
    """
    t_start = time.monotonic()

    def _log(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    if species is None:
        species = []

    # ── VEC → adaptive nextra_bands ──────────────────────────────────────────
    sp_list = list(species)
    vec_val = _phys.vec_for_system(sp_list, nonmetal) if sp_list else 8.0

    # ── Strain-source cell ───────────────────────────────────────────────────
    orig_cell = seed_dir / f"{seed}.cell"
    out_cell = seed_dir / f"{seed}-out.cell"
    base_cell = seed_dir / f"{seed}_strain_src.cell"

    if not orig_cell.exists() and not out_cell.exists():
        return {
            "_elastic_error": (
                f"no {seed}.cell or {seed}-out.cell in {seed_dir.name}"
            )
        }

    if out_cell.exists() and orig_cell.exists():
        _prep_strain_source(orig_cell, out_cell, base_cell)
    elif orig_cell.exists():
        shutil.copy2(orig_cell, base_cell)
    else:
        shutil.copy2(out_cell, base_cell)
    _strip_geomopt_tags(base_cell)

    L = _parse_lattice(base_cell)
    if L is None:
        return {"_elastic_error": "cannot parse LATTICE_CART from strain source cell"}

    # ── Strain steps ─────────────────────────────────────────────────────────
    castep_out = seed_dir / f"{seed}.castep"
    # Pass the parsed lattice L for geometric fallback when CASTEP reports P1
    # (VCA MIXTURE syntax always disables symmetry detection 2192 point group = 1).
    lattice_code = _pg_to_lattice_code(castep_out, L)
    L_orth = _phys.standardize_cubic(L)
    strain_steps = _phys.generate_strain_steps(L_orth, lattice_code, strain, n_steps)

    # ── Write strained cells + params ────────────────────────────────────────
    strained: list[str] = []
    for step in strain_steps:
        cell_out = seed_dir / f"{seed}{step.name}.cell"
        _write_strained_cell(base_cell, cell_out, step.strain_voigt)
        strained.append(f"{seed}{step.name}")

    orig_param = seed_dir / f"{seed}.param"
    sp_text = sp_param_content(orig_param, x, vec_val)
    for ss in strained:
        p = seed_dir / f"{ss}.param"
        if p.exists() or p.is_symlink():
            p.unlink()
        p.write_text(sp_text, encoding="utf-8")

    # ── Run CASTEP SinglePoint with live progress bar ────────────────────────
    stop = threading.Event()
    monitor = threading.Thread(
        target=_elastic_progress_monitor,
        args=(seed_dir, strained, stop, progress_cb),
        daemon=True,
    )
    monitor.start()

    failed: list[str] = []
    for ss in strained:
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
            _log(f"  OS error for {ss}: {exc}")
            failed.append(ss)
            continue

        log = seed_dir / f"{ss}.castep"
        empty = not log.exists() or log.stat().st_size < 100
        errs = sorted(seed_dir.glob(f"{ss}*.err"))
        if errs or empty:
            for ef in errs:
                for ln in ef.read_text(errors="replace").splitlines()[:2]:
                    if ln.strip():
                        _log(f"  [{ss}] {ln}")
            if empty and not errs:
                _log(f"  {ss}.castep empty — CASTEP crashed")
            failed.append(ss)

        if not keep_all:
            for pat in _cfg.CLEANUP_GLOBS:
                for f in seed_dir.glob(f"{ss}{pat.lstrip('*')}"):
                    try:
                        f.unlink()
                    except OSError:
                        pass

    stop.set()
    monitor.join(2)

    n_ok = len(strained) - len(failed)
    if n_ok < 3:
        return {
            "_elastic_error": f"only {n_ok}/{len(strained)} CASTEP runs succeeded"
        }

    n_atoms = count_atoms(castep_out)

    # ── Collect stresses ─────────────────────────────────────────────────────
    stresses, strains_out, missing = [], [], []
    for ss, step in zip(strained, strain_steps):
        sv = read_stress(seed_dir / f"{ss}.castep")
        if sv is None:
            missing.append(ss)
        else:
            stresses.append(sv)
            strains_out.append(step.strain_voigt)
    if missing:
        _log(
            f"  stress missing from {len(missing)} cell(s): "
            + ", ".join(missing[:3])
        )

    # ── Fit Cij ──────────────────────────────────────────────────────────────
    result = _phys.fit_cij_cubic(
        stresses,
        strains_out,
        density_gcm3=density_gcm3,
        n_atoms=n_atoms if n_atoms > 0 else None,
        volume_ang3=volume_ang3,
    )
    if "error" in result:
        return {"_elastic_error": result["error"]}

    # ── Cleanup intermediate files ─────────────────────────────────────────────
    # Binary files removed unconditionally — never needed for analysis.
    _ELASTIC_BINARY_GLOBS = (
        ".castep_bin", ".check", ".cst_esp", ".bands",
        ".usp", ".upf", ".oepr",
    )
    for ss in strained:
        for suffix in _ELASTIC_BINARY_GLOBS:
            try: (seed_dir / f"{ss}{suffix}").unlink()
            except FileNotFoundError: pass

    if not keep_all:
        # Strained inputs, BibTeX, geom trajectory, strain source.
        for ss in strained:
            for ext in (".cell", ".param", "-out.cell"):
                (seed_dir / f"{ss}{ext}").unlink(missing_ok=True)
        base_cell.unlink(missing_ok=True)
        for pat in ("*.geom", "*.bib", "*.usp", "*.upf"):
            for f in seed_dir.glob(pat):
                try: f.unlink()
                except OSError: pass

    result["elastic_wall_time_s"] = f"{time.monotonic() - t_start:.0f}"
    result["elastic_source"] = "CASTEP"
    return result
