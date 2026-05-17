"""
VASP/POSCAR_INCAR.py  —  VASP file I/O and OUTCAR parsing.
═══════════════════════════════════════════════════════════════
Pure functions: generate POSCAR/INCAR, parse OUTCAR.
No state, no subprocesses, no engine logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

import config as _cfg
from core_physics import Crystal
from engines.engine import EngineResult, read_tail, try_float


# ─────────────────────────────────────────────────────────────────────────────
# POSCAR
# ─────────────────────────────────────────────────────────────────────────────


def write_vca_poscar(
    dest: Path,
    crystal: Crystal,
) -> tuple[list[str], list[float]]:
    """Write a VASP POSCAR; return (element_order, VCA_weights) for INCAR/POTCAR.

    For VCA mode (sites with multiple species), every such site appears N times
    in the POSCAR — once per mixed species. This function sorts all atoms
    by species to create the correct POSCAR structure.
    """
    flat_atoms = []
    for i, site in enumerate(crystal.sites):
        coords = crystal.frac_coords[i]
        for species, fraction in site.items():
            flat_atoms.append({'species': species.capitalize(), 'coords': coords, 'fraction': fraction})

    # VASP requires atoms to be grouped by species in the POSCAR file.
    flat_atoms.sort(key=lambda at: at['species'])

    ordered_elements = []
    element_counts = []
    final_coords = []
    vca_weights = []

    if flat_atoms:
        current_species = flat_atoms[0]['species']
        count = 0
        for atom in flat_atoms:
            if atom['species'] == current_species:
                count += 1
            else:
                if count > 0:
                    ordered_elements.append(current_species)
                    element_counts.append(count)
                current_species = atom['species']
                count = 1
            final_coords.append(atom['coords'])
            vca_weights.append(atom['fraction'])

        if count > 0:
            ordered_elements.append(current_species)
            element_counts.append(count)

    lines = [
        f"VCAForge Crystal-centric model",
        "1.00000000000000",
    ]
    for vec in crystal.lattice:
        lines.append(f"  {vec[0]:20.15f}  {vec[1]:20.15f}  {vec[2]:20.15f}")

    lines.append("  " + "  ".join(ordered_elements))
    lines.append("  " + "  ".join(str(c) for c in element_counts))
    lines.append("Direct")

    for fc in final_coords:
        lines.append(f"  {fc[0]:15.10f}  {fc[1]:15.10f}  {fc[2]:15.10f}")

    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ordered_elements, vca_weights


# ─────────────────────────────────────────────────────────────────────────────
# INCAR
# ─────────────────────────────────────────────────────────────────────────────


def write_incar(
    path: Path,
    task_type: str,
    *,
    cutoff: int,
    spin: bool,
    smearing: float,
    vca_weights: list[float] | None = None,
    nelect: float | None = None,
    ncore: int = 0,
) -> None:
    """Write an INCAR for GeometryOptimization, ElasticIBRION6, or SinglePoint."""
    is_geom = task_type == "GeometryOptimization"
    is_elastic = task_type in ("ElasticConstants", "ElasticIBRION6")
    ediff = _cfg.EDIFF_GEOM if is_geom else _cfg.EDIFF_IBRION6

    lines = [
        f"# VCAForge INCAR - Task: {task_type}",
        "PREC   = Accurate",
        "ALGO   = Normal",
        "LREAL  = Auto",
        f"ENCUT  = {cutoff}",
        f"EDIFF  = {ediff}",
        f"ISMEAR = {_cfg.ISMEAR}",
        f"SIGMA  = {smearing:.4f}",
        f"ISPIN  = {2 if spin else 1}",
        "LWAVE  = .FALSE.",
        "LCHARG = .FALSE.",
    ]

    if vca_weights and any(abs(w - 1.0) > 1e-6 for w in vca_weights):
        lines.append("VCA    = " + " ".join(f"{w:.4f}" for w in vca_weights))
    if nelect is not None:
        lines.append(f"NELECT = {nelect:.4f}")

    if is_geom:
        lines.extend([
            f"IBRION = {_cfg.IBRION_GEOM}",
            f"ISIF   = {_cfg.ISIF}",
            f"NSW    = {_cfg.NSW_MAX_VASP}",
            f"EDIFFG = {_cfg.EDIFFG_VASP}",
        ])
        if ncore > 0:
            lines.append(f"NCORE  = {ncore}")
    elif is_elastic:
        lines.extend([
            "IBRION = 6", "ISIF   = 3", "NSW    = 1",
            "POTIM  = 0.015", "NFREE  = 2",
        ])
    else:
        lines.extend(["IBRION = -1", "NSW    = 0"])

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# Back-compat alias for existing callers — to be removed once vasp.py updated.
write_engine_params = write_incar


# ─────────────────────────────────────────────────────────────────────────────
# OUTCAR parsing — forward-scan state machine, all data in extra_data
# ─────────────────────────────────────────────────────────────────────────────


def parse_outcar(path: Path) -> EngineResult:
    """Parse a VASP OUTCAR. All engine-specific fields go in extra_data."""
    if not path.exists():
        return EngineResult(warning="OUTCAR not found")

    text = read_tail(path, max_bytes=5 * 1024 * 1024)
    lines = text.splitlines()

    energy_ev: float | None = None
    volume_ang3: float | None = None
    run_time_s: float | None = None
    extra: dict[str, Any] = {}

    # Forward scan — keep overwriting so the last (final) occurrence wins.
    i = 0
    while i < len(lines):
        line = lines[i]

        if "Reached required accuracy" in line:
            extra["geom_converged"] = "yes"

        if "free  energy   TOTEN" in line:
            val = try_float(line.split("=")[-1].strip().split()[0])
            if val is not None:
                energy_ev = val
                extra["enthalpy_eV"] = val

        if "volume of cell :" in line:
            volume_ang3 = try_float(line.split(":")[1].strip().split()[0])

        # Lattice block: "length of vectors" header, then a line of three floats.
        if "length of vectors" in line and i + 1 < len(lines):
            parts = lines[i + 1].split()
            if len(parts) >= 3:
                a, b, c = (
                    try_float(parts[0]),
                    try_float(parts[1]),
                    try_float(parts[2]),
                )
                if a and b and c:
                    extra["a_opt_ang"] = a
                    extra["b_opt_ang"] = b
                    extra["c_opt_ang"] = c
            i += 2
            continue

        if "Elapsed time" in line:
            run_time_s = try_float(line.split(":")[-1].strip())

        i += 1

    # If geom_converged was never set, default to "no" only when the geom loop
    # is known to have run (NSW > 0 implied by presence of "Iteration").
    if "geom_converged" not in extra and any(
        "Iteration" in ln for ln in lines[-50:]
    ):
        extra["geom_converged"] = "no"

    return EngineResult(
        energy_ev=energy_ev,
        volume_ang3=volume_ang3,
        run_time_s=run_time_s,
        extra_data=extra,
        warning=None if energy_ev is not None else "TOTEN not found in OUTCAR",
    )


# ─────────────────────────────────────────────────────────────────────────────
# OUTCAR — IBRION=6 elastic tensor parser (called from vasp.py::run_elastic)
# ─────────────────────────────────────────────────────────────────────────────


def parse_ibrion6_tensor(text: str) -> np.ndarray | None:
    """Parse VASP IBRION=6 elastic tensor (6x6) from OUTCAR text.

    Returns the 6x6 stiffness matrix in GPa, or None if not found.
    VASP reports the tensor in kBar; we convert to GPa (/10).
    """
    import re
    m = re.search(
        r"TOTAL ELASTIC MODULI \(kBar\)\s+Direction[^\n]+\n[^\n]+\n(.*?)(?:\n\s*\n|---)",
        text, re.DOTALL,
    )
    if not m:
        return None
    rows = []
    for line in m.group(1).splitlines():
        parts = line.split()
        if len(parts) >= 7:
            try:
                rows.append([float(x) for x in parts[1:7]])
            except ValueError:
                pass
    if len(rows) >= 6:
        return np.array(rows[:6]) / 10.0
    return None
