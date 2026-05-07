"""
VASP/POSCAR_INCAR.py  —  VASP File I/O and Parsing.
════════════════════════════════════════════════════════════════
Pure functions for generating POSCAR, INCAR, KPOINTS,
and parsing OUTCAR. No state, no subprocesses.
"""

from __future__ import annotations

import re
from pathlib import Path

import config as _cfg
import numpy as np
from core_physics import Crystal
from engines.engine import EngineResult, _try_float, read_tail


def write_vca_poscar(
    dest: Path,
    crystal: Crystal,
    template_element: str,
    target_mix: dict[str, float],
) -> tuple[list[str], list[float]]:
    L = crystal.lattice.copy()
    eps = 1e-8
    nonzero_mix = {e: f for e, f in target_mix.items() if f > eps}

    vca_coords = []
    other_coords: dict[str, list[np.ndarray]] = {}

    for sp, fc in zip(crystal.species, crystal.frac_coords):
        if sp.lower() == template_element.lower():
            vca_coords.append(fc)
        else:
            other_coords.setdefault(sp.capitalize(), []).append(fc)

    ordered_elements = []
    element_counts = []
    vca_weights = []
    final_coords = []

    for mix_el, mix_frac in nonzero_mix.items():
        ordered_elements.append(mix_el.capitalize())
        element_counts.append(len(vca_coords))
        vca_weights.append(mix_frac)
        final_coords.extend(vca_coords)

    for sp, coords in other_coords.items():
        ordered_elements.append(sp)
        element_counts.append(len(coords))
        vca_weights.append(1.0)
        final_coords.extend(coords)

    lines = [
        f"VCAForge mix={list(nonzero_mix.keys())} tmpl={template_element}",
        "1.00000000000000",
    ]
    for vec in L:
        lines.append(f"  {vec[0]:20.15f}  {vec[1]:20.15f}  {vec[2]:20.15f}")

    lines.append("  " + "  ".join(ordered_elements))
    lines.append("  " + "  ".join(str(c) for c in element_counts))
    lines.append("Direct")

    for fc in final_coords:
        lines.append(f"  {fc[0]:15.10f}  {fc[1]:15.10f}  {fc[2]:15.10f}")

    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ordered_elements, vca_weights


def write_engine_params(
    path: Path,
    task_type: str,
    xc: str,
    cutoff: int,
    spin: bool,
    smearing: float,
    *,
    vca_weights: list[float] | None = None,
    nelect: float | None = None,
    ncore: int = 0,
) -> None:
    is_geom = task_type == "GeometryOptimization"
    is_elastic = task_type in ("ElasticConstants", "ElasticIBRION6")

    ediff = (
        getattr(_cfg, "EDIFF_GEOM", "1E-5")
        if is_geom
        else getattr(_cfg, "EDIFF_IBRION6", "1E-7")
    )

    lines = [
        f"# VCAForge INCAR - Task: {task_type}",
        "PREC   = Accurate",
        "ALGO   = Normal",
        "LREAL  = Auto",
        f"ENCUT  = {cutoff}",
        f"EDIFF  = {ediff}",
        f"ISMEAR = {getattr(_cfg, 'ISMEAR', 1)}",
        f"SIGMA  = {smearing:.4f}",
        f"ISPIN  = {2 if spin else 1}",
        "LWAVE  = .FALSE.",
        "LCHARG = .FALSE.",
    ]

    if vca_weights and any(abs(w - 1.0) > 1e-6 for w in vca_weights):
        lines.append(f"VCA    = " + " ".join(f"{w:.4f}" for w in vca_weights))
    if nelect is not None:
        lines.append(f"NELECT = {nelect:.4f}")

    if is_geom:
        lines.extend(
            [
                f"IBRION = {getattr(_cfg, 'IBRION_GEOM', 2)}",
                f"ISIF   = {getattr(_cfg, 'ISIF', 3)}",
                f"NSW    = {getattr(_cfg, 'NSW_MAX_VASP', 300)}",
                f"EDIFFG = {getattr(_cfg, 'EDIFFG_VASP', '-0.01')}",
            ]
        )
        if ncore > 0:
            lines.append(f"NCORE  = {ncore}")
    elif is_elastic:
        lines.extend(
            [
                "IBRION = 6",
                "ISIF   = 3",
                "NSW    = 1",
                "POTIM  = 0.015",
                "NFREE  = 2",
            ]
        )
    else:
        lines.extend(["IBRION = -1", "NSW    = 0"])

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_outcar(path: Path) -> EngineResult:
    r = EngineResult()
    if not path.exists():
        r.warning = "OUTCAR not found"
        return r

    text = read_tail(path, max_bytes=5 * 1024 * 1024)
    lines = text.splitlines()

    la = lb = lc = None

    for i, line in enumerate(reversed(lines)):
        s = line.strip()

        if not r.geom_converged and "Reached required accuracy" in line:
            r.geom_converged = True

        if r.enthalpy_eV is None and "free  energy   TOTEN" in line:
            r.enthalpy_eV = _try_float(line.split("=")[-1].strip().split()[0])
            r.energy_ev = r.enthalpy_eV

        if r.volume_ang3 is None and "volume of cell :" in line:
            r.volume_ang3 = _try_float(line.split(":")[1].strip().split()[0])

        if la is None and "length of vectors" in line:
            try:
                parts = lines[len(lines) - 1 - i + 1].split()
                if len(parts) >= 3:
                    la, lb, lc = (
                        _try_float(parts[0]),
                        _try_float(parts[1]),
                        _try_float(parts[2]),
                    )
                    if la and lb and lc:
                        r.a_opt_ang, r.b_opt_ang, r.c_opt_ang = la, lb, lc
            except IndexError:
                pass

        if r.run_time_s is None and "Elapsed time" in line:
            r.run_time_s = _try_float(line.split(":")[-1].strip())

    return r
