"""
VASP/POSCAR_INCAR.py  —  VASP File I/O and Parsing.
════════════════════════════════════════════════════════════════
Pure functions for generating POSCAR, POTCAR, INCAR, KPOINTS,
and parsing OUTCAR. No state, no subprocesses.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

import config as _cfg
from core_physics import Crystal
from engines.engine import EngineResult, _try_float, read_tail

# ─────────────────────────────────────────────────────────────────────────────
# POSCAR Generator
# ─────────────────────────────────────────────────────────────────────────────

def write_vca_poscar(
    dest: Path,
    crystal: Crystal,
    template_element: str,
    target_mix: dict[str, float],
    vegard: bool = True,
) -> tuple[list[str], list[float]]:

    L = crystal.lattice.copy()
    eps = 1e-8
    nonzero_mix = {e: f for e, f in target_mix.items() if f > eps}

    if vegard and len(nonzero_mix) > 1:
        r_tmpl = _cfg.ELEMENTS.get(template_element.capitalize(), {}).get("rad", 0.0)
        r_mix = sum(_cfg.ELEMENTS.get(e.capitalize(), {}).get("rad", 0.0) * f for e, f in nonzero_mix.items())
        if r_tmpl > eps and r_mix > eps:
            L *= (r_mix / r_tmpl)

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
        "1.00000000000000"
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

# ─────────────────────────────────────────────────────────────────────────────
# INCAR Generator
# ─────────────────────────────────────────────────────────────────────────────

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
        lines.append(f"VCA    = " + " ".join(f"{w:.4f}" for w in vca_weights))
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
            "IBRION = 6",
            "ISIF   = 3",
            "NSW    = 1",
            "POTIM  = 0.015",
            "NFREE  = 2",
        ])
    else:
        lines.extend([
            "IBRION = -1",
            "NSW    = 0",
        ])

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

# ─────────────────────────────────────────────────────────────────────────────
# OUTCAR Parsers
# ─────────────────────────────────────────────────────────────────────────────

def parse_outcar(path: Path) -> EngineResult:
    """
    Parses the VASP OUTCAR file from the bottom up to extract final physical metrics.
    Includes advanced telemetry: Fermi energy, Max Force, Pressure, and Peak Memory.
    """
    if not path.exists():
        return EngineResult(warning="OUTCAR not found")

    text = read_tail(path, max_bytes=3 * 1024 * 1024)
    extra_data = {}

    # 1. Convergence Status
    if "reached required accuracy" in text.lower():
        extra_data["geom_converged"] = "yes"
    else:
        extra_data["geom_converged"] = "no"

    # 2. Lattice Parameters (a, b, c)
    m_lat = re.findall(
        r"length of vectors\s*\n\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)",
        text
    )
    if m_lat:
        a, b, c = m_lat[-1]
        extra_data["a_opt_ang"] = _try_float(a)
        extra_data["b_opt_ang"] = _try_float(b)
        extra_data["c_opt_ang"] = _try_float(c)

    # 3. Volume
    m_vol = re.findall(r"volume of cell :\s*([-\d.]+)", text)
    volume_ang3 = _try_float(m_vol[-1]) if m_vol else None

    # 4. Energies (TOTEN and Sigma->0)
    m_energy = re.findall(r"free\s+energy\s+TOTEN\s*=\s*([-\d.]+)\s*eV", text)
    energy_ev = None
    if m_energy:
        energy_ev = _try_float(m_energy[-1])
        # Map TOTEN to enthalpy/free_energy for orchestrator compatibility
        extra_data["enthalpy_eV"] = energy_ev
        extra_data["free_energy_ev"] = energy_ev

    m_e0 = re.findall(r"energy\(sigma->0\)\s*=\s*([-\d.]+)", text)
    if m_e0:
        extra_data["energy_0k_ev"] = _try_float(m_e0[-1])

    # 5. Fermi Energy
    m_fermi = re.findall(r"E-fermi\s*:\s*([-\d.]+)", text)
    if m_fermi:
        extra_data["fermi_ev"] = _try_float(m_fermi[-1])

    # 6. Residual Pressure (Convert kB to GPa)
    m_press = re.findall(r"external pressure\s*=\s*([-\d.]+)\s*kB", text)
    if m_press:
        p_kb = _try_float(m_press[-1])
        if p_kb is not None:
            extra_data["residual_pressure_GPa"] = round(p_kb * 0.1, 4)

    # 7. Magnetization (if ISPIN = 2)
    m_mag = re.findall(r"number of electron\s+[\d.]+\s+magnetization\s+([-\d.]+)", text)
    if m_mag:
        extra_data["mag_moment"] = _try_float(m_mag[-1])

    # 8. Maximum Force (Calculate vector norm from the last iteration)
    force_blocks = re.findall(
        r"POSITION\s+TOTAL-FORCE \(eV/Angst\)\s*\n\s*-+\n(.*?)\n\s*-+\n\s*total drift",
        text, re.DOTALL
    )
    if force_blocks:
        last_block = force_blocks[-1]
        fmax = 0.0
        for line in last_block.strip().splitlines():
            parts = line.split()
            if len(parts) >= 6:
                try:
                    fx, fy, fz = float(parts[3]), float(parts[4]), float(parts[5])
                    fnorm = (fx**2 + fy**2 + fz**2)**0.5
                    if fnorm > fmax:
                        fmax = fnorm
                except ValueError:
                    pass
        extra_data["fmax_ev_ang"] = round(fmax, 6)

    # 9. Wall Time and Peak Memory
    m_time = re.search(r"Elapsed time\s*\(sec\):\s*([-\d.]+)", text)
    run_time_s = _try_float(m_time.group(1)) if m_time else None

    m_mem = re.search(r"Maximum memory used \(kb\):\s*([-\d.]+)", text)
    if m_mem:
        mem_kb = _try_float(m_mem.group(1))
        if mem_kb is not None:
            extra_data["peak_mem_mb"] = round(mem_kb / 1024.0, 1)

    return EngineResult(
        energy_ev=energy_ev,
        volume_ang3=volume_ang3,
        run_time_s=run_time_s,
        extra_data=extra_data,
        warning=None if energy_ev else "Final energy not found. SCF failed?"
    )


def parse_ibrion6_tensor(outcar: Path) -> np.ndarray | None:
    if not outcar.exists(): return None
    text = read_tail(outcar, max_bytes=2 * 1024 * 1024)

    m = re.search(r"TOTAL ELASTIC MODULI \(kBar\)\s+Direction[^\n]+\n[^\n]+\n(.*?)(?:\n\s*\n|---)", text, re.DOTALL)
    if not m: return None

    rows = []
    for line in m.group(1).splitlines():
        parts = line.split()
        if len(parts) >= 7:
            try: rows.append([float(x) for x in parts[1:7]])
            except ValueError: pass

    if len(rows) >= 6:
        C_kbar = np.array(rows[:6])
        return C_kbar / 10.0
    return None
