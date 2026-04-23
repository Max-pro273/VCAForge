"""
CASTEP/cell_param.py  —  CASTEP file I/O, .param generation, and output parsing.
════════════════════════════════════════════════════════════════════════════════
No subprocess calls (except cif2cell), no user interaction.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import config as _cfg
from core_physics import Crystal
from engines.engine import EngineResult, _try_float, read_tail

def format_castep_symmetry_block(crystal: Crystal) -> str:
    """
    Formats mathematical symmetry operations into strict CASTEP %BLOCK syntax.
    Requests Cartesian rotations because VCAForge writes %BLOCK LATTICE_CART.
    """
    # Request Cartesian rotations for CASTEP compatibility
    rotations, translations = crystal.get_symmetry_operations(cartesian_rotations=True)

    lines = ["\n%BLOCK SYMMETRY_OPS"]
    for i, (r, t) in enumerate(zip(rotations, translations), 1):
        lines.append(f"# Symm. op. {i}")
        for row in r:
            # Exact spacing match to cif2cell
            lines.append(f"  {row[0]: 17.15f}   {row[1]: 17.15f}   {row[2]: 17.15f} ")
        lines.append(f"  {t[0]: 17.15f}   {t[1]: 17.15f}   {t[2]: 17.15f} ")

    lines.append("%ENDBLOCK SYMMETRY_OPS\n")
    return "\n".join(lines)


def write_vca_cell(
    dest: Path,
    crystal: Crystal,
    template_element: str,
    target_mix: dict[str, float],
    *,
    occ: float = 1.0,
    vegard: bool = True,
) -> None:
    """Generates .cell file from Crystal object natively. Strict 15-decimal precision."""
    L = crystal.lattice.copy()
    eps = 1e-9
    nonzero_mix = {e: f for e, f in target_mix.items() if f > eps}

    if vegard and len(nonzero_mix) > 1:
        r_template = _cfg.ELEMENTS.get(template_element.capitalize(), {}).get("rad", 0.0)
        r_mix = sum(_cfg.ELEMENTS.get(e.capitalize(), {}).get("rad", 0.0) * f for e, f in nonzero_mix.items())
        if r_template > 1e-6 and r_mix > 1e-6:
            L *= (r_mix / r_template)

    lines = [
        f"# VCAForge v{_cfg.VERSION}  mix={list(target_mix.keys())}  tmpl={template_element}\n",
        "%BLOCK LATTICE_CART\nANG\n"
    ]
    for vec in L:
        lines.append(f"  {vec[0]:20.15f}  {vec[1]:20.15f}  {vec[2]:20.15f}\n")
    lines.append("%ENDBLOCK LATTICE_CART\n\n%BLOCK POSITIONS_FRAC\n")

    for sp, fc in zip(crystal.species, crystal.frac_coords):
        if sp.lower() == template_element.lower():
            if len(nonzero_mix) == 1:
                lines.append(f"  {next(iter(nonzero_mix)):2}   {fc[0]:20.15f}   {fc[1]:20.15f}   {fc[2]:20.15f}\n")
            else:
                for mix_el, mix_frac in nonzero_mix.items():
                    lines.append(f"  {mix_el:2}   {fc[0]:20.15f}   {fc[1]:20.15f}   {fc[2]:20.15f}  MIXTURE:( 1 {mix_frac:.8f})\n")
        else:
            if occ < 1.0 - eps and _cfg.ELEMENTS.get(sp.capitalize(), {}).get("nonmetal"):
                lines.append(f"  {sp:2}   {fc[0]:20.15f}   {fc[1]:20.15f}   {fc[2]:20.15f}  MIXTURE:( 1 {occ:.8f})\n")
            else:
                lines.append(f"  {sp:2}   {fc[0]:20.15f}   {fc[1]:20.15f}   {fc[2]:20.15f}\n")

    lines.append("%ENDBLOCK POSITIONS_FRAC\n")
    lines.append(format_castep_symmetry_block(crystal))
    dest.write_text("".join(lines), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Param Generation
# ─────────────────────────────────────────────────────────────────────────────

def _scf_block(xc: str, cutoff: int, spin: bool, nextra: int, smearing: float, mix_amp: float, *, ncp: bool = False) -> str:
    """Generates the base SCF configuration block."""
    ncp_note = "  # NCP: raise cutoff >= 900 eV for C/N/O" if ncp else ""
    return (
        f"# VCAForge v{_cfg.VERSION}\n"
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

def write_engine_params(
    path: Path,
    task_type: str,
    xc: str,
    cutoff: int,
    spin: bool,
    nextra: int,
    smearing: float,
    *,
    ncp: bool = False,
) -> None:
    """Generates a .param file from scratch for GeomOpt or SinglePoint."""

    is_geom = task_type == "GeometryOptimization"
    mix_amp = _cfg.MIX_AMP_GEOM if is_geom else _cfg.MIX_AMP_SP
    elec_tol = _cfg.ELEC_TOL_GEOM if is_geom else _cfg.ELEC_TOL_SP

    body = _scf_block(xc, cutoff, spin, nextra, smearing, mix_amp, ncp=ncp)
    body += f"elec_energy_tol     : {elec_tol}\n\n"
    body += f"task                : {task_type}\n"
    body += "calculate_stress    : true\n\n"

    if is_geom:
        body += (
            f"geom_method         : LBFGS\n"
            f"geom_max_iter       : {_cfg.GEOM_MAX_ITER}\n"
            f"geom_energy_tol     : {_cfg.GEOM_E_TOL}\n"
            f"geom_force_tol      : {_cfg.GEOM_F_TOL}\n"
            f"geom_stress_tol     : {_cfg.GEOM_S_TOL}\n"
            f"geom_disp_tol       : {_cfg.GEOM_D_TOL}\n\n"
        )
    else:
        body += f"finite_basis_corr   : {_cfg.FINITE_BASIS}\n\n"

    body += (
        f"opt_strategy        : speed\n"
        f"write_checkpoint    : none\n"
        f"num_dump_cycles     : 0\n"
        f"write_cell_structure: {'true' if is_geom else 'false'}\n"
    )
    path.write_text(body, encoding="utf-8")

def patch_nextra(param_path: Path, nextra: int) -> None:
    """Updates the nextra_bands value in an existing .param file."""
    lines = param_path.read_text(encoding="utf-8").splitlines()
    replaced = False
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("nextra_bands") and ":" in line:
            lines[i] = f"nextra_bands        : {nextra}"
            replaced = True
            break
    if not replaced:
        lines.append(f"nextra_bands        : {nextra}")
    param_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

# ─────────────────────────────────────────────────────────────────────────────
# Output Parsing (CASTEP)
# ─────────────────────────────────────────────────────────────────────────────

def parse_output(output_file: Path) -> EngineResult:
    """
    Parses the .castep file from the bottom up to extract final physical metrics.
    Includes advanced telemetry: Fermi energy, Max Force, Enthalpy, and Spin.
    """
    if not output_file.exists():
        return EngineResult(warning=f"Output file missing: {output_file.name}")

    # Read tail to avoid loading massive SCF histories into memory (last 2MB is plenty)
    text = read_tail(output_file, max_bytes=2 * 1024 * 1024)
    lines = text.splitlines()

    energy_ev = None
    volume_ang3 = None
    density_gcm3 = None
    run_time_s = None

    extra: dict[str, Any] = {}

    # Read from the bottom up to grab the *final* optimized values
    for line in reversed(lines):
        # 1. Core Metrics
        if energy_ev is None and "Final energy, E" in line:
            parts = line.split("=")
            if len(parts) == 2:
                energy_ev = _try_float(parts[1].replace("eV", ""))

        elif volume_ang3 is None and "Current cell volume" in line:
            parts = line.split("=")
            if len(parts) == 2:
                volume_ang3 = _try_float(parts[1].replace("A**3", ""))

        elif density_gcm3 is None and "Density" in line and "g/cm" in line:
            parts = line.split("=")
            if len(parts) == 2:
                density_gcm3 = _try_float(parts[1].replace("g/cm**3", ""))

        elif run_time_s is None and "Total time" in line:
            parts = line.split("=")
            if len(parts) == 2:
                run_time_s = _try_float(parts[1].replace("s", ""))

        # 2. Advanced Physical Telemetry (Saved to extra_data)
        elif "Final Enthalpy" in line and "enthalpy_eV" not in extra:
            parts = line.split("=")
            if len(parts) == 2:
                extra["enthalpy_eV"] = _try_float(parts[1].replace("eV", ""))

        elif "Fermi energy" in line and "fermi_ev" not in extra:
            # Handles both "Fermi energy =" and "Fermi energy for spin 1 ="
            parts = line.split("=")
            if len(parts) == 2:
                extra["fermi_ev"] = _try_float(parts[1].replace("eV", ""))

        elif "Integrated Spin Density" in line and "mag_moment" not in extra:
            parts = line.split("=")
            if len(parts) == 2:
                extra["mag_moment"] = _try_float(parts[1])

        elif "Peak Memory Use" in line and "peak_mem_mb" not in extra:
            parts = line.split("=")
            if len(parts) == 2:
                mem_kb = _try_float(parts[1].replace("kB", ""))
                if mem_kb is not None:
                    extra["peak_mem_mb"] = round(mem_kb / 1024.0, 1)

        elif "Final free energy (E-TS)" in line and "free_energy_ev" not in extra:
            parts = line.split("=")
            if len(parts) == 2:
                extra["free_energy_ev"] = _try_float(parts[1].replace("eV", ""))

        elif "est. 0K energy (E-0.5TS)" in line and "energy_0k_ev" not in extra:
            parts = line.split("=")
            if len(parts) == 2:
                extra["energy_0k_ev"] = _try_float(parts[1].replace("eV", ""))

        elif "Final bulk modulus" in line and "B_lbfgs_GPa" not in extra:
            parts = line.split("=")
            if len(parts) == 2:
                extra["B_lbfgs_GPa"] = _try_float(parts[1].replace("GPa", ""))

        elif "Pressure:" in line and "residual_pressure_GPa" not in extra:
            # Беремо останній (фінальна геометрія)
            m = re.search(r"Pressure:\s*([-\d\.]+)", line)
            if m:
                extra["residual_pressure_GPa"] = _try_float(m.group(1))

        elif "Charge spilling" in line and "charge_spilling_pct" not in extra:
            m = re.search(r"=\s*([\d\.]+)%", line)
            if m:
                extra["charge_spilling_pct"] = _try_float(m.group(1))

    # Mulliken charges — для кожного унікального виду
    mulliken_charges = {}
    m_mull = re.findall(
        r"^\s+(\w+)\s+\d+\s+[\d\.\-]+\s+[\d\.\-]+\s+[\d\.\-]+\s+[\d\.\-]+\s+[\d\.\-]+\s+([-\d\.]+)",
        text, re.M
    )
    for species, charge in m_mull:
        key = f"mulliken_q_{species}"
        if key not in mulliken_charges:
            mulliken_charges[key] = _try_float(charge)
    extra.update(mulliken_charges)

    # Bond populations — середнє та мін/макс
    m_bonds = re.findall(
        r"^\s+\w+\s+\d+\s+--\s+\w+\s+\d+\s+([-\d\.]+)\s+([\d\.]+)",
        text, re.M
    )
    if m_bonds:
        pops = [float(p) for p, _ in m_bonds]
        lens = [float(l) for _, l in m_bonds]
        extra["bond_population_avg"] = round(sum(pops)/len(pops), 4)
        extra["bond_length_avg_ang"] = round(sum(lens)/len(lens), 5)

    # 3. Extract Max Force & BFGS steps via Regex Block
    # Look for the last BFGS convergence block
    m_force = re.search(r"\|\s*Max force \(eV/A\)\s*\|\s*([\d\.]+)\s*\|", text)
    if m_force:
        extra["fmax_ev_ang"] = _try_float(m_force.group(1))

    # geom_converged: "yes" if LBFGS converged, "no" if it just ran out of iterations
    if "LBFGS: finished iteration" in text:
        if "Geometry optimization completed successfully" in text:
            extra["geom_converged"] = "yes"
        else:
            extra["geom_converged"] = "no"

    # Look for Lattice Parameters (a, b, c) of the optimized cell
    # We grab the last instance in the file (which is the final geometry)
    m_lat = re.findall(
            r"a\s*=\s*([\d\.]+)\s+alpha\s*=\s*([\d\.]+)\s*\n\s*"
            r"b\s*=\s*([\d\.]+)\s+beta\s*=\s*([\d\.]+)\s*\n\s*"
            r"c\s*=\s*([\d\.]+)\s+gamma\s*=\s*([\d\.]+)",
            text, re.I
        )
    if m_lat:
        last_lat = m_lat[-1]
        extra["a_opt_ang"], extra["alpha"] = _try_float(last_lat[0]), _try_float(last_lat[1])
        extra["b_opt_ang"], extra["beta"]  = _try_float(last_lat[2]), _try_float(last_lat[3])
        extra["c_opt_ang"], extra["gamma"] = _try_float(last_lat[4]), _try_float(last_lat[5])

    return EngineResult(
        energy_ev=energy_ev,
        volume_ang3=volume_ang3,
        density_gcm3=density_gcm3,
        run_time_s=run_time_s,
        extra_data=extra,
        warning=None if energy_ev else "Final energy not found. SCF failed?",
    )

def parse_elastic_file(path: Path) -> dict[str, Any]:
    """Читає .elastic файл без Regex."""
    if not path.exists(): return {}
    r: dict[str, Any] = {}
    in_cij = False
    cij_rows: list[list[float]] = []

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if "Elastic Stiffness Constants" in line:
            in_cij = True
            cij_rows = []
            continue

        if in_cij:
            if not s or s.startswith("=") or s.startswith("-"):
                if len(cij_rows) == 6: in_cij = False
                continue
            parts = s.split()
            if len(parts) >= 6:
                try: cij_rows.append([float(v) for v in parts[:6]])
                except ValueError: pass
            if len(cij_rows) == 6:
                in_cij = False
                keys = ["C11", "C12", "C13", "C22", "C23", "C33", "C44", "C55", "C66"]
                indices = [(0,0), (0,1), (0,2), (1,1), (1,2), (2,2), (3,3), (4,4), (5,5)]
                for k, (i, j) in zip(keys, indices):
                    r[k] = f"{cij_rows[i][j]:.4f}"

        for label, col in [
            ("Hill bulk modulus", "B_Hill_GPa"), ("Hill shear modulus", "G_Hill_GPa"),
            ("Young modulus", "E_GPa"), ("Poisson ratio", "nu"),
            ("Debye temperature", "T_Debye_K"), ("Vickers hardness", "H_Vickers_GPa"),
        ]:
            if label in line and "=" in line:
                v = _try_float(line.split("=")[-1].strip().split()[0])
                if v is not None: r[col] = f"{v:.4f}"
    return r
