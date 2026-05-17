"""
CASTEP/cell_param.py  —  CASTEP file I/O, .param generation, output parsing.
═══════════════════════════════════════════════════════════════════════════════
Pure functions. No subprocesses, no user interaction. Vegard scaling is NOT
performed here — that is the responsibility of crystal_modes.VCAStrategy.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import config as _cfg
from core_physics import Crystal
from engines.engine import EngineResult, read_tail, try_float


# ─────────────────────────────────────────────────────────────────────────────
# Symmetry block
# ─────────────────────────────────────────────────────────────────────────────


def format_castep_symmetry_block(crystal: Crystal) -> str:
    """Format crystal symmetry into CASTEP %BLOCK SYMMETRY_OPS syntax.

    Requests Cartesian rotations because VCAForge writes %BLOCK LATTICE_CART.
    """
    rotations, translations = crystal.get_symmetry_operations(
        cartesian_rotations=True
    )

    lines = ["\n%BLOCK SYMMETRY_OPS"]
    for i, (r, t) in enumerate(zip(rotations, translations), 1):
        lines.append(f"# Symm. op. {i}")
        for row in r:
            lines.append(
                f"  {row[0]: 17.15f}   {row[1]: 17.15f}   {row[2]: 17.15f} "
            )
        lines.append(f"  {t[0]: 17.15f}   {t[1]: 17.15f}   {t[2]: 17.15f} ")
    lines.append("%ENDBLOCK SYMMETRY_OPS\n")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# .cell writer  (no Vegard — strategy already did it)
# ─────────────────────────────────────────────────────────────────────────────


def write_castep_cell(
    dest: Path,
    crystal: Crystal,
) -> None:
    """Write a CASTEP .cell file. The crystal is written verbatim — Vegard
    scaling, if applicable, was applied upstream by VCAStrategy.

    Site decoration rules:
      • Template sublattice with len(target_mix)==1 → single species, no MIXTURE:.
      • Template sublattice with len(target_mix)>1  → MIXTURE:( 1 frac ) per species.
      • Non-template sublattice: written as-is, unless occ < 1.0 and species is a
        non-metal (config.ELEMENTS[sym]["nonmetal"]=True), in which case it gets
        a partial-occupancy MIXTURE tag.
    """
    lines = [
        f"# VCAForge v{_cfg.VERSION} Crystal-centric model\n",
        "%BLOCK LATTICE_CART\nANG\n",
    ]
    for vec in crystal.lattice:
        lines.append(
            f"  {vec[0]:20.15f}  {vec[1]:20.15f}  {vec[2]:20.15f}\n"
        )
    lines.append("%ENDBLOCK LATTICE_CART\n\n%BLOCK POSITIONS_FRAC\n")

    for i, site in enumerate(crystal.sites):
        fc = crystal.frac_coords[i]
        if len(site) > 1:
            # VCA site
            for species, fraction in site.items():
                lines.append(
                    f"  {species:2}   {fc[0]:20.15f}   {fc[1]:20.15f}   "
                    f"{fc[2]:20.15f}  MIXTURE:( 1 {fraction:.8f})\n"
                )
        else:
            # Single species site
            species, fraction = next(iter(site.items()))
            if abs(fraction - 1.0) > 1e-8:
                 # Partial occupancy non-VCA site
                lines.append(
                    f"  {species:2}   {fc[0]:20.15f}   {fc[1]:20.15f}   "
                    f"{fc[2]:20.15f}  MIXTURE:( 1 {fraction:.8f})\n"
                )
            else:
                lines.append(
                    f"  {species:2}   {fc[0]:20.15f}   {fc[1]:20.15f}   "
                    f"{fc[2]:20.15f}\n"
                )

    lines.append("%ENDBLOCK POSITIONS_FRAC\n")
    lines.append(format_castep_symmetry_block(crystal))
    dest.write_text("".join(lines), encoding="utf-8")





# ─────────────────────────────────────────────────────────────────────────────
# .param writer
# ─────────────────────────────────────────────────────────────────────────────


def _scf_block(
    xc: str, cutoff: int, spin: bool, nextra: int,
    smearing: float, mix_amp: float,
) -> str:
    return (
        f"# VCAForge v{_cfg.VERSION}\n"
        f"xc_functional       : {xc}\n"
        f"cut_off_energy      : {cutoff} eV\n"
        f"spin_polarized      : {'true' if spin else 'false'}\n\n"
        f"max_scf_cycles      : {_cfg.MAX_SCF}\n"
        f"metals_method       : {_cfg.METALS_METHOD}\n"
        f"mixing_scheme       : {_cfg.MIXING_SCHEME}\n"
        f"smearing_width      : {smearing:.2f} eV\n"
        f"mix_charge_amp      : {mix_amp}\n"
        f"nextra_bands        : {nextra}\n"
    )


def write_castep_param(
    path: Path,
    task_type: str,
    xc: str,
    cutoff: int,
    spin: bool,
    nextra: int,
    smearing: float,
) -> None:
    """Generate a .param file for GeomOpt, SinglePoint, or ElasticConstants."""
    is_geom = task_type == "GeometryOptimization"
    mix_amp = _cfg.MIX_AMP_GEOM if is_geom else _cfg.MIX_AMP_SP
    elec_tol = _cfg.ELEC_TOL_GEOM if is_geom else _cfg.ELEC_TOL_SP

    body = _scf_block(xc, cutoff, spin, nextra, smearing, mix_amp)
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


# Back-compat alias.
write_engine_params = write_castep_param


def patch_nextra(param_path: Path, nextra: int) -> None:
    """Update nextra_bands in an existing .param file."""
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
# Output parsing
# ─────────────────────────────────────────────────────────────────────────────


def _extract_after_equals(line: str, unit: str = "") -> float | None:
    """Helper: parse 'key = value [unit]' format."""
    parts = line.split("=")
    if len(parts) != 2:
        return None
    return try_float(parts[1].replace(unit, ""))


def parse_castep_output(output_file: Path) -> EngineResult:
    """Parse a .castep file. All non-standard fields land in extra_data."""
    if not output_file.exists():
        return EngineResult(warning=f"Output file missing: {output_file.name}")

    text = read_tail(output_file, max_bytes=2 * 1024 * 1024)
    lines = text.splitlines()

    energy_ev: float | None = None
    volume_ang3: float | None = None
    density_gcm3: float | None = None
    run_time_s: float | None = None
    extra: dict[str, Any] = {}

    # Bottom-up scan to find FINAL values of all metrics.
    for line in reversed(lines):
        if energy_ev is None and "Final energy, E" in line:
            energy_ev = _extract_after_equals(line, "eV")
        elif volume_ang3 is None and "Current cell volume" in line:
            volume_ang3 = _extract_after_equals(line, "A**3")
        elif density_gcm3 is None and "Density" in line and "g/cm" in line:
            density_gcm3 = _extract_after_equals(line, "g/cm**3")
        elif run_time_s is None and "Total time" in line:
            run_time_s = _extract_after_equals(line, "s")
        elif "Final Enthalpy" in line and "enthalpy_eV" not in extra:
            v = _extract_after_equals(line, "eV")
            if v is not None:
                extra["enthalpy_eV"] = v
        elif "Fermi energy" in line and "fermi_ev" not in extra:
            v = _extract_after_equals(line, "eV")
            if v is not None:
                extra["fermi_ev"] = v
        elif "Integrated Spin Density" in line and "mag_moment" not in extra:
            v = _extract_after_equals(line)
            if v is not None:
                extra["mag_moment"] = v
        elif "Peak Memory Use" in line and "peak_mem_mb" not in extra:
            kb = _extract_after_equals(line, "kB")
            if kb is not None:
                extra["peak_mem_mb"] = round(kb / 1024.0, 1)
        elif "Final free energy (E-TS)" in line and "free_energy_ev" not in extra:
            v = _extract_after_equals(line, "eV")
            if v is not None:
                extra["free_energy_ev"] = v
        elif "est. 0K energy (E-0.5TS)" in line and "energy_0k_ev" not in extra:
            v = _extract_after_equals(line, "eV")
            if v is not None:
                extra["energy_0k_ev"] = v
        elif "Final bulk modulus" in line and "B_lbfgs_GPa" not in extra:
            v = _extract_after_equals(line, "GPa")
            if v is not None:
                extra["B_lbfgs_GPa"] = v
        elif "Pressure:" in line and "residual_pressure_GPa" not in extra:
            m = re.search(r"Pressure:\s*([-\d\.]+)", line)
            if m:
                extra["residual_pressure_GPa"] = try_float(m.group(1))
        elif "Charge spilling" in line and "charge_spilling_pct" not in extra:
            m = re.search(r"=\s*([\d\.]+)%", line)
            if m:
                extra["charge_spilling_pct"] = try_float(m.group(1))

    # ── Mulliken charges (N28 fix: take LAST occurrence, not first) ──────────
    m_mull = re.findall(
        r"^\s+(\w+)\s+\d+\s+[\d\.\-]+\s+[\d\.\-]+\s+[\d\.\-]+\s+[\d\.\-]+\s+"
        r"[\d\.\-]+\s+([-\d\.]+)",
        text, re.M,
    )
    mulliken: dict[str, float | None] = {}
    for species, charge in reversed(m_mull):
        key = f"mulliken_q_{species}"
        if key not in mulliken:
            mulliken[key] = try_float(charge)
    extra.update(mulliken)

    # ── Bond populations ──────────────────────────────────────────────────────
    m_bonds = re.findall(
        r"^\s+\w+\s+\d+\s+--\s+\w+\s+\d+\s+([-\d\.]+)\s+([\d\.]+)", text, re.M,
    )
    if m_bonds:
        pops = [float(p) for p, _ in m_bonds]
        lens = [float(L) for _, L in m_bonds]
        extra["bond_population_avg"] = round(sum(pops) / len(pops), 4)
        extra["bond_length_avg_ang"] = round(sum(lens) / len(lens), 5)

    # ── Max force ─────────────────────────────────────────────────────────────
    m_force = re.search(r"\|\s*Max force \(eV/A\)\s*\|\s*([\d\.]+)\s*\|", text)
    if m_force:
        extra["fmax_ev_ang"] = try_float(m_force.group(1))

    # ── Geom convergence ──────────────────────────────────────────────────────
    if "LBFGS: finished iteration" in text:
        extra["geom_converged"] = (
            "yes" if "Geometry optimization completed successfully" in text else "no"
        )

    # ── Final lattice parameters ──────────────────────────────────────────────
    m_lat = re.findall(
        r"a\s*=\s*([\d\.]+)\s+alpha\s*=\s*([\d\.]+)\s*\n\s*"
        r"b\s*=\s*([\d\.]+)\s+beta\s*=\s*([\d\.]+)\s*\n\s*"
        r"c\s*=\s*([\d\.]+)\s+gamma\s*=\s*([\d\.]+)",
        text, re.I,
    )
    if m_lat:
        a, al, b, be, c, ga = m_lat[-1]
        extra["a_opt_ang"] = try_float(a)
        extra["alpha"] = try_float(al)
        extra["b_opt_ang"] = try_float(b)
        extra["beta"] = try_float(be)
        extra["c_opt_ang"] = try_float(c)
        extra["gamma"] = try_float(ga)

    return EngineResult(
        energy_ev=energy_ev,
        volume_ang3=volume_ang3,
        density_gcm3=density_gcm3,
        run_time_s=run_time_s,
        extra_data=extra,
        warning=None if energy_ev else "Final energy not found. SCF failed?",
    )


# Back-compat alias.
parse_output = parse_castep_output


# ─────────────────────────────────────────────────────────────────────────────
# .elastic file parser
# ─────────────────────────────────────────────────────────────────────────────


def parse_elastic_file(path: Path) -> dict[str, Any]:
    """Parse a CASTEP .elastic file via a state machine (no regex on numbers)."""
    if not path.exists():
        return {}

    result: dict[str, Any] = {}
    in_cij = False
    cij_rows: list[list[float]] = []

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if "Elastic Stiffness Constants" in line:
            in_cij = True
            cij_rows = []
            continue

        if in_cij:
            if not s or s.startswith(("=", "-")):
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
                keys = ["C11", "C12", "C13", "C22", "C23", "C33",
                        "C44", "C55", "C66"]
                indices = [(0, 0), (0, 1), (0, 2),
                           (1, 1), (1, 2), (2, 2),
                           (3, 3), (4, 4), (5, 5)]
                for k, (i, j) in zip(keys, indices):
                    result[k] = f"{cij_rows[i][j]:.4f}"

        for label, col in (
            ("Hill bulk modulus", "B_Hill_GPa"),
            ("Hill shear modulus", "G_Hill_GPa"),
            ("Young modulus", "E_GPa"),
            ("Poisson ratio", "nu"),
            ("Debye temperature", "T_Debye_K"),
            ("Vickers hardness", "H_Vickers_GPa"),
        ):
            if label in line and "=" in line:
                v = try_float(line.split("=")[-1].strip().split()[0])
                if v is not None:
                    result[col] = f"{v:.4f}"

    return result
