"""
ui.py — Console UI layer for vca_tool.
───────────────────────────────────────
Responsibilities:
  • All user prompts (ask_yes_no, ask_float, ask_int, ask_choice, ask_str)
  • Setup wizards (param, species, concentration range, CASTEP command)
  • Step display (header, progress, result line, summary table)
  • ΔH_mix table rendering

No subprocess calls, no file I/O (except reading species for display),
no physics calculations. Pure presentation layer.
"""

from __future__ import annotations

import multiprocessing
import os
import sys
from pathlib import Path

import castep_io
from castep_io import ACTIVE_ENGINE, param_smart_defaults, validate_vca_pair
from workflow import (
    DONE,
    FAILED,
    PENDING,
    SKIPPED,
    STATUS_ICON,
    ExecResult,
    RunState,
    Step,
    cmd_is_valid,
    mixing_enthalpy,
)

VERSION = "5.1"


# ─────────────────────────────────────────────────────────────────────────────
# Low-level input helpers
# ─────────────────────────────────────────────────────────────────────────────


def ask_yes_no(question: str, default: bool | None = None) -> bool:
    """Prompt a yes/no question; loops until valid input."""
    hint = {True: "[Y/n]", False: "[y/N]", None: "[y/n]"}[default]
    while True:
        raw = input(f"  {question} {hint}: ").strip().lower()
        if raw == "" and default is not None:
            return default
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False


def ask_float(prompt: str, lo: float = 0.0, hi: float = 1.0) -> float:
    """Prompt for a float in [lo, hi]; loops on invalid input."""
    while True:
        try:
            value = float(input(f"  {prompt}").strip())
            if lo <= value <= hi:
                return value
            print(f"  Must be in [{lo}, {hi}].")
        except ValueError:
            print("  Not a valid number.")


def ask_int(prompt: str, lo: int = 1) -> int:
    """Prompt for an integer ≥ lo; loops on invalid input."""
    while True:
        try:
            value = int(input(f"  {prompt}").strip())
            if value >= lo:
                return value
            print(f"  Must be ≥ {lo}.")
        except ValueError:
            print("  Not a valid integer.")


def ask_str(prompt: str, default: str = "") -> str:
    """Prompt for a string; returns default on empty input."""
    raw = input(f"  {prompt}").strip()
    return raw if raw else default


def ask_choice(options: list[str], default: str) -> str:
    """
    Strict word selection from a fixed list; loops on unknown input.
    Empty input returns default.
    """
    while True:
        raw = input(f"  └  [{default}]: ").strip()
        if not raw:
            return default
        for option in options:
            if option.lower() == raw.lower():
                return option
        print(f"  ⚠  '{raw}' is not valid. Choose: {', '.join(options)}")


# ─────────────────────────────────────────────────────────────────────────────
# .param wizard
# ─────────────────────────────────────────────────────────────────────────────

_TASKS = [
    "GeometryOptimization",
    "SinglePoint",
    "ElasticConstants",
    "MolecularDynamics",
    "Phonon",
    "BandStructure",
    "Optics",
]

_XC_FUNCTIONALS = [
    "PBE",
    "PBEsol",
    "WC",
    "PW91",
    "RPBE",
    "LDA",
    "RSCAN",
    "PBE0",
    "HSE06",
]


def wizard_param(cell_path: Path, species_list: list[str], is_vca: bool) -> Path:
    """
    Interactively create a .param file if one does not exist.
    Returns the path to the (existing or newly created) .param file.
    """
    param_path = cell_path.with_suffix(".param")
    if param_path.exists():
        print(f"  .param  : {param_path.name} (found)")
        return param_path

    defaults = param_smart_defaults(species_list, is_vca)

    print("\n── Parameter setup ──")
    print("  No .param found — configuring physics settings.")
    print("  Press Enter to accept the [default] shown in brackets.\n")

    # 1/4 Task
    print("  ┌ 1/4  Task")
    for task_name in _TASKS:
        marker = "  (default)" if task_name == "GeometryOptimization" else ""
        print(f"  │  {task_name}{marker}")
    task = ask_choice(_TASKS, "GeometryOptimization")
    print()

    # 2/4 XC Functional
    print("  ┌ 2/4  XC Functional")
    print("  │  PBE     — standard for metals/alloys  (default)")
    print("  │  PBEsol  — best for ceramics/carbides (≤0.5% lattice error)")
    print("  │  WC      — Wu-Cohen, accurate lattice constants")
    print("  │  RSCAN   — best non-hybrid meta-GGA accuracy")
    print("  │  PBE0 / HSE06 — hybrid, 10-100× slower (avoid for VCA sweeps)")
    xc = ask_choice(_XC_FUNCTIONALS, "PBE")
    print()

    # 3/4 Cut-off energy
    recommended_cutoff = defaults["cut_off_energy"]
    hard_detected = defaults["hard_detected"]
    print("  ┌ 3/4  Cut-off Energy (eV)")
    if hard_detected:
        print(f"  │  ⚠  Hard elements detected: {hard_detected}")
        print("  │  Minimum 700 eV is mandatory for C, N, O, F, B.")
    else:
        print("  │  500 eV — production quality for metals/alloys")
        print("  │  700 eV — mandatory for hard elements (C, N, O, F)")
    raw_cutoff = ask_str(
        f"Cut-off energy [{recommended_cutoff}]: ", str(recommended_cutoff)
    )
    try:
        cut_off_energy = max(int(float(raw_cutoff.split()[0])), 100)
    except ValueError:
        cut_off_energy = recommended_cutoff
    print()

    # 4/4 Spin polarization
    magnetic_detected = defaults["magnetic_detected"]
    recommended_spin = defaults["spin_polarized"]
    spin_label = "true" if recommended_spin else "false"
    print("  ┌ 4/4  Spin Polarization")
    if magnetic_detected:
        print(f"  │  ⚠  Magnetic elements detected: {magnetic_detected}")
        print("  │  spin_polarized : true is mandatory.")
    else:
        print("  │  No magnetic elements detected — false is safe and ~2× faster.")
    print(f"  │  Detected {[s for s in species_list]}. Recommend spin: {spin_label}.")
    print("  │  Press Enter to accept.")
    spin_raw = ask_choice(["true", "false"], spin_label)
    spin_polarized = spin_raw == "true"
    print()

    # nextra_bands — automatic, not asked
    nextra_bands = defaults["nextra_bands"]
    print(f"  ℹ  nextra_bands : {nextra_bands}  (auto-selected for this system)")
    print()

    castep_io.write_param(
        param_path, task, xc, cut_off_energy, spin_polarized, nextra_bands
    )
    print(f"  ✓ Written: {param_path.name}")
    return param_path


# ─────────────────────────────────────────────────────────────────────────────
# Species wizard
# ─────────────────────────────────────────────────────────────────────────────


def _warn_cell_size(cell_path: Path) -> None:
    """Print warnings for oversized or conventional cells."""
    n_atoms = castep_io.atom_count(cell_path)
    if n_atoms > 8:
        print(f"\n  ⚠  Large cell: {n_atoms} atoms.")
        print(f"     ~{(n_atoms // 2) ** 3}× slower than a 2-atom primitive cell.")
        print("     Tip: export a primitive CIF from Materials Project.")
        if not ask_yes_no("Continue?", default=False):
            sys.exit(0)
    if castep_io.is_conventional_cell(cell_path):
        print(
            "\n  ⚠  Warning: Detected a conventional cell."
            " Calculation will be 64× slower!"
            " Export Primitive CIF from Materials Project."
        )


def wizard_species(cell_path: Path, cli_species: list[str] | None) -> tuple[str, str]:
    """
    Determine (species_a, species_b) pair for a VCA sweep.

    Interactive UX (no --species flag):
      Shows the elements present in the .cell and asks:
        1. Which element to substitute (must exist in .cell → A, x=0 pure)
        2. Replace with what element (B, x=1 pure)
      A == B is rejected here — use --species Ti Ti or --single for that.

    CLI shortcut:
      --species Ti Nb  → validates pair and skips wizard
      --species Ti Ti  → detected in main.py before this function is called
    """
    found_species = castep_io.read_species(cell_path)

    if cli_species:
        species_a = cli_species[0].capitalize()
        species_b = cli_species[1].capitalize()
        compat = validate_vca_pair(species_a, species_b)
        if compat.error:
            # Warn but do NOT block — user explicitly chose this pair.
            # They may be studying ceramics (Ti+C) or other cross-sublattice systems.
            print(f"\n  ⚠  Physics note: {compat.message}")
            print("     Proceeding as requested (you passed --species explicitly).\n")
        elif not compat.ok:
            print(f"\n  ⚠  {compat.message}")
        _warn_cell_size(cell_path)
        return species_a, species_b

    print("\n── Species ──")
    print(f"  Elements in .cell: {', '.join(found_species)}")
    print("  x=0 → pure A (the element you substitute FROM)")
    print("  x=1 → pure B (the element you substitute TO)")
    print("  All other sublattices (e.g. C in TiC) stay unchanged.")
    print(
        "  Tip: enter the same element twice (e.g. Ti Ti) to run a single-compound calc."
    )

    while True:
        default_a = found_species[0] if found_species else ""
        raw_a = ask_str(f"  Which element to substitute? [{default_a}]: ")
        species_a = (raw_a or default_a).capitalize()

        if not species_a:
            print("  ⚠  Required.")
            continue
        if species_a not in found_species:
            print(
                f"  ⚠  '{species_a}' not in .cell  (found: {', '.join(found_species)})"
            )
            continue

        raw_b = ask_str(f"  Replace {species_a} with (or same for single-compound): ")
        species_b = raw_b.capitalize()

        if not species_b:
            print("  ⚠  Required.")
            continue

        # A == B → signal single_mode by returning identical pair; main.py detects this
        if species_a == species_b:
            print(
                f"  ℹ  A == B ({species_a}) → will run as single-compound calculation."
            )
            _warn_cell_size(cell_path)
            return species_a, species_b

        compat = validate_vca_pair(species_a, species_b)
        if compat.error:
            # Show the physics warning but let the user decide
            print(f"\n  ⚠  Physics note: {compat.message}")
            if not ask_yes_no("  Proceed anyway?", default=False):
                continue
        elif not compat.ok:
            print(f"\n  ⚠  {compat.message}")
            if not ask_yes_no("Continue with this pair?", default=True):
                continue

        break

    _warn_cell_size(cell_path)
    return species_a, species_b


# ─────────────────────────────────────────────────────────────────────────────
# CASTEP command wizard
# ─────────────────────────────────────────────────────────────────────────────


def wizard_castep_cmd(override: str | None) -> str:
    """
    Resolve the engine MPI command string.
    Prompts for MPI process count, then validates the binary exists.
    Returns empty string for prepare-only mode.
    The default binary and command template come from ACTIVE_ENGINE in castep_io.py.
    """
    engine = ACTIVE_ENGINE
    cpu_count = multiprocessing.cpu_count()
    print(f"\n── {engine.name} — Parallelisation ──")
    print(f"  This machine: {cpu_count} logical cores")
    default_cores = min(cpu_count, 6)
    raw_cores = ask_str(f"MPI processes [{default_cores}]: ", str(default_cores))
    try:
        n_cores = max(1, int(raw_cores))
    except ValueError:
        n_cores = default_cores

    if override:
        cmd = override.replace("{ncores}", str(n_cores))
    else:
        bin_path = os.path.expanduser(engine.default_bin)
        cmd = engine.cmd_template.format(bin=bin_path, ncores=n_cores, seed="{seed}")

    cmd = os.path.expanduser(cmd.replace("{ncores}", str(n_cores)))
    print(f"\n── {engine.name} command ──\n  {cmd}")

    binary_part = cmd.replace("{seed}", "").strip()
    if cmd_is_valid(binary_part):
        print("  Binary : OK ✓")
        return cmd

    print("  ⚠  Binary not found.")
    while True:
        answer = ask_str("[Enter] fix command  /  'skip' = prepare .cell files only: ")
        if answer.lower() == "skip":
            print("  Prepare-only mode — no CASTEP will run.")
            return ""
        new_cmd = answer.replace("{ncores}", str(n_cores))
        if cmd_is_valid(new_cmd.replace("{seed}", "").strip()):
            return new_cmd
        print(f"  Not found: {new_cmd.split()[0]!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Step display
# ─────────────────────────────────────────────────────────────────────────────


def print_step_header(step: Step, total: int, state: RunState) -> None:
    x = step.concentration
    cmd_display = (
        state.castep_cmd.replace("{seed}", state.seed)
        if state.castep_cmd
        else "prepare only"
    )
    print(
        f"\n  ┌─ {step.idx}/{total - 1}"
        f"  x={x:.4f}"
        f"  {state.species_a}={round(1 - x, 4)}"
        f"  {state.species_b}={round(x, 4)}"
    )
    print(f"  │  $ {cmd_display}")


def print_step_result(
    step: Step,
    exec_result: ExecResult,
    proj_dir: Path,
    seed: str,
) -> None:
    """Print a single summary line (or error block) for a completed step."""
    if exec_result.skipped:
        print("  └─ ⊘ Skipped")
        return

    if step.status == DONE:
        conv_marker = "✓" if step.geom_converged == "yes" else "⚠ not converged"
        time_str = f"  ⏱ {float(step.wall_time_s):.0f}s" if step.wall_time_s else ""
        bulk_str = (
            f"  B={step.parsed.get('bulk_modulus_GPa', '')}GPa"
            if step.parsed.get("bulk_modulus_GPa")
            else ""
        )
        print(
            f"  └─ {conv_marker}"
            f"  H={step.enthalpy_eV}"
            f"  a={step.a_opt_ang}Å"
            f"{bulk_str}{time_str}"
        )
        if "no empty bands" in step.warnings:
            print("  │  ⚠  Increase nextra_bands in .param (try 20 or 30)")
        return

    # FAILED
    print(f"  └─ ✗ FAILED (rc={step.rc})")
    useful_stderr = [
        line
        for line in exec_result.stderr_tail
        if line.strip() and "PMIX" not in line and not line.startswith("[")
    ][-5:]
    for line in useful_stderr:
        print(f"     │ {line}")

    castep_log = proj_dir / step.step_dir / f"{seed}.castep"
    if castep_log.exists():
        all_lines = castep_log.read_text(errors="replace").splitlines()
        error_lines = [
            ln
            for ln in all_lines[-40:]
            if any(
                kw in ln.lower()
                for kw in ("error", "abort", "fatal", "failed", "warning")
            )
        ]
        for line in error_lines[-4:]:
            print(f"     │ {line.strip()}")
    else:
        print("     │ No .castep output — binary may have crashed immediately.")
        print(f"     │ Run manually:  cd '{proj_dir / step.step_dir}'")

    if step.warnings:
        print(f"     │ {step.warnings}")


def print_single_result(step: Step, seed: str) -> None:
    """Single-compound mode result line."""
    h = step.parsed.get("enthalpy_eV", "—")
    a = step.parsed.get("a_opt_ang", "—")
    t = step.parsed.get("wall_time_s", "—")
    icon = STATUS_ICON.get(step.status, "?")
    print(f"\n  [{icon}] {seed}  Enthalpy: {h} eV  |  a = {a} Å  |  Time: {t}s\n")


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────


def print_summary(state: RunState) -> None:
    """Print the full results table + ΔH_mix after the sweep completes."""
    steps = state.steps
    W = 90

    print(f"\n{'═' * W}")
    print(f"  {state.species_a}(1-x){state.species_b}(x)  —  {state.proj_dir.name}")
    print(
        f"  {'#':>3}  {'x':>7}  {'Status':<8}  {'H (eV)':>16}"
        f"  {'a (Å)':>8}  {'B (GPa)':>7}  conv"
    )
    print(
        f"  {'─' * 3}  {'─' * 7}  {'─' * 8}  {'─' * 16}  {'─' * 8}  {'─' * 7}  {'─' * 4}"
    )

    for step in steps:
        flag = " ⚠" if step.geom_converged == "no" and step.status == DONE else ""
        bulk = step.parsed.get("bulk_modulus_GPa", "—") or "—"
        print(
            f"  {STATUS_ICON.get(step.status, '?')}{step.idx:>2}"
            f"  {step.concentration:>7.4f}"
            f"  {step.status:<8}"
            f"  {(step.enthalpy_eV or '—'):>16}"
            f"  {(step.a_opt_ang or '—'):>8}"
            f"  {str(bulk):>7}"
            f"  {(step.geom_converged or '—')}{flag}"
        )

    counts = {
        st: sum(1 for s in steps if s.status == st)
        for st in (DONE, SKIPPED, FAILED, PENDING)
    }
    print(f"{'═' * W}")
    print(
        f"  ✓ {counts[DONE]}  ⊘ {counts[SKIPPED]}"
        f"  ✗ {counts[FAILED]}  · {counts[PENDING]}"
    )

    dh_data = mixing_enthalpy(steps)
    if dh_data:
        print("\n  ── ΔH_mix ──────────────────────────────────────")
        print(f"  {'x':>7}   {'H (eV)':>16}   {'ΔH (meV/f.u.)':>14}")
        print(f"  {'─' * 7}   {'─' * 16}   {'─' * 14}")
        for x_val, h_val, dh_val in dh_data:
            print(f"  {x_val:>7.4f}   {h_val:>16.6f}   {dh_val:>+14.2f}")
        print()
    elif counts[DONE] > 0 and counts[FAILED] == 0:
        print("  ⚠  ΔH_mix skipped — x=0 or x=1 endpoint missing/not converged.")
