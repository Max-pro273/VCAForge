"""
ui.py — Console UI layer for vca_tool.
───────────────────────────────────────
Responsibilities:
  • All user prompts (ask_yes_no, ask_float, ask_int, ask_choice, ask_str)
  • Setup wizards (param, species, concentration range, CASTEP command)
  • Step display (header, progress, result line, summary table)
  • VEC Stability Predictor (pre-run analysis + interactive guard)
  • Elastic step display (unified inline format inside step box)
  • ΔH_mix table rendering

No subprocess calls, no file I/O (except reading species for display),
no physics calculations. Pure presentation layer.
"""

from __future__ import annotations

import multiprocessing
import os
import shutil
import sys
import time
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

VERSION = "5.2"

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _fmt_time(seconds: float) -> str:
    """Format a duration as '45s' or '1m 12s'."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    return f"{s // 60}m {s % 60}s"


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

# Tasks always available (compatible with VCA MIXTURE + ultrasoft pseudopotentials)
_TASKS = [
    "GeometryOptimization",
    "SinglePoint",
    "MolecularDynamics",
    "BandStructure",
    "Optics",
]

# Tasks requiring NCP + no MIXTURE — available only in single-compound mode.
# CASTEP explicitly blocks strain-field response for MIXTURE atoms.
# For VCA elastic constants use the finite-strain workflow:
#   GeometryOptimization → elastic_workflow.run_finite_strain_elastic()
_NCP_ONLY_TASKS = ["ElasticConstants", "Phonon"]

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

    # 1/4 Task — NCP tasks (ElasticConstants, Phonon) only shown for single-compound mode
    available_tasks = _TASKS if is_vca else (_TASKS + _NCP_ONLY_TASKS)
    print("  ┌ 1/4  Task")
    for task_name in available_tasks:
        marker = "  (default)" if task_name == "GeometryOptimization" else ""
        print(f"  │  {task_name}{marker}")
    if is_vca:
        print(
            "  │  ℹ  ElasticConstants / Phonon are disabled for VCA sweeps.\n"
            "  │     After GeometryOptimization completes, use --elastic flag\n"
            "  │     to run the finite-strain Cij workflow automatically."
        )
    task = ask_choice(available_tasks, "GeometryOptimization")

    # NCP: only ElasticConstants/Phonon in single-compound mode need norm-conserving PSP
    _NCP_TASKS = {"ElasticConstants", "Phonon"}
    needs_ncp = task in _NCP_TASKS
    if needs_ncp:
        print(
            f"\n  ⚠  {task} requires norm-conserving pseudopotentials (NCP).\n"
            "     Ultrasoft (default) will crash: "
            "'strain field response with ultrasoft PSP not implemented'.\n"
            "\n"
            "     The tool will automatically inject SPECIES_POT NCP into your .cell.\n"
            "     Cutoff is raised to ≥900 eV for hard elements (C, N, O).\n"
            "\n"
            "     ── Recommended workflow ──────────────────────────────────\n"
            "     Step 1: GeometryOptimization (USP, fast)\n"
            "             → produces TiC-out.cell with relaxed geometry\n"
            "     Step 2: ElasticConstants on the relaxed cell (NCP, slower)\n"
            "             python main.py TiC-out.cell --single\n"
            "             → results written to TiC-out.elastic\n"
            "\n"
            "     The .elastic file contains: Cij tensor, bulk/shear/Young modulus,\n"
            "     Poisson ratio, Debye temperature, Vickers hardness.\n"
            "     Parse it with: castep_io.parse_elastic_file(path)\n"
        )
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

    # 3/4 Cut-off energy — NCP needs higher cutoff than USP
    recommended_cutoff = defaults["cut_off_energy"]
    hard_detected = defaults["hard_detected"]
    if needs_ncp and hard_detected:
        recommended_cutoff = max(recommended_cutoff, 900)
    elif needs_ncp:
        recommended_cutoff = max(recommended_cutoff, 700)
    print("  ┌ 3/4  Cut-off Energy (eV)")
    if needs_ncp:
        print("  │  ℹ  NCP requires higher cutoff than USP:")
        print("  │     ≥900 eV for hard elements (C, N, O), ≥700 eV for pure metals.")
    if hard_detected:
        print(f"  │  ⚠  Hard elements detected: {hard_detected}")
        print("  │  Minimum 700 eV (USP) / 900 eV (NCP) for C, N, O, F, B.")
    else:
        print("  │  500 eV — production quality for metals/alloys (USP)")
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
    if needs_ncp:
        print("  ℹ  Injecting SPECIES_POT NCP block into .cell file …")
        castep_io.inject_species_pot_ncp(cell_path)
        print("  ✓  .cell updated with NCP pseudopotentials")
    print()

    castep_io.write_param(
        param_path,
        task,
        xc,
        cut_off_energy,
        spin_polarized,
        nextra_bands,
        ncp=needs_ncp,
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
# CASTEP binary auto-discovery
# ─────────────────────────────────────────────────────────────────────────────

# Common install locations to probe when castep.mpi is not in PATH.
# Ordered from most-specific to most-generic.
_CASTEP_SEARCH_DIRS = [
    "~/Applications/CASTEP*/bin/linux*/castep.mpi",
    "~/castep*/bin/linux*/castep.mpi",
    "/opt/castep*/bin/castep.mpi",
    "/usr/local/bin/castep.mpi",
    "/usr/bin/castep.mpi",
    "~/bin/castep.mpi",
]

_CASTEP_SERIAL_NAMES = ["castep.serial", "castep"]


def _find_castep_binary() -> str | None:
    """
    Auto-discover the CASTEP MPI binary without any user interaction.

    Search order:
      1. PATH  (castep.mpi, then castep.serial, then castep)
      2. Common installation directories via glob expansion
      3. Returns None if nothing is found
    """
    import glob as _glob

    # 1. PATH
    for name in ["castep.mpi"] + _CASTEP_SERIAL_NAMES:
        found = shutil.which(name)
        if found:
            return found

    # 2. Common directories (glob, expand ~)
    for pattern in _CASTEP_SEARCH_DIRS:
        matches = sorted(_glob.glob(os.path.expanduser(pattern)))
        if matches:
            return matches[-1]  # most recent version (alphabetically last)

    return None


def _build_castep_cmd(bin_path: str, n_cores: int) -> str:
    """
    Build the final CASTEP command from a binary path and core count.
    Uses mpirun for .mpi binaries, plain path for serial.
    """
    name = Path(bin_path).name
    if "mpi" in name.lower():
        return f"mpirun -n {n_cores} {bin_path} {{seed}}"
    return f"{bin_path} {{seed}}"


# ─────────────────────────────────────────────────────────────────────────────
# CASTEP command wizard
# ─────────────────────────────────────────────────────────────────────────────


def wizard_castep_cmd(override: str | None) -> str:
    """
    Resolve the CASTEP command string with minimal friction.

    UX design
    ─────────
    The user should never need to type a full path manually.
    This wizard:
      1. If --castep-cmd was supplied → extract binary, validate, ask cores only.
      2. Otherwise → auto-discover binary, show what was found, ask cores only.
      3. If auto-discovery fails → ask for path once, then ask cores.
      4. 'skip' at any prompt → prepare-only mode (no CASTEP runs).

    The final command is always of the form:
        mpirun -n N /path/to/castep.mpi {seed}   (MPI binary)
        /path/to/castep.serial {seed}             (serial binary)

    The user never needs to type {seed} — it is appended automatically.
    """
    cpu_count = multiprocessing.cpu_count()

    # ── Extract binary from --castep-cmd if supplied ──────────────────────────
    if override:
        # Accept any of these forms:
        #   mpirun -n 6 /path/castep.mpi {seed}
        #   /path/castep.mpi {seed}
        #   /path/castep.mpi
        #   mpirun -n 6 /path/castep.mpi
        # We extract only the binary (the last token that looks like a path)
        tokens = override.replace("{seed}", "").split()
        bin_path = None
        for tok in reversed(tokens):
            tok_exp = os.path.expanduser(tok)
            if "castep" in tok.lower() or Path(tok_exp).is_file():
                bin_path = tok_exp
                break
        if bin_path is None:
            bin_path = os.path.expanduser(tokens[-1]) if tokens else ""
    else:
        bin_path = _find_castep_binary()

    print(f"\n── CASTEP — setup ──")
    print(f"  This machine: {cpu_count} logical cores")

    # ── Validate binary or ask for it ─────────────────────────────────────────
    while True:
        if bin_path and cmd_is_valid(bin_path):
            print(f"  Binary : {bin_path}  ✓")
            break
        if bin_path:
            print(f"  ✗  Not found: {bin_path!r}")
        else:
            print("  ✗  CASTEP binary not found in PATH or common install locations.")

        print("  Enter path to castep.mpi  (or 'skip' for prepare-only mode):")
        answer = ask_str("  Path: ").strip()
        if answer.lower() == "skip":
            print(
                "  Prepare-only mode — .cell files will be written but CASTEP won't run."
            )
            return ""
        bin_path = os.path.expanduser(answer)

    # ── Ask for MPI process count ──────────────────────────────────────────────
    raw_cores = ask_str(f"  MPI processes [{cpu_count}]: ", str(cpu_count))
    try:
        n_cores = max(1, int(raw_cores))
    except ValueError:
        n_cores = cpu_count

    cmd = _build_castep_cmd(bin_path, n_cores)
    print(f"  Command: {cmd.replace('{seed}', '<seed>')}")
    return cmd


# ─────────────────────────────────────────────────────────────────────────────
# Draw box
# ─────────────────────────────────────────────────────────────────────────────

import textwrap


def draw_box(text: str, width: int = 74, padding: int = 2) -> str:
    """
    Draw an ASCII box around multiline text, wrapping long lines automatically.
    Maintains the UI style of vca_tool.
    """
    inner_width = width - (padding * 2) - 2
    out_lines = [f"  ┌{'─' * (width - 2)}┐"]

    for paragraph in text.splitlines():
        if not paragraph.strip():
            # Preserve empty lines
            out_lines.append(f"  │{' ' * (width - 2)}│")
            continue

        # Wrap long paragraphs safely
        wrapped = textwrap.wrap(paragraph, width=inner_width)
        for line in wrapped:
            # Left-align the text, pad the right side with spaces
            out_lines.append(f"  │{' ' * padding}{line:<{inner_width}}{' ' * padding}│")

    out_lines.append(f"  └{'─' * (width - 2)}┘")
    return "\n".join(out_lines)


# ─────────────────────────────────────────────────────────────────────────────
# Step display
# ─────────────────────────────────────────────────────────────────────────────


def print_step_header(step: Step, total: int, state: RunState) -> None:
    x = step.concentration
    cmd = (
        state.castep_cmd.replace("{seed}", state.seed)
        if state.castep_cmd
        else "prepare only"
    )
    # Inline VEC annotation
    vec_tag = ""
    try:
        from castep_io import _NONMETALS as _NM
        from elasticity import vec_for_concentration, vec_stability_band

        all_sp = castep_io.read_species(state.proj_dir.parent / f"{state.seed}.cell")
        nonmetal = next(
            (
                s
                for s in all_sp
                if s.capitalize() in _NM
                and s.capitalize() not in {state.species_a, state.species_b}
            ),
            None,
        )
        vec = vec_for_concentration(state.species_a, state.species_b, x, nonmetal)
        band = vec_stability_band(vec)
        icon = {"green": "●", "yellow": "◑", "red": "○"}[band]
        vec_tag = f"  VEC={vec:.2f}{icon}"
    except Exception:
        pass

    print(
        f"\n  ┌─ {step.idx}/{total - 1}"
        f"  x={x:.4f}"
        f"  {state.species_a}={round(1 - x, 4)}"
        f"  {state.species_b}={round(x, 4)}"
        f"{vec_tag}"
    )
    print(f"  │  $ {cmd}")


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
        t_str = _fmt_time(float(step.wall_time_s)) if step.wall_time_s else "—"
        a_str = f"a={step.a_opt_ang}Å" if step.a_opt_ang else ""
        print(
            f"  │  ▶ Geometry Optimization … {conv_marker}"
            f"  ({t_str})  [{a_str}  H={step.enthalpy_eV} eV]"
        )
        if "no empty bands" in step.warnings:
            print("  │  ⚠  Increase nextra_bands in .param (try 20 or 30)")
        # Closing line printed by caller after elastic sub-step (if any)
        return

    # FAILED
    print(f"  │  ✗ FAILED (rc={step.rc})")
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
    print("  └─ ✗ Step failed.")


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
        print("\n  ── H(x) deviation from Vegard linear mixing ─────────────────────")
        print("  ⚠  VCA ΔH values are dominated by pseudopotential offsets between")
        print("     species (~eV range), NOT chemical mixing energy (~meV range).")
        print("     Use lattice parameter and Cij vs x for quantitative analysis.")
        print(f"\n  {'x':>7}   {'H_total (eV)':>16}   {'ΔH_Vegard (meV/cell)':>22}")
        print(f"  {'─' * 7}   {'─' * 16}   {'─' * 22}")
        for x_val, h_val, dh_val in dh_data:
            print(f"  {x_val:>7.4f}   {h_val:>16.6f}   {dh_val:>+22.2f}")
        print()
    elif counts[DONE] > 0 and counts[FAILED] == 0:
        print("  ⚠  ΔH skipped — x=0 or x=1 endpoint missing/not converged.")


# ─────────────────────────────────────────────────────────────────────────────
# VEC Stability Predictor
# ─────────────────────────────────────────────────────────────────────────────


def print_vec_stability_table(
    species_a: str,
    species_b: str,
    concentrations: list[float],
    nonmetal: str | None,
) -> None:
    """
    Print a VEC stability table for all planned concentration steps.

    Example output:
      ── VEC Stability Forecast ──────────────────────────────────────
       x       VEC    Status
      ─────  ──────  ──────────────────────────────────────────────────
       0.00   8.00   ● Stable — SCF converges quickly
       0.25   8.25   ◑ Yellow zone — SCF may need extra iterations
       0.50   8.50   ○ RED ZONE — Born instability likely (C44 → 0)
    """
    from elasticity import vec_for_concentration, vec_stability_band

    _BAND_LABEL = {
        "green": "● Stable — SCF converges quickly",
        "yellow": "◑ Yellow zone — SCF may need extra iterations",
        "red": "○ RED ZONE — Born instability likely (C44 → 0)",
    }

    print("\n  ── VEC Stability Forecast " + "─" * 46)
    print(f"  {'x':>6}   {'VEC':>6}   Status")
    print(f"  {'─' * 6}   {'─' * 6}   {'─' * 50}")
    for x in concentrations:
        vec = vec_for_concentration(species_a, species_b, x, nonmetal)
        band = vec_stability_band(vec)
        print(f"  {x:>6.4f}   {vec:>6.2f}   {_BAND_LABEL[band]}")
    print()


def ask_vec_guard(
    species_a: str,
    species_b: str,
    concentrations: list[float],
    nonmetal: str | None,
    vec_threshold: float = 8.4,
) -> list[float] | None:
    """
    Analyse VEC stability for all planned steps.  If any step has
    VEC > vec_threshold, display an interactive warning and ask the user
    whether to skip the dangerous steps or proceed anyway.

    Returns
    ───────
    • list[float]   — concentrations to actually run (may be filtered)
    • None          — if no steps exceed the threshold (nothing to do)

    The VEC > 8.4 threshold corresponds to the Band Jahn-Teller instability
    observed in Ti(1-x)Nb(x)C systems.  CASTEP will typically fail to
    converge the SCF cycle or yield C44 < 0 for these compositions.
    """
    from elasticity import vec_for_concentration, vec_stability_band

    red_steps = [
        x
        for x in concentrations
        if vec_for_concentration(species_a, species_b, x, nonmetal) > vec_threshold
    ]
    if not red_steps:
        return None  # all steps are safe — caller can proceed normally

    # Find the first dangerous concentration to include in the warning
    x_first_red = red_steps[0]
    vec_first = vec_for_concentration(species_a, species_b, x_first_red, nonmetal)

    # Boundary: last safe x
    safe_steps = [x for x in concentrations if x not in red_steps]
    x_safe_max = max(safe_steps) if safe_steps else None
    skip_label = (
        f"run only x ≤ {x_safe_max:.4f}"
        if x_safe_max is not None
        else "skip all unstable steps"
    )

    message = (
        f"⚠ WARNING: High Valence Electron Concentration (VEC > {vec_threshold}) "
        f"detected for x ≥ {x_first_red:.4f} (VEC = {vec_first:.2f}).\n\n"
        f"VCA pseudo-atoms with VEC > {vec_threshold} exhibit strong electronic "
        f"instability (Band Jahn-Teller effect). CASTEP will likely fail "
        f"to converge the SCF cycle or yield C44 < 0 (Born instability).\n\n"
        f"Affected steps ({len(red_steps)}): "
        + "  ".join(f"x={v:.4f}" for v in red_steps[:6])
        + (" …" if len(red_steps) > 6 else "")
        + "\n\n"
        f"[1] Skip unstable steps ({skip_label}) ← Recommended\n"
        f"[2] Proceed anyway (may hang for hours or crash)"
    )

    print(f"\n{draw_box(message)}")

    while True:
        try:
            raw = input("  Choice [1]: ").strip()
        except EOFError:
            raw = "1"
        if raw in {"", "1"}:
            skipped_note = (
                f"Only x ≤ {x_safe_max:.4f} will be computed."
                if x_safe_max is not None
                else "All planned steps are unstable and will be skipped."
            )
            print(
                f"\n  ✓  Skipping {len(red_steps)} unstable step(s) with VEC > {vec_threshold}.\n"
                f"     {skipped_note}\n"
                f"     Elastic constants for skipped x will use Vegard interpolation.\n"
            )
            return safe_steps
        if raw == "2":
            print(
                f"\n  ⚠  Proceeding with all steps.  Watch for 'Born stability violated'\n"
                f"     in the elastic output for x ≥ {x_first_red:.4f}.\n"
            )
            return list(concentrations)
        print("  Enter 1 or 2.")


# ─────────────────────────────────────────────────────────────────────────────
# Elastic step inline display
# ─────────────────────────────────────────────────────────────────────────────


def print_elastic_progress(n_strains: int, vec: float, nextra: int) -> None:
    """
    Print the 'starting' line for the elastic sub-step.

      │  ▶ Elastic Tensors (6 strains, nextra=19, VEC=8.30) ...
    """
    print(
        f"  │  ▶ Elastic Tensors ({n_strains} strains,"
        f"  nextra_bands={nextra},  VEC={vec:.2f}) …",
        flush=True,
    )


def print_elastic_result(elastic_data: dict, elapsed_s: float) -> None:
    """
    Print the elastic result line integrated into the step box.

    Success:
      │  ▶ Elastic Tensors (6 strains) ... ✓ (1m 12s) [B=259 GPa, E=474 GPa]

    Failure:
      │  ⚠ Elasticity failed: Born stability violated (Expected for VEC > 8.4).
    """
    t = _fmt_time(elapsed_s)

    if "_elastic_error" in elastic_data:
        err = elastic_data["_elastic_error"]
        print(f"  │  ⚠  Elasticity failed: {err}")
        print("  └─ ✗ Elastic step failed.")
        return

    if "C11" not in elastic_data:
        print(f"  │  ⚠  Elasticity: no usable data returned")
        print("  └─ ✗ Elastic step failed.")
        return

    b = elastic_data.get("B_Hill_GPa", "—")
    g = elastic_data.get("G_Hill_GPa", "—")
    e = elastic_data.get("E_GPa", "—")
    c11 = elastic_data.get("C11", "—")
    c12 = elastic_data.get("C12", "—")
    c44 = elastic_data.get("C44", "—")
    src = elastic_data.get("elastic_source", "CASTEP")
    tag = "  [Vegard]" if "Vegard" in src else ""
    r2 = elastic_data.get("elastic_R2_min", "")
    r2_tag = f"  R²={r2}" if r2 and r2 != "N/A" else ""
    note = elastic_data.get("elastic_quality_note", "")

    print(f"  │  ✓ Elastic Tensors ({t}){tag}  [B={b}  G={g}  E={e} GPa]{r2_tag}")
    print(f"  │     C11={c11}  C12={c12}  C44={c44} GPa")
    if note:
        print(f"  │  ⚠  {note}")
    print("  └─ ✓ Elastic step completed.")
