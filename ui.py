"""
ui.py  —  All console prompts and wizards for VCAForge.
════════════════════════════════════════════════════════
No subprocess calls, no file I/O beyond reading .cell for hints.
Pure presentation: prompts, wizards, step boxes, tables.

New UX (v6.0)
─────────────
One composite prompt replaces the old multi-step species wizard:

  Enter elements for the VCA mix …
  or press Enter to run the pure template [TiC]:

  * Enter alone       → single_mode, pure template compound
  * One element       → single_mode, that element on the template sublattice
  * Two+ elements     → VCA sweep on that sublattice

This lets any .cell file act as a pure topology template:
  template = TiC,  input = "Nb V"  →  Nb(1-x)V(x)C sweep.
"""

from __future__ import annotations

import multiprocessing
import os
import shutil
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import config
import core_physics as phys
from orchestrator import (
    DONE,
    FAILED,
    PENDING,
    SKIPPED,
    ExecResult,
    RunState,
    Step,
    mixing_enthalpy,
)

# ─────────────────────────────────────────────────────────────────────────────
# Public dataclass: wizard result
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class WizardResult:
    """Everything the wizard learned from the user.

    Attributes:
        template_element: The element in the .cell template to be replaced.
        target_mix:       ``{element: fraction}`` for x = 1 endpoint.
                          Single element has fraction = 1.0.
        single_mode:      ``True`` when no sweep is needed.
        c_start:          Sweep start (0.0 for single mode).
        c_end:            Sweep end   (0.0 for single mode).
        n_steps:          Number of intervals (0 for single mode).
        nonmetal:         Detected anion element, or ``""``.
        run_elastic:      Whether to run the elastic sub-step.
    """

    template_element: str
    target_mix: dict[str, float]
    single_mode: bool
    c_start: float
    c_end: float
    n_steps: int
    nonmetal: str
    run_elastic: bool


# ─────────────────────────────────────────────────────────────────────────────
# Terminal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _tw(fallback: int = 88) -> int:
    """Return the terminal width, capped at 120 columns."""
    try:
        return min(shutil.get_terminal_size().columns, 120)
    except Exception:
        return fallback


def _section(title: str) -> None:
    """Print a  ── Title ──────────  section divider at terminal width."""
    w = _tw()
    inner = f"── {title} "
    print(f"\n{inner}{'─' * max(2, w - len(inner))}")


def _fmt_time(s: float) -> str:
    """Format seconds as ``Xs`` or ``Xm Ys``."""
    s = int(s)
    return f"{s}s" if s < 60 else f"{s // 60}m {s % 60}s"


# ─────────────────────────────────────────────────────────────────────────────
# Low-level input helpers
# ─────────────────────────────────────────────────────────────────────────────


def ask_yes_no(question: str, default: bool | None = None) -> bool:
    """Prompt for a yes/no answer and return a bool.

    Args:
        question: Question text (without trailing ``?``).
        default:  Pre-selected answer shown in brackets.
    """
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
    """Prompt for a float in [lo, hi]."""
    while True:
        try:
            v = float(input(f"  {prompt}").strip())
            if lo <= v <= hi:
                return v
            print(f"  Must be in [{lo:.4g}, {hi:.4g}].")
        except ValueError:
            print("  Not a valid number.")


def ask_int(prompt: str, lo: int = 1) -> int:
    """Prompt for an integer >= lo."""
    while True:
        try:
            v = int(input(f"  {prompt}").strip())
            if v >= lo:
                return v
            print(f"  Must be >= {lo}.")
        except ValueError:
            print("  Not a valid integer.")


def ask_str(prompt: str, default: str = "") -> str:
    """Prompt for a non-empty string, returning *default* on bare Enter."""
    raw = input(f"  {prompt}").strip()
    return raw if raw else default


def ask_choice(options: list[str], default: str) -> str:
    """Prompt for one of *options*, case-insensitive.

    Args:
        options: Valid choices.
        default: Shown in brackets; returned on bare Enter.
    """
    while True:
        raw = input(f"  [{default}]: ").strip()
        if not raw:
            return default
        for o in options:
            if o.lower() == raw.lower():
                return o
        print(f"  '{raw}' not valid.  Options: {', '.join(options)}")


# ─────────────────────────────────────────────────────────────────────────────
# .param wizard
# ─────────────────────────────────────────────────────────────────────────────

_TASKS_VCA = [
    "GeometryOptimization",
    "SinglePoint",
    "MolecularDynamics",
    "BandStructure",
    "Optics",
]
_TASKS_FULL = _TASKS_VCA + ["ElasticConstants", "Phonon"]
_XC_LIST = [
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
    """Return path to a .param file, creating it interactively if absent.

    Args:
        cell_path:    Path to the template .cell file.
        species_list: Elements detected in the .cell file.
        is_vca:       ``True`` for a VCA sweep run.
    """
    from castep.castep import (
        inject_ncp,
        smart_defaults,
        write_geomopt_param,
        write_singlepoint_param,
    )

    param_path = cell_path.with_suffix(".param")
    if param_path.exists():
        print(f"  .param : {param_path.name}  (found)")
        return param_path

    defs = smart_defaults(species_list, is_vca)
    _section("Parameter setup")
    print("  No .param found — configuring DFT settings.")
    print("  Press Enter to accept the default shown in [brackets].\n")

    # 1 / 5  Task
    tasks = _TASKS_VCA if is_vca else _TASKS_FULL
    task_hint = "\n".join(
        f"  │    {t}{'  <- default' if t == 'GeometryOptimization' else ''}"
        for t in tasks
    )
    vca_note = (
        "\n  │\n"
        "  │  · ElasticConstants / Phonon are disabled for VCA sweeps.\n"
        "  │    Use the integrated elastic workflow (prompted below)."
        if is_vca
        else ""
    )
    print(f"  ┌ 1/5  Task\n{task_hint}{vca_note}")
    task = ask_choice(tasks, "GeometryOptimization")
    needs_ncp = task in {"ElasticConstants", "Phonon"}
    if needs_ncp:
        print(
            textwrap.dedent(f"""
          ⚠  {task} requires norm-conserving pseudopotentials (NCP).
             Ultrasoft PSP will crash: 'strain field response not implemented'.
             The wizard will inject SPECIES_POT NCP into your .cell.
             Recommended cutoff: >= 900 eV for hard elements (C, N, O).
        """)
        )

    # 2 / 5  XC functional
    print(
        textwrap.dedent("""
      ┌ 2/5  XC Functional
      │    PBE    — standard for metals and alloys              <- default
      │    PBEsol — better lattice constants for carbides/nitrides
      │    WC     — Wu-Cohen, accurate lattice constants
      │    RSCAN  — best non-hybrid meta-GGA accuracy
      │    PBE0 / HSE06 — hybrid, 10-100x slower (avoid for VCA sweeps)""")
    )
    xc = ask_choice(_XC_LIST, "PBE")

    # 3 / 5  Cut-off energy
    rec = defs["cut_off_energy"]
    hard = defs["hard_detected"]
    if needs_ncp and hard:
        rec = max(rec, 900)
    elif needs_ncp:
        rec = max(rec, 700)
    hard_note = (
        f"  │  ⚠  Hard elements detected: {hard}\n"
        "  │     Minimum 700 eV (USP) / 900 eV (NCP) for C, N, O, F, B, H."
        if hard
        else "  │    500 eV — metals/alloys (USP)\n"
        "  │    700 eV — hard elements (USP, mandatory for C/N/O)"
    )
    print(f"\n  ┌ 3/5  Cut-off energy (eV)\n{hard_note}")
    raw = ask_str(f"  Cut-off [{rec}]: ", str(rec))
    try:
        cutoff = max(int(float(raw.split()[0])), 100)
    except (ValueError, IndexError):
        cutoff = rec

    # 4 / 5  Spin polarisation
    mag = defs["magnetic_detected"]
    spin = defs["spin_polarized"]
    mag_note = (
        f"  │  ⚠  Magnetic elements detected: {mag}\n"
        "  │     spin_polarized : true is mandatory."
        if mag
        else "  │    No magnetic elements detected.  false is safe and ~2x faster."
    )
    print(f"\n  ┌ 4/5  Spin polarisation\n{mag_note}")
    print(
        f"  │    Detected {species_list}.  Recommended: {'true' if spin else 'false'}."
    )
    spin = ask_choice(["true", "false"], "true" if spin else "false") == "true"

    # 5 / 5  Smearing width
    default_sw = config.SMEARING_VCA if is_vca else config.SMEARING_SINGLE
    smear_note = (
        "  │    VCA:    0.20 eV — fractional nuclear charge broadens bands.\n"
        "  │    Sharper values often diverge for intermediate x."
        if is_vca
        else "  │    Single compound:  0.10 eV — sharp Fermi edge."
    )
    print(
        "\n  ┌ 5/5  Smearing width (eV)\n  │    Electronic temperature for metals_method: dm."
    )
    print(smear_note)
    raw = ask_str(f"  Smearing [{default_sw:.2f}]: ", f"{default_sw:.2f}")
    try:
        smearing = max(0.01, float(raw.strip().rstrip("eEvV")))
    except ValueError:
        smearing = default_sw

    # Write
    nextra = defs["nextra_bands"]
    print(f"\n  · nextra_bands : {nextra}  (auto-selected)")
    if needs_ncp:
        print("  · Injecting SPECIES_POT NCP block into .cell …")
        inject_ncp(cell_path)
        print("  ✓ .cell updated with NCP pseudopotentials")
    print()

    if task in {"GeometryOptimization", "FiniteStrainElastic"}:
        write_geomopt_param(
            param_path, xc, cutoff, spin, nextra, smearing, ncp=needs_ncp
        )
    elif task == "SinglePoint":
        write_singlepoint_param(param_path, xc, cutoff, spin, nextra, ncp=needs_ncp)
    else:
        write_geomopt_param(
            param_path, xc, cutoff, spin, nextra, smearing, ncp=needs_ncp
        )

    print(f"  ✓ Written: {param_path.name}")
    return param_path


# ─────────────────────────────────────────────────────────────────────────────
# New composite species / mode wizard  (Block 2 of the spec)
# ─────────────────────────────────────────────────────────────────────────────


def wizard_mode(
    cell_path: Path,
    cli_elements: list[str] | None,
    cli_range: tuple[float, float, int] | None,
) -> WizardResult:
    """Ask one composite question that covers mode, elements, and range.

    Implements the three scenarios from the spec:

    * **Scenario A** — user presses Enter:
      ``single_mode=True``, pure template compound (e.g. TiC).

    * **Scenario B** — user enters one element (e.g. ``Nb``):
      ``single_mode=True``, that element replaces the template (e.g. NbC).

    * **Scenario C** — user enters two or more elements (e.g. ``Nb V``):
      ``single_mode=False``, VCA sweep on the template sublattice.

    Args:
        cell_path:    Path to the primitive template .cell file.
        cli_elements: Elements from ``--species`` CLI flag (or ``None``).
        cli_range:    ``(x0, x1, n)`` from ``--range`` CLI flag (or ``None``).

    Returns:
        :class:`WizardResult` with all parameters needed to build a run.
    """
    from castep import config as _ccfg
    from castep.castep import read_species

    found = read_species(cell_path)
    nonmetal = next((s for s in found if s.capitalize() in _ccfg.NONMETALS), "")
    metals = [s for s in found if s.capitalize() not in _ccfg.NONMETALS]
    template = metals[0] if metals else (found[0] if found else "")

    template_label = "".join(found)  # e.g. "TiC"
    _section("Species / mode")
    print(f"  Found sublattices in template: {found}")
    print(f"  Template compound             : {template_label}")
    print(
        textwrap.dedent(f"""
      Enter elements for the VCA mix separated by space (e.g. 'Nb V'),
      replace a single element (e.g. 'Nb'),
      or press Enter to run the pure template [{template_label}]:""")
    )

    # Resolve from CLI or interactive input.
    if cli_elements is not None:
        raw_elems = [e.capitalize() for e in cli_elements]
        print(f"  (using --species: {' '.join(raw_elems)})")
    else:
        raw = input("  > ").strip()
        raw_elems = [e.capitalize() for e in raw.split()] if raw else []

    # ── Scenario A: pure template ──────────────────────────────────────────
    if not raw_elems:
        print(f"\n  · Mode: single compound  ({template_label})")
        run_elastic = ask_yes_no(
            "  Run elastic constants after GeomOpt?", default=False
        )
        return WizardResult(
            template_element=template,
            target_mix={template: 1.0},
            single_mode=True,
            c_start=0.0,
            c_end=0.0,
            n_steps=0,
            nonmetal=nonmetal,
            run_elastic=run_elastic,
        )

    # ── Scenario B: single replacement ────────────────────────────────────
    if len(raw_elems) == 1:
        new_elem = raw_elems[0]
        nm_label = nonmetal if nonmetal else ""
        compound = f"{new_elem}{nm_label}"
        print(
            f"\n  · Mode: single compound  ({compound}  using {template_label} geometry)"
        )
        _warn_nonmetal_mix(new_elem, nonmetal)
        run_elastic = ask_yes_no(
            "  Run elastic constants after GeomOpt?", default=False
        )
        return WizardResult(
            template_element=template,
            target_mix={new_elem: 1.0},
            single_mode=True,
            c_start=0.0,
            c_end=0.0,
            n_steps=0,
            nonmetal=nonmetal,
            run_elastic=run_elastic,
        )

    # ── Scenario C: VCA sweep ─────────────────────────────────────────────
    elems = raw_elems
    if len(elems) == 2:
        # Binary: element A at x=0, element B at x=1.
        target_mix_x1 = {elems[0]: 0.0, elems[1]: 1.0}
        sweep_a, sweep_b = elems[0], elems[1]
    else:
        # Ternary+: ask for endpoint fractions.
        target_mix_x1, sweep_a, sweep_b = _ask_ternary_fracs(elems)

    print(
        f"\n  · Mode: VCA sweep  "
        f"{sweep_a}(1-x) → {sweep_b}(x)  on {template} sublattice"
    )

    # Range
    if cli_range:
        c_start, c_end, n_steps = cli_range
    else:
        print("\n── Concentration range ──────────────────────────────────────────────")
        c_start = ask_float("Start [0-1]: ")
        c_end = ask_float("End   [0-1]: ")
        n_steps = ask_int("Intervals (e.g. 8): ")

    # VEC reference table (informational — no pre-flight blocking).
    if len(elems) == 2 and nonmetal:
        import numpy as np

        planned_x = list(np.linspace(c_start, c_end, n_steps + 1))
        print_vec_table(elems[0], elems[1], planned_x, nonmetal)

    run_elastic = ask_yes_no(
        "\n  Run elastic constants after each GeomOpt?", default=False
    )
    return WizardResult(
        template_element=template,
        target_mix=target_mix_x1,
        single_mode=False,
        c_start=c_start,
        c_end=c_end,
        n_steps=n_steps,
        nonmetal=nonmetal,
        run_elastic=run_elastic,
    )


def _ask_ternary_fracs(
    elems: list[str],
) -> tuple[dict[str, float], str, str]:
    """Interactively gather endpoint fractions for 3+ elements.

    The first element is phased out (fraction = 0 at x=1).  The remaining
    elements share the sublattice at x=1; their fractions must sum to 1.
    The **last** element automatically receives whatever fraction is left,
    preventing under-specification.  Earlier elements are bounded so that
    the last element always retains at least a minimal share.

    Args:
        elems: Element symbols (first = dominant at x=0, phased out as x→1).

    Returns:
        ``(target_mix_at_x1, label_a, label_b)``
    """
    import sys as _sys

    new_elems = elems[1:]
    print(
        f"\n  {len(elems)} elements: {', '.join(elems)}\n"
        f"  \047{elems[0]}\047 is phased out (weight → 0 as x → 1).\n"
        f"  Assign fractions for the {len(new_elems)} elements at the x=1 endpoint\n"
        f"  (must sum to 1.000 — last element receives the remainder automatically):"
    )

    fracs: dict[str, float] = {elems[0]: 0.0}
    remaining = 1.0

    for i, e in enumerate(new_elems):
        is_last = i == len(new_elems) - 1
        if is_last:
            f = round(remaining, 10)
            if f < 1e-6:
                print(
                    f"\n  \u274c  No fraction left for \047{e}\047 "
                    f"(remaining = {remaining:.6f}).\n"
                    f"     Earlier fractions already sum to 1. "
                    f"Re-run and allocate less to the other elements."
                )
                _sys.exit(1)
            print(f"  Fraction of {e} at x=1  (auto-assigned remainder): {f:.4f}")
        else:
            # Reserve at least 1e-4 per remaining element after this one.
            n_after = len(new_elems) - i - 1
            hi = max(0.0, round(remaining - 1e-4 * n_after, 10))
            if hi < 1e-6:
                print(
                    f"\n  \u274c  No fraction available for \047{e}\047 — "
                    f"earlier allocations filled the budget.\n"
                    f"     Re-run and allocate less to the previous elements."
                )
                _sys.exit(1)
            f = ask_float(
                f"  Fraction of {e} at x=1  (remaining: {remaining:.4f}, max: {hi:.4f}): ",
                lo=0.0,
                hi=hi,
            )

        fracs[e] = f
        remaining = round(remaining - f, 10)

    # Float-drift guard: re-normalise the new elements only.
    total_new = sum(v for k, v in fracs.items() if k != elems[0])
    if abs(total_new - 1.0) > 1e-3 and total_new > 1e-9:
        scale = 1.0 / total_new
        fracs = {
            k: (0.0 if k == elems[0] else round(v * scale, 10))
            for k, v in fracs.items()
        }

    parts = "  ".join(f"{e}={fracs[e]:.4f}" for e in new_elems)
    chk = sum(fracs[e] for e in new_elems)
    print(f"\n  ✓  Mix at x=1:  {parts}  (sum={chk:.4f})")
    return fracs, elems[0], elems[-1]


def _warn_nonmetal_mix(elem: str, nonmetal: str) -> None:
    """Warn if the user is mixing a metal with the nonmetal sublattice."""
    from castep import config as _ccfg

    if elem in _ccfg.NONMETALS and nonmetal and elem != nonmetal:
        print(
            f"\n  ⚠  '{elem}' is a nonmetal — VCA on the anion sublattice is unusual.\n"
        )


# ─────────────────────────────────────────────────────────────────────────────
# CASTEP binary wizard
# ─────────────────────────────────────────────────────────────────────────────

_BINARY_SEARCH = [
    "~/Applications/CASTEP*/bin/linux*/castep.mpi",
    "~/castep*/bin/linux*/castep.mpi",
    "/opt/castep*/bin/castep.mpi",
    "/usr/local/bin/castep.mpi",
    "~/bin/castep.mpi",
]


def _find_binary() -> str | None:
    """Search PATH and common install locations for a CASTEP binary."""
    import glob

    for name in ["castep.mpi", "castep.serial", "castep"]:
        found = shutil.which(name)
        if found:
            return found
    for pat in _BINARY_SEARCH:
        matches = sorted(glob.glob(os.path.expanduser(pat)))
        if matches:
            return matches[-1]
    return None


def wizard_castep_cmd(override: str | None) -> str:
    """Prompt for the CASTEP execution command.

    Args:
        override: Value from ``--castep-cmd`` CLI flag, or ``None``.

    Returns:
        Shell command template with ``{seed}`` placeholder, or ``""`` for
        prepare-only mode.
    """
    from orchestrator import cmd_is_valid

    cpu = multiprocessing.cpu_count()
    if override:
        tokens = override.replace("{seed}", "").split()
        bin_path = next(
            (
                os.path.expanduser(t)
                for t in reversed(tokens)
                if "castep" in t.lower() or Path(os.path.expanduser(t)).is_file()
            ),
            os.path.expanduser(tokens[-1]) if tokens else "",
        )
    else:
        bin_path = _find_binary()

    _section("CASTEP")
    print(f"  Machine : {cpu} logical cores")

    while True:
        if bin_path and cmd_is_valid(bin_path):
            print(f"  Binary  : {bin_path}  ✓")
            break
        msg = (
            f"  ✗  Not found: {bin_path!r}"
            if bin_path
            else "  ✗  CASTEP binary not found in PATH or common install locations."
        )
        print(msg)
        ans = ask_str(
            "  Path to castep.mpi  (or 'skip' for prepare-only mode): "
        ).strip()
        if ans.lower() == "skip":
            print("  · Prepare-only mode — .cell files written, CASTEP will not run.")
            return ""
        bin_path = os.path.expanduser(ans)

    raw = ask_str(f"  MPI processes [{cpu}]: ", str(cpu))
    try:
        n = max(1, int(raw))
    except ValueError:
        n = cpu

    name = Path(bin_path).name
    cmd = (
        f"mpirun -n {n} {bin_path} {{seed}}"
        if "mpi" in name.lower()
        else f"{bin_path} {{seed}}"
    )
    print(f"  Command : {cmd.replace('{seed}', '<seed>')}")
    return cmd


# ─────────────────────────────────────────────────────────────────────────────
# Step box display
# ─────────────────────────────────────────────────────────────────────────────


def _vec_tag(state: RunState, x: float) -> str:
    """Build the ``VEC=8.00`` tag shown in the step header."""
    try:
        sp_a = state.species[0][0]
        fracs = [(sp_a, 1.0 - x)] + [(e, f * x) for e, f in state.species[1:]]
        vec = phys.vec_for_system(fracs, state.nonmetal or None)
        return f"  VEC={vec:.2f}"
    except Exception:
        return ""


def print_step_header(step: Step, total: int, state: RunState) -> None:
    """Print the opening ┌─ line of a step box.

    Args:
        step:  The current :class:`~orchestrator.Step`.
        total: Total number of steps in this run.
        state: Parent :class:`~orchestrator.RunState`.
    """
    x = step.concentration
    cmd = (
        state.castep_cmd.replace("{seed}", state.seed)
        if state.castep_cmd
        else "prepare only"
    )
    if len(state.species) == 2:
        sp_label = (
            f"  {state.species[0][0]}={round(1 - x, 4)}"
            f"  {state.species[1][0]}={round(x, 4)}"
        )
    else:
        fracs = [(state.species[0][0], round(1 - x, 4))] + [
            (e, round(f * x, 4)) for e, f in state.species[1:]
        ]
        sp_label = "  " + "  ".join(f"{e}={v}" for e, v in fracs)

    vtag = _vec_tag(state, x)
    print(f"\n  ┌─ {step.idx}/{total - 1}  x={x:.4f}{sp_label}{vtag}")
    print(f"  │  $ {cmd}")


def print_step_result(
    result: ExecResult, step: Step, proj_dir: Path, seed: str
) -> None:
    """Print the result line inside the step box.

    Args:
        result:   Subprocess execution result.
        step:     The completed :class:`~orchestrator.Step`.
        proj_dir: Project directory (for locating the .castep log on failure).
        seed:     CASTEP seed name.
    """
    if result.skipped:
        print("  └─ ⊘ Skipped")
        return

    if step.status == DONE:
        conv = "✓" if step.geom_converged == "yes" else "⚠ not converged"
        t = _fmt_time(float(step.wall_time_s)) if step.wall_time_s else "—"
        a = f"a={step.a_opt_ang} Å" if step.a_opt_ang else ""
        print(
            f"  └  {conv} Geometry Optimization   ({t})  [{a}  H={step.enthalpy_eV} eV]"
        )
        if "no empty bands" in step.warnings:
            print("     ⚠  Increase nextra_bands in .param (try 20 or 30)")
        return

    # FAILED
    print(f"  │  ✗  FAILED  (rc={step.rc})")
    useful = [
        ln
        for ln in result.stderr_tail
        if ln.strip() and "PMIX" not in ln and not ln.startswith("[")
    ][-5:]
    for ln in useful:
        print(f"  │     {ln}")

    log = proj_dir / step.step_dir / f"{seed}.castep"
    if log.exists():
        err_lines = [
            ln
            for ln in log.read_text(errors="replace").splitlines()[-40:]
            if any(k in ln.lower() for k in ("error", "abort", "fatal", "failed"))
        ]
        for ln in err_lines[-4:]:
            print(f"  │     {ln.strip()}")
    else:
        print("  │     No .castep output — binary may have crashed immediately.")
        print(f"  │     Run manually:  cd '{proj_dir / step.step_dir}'")

    if step.warnings:
        print(f"  │     {step.warnings}")
    print("  └─ ✗ Step failed.")


def print_single_result(step: Step, seed: str) -> None:
    """Print a compact result line for a single-compound run."""
    h = step.parsed.get("enthalpy_eV", "—")
    a = step.parsed.get("a_opt_ang", "—")
    t = step.parsed.get("wall_time_s", "—")
    ok = "✓" if step.status == DONE else "✗"
    print(f"\n  [{ok}] {seed}  H = {h} eV  |  a = {a} Å  |  {t}s\n")


# ─────────────────────────────────────────────────────────────────────────────
# Elastic sub-step display
# ─────────────────────────────────────────────────────────────────────────────


def print_elastic_start(n_strains: int, vec: float, nextra: int) -> None:
    """Print the elastic header line inside the step box."""
    print(
        f"  │  ▶ Elastic Tensors"
        f"  ({n_strains} strains  nextra={nextra}  VEC={vec:.2f}) …",
        flush=True,
    )


def elastic_progress_cb(line: str) -> None:
    """Callback for the elastic progress monitor — overwrites the current line."""
    print(line + " " * 4, end="\r", flush=True)


def elastic_progress_clear() -> None:
    """Clear the elastic progress bar line."""
    print(" " * 76, end="\r")


def print_elastic_result(data: dict[str, Any], elapsed: float) -> None:
    """Print the elastic result summary inside the step box.

    Args:
        data:    Dict returned by :func:`~castep.elastic.run_elastic`.
        elapsed: Wall-clock seconds for the elastic sub-step.
    """
    t = _fmt_time(elapsed)
    b = data.get("B_Hill_GPa", "—")
    g = data.get("G_Hill_GPa", "—")
    e = data.get("E_GPa", "—")
    c11 = data.get("C11", "—")
    c12 = data.get("C12", "—")
    c44 = data.get("C44", "—")
    r2 = data.get("elastic_R2_min", "")
    r2t = f"  R²={r2}" if r2 and r2 != "N/A" else ""
    src = data.get("elastic_source", "CASTEP")
    tag = "  [Vegard]" if "Vegard" in src else ""
    note = data.get("elastic_quality_note", "")
    print(f"  │  ✓ Elastic Tensors ({t}){tag}  [B={b}  G={g}  E={e} GPa]{r2t}")
    print(f"  │     C11={c11}  C12={c12}  C44={c44} GPa")
    if note:
        print(f"  │  ⚠  {note}")
    print("  └─ ✓ Step completed.")


def print_elastic_error(msg: str) -> None:
    """Print a failed-elastic notice and close the step box."""
    print(f"  │  ⚠  Elastic failed: {msg}")
    print("  └─ ✗ Elastic step failed.")


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

_STATUS_ICON = {DONE: "✓", SKIPPED: "⊘", FAILED: "✗", PENDING: "·"}


def print_summary(state: RunState) -> None:
    """Print the final results table for the run.

    Args:
        state: Completed (or partial) :class:`~orchestrator.RunState`.
    """
    steps = state.steps
    W = min(_tw(), 92)

    if len(state.species) == 2:
        sp_label = f"{state.species[0][0]}(1-x){state.species[1][0]}(x)"
    else:
        sp_label = " + ".join(
            f"{e}({f:.0%})" if i > 0 else f"{e}(1-x)"
            for i, (e, f) in enumerate(state.species)
        )

    print(f"\n{'═' * W}")
    print(f"  {sp_label}  —  {state.proj_dir.name}")
    print(
        f"  {'#':>4}  {'x':>7}  {'Status':<8}  {'H (eV)':>16}"
        f"  {'a (Å)':>8}  {'B (GPa)':>7}  conv"
    )
    print(
        f"  {'─' * 4}  {'─' * 7}  {'─' * 8}  {'─' * 16}"
        f"  {'─' * 8}  {'─' * 7}  {'─' * 4}"
    )
    for s in steps:
        flag = " ⚠" if s.geom_converged == "no" and s.status == DONE else ""
        B = s.parsed.get("bulk_modulus_GPa", "—") or "—"
        icon = _STATUS_ICON.get(s.status, "?")
        print(
            f"  {icon}{s.idx:>3}  {s.concentration:>7.4f}  {s.status:<8}"
            f"  {(s.enthalpy_eV or '—'):>16}"
            f"  {(s.a_opt_ang or '—'):>8}"
            f"  {str(B):>7}"
            f"  {(s.geom_converged or '—')}{flag}"
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

    dh = mixing_enthalpy(steps)
    if dh:
        print(
            textwrap.dedent("""
          ── ΔH deviation from Vegard linear mixing ──────────────────────
          · VCA ΔH values are dominated by pseudopotential offsets (~eV),
            not by chemical mixing energy (~meV).
            Use lattice parameter and Cij vs x for quantitative analysis.
        """)
        )
        print(f"  {'x':>7}   {'H_total (eV)':>16}   {'ΔH_Vegard (meV/cell)':>22}")
        print(f"  {'─' * 7}   {'─' * 16}   {'─' * 22}")
        for xv, hv, dhv in dh:
            print(f"  {xv:>7.4f}   {hv:>16.6f}   {dhv:>+22.2f}")
        print()
    elif counts[DONE] > 0:
        print("  · ΔH skipped — x=0 or x=1 endpoint not yet completed.")


# ─────────────────────────────────────────────────────────────────────────────
# VEC stability table + guard
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# VEC reference table  (informational only — no pre-flight blocking)
# ─────────────────────────────────────────────────────────────────────────────


def print_vec_table(
    elem_a: str,
    elem_b: str,
    concentrations: list[float],
    nonmetal: str | None,
) -> None:
    """Print the VEC reference table for the planned sweep.

    VEC is shown as a reference value only — no traffic-light colours, no
    pre-flight blocking.  The threshold at which SCF diverges depends on the
    anion (C: ~8.4, N: ~9.0) and on the pseudopotential set, so the code no
    longer makes pass/fail decisions before the calculation runs.  If a step
    actually stalls, the watchdog kills it and the smart-retry logic applies.

    Args:
        elem_a:         Element at x = 0.
        elem_b:         Element at x = 1.
        concentrations: Planned x values.
        nonmetal:       Anion element (e.g. ``"C"``), or ``None``.
    """
    print("\n  ── VEC Reference " + "─" * 49)
    print(f"  {'x':>6}   {'VEC':>6}   (reference — watchdog handles divergences)")
    print(f"  {'─' * 6}   {'─' * 6}   {'─' * 45}")
    for x in concentrations:
        sp_eval = [(elem_a, 1.0 - x), (elem_b, x)]
        vec = phys.vec_for_system(sp_eval, nonmetal)
        print(f"  {x:>6.4f}   {vec:>6.2f}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Smart retry UI  (Task 2)
# ─────────────────────────────────────────────────────────────────────────────

#: Human-readable descriptions for each kill/failure token.
_ERROR_DESC: dict[str, tuple[str, str]] = {
    "scf_nosconv": (
        "SCF convergence failure (charge sloshing / hard divergence)",
        "smearing_width → 0.20 eV,  mix_charge_amp → 0.05",
    ),
    "smax_stall": (
        "Geometry optimisation stalled (Smax oscillating above threshold)",
        "smearing_width → 0.20 eV,  mix_charge_amp → 0.05",
    ),
    "geom_high_stress": (
        "High initial stress (possible Vegard-law mismatch in cell geometry)",
        "geom_stress_tol → 0.10 GPa  (wider acceptance window for LBFGS)",
    ),
    "timeout": (
        "Step timed out (exceeded wall-clock limit)",
        "smearing_width → 0.20 eV,  mix_charge_amp → 0.05",
    ),
    "unknown": (
        "Unknown failure — no recognisable pattern in .castep output",
        "no automatic patch available",
    ),
}


def print_smart_retry(
    failed_steps: list["Step"],
    state: "RunState",
) -> list["Step"]:
    """Display a per-step failure analysis and optionally patch + return steps to retry.

    This replaces the old blunt ``"Retry failed steps? [Y/n]"`` question.
    For each failed step the failure mode is classified, a human-readable
    explanation is printed, and the targeted .param patch is described.
    The user can then approve patching + retry for the fixable steps.

    Args:
        failed_steps: Steps with status ``FAILED``.
        state:        Parent :class:`~orchestrator.RunState`.

    Returns:
        List of steps that should be retried (empty if the user declines).
    """
    from orchestrator import FAILED, patch_for_recovery

    _section("Sweep completed — failure analysis")

    n_ok = sum(1 for s in state.steps if s.status != FAILED)
    n_fail = len(failed_steps)
    print(f"  ✓  {n_ok} step(s) succeeded")
    print(f"  ✗  {n_fail} step(s) failed or timed out\n")

    patchable: list["Step"] = []
    for s in failed_steps:
        reason = s.parsed.get("kill_reason", "unknown")
        desc, fix = _ERROR_DESC.get(reason, _ERROR_DESC["unknown"])
        x_str = f"x={s.concentration:.4f}"
        print(f"  ✗  {x_str}  —  {desc}")
        if reason != "unknown":
            print(f"        Fix: {fix}")
            patchable.append(s)
        else:
            print("        No automatic fix available.  Inspect the .castep file.")
        print()

    if not patchable:
        print("  No automatically fixable failures found.")
        return []

    if not ask_yes_no(
        f"  Auto-patch .param and retry {len(patchable)} fixable step(s)?",
        default=True,
    ):
        return []

    step_dir_base = state.proj_dir
    for s in patchable:
        step_dir = step_dir_base / s.step_dir
        reason = s.parsed.get("kill_reason", "unknown")
        ok = patch_for_recovery(step_dir, state.seed, reason)
        if ok:
            print(f"  ✓  Patched  {s.step_dir}")
        else:
            print(f"  ⚠  Could not patch {s.step_dir} (.param missing?)")
        s.status = "pending"
        s.rc = ""

    return patchable
