#!/usr/bin/env python3
"""
main.py  —  VCAForge  v{VERSION}
════════════════════════════════
Automated VCA concentration sweeps with integrated elastic constants.

Usage
─────
  python main.py TiC.cif                            # interactive wizard
  python main.py TiC.cell --species Nb              # single NbC
  python main.py TiC.cell --species Nb V --range 0 1 8
  python main.py TiC.cell --species Nb V W          # ternary VCA (asks fracs)
  python main.py TiC.cell --single                  # pure template compound
  python main.py TiC.cell --resume
  python main.py TiC.cell --castep-cmd 'castep.serial {seed}'

Adaptive species wizard (no --species flag)
───────────────────────────────────────────
  Enter alone            →  single_mode  (pure template, e.g. TiC)
  One element  'Nb'      →  single_mode  (NbC using TiC geometry)
  Two elements 'Nb V'    →  VCA sweep   Nb(1-x)V(x)C
  N elements   'Nb V W'  →  VCA sweep, asks end-point fractions
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import config
import ui
from orchestrator import (
    DONE,
    FAILED,
    SKIPPED,
    ExecResult,
    RunState,
    Step,
    cmd_is_valid,
    execute_step,
    load_run,
    new_run,
    run_process,
    save_run,
    write_csv,
)

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vcaforge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=f"VCAForge {config.VERSION} — automated VCA concentration sweeps",
        epilog=(
            "Examples:\n"
            "  python main.py TiC.cif\n"
            "  python main.py TiC.cell --species Nb V --range 0 1 8\n"
            "  python main.py TiC.cell --single\n"
            "  python main.py TiC.cell --resume\n"
        ),
    )
    p.add_argument("file", type=Path, metavar="STRUCTURE")
    p.add_argument(
        "--species",
        nargs="+",
        metavar="ELEM",
        help=(
            "One element → single compound on that sublattice; two or more → VCA sweep"
        ),
    )
    p.add_argument(
        "--range",
        nargs=3,
        metavar=("X0", "X1", "N"),
        help="Sweep: start end n_intervals",
    )
    p.add_argument("--single", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--resume-dir", type=Path, metavar="DIR")
    p.add_argument("--interactive", action="store_true")
    p.add_argument("--keep-all", action="store_true")
    p.add_argument("--castep-cmd", dest="castep_cmd", metavar="CMD")
    return p


def _validate_args(p: argparse.ArgumentParser) -> argparse.Namespace:
    args = p.parse_args()
    if not args.file.exists():
        p.error(f"File not found: '{args.file}'")
    if args.file.suffix.lower() not in {".cif", ".cell"}:
        p.error(f"Expected .cell or .cif, got '{args.file.suffix}'")
    if args.range is not None:
        try:
            x0, x1, n = float(args.range[0]), float(args.range[1]), int(args.range[2])
        except ValueError:
            p.error("--range needs two floats and one int: X0 X1 N")
        if not (0 <= x0 <= 1 and 0 <= x1 <= 1):
            p.error("X0 and X1 must be in [0, 1]")
        if n < 1:
            p.error("N must be >= 1")
        args.range = (x0, x1, n)
    return args


# ─────────────────────────────────────────────────────────────────────────────
# CIF → .cell resolution
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_cell(path: Path) -> Path:
    """Return a .cell file path, converting from .cif via cif2cell if needed."""
    if path.suffix.lower() == ".cell":
        return path

    if config.CELL_AUTO_REDUCE:
        try:
            from castep.castep import reduce_cif

            r = reduce_cif(path)
            ratio = r.n_original / max(r.n_primitive, 1)
            if ratio > 1.001:
                print(
                    f"  Cell reduction  : {r.n_original} atoms"
                    f" -> {r.n_primitive} atoms"
                    f"  (x{ratio:.0f} reduction, primitive cell)"
                )
            else:
                print(f"  Cell            : {r.n_original} atoms (already primitive)")
        except Exception as exc:
            print(f"  Cell reduction skipped: {exc}")

    cell_path = path.with_suffix(".cell")
    if cell_path.exists():
        print(f"  .cell  : {cell_path.name} (found)")
        return cell_path

    if not shutil.which("cif2cell"):
        sys.exit("  ERROR: cif2cell not in PATH — supply a .cell file directly.")
    try:
        subprocess.run(
            ["cif2cell", "-f", str(path), "-p", "castep", "-o", str(cell_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"  .cell  : {cell_path.name} (generated by cif2cell)")
    except subprocess.CalledProcessError as exc:
        sys.exit(f"  ERROR: cif2cell failed:\n{exc.stderr.strip() or 'no output'}")
    return cell_path


# ─────────────────────────────────────────────────────────────────────────────
# Resume
# ─────────────────────────────────────────────────────────────────────────────


def _try_resume(
    work_dir: Path,
    stem: str,
    explicit: Path | None,
) -> RunState | None:
    if explicit:
        state = load_run(explicit.resolve())
    else:
        candidates = sorted(
            [
                d
                for d in work_dir.iterdir()
                if d.is_dir() and d.name.startswith(f"{stem}_")
            ],
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )
        state = next((s for d in candidates if (s := load_run(d)) is not None), None)

    if state is None:
        return None

    for step in state.steps:
        if step.status == "running":
            step.status = "pending"
            step.rc = "interrupted"
            print(f"  Step {step.idx:02d} reset (was RUNNING — crash recovery)")

    print(
        f"\n  Found: {state.proj_dir.name}  [{state.system_label()}]"
        f"  {state.n_done} done / {len(state.steps)} total"
    )
    if ui.ask_yes_no("Resume?", default=True):
        save_run(state)
        return state
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────


def _log(log_path: Path, msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[{ts}]  {msg}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Cell-write callbacks
# ─────────────────────────────────────────────────────────────────────────────


def _make_vca_write_fn(
    cell_src: Path,
    template_element: str,
    species: list[tuple[str, float]],
    nonmetal_occ: float,
):
    """Return a ``(dest, x) -> None`` callback that writes a VCA .cell file.

    ``species[0]``  — element being replaced at x = 0  (weight = 1 - x).
    ``species[1:]`` — new elements at x = 1 endpoint.
                      Each has an *end_frac* that gives its fraction of the
                      mixed sublattice at x = 1 (end_fracs must sum to 1).
                      At concentration x the weight is:  end_frac * x.
    """
    from castep.castep import write_vca_cell

    def _fn(dest: Path, x: float) -> None:
        # Build the per-atom mix dict for this concentration.
        target_mix: dict[str, float] = {species[0][0]: 1.0 - x}
        for elem, end_frac in species[1:]:
            target_mix[elem] = end_frac * x

        # Normalise to guard against floating-point drift.
        total = sum(target_mix.values())
        if abs(total - 1.0) > 1e-9 and total > 1e-9:
            target_mix = {e: v / total for e, v in target_mix.items()}

        write_vca_cell(dest, cell_src, template_element, target_mix, occ=nonmetal_occ)

    return _fn


def _make_copy_fn(cell_src: Path):
    def _fn(dest: Path, _x: float) -> None:
        shutil.copy2(cell_src, dest)

    return _fn


# ─────────────────────────────────────────────────────────────────────────────
# Run loops
# ─────────────────────────────────────────────────────────────────────────────


def _run_loop(
    state: RunState,
    cell_src: Path,
    param_src: Path,
    write_fn: object,
    *,
    interactive: bool,
    keep_all: bool,
) -> None:
    steps = state.steps
    total = len(steps)
    log = state.proj_dir / config.LOG_FILE

    if state.n_pending == 0:
        print("\n  All steps already completed.")
        ui.print_summary(state)
        return

    print(f"\n-- {state.n_pending} pending / {total} total")
    if interactive:
        print("  s = skip  q = quit  Ctrl+C = skip current step\n")
    else:
        print("  Ctrl+C = skip current step  |  --interactive to pause\n")

    for step in steps:
        if step.status in {DONE, SKIPPED}:
            icon = "+" if step.status == DONE else "o"
            print(
                f"  {icon} x={step.concentration:.4f}"
                f"  H={step.enthalpy_eV or '—'}"
                f"  a={step.a_opt_ang or '—'} Å"
                f"  {step.wall_time_s or '—'}s"
            )
            continue

        ui.print_step_header(step, total, state)
        _log(log, f"Step {step.idx:02d}  x={step.concentration:.4f}  START")

        if interactive:
            ans = input("  [Enter=run  s=skip  q=quit]: ").strip().lower()
            if ans == "q":
                print("  Quit requested.")
                save_run(state)
                write_csv(state)
                return
            if ans == "s":
                step.status = SKIPPED
                step.rc = "user"
                save_run(state)
                write_csv(state)
                print("  └─ ⊘ Skipped")
                continue

        result = execute_step(
            state,
            step,
            cell_src,
            param_src,
            write_fn,
            keep_all=keep_all,
        )
        ui.print_step_result(result, step, state.proj_dir, state.seed)
        _log(
            log,
            f"Step {step.idx:02d}  x={step.concentration:.4f}"
            f"  {step.status.upper()}  rc={step.rc}",
        )

        if not result.skipped and step.status == DONE and state.run_elastic:
            _run_elastic_substep(state, step)

    ui.print_summary(state)

    # Smart retry: analyse each failed step, offer targeted .param patches.
    failed = [s for s in steps if s.status == FAILED]
    if failed and state.castep_cmd:
        retry = ui.print_smart_retry(failed, state)
        if retry:
            save_run(state)
            write_csv(state)
            _run_loop(
                state,
                cell_src,
                param_src,
                write_fn,
                interactive=interactive,
                keep_all=keep_all,
            )


def _run_single(
    state: RunState,
    cell_src: Path,
    param_src: Path,
    *,
    keep_all: bool,
) -> None:
    step = state.steps[0]
    seed = state.seed
    log = state.proj_dir / config.LOG_FILE

    if step.status in {DONE, SKIPPED}:
        print(f"\n  Already {step.status}.")
        ui.print_single_result(step, seed)
        return

    print(f"\n  Running single-compound: {state.system_label()}")
    _log(log, f"Single-compound  {state.system_label()}  START")

    result = execute_step(
        state,
        step,
        cell_src,
        param_src,
        _make_copy_fn(cell_src),
        keep_all=keep_all,
    )
    ui.print_single_result(step, seed)
    _log(log, f"Single-compound  {state.system_label()}  {step.status.upper()}")

    if not result.skipped and step.status == DONE and state.run_elastic:
        _run_elastic_substep(state, step)


# ─────────────────────────────────────────────────────────────────────────────
# Elastic sub-step
# ─────────────────────────────────────────────────────────────────────────────


def _run_elastic_substep(state: RunState, step: Step) -> None:
    """Run the finite-strain elastic workflow for a completed GeomOpt step.

    Delegates fully to ``castep.elastic.run_elastic``, which:
      - reads the lattice from ``{seed}-out.cell`` internally,
      - computes nextra_bands from x and species via VEC,
      - runs all strained single-points,
      - fits and returns C_ij.

    We do NOT pass a lattice matrix here — run_elastic reads it itself.
    We DO pass x and species so it can compute the correct nextra_bands.
    """
    import core_physics as phys
    from castep.castep import run_elastic

    x = step.concentration

    # Build the species fractions at this specific concentration for VEC.
    sp_at_x: list[tuple[str, float]] = [(state.species[0][0], 1.0 - x)] + [
        (e, f * x) for e, f in state.species[1:]
    ]
    vec = phys.vec_for_system(sp_at_x, state.nonmetal or None)

    step_dir = state.proj_dir / step.step_dir

    # Pull density and volume from the GeomOpt result for Debye/acoustic
    # Grüneisen calculations.
    density = _float_or_none(step.parsed.get("density_gcm3"))
    volume = _float_or_none(step.parsed.get("volume_ang3"))

    ui.print_elastic_start(
        n_strains=config.ELASTIC_N_STEPS * 2,
        vec=vec,
        nextra=phys.nextra_bands_for(x, vec),
    )

    t0 = time.monotonic()
    # run_elastic signature:
    #   run_elastic(seed_dir, seed, castep_cmd, *, x, species, nonmetal,
    #               strain, n_steps, keep_all, progress_cb,
    #               density_gcm3, volume_ang3)
    # — all physics params are keyword-only; it reads the lattice itself.
    data = run_elastic(
        step_dir,
        state.seed,
        state.castep_cmd,
        x=x,
        species=sp_at_x,
        nonmetal=state.nonmetal or None,
        progress_cb=ui.elastic_progress_cb,
        density_gcm3=density,
        volume_ang3=volume,
    )
    ui.elastic_progress_clear()
    elapsed = time.monotonic() - t0

    step.parsed["total_wall_time_s"] = (
        f"{float(step.parsed.get('wall_time_s') or 0) + elapsed:.0f}"
    )

    if data and "C11" in data:
        step.parsed.update(data)
        step.parsed.setdefault("VEC", f"{vec:.4f}")
        ui.print_elastic_result(data, elapsed)
    elif data and "_elastic_error" in data:
        step.parsed.update(data)
        ui.print_elastic_error(data["_elastic_error"])
    else:
        ui.print_elastic_error("no usable data returned")

    save_run(state)
    write_csv(state)


def _float_or_none(val: object) -> float | None:
    """Safely convert a parsed string value to float, returning None on failure."""
    try:
        return float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Vegard interpolation pass
# ─────────────────────────────────────────────────────────────────────────────


def _run_vegard_pass(state: RunState) -> None:
    """Fill elastic constants for failed intermediate steps via Vegard law."""
    from core_physics import vegard_interpolate

    done = [s for s in state.steps if s.status == DONE]
    e0 = next(
        (s.parsed for s in done if abs(s.concentration) < 1e-5 and "C11" in s.parsed),
        {},
    )
    e1 = next(
        (
            s.parsed
            for s in done
            if abs(s.concentration - 1) < 1e-5 and "C11" in s.parsed
        ),
        {},
    )
    if not e0 or not e1:
        return

    changed = False
    for s in done:
        if "C11" in s.parsed:
            continue
        x = s.concentration
        if not (1e-5 < x < 1 - 1e-5):
            continue
        interp = vegard_interpolate(x, e0, e1)
        if interp:
            s.parsed.update(interp)
            b = interp.get("B_Hill_GPa", "—")
            print(f"  Vegard x={x:.4f}  B={b} GPa")
            changed = True

    if changed:
        save_run(state)
        write_csv(state)


# ─────────────────────────────────────────────────────────────────────────────
# main()
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = _build_parser()
    args = _validate_args(parser)

    print(f"\n  VCAForge {config.VERSION}\n")

    src = args.file.resolve()
    work_dir = src.parent
    print(f"  Input : {src}")

    cell_path = _resolve_cell(src)

    # ── Resume path ──────────────────────────────────────────────────────────
    state: RunState | None = None
    if args.resume or args.resume_dir is not None:
        state = _try_resume(work_dir, src.stem, args.resume_dir)

    # ── New run wizard ────────────────────────────────────────────────────────
    if state is None:
        from castep.castep import read_species as _read_sp

        species_in_cell = _read_sp(cell_path)

        # Normalise CLI species; --single forces Scenario A (Enter path).
        if args.single:
            cli_species: list[str] | None = []  # → wizard_mode Scenario A
        elif args.species:
            cli_species = [e.capitalize() for e in args.species]
        else:
            cli_species = None  # fully interactive

        # wizard_param is idempotent when .param already exists.
        ui.wizard_param(cell_path, species_in_cell, is_vca=True)
        castep_cmd = ui.wizard_castep_cmd(args.castep_cmd)

        # ── Composite species / mode wizard ───────────────────────────────────
        result = ui.wizard_mode(
            cell_path=cell_path,
            cli_elements=cli_species,
            cli_range=args.range,
        )

        # ── Convert WizardResult → species list ───────────────────────────────
        #
        # The wizard returns:
        #   result.template_element  — e.g. "Ti"  (the sublattice being replaced)
        #   result.target_mix        — {elem: fraction_at_x1}
        #                              Binary:  {"Nb": 0.0, "Zr": 1.0}
        #                              Ternary: {"Nb": 0.6, "V": 0.4}
        #
        # We build species_list as:
        #   species[0] = (template_element, 0.0)   ← dominant at x = 0
        #   species[1:] = [(new_elem, end_frac)]    ← fractions at x = 1
        #
        # For a binary Nb V sweep on TiC:
        #   template_element = "Ti"
        #   target_mix       = {"Nb": 0.0, "V": 1.0}
        #   → species_list   = [("Ti", 0.0), ("Nb", 0.0), ("V", 1.0)]
        #
        # Wait — that would give Ti weight at x=0 and Nb weight=0 everywhere.
        # The correct interpretation is:
        #   - elems[0] = "Nb" is the element that starts (present at x=0)
        #   - elems[1] = "V"  is the element that ends   (present at x=1)
        #   - template = "Ti" is what gets replaced in the .cell file
        #
        # So species_list should use the *new* elements, not the template:
        #   species[0] = ("Nb", 0.0)   ← new element at x = 0
        #   species[1] = ("V",  1.0)   ← new element at x = 1
        #
        # And template_element stays "Ti" for write_vca_cell.
        #
        if result.single_mode:
            dominant = next(iter(result.target_mix))
            species_list: list[tuple[str, float]] = [(dominant, 1.0)]
        else:
            # target_mix at x=1 endpoint: {elem: frac}.  elem with frac≈0 is
            # the "start" element (dominant at x=0).
            # Sort so the x=0 element (frac≈0) is first.
            items = sorted(result.target_mix.items(), key=lambda kv: kv[1])
            species_list = list(items)

        print(f"\n  System : {_format_system(species_list, result.nonmetal)}")

        timestamp = datetime.now().strftime("%b%-d_%H-%M")
        proj_dir = work_dir / f"{src.stem}_{timestamp}"

        state = new_run(
            seed=src.stem,
            proj_dir=proj_dir,
            template_element=result.template_element,
            species=species_list,
            castep_cmd=castep_cmd,
            c_start=result.c_start,
            c_end=result.c_end,
            n_steps=result.n_steps,
            single_mode=result.single_mode,
            nonmetal=result.nonmetal,
            nonmetal_occ=1.0,
            run_elastic=result.run_elastic,
        )

    else:
        # Resumed run — re-validate binary.
        if state.castep_cmd:
            binary = state.castep_cmd.replace("{seed}", "").strip().split()[-1]
            if not cmd_is_valid(binary):
                print("  Stored CASTEP command no longer valid.")
                state.castep_cmd = ui.wizard_castep_cmd(None)
                save_run(state)

    param_src = cell_path.with_suffix(".param")
    write_csv(state)

    print(f"\n  Project : {state.proj_dir.name}")
    print(f"  CSV     : {config.CSV_FILE}")
    if not state.castep_cmd:
        print("  Mode    : prepare-only (no CASTEP)")
    if state.run_elastic:
        print("  Elastic : integrated (runs after each GeomOpt)")
    print()

    log = state.proj_dir / config.LOG_FILE
    _log(log, f"Session start  proj={state.proj_dir.name}")

    # Build the per-step cell-write callback.
    write_fn = (
        _make_copy_fn(cell_path)
        if state.single_mode
        else _make_vca_write_fn(
            cell_path,
            state.template_element,
            state.species,
            state.nonmetal_occ,
        )
    )

    if state.single_mode:
        _run_single(state, cell_path, param_src, keep_all=args.keep_all)
    else:
        _run_loop(
            state,
            cell_path,
            param_src,
            write_fn,
            interactive=args.interactive,
            keep_all=args.keep_all,
        )

    if state.run_elastic:
        _run_vegard_pass(state)

    _log(log, "Session complete.")
    print(f"\n  Results -> {state.proj_dir / config.CSV_FILE}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────


def _format_system(species: list[tuple[str, float]], nonmetal: str) -> str:
    """Return a human-readable formula for the console System: line.

    Examples::

        [("Ti", 0.0), ("Zr", 1.0)], "C"              → "Ti(1-x)Zr(x)C"
        [("Ti", 0.0), ("Nb", 0.7), ("V", 0.3)], "C"  → "Ti(1-x)[Nb0.70V0.30](x)C"
        [("NbC", 1.0)], ""                            → "NbC"
    """
    nm = nonmetal
    if len(species) == 1:
        return f"{species[0][0]}{nm}" if nm else species[0][0]
    if len(species) == 2:
        metal = f"{species[0][0]}(1-x){species[1][0]}(x)"
    else:
        inner = "".join(f"{e}{f:.2f}" for e, f in species[1:])
        metal = f"{species[0][0]}(1-x)[{inner}](x)"
    return f"{metal}{nm}" if nm else metal


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n  Interrupted. State saved.\n")
        sys.exit(0)
