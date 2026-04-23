#!/usr/bin/env python3
"""
main.py  —  VCAForge entry point.
══════════════════════════════════
Engine discovery is automatic: engines/ subdirectories are scanned via
discover_engines() — no hardcoded engine names here.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import config
import ui
from engines.engine import ENGINES, discover_engines, is_engine_available
from orchestrator import (
    DONE,
    FAILED,
    SKIPPED,
    ExecResult,
    RunState,
    Step,
    execute_step,
    load_run,
    new_run,
    run_elastic_for_step,
    save_run,
    write_csv,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vcaforge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=f"VCAForge {config.VERSION}",
    )
    p.add_argument("file", type=Path, metavar="STRUCTURE")
    p.add_argument("--species", nargs="+", metavar="ELEM")
    p.add_argument("--range", nargs=3, metavar=("X0", "X1", "N"))
    p.add_argument("--single", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--resume-dir", type=Path, metavar="DIR")
    p.add_argument("--interactive", action="store_true")
    p.add_argument("--keep-all", action="store_true")
    p.add_argument("--engine", choices=list(ENGINES), default=None)
    # Legacy per-engine cmd flags — mapped to --engine-cmd internally
    p.add_argument("--engine-cmd", dest="engine_cmd_override", metavar="CMD",
                   help="Override the engine binary command")
    return p


def _validate_args(p: argparse.ArgumentParser) -> argparse.Namespace:
    args = p.parse_args()
    if not args.file.exists():
        p.error(f"File not found: {args.file!r}")
    if args.range:
        try:
            x0, x1, n = float(args.range[0]), float(args.range[1]), int(args.range[2])
            args.range = (x0, x1, n)
        except ValueError:
            p.error("--range needs two floats and one int: X0 X1 N")
    return args


def _detect_engine(args: argparse.Namespace) -> str:
    """Return engine name — prefer explicit flags, then auto-detect via config."""
    if not ENGINES:
        sys.exit("  Fatal: No engines registered. Check engines/ directory and @register_engine decorators.")

    if args.engine:
        return args.engine

    found = [name for name in ENGINES if is_engine_available(name)]

    ui.section("Engines")
    for name in ENGINES:
        status = "(detected)" if name in found else "(not in PATH)"
        print(f"  {name.upper():<10} {status}")

    if not found:
        print("  (no engines found in PATH — prepare-only mode)")
        # Safe because we already checked `if not ENGINES` above
        return list(ENGINES.keys())[0]

    if len(found) == 1:
        return found[0]

    opts = "/".join(found)
    ans = input(f"  Select engine [{found[0]}] ({opts}): ").strip().lower()
    return ans if ans in found else found[0]


def _try_resume(work_dir: Path, stem: str, args: argparse.Namespace) -> RunState | None:
    explicit = getattr(args, "resume_dir", None)
    if explicit:
        state = load_run(explicit.resolve())
    else:
        candidates = sorted(
            [d for d in work_dir.iterdir() if d.is_dir() and d.name.startswith(f"{stem}_")],
            key=lambda d: d.stat().st_mtime, reverse=True,
        )
        state = next((s for d in candidates if (s := load_run(d)) is not None), None)

    if state is None:
        return None

    for step in state.steps:
        if step.status == "running":
            step.status, step.rc = "pending", "interrupted"

    print(f"\n  Found: {state.proj_dir.name}  [{state.system_label()}]  {state.n_done} done / {len(state.steps)} total")
    if ui.ask_yes_no("Resume?", default=True):
        save_run(state)
        return state
    return None


def _log(log_path: Path, msg: str) -> None:
    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[{ts}]  {msg}\n")


def _run_sweep(state: RunState, crystal: Any, engine: Any, *, interactive: bool, keep_all: bool) -> None:
    n_total = len(state.steps)
    for step in state.steps:
        already_done = step.status in (DONE, SKIPPED)
        ui.print_step_header(step, n_total, state)

        if already_done:
            ui.print_step_result(
                ExecResult(rc=0, skipped=False, stderr_tail=[]),
                step, state.proj_dir, state.seed, state.run_elastic,
            )
            continue

        result = execute_step(state, step, crystal, engine, keep_all)
        ui.print_step_result(result, step, state.proj_dir, state.seed, state.run_elastic)

        if result.skipped:
            return

        if state.run_elastic and step.status == DONE:
            t0 = time.monotonic()
            try:
                el_res = run_elastic_for_step(state, step, engine)
                step.parsed.update(el_res)
                if "_elastic_error" in el_res:
                    ui.print_elastic_error(el_res["_elastic_error"])
                else:
                    ui.print_elastic_result(el_res, time.monotonic() - t0)
            except Exception as e:
                print(f"  │  ⚠  Elastic Engine Crashed: {e}")
                ui.print_elastic_error("Engine crash")

            save_run(state)
            write_csv(state)

        if interactive and step.status == FAILED:
            if not ui.ask_yes_no(ui._T("continue_q"), default=False):
                return

    # Smart retry — only for engines that support .param patching (CASTEP)
    if hasattr(engine, "parse_stress_tensor"):
        failed = [s for s in state.steps if s.status == FAILED]
        if failed:
            for s in ui.print_smart_retry(failed, state):
                ui.print_step_header(s, n_total, state)
                res = execute_step(state, s, crystal, engine, keep_all)
                ui.print_step_result(res, s, state.proj_dir, state.seed, state.run_elastic)


def _vegard_pass(state: RunState) -> None:
    from orchestrator import vegard_interpolate
    done = [s for s in state.steps if s.status == DONE]
    if len(done) < 2:
        return
    d0, d1 = done[0].parsed, done[-1].parsed
    changed = False
    for step in state.steps:
        if step.status == DONE and not step.parsed.get("C11"):
            if r := vegard_interpolate(step.concentration, d0, d1):
                step.parsed.update(r)
                changed = True
    if changed:
        save_run(state)
        write_csv(state)


def main() -> None:
    # 1. Discover all engines from engines/ subdirectories
    discover_engines()

    p = _build_parser()
    args = _validate_args(p)

    src = args.file.resolve()
    print(f"  Input : {src}")

    # 2. LOAD AND STANDARDIZE CRYSTAL
    try:
        from core_physics import load_crystal
        crystal = load_crystal(src)
        print(f"  Structure: {crystal.num_atoms} atoms (primitive cell) loaded via spglib.")
    except Exception as e:
        sys.exit(f"  ERROR loading structure: {e}")

    engine_name = _detect_engine(args)
    state = _try_resume(src.parent, src.stem, args) if args.resume else None
    engine: Any = None

    if state is None:
        if engine_name not in ENGINES:
            sys.exit(f"  ERROR: Engine '{engine_name}' not registered. Available: {list(ENGINES)}")

        engine_cls = ENGINES[engine_name]
        cli_species = [] if args.single else ([e.capitalize() for e in args.species] if args.species else None)

        override = getattr(args, "engine_cmd_override", None)
        if not hasattr(engine_cls, "setup_interactive"):
            sys.exit(f"  ERROR: Engine '{engine_name}' has no setup_interactive classmethod.")

        # Initialize Engine
        engine, engine_cmd = engine_cls.setup_interactive(src, crystal, override)

        wr = ui.wizard_mode(crystal=crystal, cli_elements=cli_species, cli_range=args.range)
        species_list = (
            [(next(iter(wr.target_mix)), 1.0)]
            if wr.single_mode
            else sorted(wr.target_mix.items(), key=lambda kv: kv[1])
        )

        # Dynamically collect engine constructor kwargs for crash-safe resume
        # vars(engine) grabs initialized parameters like param_src, cell_src, etc.
        engine_kwargs = {k: v for k, v in vars(engine).items() if not k.startswith('_')}
        serializable_kwargs = {}
        for k, v in engine_kwargs.items():
            if isinstance(v, Path):
                serializable_kwargs[k] = str(v)
            else:
                serializable_kwargs[k] = v

        state = new_run(
            seed=src.stem,
            proj_dir=src.parent / f"{src.stem}_{datetime.now().strftime('%b%-d_%H-%M')}",
            template_element=wr.template_element,
            species=species_list,
            engine_cmd=engine_cmd,
            engine_kwargs=serializable_kwargs,
            c_start=wr.c_start,
            c_end=wr.c_end,
            n_steps=wr.n_steps,
            single_mode=wr.single_mode,
            nonmetal=wr.nonmetal,
            nonmetal_occ=1.0,
            run_elastic=wr.run_elastic,
        )
    else:
        # Resume: reconstruct engine from saved engine_kwargs
        if engine_name not in ENGINES:
            sys.exit(f"  ERROR: Engine '{engine_name}' not available for resume.")
        engine_cls = ENGINES[engine_name]
        kwargs = dict(state.engine_kwargs)

        # Ensure Path objects are reconstructed safely without stripping booleans/ints
        for k, v in kwargs.items():
            if isinstance(v, str) and (v.startswith("/") or v.startswith("~")):
                kwargs[k] = Path(os.path.expanduser(v))
        try:
            engine = engine_cls(**kwargs)
        except Exception as exc:
            print(f"  ⚠  Could not restore engine from saved kwargs ({exc}), re-running setup...")
            override = getattr(args, "engine_cmd_override", None)
            engine, _ = engine_cls.setup_interactive(src, crystal, override)

    if engine is None:
        sys.exit("  Fatal: Engine not initialized.")

    write_csv(state)
    print(f"\n  Project : {state.proj_dir.name}")
    if state.run_elastic:
        print("  Elastic : integrated")
    print()

    log = state.proj_dir / config.LOG_FILE
    _log(log, f"start  engine={engine.name}  proj={state.proj_dir.name}")

    _run_sweep(state, crystal, engine, interactive=args.interactive, keep_all=args.keep_all)

    if state.run_elastic:
        _vegard_pass(state)
    _log(log, "complete")
    ui.print_summary(state)
    print(f"\n  Results → {state.proj_dir / config.CSV_FILE}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n  Interrupted. State saved.\n")
        sys.exit(0)
