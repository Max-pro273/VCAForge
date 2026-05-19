#!/usr/bin/env python3
"""
main.py  —  VCAForge entry point.
══════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import config
import ui
from core_physics import Crystal, load_crystal
from crystal_modes import (
    MODES,
    SQSStrategy,
    exact_supercell_extent,
    register_strategy,
)
from engines.engine import (
    ENGINES,
    WizardBypassCapable,
    discover_engines,
    is_engine_available,
)
from orchestrator import ElasticTask, GeomOptTask, TaskRunner
from runstate import (
    DONE,
    PENDING,
    RUNNING,
    SKIPPED,
    RunState,
    StateIO,
)

if TYPE_CHECKING:
    from engines.engine import BaseEngine
    from orchestrator import Task


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vcaforge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=f"VCAForge {config.VERSION}",
        epilog=(
            "Subcommands\n"
            "  run        Run a VCA/SQS/direct sweep (default when STRUCTURE given).\n"
            "  navigate   Bayesian optimisation over composition space.\n"
            "  aggregate  Collect CSVs from multiple runs.\n"
            "  visualize  Render GP landscape from navigator output.\n"
            "  models     List available / installed MLIP models.\n\n"
            "Examples\n"
            "  vcaforge run TiC.cell --engine castep --mode sqs --species Ti Nb\n"
            "  vcaforge run TiC.cell --engine mlip --model mace-mp-0b --device cuda\n"
            "  vcaforge navigate --dir ./TiNbC_Jan12 --target H_Vickers_Chen_GPa\n"
            "  vcaforge models\n"
        ),
    )
    p.add_argument(
        "--version", action="version", version=f"VCAForge {config.VERSION}",
    )

    sub = p.add_subparsers(dest="subcommand", metavar="SUBCOMMAND")

    # ── run (default) ─────────────────────────────────────────────────────────
    run_p = sub.add_parser(
        "run",
        help="Run a VCA/SQS/direct composition sweep.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_run_args(run_p)

    # ── navigate ──────────────────────────────────────────────────────────────
    nav_p = sub.add_parser(
        "navigate",
        help="Bayesian optimisation loop over composition space.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    nav_p.add_argument("--dir", type=Path, metavar="DIR",
        help="Directory containing vca_results.csv (or parent to scan).")
    nav_p.add_argument("--target", default=None,
        help=f"CSV column to optimise (default: {config.NAVIGATOR_TARGET}).")
    nav_p.add_argument("--mode", choices=["maximize", "minimize"], default=None)
    nav_p.add_argument("--batch", type=int, default=None, metavar="N",
        help="Suggest N new compositions per iteration.")
    nav_p.add_argument("--max-iter", type=int, default=None,
        help=f"Safety cap on iterations (default: {config.NAVIGATOR_LOOP_MAX_ITER}).")
    nav_p.add_argument("--no-auto-run", action="store_true",
        help="Print suggestions only; do not launch VCAForge automatically.")

    # ── aggregate ─────────────────────────────────────────────────────────────
    agg_p = sub.add_parser(
        "aggregate",
        help="Collect vca_results.csv files from multiple runs into one master CSV.",
    )
    agg_p.add_argument("dirs", nargs="*", type=Path, metavar="DIR",
        help="Directories to aggregate (default: scan current directory).")
    agg_p.add_argument("--out", type=Path, default=None,
        help="Output CSV path (default: <cwd>/<stem>_master.csv).")
    agg_p.add_argument("--mode", choices=["sqs", "vca", "direct"], default=None,
        help="Calculation mode to aggregate (sqs|vca|direct). Baseline 'direct' points are always included.")
    agg_p.add_argument("--target", default=None,
        help="Property column used for deduplication (keeping best).")
    agg_p.add_argument("--maximize", action="store_const", const="maximize", dest="direction",
        help="Maximize target value during deduplication (default).")
    agg_p.add_argument("--minimize", action="store_const", const="minimize", dest="direction",
        help="Minimize target value during deduplication.")

    # ── visualize ─────────────────────────────────────────────────────────────
    viz_p = sub.add_parser(
        "visualize",
        help="Render the navigator GP landscape (mu, sigma, P_f) as a plot.",
    )
    viz_p.add_argument("csv", type=Path, metavar="CSV",
        help="vca_results.csv (or master CSV) to visualise.")
    viz_p.add_argument("--target", default=None,
        help="Property column to plot (default: config.NAVIGATOR_TARGET).")
    viz_p.add_argument("--out", type=Path, default=None,
        help="Save plot to file instead of showing interactively.")

    # ── models ────────────────────────────────────────────────────────────────
    models_p = sub.add_parser(
        "models",
        help="List all registered MLIP models and their install status.",
    )
    models_p.add_argument("--available", action="store_true",
        help="Show only models whose library is currently installed.")
    models_p.add_argument("--install", metavar="MODEL_KEY",
        help="Print the uv install command for a specific model.")
    models_p.add_argument("--device", default="cpu",
        choices=["cpu", "cuda", "mps", "rocm"],
        help="Device to show install command for (cpu/cuda/mps/rocm).")

    return p, run_p   # return both so _validate_args can use run_p


def _add_run_args(p: argparse.ArgumentParser) -> None:
    """Add all 'run' subcommand arguments (also used as top-level fallback)."""
    p.add_argument("file", type=Path, metavar="STRUCTURE")

    # Engine + mode
    p.add_argument(
        "--engine", choices=list(ENGINES) or None, default=None,
        help="DFT/ML engine. Auto-detected if omitted.",
    )
    p.add_argument(
        "--mode", choices=sorted(MODES), default=None,
        help="Crystal preparation mode (vca | sqs | direct). Asked interactively if omitted.",
    )

    # Composition / sweep
    p.add_argument("--template", metavar="ELEM",
        help="Sublattice element to be substituted (e.g. Ti in TiC).")
    p.add_argument("--composition", "-c", nargs="+", metavar="SPEC",
        help="Mixture at x=1, e.g. 'Nb:0.5 Ti:0.25 Zr:0.25'.")
    p.add_argument("--species", nargs="+", metavar="ELEM",
        help="Binary shortcut: '--species Nb Zr' = Nb(1-x) → Zr(x).")
    p.add_argument("--points", metavar="LIST",
        help="Explicit concentration list, e.g. '0.0,0.3,0.5,0.7,1.0'.")
    p.add_argument("--x", type=float, metavar="VAL",
        help="Single concentration point (shortcut for --points VAL).")
    p.add_argument("--range", nargs=3, metavar=("X0", "X1", "N"),
        help="Linear sweep: start, end, intervals.")
    p.add_argument("--single", action="store_true",
        help="Single-compound run (no sweep, no composition).")

    # SQS
    p.add_argument(
        "--sqs-max", type=int, default=None, dest="sqs_max",
        help=f"Max cubic supercell extent for SQS (default: {config.SQS_MAX_EXTENT}).",
    )
    p.add_argument(
        "--sqs-iterations", type=int, default=None, dest="sqs_iterations",
        help=f"sqsgenerator MC iterations (default: {config.SQS_ITERATIONS}).",
    )
    p.add_argument(
        "--sqs-threads", type=int, default=None, dest="sqs_threads",
        help=(
            "sqsgenerator parallel MC threads (default: config.SQS_THREADS = 0 = auto). "
            "Set to 0 to use all available CPU cores."
        ),
    )

    # MLIP-specific
    p.add_argument(
        "--model", default=None, metavar="MODEL",
        help=(
            "MLIP model key (e.g. mace-mp-0b, chgnet, 7net-mf-ompa). "
            "Used with --engine mlip. Run 'vcaforge models' for full list."
        ),
    )
    p.add_argument(
        "--device", default=None, choices=["cpu", "cuda", "mps", "rocm"],
        help="Compute device for MLIP. cpu/cuda/mps/rocm. Default: config.MLIP_DEVICE.",
    )
    p.add_argument(
        "--mlip-fmax", type=float, default=None, dest="mlip_fmax",
        help=f"Force convergence criterion eV/Å for ASE relaxer (default: {config.MLIP_FMAX}).",
    )
    p.add_argument(
        "--mlip-file", default=None, dest="mlip_file", metavar="PATH",
        help="Path to a local checkpoint file (for nequip-local, deepmd-local, local-ase, local-torch).",
    )
    p.add_argument(
        "--mlip-optimizer", default=None, dest="mlip_optimizer",
        choices=["BFGS", "FIRE", "LBFGS"],
        help=f"ASE optimizer for MLIP relaxation (default: {config.MLIP_OPTIMIZER}).",
    )
    p.add_argument(
        "--parallel", type=int, default=1, metavar="N",
        help="Run N concentration steps concurrently (MLIP GPU: set to number of cores).",
    )

    # Run control
    p.add_argument("--resume", action="store_true")
    p.add_argument("--resume-dir", type=Path, metavar="DIR")
    p.add_argument("--interactive", action="store_true")
    p.add_argument("--keep-all", action="store_true")
    p.add_argument("-n", "--cores", type=int, default=None, metavar="N",
        help="MPI processes (DFT engines only; skips interactive prompt).")
    p.add_argument("--elastic", action="store_true",
        help="Force-enable elastic constants.")
    p.add_argument("--no-elastic", action="store_true",
        help="Force-skip elastic constants.")
    p.add_argument("--engine-cmd", dest="engine_cmd_override", metavar="CMD",
        help="Override the engine binary command (DFT engines only).")


def _validate_args(
    p: argparse.ArgumentParser, run_p: argparse.ArgumentParser
) -> argparse.Namespace:
    import sys as _sys  # noqa: PLC0415
    argv = _sys.argv[1:]

    # Backward compat: bare `vcaforge TiC.cell ...` (no subcommand token).
    # Detect this by checking if the first token looks like a file path rather
    # than a known subcommand name. If so, prepend "run" and re-parse with p.
    _subcommands = {"run", "navigate", "aggregate", "visualize", "models"}
    _top_level_flags = {"-h", "--help", "--version"}
    first = argv[0] if argv else ""
    if first in _top_level_flags:
        # Let the top-level parser handle -h / --version directly.
        pass
    elif first not in _subcommands and not first.startswith("-"):
        # Positional file path with no subcommand — old-style invocation.
        argv = ["run"] + argv
    elif first.startswith("-") and not any(t in _subcommands for t in argv):
        # All flags, no subcommand (e.g. `vcaforge --mode sqs file.cell`).
        argv = ["run"] + argv

    args = p.parse_args(argv)

    if args.subcommand == "run":
        if not hasattr(args, "file") or args.file is None:
            p.error("STRUCTURE file is required.")
        if not args.file.exists():
            p.error(f"File not found: {args.file!r}")
        if getattr(args, "range", None):
            try:
                x0, x1, n = (
                    float(args.range[0]), float(args.range[1]), int(args.range[2]),
                )
                args.range = (x0, x1, n)
            except ValueError:
                p.error("--range needs two floats and one int: X0 X1 N")

    return args

    return args


# ─────────────────────────────────────────────────────────────────────────────
# Engine selection
# ─────────────────────────────────────────────────────────────────────────────


def _detect_engine(args: argparse.Namespace) -> str:
    if not ENGINES:
        sys.exit(
            "  Fatal: No engines registered. Check engines/ directory "
            "and @register_engine decorators."
        )
    if args.engine:
        return args.engine

    found = [name for name in ENGINES if is_engine_available(name)]
    ui.section("Engines")
    for name in ENGINES:
        print(f"  {name.upper():<10} (loaded)")
    if not found:
        print("  (no engines registered — prepare-only mode)")
        return next(iter(ENGINES))
    if len(found) == 1:
        return found[0]
    opts = "/".join(found)
    ans = input(f"  Select engine [{found[0]}] ({opts}): ").strip().lower()
    return ans if ans in found else found[0]


def _select_mode(
    args: argparse.Namespace, engine: "BaseEngine",
) -> str:
    """Pick the crystal-preparation mode AFTER the engine is known."""
    supported = sorted(engine.SUPPORTED_MODES)
    if not supported:
        sys.exit(f"  Fatal: engine '{engine.name}' supports no modes.")

    if args.mode:
        if args.mode not in supported:
            sys.exit(
                f"  ERROR: engine '{engine.name}' does not support "
                f"mode '{args.mode}'. Supported: {supported}"
            )
        return args.mode

    ui.section(f"Crystal mode  ({engine.name.upper()})")
    print(f"  Supported by this engine: {', '.join(supported)}")
    default = (
        "vca" if "vca" in supported
        else ("sqs" if "sqs" in supported else supported[0])
    )
    return ui.ask_choice(supported, default)


def _announce_mode_warnings(engine: "BaseEngine", mode: str) -> None:
    warnings = getattr(engine, "MODE_WARNINGS", {})
    if mode in warnings:
        ui.section(f"Advisory — {engine.name.upper()} + {mode.upper()}")
        for line in warnings[mode].splitlines():
            print(f"  ⚠  {line}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# SQS planning + per-run strategy registration
# ─────────────────────────────────────────────────────────────────────────────


def _setup_sqs_strategy(
    args: argparse.Namespace,
    crystal: Crystal,
    template_element: str,
    target_mix_at_x1: dict[str, float],
) -> None:
    """Build a per-run SQSStrategy honoring --sqs-max / --sqs-iterations /
    --sqs-threads, print the planning report, and register it as MODES['sqs'].
    """
    max_extent = args.sqs_max or config.SQS_MAX_EXTENT
    iterations = args.sqs_iterations or config.SQS_ITERATIONS
    # --sqs-threads 0 means auto (all cores); None means use config default.
    n_threads_arg = getattr(args, "sqs_threads", None)
    n_threads = n_threads_arg if n_threads_arg is not None else config.SQS_THREADS

    strategy = SQSStrategy(
        max_extent=max_extent,
        iterations=iterations,
        shell_weights=config.SQS_SHELL_WEIGHTS,
        n_threads=n_threads,
    )
    register_strategy("sqs", strategy)

    n_template_primitive = sum(
        1 for site in crystal.sites
        if next(iter(site.keys())).casefold() == template_element.casefold()
    )
    if n_template_primitive == 0:
        return
    plan = SQSStrategy.plan(
        crystal, template_element, target_mix_at_x1, max_extent,
    )
    exact_ext = None
    if not plan.is_exact:
        exact_ext = exact_supercell_extent(
            target_mix_at_x1, n_template_primitive,
        )
    ui.print_sqs_plan(
        plan, target_mix_at_x1, template_element,
        base_atom_count=crystal.num_atoms,
        exact_extent=exact_ext,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Resume
# ─────────────────────────────────────────────────────────────────────────────


def _try_resume(
    work_dir: Path, stem: str, args: argparse.Namespace,
) -> RunState | None:
    explicit = getattr(args, "resume_dir", None)
    if explicit:
        state = StateIO.load_run(explicit.resolve())
        if not state:
            sys.exit(f"  ERROR: Could not load state from {explicit}")
    else:
        candidates = sorted(
            [d for d in work_dir.iterdir()
             if d.is_dir() and d.name.startswith(f"{stem}_")],
            key=lambda d: d.stat().st_mtime, reverse=True,
        )
        state = next(
            (s for d in candidates if (s := StateIO.load_run(d)) is not None), None,
        )

    if state is None:
        if getattr(args, "resume", False):
            print("  ℹ  No resumable state found. Starting fresh.")
        return None

    if state.n_done == len(state.steps):
        return None

    for step in state.steps:
        if step.status == RUNNING:
            step.status = PENDING
            step.rc = "interrupted"

    print(f"ℹ  Found incomplete run : {state.proj_dir.name}")
    print(f"     System: {state.system_label()} | "
          f"Progress: {state.n_done}/{len(state.steps)} steps done")
    if ui.ask_yes_no("  Resume this calculation?", default=True):
        StateIO.save_run(state)
        return state
    print("  · Starting fresh run instead.")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Task Queue Builder
# ─────────────────────────────────────────────────────────────────────────────


def build_task_queue(state: RunState, crystal: Crystal, engine: "BaseEngine", args: Any) -> list["Task"]:
    queue: list["Task"] = []

    for step in state.steps:
        if step.status in (DONE, SKIPPED):
            continue
        queue.append(GeomOptTask(state, step, engine, crystal, keep_all=args.keep_all))

        if state.run_elastic:
            queue.append(ElasticTask(state, step, engine, crystal))

    return queue


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    discover_engines()

    p, run_p = _build_parser()
    args = _validate_args(p, run_p)

    # ── Subcommand dispatch ───────────────────────────────────────────────────

    if args.subcommand == "models":
        _cmd_models(args)
        return

    if args.subcommand == "navigate":
        _cmd_navigate(args)
        return

    if args.subcommand == "aggregate":
        _cmd_aggregate(args)
        return

    if args.subcommand == "visualize":
        _cmd_visualize(args)
        return

    # ── run (default) ────────────────────────────────────────────────────────

    src = args.file.resolve()
    print(f"  Input : {src}")

    try:
        crystal = load_crystal(src)
        print(
            f"  Structure: {crystal.num_atoms} atoms (primitive cell) "
            f"loaded via spglib."
        )
    except Exception as e:
        sys.exit(f"  ERROR loading structure: {e}")

    engine_name = _detect_engine(args)
    state = _try_resume(src.parent, src.stem, args)
    engine: "BaseEngine" | None = None

    if state is None:
        engine, mode, wr = _setup_new_run(args, src, crystal, engine_name)
        state = _create_run(args, src, crystal, engine, engine_name, mode, wr)
    else:
        engine = _restore_engine(args, src, crystal, state, engine_name)

    if engine is None:
        sys.exit("  Fatal: Engine not initialized.")

    _announce_mode_warnings(engine, state.crystal_mode)

    StateIO.write_csv(state)
    print(f"Project : {state.proj_dir.name}")
    print(f"  Mode    : {state.crystal_mode}")
    if state.run_elastic:
        print("  Elastic : integrated")
    print()

    parallel = max(1, getattr(args, "parallel", 1))
    queue = build_task_queue(state, crystal, engine, args)
    if parallel > 1:
        from orchestrator import ParallelTaskRunner  # noqa: PLC0415
        runner = ParallelTaskRunner(queue, max_workers=parallel)
    else:
        runner = TaskRunner(queue)
    runner.run()

    StateIO.write_csv(state)
    ui.print_summary(state)
    print(f"Results → {state.proj_dir / config.CSV_FILE}")


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand handlers
# ─────────────────────────────────────────────────────────────────────────────


def _cmd_models(args: argparse.Namespace) -> None:
    """vcaforge models — list MLIP model catalogue."""
    # Ensure MLIP engine is loaded so registry is populated
    try:
        import importlib
        importlib.import_module("engines.MLIP.mlip")
    except ImportError:
        pass

    from engines.MLIP.model_registry import (  # noqa: PLC0415
        all_models, available_models, install_hint,
    )

    if args.install:
        hint = install_hint(args.install, args.device)
        if hint:
            print(f"uv pip install {hint}")
        else:
            print(f"Unknown model key: {args.install!r}")
        return

    models = available_models() if args.available else all_models()
    avail_keys = {m["key"] for m in available_models()}

    current_lib = ""
    for m in models:
        if m["library"] != current_lib:
            current_lib = m["library"]
            print(f"\n  ── {current_lib} ──")
        installed = "✓" if m["key"] in avail_keys else "✗"
        stress    = "stress ✓" if m["supports_stress"] else "no stress"
        univ      = "universal" if m["universal"] else "local file"
        print(f"  {installed}  {m['key']:<22}  {m['label'][:38]:<38}  [{stress}, {univ}]")
        if m["key"] not in avail_keys:
            print(f"       install: uv pip install {m['install_cpu']}")

    print()
    if not args.available:
        n_avail = len(avail_keys)
        n_total = len(all_models())
        print(f"  {n_avail}/{n_total} libraries installed. "
              f"Use 'vcaforge models --available' to show only installed models.")


def _cmd_navigate(args: argparse.Namespace) -> None:
    """vcaforge navigate — delegate to navigator module."""
    try:
        import navigator  # noqa: PLC0415
    except ImportError:
        sys.exit(
            "  ERROR: navigator module not found. "
            "Ensure navigator.py is in your Python path."
        )
    nav_args = argparse.Namespace(
        dir=getattr(args, "dir", None),
        target=getattr(args, "target", None) or config.NAVIGATOR_TARGET,
        mode=getattr(args, "mode", None) or config.NAVIGATOR_MODE,
        batch=getattr(args, "batch", None),
        max_iter=getattr(args, "max_iter", None) or config.NAVIGATOR_LOOP_MAX_ITER,
        no_auto_run=getattr(args, "no_auto_run", False),
    )
    if hasattr(navigator, "main_cli"):
        navigator.main_cli(nav_args)
    else:
        sys.exit(
            "  ERROR: navigator.py does not expose main_cli(). "
            "Update navigator.py to support CLI invocation."
        )


def _cmd_aggregate(args: argparse.Namespace) -> None:
    """vcaforge aggregate — delegate to aggregator module."""
    try:
        import aggregator  # noqa: PLC0415
    except ImportError:
        sys.exit("  ERROR: aggregator module not found.")
    dirs = getattr(args, "dirs", None) or [Path(".")]
    out  = getattr(args, "out", None)
    if hasattr(aggregator, "aggregate_dirs"):
        aggregator.aggregate_dirs(
            dirs, output=out, 
            crystal_mode=getattr(args, "mode", None),
            target=getattr(args, "target", None),
            mode=getattr(args, "direction", "maximize") or "maximize"
        )
    else:
        sys.exit("  ERROR: aggregator.py does not expose aggregate_dirs().")


def _cmd_visualize(args: argparse.Namespace) -> None:
    """vcaforge visualize — delegate to vca_visualizer module."""
    try:
        import vca_visualizer  # noqa: PLC0415
    except ImportError:
        sys.exit("  ERROR: vca_visualizer module not found.")
    if hasattr(vca_visualizer, "main_cli"):
        vca_visualizer.main_cli(args)
    else:
        sys.exit("  ERROR: vca_visualizer.py does not expose main_cli().")



def _setup_new_run(
    args: argparse.Namespace, src: Path, crystal: Crystal, engine_name: str,
) -> tuple["BaseEngine", str, Any]:
    """Returns (engine, mode, wizard_result)."""
    if engine_name not in ENGINES:
        sys.exit(
            f"  ERROR: Engine '{engine_name}' not registered. "
            f"Available: {list(ENGINES)}"
        )

    engine_cls = ENGINES[engine_name]
    override = getattr(args, "engine_cmd_override", None)
    engine, _ = engine_cls.setup_interactive(src, crystal, override, args)

    mode = _select_mode(args, engine)

    if isinstance(engine, WizardBypassCapable):
        wr = engine.bypass_wizard(crystal, args)
    else:
        wr = ui.wizard_mode(crystal=crystal, args=args)

    if mode == "sqs" and not wr.single_mode:
        target_at_x1 = {e: f for e, f in wr.target_mix.items() if f > 1e-9}
        if target_at_x1:
            _setup_sqs_strategy(args, crystal, wr.template_element, target_at_x1)

    return engine, mode, wr


def _create_run(
    args: argparse.Namespace, src: Path, crystal: Crystal,
    engine: "BaseEngine", engine_name: str, mode: str, wr: Any,
) -> RunState:
    if wr.single_mode:
        species_list = [(next(iter(wr.target_mix)), 1.0)]
    else:
        # The template element MUST be the first element in species_list,
        # as it is the one that phases out (1-x) in _build_target_mix.
        template = wr.template_element
        species_list = [(template, wr.target_mix.get(template, 0.0))]
        for k, v in sorted(wr.target_mix.items()):
            if k != template:
                species_list.append((k, v))

    crystal_kwargs: dict[str, object] = {}
    if mode == "sqs":
        crystal_kwargs["max_extent"] = (
            args.sqs_max or config.SQS_MAX_EXTENT
        )
        crystal_kwargs["iterations"] = (
            args.sqs_iterations or config.SQS_ITERATIONS
        )

    engine_cmd = getattr(engine, "engine_cmd", "")
    return StateIO.new_run(
        seed=src.stem,
        proj_dir=src.parent
        / f"{src.stem}_{datetime.now().strftime('%b%d_%H-%M')}",
        template_element=wr.template_element,
        species=species_list,
        engine_cmd=engine_cmd,
        engine_name=engine_name,
        engine_kwargs=engine.to_dict(),
        c_start=wr.c_start, c_end=wr.c_end, n_steps=wr.n_steps,
        points=wr.points,
        single_mode=wr.single_mode,
        nonmetal=wr.nonmetal, nonmetal_occ=1.0,
        run_elastic=wr.run_elastic,
        crystal_mode=mode,
        crystal_kwargs=crystal_kwargs,
    )


def _restore_engine(
    args: argparse.Namespace, src: Path, crystal: Crystal,
    state: RunState, engine_name: str,
) -> "BaseEngine":
    if state.engine_name and state.engine_name != engine_name:
        print(
            f"  ⚠  Run was created with '{state.engine_name}' but current "
            f"engine is '{engine_name}'. Using '{state.engine_name}'."
        )
        engine_name = state.engine_name

    if engine_name not in ENGINES:
        sys.exit(f"  ERROR: Engine '{engine_name}' not available for resume.")

    engine_cls = ENGINES[engine_name]

    if state.crystal_mode == "sqs" and state.crystal_kwargs:
        register_strategy("sqs", SQSStrategy(**state.crystal_kwargs))

    try:
        return engine_cls.from_dict(dict(state.engine_kwargs))
    except Exception as exc:
        print(
            f"  ⚠  Could not restore engine from saved kwargs ({exc}), "
            f"re-running setup..."
        )
        override = getattr(args, "engine_cmd_override", None)
        engine, _ = engine_cls.setup_interactive(src, crystal, override, args)
        return engine


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted. State saved.")
        sys.exit(0)
