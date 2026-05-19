"""
navigator_loop.py  —  Active-learning loop for VCAForge (N-component aware).
═══════════════════════════════════════════════════════════════════════════
Drives the cycle:
    suggest x → run VCAForge for that composition → aggregate → reassess → repeat

Major design points
───────────────────
1.  **stdin/stdout pass-through.** VCAForge has interactive prompts. We do
    NOT capture its output; subprocess inherits the parent's stdio so the
    user sees and answers prompts in real time.

2.  **N-component aware.** Builds commands using the new --species M:f format
    that the user's main.py accepts:
        python main.py STRUCT --species Ti:0.5 Nb:0.3 V:0.2 --engine castep -n 0 --elastic

3.  **Aggregator integration.** After every iteration (if config flag is on),
    re-aggregate all CSVs into the master CSV. Subsequent iterations read
    from the master, which means the GP sees a clean, deduplicated picture
    of all data so far.

4.  **Completion-aware stop.** Uses navigator.assess_completion() to detect
    when more iterations have diminishing returns. Reports a final summary
    with elapsed time, peak composition, target value, and confidence.

5.  **Stop conditions** (any of):
      - max_iter reached
      - assess_completion returns is_done=True
      - new x is within NAVIGATOR_LOOP_DEDUPE_TOL of an existing point
        AND that has happened ≥ NAVIGATOR_DONE_DEDUPE_REPEATS times in a row
      - VCAForge returned non-zero AND user declined to continue
      - VCAForge ran but new result didn't appear in any CSV
      - user pressed Ctrl-C (graceful — prints summary, preserves state)
"""

from __future__ import annotations

import logging
import shlex
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

import config
from navigator import (
    Candidate,
    NavigatorOrchestrator,
    SystemSpec,
    assess_completion,
    build_vcaforge_command,
    parse_system_from_string,
)

log = logging.getLogger("vcaforge.navigator_loop")


def _cfg(name: str, default: Any) -> Any:
    return getattr(config, name, default)


def _dedupe_tol() -> float:
    return float(_cfg("NAVIGATOR_LOOP_DEDUPE_TOL", 1e-3))


def _aggregate_on_iter() -> bool:
    return bool(_cfg("NAVIGATOR_AGGREGATE_ON_ITER", True))


def _scan_parent() -> bool:
    return bool(_cfg("NAVIGATOR_LOOP_SCAN_PARENT", True))


def _mpi_procs() -> int:
    return int(_cfg("NAVIGATOR_LOOP_MPI_PROCS", 0))


def _run_elastic_default() -> bool:
    return bool(_cfg("NAVIGATOR_LOOP_RUN_ELASTIC", True))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_banner(text: str) -> None:
    print()
    print("  " + "═" * 65)
    print(f"  {text}")
    print("  " + "═" * 65)


def _composition_distance(a: dict[str, float], b: dict[str, float]) -> float:
    """Euclidean distance between two compositions (must have same keys)."""
    keys = set(a) | set(b)
    return float(np.sqrt(sum((a.get(k, 0.0) - b.get(k, 0.0)) ** 2 for k in keys)))


def _x_already_evaluated(
    cand_comp: dict[str, float],
    existing_comps: list[dict[str, float]],
    tol: float,
) -> dict[str, float] | None:
    if not existing_comps:
        return None
    dists = [_composition_distance(cand_comp, c) for c in existing_comps]
    idx = int(np.argmin(dists))
    if dists[idx] < tol:
        return existing_comps[idx]
    return None


def _existing_compositions_for_system(
    nav: NavigatorOrchestrator, system: SystemSpec,
) -> list[dict[str, float]]:
    return [r.composition for r in nav.ingestor.loaded_rows() if r.system == system]


def _confirm(prompt: str, default: bool = True) -> bool:
    suffix = " [Y/n]: " if default else " [y/N]: "
    if not sys.stdin.isatty():
        print(f"{prompt}{suffix}(non-interactive → {default})")
        return default
    try:
        ans = input(prompt + suffix).strip().lower()
    except EOFError:
        return default
    if not ans:
        return default
    return ans.startswith("y")


def _format_duration(seconds: float) -> str:
    td = timedelta(seconds=int(seconds))
    return str(td)


# ─────────────────────────────────────────────────────────────────────────────
# Subprocess driver
# ─────────────────────────────────────────────────────────────────────────────

def _run_vcaforge(cmd: str, cwd: Path | None = None) -> int:
    print()
    print(f"  $ {cmd}")
    if cwd is not None:
        print(f"  (cwd: {cwd})")
    print()
    parts = shlex.split(cmd)
    if parts and parts[0] in ("python", "python3"):
        parts[0] = sys.executable
    try:
        proc = subprocess.run(
            parts,
            cwd=str(cwd) if cwd else None,
            stdin=None, stdout=None, stderr=None,    # inherit
            check=False,
        )
        return proc.returncode
    except FileNotFoundError as exc:
        print(f"  ERROR: command not found: {exc}", file=sys.stderr)
        return 127


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation hook (lazy import — aggregator is optional)
# ─────────────────────────────────────────────────────────────────────────────

def _aggregate_if_configured(
    base_dir: Path,
    system: SystemSpec,
    target: str | None,
    mode: str | None,
    crystal_mode: str | None = None,
) -> Path | None:
    if not _aggregate_on_iter():
        return None
    try:
        from aggregator import ResultAggregator
    except ImportError:
        log.debug("aggregator.py not available — skipping master CSV update.")
        return None
    try:
        agg = ResultAggregator(target=target, mode=mode or "maximize",
                               dedupe_tol=_dedupe_tol(),
                               crystal_mode=crystal_mode)
        return agg.aggregate(base_dir, system)
    except (FileNotFoundError, ValueError) as exc:
        log.debug("Aggregation skipped (%s)", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Loop
# ─────────────────────────────────────────────────────────────────────────────

def run_loop(
    structure_file: Path,
    csv_path: Path | None,
    base_dir: Path | None,
    target: str | None,
    mode: str | None,
    acquisition: str | None,
    target_system: str | None,
    engine: str = "castep",
    max_iter: int = 30,
    verbose: bool = False,
    sqs_sites: int | None = None,
    crystal_mode: str | None = None,
    template: str | None = None,
) -> int:
    """Active-learning loop. Returns shell exit code (0 success)."""

    structure_file = Path(structure_file).expanduser().resolve()
    if not structure_file.exists():
        print(f"  ERROR: Structure file not found: {structure_file}", file=sys.stderr)
        return 2

    if csv_path is None and base_dir is None:
        if _scan_parent():
            base_dir = structure_file.parent
        else:
            base_dir = Path.cwd()

    target_sys: SystemSpec | None = None
    if target_system:
        try:
            target_sys = parse_system_from_string(target_system)
        except ValueError as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            return 2

    vca_cwd = Path.cwd()
    if not (vca_cwd / "main.py").exists():
        print(f"  WARNING: main.py not found in {vca_cwd}.\n"
              f"  Run navigator from VCAForge's project directory.",
              file=sys.stderr)

    _print_banner(f"VCAForge Active Learning Loop  (max {max_iter} iterations)")
    print(f"    Structure : {structure_file}")
    print(f"    Scan dir  : {base_dir or csv_path}")
    print(f"    Engine    : {engine}")
    print(f"    Target    : {target or _cfg('NAVIGATOR_TARGET', 'H_Vickers_GPa')}  "
          f"({mode or _cfg('NAVIGATOR_MODE', 'maximize')})")
    if target_sys:
        print(f"    System    : {target_sys.label()}  (forced, N={target_sys.n_metals})")
    else:
        print(f"    System    : auto-detect from data")
    print(f"    Aggregate : {'on' if _aggregate_on_iter() else 'off'} per iteration")

    # Loop state
    consecutive_dupes = 0
    last_candidate: Candidate | None = None
    iter_history: list[dict[str, Any]] = []
    loop_start = time.monotonic()
    iteration = 0

    try:
        for iteration in range(1, max_iter + 1):
            _print_banner(f"Iteration {iteration} / {max_iter}")
            iter_start = time.monotonic()

            # 1. Suggest next point based on current state of CSVs.
            try:
                nav = NavigatorOrchestrator(
                    csv_path=csv_path,
                    base_dir=base_dir,
                    target=target,
                    mode=mode,
                    acquisition=acquisition,
                    target_system=target_sys,
                    sqs_sites=sqs_sites,
                )
                cand = nav.suggest()
                report = nav.report()
            except (FileNotFoundError, ValueError) as exc:
                print(f"  ERROR: {exc}", file=sys.stderr)
                return 2

            resolved_sys = report.target_system or cand.system
            if resolved_sys is None:
                print("  ERROR: Could not resolve target system. Use --system.",
                      file=sys.stderr)
                return 2
            if target_sys is None:
                target_sys = resolved_sys
                print(f"    Auto-detected system: {target_sys.label()} (N={target_sys.n_metals})")

            print()
            print(f"    Suggestion : {cand.composition_str() if cand.composition else f'x={cand.concentration:.4f}'}")
            if np.isfinite(cand.predicted_mean):
                print(f"    Predicted  : {cand.predicted_mean:.3f} ± {cand.predicted_std:.3f}")
            if cand.p_feasible is not None and np.isfinite(cand.p_feasible):
                print(f"    P(stable)  : {int(round(cand.p_feasible * 100))}%")
            if cand.acq_value is not None:
                print(f"    {report.acquisition} score  : {cand.acq_value:.4f}")
            print(f"    Source     : {cand.source}")

            # 2. Completion check
            completion = assess_completion(nav, consecutive_duplicate_suggestions=consecutive_dupes)
            if completion.is_done:
                _print_banner(f"Study complete: {completion.reason}")
                last_candidate = cand
                break

            # 3. Dedupe check
            existing = _existing_compositions_for_system(nav, target_sys)
            tol = _dedupe_tol()
            collision = (
                _x_already_evaluated(cand.composition, existing, tol)
                if cand.composition else None
            )
            if collision is not None:
                consecutive_dupes += 1
                print()
                print(f"  ⚠  Suggested composition duplicates existing data:")
                print(f"     suggested = {cand.composition_str()}")
                print(f"     existing  = " + " ".join(f"{m}={v:.4f}" for m, v in collision.items()))
                print(f"     consecutive duplicates: {consecutive_dupes}")
                if consecutive_dupes >= int(_cfg("NAVIGATOR_DONE_DEDUPE_REPEATS", 3)):
                    _print_banner("Stop — too many consecutive duplicate suggestions.")
                    last_candidate = cand
                    break
                # Otherwise let the user decide whether to skip
                if not _confirm("  Re-run anyway (will probably suggest same point next time)?",
                                default=False):
                    last_candidate = cand
                    break
            else:
                consecutive_dupes = 0

            # 4. Build & run command
            cand.system = target_sys
            cmd = build_vcaforge_command(
                structure_file, cand,
                engine=engine, n_steps=0,
                run_elastic=_run_elastic_default(),
                mpi_procs=_mpi_procs() if _mpi_procs() > 0 else None,
                crystal_mode=crystal_mode,
                template=template,
            )
            print()
            print("  Run this command? (terminal pass-through — answer prompts as they appear)")

            t0 = time.monotonic()
            rc = _run_vcaforge(cmd, cwd=vca_cwd)
            dt = time.monotonic() - t0

            if rc != 0:
                print()
                print(f"  ⚠  VCAForge exited with code {rc} after {dt:.1f}s.")
                if not _confirm("  Continue with next iteration anyway?", default=False):
                    last_candidate = cand
                    break
            else:
                print()
                print(f"  ✓ VCAForge finished in {dt:.1f}s.")

            # 5. Sanity: did the new point appear?
            try:
                check_nav = NavigatorOrchestrator(
                    csv_path=csv_path, base_dir=base_dir,
                    target=target, mode=mode, acquisition=acquisition,
                    target_system=target_sys,
                    sqs_sites=sqs_sites,
                )
                check_nav.ingestor.load()
                xs_after = _existing_compositions_for_system(check_nav, target_sys)
                if cand.composition and _x_already_evaluated(cand.composition, xs_after, tol) is None:
                    print(f"  ⚠  New composition did not appear in any CSV. "
                          f"Result file may be in an unscanned directory.")
                    if not _confirm("  Continue anyway?", default=False):
                        last_candidate = cand
                        break
            except (FileNotFoundError, ValueError) as exc:
                log.debug("Post-run rescan failed: %s", exc)

            # 6. Aggregate (master CSV update)
            master = _aggregate_if_configured(
                base_dir if base_dir else structure_file.parent,
                target_sys, target, mode,
                crystal_mode=crystal_mode,
            )
            if master:
                print(f"  Master CSV updated: {master.name}")
                # Subsequent iterations will read from this master automatically
                # because DataIngestor scans for both vca_results.csv AND *_master.csv

            iter_history.append({
                "iteration": iteration,
                "duration_s": round(time.monotonic() - iter_start, 1),
                "suggested_composition": cand.composition,
                "predicted_mean": cand.predicted_mean,
                "predicted_std": cand.predicted_std,
                "acq_value": cand.acq_value,
            })
            last_candidate = cand
        else:
            _print_banner(f"Reached max iterations ({max_iter}).")

    except KeyboardInterrupt:
        print()
        _print_banner("Interrupted by user (Ctrl-C). State preserved on disk.")
        _print_summary(loop_start, iteration, iter_history, last_candidate, target_sys, target,
                       base_dir, mode, acquisition, csv_path, structure_file, sqs_sites)
        return 130

    # ── Final summary ──────────────────────────────────────────────────────
    _print_summary(loop_start, iteration, iter_history, last_candidate, target_sys, target,
                   base_dir, mode, acquisition, csv_path, structure_file, sqs_sites)
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Final summary
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary(
    loop_start: float,
    iteration: int,
    iter_history: list[dict[str, Any]],
    last_candidate: Candidate | None,
    target_sys: SystemSpec | None,
    target: str | None,
    base_dir: Path | None,
    mode: str | None,
    acquisition: str | None,
    csv_path: Path | None,
    structure_file: Path,
    sqs_sites: int | None = None,
) -> None:
    elapsed = time.monotonic() - loop_start
    _print_banner("Loop summary")
    print(f"    Total time    : {_format_duration(elapsed)}")
    print(f"    Iterations    : {iteration}")
    if iter_history:
        avg = sum(h["duration_s"] for h in iter_history) / max(len(iter_history), 1)
        print(f"    Avg per iter  : {avg:.1f} s")

    if target_sys is None:
        return

    # Final report
    try:
        nav = NavigatorOrchestrator(
            csv_path=csv_path, base_dir=base_dir,
            target=target, mode=mode, acquisition=acquisition,
            target_system=target_sys,
            sqs_sites=sqs_sites,
        )
        nav.suggest()
        rep = nav.report()
        completion = assess_completion(nav)

        print()
        print(f"    System        : {rep.target_system.label() if rep.target_system else '?'}")
        print(f"    Stable points : {rep.n_stable}")
        print(f"    Failed points : {rep.n_failed}")
        if np.isfinite(rep.current_best_value):
            print(f"    Best {rep.target}: {rep.current_best_value:.3f}")
            if rep.current_best_composition:
                comp = rep.current_best_composition
                if len(comp) > 2:
                    comp_str = "  ".join(f"{m}={v:.4f}" for m, v in comp.items())
                    print(f"    At composition: {comp_str}")
                else:
                    print(f"    At x          : {rep.current_best_x:.4f}")
        print(f"    Status        : {completion.short_summary()}")
        print(f"    Confidence    : {completion.confidence_pct}%")

        if completion.is_done:
            print()
            print("    Recommendation: stop — further iterations have diminishing return.")
        elif last_candidate is not None and not last_candidate.converged:
            print()
            print(f"    Continue with : python navigator.py --loop {structure_file} \\")
            print(f"                       --system {target_sys.label()}")
    except Exception as exc:  # noqa: BLE001
        log.warning("Final summary failed: %s", exc)
