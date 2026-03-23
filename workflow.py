"""
workflow.py — Process orchestration for VCA concentration sweeps.
─────────────────────────────────────────────────────────────────
Responsibilities:
  • RunState / Step dataclasses with crash-safe JSON persistence
  • CASTEP subprocess execution with live progress monitoring
  • Ctrl+C → skip current step (does NOT lose completed results)
  • Dynamic CSV export (column set adapts to parsed .castep keys)
  • ΔH_mix calculation from completed steps
  • Single-compound (non-VCA) execution mode

No user prompts here. All UI text is in ui.py.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from castep_io import CastepResult, nextra_for_step, parse_castep_log, patch_nextra_bands

VERSION = "5.1"

# ─────────────────────────────────────────────────────────────────────────────
# Step status constants
# ─────────────────────────────────────────────────────────────────────────────

PENDING = "pending"
RUNNING = "running"
DONE = "done"
SKIPPED = "skipped"
FAILED = "failed"

STATUS_ICON: dict[str, str] = {
    DONE: "✓",
    SKIPPED: "⊘",
    FAILED: "✗",
    PENDING: "·",
    RUNNING: "▶",
}

# Large CASTEP binary output files — deleted after successful steps (unless --keep-all)
_CLEANUP_GLOBS = (
    "*.castep_bin",
    "*.check",
    "*.cst_esp",
    "*.bands",
    "*.bib",
    "*.usp",
    "*.upf",
    "*.oepr",
)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Step:
    """One concentration point in a VCA sweep."""

    idx: int
    concentration: float
    status: str = PENDING
    step_dir: str = ""
    started_at: str = ""
    finished_at: str = ""
    rc: str = ""
    # Physical results — populated after parsing .castep output
    parsed: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Flatten to a row dict suitable for CSV / JSON export."""
        base: dict[str, Any] = {
            "step": self.idx,
            "concentration": self.concentration,
            "status": self.status,
            "step_dir": self.step_dir,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "rc": self.rc,
        }
        base.update(self.parsed)
        return base

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Step":
        known = {"idx", "step", "concentration", "status", "step_dir",
                 "started_at", "finished_at", "rc", "parsed"}
        idx = data.get("step", data.get("idx", 0))
        parsed = data.get("parsed", {})
        # Back-compat: old format stored fields directly on Step
        extra = {k: v for k, v in data.items() if k not in known}
        parsed.update(extra)
        return cls(
            idx=idx,
            concentration=data.get("concentration", 0.0),
            status=data.get("status", PENDING),
            step_dir=data.get("step_dir", ""),
            started_at=data.get("started_at", ""),
            finished_at=data.get("finished_at", ""),
            rc=data.get("rc", ""),
            parsed=parsed,
        )

    # Convenience accessors (read-only) for UI display
    @property
    def enthalpy_eV(self) -> str:
        return str(self.parsed.get("enthalpy_eV", ""))

    @property
    def a_opt_ang(self) -> str:
        return str(self.parsed.get("a_opt_ang", ""))

    @property
    def wall_time_s(self) -> str:
        return str(self.parsed.get("wall_time_s", ""))

    @property
    def geom_converged(self) -> str:
        return str(self.parsed.get("geom_converged", ""))

    @property
    def warnings(self) -> str:
        return str(self.parsed.get("warnings", ""))


@dataclass
class RunState:
    """Full persistent state for one VCA run session."""

    version: str
    seed: str               # CASTEP job name (= structure stem)
    proj_dir: Path
    species_a: str
    species_b: str
    castep_cmd: str
    c_start: float
    c_end: float
    n_steps: int
    created_at: str
    single_mode: bool = False  # True = single compound, no VCA sweep
    steps: list[Step] = field(default_factory=list)

    # ── JSON round-trip ──────────────────────────────────────────────────────

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["proj_dir"] = str(self.proj_dir)
        data["steps"] = [s.to_dict() for s in self.steps]
        return data

    @classmethod
    def from_json(cls, data: dict[str, Any], proj_dir: Path) -> "RunState":
        steps = [Step.from_dict(s) for s in data.get("steps", [])]
        return cls(
            version=data.get("version", "?"),
            seed=data["seed"],
            proj_dir=proj_dir,
            species_a=data.get("species_a", ""),
            species_b=data.get("species_b", ""),
            castep_cmd=data.get("castep_cmd", ""),
            c_start=data.get("c_start", 0.0),
            c_end=data.get("c_end", 1.0),
            n_steps=data.get("n_steps", 0),
            created_at=data.get("created_at", ""),
            single_mode=data.get("single_mode", False),
            steps=steps,
        )

    # ── Convenience counts ───────────────────────────────────────────────────

    @property
    def n_done(self) -> int:
        return sum(1 for s in self.steps if s.status == DONE)

    @property
    def n_pending(self) -> int:
        return sum(1 for s in self.steps if s.status == PENDING)

    @property
    def n_failed(self) -> int:
        return sum(1 for s in self.steps if s.status == FAILED)


# ─────────────────────────────────────────────────────────────────────────────
# State persistence (crash-safe: write-then-rename)
# ─────────────────────────────────────────────────────────────────────────────


def _state_path(proj_dir: Path) -> Path:
    return proj_dir / "vca_state.json"


def save_run(state: RunState) -> None:
    """Atomically persist RunState to disk (write tmp → rename)."""
    dst = _state_path(state.proj_dir)
    tmp = dst.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(state.to_json(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    tmp.replace(dst)


def load_run(proj_dir: Path) -> RunState | None:
    """Load RunState from disk; returns None if not found or corrupted."""
    state_file = _state_path(proj_dir)
    if not state_file.exists():
        return None
    try:
        data = json.loads(state_file.read_text(encoding="utf-8"))
        return RunState.from_json(data, proj_dir)
    except (json.JSONDecodeError, KeyError):
        return None


def new_run(
    seed: str,
    proj_dir: Path,
    species_a: str,
    species_b: str,
    castep_cmd: str,
    c_start: float,
    c_end: float,
    n_steps: int,
    single_mode: bool = False,
) -> RunState:
    """Create and persist a fresh RunState with evenly-spaced concentration steps."""
    proj_dir.mkdir(parents=True, exist_ok=True)
    if single_mode:
        steps = [Step(idx=0, concentration=0.0)]
    else:
        delta = (c_end - c_start) / n_steps
        steps = [
            Step(
                idx=i,
                concentration=round(
                    c_end if i == n_steps else c_start + i * delta, 10
                ),
            )
            for i in range(n_steps + 1)
        ]
    state = RunState(
        version=VERSION,
        seed=seed,
        proj_dir=proj_dir,
        species_a=species_a,
        species_b=species_b,
        castep_cmd=castep_cmd,
        c_start=c_start,
        c_end=c_end,
        n_steps=n_steps,
        created_at=_now(),
        single_mode=single_mode,
        steps=steps,
    )
    save_run(state)
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic CSV export
# ─────────────────────────────────────────────────────────────────────────────

_CSV_FIXED_FIELDS = [
    "step", "concentration", "status", "step_dir",
    "started_at", "finished_at", "rc",
]


def write_csv(state: RunState) -> Path:
    """
    Write vca_results.csv with dynamic column headers.

    The column set is built from all keys present in any step's parsed dict,
    so new physical fields (elastic constants, spin, etc.) appear automatically
    without any hardcoding.
    """
    # Collect all dynamic field names across all steps (preserving insertion order)
    dynamic_fields: list[str] = []
    seen_fields: set[str] = set()
    for step in state.steps:
        for key in step.parsed:
            if key not in seen_fields:
                seen_fields.add(key)
                dynamic_fields.append(key)

    all_fields = _CSV_FIXED_FIELDS + dynamic_fields
    out_path = state.proj_dir / "vca_results.csv"

    with out_path.open("w", newline="", encoding="utf-8") as csv_file:
        csv_file.write(f"# vca_tool v{VERSION} — VCA Results\n")
        csv_file.write(f"# System  : {state.species_a}(1-x){state.species_b}(x)\n")
        csv_file.write(f"# Seed    : {state.seed}\n")
        csv_file.write(
            f"# Range   : {state.c_start} → {state.c_end}"
            f"  ({state.n_steps} intervals)\n"
        )
        csv_file.write(f"# Updated : {_now()}\n#\n")
        writer = csv.DictWriter(
            csv_file, fieldnames=all_fields, extrasaction="ignore", restval="N/A"
        )
        writer.writeheader()
        writer.writerows(s.to_dict() for s in state.steps)

    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# ΔH_mix calculation
# ─────────────────────────────────────────────────────────────────────────────


def mixing_enthalpy(
    steps: list[Step],
) -> list[tuple[float, float, float]]:
    """
    Compute ΔH_mix(x) = H(x) − [(1−x)·H(0) + x·H(1)] in meV/f.u.

    Returns list of (x, H_eV, dH_meV) sorted by x.
    Returns empty list if endpoint enthalpies are unavailable.
    """
    done_steps = [
        s for s in steps if s.status == DONE and s.parsed.get("enthalpy_eV")
    ]

    try:
        h_at_zero = next(
            float(s.parsed["enthalpy_eV"])
            for s in done_steps
            if abs(s.concentration) < 1e-6
        )
        h_at_one = next(
            float(s.parsed["enthalpy_eV"])
            for s in done_steps
            if abs(s.concentration - 1.0) < 1e-6
        )
    except StopIteration:
        return []

    result: list[tuple[float, float, float]] = []
    for step in sorted(done_steps, key=lambda s: s.concentration):
        x = step.concentration
        h = float(step.parsed["enthalpy_eV"])
        dh_meV = (h - ((1 - x) * h_at_zero + x * h_at_one)) * 1000.0
        result.append((x, h, dh_meV))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Binary validation
# ─────────────────────────────────────────────────────────────────────────────


def cmd_is_valid(cmd: str) -> bool:
    """Return True if the first token of cmd resolves to an executable."""
    binary = os.path.expanduser(cmd.split()[0])
    if Path(binary).is_absolute() or "/" in binary:
        return Path(binary).is_file() and os.access(binary, os.X_OK)
    return shutil.which(binary) is not None


# ─────────────────────────────────────────────────────────────────────────────
# Ctrl+C → skip (not quit)
# ─────────────────────────────────────────────────────────────────────────────

_SKIP_CURRENT_STEP = False


def _arm_skip_signal() -> None:
    global _SKIP_CURRENT_STEP
    _SKIP_CURRENT_STEP = False

    def _handler(sig: int, frame: Any) -> None:  # noqa: ANN401
        global _SKIP_CURRENT_STEP
        _SKIP_CURRENT_STEP = True
        print("\n  [Ctrl+C] — terminating step …\n")

    signal.signal(signal.SIGINT, _handler)


def _disarm_skip_signal() -> None:
    signal.signal(signal.SIGINT, signal.default_int_handler)


# ─────────────────────────────────────────────────────────────────────────────
# Live progress monitor (background thread)
# ─────────────────────────────────────────────────────────────────────────────

_RE_GEO_ITER = re.compile(r"LBFGS: finished iteration\s+(\d+)", re.I)
_RE_SCF_LINE = re.compile(r"^\s+(\d+)\s+[-\d.E+]+\s+[-\d.E+]+", re.M)
_RE_MAX_GEO = re.compile(r"max\. number of steps\s+:\s+(\d+)", re.I)
_RE_MAX_SCF = re.compile(r"max\. number of SCF cycles\s+:\s+(\d+)", re.I)


def _progress_monitor(castep_file: Path, stop_event: threading.Event) -> None:
    """Background thread: read growing .castep file, print one-line progress bar."""
    BAR_WIDTH = 22
    geo_iter = 0
    max_geo = 100
    scf_cycle = 0
    last_size = 0
    t_start = time.time()

    while not stop_event.is_set():
        try:
            if not castep_file.exists():
                time.sleep(1)
                continue
            current_size = castep_file.stat().st_size
            if current_size == last_size:
                time.sleep(0.8)
                continue
            last_size = current_size
            text = castep_file.read_text(encoding="utf-8", errors="replace")

            match = _RE_MAX_GEO.search(text)
            if match:
                max_geo = int(match.group(1))
            match = _RE_MAX_SCF.search(text)
            if match:
                pass  # max_scf unused in display but parsed for completeness

            geo_hits = _RE_GEO_ITER.findall(text)
            if geo_hits:
                geo_iter = int(geo_hits[-1])
            scf_hits = _RE_SCF_LINE.findall(text)
            if scf_hits:
                scf_cycle = int(scf_hits[-1])

            elapsed = int(time.time() - t_start)
            minutes, seconds = divmod(elapsed, 60)
            pct = min(100, int(geo_iter / max(max_geo, 1) * 100))
            filled = pct * BAR_WIDTH // 100
            bar = "█" * filled + "░" * (BAR_WIDTH - filled)
            line = (
                f"  │  [{bar}] {pct:3d}%"
                f"  geo {geo_iter}/{max_geo}"
                f"  scf {scf_cycle}"
                f"  ⏱ {minutes:02d}:{seconds:02d}"
            )
            print(line + " " * 4, end="\r", flush=True)
        except OSError:
            pass
        time.sleep(0.8)

    print(" " * 76, end="\r")  # clear progress line


# ─────────────────────────────────────────────────────────────────────────────
# Subprocess execution
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ExecResult:
    rc: int | None
    skipped: bool
    stderr_tail: list[str]


def _run_castep_process(cmd: str, cwd: Path, castep_out: Path) -> ExecResult:
    """
    Run cmd in cwd, monitoring castep_out for progress.
    Ctrl+C sends SIGTERM to the child process and returns skipped=True.
    stderr is captured (last 40 lines retained).
    """
    stop_event = threading.Event()
    progress_thread = threading.Thread(
        target=_progress_monitor, args=(castep_out, stop_event), daemon=True
    )
    progress_thread.start()

    proc: subprocess.Popen | None = None  # type: ignore[type-arg]
    stderr_lines: list[str] = []
    return_code = -1

    _arm_skip_signal()
    try:
        proc = subprocess.Popen(
            cmd, shell=True, cwd=cwd,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )

        def _drain_stderr() -> None:
            try:
                assert proc is not None
                for raw_line in proc.stderr:  # type: ignore[union-attr]
                    decoded = raw_line.decode(errors="replace").rstrip()
                    stderr_lines.append(decoded)
                    if len(stderr_lines) > 40:
                        stderr_lines.pop(0)
            except OSError:
                pass

        drain_thread = threading.Thread(target=_drain_stderr, daemon=True)
        drain_thread.start()

        while proc.poll() is None:
            if _SKIP_CURRENT_STEP:
                stop_event.set()
                progress_thread.join(timeout=2)
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                _disarm_skip_signal()
                return ExecResult(rc=None, skipped=True, stderr_tail=[])
            time.sleep(0.5)

        drain_thread.join(timeout=3)
        return_code = proc.returncode

    except OSError as exc:
        stderr_lines.append(str(exc))
    finally:
        stop_event.set()
        progress_thread.join(timeout=2)
        _disarm_skip_signal()
        if proc is not None and proc.poll() is None:
            proc.kill()

    return ExecResult(rc=return_code, skipped=False, stderr_tail=stderr_lines)


# ─────────────────────────────────────────────────────────────────────────────
# Step execution
# ─────────────────────────────────────────────────────────────────────────────


def execute_step(
    state: RunState,
    step: Step,
    cell_src: Path,
    param_src: Path,
    *,
    write_cell_fn: Callable[[Path, float], None],
    keep_all: bool = False,
) -> ExecResult:
    """
    Prepare files, run CASTEP for one step, parse results, persist state.

    write_cell_fn(dest_path, concentration) is a callback that writes the .cell
    file — keeps workflow.py decoupled from castep_io.py cell generation logic.

    The original .cell and .param are also copied once into the project root
    as backups (original_<seed>.cell / original_<seed>.param) so the source
    files in the user's working directory are never modified.
    """
    x = step.concentration
    seed = state.seed
    step_dir_name = f"x{x:.4f}"
    step_dir = state.proj_dir / step_dir_name
    step_dir.mkdir(parents=True, exist_ok=True)
    step.step_dir = step_dir_name

    # Back up original source files into the project root (once, non-destructive)
    _backup_source(state.proj_dir, cell_src, f"original_{seed}.cell")
    _backup_source(state.proj_dir, param_src, f"original_{seed}.param")

    # Write VCA .cell (via callback) and write per-step .param with correct nextra_bands
    write_cell_fn(step_dir / f"{seed}.cell", x)
    step_param = step_dir / f"{seed}.param"
    shutil.copy2(param_src, step_param)
    # Patch nextra_bands: interpolated per-step value (pure endpoints get less,
    # VCA middle gets +10 overhead for fractional-charge band complexity)
    # In single_mode species_b is empty — use species_a only (no VCA overhead)
    if state.single_mode or not state.species_b:
        from castep_io import nextra_for_element
        nextra = nextra_for_element(state.species_a)
    else:
        nextra = nextra_for_step(state.species_a, state.species_b, x)
    patch_nextra_bands(step_param, nextra)
    step.parsed["nextra_bands_used"] = nextra

    # Mark RUNNING before launch (crash-safe: resets to PENDING on next resume)
    step.status = RUNNING
    step.started_at = _now()
    save_run(state)
    write_csv(state)

    if not state.castep_cmd:
        # Prepare-only mode — no CASTEP execution
        step.status = DONE
        step.rc = "N/A"
        step.finished_at = _now()
        save_run(state)
        write_csv(state)
        return ExecResult(rc=0, skipped=False, stderr_tail=[])

    cmd = os.path.expanduser(state.castep_cmd.replace("{seed}", seed))
    exec_result = _run_castep_process(cmd, step_dir, step_dir / f"{seed}.castep")
    step.finished_at = _now()

    if exec_result.skipped:
        step.status = SKIPPED
        step.rc = "ctrl-c"
    else:
        step.rc = str(exec_result.rc) if exec_result.rc is not None else "unknown"
        castep_result: CastepResult = parse_castep_log(step_dir / f"{seed}.castep")
        # Store all parsed fields dynamically — no hardcoded field list
        step.parsed = castep_result.to_dict()

        if exec_result.rc == 0:
            step.status = DONE
            if not keep_all:
                _cleanup_large_files(step_dir)
        else:
            step.status = FAILED

    save_run(state)
    write_csv(state)
    return exec_result


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _backup_source(proj_dir: Path, src: Path, dest_name: str) -> None:
    """
    Copy src into proj_dir/dest_name if it does not already exist.
    Preserves the original file untouched in the user's working directory.
    """
    dest = proj_dir / dest_name
    if not dest.exists() and src.exists():
        try:
            shutil.copy2(src, dest)
        except OSError:
            pass  # non-critical — log file backup failure silently


def _cleanup_large_files(step_dir: Path) -> None:
    for glob_pattern in _CLEANUP_GLOBS:
        for file_path in step_dir.glob(glob_pattern):
            try:
                file_path.unlink()
            except OSError:
                pass
