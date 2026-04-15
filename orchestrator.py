"""
orchestrator.py  —  Run state, persistence, CSV, subprocess execution.
════════════════════════════════════════════════════════════════════════
No user prompts, no physics.  Pure orchestration:

  RunState / Step  — data model + crash-safe JSON persistence
  execute_step     — engine-agnostic: runs one DFT job, parses results
  write_csv        — live CSV export
  mixing_enthalpy  — ΔH_mix from completed steps

Watchdog
────────
``run_process`` embeds a :class:`_Watchdog` that monitors the CASTEP output
while the subprocess runs and kills it automatically when:

  1. Wall-clock elapsed > ``config.STEP_TIMEOUT_S`` (default 900 s / 15 min).
  2. Smax oscillates above ``config.SMAX_KILL_GPa`` (default 50 GPa) for
     ``config.SMAX_STALL_ITERS`` (default 8) consecutive LBFGS steps.
     This is the exact signature of the TiC_base run that ran for one hour
     with Smax bouncing between 14–270 GPa and never converging.
  3. ``"Reached maximum number of SCF cycles"`` appears in the output
     (charge sloshing / hard divergence).

The kill reason is stored in :attr:`ExecResult.kill_reason` so
:func:`patch_for_recovery` can apply targeted .param fixes before a retry.

Engine contract
───────────────
``execute_step`` calls ``write_cell_fn(dest, x)`` and delegates output
parsing to the Engine protocol from ``engine.py``.  Adding VASP or QE
requires implementing that protocol only — no changes here.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import config

# ─────────────────────────────────────────────────────────────────────────────
# Status constants
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

# Kill-reason tokens — written into ExecResult.kill_reason and Step.parsed.
_KR_TIMEOUT = "timeout"
_KR_SMAX = "smax_stall"
_KR_SCF = "scf_nosconv"
_KR_CTRL_C = "ctrl-c"
_KR_STRESS = "geom_high_stress"


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Step:
    """One concentration point in a VCA sweep, or a single-compound run."""

    idx: int
    concentration: float
    status: str = PENDING
    step_dir: str = ""
    started_at: str = ""
    finished_at: str = ""
    rc: str = ""
    parsed: dict[str, Any] = field(default_factory=dict)

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

    def to_dict(self) -> dict[str, Any]:
        base = {
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
    def from_dict(cls, d: dict[str, Any]) -> "Step":
        known = {
            "idx",
            "step",
            "concentration",
            "status",
            "step_dir",
            "started_at",
            "finished_at",
            "rc",
            "parsed",
        }
        parsed = dict(d.get("parsed") or {})
        parsed.update({k: v for k, v in d.items() if k not in known})
        return cls(
            idx=d.get("step", d.get("idx", 0)),
            concentration=d.get("concentration", 0.0),
            status=d.get("status", PENDING),
            step_dir=d.get("step_dir", ""),
            started_at=d.get("started_at", ""),
            finished_at=d.get("finished_at", ""),
            rc=d.get("rc", ""),
            parsed=parsed,
        )


@dataclass
class RunState:
    """Complete, serialisable state of one VCAForge run."""

    version: str
    seed: str
    proj_dir: Path
    template_element: str
    species: list[tuple[str, float]]
    castep_cmd: str
    c_start: float
    c_end: float
    n_steps: int
    created_at: str
    single_mode: bool = False
    nonmetal: str = ""
    nonmetal_occ: float = 1.0
    run_elastic: bool = False
    steps: list[Step] = field(default_factory=list)

    @property
    def species_a(self) -> str:
        return self.species[0][0] if self.species else ""

    @property
    def species_b(self) -> str:
        return self.species[1][0] if len(self.species) > 1 else ""

    @property
    def n_done(self) -> int:
        return sum(1 for s in self.steps if s.status == DONE)

    @property
    def n_pending(self) -> int:
        return sum(1 for s in self.steps if s.status == PENDING)

    @property
    def n_failed(self) -> int:
        return sum(1 for s in self.steps if s.status == FAILED)

    def system_label(self) -> str:
        """Human-readable formula: ``Ti(1-x)Zr(x)C``, ``Ti(1-x)[Nb0.70V0.30](x)C``."""
        if self.single_mode:
            return self.seed
        sp = self.species
        nm = self.nonmetal
        if len(sp) == 2:
            metal = f"{sp[0][0]}(1-x){sp[1][0]}(x)"
        else:
            inner = "".join(f"{e}{f:.2f}" for e, f in sp[1:])
            metal = f"{sp[0][0]}(1-x)[{inner}](x)"
        return f"{metal}{nm}" if nm else metal

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["proj_dir"] = str(self.proj_dir)
        d["steps"] = [s.to_dict() for s in self.steps]
        return d

    @classmethod
    def from_json(cls, d: dict[str, Any], proj_dir: Path) -> "RunState":
        raw_species = d.get("species")
        if raw_species is None:
            sa = d.get("species_a", "")
            sb = d.get("species_b", "")
            raw_species = [(sa, 0.0), (sb, 1.0)] if sa and sb else [(sa, 0.0)]
        tmpl = d.get("template_element") or (raw_species[0][0] if raw_species else "")
        return cls(
            version=d.get("version", "?"),
            seed=d["seed"],
            proj_dir=proj_dir,
            template_element=tmpl,
            species=raw_species,
            castep_cmd=d.get("castep_cmd", ""),
            c_start=d.get("c_start", 0.0),
            c_end=d.get("c_end", 1.0),
            n_steps=d.get("n_steps", 0),
            created_at=d.get("created_at", ""),
            single_mode=d.get("single_mode", False),
            nonmetal=d.get("nonmetal", ""),
            nonmetal_occ=d.get("nonmetal_occ", 1.0),
            run_elastic=d.get("run_elastic", False),
            steps=[Step.from_dict(s) for s in d.get("steps", [])],
        )


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────


def _state_path(proj_dir: Path) -> Path:
    return proj_dir / config.STATE_FILE


def save_run(state: RunState) -> None:
    """Atomically persist *state* to JSON (write-then-rename)."""
    dst = _state_path(state.proj_dir)
    tmp = dst.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(state.to_json(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp.replace(dst)


def load_run(proj_dir: Path) -> RunState | None:
    f = _state_path(proj_dir)
    if not f.exists():
        return None
    try:
        return RunState.from_json(json.loads(f.read_text(encoding="utf-8")), proj_dir)
    except (json.JSONDecodeError, KeyError):
        return None


def new_run(
    *,
    seed: str,
    proj_dir: Path,
    template_element: str,
    species: list[tuple[str, float]],
    castep_cmd: str,
    c_start: float,
    c_end: float,
    n_steps: int,
    single_mode: bool = False,
    nonmetal: str = "",
    nonmetal_occ: float = 1.0,
    run_elastic: bool = False,
) -> RunState:
    proj_dir.mkdir(parents=True, exist_ok=True)
    if single_mode:
        steps = [Step(idx=0, concentration=0.0)]
    else:
        d = (c_end - c_start) / max(n_steps, 1)
        steps = [
            Step(
                idx=i,
                concentration=round(c_end if i == n_steps else c_start + i * d, 10),
            )
            for i in range(n_steps + 1)
        ]
    state = RunState(
        version=config.VERSION,
        seed=seed,
        proj_dir=proj_dir,
        template_element=template_element,
        species=species,
        castep_cmd=castep_cmd,
        c_start=c_start,
        c_end=c_end,
        n_steps=n_steps,
        created_at=_now(),
        single_mode=single_mode,
        nonmetal=nonmetal,
        nonmetal_occ=nonmetal_occ,
        run_elastic=run_elastic,
        steps=steps,
    )
    save_run(state)
    return state


# ─────────────────────────────────────────────────────────────────────────────
# CSV export
# ─────────────────────────────────────────────────────────────────────────────

_CSV_FIXED = [
    "step",
    "concentration",
    "status",
    "started_at",
    "finished_at",
    "wall_time_s",
    "elastic_wall_time_s",
    "total_wall_time_s",
]

_CSV_ORDER = [
    "concentration",
    "VEC",
    "a_opt_ang",
    "b_opt_ang",
    "c_opt_ang",
    "volume_ang3",
    "density_gcm3",
    "enthalpy_eV",
    "dH_mix_meV_per_fu",
    "C11",
    "C12",
    "C44",
    "B_Voigt_GPa",
    "B_Reuss_GPa",
    "B_Hill_GPa",
    "G_Voigt_GPa",
    "G_Reuss_GPa",
    "G_Hill_GPa",
    "E_GPa",
    "nu",
    "Zener_A",
    "Pugh_ratio",
    "Cauchy_pressure_GPa",
    "C_prime_GPa",
    "Kleinman_zeta",
    "lambda_Lame_GPa",
    "mu_Lame_GPa",
    "H_Vickers_GPa",
    "v_longitudinal_ms",
    "v_transverse_ms",
    "v_mean_ms",
    "T_Debye_K",
    "acoustic_Gruneisen",
    "elastic_source",
    "elastic_n_points",
    "elastic_R2_min",
    "elastic_quality_note",
    "geom_converged",
    "nextra_bands_used",
    "kill_reason",
    "warnings",
    "step_dir",
    "rc",
]


def write_csv(state: RunState) -> Path:
    """Write/overwrite the live results CSV inside *state.proj_dir*."""
    dh_data = mixing_enthalpy(state.steps)
    if dh_data:
        dh_map = {x: dh for x, _, dh in dh_data}
        for s in state.steps:
            if s.status == DONE and s.concentration in dh_map:
                s.parsed.setdefault(
                    "dH_mix_meV_per_fu", f"{dh_map[s.concentration]:.3f}"
                )

    all_keys: set[str] = set()
    for s in state.steps:
        all_keys.update(s.parsed.keys())

    ordered: list[str] = []
    seen: set[str] = set(_CSV_FIXED)
    for k in _CSV_ORDER:
        if k not in seen and (k in all_keys or k in _CSV_FIXED):
            ordered.append(k)
            seen.add(k)
    for s in state.steps:
        for k in s.parsed:
            if k not in seen:
                ordered.append(k)
                seen.add(k)

    all_fields = _CSV_FIXED + ordered
    out = state.proj_dir / config.CSV_FILE

    with out.open("w", newline="", encoding="utf-8") as f:
        f.write(f"# VCAForge v{config.VERSION} — Elastic Constants & Structural Data\n")
        f.write(f"# System  : {state.system_label()}\n")
        f.write(f"# Seed    : {state.seed}\n")
        f.write(f"# Updated : {_now()}\n#\n")
        w = csv.DictWriter(
            f, fieldnames=all_fields, extrasaction="ignore", restval="N/A"
        )
        w.writeheader()
        w.writerows(s.to_dict() for s in state.steps)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# ΔH_mix
# ─────────────────────────────────────────────────────────────────────────────


def mixing_enthalpy(steps: list[Step]) -> list[tuple[float, float, float]]:
    """Return ``[(x, H_eV, dH_meV/cell)]`` sorted by x.

    Requires both x = 0 and x = 1 endpoints to be complete.
    """
    done = [s for s in steps if s.status == DONE and s.parsed.get("enthalpy_eV")]
    try:
        h0 = next(
            float(s.parsed["enthalpy_eV"]) for s in done if abs(s.concentration) < 1e-6
        )
        h1 = next(
            float(s.parsed["enthalpy_eV"])
            for s in done
            if abs(s.concentration - 1) < 1e-6
        )
    except StopIteration:
        return []

    return sorted(
        [
            (
                s.concentration,
                float(s.parsed["enthalpy_eV"]),
                (
                    float(s.parsed["enthalpy_eV"])
                    - ((1 - s.concentration) * h0 + s.concentration * h1)
                )
                * 1000,
            )
            for s in done
        ],
        key=lambda t: t[0],
    )


# ─────────────────────────────────────────────────────────────────────────────
# CASTEP output analysis  (used by watchdog + retry)
# ─────────────────────────────────────────────────────────────────────────────


def _parse_smax_history(text: str) -> list[float]:
    """Return Smax (GPa) for every completed LBFGS iteration seen so far.

    Parses CASTEP convergence table lines such as::

        |   Smax    |   2.673547E+002 |   3.000000E-002 |  GPa | No  | <-- LBFGS

    Only the *value* column (index 2 after split on ``|``) is taken.

    Args:
        text: Contents of the ``.castep`` output file.

    Returns:
        List of Smax values in order of appearance.
    """
    smax: list[float] = []
    for ln in text.splitlines():
        if "Smax" in ln and "<-- LBFGS" in ln:
            parts = ln.split("|")
            if len(parts) >= 4:
                try:
                    smax.append(float(parts[2].strip()))
                except ValueError:
                    pass
    return smax


def _parse_max_scf_reached(text: str) -> bool:
    """Return ``True`` if CASTEP reported hitting the SCF cycle limit."""
    return "Reached maximum number of SCF cycles" in text


def _classify_failure(castep_path: Path) -> str:
    """Classify why a step failed by inspecting its ``.castep`` output.

    Returns one of the ``_KR_*`` tokens or ``"unknown"``.

    Args:
        castep_path: Path to the ``.castep`` output file (may not exist).
    """
    if not castep_path.exists():
        return "unknown"
    try:
        text = castep_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "unknown"

    if _parse_max_scf_reached(text):
        return _KR_SCF

    smax = _parse_smax_history(text)
    if smax:
        # High initial stress → Vegard mismatch or wrong geometry.
        if smax[0] > config.SMAX_KILL_GPa:
            return _KR_STRESS
        # Stall: last N values all above threshold.
        window = smax[-config.SMAX_STALL_ITERS :]
        if (
            len(window) >= config.SMAX_STALL_ITERS
            and min(window) > config.SMAX_KILL_GPa
        ):
            return _KR_SMAX

    return "unknown"


def patch_for_recovery(step_dir: Path, seed: str, error_type: str) -> bool:
    """Patch ``{seed}.param`` in *step_dir* with conservative recovery settings.

    Failure-mode → .param changes:

    * ``"scf_nosconv"`` / ``"smax_stall"`` / ``"timeout"`` —
      ``smearing_width : 0.20 eV``,  ``mix_charge_amp : 0.05``
      (wider smearing reduces charge sloshing; lower mixing damps oscillations).

    * ``"geom_high_stress"`` —
      ``geom_stress_tol : 0.10 GPa``
      (looser tolerance helps LBFGS escape a bad starting geometry).

    Existing values are replaced in-place; missing keys are appended.

    Args:
        step_dir:   Directory containing the .param file.
        seed:       CASTEP seed name.
        error_type: Classification token from :func:`_classify_failure`.

    Returns:
        ``True`` if the file was found and patched, ``False`` otherwise.
    """
    param = step_dir / f"{seed}.param"
    if not param.exists():
        return False

    scf_types = {_KR_SCF, _KR_SMAX, _KR_TIMEOUT}
    apply_scf = error_type in scf_types
    apply_stress = error_type == _KR_STRESS

    lines = param.read_text(encoding="utf-8").splitlines()
    patched: list[str] = []
    keys_done: set[str] = set()

    for ln in lines:
        kv = ln.split(":", 1)
        key = kv[0].strip().lower() if len(kv) == 2 else ""

        if apply_scf and key == "smearing_width":
            patched.append("smearing_width      : 0.20 eV")
            keys_done.add("smearing_width")
            continue
        if apply_scf and key == "mix_charge_amp":
            patched.append("mix_charge_amp      : 0.05")
            keys_done.add("mix_charge_amp")
            continue
        if apply_stress and key == "geom_stress_tol":
            patched.append("geom_stress_tol     : 0.10 GPa")
            keys_done.add("geom_stress_tol")
            continue
        patched.append(ln)

    # Append any keys not found in the original file.
    if apply_scf:
        if "smearing_width" not in keys_done:
            patched.append("smearing_width      : 0.20 eV")
        if "mix_charge_amp" not in keys_done:
            patched.append("mix_charge_amp      : 0.05")
    if apply_stress and "geom_stress_tol" not in keys_done:
        patched.append("geom_stress_tol     : 0.10 GPa")

    param.write_text("\n".join(patched) + "\n", encoding="utf-8")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Subprocess helpers
# ─────────────────────────────────────────────────────────────────────────────

_SKIP_FLAG: bool = False


def _arm_skip() -> None:
    global _SKIP_FLAG
    _SKIP_FLAG = False

    def _handler(sig: int, frame: object) -> None:
        global _SKIP_FLAG
        _SKIP_FLAG = True

    signal.signal(signal.SIGINT, _handler)


def _disarm_skip() -> None:
    signal.signal(signal.SIGINT, signal.SIG_DFL)


def _progress_monitor(output_file: Path, stop: threading.Event) -> None:
    """Background thread: one-line adaptive LBFGS + SCF progress bar.

    Outer: LBFGS steps (adaptive denominator = observed_max + 2, min 5).
    Inner: SCF cycle number within the current step.

    Output::

        │  [████████████░░░░░░░░░░]  60%  geo 3/3  scf 7  ⏱ 01:23
    """
    _BAR = 22
    last_sz = 0
    t0 = time.time()
    geo = 0
    geo_seen = 0
    scf = 0

    while not stop.is_set():
        try:
            sz = output_file.stat().st_size if output_file.exists() else 0
            if sz != last_sz:
                last_sz = sz
                text = output_file.read_text(errors="replace")
                for ln in text.splitlines():
                    if "LBFGS: finished iteration" in ln:
                        parts = ln.split()
                        for i, tok in enumerate(parts):
                            if tok == "iteration" and i + 1 < len(parts):
                                try:
                                    geo = int(parts[i + 1])
                                    geo_seen = max(geo_seen, geo)
                                except ValueError:
                                    pass
                    if "<-- SCF" in ln:
                        s = ln.lstrip()
                        if s and s[0].isdigit():
                            try:
                                scf = int(s.split()[0])
                            except ValueError:
                                pass
        except OSError:
            pass

        geo_denom = max(geo_seen + 2, 5)
        pct = min(100, geo * 100 // geo_denom)
        filled = pct * _BAR // 100
        bar = "█" * filled + "░" * (_BAR - filled)
        el = int(time.time() - t0)
        mm, ss = divmod(el, 60)
        geo_str = f"{geo}/{geo_seen}" if geo_seen >= 1 else f"{geo}/—"
        print(
            f"  │  [{bar}] {pct:3d}%  geo {geo_str}  scf {scf}"
            f"  ⏱ {mm:02d}:{ss:02d}    ",
            end="\r",
            flush=True,
        )
        stop.wait(0.8)

    print(" " * 80, end="\r")


class _Watchdog:
    """Kill CASTEP automatically when it stalls or exceeds the time limit.

    Three independent kill conditions (checked every ``_POLL_S`` seconds):

    1. Wall-clock elapsed > ``config.STEP_TIMEOUT_S``.
    2. Smax > ``config.SMAX_KILL_GPa`` for ``config.SMAX_STALL_ITERS``
       consecutive LBFGS steps without ever dropping below the threshold.
    3. ``"Reached maximum number of SCF cycles"`` in the output.

    Args:
        output_file: Path to the ``.castep`` file being written.
        proc:        Running :class:`subprocess.Popen` instance.
        stop:        Event set by the caller when the process finishes normally.
    """

    _POLL_S = 10.0

    def __init__(
        self,
        output_file: Path,
        proc: "subprocess.Popen[bytes]",
        stop: threading.Event,
    ) -> None:
        self._file = output_file
        self._proc = proc
        self._stop = stop
        self.reason = ""  # populated when a kill fires

    def run(self) -> None:
        """Main loop — runs in a daemon thread."""
        t_start = time.monotonic()
        while not self._stop.is_set():
            elapsed = time.monotonic() - t_start

            # 1. Hard timeout.
            if elapsed > config.STEP_TIMEOUT_S:
                self.reason = _KR_TIMEOUT
                self._kill(
                    f"step timed out after {int(elapsed / 60)}m"
                    f" (limit {int(config.STEP_TIMEOUT_S / 60)}m)"
                )
                return

            if self._file.exists():
                try:
                    text = self._file.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    self._stop.wait(self._POLL_S)
                    continue

                # 2. SCF non-convergence.
                if _parse_max_scf_reached(text):
                    self.reason = _KR_SCF
                    self._kill("max SCF cycles reached — charge sloshing likely")
                    return

                # 3. Smax stall.
                smax = _parse_smax_history(text)
                if len(smax) >= config.SMAX_STALL_ITERS:
                    window = smax[-config.SMAX_STALL_ITERS :]
                    if min(window) > config.SMAX_KILL_GPa:
                        self.reason = _KR_SMAX
                        self._kill(
                            f"Smax stalled > {config.SMAX_KILL_GPa} GPa for"
                            f" {config.SMAX_STALL_ITERS} steps"
                            f" (last: {window[-1]:.1f} GPa)"
                        )
                        return

            self._stop.wait(self._POLL_S)

    def _kill(self, msg: str) -> None:
        print(f"\n  │  ⚠  Watchdog: {msg}", flush=True)
        try:
            self._proc.terminate()
            try:
                self._proc.wait(10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        except OSError:
            pass


@dataclass
class ExecResult:
    """Outcome of one DFT step subprocess.

    Attributes:
        rc:          Return code, or ``None`` when killed externally.
        skipped:     ``True`` when the user pressed Ctrl+C.
        stderr_tail: Last 40 stderr lines (for failure diagnostics).
        kill_reason: Non-empty when the watchdog terminated the process.
    """

    rc: int | None
    skipped: bool
    stderr_tail: list[str]
    kill_reason: str = ""


def run_process(cmd: str, cwd: Path, output_file: Path) -> ExecResult:
    """Run *cmd* in *cwd* with progress bar, watchdog, and graceful Ctrl+C.

    Args:
        cmd:         Shell command (``{seed}`` already substituted).
        cwd:         Subprocess working directory.
        output_file: ``.castep`` file path for monitoring.

    Returns:
        :class:`ExecResult` with all outcome fields populated.
    """
    stop = threading.Event()
    monitor = threading.Thread(
        target=_progress_monitor, args=(output_file, stop), daemon=True
    )
    monitor.start()

    proc: "subprocess.Popen[bytes] | None" = None
    stderr: list[str] = []
    rc = -1
    watchdog: _Watchdog | None = None

    _arm_skip()
    try:
        proc = subprocess.Popen(
            cmd,
            shell=True,
            cwd=cwd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        watchdog = _Watchdog(output_file, proc, stop)
        wd_thread = threading.Thread(target=watchdog.run, daemon=True)
        wd_thread.start()

        def _drain() -> None:
            for raw in proc.stderr:  # type: ignore[union-attr]
                line = raw.decode(errors="replace").rstrip()
                stderr.append(line)
                if len(stderr) > 40:
                    stderr.pop(0)

        drain = threading.Thread(target=_drain, daemon=True)
        drain.start()

        while proc.poll() is None:
            if _SKIP_FLAG:
                stop.set()
                monitor.join(2)
                proc.terminate()
                try:
                    proc.wait(15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                _disarm_skip()
                return ExecResult(
                    rc=None, skipped=True, stderr_tail=[], kill_reason=_KR_CTRL_C
                )
            time.sleep(0.5)

        drain.join(3)
        rc = proc.returncode
    except OSError as e:
        stderr.append(str(e))
    finally:
        stop.set()
        monitor.join(2)
        _disarm_skip()
        if proc and proc.poll() is None:
            proc.kill()

    kill_reason = watchdog.reason if watchdog else ""
    return ExecResult(rc=rc, skipped=False, stderr_tail=stderr, kill_reason=kill_reason)


def cmd_is_valid(cmd: str) -> bool:
    binary = os.path.expanduser(cmd.split()[0])
    if Path(binary).is_absolute() or "/" in binary:
        return Path(binary).is_file() and os.access(binary, os.X_OK)
    return shutil.which(binary) is not None


# ─────────────────────────────────────────────────────────────────────────────
# Step execution  (engine-agnostic)
# ─────────────────────────────────────────────────────────────────────────────


def execute_step(
    state: RunState,
    step: Step,
    cell_src: Path,
    param_src: Path,
    write_cell_fn: Callable[[Path, float], None],
    output_suffix: str = ".castep",
    keep_all: bool = False,
) -> ExecResult:
    """Prepare inputs, run the engine, parse outputs, persist state.

    When the watchdog kills the process the kill reason is written into
    ``step.parsed["kill_reason"]`` so the smart-retry logic in ``main.py``
    can call :func:`patch_for_recovery` with the correct error type.
    """
    x = step.concentration
    seed = state.seed
    castep_base = state.proj_dir / config.CASTEP_SUBDIR
    castep_base.mkdir(parents=True, exist_ok=True)
    step_dir = castep_base / f"x{x:.4f}"
    step_dir.mkdir(parents=True, exist_ok=True)
    step.step_dir = f"{config.CASTEP_SUBDIR}/x{x:.4f}"

    for src, name in [
        (cell_src, f"original_{seed}.cell"),
        (param_src, f"original_{seed}.param"),
    ]:
        dst = state.proj_dir / name
        if not dst.exists() and src.exists():
            shutil.copy2(src, dst)

    write_cell_fn(step_dir / f"{seed}.cell", x)
    shutil.copy2(param_src, step_dir / f"{seed}.param")

    from castep.castep import nextra_for_step, patch_nextra

    nextra = (
        config.ELASTIC_NEXTRA_PURE
        if state.single_mode
        else nextra_for_step(state.species_a, state.species_b, x)
    )
    patch_nextra(step_dir / f"{seed}.param", nextra)
    step.parsed["nextra_bands_used"] = nextra

    step.status = RUNNING
    step.started_at = _now()
    save_run(state)
    write_csv(state)

    if not state.castep_cmd:
        step.status = DONE
        step.rc = "N/A"
        step.finished_at = _now()
        save_run(state)
        write_csv(state)
        return ExecResult(rc=0, skipped=False, stderr_tail=[])

    cmd = os.path.expanduser(state.castep_cmd.replace("{seed}", seed))
    result = run_process(cmd, step_dir, step_dir / f"{seed}{output_suffix}")
    step.finished_at = _now()

    if result.skipped:
        step.status = SKIPPED
        step.rc = "ctrl-c"
    else:
        step.rc = str(result.rc) if result.rc is not None else "unknown"

        from castep.castep import parse_elastic_file, parse_output

        parsed = parse_output(step_dir / f"{seed}{output_suffix}")
        step.parsed.update(parsed.to_dict())

        elastic_path = step_dir / f"{seed}.elastic"
        if elastic_path.exists():
            step.parsed.update(parse_elastic_file(elastic_path))

        # Store kill/failure reason for smart retry.
        if result.kill_reason:
            step.parsed["kill_reason"] = result.kill_reason
        elif result.rc != 0:
            step.parsed["kill_reason"] = _classify_failure(
                step_dir / f"{seed}{output_suffix}"
            )

        step.status = DONE if result.rc == 0 else FAILED
        if result.rc == 0 and not keep_all:
            _cleanup(step_dir)

    save_run(state)
    write_csv(state)
    return result


def _cleanup(step_dir: Path) -> None:
    for pat in (
        "*.castep_bin",
        "*.check",
        "*.cst_esp",
        "*.bands",
        "*.bib",
        "*.usp",
        "*.upf",
        "*.oepr",
    ):
        for f in step_dir.glob(pat):
            try:
                f.unlink()
            except OSError:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")
