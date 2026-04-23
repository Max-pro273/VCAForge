"""
orchestrator.py  —  Run state, persistence, CSV, subprocess execution.
════════════════════════════════════════════════════════════════════════
RunState / Step  — data model + crash-safe JSON persistence
execute_step     — engine-agnostic: runs one DFT job, parses results
write_csv        — live CSV export
mixing_enthalpy  — ΔH_mix from completed steps

Watchdog
────────
``run_process`` embeds a :class:`_Watchdog` that monitors the engine output
while the subprocess runs and kills it automatically when:

  1. Wall-clock elapsed > ``config.STEP_TIMEOUT_S`` (default 1800 s).
  2. Smax oscillates above ``config.SMAX_KILL_GPa`` (default 50 GPa) for
     ``config.SMAX_STALL_ITERS`` consecutive LBFGS steps.
  3. ``"Reached maximum number of SCF cycles"`` appears in the output.

Engine contract
───────────────
All engine-specific logic is in the engine class.  Orchestrator detects
capabilities via ``hasattr`` (duck-typing) — no ``if engine.name ==`` anywhere.

Elastic routing:
  hasattr(engine, "run_internal_elastic")
      → True:  engine handles it entirely (e.g. VASP IBRION=6)
      → False: orchestrator runs finite-strain loop using
               engine.write_singlepoint_input / engine.parse_stress_tensor

Progress monitoring:
  engine.progress_monitor(output_ref, stop_event) called in background thread.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field, asdict  # <-- Add asdict here
from pathlib import Path
from typing import Any

import config
from core_physics import Crystal  # canonical location — do not redefine here

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

_KR_TIMEOUT = "timeout"
_KR_SMAX    = "smax_stall"
_KR_SCF     = "scf_nosconv"
_KR_CTRL_C  = "ctrl-c"
_KR_STRESS  = "geom_high_stress"


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
            "idx", "step", "concentration", "status",
            "step_dir", "started_at", "finished_at", "rc", "parsed",
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
    engine_cmd: str          # renamed from castep_cmd — engine-agnostic
    c_start: float
    c_end: float
    n_steps: int
    created_at: str
    single_mode: bool = False
    nonmetal: str = ""
    nonmetal_occ: float = 1.0
    run_elastic: bool = False
    engine_kwargs: dict = field(default_factory=dict)
    steps: list[Step] = field(default_factory=list)

    # Legacy alias so old JSON state files still load
    @property
    def castep_cmd(self) -> str:
        return self.engine_cmd

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
        d["engine_cmd"] = self.engine_cmd
        d["castep_cmd"] = self.engine_cmd   # keep for backward compat
        d["engine_kwargs"] = self.engine_kwargs
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
        # Accept both old "castep_cmd" and new "engine_cmd"
        cmd = d.get("engine_cmd") or d.get("castep_cmd", "")
        return cls(
            version=d.get("version", "?"),
            seed=d["seed"],
            proj_dir=proj_dir,
            template_element=tmpl,
            species=raw_species,
            engine_cmd=cmd,
            c_start=d.get("c_start", 0.0),
            c_end=d.get("c_end", 1.0),
            n_steps=d.get("n_steps", 0),
            created_at=d.get("created_at", ""),
            single_mode=d.get("single_mode", False),
            nonmetal=d.get("nonmetal", ""),
            nonmetal_occ=d.get("nonmetal_occ", 1.0),
            run_elastic=d.get("run_elastic", False),
            engine_kwargs=d.get("engine_kwargs", {}),
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
    engine_cmd: str,
    c_start: float,
    c_end: float,
    n_steps: int,
    single_mode: bool = False,
    nonmetal: str = "",
    nonmetal_occ: float = 1.0,
    run_elastic: bool = False,
    engine_kwargs: dict | None = None,
    # legacy kwarg alias
    castep_cmd: str = "",
) -> RunState:
    proj_dir.mkdir(parents=True, exist_ok=True)
    cmd = engine_cmd or castep_cmd
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
        engine_cmd=cmd,
        c_start=c_start,
        c_end=c_end,
        n_steps=n_steps,
        created_at=_now(),
        single_mode=single_mode,
        nonmetal=nonmetal,
        nonmetal_occ=nonmetal_occ,
        run_elastic=run_elastic,
        engine_kwargs=engine_kwargs or {},
        steps=steps,
    )
    save_run(state)
    return state


# ─────────────────────────────────────────────────────────────────────────────
# CSV export
# ─────────────────────────────────────────────────────────────────────────────

_CSV_FIXED = [
    "step", "concentration", "status",
    "started_at", "finished_at", "wall_time_s",
    "elastic_wall_time_s", "total_wall_time_s",
]

_CSV_ORDER = [
    # ── Structure ──────────────────────────────────────────────────────────────
    "concentration", "VEC",
    "a_opt_ang", "b_opt_ang", "c_opt_ang", "a_prim_ang",
    "alpha", "beta", "gamma",
    "volume_ang3", "density_gcm3",
    # ── Energetics ─────────────────────────────────────────────────────────────
    "energy_ev", "free_energy_ev", "energy_0k_ev",
    "enthalpy_eV", "dH_mix_meV_per_fu",
    # ── Electronic ─────────────────────────────────────────────────────────────
    "fermi_ev", "mag_moment",
    "charge_spilling_pct",
    "mulliken_q_C", "mulliken_q_N", "mulliken_q_O",       # nonmetals first
    "mulliken_q_Ti", "mulliken_q_Zr", "mulliken_q_Nb",    # metals — extend as needed
    "mulliken_q_V", "mulliken_q_Hf", "mulliken_q_Mo",
    "bond_population_avg", "bond_length_avg_ang",
    # ── Mechanical (elastic) ───────────────────────────────────────────────────
    "B_lbfgs_GPa",
    "C11", "C12", "C44",
    "B_Voigt_GPa", "B_Reuss_GPa", "B_Hill_GPa",
    "G_Voigt_GPa", "G_Reuss_GPa", "G_Hill_GPa",
    "E_GPa", "nu", "Zener_A", "Pugh_ratio",
    "Cauchy_pressure_GPa", "C_prime_GPa",
    "Kleinman_zeta", "lambda_Lame_GPa", "mu_Lame_GPa",
    "H_Vickers_GPa",
    # ── Acoustic / Thermal ────────────────────────────────────────────────────
    "v_longitudinal_ms", "v_transverse_ms", "v_mean_ms",
    "T_Debye_K", "acoustic_Gruneisen",
    # ── Convergence / QC ──────────────────────────────────────────────────────
    "residual_pressure_GPa", "fmax_ev_ang",
    "geom_converged", "nextra_bands_used", "kill_reason", "warnings",
    # ── Elastic metadata ──────────────────────────────────────────────────────
    "elastic_source", "elastic_n_points", "elastic_R2_min", "elastic_quality_note",
    "elastic_wall_time_s",
    # ── Run metadata ──────────────────────────────────────────────────────────
    "peak_mem_mb", "step_dir", "rc",
]


def write_csv(state: RunState) -> Path:
    dh_data = mixing_enthalpy(state.steps)
    if dh_data:
        dh_map = {x: dh for x, _, dh in dh_data}
        for s in state.steps:
            if s.status == DONE and s.concentration in dh_map:
                s.parsed.setdefault("dH_mix_meV_per_fu", f"{dh_map[s.concentration]:.3f}")

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
        w = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore", restval="N/A")
        w.writeheader()
        w.writerows(s.to_dict() for s in state.steps)
    return out

_VEGARD_KEYS = ("C11", "C12", "C44", "B_Hill_GPa", "G_Hill_GPa", "E_GPa", "nu", "Zener_A", "Pugh_ratio")
def vegard_interpolate(x: float, d0: dict[str, Any], d1: dict[str, Any]) -> dict[str, Any]:
    if not d0 or not d1: return {}
    r = {}
    for k in _VEGARD_KEYS:
        if k in d0 and k in d1:
            try: r[k] = f"{(1 - x) * float(d0[k]) + x * float(d1[k]):.4f}"
            except ValueError: pass
    if r: r.update({"elastic_source": "Vegard_interpolation", "elastic_n_points": "0"})
    return r

# ─────────────────────────────────────────────────────────────────────────────
# ΔH_mix
# ─────────────────────────────────────────────────────────────────────────────

def mixing_enthalpy(steps: list[Step]) -> list[tuple[float, float, float]]:
    """Return ``[(x, H_eV, dH_meV/cell)]`` sorted by x."""

    def _h(s: Step) -> float | None:
        try:
            return float(s.parsed["enthalpy_eV"])
        except (KeyError, TypeError, ValueError):
            return None

    done = [s for s in steps if s.status == DONE and _h(s) is not None]
    try:
        h0 = next(_h(s) for s in done if abs(s.concentration) < 1e-4)
        h1 = next(_h(s) for s in done if abs(s.concentration - 1) < 1e-4)
    except StopIteration:
        return []

    result = []
    for s in done:
        h = _h(s)
        if h is None:
            continue
        dh = (h - ((1 - s.concentration) * h0 + s.concentration * h1)) * 1000
        result.append((s.concentration, h, dh))
    return sorted(result, key=lambda t: t[0])


# ─────────────────────────────────────────────────────────────────────────────
# Failure analysis helpers (engine-agnostic watchdog patterns)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_smax_history(text: str) -> list[float]:
    """Parse Smax (GPa) history from CASTEP LBFGS convergence lines."""
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

def patch_for_recovery(step_dir: Path, seed: str, error_type: str) -> bool:
    """Patch ``{seed}.param`` in *step_dir* with conservative recovery settings."""
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


class _Watchdog:
    """Kill engine automatically when it stalls or exceeds the time limit."""

    _POLL_S = 10.0

    def __init__(self, output_file: Path, proc: "subprocess.Popen[bytes]", stop: threading.Event) -> None:
        self._file = output_file
        self._proc = proc
        self._stop = stop
        self.reason = ""

    def run(self) -> None:
        t_start = time.monotonic()
        while not self._stop.is_set():
            elapsed = time.monotonic() - t_start

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

                if "Reached maximum number of SCF cycles" in text:
                    self.reason = _KR_SCF
                    self._kill("max SCF cycles reached — charge sloshing likely")
                    return

                smax = _parse_smax_history(text)
                if len(smax) >= config.SMAX_STALL_ITERS:
                    window = smax[-config.SMAX_STALL_ITERS:]
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
    rc: int | None
    skipped: bool
    stderr_tail: list[str]
    kill_reason: str = ""


def run_process(cmd: str, cwd: Path, output_file: Path, *, engine: Engine) -> ExecResult:
    """Run cmd in cwd with real-time stdout progress monitoring and watchdog."""
    stop = threading.Event()
    proc: "subprocess.Popen[bytes] | None" = None
    stderr_tail: list[str] = []
    rc = -1
    watchdog: _Watchdog | None = None

    _arm_skip()
    try:
        # Запускаємо процес, перехоплюємо stdout та stderr
        proc = subprocess.Popen(
            cmd,
            shell=True,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        proc._cwd = cwd  # consumed by file-polling progress monitors (e.g. CASTEP)

        # Запускаємо прогрес-бар РУШІЯ, передаючи йому потік процесу
        monitor = threading.Thread(
            target=engine.progress_monitor, args=(proc, stop), daemon=True
        )
        monitor.start()

        # Watchdog для відстеження зависань
        watchdog = _Watchdog(output_file, proc, stop)
        wd_thread = threading.Thread(target=watchdog.run, daemon=True)
        wd_thread.start()

        def _drain_stderr() -> None:
            if proc.stderr:
                for raw in proc.stderr:
                    line = raw.decode(errors="replace").rstrip()
                    if line:
                        stderr_tail.append(line)
                        if len(stderr_tail) > 40:
                            stderr_tail.pop(0)

        drain = threading.Thread(target=_drain_stderr, daemon=True)
        drain.start()

        while proc.poll() is None:
            if _SKIP_FLAG:
                stop.set()
                monitor.join(2)
                proc.terminate()
                try:
                    proc.wait(10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                _disarm_skip()
                return ExecResult(rc=None, skipped=True, stderr_tail=[], kill_reason=_KR_CTRL_C)
            time.sleep(0.2)

        drain.join(2)
        rc = proc.returncode
    except OSError as e:
        stderr_tail.append(str(e))
    finally:
        stop.set()
        if 'monitor' in locals(): monitor.join(2)
        _disarm_skip()
        if proc and proc.poll() is None:
            proc.kill()

    kill_reason = watchdog.reason if watchdog else ""
    return ExecResult(rc=rc, skipped=False, stderr_tail=stderr_tail, kill_reason=kill_reason)



# ─────────────────────────────────────────────────────────────────────────────
# Step execution  (engine-agnostic)
# ─────────────────────────────────────────────────────────────────────────────

def execute_step(
    state: RunState,
    step: Step,
    crystal: Crystal,
    engine,
    keep_all: bool = False,
) -> ExecResult:
    """Prepare inputs, run the engine, parse outputs, persist state."""
    x = step.concentration
    seed = state.seed

    # Engine declares its own subdirectory name (e.g. CastepEngine.subdir_name = "CASTEP")
    subdir_name = getattr(engine, "subdir_name", engine.name.upper())

    base_dir = state.proj_dir / subdir_name
    base_dir.mkdir(parents=True, exist_ok=True)
    step_dir = base_dir / f"x{x:.4f}"
    step_dir.mkdir(parents=True, exist_ok=True)
    step.step_dir = f"{subdir_name}/x{x:.4f}"

    engine.write_input(step_dir, seed, crystal, state.species, x)

    step.status = RUNNING
    step.started_at = _now()
    save_run(state)
    write_csv(state)

    if not state.engine_cmd:
        step.status = DONE
        step.rc = "N/A"
        step.finished_at = _now()
        save_run(state)
        write_csv(state)
        return ExecResult(rc=0, skipped=False, stderr_tail=[])

    output_file = (
        step_dir / f"{seed}{engine.output_suffix}"
        if engine.output_suffix.startswith(".")
        else step_dir / engine.output_suffix
    )
    cmd = os.path.expanduser(state.engine_cmd.replace("{seed}", seed))

    result = run_process(cmd, step_dir, output_file, engine=engine)
    step.finished_at = _now()

    if result.skipped:
        step.status = SKIPPED
        step.rc = "ctrl-c"
    else:
        step.rc = str(result.rc) if result.rc is not None else "unknown"

        parsed = engine.parse_output(output_file)
        result_dict = asdict(parsed)
        extra = result_dict.pop("extra_data", {})

        clean_dict = {k: v for k, v in result_dict.items() if v is not None}
        clean_dict.update(extra)  # Flatten custom engine metrics into the root

        # Normalise run_time_s → wall_time_s (Step.wall_time_s property reads this key)
        if "run_time_s" in clean_dict and "wall_time_s" not in clean_dict:
            clean_dict["wall_time_s"] = clean_dict["run_time_s"]

        step.parsed.update(clean_dict)

        # Engine-specific post-parse hook (e.g. CASTEP .elastic file)
        if hasattr(engine, "parse_extra_outputs"):
            extra = engine.parse_extra_outputs(step_dir, seed)
            if extra:
                step.parsed.update(extra)

        if result.kill_reason:
            step.parsed["kill_reason"] = result.kill_reason

        step.status = DONE if result.rc == 0 else FAILED

        if result.rc == 0 and not keep_all:
            engine.cleanup(step_dir)

    save_run(state)
    write_csv(state)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Elastic workflow — capability-based routing
# ─────────────────────────────────────────────────────────────────────────────

def run_elastic_for_step(
    state: RunState,
    step: Step,
    engine,
) -> dict[str, str]:
    """Route elastic calculation based on engine capabilities.

    - If engine has ``run_internal_elastic``: delegate entirely.
    - Otherwise: finite-strain loop driven by orchestrator.
    """
    step_dir = state.proj_dir / step.step_dir
    seed = state.seed
    x = step.concentration
    density = float(step.parsed.get("density_gcm3") or 0) or None
    volume = float(step.parsed.get("volume_ang3") or 0) or None

    if hasattr(engine, "run_internal_elastic"):
        return engine.run_internal_elastic(
            step_dir, seed, x, state.species,
            state.nonmetal or None, density, volume,
        )

    # ── Finite-strain fallback ────────────────────────────────────────────────
    return _finite_strain_elastic(state, step, engine, step_dir, seed, x, density, volume)


def _finite_strain_elastic(
    state: RunState,
    step: Step,
    engine,
    step_dir: Path,
    seed: str,
    x: float,
    density_gcm3: float | None,
    volume_ang3: float | None,
) -> dict[str, str]:
    """Orchestrator-driven finite-strain elastic loop.

    generate_strain_steps(crystal) reads crystal.strain_pattern_code and
    crystal.lattice_type internally — no external lattice-code mapping needed.

    Requires engine to implement:
        load_optimised_crystal(step_dir, seed) -> Crystal
        write_singlepoint_input(dest_dir, crystal, seed, species_mix, x, strain_voigt)
        parse_stress_tensor(output_file) -> np.ndarray  (Voigt 6-vector, GPa)
    """
    import numpy as np
    from core_physics import generate_strain_steps, fit_cij_cubic

    if not (
        hasattr(engine, "write_singlepoint_input")
        and hasattr(engine, "parse_stress_tensor")
        and hasattr(engine, "load_optimised_crystal")
    ):
        return {
            "_elastic_error": (
                "Engine does not support finite-strain fallback — "
                "missing write_singlepoint_input / parse_stress_tensor / load_optimised_crystal"
            )
        }

    t0 = time.monotonic()

    try:
        opt_crystal = engine.load_optimised_crystal(step_dir, seed)
    except (FileNotFoundError, ValueError) as exc:
        return {"_elastic_error": f"load_optimised_crystal failed: {exc}"}

    # generate_strain_steps uses crystal.strain_pattern_code (from spglib) and
    # handles cubic → conventional cell conversion internally.
    strain_steps = generate_strain_steps(
        opt_crystal,
        max_strain=config.ELASTIC_MAX_STRAIN,
        n_steps=config.ELASTIC_N_STEPS,
    )

    stresses: list[np.ndarray] = []
    strains: list[np.ndarray] = []
    step_errors: list[str] = []

    for ss in strain_steps:
        sub_seed = f"{seed}{ss.name}"

        try:
            engine.write_singlepoint_input(
                step_dir, opt_crystal, sub_seed, state.species, x, ss.strain_voigt
            )
        except (FileNotFoundError, ValueError, OSError) as exc:
            step_errors.append(f"{ss.name}: write failed: {exc}")
            continue

        output_file = (
            step_dir / f"{sub_seed}{engine.output_suffix}"
            if engine.output_suffix.startswith(".")
            else step_dir / engine.output_suffix
        )
        cmd = os.path.expanduser(state.engine_cmd.replace("{seed}", sub_seed))
        run_process(cmd, step_dir, output_file, engine=engine)

        try:
            sv = engine.parse_stress_tensor(output_file)
        except (FileNotFoundError, ValueError) as exc:
            step_errors.append(f"{ss.name}: parse failed: {exc}")
            continue

        stresses.append(sv)
        strains.append(ss.strain_voigt)

    if len(stresses) < 3:
        error_detail = "; ".join(step_errors) if step_errors else "no step errors recorded"
        return {
            "_elastic_error": (
                f"Not enough stress tensors ({len(stresses)}/{len(strain_steps)}). "
                f"Step errors: {error_detail}"
            )
        }

    result = fit_cij_cubic(
        stresses, strains,
        density_gcm3=density_gcm3,
        n_atoms=opt_crystal.num_atoms,
        volume_ang3=volume_ang3,
    )
    result["elastic_wall_time_s"] = f"{time.monotonic() - t0:.0f}"
    result["elastic_source"] = f"{engine.name.upper()}-FiniteStrain-{opt_crystal.lattice_type}"
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")
