"""
orchestrator.py  —  Run state, persistence, CSV, subprocess execution.
════════════════════════════════════════════════════════════════════════
No user prompts, no physics.  Pure orchestration:

  RunState / Step  — data model + crash-safe JSON persistence
  execute_step     — engine-agnostic: runs one DFT job, parses results
  write_csv        — live CSV export
  mixing_enthalpy  — ΔH_mix from completed steps

Engine contract
───────────────
``execute_step`` calls ``write_cell_fn(dest, x)`` and delegates output
parsing to the ``Engine`` protocol from ``engine.py``.  Adding VASP or
QE requires implementing that protocol only — no changes here.
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
DONE    = "done"
SKIPPED = "skipped"
FAILED  = "failed"

STATUS_ICON: dict[str, str] = {
    DONE: "✓", SKIPPED: "⊘", FAILED: "✗", PENDING: "·", RUNNING: "▶",
}


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Step:
    """One concentration point in a VCA sweep, or a single-compound run."""

    idx:           int
    concentration: float
    status:        str            = PENDING
    step_dir:      str            = ""
    started_at:    str            = ""
    finished_at:   str            = ""
    rc:            str            = ""
    parsed:        dict[str, Any] = field(default_factory=dict)

    # ── read-only convenience accessors ──────────────────────────────────────

    @property
    def enthalpy_eV(self)    -> str: return str(self.parsed.get("enthalpy_eV",    ""))
    @property
    def a_opt_ang(self)      -> str: return str(self.parsed.get("a_opt_ang",      ""))
    @property
    def wall_time_s(self)    -> str: return str(self.parsed.get("wall_time_s",    ""))
    @property
    def geom_converged(self) -> str: return str(self.parsed.get("geom_converged", ""))
    @property
    def warnings(self)       -> str: return str(self.parsed.get("warnings",       ""))

    def to_dict(self) -> dict[str, Any]:
        base = {
            "step":          self.idx,
            "concentration": self.concentration,
            "status":        self.status,
            "step_dir":      self.step_dir,
            "started_at":    self.started_at,
            "finished_at":   self.finished_at,
            "rc":            self.rc,
        }
        base.update(self.parsed)
        return base

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Step":
        known = {
            "idx", "step", "concentration", "status", "step_dir",
            "started_at", "finished_at", "rc", "parsed",
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
    """Complete, serialisable state of one VCAForge run.

    Attributes:
        template_element: Element in the .cell template that is substituted
                          (e.g. ``"Ti"`` in TiC).  Persisted so resumed
                          runs never need to re-read the .cell file.
        species:          ``[(element, end_fraction), ...]``.  The first
                          element is present at x = 0; subsequent elements
                          appear at x = 1 with the given fractions
                          (which must sum to 1).
        single_mode:      ``True`` for a single compound (no sweep).
        nonmetal:         Anion element detected in the template, e.g. ``"C"``.
    """

    version:          str
    seed:             str
    proj_dir:         Path
    template_element: str                     # which sublattice to replace
    species:          list[tuple[str, float]] # [(elem, end_frac), ...]
    castep_cmd:       str
    c_start:          float
    c_end:            float
    n_steps:          int
    created_at:       str
    single_mode:      bool  = False
    nonmetal:         str   = ""
    nonmetal_occ:     float = 1.0
    run_elastic:      bool  = False
    steps:            list[Step] = field(default_factory=list)

    # ── convenience ──────────────────────────────────────────────────────────

    @property
    def species_a(self) -> str:
        return self.species[0][0] if self.species else ""

    @property
    def species_b(self) -> str:
        return self.species[1][0] if len(self.species) > 1 else ""

    @property
    def n_done(self)    -> int: return sum(1 for s in self.steps if s.status == DONE)
    @property
    def n_pending(self) -> int: return sum(1 for s in self.steps if s.status == PENDING)
    @property
    def n_failed(self)  -> int: return sum(1 for s in self.steps if s.status == FAILED)

    def system_label(self) -> str:
        """Human-readable chemical formula, e.g. ``Ti(1-x)Zr(x)C``.

        Works for any number of mixing components::

            single          → "TiC"          (seed name)
            binary VCA      → "Ti(1-x)Zr(x)C"
            ternary VCA     → "Ti(1-x)[Nb0.70V0.30](x)C"
            4-component VCA → "Ti(1-x)[Nb0.50V0.30W0.20](x)C"
        """
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

    # ── serialisation ─────────────────────────────────────────────────────────

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["proj_dir"] = str(self.proj_dir)
        d["steps"]    = [s.to_dict() for s in self.steps]
        return d

    @classmethod
    def from_json(cls, d: dict[str, Any], proj_dir: Path) -> "RunState":
        # Back-compat: very old format stored species_a/species_b as strings.
        raw_species = d.get("species")
        if raw_species is None:
            sa = d.get("species_a", "")
            sb = d.get("species_b", "")
            raw_species = [(sa, 0.0), (sb, 1.0)] if sa and sb else [(sa, 0.0)]
        # Back-compat: template_element was not stored in older state files.
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
    """Atomically persist *state* to JSON (write-then-rename pattern)."""
    dst = _state_path(state.proj_dir)
    tmp = dst.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(state.to_json(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp.replace(dst)


def load_run(proj_dir: Path) -> RunState | None:
    """Return a :class:`RunState` from *proj_dir*, or ``None`` if absent/corrupt."""
    f = _state_path(proj_dir)
    if not f.exists():
        return None
    try:
        return RunState.from_json(json.loads(f.read_text(encoding="utf-8")), proj_dir)
    except (json.JSONDecodeError, KeyError):
        return None


def new_run(
    *,
    seed:             str,
    proj_dir:         Path,
    template_element: str,
    species:          list[tuple[str, float]],
    castep_cmd:       str,
    c_start:          float,
    c_end:            float,
    n_steps:          int,
    single_mode:      bool  = False,
    nonmetal:         str   = "",
    nonmetal_occ:     float = 1.0,
    run_elastic:      bool  = False,
) -> RunState:
    """Create a :class:`RunState`, generate concentration steps, persist, return."""
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
    "step", "concentration", "status",
    "started_at", "finished_at",
    "wall_time_s", "elastic_wall_time_s", "total_wall_time_s",
]

_CSV_ORDER = [
    "concentration", "VEC",
    "a_opt_ang", "b_opt_ang", "c_opt_ang", "volume_ang3", "density_gcm3",
    "enthalpy_eV", "dH_mix_meV_per_fu",
    "C11", "C12", "C44",
    "B_Voigt_GPa", "B_Reuss_GPa", "B_Hill_GPa",
    "G_Voigt_GPa", "G_Reuss_GPa", "G_Hill_GPa",
    "E_GPa", "nu", "Zener_A", "Pugh_ratio", "Cauchy_pressure_GPa",
    "C_prime_GPa", "Kleinman_zeta", "lambda_Lame_GPa", "mu_Lame_GPa",
    "H_Vickers_GPa", "v_longitudinal_ms", "v_transverse_ms", "v_mean_ms",
    "T_Debye_K", "acoustic_Gruneisen",
    "elastic_source", "elastic_n_points", "elastic_R2_min", "elastic_quality_note",
    "geom_converged", "nextra_bands_used", "warnings", "step_dir", "rc",
]


def write_csv(state: RunState) -> Path:
    """Write/overwrite the live results CSV inside *state.proj_dir*."""
    # Inject ΔH_mix where both endpoints are available.
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
            float(s.parsed["enthalpy_eV"])
            for s in done if abs(s.concentration) < 1e-6
        )
        h1 = next(
            float(s.parsed["enthalpy_eV"])
            for s in done if abs(s.concentration - 1) < 1e-6
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
                ) * 1000,
            )
            for s in done
        ],
        key=lambda t: t[0],
    )


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
    """Background thread: display a one-line adaptive progress bar.

    Tracks two nested loops from the CASTEP output:

    * **Outer** — LBFGS geometry steps.  The bar denominator adapts to the
      observed maximum rather than the hard ``geom_max_iter`` limit, so the
      bar is visually meaningful from the first step even for fast-converging
      systems (typical: 3-20 steps vs the 150 hard limit).

    * **Inner** — SCF cycles within the current LBFGS step.  Displayed as a
      raw counter alongside the bar.

    Output format (single overwriting line, never scrolls)::

        │  [████████████░░░░░░░░░░]  geo 4/6  scf 7  ⏱ 01:23
    """
    _BAR       = 22
    last_size  = 0
    t0         = time.time()
    geo        = 0    # last completed LBFGS iteration number
    geo_seen   = 0    # running maximum (adaptive denominator)
    scf        = 0    # SCF cycle number within the current step

    while not stop.is_set():
        try:
            sz = output_file.stat().st_size if output_file.exists() else 0
            if sz != last_size:
                last_size = sz
                text = output_file.read_text(errors="replace")

                # Completed LBFGS steps: "LBFGS: finished iteration N …"
                # Parsed with plain string ops — no regex needed.
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

                # Current SCF cycle: lines ending with "<-- SCF" whose
                # first non-space token is an integer cycle number.
                for ln in text.splitlines():
                    if "<-- SCF" in ln:
                        s = ln.lstrip()
                        if s and s[0].isdigit():
                            tok = s.split()[0]
                            try:
                                scf = int(tok)
                            except ValueError:
                                pass
        except OSError:
            pass

        # Adaptive denominator: at least 5, grows 2 ahead of observations.
        geo_denom = max(geo_seen + 2, 5)
        pct       = min(100, geo * 100 // geo_denom)
        filled    = pct * _BAR // 100
        bar       = "█" * filled + "░" * (_BAR - filled)
        el        = int(time.time() - t0)
        mm, ss    = divmod(el, 60)
        geo_str   = f"{geo}/{geo_seen}" if geo_seen >= 1 else f"{geo}/—"
        line = (
            f"  │  [{bar}] {pct:3d}%  geo {geo_str}  scf {scf}"
            f"  ⏱ {mm:02d}:{ss:02d}"
        )
        print(line + "    ", end="\r", flush=True)
        stop.wait(0.8)

    print(" " * 80, end="\r")  # Clear the line when the process finishes.


@dataclass
class ExecResult:
    rc:          int | None
    skipped:     bool
    stderr_tail: list[str]


def run_process(cmd: str, cwd: Path, output_file: Path) -> ExecResult:
    """Run *cmd* in *cwd*, monitoring *output_file*; Ctrl+C = graceful skip."""
    stop   = threading.Event()
    thread = threading.Thread(
        target=_progress_monitor, args=(output_file, stop), daemon=True
    )
    thread.start()

    proc:   subprocess.Popen | None = None
    stderr: list[str] = []
    rc = -1

    _arm_skip()
    try:
        proc = subprocess.Popen(
            cmd, shell=True, cwd=cwd,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )

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
                thread.join(2)
                proc.terminate()
                try:
                    proc.wait(15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                _disarm_skip()
                return ExecResult(rc=None, skipped=True, stderr_tail=[])
            time.sleep(0.5)

        drain.join(3)
        rc = proc.returncode
    except OSError as e:
        stderr.append(str(e))
    finally:
        stop.set()
        thread.join(2)
        _disarm_skip()
        if proc and proc.poll() is None:
            proc.kill()

    return ExecResult(rc=rc, skipped=False, stderr_tail=stderr)


def cmd_is_valid(cmd: str) -> bool:
    """Return ``True`` if the binary in *cmd* is executable."""
    binary = os.path.expanduser(cmd.split()[0])
    if Path(binary).is_absolute() or "/" in binary:
        return Path(binary).is_file() and os.access(binary, os.X_OK)
    return shutil.which(binary) is not None


# ─────────────────────────────────────────────────────────────────────────────
# Step execution  (engine-agnostic)
# ─────────────────────────────────────────────────────────────────────────────

def execute_step(
    state:         RunState,
    step:          Step,
    cell_src:      Path,
    param_src:     Path,
    write_cell_fn: Callable[[Path, float], None],
    output_suffix: str  = ".castep",
    keep_all:      bool = False,
) -> ExecResult:
    """Prepare inputs, run the engine, parse outputs, and persist state.

    Intentionally engine-agnostic: all engine-specific knowledge lives in
    *write_cell_fn* (cell writing) and in the engine's output parser.
    Swapping to VASP means supplying a different *write_cell_fn* and
    pointing *output_suffix* at the VASP output — nothing here changes.

    Args:
        state:         Active :class:`RunState`.
        step:          The step to execute (mutated in-place).
        cell_src:      Template .cell file; backed up once per project.
        param_src:     Template .param file copied into each step directory.
        write_cell_fn: ``(dest_path, x) -> None`` — writes the engine input.
        output_suffix: File extension for engine output (default ``.castep``).
        keep_all:      When ``False``, bulky binary files are removed on success.
    """
    x    = step.concentration
    seed = state.seed
    castep_base = state.proj_dir / config.CASTEP_SUBDIR
    castep_base.mkdir(parents=True, exist_ok=True)
    step_dir = castep_base / f"x{x:.4f}"
    step_dir.mkdir(parents=True, exist_ok=True)
    step.step_dir = f"{config.CASTEP_SUBDIR}/x{x:.4f}"

    # Back up template files once per project (non-destructive).
    for src, name in [
        (cell_src,  f"original_{seed}.cell"),
        (param_src, f"original_{seed}.param"),
    ]:
        dst = state.proj_dir / name
        if not dst.exists() and src.exists():
            shutil.copy2(src, dst)

    # Write per-step inputs.
    write_cell_fn(step_dir / f"{seed}.cell", x)
    shutil.copy2(param_src, step_dir / f"{seed}.param")

    # Adjust nextra_bands per-step based on VEC.
    from castep.castep import nextra_for_step, patch_nextra
    nextra = (
        config.ELASTIC_NEXTRA_PURE
        if state.single_mode
        else nextra_for_step(state.species_a, state.species_b, x)
    )
    patch_nextra(step_dir / f"{seed}.param", nextra)
    step.parsed["nextra_bands_used"] = nextra

    step.status     = RUNNING
    step.started_at = _now()
    save_run(state)
    write_csv(state)

    # Prepare-only mode — no binary configured.
    if not state.castep_cmd:
        step.status      = DONE
        step.rc          = "N/A"
        step.finished_at = _now()
        save_run(state)
        write_csv(state)
        return ExecResult(rc=0, skipped=False, stderr_tail=[])

    cmd    = os.path.expanduser(state.castep_cmd.replace("{seed}", seed))
    result = run_process(cmd, step_dir, step_dir / f"{seed}{output_suffix}")
    step.finished_at = _now()

    if result.skipped:
        step.status = SKIPPED
        step.rc     = "ctrl-c"
    else:
        step.rc = str(result.rc) if result.rc is not None else "unknown"

        from castep.castep import parse_elastic_file, parse_output
        parsed = parse_output(step_dir / f"{seed}{output_suffix}")
        step.parsed.update(parsed.to_dict())

        elastic_path = step_dir / f"{seed}.elastic"
        if elastic_path.exists():
            step.parsed.update(parse_elastic_file(elastic_path))

        step.status = DONE if result.rc == 0 else FAILED
        if result.rc == 0 and not keep_all:
            _cleanup(step_dir)

    save_run(state)
    write_csv(state)
    return result


def _cleanup(step_dir: Path) -> None:
    for pat in (
        "*.castep_bin", "*.check", "*.cst_esp",
        "*.bands", "*.bib", "*.usp", "*.upf", "*.oepr",
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
