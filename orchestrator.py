
from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

import config
from core_physics import Crystal, fit_cij_universal, generate_strain_steps
from crystal_modes import get_strategy
from engines.engine import (
    ElasticCapable,
    FiniteStrainCapable,
    InProcessCapable,
    InProcessElasticCapable,
    KillReason,
    WatchdogCapable,
    read_tail,
)
from runstate import (
    DONE,
    FAILED,
    PENDING,
    RUNNING,
    SKIPPED,
    StateIO,
)

if TYPE_CHECKING:
    from engines.engine import BaseEngine
    from runstate import RunState, Step


def _load_mlip_relaxed_crystal(step_dir: Path) -> Crystal | None:
    """Load the relaxed Crystal saved by MlipEngine.run_in_process.

    MlipEngine writes three .npy files after a successful GeomOpt:
      relaxed_cell.npy   — (3,3) cell matrix in Å
      relaxed_pos.npy    — (N,3) fractional coordinates
      relaxed_nums.npy   — (N,) atomic numbers (int)

    Returns None if the files are absent (non-MLIP engine, or failed run),
    so callers can fall back to the pre-relaxation Crystal without crashing.
    """
    cell_f = step_dir / "relaxed_cell.npy"
    pos_f  = step_dir / "relaxed_pos.npy"
    nums_f = step_dir / "relaxed_nums.npy"
    if not (cell_f.exists() and pos_f.exists() and nums_f.exists()):
        return None
    try:
        import ase.data
        cell = np.load(cell_f)
        pos  = np.load(pos_f)
        nums = np.load(nums_f).astype(int)
        symbols = [ase.data.chemical_symbols[z] for z in nums]
        sites   = [{s: 1.0} for s in symbols]
        return Crystal(lattice=cell, frac_coords=pos, sites=sites)
    except Exception:
        return None  # non-fatal — caller falls back to pre-relaxation crystal


class Task(ABC):
    """A single task to be executed by the orchestrator."""

    def __init__(self, state: "RunState", step: "Step", engine: "BaseEngine", crystal: "Crystal"):
        self.state = state
        self.step = step
        self.engine = engine
        self.crystal = crystal

    @abstractmethod
    def execute(self) -> bool:
        """Execute the task and return a result."""
        raise NotImplementedError

    def _build_target_mix(
        self, species: list[tuple[str, float]], x: float,
    ) -> dict[str, float]:
        """Convert state.species + x → per-species fraction on the template
        sublattice. species[0] phases out as x → 1."""
        mix = {species[0][0]: 1.0 - x}
        for e, f in species[1:]:
            mix[e] = f * x
        return mix

    def _run_process(self, cmd: str, cwd: Path, output_file: Path) -> "ExecResult":
        """Shared subprocess runner with Watchdog and SIGINT skip.

        Defined once in Task base — GeomOptTask and ElasticTask both inherit it.
        Prevents the DRY violation of having two identical copies.
        """
        stop = threading.Event()
        proc: subprocess.Popen | None = None
        stderr_tail: list[str] = []
        rc: int | None = -1
        watchdog: "_Watchdog | None" = None

        _arm_skip()
        try:
            proc = subprocess.Popen(
                cmd, shell=True, cwd=cwd,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            monitor = threading.Thread(
                target=self.engine.progress_monitor,
                args=(proc, stop, cwd),
                daemon=True,
            )
            monitor.start()

            watchdog = _Watchdog(output_file, proc, stop, self.engine)
            wd_thread = threading.Thread(target=watchdog.run, daemon=True)
            wd_thread.start()

            def _drain_stderr() -> None:
                if proc is not None and proc.stderr:
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
                    return ExecResult(
                        rc=None, skipped=True, stderr_tail=[],
                        kill_reason=KillReason.CTRL_C,
                    )
                time.sleep(0.2)

            drain.join(2)
            rc = proc.returncode
        except OSError as e:
            stderr_tail.append(str(e))
        finally:
            stop.set()
            _disarm_skip()
            if proc and proc.poll() is None:
                proc.kill()

        kill_reason = watchdog.reason if watchdog else ""
        return ExecResult(
            rc=rc, skipped=False, stderr_tail=stderr_tail,
            kill_reason=kill_reason,
        )


@dataclass
class ExecResult:
    rc: int | None
    skipped: bool
    stderr_tail: list[str]
    kill_reason: str = ""


class GeomOptTask(Task):
    """A geometry optimization task."""

    def __init__(
        self,
        state: "RunState",
        step: "Step",
        engine: "BaseEngine",
        crystal: "Crystal",
        keep_all: bool = False,
    ):
        super().__init__(state, step, engine, crystal)
        self.keep_all = keep_all

    def execute(self) -> bool:
        x = self.step.concentration
        seed = self.state.seed

        if self.state.crystal_mode not in self.engine.SUPPORTED_MODES:
            print(
                f"ERROR: {self.engine.name} does not support '{self.state.crystal_mode}' mode. "
                f"Supported: {sorted(self.engine.SUPPORTED_MODES)}"
            )
            return False

        base_dir = self.state.proj_dir / self.engine.subdir_name
        base_dir.mkdir(parents=True, exist_ok=True)
        step_dir = base_dir / f"x{x:.4f}"
        step_dir.mkdir(parents=True, exist_ok=True)
        self.step.step_dir = f"{self.engine.subdir_name}/x{x:.4f}"

        target_mix = self._build_target_mix(self.state.species, x)
        strategy = get_strategy(self.state.crystal_mode)
        prepared = strategy.prepare(
            self.crystal, self.state.template_element, target_mix, x, step_dir,
        )
        self.step.parsed.update(prepared.metadata)

        self.engine.write_input(
            step_dir, seed, prepared.crystal,
        )

        self.step.status = RUNNING
        self.step.started_at = StateIO._now()
        StateIO.save_run(self.state)

        if isinstance(self.engine, InProcessCapable):
            result = self._execute_in_process(
                prepared, x, seed
            )
        elif not self.state.engine_cmd:
            self.step.status = DONE
            self.step.rc = "N/A"
            self.step.finished_at = StateIO._now()
            StateIO.save_run(self.state)
            return True
        else:
            result = self._execute_subprocess(seed, step_dir)

        StateIO.save_run(self.state)
        return result.rc == 0

    def _execute_subprocess(
        self, seed: str, step_dir: Path,
    ) -> ExecResult:
        output_file = (
            step_dir / f"{seed}{self.engine.output_suffix}"
            if self.engine.output_suffix.startswith(".")
            else step_dir / self.engine.output_suffix
        )
        cmd = os.path.expanduser(self.state.engine_cmd.replace("{seed}", seed))

        result = self._run_process(cmd, step_dir, output_file)
        self.step.finished_at = StateIO._now()

        if result.skipped:
            self.step.status = SKIPPED
            self.step.rc = KillReason.CTRL_C
        else:
            self.step.rc = str(result.rc) if result.rc is not None else "unknown"
            self._ingest_engine_result(
                output_file, step_dir, seed, result,
            )
            self.step.status = DONE if result.rc == 0 else FAILED
            if result.rc == 0 and not self.keep_all:
                self.engine.cleanup(step_dir)

        return result

    def _execute_in_process(
        self, prepared, x: float, seed: str,
    ) -> ExecResult:
        """Run an InProcessCapable engine (e.g. MLIP) in a worker thread.

        Uses concurrent.futures.ThreadPoolExecutor so:
          • The main thread stays free to catch KeyboardInterrupt (Ctrl-C).
          • STEP_TIMEOUT_S applies uniformly — same constant as subprocess path.
          • stop_event is passed to the engine for cooperative cancellation
            (checked every ionic step via _StopCallback).
        """
        import concurrent.futures  # noqa: PLC0415

        step_dir  = self.state.proj_dir / self.step.step_dir
        stop_event = threading.Event()

        # Start progress monitor in background (reads .mlip_progress.json)
        stop_monitor = threading.Event()
        monitor_t = threading.Thread(
            target=self.engine.progress_monitor,
            args=(None, stop_monitor, step_dir),
            daemon=True,
        )
        monitor_t.start()

        _arm_skip()
        exec_result: ExecResult = ExecResult(rc=1, skipped=False, stderr_tail=[])
        result = None

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(
                    self.engine.run_in_process,
                    step_dir, seed, prepared.crystal, stop_event,
                )
                deadline = time.monotonic() + config.STEP_TIMEOUT_S
                while not future.done():
                    if _SKIP_FLAG:
                        stop_event.set()
                        future.cancel()
                        _disarm_skip()
                        self.step.status = SKIPPED
                        self.step.rc     = KillReason.CTRL_C
                        self.step.finished_at = StateIO._now()
                        return ExecResult(
                            rc=None, skipped=True, stderr_tail=[],
                            kill_reason=KillReason.CTRL_C,
                        )
                    if time.monotonic() > deadline:
                        stop_event.set()
                        future.cancel()
                        _disarm_skip()
                        self.step.status = FAILED
                        self.step.rc     = KillReason.TIMEOUT
                        self.step.finished_at = StateIO._now()
                        return ExecResult(
                            rc=None, skipped=False, stderr_tail=[],
                            kill_reason=KillReason.TIMEOUT,
                        )
                    time.sleep(0.5)

                result = future.result()   # re-raises engine exceptions

        except KeyboardInterrupt:
            stop_event.set()
            _disarm_skip()
            self.step.status = SKIPPED
            self.step.rc     = KillReason.CTRL_C
            self.step.finished_at = StateIO._now()
            return ExecResult(rc=None, skipped=True, stderr_tail=[],
                              kill_reason=KillReason.CTRL_C)
        except Exception as exc:
            stop_event.set()
            _disarm_skip()
            self.step.status = FAILED
            self.step.rc     = f"in_process_error: {type(exc).__name__}"
            self.step.finished_at = StateIO._now()
            self.step.parsed["warnings"] = str(exc)
            return ExecResult(rc=1, skipped=False, stderr_tail=[str(exc)])
        finally:
            stop_event.set()
            stop_monitor.set()
            _disarm_skip()

        self.step.finished_at = StateIO._now()
        self._merge_engine_result(result, self.step)
        self.step.status = DONE if result.warning is None else FAILED
        rc = 0 if result.warning is None else 1
        self.step.rc = str(rc)
        exec_result = ExecResult(rc=rc, skipped=False, stderr_tail=[])

        if rc == 0 and not self.keep_all:
            self.engine.cleanup(step_dir)
        return exec_result

    def _ingest_engine_result(
        self, output_file: Path, step_dir: Path, seed: str,
        exec_result: ExecResult,
    ) -> None:
        parsed = self.engine.parse_output(output_file)
        self._merge_engine_result(parsed, self.step)

        extra_out = self.engine.parse_extra_outputs(step_dir, seed)
        if extra_out:
            self.step.parsed.update(extra_out)

        if exec_result.kill_reason:
            self.step.parsed["kill_reason"] = exec_result.kill_reason

    def _merge_engine_result(self, parsed, step: "Step") -> None:
        result_dict = asdict(parsed)
        extra = result_dict.pop("extra_data", {})
        clean = {k: v for k, v in result_dict.items() if v is not None}
        clean.update(extra)
        if "run_time_s" in clean and "wall_time_s" not in clean:
            clean["wall_time_s"] = clean["run_time_s"]
        step.parsed.update(clean)

class ElasticTask(Task):
    """An elastic constants calculation task."""

    def execute(self) -> bool:
        step_dir = self.state.proj_dir / self.step.step_dir
        seed = self.state.seed
        x = self.step.concentration
        density = float(self.step.parsed.get("density_gcm3") or 0) or None
        volume = float(self.step.parsed.get("volume_ang3") or 0) or None

        # Build the prepared crystal (correct species/SQS) even if loading relaxed fails
        target_mix = self._build_target_mix(self.state.species, x)
        strategy = get_strategy(self.state.crystal_mode)
        # We don't need to write files here, strategy.prepare might be called with dummy dir
        prepared = strategy.prepare(
            self.crystal, self.state.template_element, target_mix, x, step_dir,
        )

        if isinstance(self.engine, ElasticCapable):
            result = self.engine.run_elastic(
                step_dir, seed, x, self.state.species,
                self.state.nonmetal or None, density, volume,
            )
        elif isinstance(self.engine, InProcessElasticCapable):
            # Load the relaxed geometry saved by MlipEngine.run_in_process.
            # _load_mlip_relaxed_crystal returns None if the files are absent,
            # in which case we fall back to prepared.crystal (unrelaxed but correct species).
            opt_crystal = _load_mlip_relaxed_crystal(step_dir)
            if opt_crystal is None:
                import logging
                logging.warning(
                    "ElasticTask: relaxed_cell.npy not found in %s — "
                    "falling back to unrelaxed (prepared) crystal. "
                    "Elastic constants may be unreliable.",
                    step_dir,
                )
                opt_crystal = prepared.crystal
            
            import concurrent.futures  # noqa: PLC0415
            stop_event = threading.Event()
            _arm_skip()
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    future = ex.submit(
                        self.engine.run_elastic_in_process,
                        opt_crystal, density, volume, stop_event,
                    )
                    deadline = time.monotonic() + config.STEP_TIMEOUT_S
                    while not future.done():
                        if _SKIP_FLAG:
                            stop_event.set(); future.cancel()
                            result = {"_elastic_error": "Interrupted (Ctrl-C)"}
                            break
                        if time.monotonic() > deadline:
                            stop_event.set(); future.cancel()
                            result = {"_elastic_error": "Elastic timed out"}
                            break
                        time.sleep(0.5)
                    else:
                        result = future.result()
            except Exception as exc:
                result = {"_elastic_error": f"in_process_elastic: {exc}"}
            finally:
                stop_event.set()
                _disarm_skip()
        else:
            result = self._finite_strain_elastic(
                step_dir, seed, x, density, volume,
            )

        self.step.parsed.update(result)
        StateIO.save_run(self.state)
        # Born instability is a physical result, not a task execution failure.
        # Only fatal when _elastic_error is present AND no Cij were recorded.
        has_cij = "C11" in result and "C12" in result and "C44" in result
        fatal = "_elastic_error" in result and not has_cij
        return not fatal

    def _finite_strain_elastic(
        self, step_dir: Path, seed: str, x: float,
        density_gcm3: float | None, volume_ang3: float | None,
    ) -> dict[str, str]:
        if not isinstance(self.engine, FiniteStrainCapable):
            return {
                "_elastic_error": (
                    f"{self.engine.name} supports neither ElasticCapable nor "
                    f"FiniteStrainCapable — cannot compute elastic constants."
                )
            }

        t0 = time.monotonic()
        try:
            opt_crystal = self.engine.load_optimised_crystal(step_dir, seed)
        except (FileNotFoundError, ValueError) as exc:
            return {"_elastic_error": f"load_optimised_crystal failed: {exc}"}

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
                self.engine.write_singlepoint_input(
                    step_dir, opt_crystal, sub_seed, ss.strain_voigt,
                )
            except (FileNotFoundError, ValueError, OSError) as exc:
                step_errors.append(f"{ss.name}: write failed: {exc}")
                continue

            output_file = (
                step_dir / f"{sub_seed}{self.engine.output_suffix}"
                if self.engine.output_suffix.startswith(".")
                else step_dir / self.engine.output_suffix
            )
            cmd = os.path.expanduser(self.state.engine_cmd.replace("{seed}", sub_seed))
            exec_result = self._run_process(cmd, step_dir, output_file)

            if exec_result.skipped:
                return {"_elastic_error": "Interrupted (Ctrl-C)"}
            if exec_result.rc not in (0, None):
                step_errors.append(
                    f"{ss.name}: rc={exec_result.rc} reason={exec_result.kill_reason}"
                )
                continue

            try:
                sv = self.engine.parse_stress_tensor(output_file)
            except (FileNotFoundError, ValueError) as exc:
                step_errors.append(f"{ss.name}: parse failed: {exc}")
                continue

            stresses.append(sv)
            strains.append(ss.strain_voigt)

        if len(stresses) < 3:
            detail = "; ".join(step_errors) if step_errors else "no step errors"
            return {
                "_elastic_error": (
                    f"Not enough stress tensors ({len(stresses)}/"
                    f"{len(strain_steps)}). Step errors: {detail}"
                )
            }

        result = fit_cij_universal(
            stresses, strains,
            density_gcm3=density_gcm3,
            n_atoms=opt_crystal.num_atoms,
            volume_ang3=volume_ang3,
        )
        result["elastic_wall_time_s"] = f"{time.monotonic() - t0:.0f}"
        result["elastic_source"] = (
            f"{self.engine.name.upper()}-FiniteStrain-{opt_crystal.lattice_type}"
        )
        self.engine.cleanup(step_dir)
        return result

class TaskRunner:
    def __init__(self, queue: list[Task]):
        self.queue = queue

    def run(self):
        for task in self.queue:
            success = task.execute()
            if not success:
                print(f"Task {type(task).__name__} failed. Halting queue.")
                break


class ParallelTaskRunner:
    """Runs GeomOptTasks concurrently; ElasticTasks sequentially after their step.

    Designed for MLIP GPU workflows where N concentration points can be
    computed independently in parallel on the same GPU (or across GPUs).
    DFT engines (CASTEP, VASP) should use max_workers=1 to avoid I/O conflicts.

    Task dependency rule: ElasticTask for step S must run AFTER GeomOptTask
    for the same step S (ElasticTask reads volume/density from step.parsed).
    All other GeomOptTasks are independent and can run concurrently.
    """

    def __init__(self, queue: list[Task], max_workers: int = 4):
        self.queue = queue
        self.max_workers = max(1, max_workers)

    def run(self) -> None:
        import concurrent.futures  # noqa: PLC0415

        # Split into per-step pairs: {step_id: (GeomOptTask, ElasticTask|None)}
        geom_tasks:    list[GeomOptTask] = []
        elastic_map:   dict[int, ElasticTask] = {}

        for task in self.queue:
            if isinstance(task, GeomOptTask):
                geom_tasks.append(task)
            elif isinstance(task, ElasticTask):
                elastic_map[id(task.step)] = task
            else:
                # Unknown task type — run sequentially for safety
                task.execute()

        def _run_step(geom: GeomOptTask) -> bool:
            success = geom.execute()
            if success:
                elastic = elastic_map.get(id(geom.step))
                if elastic:
                    success = elastic.execute()
            else:
                print(
                    f"  [parallel] GeomOptTask x={geom.step.concentration:.4f} "
                    f"failed — skipping elastic for this step."
                )
            return success

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            futures = {
                executor.submit(_run_step, g): g for g in geom_tasks
            }
            for future in concurrent.futures.as_completed(futures):
                task = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    print(
                        f"  [parallel] Step x={task.step.concentration:.4f} "
                        f"raised: {type(exc).__name__}: {exc}"
                    )


class _Watchdog:
    """Kills the engine on timeout or engine-reported health failure."""

    _POLL_S = 10.0

    def __init__(
        self, output_file: Path, proc: subprocess.Popen,
        stop: threading.Event, engine: "BaseEngine",
    ) -> None:
        self._file = output_file
        self._proc = proc
        self._stop = stop
        self._engine = engine
        self.reason = ""

    def run(self) -> None:
        t_start = time.monotonic()
        while not self._stop.is_set():
            elapsed = time.monotonic() - t_start
            if elapsed > config.STEP_TIMEOUT_S:
                self.reason = KillReason.TIMEOUT
                self._kill(
                    f"step timed out after {int(elapsed / 60)}m "
                    f"(limit {int(config.STEP_TIMEOUT_S / 60)}m)"
                )
                return
            if self._file.exists() and isinstance(self._engine, WatchdogCapable):
                try:
                    text = read_tail(self._file, max_bytes=2 * 1024 * 1024)
                except OSError:
                    self._stop.wait(self._POLL_S)
                    continue
                kill_reason = self._engine.check_health(text)
                if kill_reason:
                    self.reason = kill_reason
                    self._kill(f"engine health check failed: {kill_reason}")
                    return
            self._stop.wait(self._POLL_S)

    def _kill(self, msg: str) -> None:
        print(f"  │  ⚠  Watchdog: {msg}", flush=True)
        try:
            self._proc.terminate()
            try:
                self._proc.wait(10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        except OSError:
            pass


_SKIP_FLAG: bool = False


def _arm_skip() -> None:
    global _SKIP_FLAG
    _SKIP_FLAG = False

    def _handler(sig: int, frame: Any | None) -> None:
        global _SKIP_FLAG
        _SKIP_FLAG = True

    signal.signal(signal.SIGINT, _handler)


def _disarm_skip() -> None:
    signal.signal(signal.SIGINT, signal.SIG_DFL)
