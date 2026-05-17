"""
engines/engine.py  —  Engine protocol, EngineResult, BaseEngine ABC, registry.
═══════════════════════════════════════════════════════════════════════════════
STRICT CONTRACT:
  - 100% engine-agnostic. No CASTEP/VASP/MLIP-specific logic.
  - Capability routing via @runtime_checkable Protocols and isinstance().
  - Engines never reference each other; orchestrator never names them.
"""

from __future__ import annotations

import argparse
import glob
import importlib
import os
import pkgutil
import shutil
import subprocess
import threading
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from core_physics import Crystal
    from crystal_modes import PreparedCrystal


# ─────────────────────────────────────────────────────────────────────────────
# Kill-reason constants — single source of truth for orchestrator & engines
# ─────────────────────────────────────────────────────────────────────────────


class KillReason:
    """Shared kill-reason strings. Orchestrator owns TIMEOUT + CTRL_C only;
    engine-specific reasons (SCF, stress) live in their own engine module."""

    TIMEOUT = "timeout"
    CTRL_C = "ctrl-c"


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────


def try_float(s: str) -> float | None:
    """Best-effort float() — returns None on failure rather than raising."""
    try:
        return float(s.strip())
    except (ValueError, AttributeError):
        return None


# Back-compat alias — to be removed once all engines migrate to try_float.
_try_float = try_float


def read_tail(file_path: Path, max_bytes: int = 5 * 1024 * 1024) -> str:
    """Read the tail of a file without loading the whole thing into RAM."""
    if not file_path.exists():
        return ""
    size = file_path.stat().st_size
    with open(file_path, "rb") as f:
        if size > max_bytes:
            f.seek(-max_bytes, os.SEEK_END)
        return f.read().decode("utf-8", errors="replace")


# ─────────────────────────────────────────────────────────────────────────────
# Terminal progress bar — shared by every engine.progress_monitor
# ─────────────────────────────────────────────────────────────────────────────


class ProgressBar:
    """Terminal progress bar: width detection, mm:ss elapsed, bar rendering."""

    def __init__(self, bar_width: int = 22) -> None:
        self._t0 = time.time()
        self._bar_width = bar_width

    def bar(self, pct: float) -> str:
        filled = int(max(0.0, min(1.0, pct)) * self._bar_width)
        return "█" * filled + "░" * (self._bar_width - filled)

    def elapsed(self) -> str:
        mm, ss = divmod(int(time.time() - self._t0), 60)
        return f"{mm:02d}:{ss:02d}"

    def render(self, content: str) -> None:
        w = shutil.get_terminal_size((80, 24)).columns
        line = f"  │  {content}"
        if len(line) > w - 1:
            line = line[: w - 4] + "..."
        print(f"\r{line.ljust(w - 1)}", end="", flush=True)

    @staticmethod
    def clear() -> None:
        w = shutil.get_terminal_size((80, 24)).columns
        print("\r" + " " * w, end="\r", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# EngineResult — unified output envelope
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class EngineResult:
    """Standardized output from any DFT/ML engine.

    Engines MUST NOT add fields here; engine-specific outputs go into
    extra_data and are flattened into step.parsed by the orchestrator.
    """

    energy_ev: float | None = None
    volume_ang3: float | None = None
    density_gcm3: float | None = None
    run_time_s: float | None = None
    extra_data: dict[str, Any] = field(default_factory=dict)
    warning: str | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Capability Protocols  (use isinstance(), NEVER hasattr())
# ─────────────────────────────────────────────────────────────────────────────


@runtime_checkable
class ElasticCapable(Protocol):
    """Engine runs its own internal elastic constants workflow (e.g. VASP IBRION=6)."""

    def run_elastic(
        self,
        step_dir: Path,
        seed: str,
        x: float,
        species: list[tuple[str, float]],
        nonmetal: str | None,
        density_gcm3: float | None,
        volume_ang3: float | None,
    ) -> dict[str, str]: ...


@runtime_checkable
class FiniteStrainCapable(Protocol):
    """Engine supports orchestrator-driven finite-strain elastic fallback."""

    def load_optimised_crystal(self, step_dir: Path, seed: str) -> Crystal: ...

    def write_singlepoint_input(
        self,
        dest_dir: Path,
        crystal: Crystal,
        seed: str,
        strain_voigt: np.ndarray,
    ) -> None: ...

    def parse_stress_tensor(self, output_file: Path) -> np.ndarray: ...


@runtime_checkable
class RecoveryCapable(Protocol):
    """Engine knows how to patch its own input files after failure."""

    def patch_for_recovery(
        self, step_dir: Path, seed: str, error_type: str
    ) -> bool: ...

    def retry_schema(self) -> list[dict[str, str]]: ...


@runtime_checkable
class WatchdogCapable(Protocol):
    """Engine evaluates its own log tail to decide if it has stalled/died."""

    def check_health(self, log_tail: str) -> str | None:
        """Return a kill_reason string if dead, else None."""
        ...


@runtime_checkable
class WizardBypassCapable(Protocol):
    """
    Engine provides its own composition/sweep parameters and skips ui.wizard_mode.
    Replaces the legacy `getattr(engine, "skip_vca_wizard")` magic attribute.
    Must return a WizardResult — imported lazily to avoid circular dependency.
    """

    def bypass_wizard(
        self, crystal: "Crystal", args: argparse.Namespace
    ) -> "Any": ...   # return type is WizardResult; Any avoids circular import


@runtime_checkable
class InProcessCapable(Protocol):
    """
    Engine executes its calculation in-process (no subprocess). The orchestrator
    invokes `run_in_process` directly and skips run_process/Watchdog entirely.
    Designed for MLIP-style engines that drive ASE relaxers in the same Python.

    stop_event: threading.Event set by the orchestrator when a timeout or
    Ctrl-C is detected. Engines must check it periodically and raise
    InterruptedError to abort cleanly.
    """

    def run_in_process(
        self,
        step_dir: Path,
        seed: str,
        crystal: "Crystal",
        stop_event: "threading.Event | None",
    ) -> EngineResult: ...


@runtime_checkable
class InProcessElasticCapable(Protocol):
    """
    Engine computes elastic constants in-process (no subprocess).
    Used by MLIP engines that drive ASE-based strain evaluations.
    The orchestrator checks isinstance(engine, InProcessElasticCapable)
    as the third branch in ElasticTask — after ElasticCapable (VASP IBRION=6)
    and FiniteStrainCapable (CASTEP subprocess).

    stop_event: threading.Event for cancellation (same convention as
    InProcessCapable). Implementations must return early if set.
    """

    def run_elastic_in_process(
        self,
        crystal: "Crystal",
        density_gcm3: "float | None",
        volume_ang3: "float | None",
        stop_event: "threading.Event | None",
    ) -> dict[str, str]: ...


# ─────────────────────────────────────────────────────────────────────────────
# BaseEngine ABC
# ─────────────────────────────────────────────────────────────────────────────


class BaseEngine(ABC):
    """Abstract base every DFT/ML backend must subclass."""

    name: str
    output_suffix: str
    subdir_name: str
    SUPPORTED_MODES: frozenset[str]
    # Per-mode advisory text — printed once at run start when the user picks
    # this mode. Use for "physically unreliable" warnings (e.g. VASP+VCA).
    MODE_WARNINGS: dict[str, str] = {}
    _cleanup_globs: list[str] = []

    # ── Abstract ──────────────────────────────────────────────────────────────

    @abstractmethod
    def write_input(
        self,
        dest_dir: Path,
        seed: str,
        crystal: Crystal,
    ) -> None: ...

    @abstractmethod
    def parse_output(self, output_file: Path) -> EngineResult: ...

    @classmethod
    @abstractmethod
    def get_wizard_schema(
        cls, crystal: Crystal, is_vca: bool
    ) -> list[dict[str, Any]]: ...

    @classmethod
    @abstractmethod
    def setup_interactive(
        cls,
        src: Path,
        crystal: Crystal,
        override_cmd: str | None,
        args: argparse.Namespace,
    ) -> tuple[BaseEngine, str]: ...

    # ── Concrete defaults ─────────────────────────────────────────────────────

    def cleanup(self, step_dir: Path) -> None:
        """Iterate self._cleanup_globs. Logs OSError — never silently swallows."""
        for pat in self._cleanup_globs:
            for f in step_dir.glob(pat):
                try:
                    f.unlink()
                except OSError as exc:
                    print(f"  [cleanup] Could not remove {f}: {exc}")

    def progress_monitor(
        self, proc: subprocess.Popen, stop: threading.Event, cwd: Path
    ) -> None:
        """No-op default. Engines override for live monitoring.

        cwd is passed explicitly (W19 fix) — no more monkey-patching proc._cwd.
        """
        pass

    def parse_extra_outputs(self, step_dir: Path, seed: str) -> dict[str, Any]:
        """Post-parse hook (e.g., for CASTEP's .elastic file)."""
        return {}

    # ── Serialization (Path-aware, no string heuristics) ──────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize engine state for JSON persistence.

        Path objects → {"__type__": "path", "val": str(v)}. No string
        heuristics: anything that's not a Path is stored as-is.
        """
        result: dict[str, Any] = {}
        for k, v in vars(self).items():
            if k.startswith("_"):
                continue
            if isinstance(v, Path):
                result[k] = {"__type__": "path", "val": str(v)}
            else:
                result[k] = v
        return result

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BaseEngine:
        """Reconstruct engine from to_dict() output.

        Only the explicit {"__type__": "path"} envelope is treated as a Path.
        F4 fix: NO string heuristic — strings stay strings.
        """
        kwargs: dict[str, Any] = {}
        for k, v in d.items():
            if isinstance(v, dict) and v.get("__type__") == "path":
                kwargs[k] = Path(v["val"])
            else:
                kwargs[k] = v
        return cls(**kwargs)

    # ── Resource discovery ────────────────────────────────────────────────────

    @classmethod
    def find_resource(
        cls,
        search_paths: list[str],
        *,
        is_dir: bool = False,
        must_be_executable: bool = False,
    ) -> Path | None:
        """Unified file/dir/executable finder with expansion and globbing."""

        def _ok(p: Path) -> bool:
            if is_dir:
                return p.is_dir()
            if not p.is_file():
                return False
            return os.access(p, os.X_OK) if must_be_executable else True

        for pat in search_paths:
            expanded = os.path.expanduser(pat)
            if any(c in expanded for c in "*?["):
                hits = sorted(glob.glob(expanded))
            elif "/" in expanded or expanded.startswith("~"):
                hits = [expanded]
            else:
                which = shutil.which(expanded)
                hits = [which] if which else []

            for hit in reversed(hits):
                p = Path(hit)
                if _ok(p):
                    return p
        return None

    # ── Shared setup helpers (DRY between CASTEP/VASP/MLIP) ──────────────────

    @classmethod
    def _resolve_binary(
        cls, search_paths: list[str], override_cmd: str | None, name: str
    ) -> str:
        """Locate an engine binary, prompting the user as a last resort."""
        import ui  # local — avoids circular dependency

        found = (
            Path(override_cmd) if override_cmd
            else cls.find_resource(search_paths, must_be_executable=True)
        )
        bin_path = str(found) if found else ""
        while not bin_path:
            print(f"  ⚠  {name} binary not found automatically.")
            ans = ui.ask_str(f"  Path to {name} (or 'skip'): ").strip()
            if ans.lower() == "skip":
                return ""
            bin_path = os.path.expanduser(ans)
        print(f"  ✓  Found executable: {bin_path}")
        return bin_path

    @classmethod
    def _resolve_mpi_procs(cls, args: argparse.Namespace) -> int:
        """Return MPI process count from --cores or interactive prompt."""
        import ui

        cpu = os.cpu_count() or 4
        cores = getattr(args, "cores", None)
        if cores is not None:
            n = max(1, int(cores))
            print(f"  MPI processes : {n}  (from --cores)")
            return n
        raw = ui.ask_str(f"  MPI processes [{cpu}]: ", str(cpu))
        try:
            return max(1, int(raw))
        except ValueError:
            return cpu

    @classmethod
    def _build_mpi_cmd(cls, bin_path: str, n_procs: int) -> str:
        """Assemble an mpirun command string from a binary path and proc count."""
        if not bin_path:
            return ""
        binary_name = Path(bin_path).name.lower()
        already_mpi = "mpi" in binary_name
        if n_procs > 1 or already_mpi:
            return f"mpirun -n {n_procs} {bin_path}"
        return bin_path

    @classmethod
    def _detect_species_flags(cls, crystal: Crystal) -> tuple[bool, bool]:
        """Inspect species in crystal for (has_hard, has_mag) via config.ELEMENTS."""
        import config as _cfg
        species = list(dict.fromkeys(crystal.species)) if crystal else []
        has_hard = any(
            _cfg.ELEMENTS.get(s.capitalize(), {}).get("hard") for s in species
        )
        has_mag = any(
            _cfg.ELEMENTS.get(s.capitalize(), {}).get("mag") for s in species
        )
        return has_hard, has_mag


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────


ENGINES: dict[str, type[BaseEngine]] = {}


def register_engine(name: str) -> Callable:
    def wrapper(cls: type[BaseEngine]) -> type[BaseEngine]:
        if not (isinstance(cls, type) and issubclass(cls, BaseEngine)):
            raise TypeError(
                f"@register_engine('{name}'): {cls.__name__} must subclass BaseEngine."
            )
        ENGINES[name] = cls
        return cls

    return wrapper


@dataclass
class DiscoveryResult:
    """Structured outcome of importing one engine subpackage."""
    ok: bool
    error: BaseException | None = None

    @property
    def summary(self) -> str:
        if self.ok:
            return "loaded"
        et = type(self.error).__name__ if self.error else "Error"
        msg = str(self.error) if self.error else "unknown"
        return f"{et}: {msg.splitlines()[0] if msg else ''}"


def discover_engines(package_name: str = "engines") -> dict[str, DiscoveryResult]:
    """Import all engine subpackages. Returns dict[name, DiscoveryResult]."""
    results: dict[str, DiscoveryResult] = {}
    try:
        pkg = importlib.import_module(package_name)
    except ImportError:
        return results

    if not hasattr(pkg, "__path__"):
        return results

    for _, subpkg_name, is_pkg in pkgutil.iter_modules(pkg.__path__):
        if not is_pkg:
            continue
        subpkg_path = f"{package_name}.{subpkg_name}"
        try:
            subpkg = importlib.import_module(subpkg_path)
            if hasattr(subpkg, "__path__"):
                for _, module_name, _ in pkgutil.iter_modules(subpkg.__path__):
                    importlib.import_module(f"{subpkg_path}.{module_name}")
            results[subpkg_name] = DiscoveryResult(ok=True)
        except Exception as exc:
            # Keep full traceback available via the exception object
            exc.__traceback_str__ = traceback.format_exc()  # type: ignore[attr-defined]
            results[subpkg_name] = DiscoveryResult(ok=False, error=exc)
    return results


def is_engine_available(name: str) -> bool:
    return name in ENGINES
