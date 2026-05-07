"""
engines/engine.py  —  Engine protocol, EngineResult, BaseEngine ABC, and plugin registry.
══════════════════════════════════════════════════════════════════════════════════════════
STRICT CONTRACT:
  - 100% Engine-agnostic. No mention of CASTEP, VASP, MLIP, or their specific logic.
  - No legacy typing (uses built-in dict, type, list).
  - Uses @runtime_checkable Protocols for capability routing.
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
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

try:
    from typing import Self
except ImportError:
    from typing import TypeVar

    Self = TypeVar("Self", bound="BaseEngine")  # type: ignore[assignment]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core_physics import Crystal

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────


def _try_float(s: str) -> float | None:
    """Returns None on failure instead of crashing."""
    try:
        return float(s.strip())
    except (ValueError, AttributeError):
        return None


def read_tail(file_path: Path, max_bytes: int = 5 * 1024 * 1024) -> str:
    """Reads the end of a file without loading the whole file into RAM."""
    if not file_path.exists():
        return ""
    size = file_path.stat().st_size
    with open(file_path, "rb") as f:
        if size > max_bytes:
            f.seek(-max_bytes, os.SEEK_END)
        return f.read().decode("utf-8", errors="replace")


# ─────────────────────────────────────────────────────────────────────────────
# Core Data Model
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class EngineResult:
    """Standardized output from any DFT/ML engine."""

    energy_ev: float | None = None
    volume_ang3: float | None = None
    density_gcm3: float | None = None
    run_time_s: float | None = None
    extra_data: dict[str, Any] = field(default_factory=dict)
    warning: str | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Capability Protocols  (use isinstance(), never hasattr())
# ─────────────────────────────────────────────────────────────────────────────


@runtime_checkable
class ElasticCapable(Protocol):
    """Engine can run its own internal elastic constants workflow."""

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
        species_mix: list[tuple[str, float]],
        x: float,
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
    """
    Engine evaluates its own log tail to determine if it has stalled/failed.
    Replaces brittle regex pattern passing.
    """

    def check_health(self, log_tail: str) -> str | None:
        """Returns a kill_reason string (e.g. 'scf_nosconv') if dead, else None."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# BaseEngine ABC
# ─────────────────────────────────────────────────────────────────────────────


class BaseEngine(ABC):
    """Abstract base every DFT/ML backend must subclass."""

    name: str
    output_suffix: str
    subdir_name: str
    SUPPORTED_MODES: frozenset[str]
    _cleanup_globs: list[str] = []

    # ── Abstract ──────────────────────────────────────────────────────────────

    @abstractmethod
    def write_input(
        self,
        dest_dir: Path,
        seed: str,
        crystal: Crystal,
        species_mix: list[tuple[str, float]],
        x: float,
        template_element: str,
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
        """Iterates self._cleanup_globs. Logs OSError — never silently swallows."""
        for pat in self._cleanup_globs:
            for f in step_dir.glob(pat):
                try:
                    f.unlink()
                except OSError as exc:
                    print(f"  [cleanup] Could not remove {f}: {exc}")

    def progress_monitor(self, proc: subprocess.Popen, stop: threading.Event) -> None:
        """No-op default. Engines override with live monitoring."""
        pass

    def parse_extra_outputs(self, step_dir: Path, seed: str) -> dict[str, Any]:
        """Post-parse hook (e.g., for CASTEP's .elastic file)."""
        return {}

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize engine state for JSON persistence.
        Path objects → {"__type__": "path", "val": str(v)}
        """
        result: dict[str, Any] = {}
        for k, v in vars(self).items():
            if k.startswith("_") or callable(v):
                continue
            if isinstance(v, Path):
                result[k] = {"__type__": "path", "val": str(v)}
            else:
                result[k] = v
        return result

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BaseEngine:
        """Reconstruct engine from serialized dict."""
        kwargs: dict[str, Any] = {}
        for k, v in d.items():
            if isinstance(v, dict) and v.get("__type__") == "path":
                kwargs[k] = Path(v["val"])
            elif isinstance(v, str) and (v.startswith("/") or v.startswith("~")):
                kwargs[k] = Path(os.path.expanduser(v))
            else:
                kwargs[k] = v
        return cls(**kwargs)

    @classmethod
    def find_resource(
        cls,
        search_paths: list[str],
        *,
        is_dir: bool = False,
        must_be_executable: bool = False,
    ) -> str | None:
        """Unified resource finder with expansion and globbing."""
        for pat in search_paths:
            expanded = os.path.expanduser(pat)

            if "/" not in expanded and not expanded.startswith("~"):
                found = shutil.which(expanded)
                if found:
                    candidate = Path(found)
                    if is_dir and not candidate.is_dir():
                        continue
                    if must_be_executable and not os.access(found, os.X_OK):
                        continue
                    return found
                continue

            candidates = (
                sorted(glob.glob(expanded))
                if any(c in expanded for c in "*?[")
                else [expanded]
            )
            for hit in reversed(candidates):
                p = Path(hit)
                if is_dir and p.is_dir():
                    return hit
                elif not is_dir and p.is_file():
                    if must_be_executable and not os.access(hit, os.X_OK):
                        continue
                    return hit
        return None


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


def discover_engines(package_name: str = "engines") -> dict[str, str | None]:
    """Dynamically imports all engine subpackages. Returns dict[name, error_str]."""
    results: dict[str, str | None] = {}
    try:
        pkg = importlib.import_module(package_name)
    except ImportError:
        return results

    if not hasattr(pkg, "__path__"):
        return results

    for _, subpkg_name, is_pkg in pkgutil.iter_modules(pkg.__path__):  # type: ignore
        if not is_pkg:
            continue
        subpkg_path = f"{package_name}.{subpkg_name}"
        try:
            subpkg = importlib.import_module(subpkg_path)
            if hasattr(subpkg, "__path__"):
                for _, module_name, _ in pkgutil.iter_modules(subpkg.__path__):  # type: ignore
                    importlib.import_module(f"{subpkg_path}.{module_name}")
            results[subpkg_name] = None
        except Exception:
            results[subpkg_name] = traceback.format_exc()
    return results


def is_engine_available(name: str) -> bool:
    return name in ENGINES
