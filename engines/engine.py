"""
engines/engine.py  —  Engine protocol, EngineResult, and plugin registry.
══════════════════════════════════════════════════════════════════════════
Single Source of Truth for:
  - EngineResult dataclass (uniform output shape for all engines)
  - Engine Protocol (the contract every backend must implement)
  - Plugin registry (@register_engine decorator + discover_engines())
  - Reusable I/O and Parsing Utilities
"""

from __future__ import annotations

import importlib
import os
import pkgutil
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Protocol, Type, runtime_checkable

# ─────────────────────────────────────────────────────────────────────────────
# Shared Utilities (DRY code reductions)
# ─────────────────────────────────────────────────────────────────────────────

def _try_float(s: str) -> float | None:
    """Unified float parsing utility. Returns None on failure instead of crashing."""
    try:
        return float(s.strip())
    except ValueError:
        return None

def read_tail(file_path: Path, max_bytes: int = 5 * 1024 * 1024) -> str:
    """
    Reads the end of a file efficiently without loading the whole file into RAM.
    Defaults to the last 5MB. Crucial for massive OUTCAR or .castep files.
    """
    if not file_path.exists():
        return ""
    size = file_path.stat().st_size
    with open(file_path, "rb") as f:
        if size > max_bytes:
            f.seek(-max_bytes, os.SEEK_END)
        return f.read().decode("utf-8", errors="replace")

# ─────────────────────────────────────────────────────────────────────────────
# Core Data Models
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
# Engine Protocol
# ─────────────────────────────────────────────────────────────────────────────

@runtime_checkable
class Engine(Protocol):
    """Structural protocol every DFT/ML backend must satisfy."""
    name: str
    output_suffix: str

    def write_input(
        self,
        dest_dir: Path,
        seed: str,
        crystal: Any, # Crystal from core_physics
        species_mix: list[tuple[str, float]],
        x: float,
    ) -> None: ...

    def parse_output(self, output_file: Path) -> EngineResult: ...

    def cleanup(self, step_dir: Path) -> None: ...

    @classmethod
    def get_wizard_schema(cls, crystal: Any, is_vca: bool) -> list[dict]: ...

    def progress_monitor(self, proc: Any, stop: Any) -> None: ...

# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

ENGINES: Dict[str, Type[Engine]] = {}

def register_engine(name: str) -> Callable:
    def wrapper(cls: Type[Engine]) -> Type[Engine]:
        ENGINES[name] = cls
        return cls
    return wrapper

def discover_engines(package_name: str = "engines") -> None:
    """
    Dynamically discovers and imports all Python modules inside engine subdirectories.
    This ensures the @register_engine decorators are executed.
    """
    try:
        pkg = importlib.import_module(package_name)
    except ImportError:
        return

    if not hasattr(pkg, "__path__"):
        return

    # 1. Iterate over subdirectories (e.g., CASTEP, VASP)
    for _, subpkg_name, is_pkg in pkgutil.iter_modules(pkg.__path__): # type: ignore
        if is_pkg:
            subpkg_path = f"{package_name}.{subpkg_name}"
            try:
                subpkg = importlib.import_module(subpkg_path)

                # 2. Iterate over the actual .py files inside (e.g., castep.py)
                if hasattr(subpkg, "__path__"):
                    for _, module_name, _ in pkgutil.iter_modules(subpkg.__path__): # type: ignore
                        importlib.import_module(f"{subpkg_path}.{module_name}")

            except Exception as e:
                print(f"  [Warning] Failed to load engine package '{subpkg_name}': {e}")

def is_engine_available(engine_name: str) -> bool:
    return engine_name in ENGINES

def find_engine_binary(cmd_template: str) -> str | None:
    bin_name = cmd_template.split()[0]
    return shutil.which(bin_name)
