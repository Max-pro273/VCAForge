"""
engine.py  —  Abstract engine interface.

Any DFT engine (CASTEP, VASP, QE, ...) must expose:
  - write_input(dest, cell_template, species_mix, x)
  - write_params(dest, task, xc, cutoff, spin, nextra, smearing)
  - parse_output(path) -> EngineResult
  - output_suffix: str
  - name: str

Concrete implementations live in castep/ (and future vasp/, qe/).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class EngineResult:
    """Parsed output common to all engines."""

    enthalpy_eV: float | None = None
    a_opt_ang: float | None = None
    b_opt_ang: float | None = None
    c_opt_ang: float | None = None
    volume_ang3: float | None = None
    density_gcm3: float | None = None
    bulk_modulus_GPa: float | None = None
    wall_time_s: float | None = None
    geom_converged: bool = False
    task_type: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        def fmt(v, fmt_str):
            return fmt_str.format(v) if v is not None else ""

        return {
            "enthalpy_eV": fmt(self.enthalpy_eV, "{:.6f}"),
            "a_opt_ang": fmt(self.a_opt_ang, "{:.5f}"),
            "b_opt_ang": fmt(self.b_opt_ang, "{:.5f}"),
            "c_opt_ang": fmt(self.c_opt_ang, "{:.5f}"),
            "volume_ang3": fmt(self.volume_ang3, "{:.4f}"),
            "density_gcm3": fmt(self.density_gcm3, "{:.4f}"),
            "bulk_modulus_GPa": fmt(self.bulk_modulus_GPa, "{:.2f}"),
            "wall_time_s": fmt(self.wall_time_s, "{:.1f}"),
            "geom_converged": "yes" if self.geom_converged else "no",
            "warnings": "; ".join(self.warnings[:3]),
        }


@runtime_checkable
class Engine(Protocol):
    name: str
    output_suffix: str

    def write_input(
        self,
        dest_dir: Any,
        cell_template: Any,
        species_mix: list[tuple[str, float]],
        x: float,
    ) -> None: ...

    def write_params(
        self,
        dest: Any,
        task: str,
        xc: str,
        cutoff: int,
        spin: bool,
        nextra: int,
        smearing: float,
    ) -> None: ...

    def parse_output(self, path: Any) -> EngineResult: ...
