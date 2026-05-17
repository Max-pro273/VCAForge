"""
engines/CASTEP/castep.py  —  CASTEP DFT engine.
═══════════════════════════════════════════════════════════════
Registered as "castep" via @register_engine decorator.
Elastic strategy: orchestrator-driven finite-strain fallback
(FiniteStrainCapable). Recovery via .param patching (RecoveryCapable).
Watchdog via SCF/Smax log inspection (WatchdogCapable).
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

import config as _cfg
from core_physics import Crystal, load_crystal, read_geometry, vec_for_system
from engines.CASTEP.cell_param import (
    parse_castep_output,
    parse_elastic_file,
    patch_nextra,
    write_castep_param,
    write_castep_cell,
)
from engines.engine import (
    BaseEngine,
    EngineResult,
    ProgressBar,
    read_tail,
    register_engine,
)


# ─────────────────────────────────────────────────────────────────────────────
# Kill-reason constants — CASTEP-specific.
# Generic ones (TIMEOUT, CTRL_C) come from engines.engine.KillReason.
# ─────────────────────────────────────────────────────────────────────────────

_KR_SCF = "scf_nosconv"
_KR_SMAX = "smax_stall"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
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


def _nextra_bands(crystal: Crystal) -> int:
    """Extra bands heuristic — wider VCA → more bands needed."""
    is_pure = all(len(site) == 1 for site in crystal.sites)
    if is_pure:
        return _cfg.ELASTIC_NEXTRA_PURE

    vec = crystal.vec()
    return _cfg.ELASTIC_NEXTRA_BASE + int(abs(vec - 8.0) * 20)


# ─────────────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────────────


@register_engine("castep")
class CastepEngine(BaseEngine):
    """CASTEP DFT engine. Supports all crystal modes natively."""

    name = "castep"
    output_suffix = ".castep"
    subdir_name = _cfg.CASTEP_SUBDIR
    SUPPORTED_MODES = frozenset({"vca", "sqs", "direct"})

    @property
    def _cleanup_globs(self) -> list[str]:
        return _cfg.CASTEP_CLEANUP_GLOBS

    def __init__(self, engine_cmd: str, param_src: Path | str) -> None:
        self.engine_cmd = engine_cmd
        self.param_src = Path(param_src)

    # ── Setup ─────────────────────────────────────────────────────────────────

    @classmethod
    def setup_interactive(
        cls,
        src: Path,
        crystal: Crystal,
        override_cmd: str | None,
        args: argparse.Namespace,
    ) -> tuple[CastepEngine, str]:
        import ui
        ui.section("Engine execution (CASTEP)")

        bin_path = cls._resolve_binary(
            _cfg.CASTEP_SEARCH_PATHS, override_cmd, "castep"
        )
        n_procs = cls._resolve_mpi_procs(args)
        cmd_template = (
            f"{cls._build_mpi_cmd(bin_path, n_procs)} {{seed}}" if bin_path else ""
        )

        param_src = src.with_suffix(".param")
        if not param_src.exists():
            ui.section("Parameter setup (CASTEP)")
            answers = ui.render_wizard(cls.get_wizard_schema(crystal, is_vca=True))

            has_d_block = any(
                _cfg.ELEMENTS.get(s.capitalize(), {}).get("Z", 0) > 20
                for s in dict.fromkeys(crystal.species)
            )
            nextra = 20 if has_d_block else 10

            write_castep_param(
                param_src,
                task_type=answers["task"],
                xc=answers["xc"],
                cutoff=answers["cutoff"],
                spin=answers["spin"],
                nextra=nextra,
                smearing=answers["smearing"],
            )
            print(f"\n  ✓ Written: {param_src.name}")
        else:
            print(f"  .param : {param_src.name}  (found)")

        return cls(cmd_template, param_src), cmd_template

    # ── Wizard schema ─────────────────────────────────────────────────────────

    @classmethod
    def get_wizard_schema(
        cls, crystal: Crystal, is_vca: bool
    ) -> list[dict[str, Any]]:
        has_hard, has_mag = cls._detect_species_flags(crystal)
        rec_cut = _cfg.ENCUT_HARD if has_hard else _cfg.ENCUT_SOFT
        rec_smear = _cfg.SMEARING_VCA if is_vca else _cfg.SMEARING_SINGLE

        return [
            {
                "key": "task", "type": "choice",
                "label": "1/5  What calculation to run?",
                "options": _cfg.TASKS_VCA if is_vca else _cfg.TASKS_FULL,
                "default": "GeometryOptimization",
                "help": "GeometryOptimization (relax) or SinglePoint (energy only)",
            },
            {
                "key": "xc", "type": "choice",
                "label": "2/5  Exchange-correlation functional",
                "options": _cfg.XC_LIST,
                "default": _cfg.XC_DEFAULT,
                "help": "PBE (default) / PBEsol (best for ceramics) / LDA",
            },
            {
                "key": "cutoff", "type": "int",
                "label": "3/5  Plane-wave cutoff energy (eV)",
                "default": rec_cut,
                "help": (
                    "500-600 eV standard. "
                    + ("⚠ Hard elements detected: 700+ eV required" if has_hard else "")
                ),
            },
            {
                "key": "spin", "type": "bool",
                "label": "4/5  Spin polarization",
                "default": has_mag,
                "help": (
                    "⚠ Magnetic elements detected" if has_mag else "Non-magnetic default"
                ),
            },
            {
                "key": "smearing", "type": "float",
                "label": "5/5  Fermi smearing width (eV)",
                "default": rec_smear,
                "help": "0.10-0.20 eV — helps SCF convergence",
            },
        ]

    # ── Core engine methods ───────────────────────────────────────────────────

    def write_input(
        self,
        dest_dir: Path,
        seed: str,
        crystal: Crystal,
    ) -> None:
        write_castep_cell(
            dest_dir / f"{seed}.cell", crystal
        )

        dest_param = dest_dir / f"{seed}.param"
        shutil.copy2(self.param_src, dest_param)
        # Assuming x=0 for pure, x=1 for full mixture, and interpolating vec for VCA
        # This is a simplification. The `x` is not available anymore.
        # I'll use the VEC of the crystal.
        patch_nextra(
            dest_param, _nextra_bands(crystal)
        )

    def parse_output(self, output_file: Path) -> EngineResult:
        return parse_castep_output(output_file)

    def parse_extra_outputs(self, step_dir: Path, seed: str) -> dict[str, Any]:
        ep = step_dir / f"{seed}.elastic"
        return parse_elastic_file(ep) if ep.exists() else {}

    # ── WatchdogCapable ───────────────────────────────────────────────────────

    def check_health(self, log_tail: str) -> str | None:
        if "Reached maximum number of SCF cycles" in log_tail:
            return _KR_SCF
        smax = _parse_smax_history(log_tail)
        window_size = _cfg.SMAX_STALL_ITERS
        if len(smax) >= window_size:
            window = smax[-window_size:]
            if min(window) > _cfg.SMAX_KILL_GPa:
                return _KR_SMAX
        return None

    # ── RecoveryCapable ───────────────────────────────────────────────────────

    def patch_for_recovery(
        self, step_dir: Path, seed: str, error_type: str
    ) -> bool:
        """Patch .param to address SCF or stall errors. No-op for unknown errors."""
        param = step_dir / f"{seed}.param"
        if not param.exists():
            return False
        if error_type not in {_KR_SCF, _KR_SMAX, "timeout"}:
            return False

        lines = param.read_text(encoding="utf-8").splitlines()
        patched: list[str] = []
        keys_done: set[str] = set()

        for ln in lines:
            kv = ln.split(":", 1)
            key = kv[0].strip().lower() if len(kv) == 2 else ""

            if key == "smearing_width":
                patched.append("smearing_width      : 0.20 eV")
                keys_done.add("smearing_width")
                continue
            if key == "mix_charge_amp":
                patched.append("mix_charge_amp      : 0.05")
                keys_done.add("mix_charge_amp")
                continue
            patched.append(ln)

        if "smearing_width" not in keys_done:
            patched.append("smearing_width      : 0.20 eV")
        if "mix_charge_amp" not in keys_done:
            patched.append("mix_charge_amp      : 0.05")

        param.write_text("\n".join(patched) + "\n", encoding="utf-8")
        return True

    def retry_schema(self) -> list[dict[str, str]]:
        return [
            {
                "kill_reason": _KR_SCF,
                "description": "SCF did not converge.",
                "fix": "Increased smearing, reduced mix_amp.",
            },
            {
                "kill_reason": _KR_SMAX,
                "description": "Geom opt stalled at high stress.",
                "fix": "Increased smearing, reduced mix_amp.",
            },
            {
                "kill_reason": "timeout",
                "description": "Step timed out.",
                "fix": "Increased smearing, reduced mix_amp.",
            },
        ]

    # ── FiniteStrainCapable ───────────────────────────────────────────────────

    def load_optimised_crystal(self, step_dir: Path, seed: str) -> Crystal:
        orig_cell = step_dir / f"{seed}.cell"
        out_cell = step_dir / f"{seed}-out.cell"

        if not orig_cell.exists():
            raise FileNotFoundError(f"Base cell missing: {orig_cell}")

        crystal = load_crystal(orig_cell)

        if out_cell.exists():
            # Use public read_geometry — no spglib standardization, preserves
            # the orientation that the strain frame depends on.
            relaxed = read_geometry(out_cell)
            crystal.lattice = relaxed.lattice
            crystal.clear_cache()
        else:
            print(
                f"\n  [Warning] {out_cell.name} not found — using unrelaxed lattice."
            )

        return crystal

    def write_singlepoint_input(
        self,
        dest_dir: Path,
        crystal: Crystal,
        seed: str,
        strain_voigt: np.ndarray,
    ) -> None:
        if not self.param_src.exists():
            raise FileNotFoundError(f"param_src not found: {self.param_src}")

        # Voigt → deformation gradient F.
        e11, e22, e33 = strain_voigt[0], strain_voigt[1], strain_voigt[2]
        e23 = strain_voigt[3] / 2
        e13 = strain_voigt[4] / 2
        e12 = strain_voigt[5] / 2
        F = np.array([
            [1 + e11, e12, e13],
            [e12, 1 + e22, e23],
            [e13, e23, 1 + e33],
        ])

        strained = Crystal(
            lattice=crystal.lattice @ F.T,
            frac_coords=crystal.frac_coords,
            sites=crystal.sites,
        )

        write_castep_cell(
            dest_dir / f"{seed}.cell", strained
        )

        # Reuse XC and cutoff from the master .param; rewrite as SinglePoint.
        param_text = self.param_src.read_text(encoding="utf-8", errors="replace")
        m_xc = re.search(r"xc_functional\s*:\s*(\S+)", param_text, re.I)
        m_cut = re.search(r"cut_off_energy\s*:\s*(\d+)", param_text, re.I)
        if not m_xc or not m_cut:
            raise ValueError(
                f"Missing xc_functional or cut_off_energy in {self.param_src}"
            )

        write_castep_param(
            dest_dir / f"{seed}.param",
            task_type="SinglePoint",
            xc=m_xc.group(1),
            cutoff=int(m_cut.group(1)),
            spin=False,
            nextra=_nextra_bands(crystal),
            smearing=_cfg.SMEARING_SINGLE,
        )

    def parse_stress_tensor(self, output_file: Path) -> np.ndarray:
        if not output_file.exists():
            raise FileNotFoundError(f"Stress tensor output not found: {output_file}")

        text = read_tail(output_file, max_bytes=2 * 1024 * 1024)

        # Prefer the geom-opt stress block (3 rows of (sx,sy,sz) <-- S).
        geomopt_blocks = re.findall(
            r"^\s*([-+]?\d+\.\d+[Ee][+-]?\d+)\s+"
            r"([-+]?\d+\.\d+[Ee][+-]?\d+)\s+"
            r"([-+]?\d+\.\d+[Ee][+-]?\d+)\s+<-- S",
            text, re.M,
        )
        if len(geomopt_blocks) >= 3:
            m = np.array([[float(v) for v in r] for r in geomopt_blocks[-3:]])
            return np.array([m[0, 0], m[1, 1], m[2, 2],
                             m[1, 2], m[0, 2], m[0, 1]])

        # Fallback: Symmetrised Stress Tensor block from SinglePoint output.
        sp_blocks = re.findall(
            r"Symmetrised Stress Tensor.*?"
            r"\*\s+x\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s*\*\s*"
            r"\*\s+y\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s*\*\s*"
            r"\*\s+z\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s*\*",
            text, re.S,
        )
        if sp_blocks:
            v = [float(val) for val in sp_blocks[-1]]
            return np.array([v[0], v[4], v[8], v[5], v[2], v[1]])

        raise ValueError("Stress tensor not found in CASTEP output.")

    # ── Progress monitor (uses shared ProgressBar) ────────────────────────────

    def progress_monitor(
        self,
        proc: subprocess.Popen,
        stop: threading.Event,
        cwd: Path,
    ) -> None:
        # Drain stdout so the pipe never blocks.
        if proc.stdout is not None:
            threading.Thread(
                target=lambda: [_ for _ in iter(proc.stdout.readline, b"")],
                daemon=True,
            ).start()

        bar = ProgressBar()
        geo = scf = scf_seen = 0
        last_size = 0
        castep_file: Path | None = None

        # Discover the .castep file (it may not exist yet).
        deadline = time.monotonic() + 15.0
        while not stop.is_set() and time.monotonic() < deadline:
            candidates = list(cwd.glob("*.castep"))
            if candidates:
                castep_file = max(candidates, key=lambda p: p.stat().st_mtime)
                break
            time.sleep(0.5)

        try:
            while not stop.is_set():
                if castep_file is None or not castep_file.exists():
                    bar.render(f"waiting for .castep file...   ⏱ {bar.elapsed()}")
                    time.sleep(1.0)
                    continue

                size = castep_file.stat().st_size
                if size == last_size:
                    bar.render(self._fmt_progress(bar, geo, scf, scf_seen))
                    time.sleep(0.8)
                    continue

                text = read_tail(castep_file, max_bytes=2 * 1024 * 1024)
                last_size = size

                for line in text.splitlines():
                    if "LBFGS: finished iteration" in line:
                        parts = line.split()
                        try:
                            geo = int(parts[parts.index("iteration") + 1])
                            scf = 0
                        except (ValueError, IndexError):
                            pass
                    elif "<-- SCF" in line:
                        s = line.lstrip()
                        if s and s[0].isdigit():
                            try:
                                scf = int(s.split()[0])
                                scf_seen = max(scf_seen, scf)
                            except ValueError:
                                pass
                bar.render(self._fmt_progress(bar, geo, scf, scf_seen))
                time.sleep(0.8)
        finally:
            ProgressBar.clear()

    @staticmethod
    def _fmt_progress(bar: ProgressBar, geo: int, scf: int, scf_seen: int) -> str:
        scf_denom = max(scf_seen + 2, 5)
        pct = min(1.0, scf / scf_denom)
        scf_str = f"{scf}/{scf_seen}" if scf_seen else f"{scf}/—"
        return f"[{bar.bar(pct)}] {int(pct * 100):3d}%  geo {geo}  scf {scf_str}  ⏱ {bar.elapsed()}"
