"""
engines/CASTEP/castep.py  —  CASTEP Engine Implementation.
═══════════════════════════════════════════════════════════
Registered as "castep" via @register_engine decorator.
Elastic strategy: finite-strain fallback driven by orchestrator.
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

import config as _cfg
import numpy as np
from core_physics import Crystal, load_crystal
from engines.CASTEP.cell_param import (
    parse_elastic_file,
    parse_output,
    patch_nextra,
    write_engine_params,
    write_vca_cell,
)
from engines.engine import (
    BaseEngine,
    EngineResult,
    FiniteStrainCapable,
    RecoveryCapable,
    WatchdogCapable,
    read_tail,
    register_engine,
)

# Kill-reason constants
_KR_SCF = "scf_nosconv"
_KR_SMAX = "smax_stall"
_KR_TIMEOUT = "timeout"
_KR_STRESS = "geom_high_stress"


def parse_smax_history(text: str) -> list[float]:
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


def _nextra_bands(x: float, vec: float) -> int:
    """Extra bands for elastic SCF."""
    if x < 1e-5 or x > 1.0 - 1e-5:
        return getattr(_cfg, "ELASTIC_NEXTRA_PURE", 10)
    return getattr(_cfg, "ELASTIC_NEXTRA_BASE", 15) + int(abs(vec - 8.0) * 20)


@register_engine("castep")
class CastepEngine(BaseEngine):
    """CASTEP DFT engine — finite-strain elastic fallback."""

    name = "castep"
    output_suffix = ".castep"
    subdir_name = _cfg.CASTEP_SUBDIR
    SUPPORTED_MODES = frozenset({"vca", "sqs", "direct"})
    _cleanup_globs = getattr(_cfg, "CASTEP_CLEANUP_GLOBS", [])

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
    ) -> tuple["CastepEngine", str]:
        import ui

        cpu = os.cpu_count() or 4
        ui.section("Engine execution (CASTEP)")

        # Find binary using the unified BaseEngine method
        bin_path = (
            override_cmd
            if override_cmd
            else cls.find_resource(_cfg.CASTEP_SEARCH_PATHS, must_be_executable=True)
        )

        while not bin_path:
            print("  ⚠  CASTEP executable not found automatically.")
            ans = ui.ask_str(
                "  Provide absolute path to castep.mpi (or 'skip'): "
            ).strip()
            if ans.lower() == "skip":
                bin_path = ""
                break
            bin_path = os.path.expanduser(ans)

        if bin_path:
            print(f"  ✓  Found executable: {bin_path}")

        cores = getattr(args, "cores", None)
        if cores is not None:
            n = max(1, cores)
            print(f"  MPI processes : {n}  (from --cores)")
        else:
            raw = ui.ask_str(f"  MPI processes [{cpu}]: ", str(cpu))
            try:
                n = max(1, int(raw))
            except ValueError:
                n = cpu

        cmd = ""
        if bin_path:
            name = Path(bin_path).name
            cmd = (
                f"mpirun -n {n} {bin_path} {{seed}}"
                if "mpi" in name.lower()
                else f"{bin_path} {{seed}}"
            )

        # Build .param
        species_list = list(dict.fromkeys(crystal.species))
        param_src = src.with_suffix(".param")
        if not param_src.exists():
            schema = cls.get_wizard_schema(crystal, is_vca=True)
            ui.section("Parameter setup (CASTEP)")
            answers = ui.render_wizard(schema)

            has_d = any(
                _cfg.ELEMENTS.get(s.capitalize(), {}).get("Z", 0) > 20
                for s in species_list
            )
            nextra = 20 if has_d else 10

            write_engine_params(
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

        return cls(cmd, param_src), cmd

    # ── Wizard Schema ─────────────────────────────────────────────────────────

    @classmethod
    def get_wizard_schema(cls, crystal: Crystal, is_vca: bool) -> list[dict]:
        species = list(dict.fromkeys(crystal.species))
        has_hard = any(
            _cfg.ELEMENTS.get(s.capitalize(), {}).get("hard", False) for s in species
        )
        has_mag = any(
            _cfg.ELEMENTS.get(s.capitalize(), {}).get("mag", False) for s in species
        )
        rec_cut = _cfg.ENCUT_HARD if has_hard else _cfg.ENCUT_SOFT
        rec_smear = (
            _cfg.SMEARING_VCA if is_vca else getattr(_cfg, "SMEARING_SINGLE", 0.10)
        )

        return [
            {
                "key": "task",
                "label": "1/5  What calculation to run?",
                "type": "choice",
                "options": _cfg.TASKS_VCA if is_vca else _cfg.TASKS_FULL,
                "default": "GeometryOptimization",
                "help": "GeometryOptimization (relax) or SinglePoint (energy only)",
            },
            {
                "key": "xc",
                "label": "2/5  Exchange-correlation functional",
                "type": "choice",
                "options": _cfg.XC_LIST,
                "default": _cfg.XC_DEFAULT,
                "help": "PBE (default) / PBEsol (best for ceramics) / LDA",
            },
            {
                "key": "cutoff",
                "label": "3/5  Plane-wave cutoff energy (eV)",
                "type": "int",
                "default": rec_cut,
                "help": f"500-600 eV standard. {'⚠ Hard elements detected: 700+ eV required' if has_hard else ''}",
            },
            {
                "key": "spin",
                "label": "4/5  Spin polarization",
                "type": "bool",
                "default": has_mag,
                "help": f"{'⚠ Magnetic elements detected' if has_mag else 'Non-magnetic default'}",
            },
            {
                "key": "smearing",
                "label": "5/5  Fermi smearing width (eV)",
                "type": "float",
                "default": rec_smear,
                "help": "0.10-0.20 eV — helps SCF convergence",
            },
        ]

    # ── Core Engine Methods ───────────────────────────────────────────────────

    def write_input(
        self,
        dest_dir: Path,
        seed: str,
        crystal: Crystal,
        species_mix: list[tuple[str, float]],
        x: float,
        template_element: str,
    ) -> None:
        target_mix = {species_mix[0][0]: 1.0 - x}
        for e, f in species_mix[1:]:
            target_mix[e] = f * x

        scaled = crystal.with_vegard(template_element, target_mix)
        write_vca_cell(dest_dir / f"{seed}.cell", scaled, template_element, target_mix)
        dest_param = dest_dir / f"{seed}.param"
        shutil.copy2(self.param_src, dest_param)
        patch_nextra(dest_param, _nextra_bands(x, crystal.vec(species_mix)))

    def parse_output(self, output_file: Path) -> EngineResult:
        return parse_output(output_file)

    def parse_extra_outputs(self, step_dir: Path, seed: str) -> dict:
        ep = step_dir / f"{seed}.elastic"
        return parse_elastic_file(ep) if ep.exists() else {}

    # ── WatchdogCapable ───────────────────────────────────────────────────────

    def check_health(self, log_tail: str) -> str | None:
        """Analyzes CASTEP output tail. Returns a kill reason if stalled/dead."""
        if "Reached maximum number of SCF cycles" in log_tail:
            return _KR_SCF
        smax = parse_smax_history(log_tail)
        if len(smax) >= getattr(_cfg, "SMAX_STALL_ITERS", 8):
            window = smax[-getattr(_cfg, "SMAX_STALL_ITERS", 8) :]
            if min(window) > getattr(_cfg, "SMAX_KILL_GPa", 50.0):
                return _KR_SMAX
        return None

    # ── RecoveryCapable ───────────────────────────────────────────────────────

    def patch_for_recovery(self, step_dir: Path, seed: str, error_type: str) -> bool:
        param = step_dir / f"{seed}.param"
        if not param.exists():
            return False

        apply_scf = error_type in {_KR_SCF, _KR_SMAX, _KR_TIMEOUT}
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

    def retry_schema(self) -> list[dict]:
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
                "kill_reason": _KR_TIMEOUT,
                "description": "Step timed out.",
                "fix": "Increased smearing, reduced mix_amp.",
            },
            {
                "kill_reason": _KR_STRESS,
                "description": "Residual stress too high for elastic.",
                "fix": "Relaxed geom_stress_tol to 0.10 GPa.",
            },
        ]

    # ── FiniteStrainCapable ───────────────────────────────────────────────────

    def load_optimised_crystal(self, step_dir: Path, seed: str) -> Crystal:
        from core_physics import _read_raw

        orig_cell = step_dir / f"{seed}.cell"
        out_cell = step_dir / f"{seed}-out.cell"

        if not orig_cell.exists():
            raise FileNotFoundError(f"Base cell missing: {orig_cell}")

        crystal = load_crystal(orig_cell)

        if out_cell.exists():
            relaxed = _read_raw(out_cell)
            crystal.lattice = relaxed.lattice
            crystal.clear_cache()
        else:
            print(f"\n  [Warning] {out_cell.name} not found — using unrelaxed lattice.")

        return crystal

    def write_singlepoint_input(
        self,
        dest_dir: Path,
        crystal: Crystal,
        seed: str,
        species_mix: list[tuple[str, float]],
        x: float,
        strain_voigt: np.ndarray,
    ) -> None:
        if not self.param_src.exists():
            raise FileNotFoundError(f"param_src not found: {self.param_src}")

        tmpl_elem = species_mix[0][0]
        target_mix = {tmpl_elem: 1.0 - x}
        for e, f in species_mix[1:]:
            target_mix[e] = f * x

        e11, e22, e33 = strain_voigt[0], strain_voigt[1], strain_voigt[2]
        e23, e13, e12 = strain_voigt[3] / 2, strain_voigt[4] / 2, strain_voigt[5] / 2
        F = np.array(
            [
                [1 + e11, e12, e13],
                [e12, 1 + e22, e23],
                [e13, e23, 1 + e33],
            ]
        )

        strained_crystal = Crystal(
            lattice=crystal.lattice @ F.T,
            frac_coords=crystal.frac_coords,
            species=crystal.species,
        )

        write_vca_cell(
            dest_dir / f"{seed}.cell",
            strained_crystal,
            tmpl_elem,
            target_mix,
            vegard=False,
        )

        param_text = self.param_src.read_text(encoding="utf-8", errors="replace")
        m_xc = re.search(r"xc_functional\s*:\s*(\S+)", param_text, re.I)
        m_cut = re.search(r"cut_off_energy\s*:\s*(\d+)", param_text, re.I)

        if not m_xc or not m_cut:
            raise ValueError(
                f"Missing xc_functional or cut_off_energy in {self.param_src}"
            )

        write_engine_params(
            dest_dir / f"{seed}.param",
            task_type="SinglePoint",
            xc=m_xc.group(1),
            cutoff=int(m_cut.group(1)),
            spin=False,
            nextra=_nextra_bands(x, crystal.vec(species_mix)),
            smearing=getattr(_cfg, "SMEARING_SINGLE", 0.1),
        )

    def parse_stress_tensor(self, output_file: Path) -> np.ndarray:
        if not output_file.exists():
            raise FileNotFoundError(f"Stress tensor output not found: {output_file}")

        text = read_tail(output_file, max_bytes=2 * 1024 * 1024)

        geomopt_blocks = re.findall(
            r"^\s*([-+]?\d+\.\d+[Ee][+-]?\d+)\s+"
            r"([-+]?\d+\.\d+[Ee][+-]?\d+)\s+"
            r"([-+]?\d+\.\d+[Ee][+-]?\d+)\s+<-- S",
            text,
            re.M,
        )
        if len(geomopt_blocks) >= 3:
            m = np.array([[float(v) for v in r] for r in geomopt_blocks[-3:]])
            return np.array([m[0, 0], m[1, 1], m[2, 2], m[1, 2], m[0, 2], m[0, 1]])

        sp_blocks = re.findall(
            r"Symmetrised Stress Tensor.*?"
            r"\*\s+x\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s*\*\s*"
            r"\*\s+y\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s*\*\s*"
            r"\*\s+z\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s*\*",
            text,
            re.S,
        )
        if sp_blocks:
            v = [float(val) for val in sp_blocks[-1]]
            return np.array([v[0], v[4], v[8], v[5], v[2], v[1]])

        raise ValueError("Stress tensor not found in CASTEP output.")

    # ── Progress Monitor ──────────────────────────────────────────────────────

    def progress_monitor(self, proc: subprocess.Popen, stop: threading.Event) -> None:
        if proc.stdout is not None:
            threading.Thread(
                target=lambda: [_ for _ in iter(proc.stdout.readline, b"")], daemon=True
            ).start()

        cwd_path: Path | None = getattr(proc, "_cwd", None)
        t0 = time.time()
        geo, geo_seen = 0, 0
        scf, scf_seen = 0, 0
        last_size = 0
        castep_file: Path | None = None

        def _render() -> None:
            width = shutil.get_terminal_size().columns or 80
            scf_denom = max(scf_seen + 2, 5)
            pct = min(100, int(scf * 100 / scf_denom))
            filled = pct * 30 // 100
            bar = "█" * filled + "░" * (30 - filled)
            el = int(time.time() - t0)
            mm, ss_v = divmod(el, 60)
            scf_str = f"{scf}/{scf_seen}" if scf_seen >= 1 else f"{scf}/—"

            line = f"  │  [{bar}] {pct:3d}%  geo {geo}  scf {scf_str}  ⏱ {mm:02d}:{ss_v:02d}"
            if len(line) > width - 1:
                line = line[: width - 4] + "..."
            print(f"\r{line.ljust(width - 1)}", end="", flush=True)

        if cwd_path is not None:
            deadline = time.monotonic() + 15.0
            while not stop.is_set() and time.monotonic() < deadline:
                candidates = list(cwd_path.glob("*.castep"))
                if candidates:
                    castep_file = max(candidates, key=lambda p: p.stat().st_mtime)
                    break
                time.sleep(0.5)

        while not stop.is_set():
            if castep_file is None or not castep_file.exists():
                _render()
                time.sleep(1.0)
                continue

            try:
                size = castep_file.stat().st_size
                if size == last_size:
                    _render()
                    time.sleep(0.8)
                    continue

                text = read_tail(castep_file, max_bytes=2 * 1024 * 1024)
                last_size = size

                for line in text.splitlines():
                    if "LBFGS: finished iteration" in line:
                        parts = line.split()
                        try:
                            geo = int(parts[parts.index("iteration") + 1])
                            geo_seen = max(geo_seen, geo)
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
                _render()
                time.sleep(0.8)
            except FileNotFoundError:
                time.sleep(0.5)
            except Exception:
                time.sleep(1.0)

        print(
            "\r" + " " * (shutil.get_terminal_size().columns or 80),
            end="\r",
            flush=True,
        )
