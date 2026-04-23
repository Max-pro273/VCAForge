"""
engines/CASTEP/castep.py  —  CASTEP Engine Implementation.
═══════════════════════════════════════════════════════════
Registered as "castep" via @register_engine decorator.
Elastic strategy: finite-strain fallback driven by orchestrator.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

import config as _cfg
from core_physics import Crystal, nextra_bands_for, load_crystal
from engines.engine import EngineResult, find_engine_binary, register_engine, read_tail

from engines.CASTEP.cell_param import (
    parse_elastic_file,
    parse_output,
    patch_nextra,
    write_engine_params,
    write_vca_cell,
)

def _species_vec(species_mix: list[tuple[str, float]]) -> float:
    from core_physics import vec_for_system
    return vec_for_system(species_mix)

def _wizard_engine_cmd(override_cmd: str | None) -> str:
    import ui
    import glob
    if override_cmd: return override_cmd

    cpu = os.cpu_count() or 4
    ui.section("Engine execution (CASTEP)")

    bin_path = None
    # Smart discovery using config paths
    for pat in _cfg.CASTEP_SEARCH_PATHS:
        # Check standard PATH first
        if pat in ("castep.mpi", "castep"):
            w = shutil.which(pat)
            if w:
                bin_path = w
                break
        else:
            # Glob search for absolute/relative paths
            expanded = os.path.expanduser(pat)
            matches = glob.glob(expanded, recursive=True)
            for m in matches:
                if os.path.isfile(m) and os.access(m, os.X_OK):
                    bin_path = m
                    break
        if bin_path: break

    if not bin_path:
        print("  ⚠  CASTEP executable not found automatically.")
        ans = ui.ask_str("  Provide absolute path to castep.mpi (or 'skip'): ").strip()
        if ans.lower() == "skip": return ""
        bin_path = os.path.expanduser(ans)
    else:
        print(f"  ✓  Found executable: {bin_path}")

    raw = ui.ask_str(f"  MPI processes [{cpu}]: ", str(cpu))
    try: n = max(1, int(raw))
    except ValueError: n = cpu

    name = Path(bin_path).name
    cmd = f"mpirun -n {n} {bin_path} {{seed}}" if "mpi" in name.lower() else f"{bin_path} {{seed}}"
    return cmd

def _wizard_param(src_file: Path, crystal: Crystal, species_list: list[str], is_vca: bool) -> Path:
    import ui
    param_path = src_file.with_suffix(".param")
    if param_path.exists():
        print(f"  .param : {param_path.name}  (found)")
        return param_path

    schema = CastepEngine.get_wizard_schema(crystal, is_vca)
    ui.section("Parameter setup (CASTEP)")
    answers = ui.render_wizard(schema)

    # We use a simple heuristic for nextra bands during initial setup
    has_d = any(_cfg.ELEMENTS.get(s.capitalize(), {}).get("Z", 0) > 20 for s in species_list)
    nextra = (20 if has_d else 10) if is_vca else 10

    write_engine_params(
        param_path,
        task_type=answers["task"],
        xc=answers["xc"],
        cutoff=answers["cutoff"],
        spin=answers["spin"],
        nextra=nextra,
        smearing=answers["smearing"],
    )
    return param_path


@register_engine("castep")
class CastepEngine:
    """CASTEP DFT engine — finite-strain elastic fallback."""

    name: str = "castep"
    output_suffix: str = ".castep"
    subdir_name: str = _cfg.CASTEP_SUBDIR
    _cleanup_globs = _cfg.CASTEP_CLEANUP_GLOBS

    def __init__(self, engine_cmd: str, param_src: Path | str) -> None:
        self.engine_cmd = engine_cmd
        self.param_src = Path(param_src)

    @classmethod
    def setup_interactive(
        cls,
        src: Path,
        crystal: Crystal,
        override_cmd: str | None = None,
    ) -> tuple["CastepEngine", str]:
        species_list = list(dict.fromkeys(crystal.species))
        param_src = _wizard_param(src, crystal, species_list, is_vca=True)
        engine_cmd = _wizard_engine_cmd(override_cmd)
        return cls(engine_cmd, param_src), engine_cmd

    def write_input(
        self,
        dest_dir: Path,
        seed: str,
        crystal: Crystal,
        species_mix: list[tuple[str, float]],
        x: float,
    ) -> None:
        tmpl_elem = species_mix[0][0]
        target_mix = {tmpl_elem: 1.0 - x}
        for e, f in species_mix[1:]:
            target_mix[e] = f * x

        write_vca_cell(dest_dir / f"{seed}.cell", crystal, tmpl_elem, target_mix)
        dest_param = dest_dir / f"{seed}.param"
        shutil.copy2(self.param_src, dest_param)
        patch_nextra(dest_param, nextra_bands_for(x, _species_vec(species_mix)))

    def parse_output(self, output_file: Path) -> EngineResult:
        return parse_output(output_file)

    def parse_extra_outputs(self, step_dir: Path, seed: str) -> dict:
        ep = step_dir / f"{seed}.elastic"
        return parse_elastic_file(ep) if ep.exists() else {}

    def progress_monitor(self, proc: subprocess.Popen, stop: threading.Event) -> None:
        if proc.stdout is not None:
            threading.Thread(target=lambda: [_ for _ in iter(proc.stdout.readline, b"")], daemon=True).start()

        cwd_path: Path | None = getattr(proc, "_cwd", None)
        t0 = time.time()
        geo, geo_seen = 0, 0
        scf, scf_seen = 0, 0
        last_size = 0
        castep_file: Path | None = None

        def _render() -> None:
            scf_denom = max(scf_seen + 2, 5)
            pct = min(100, int(scf * 100 / scf_denom))
            filled = pct * 30 // 100
            bar = "█" * filled + "░" * (30 - filled)
            el = int(time.time() - t0)
            mm, ss_v = divmod(el, 60)
            scf_str = f"{scf}/{scf_seen}" if scf_seen >= 1 else f"{scf}/—"
            print(f"\r  │  [{bar}] {pct:3d}%  geo {geo}  scf {scf_str}  ⏱ {mm:02d}:{ss_v:02d}    ", end="", flush=True)

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

                # MEMORY FIX: Only read the last 2MB to prevent RAM crashes on long runs
                text = read_tail(castep_file, max_bytes=2 * 1024 * 1024)
                last_size = size

                for line in text.splitlines():
                    if "LBFGS: finished iteration" in line:
                        parts = line.split()
                        try:
                            geo = int(parts[parts.index("iteration") + 1])
                            geo_seen = max(geo_seen, geo)
                            scf = 0
                        except (ValueError, IndexError): pass
                    elif "<-- SCF" in line:
                        s = line.lstrip()
                        if s and s[0].isdigit():
                            try:
                                scf = int(s.split()[0])
                                scf_seen = max(scf_seen, scf)
                            except ValueError: pass
                _render()
                time.sleep(0.8)
            except FileNotFoundError:
                time.sleep(0.5)
            except Exception as e:
                print(f"\n  [Monitor Error] {e}")
                time.sleep(1.0)

        print("\r" + " " * 90, end="\r", flush=True)

    @classmethod
    def get_wizard_schema(cls, crystal: Crystal, is_vca: bool) -> list[dict]:
        species = list(dict.fromkeys(crystal.species))
        has_hard = any(_cfg.ELEMENTS.get(s.capitalize(), {}).get("hard", False) for s in species)
        has_mag = any(_cfg.ELEMENTS.get(s.capitalize(), {}).get("mag", False) for s in species)
        rec_cut = _cfg.ENCUT_HARD if has_hard else _cfg.ENCUT_SOFT
        rec_smear = _cfg.SMEARING_VCA if is_vca else _cfg.SMEARING_SINGLE

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
            crystal.clear_cache() # Safe symmetric reset
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
        F = np.array([
            [1 + e11,   e12,   e13],
            [   e12, 1 + e22,  e23],
            [   e13,   e23, 1 + e33],
        ])

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
            vegard=False # Vegard already applied to base relaxed cell
        )

        param_text = self.param_src.read_text(encoding="utf-8", errors="replace")
        m_xc = re.search(r"xc_functional\s*:\s*(\S+)", param_text, re.I)
        m_cut = re.search(r"cut_off_energy\s*:\s*(\d+)", param_text, re.I)

        if not m_xc or not m_cut:
            raise ValueError(f"Missing xc_functional or cut_off_energy in {self.param_src}")

        write_engine_params(
            dest_dir / f"{seed}.param",
            task_type="SinglePoint",
            xc=m_xc.group(1),
            cutoff=int(m_cut.group(1)),
            spin=False,
            nextra=nextra_bands_for(x, _species_vec(species_mix)),
            smearing=_cfg.SMEARING_SINGLE,
        )

    def parse_stress_tensor(self, output_file: Path) -> np.ndarray:
        if not output_file.exists():
            raise FileNotFoundError(f"Stress tensor output not found: {output_file}")

        text = read_tail(output_file, max_bytes=2 * 1024 * 1024)

        # Format 1 — GeomOpt: three rows with scientific notation ending in "<-- S"
        #   -1.234567E+002  0.000000E+000  0.000000E+000  <-- S
        geomopt_blocks = re.findall(
            r"^\s*([-+]?\d+\.\d+[Ee][+-]?\d+)\s+"
            r"([-+]?\d+\.\d+[Ee][+-]?\d+)\s+"
            r"([-+]?\d+\.\d+[Ee][+-]?\d+)\s+<-- S",
            text, re.M,
        )
        if len(geomopt_blocks) >= 3:
            m = np.array([[float(v) for v in r] for r in geomopt_blocks[-3:]])
            return np.array([m[0, 0], m[1, 1], m[2, 2], m[1, 2], m[0, 2], m[0, 1]])

        # Format 2 — SinglePoint: rows inside the "Symmetrised Stress Tensor" box
        #  *  x      0.046525      0.000000      0.000000  *
        #  *  y      0.000000      0.046525     -0.000000  *
        #  *  z      0.000000     -0.000000      0.046525  *
        # Take the LAST such block in the file (final SCF).
        sp_blocks = re.findall(
            r"Symmetrised Stress Tensor.*?"
            r"\*\s+x\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s*\*\s*"
            r"\*\s+y\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s*\*\s*"
            r"\*\s+z\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s*\*",
            text, re.S,
        )
        if sp_blocks:
            v = [float(x) for x in sp_blocks[-1]]
            # v = [xx, xy, xz, yx, yy, yz, zx, zy, zz]
            return np.array([v[0], v[4], v[8], v[5], v[2], v[1]])

        raise ValueError(
            "Stress tensor not found in CASTEP output — "
            "neither GeomOpt (<-- S) nor SinglePoint (Symmetrised Stress Tensor) "
            "format matched. SCF may have failed."
        )

    def cleanup(self, step_dir: Path) -> None:
        for glob_pat in self._cleanup_globs:
            for f in step_dir.glob(glob_pat):
                try: f.unlink()
                except OSError: pass
