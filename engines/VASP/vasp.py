"""
engines/VASP/vasp.py  —  VASP DFT engine.
═══════════════════════════════════════════════════════════════
Registered as "vasp" via @register_engine decorator.
Elastic strategy: internal IBRION=6 (ElasticCapable).
Watchdog via OUTCAR error inspection (WatchdogCapable).

VCA mode emits a stern MODE_WARNINGS advisory — VASP's INCAR `VCA = ...`
tag is qualitative at best. Use SQS for physical accuracy.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np  # F7 fix — was missing

import config as _cfg
import core_physics as _phys
from core_physics import Crystal
from engines.engine import (
    BaseEngine,
    ElasticCapable,
    EngineResult,
    ProgressBar,
    read_tail,
    register_engine,
)
from engines.VASP.POSCAR_INCAR import (
    parse_ibrion6_tensor,
    parse_outcar,
    write_incar,
    write_vca_poscar,
)


_KR_CONVERGENCE = "vasp_no_convergence"


@register_engine("vasp")
class VaspEngine(BaseEngine):
    """VASP DFT engine with native IBRION=6 elastic constants."""

    name = "vasp"
    output_suffix = "OUTCAR"
    subdir_name = _cfg.VASP_SUBDIR
    SUPPORTED_MODES = frozenset({"vca", "sqs", "direct"})

    MODE_WARNINGS = {
        "vca": (
            "VASP VCA via the INCAR 'VCA = ...' tag interpolates POTCARs "
            "site-by-site. This is NOT a quantum-mechanically rigorous "
            "mixture: it neglects local lattice distortions, charge "
            "transfer between mixed species, and short-range ordering. "
            "Results are qualitative trends only. For physical accuracy "
            "use --mode sqs (VASP supercell with explicit site occupation) "
            "or switch to CASTEP."
        ),
    }

    @property
    def _cleanup_globs(self) -> list[str]:
        return _cfg.VASP_CLEANUP_GLOBS

    def __init__(
        self,
        engine_cmd: str,
        potcar_dir: str | Path,
        vaspkit_cmd: str | None = None,
        cutoff: int = _cfg.ENCUT_SOFT,
        spin: bool = False,
        smearing: float = _cfg.SMEARING_VCA,
    ) -> None:
        self.engine_cmd = engine_cmd
        self.potcar_dir = Path(potcar_dir) if potcar_dir else Path("")
        self.vaspkit_cmd = str(vaspkit_cmd) if vaspkit_cmd else None
        self.cutoff = int(cutoff)
        self.spin = bool(spin)
        self.smearing = float(smearing)

    # ── Setup ─────────────────────────────────────────────────────────────────

    @classmethod
    def setup_interactive(
        cls,
        src: Path,
        crystal: Crystal,
        override_cmd: str | None,
        args: argparse.Namespace,
    ) -> tuple[VaspEngine, str]:
        import ui
        ui.section("Engine execution (VASP)")

        bin_path = cls._resolve_binary(
            _cfg.VASP_SEARCH_PATHS, override_cmd, "vasp_std"
        )
        n_procs = cls._resolve_mpi_procs(args)
        cmd = cls._build_mpi_cmd(bin_path, n_procs)

        # POTCAR directory + VASPkit (optional)
        potcar_dir_found = cls.find_resource(_cfg.POTCAR_SEARCH_PATHS, is_dir=True)
        potcar_dir = str(potcar_dir_found) if potcar_dir_found else ""
        while not potcar_dir:
            ans = ui.ask_str("  PAW_PBE dir not found. Path to POTCARs: ").strip()
            if not ans:
                break
            potcar_dir = os.path.expanduser(ans)

        vk_found = cls.find_resource(
            _cfg.VASPKIT_SEARCH_PATHS, must_be_executable=True
        )
        vk = str(vk_found) if vk_found else None

        print(f"  Command : {cmd}")
        print(f"  POTCARs : {potcar_dir or 'Not found'}")
        print(f"  VASPkit : {vk or 'Not found (using internal parser)'}")

        # Parameter file caches user-chosen wizard answers.
        param_src = src.with_suffix(".vasp_param")
        cutoff = _cfg.ENCUT_SOFT
        spin = False
        smearing = _cfg.SMEARING_VCA

        if not param_src.exists():
            ui.section("Parameter setup (VASP)")
            answers = ui.render_wizard(cls.get_wizard_schema(crystal, is_vca=True))
            cutoff = int(answers["cutoff"])
            spin = bool(answers["spin"])
            smearing = float(answers["smearing"])
            param_src.write_text(
                json.dumps({"cutoff": cutoff, "spin": spin, "smearing": smearing}),
                encoding="utf-8",
            )
            print(f"\n  ✓ Written: {param_src.name}")
        else:
            print(f"  .vasp_param : {param_src.name}  (found)")
            try:
                p = json.loads(param_src.read_text())
                cutoff = p.get("cutoff", cutoff)
                spin = p.get("spin", spin)
                smearing = p.get("smearing", smearing)
            except (json.JSONDecodeError, OSError):
                pass

        engine = cls(
            engine_cmd=cmd,
            potcar_dir=potcar_dir,
            vaspkit_cmd=vk,
            cutoff=cutoff,
            spin=spin,
            smearing=smearing,
        )
        return engine, cmd

    # ── Wizard schema ─────────────────────────────────────────────────────────

    @classmethod
    def get_wizard_schema(
        cls, crystal: Crystal, is_vca: bool
    ) -> list[dict[str, Any]]:
        has_hard, has_mag = cls._detect_species_flags(crystal)
        def_cut = _cfg.ENCUT_HARD if has_hard else _cfg.ENCUT_SOFT
        def_smear = _cfg.SMEARING_VCA if is_vca else _cfg.SMEARING_SINGLE

        return [
            {
                "key": "cutoff", "type": "int",
                "label": "1/3  Plane-wave cutoff energy (ENCUT, eV)",
                "default": def_cut,
                "help": (
                    f"⚠ Hard elements detected: {def_cut}+ eV required."
                    if has_hard
                    else "400-500 eV is a good default."
                ),
            },
            {
                "key": "spin", "type": "bool",
                "label": "2/3  Spin polarization (ISPIN)",
                "default": has_mag,
                "help": (
                    "⚠ Magnetic elements detected — ISPIN=2 recommended."
                    if has_mag
                    else "Non-magnetic — no is correct."
                ),
            },
            {
                "key": "smearing", "type": "float",
                "label": "3/3  Fermi smearing width (SIGMA, eV)",
                "default": def_smear,
                "help": "0.10-0.20 eV helps convergence for VCA.",
            },
        ]

    # ── Core engine methods ───────────────────────────────────────────────────

    def write_input(
        self,
        dest_dir: Path,
        seed: str,
        crystal: Crystal,
    ) -> None:
        # The new Crystal object contains all site info.
        ord_elems, vca_weights = write_vca_poscar(
            dest_dir / "POSCAR", crystal
        )

        zval_map = self._build_potcar(dest_dir, ord_elems)

        # NELECT = total number of valence electrons in the cell.
        # Crystal.vec() gives average VEC per atom.
        nelect = crystal.vec() * crystal.num_atoms

        cpu = os.cpu_count() or 4
        ncore = max(1, int(cpu ** 0.5))

        write_incar(
            dest_dir / "INCAR",
            task_type="GeometryOptimization",
            cutoff=self.cutoff,
            spin=self.spin,
            smearing=self.smearing,
            vca_weights=vca_weights,
            nelect=nelect,
            ncore=ncore,
        )

    def parse_output(self, output_file: Path) -> EngineResult:
        return parse_outcar(output_file)

    # ── WatchdogCapable ───────────────────────────────────────────────────────

    def check_health(self, log_tail: str) -> str | None:
        if (
            "ZBRENT: fatal error" in log_tail
            or "EDDDAV: Call to ZHEGV failed" in log_tail
        ):
            return _KR_CONVERGENCE
        return None

    # ── ElasticCapable ────────────────────────────────────────────────────────
    # NOTE: This still uses subprocess.run directly, NOT run_process (W25).
    # Reason: run_process is in orchestrator.py and would create a circular
    # import. Proper fix is to expose run_process in engines.engine OR move
    # elastic execution back into orchestrator. The architecture document
    # already flags this — see Round 2, W25. Implementation deferred until
    # the orchestrator.run_process boundary is settled.

    def run_elastic(
        self,
        step_dir: Path,
        seed: str,
        x: float,
        species: list[tuple[str, float]],
        nonmetal: str | None,
        density_gcm3: float | None,
        volume_ang3: float | None,
    ) -> dict[str, str]:
        t0 = time.monotonic()

        contcar = step_dir / "CONTCAR"
        if not contcar.exists():
            return {"_elastic_error": "CONTCAR not found. Geom optimization failed?"}

        shutil.copy2(contcar, step_dir / "POSCAR")

        # Reuse VCA weights and NELECT from the master INCAR.
        incar_text = (step_dir / "INCAR").read_text(
            encoding="utf-8", errors="replace"
        )
        m_vca = re.search(r"VCA\s*=\s*(.*)", incar_text)
        vca_weights = (
            [float(w) for w in m_vca.group(1).split()] if m_vca else []
        )
        m_nelect = re.search(r"NELECT\s*=\s*([\d.]+)", incar_text)
        nelect = float(m_nelect.group(1)) if m_nelect else None

        write_incar(
            step_dir / "INCAR",
            task_type="ElasticIBRION6",
            cutoff=self.cutoff,
            spin=self.spin,
            smearing=_cfg.SMEARING_SINGLE,
            vca_weights=vca_weights,
            nelect=nelect,
            ncore=0,
        )

        try:
            subprocess.run(
                self.engine_cmd, shell=True, cwd=step_dir,
                check=True, capture_output=True,
                timeout=_cfg.STEP_TIMEOUT_S,
            )
        except subprocess.CalledProcessError as e:
            return {
                "_elastic_error": (
                    f"VASP crashed: {e.stderr.decode(errors='replace')[:500]}"
                )
            }
        except subprocess.TimeoutExpired:
            return {"_elastic_error": "VASP elastic timed out."}
        except OSError as e:
            return {"_elastic_error": f"OS error running VASP: {e}"}

        outcar = step_dir / "OUTCAR"
        if not outcar.exists():
            return {"_elastic_error": "OUTCAR not found."}

        text = read_tail(outcar, 2 * 1024 * 1024)
        C_GPa = parse_ibrion6_tensor(text)
        if C_GPa is None or C_GPa.shape != (6, 6):
            return {"_elastic_error": "Elastic tensor not found in OUTCAR."}

        c11 = float(C_GPa[0, 0])
        c12 = float(C_GPa[0, 1])
        c44 = float(C_GPa[3, 3])

        props = _phys.cubic_vrh(
            c11, c12, c44, density=density_gcm3, vol=volume_ang3
        )
        if not props.get("born_stable", False):
            return {
                "_elastic_error": "Born stability violated.",
                "C11": f"{c11:.4f}",
                "C12": f"{c12:.4f}",
                "C44": f"{c44:.4f}",
                "born_stable": "no",
            }

        result: dict[str, str] = {
            k: f"{v:.4f}" for k, v in props.items() if isinstance(v, float)
        }
        result.update({
            "C11": f"{c11:.4f}", "C12": f"{c12:.4f}", "C44": f"{c44:.4f}",
            "born_stable": "yes",
            "elastic_source": "VASP-IBRION6",
            "elastic_n_points": "1",
            "elastic_R2_min": "N/A",
            "elastic_wall_time_s": f"{time.monotonic() - t0:.0f}",
        })

        self.cleanup(step_dir)
        return result

    # ── POTCAR builder ────────────────────────────────────────────────────────

    def _build_potcar(
        self, dest_dir: Path, ord_elems: list[str]
    ) -> dict[str, float]:
        """Build POTCAR, return {element → ZVAL} map for NELECT calculation."""
        if self.vaspkit_cmd:
            try:
                proc = subprocess.run(
                    f"echo '103' | {self.vaspkit_cmd}",
                    shell=True, cwd=dest_dir, timeout=60,
                    capture_output=True,
                )
                if proc.returncode == 0 and (dest_dir / "POTCAR").exists():
                    text = (dest_dir / "POTCAR").read_text(
                        encoding="utf-8", errors="replace"
                    )
                    zvals = re.findall(r"ZVAL\s*=\s*([\d.]+)", text)
                    return {el: float(z) for el, z in zip(ord_elems, zvals)}
            except (OSError, subprocess.TimeoutExpired):
                pass

        zval_map: dict[str, float] = {}
        potcar_chunks: list[str] = []
        for el in ord_elems:
            sub_dir = _cfg.POTCAR_PREFERRED.get(el.capitalize(), el.capitalize())
            p_path = self.potcar_dir / sub_dir / "POTCAR"
            if not p_path.exists():
                p_path = self.potcar_dir / el.capitalize() / "POTCAR"
                if not p_path.exists():
                    raise FileNotFoundError(
                        f"POTCAR not found for {el} in {self.potcar_dir}"
                    )
            text = p_path.read_text(encoding="utf-8", errors="replace")
            potcar_chunks.append(text)
            m = re.search(r"ZVAL\s*=\s*([\d.]+)", text)
            zval_map[el] = float(m.group(1)) if m else 0.0

        (dest_dir / "POTCAR").write_text("".join(potcar_chunks), encoding="utf-8")
        return zval_map

    # ── Progress monitor (shared ProgressBar) ─────────────────────────────────

    def progress_monitor(
        self,
        proc: subprocess.Popen,
        stop: threading.Event,
        cwd: Path,
    ) -> None:
        bar = ProgressBar()
        ionic = 0
        scf = 0
        nsw = _cfg.NSW_MAX_VASP
        disp_cur = 0
        disp_tot = 0

        def _render() -> None:
            if disp_tot > 0:
                pct = disp_cur / max(disp_tot, 1)
                content = (
                    f"[{bar.bar(pct)}] {int(pct * 100):3d}%  "
                    f"displ {disp_cur}/{disp_tot}  scf {scf}  ⏱ {bar.elapsed()}"
                )
            else:
                pct = ionic / max(nsw, 1)
                content = (
                    f"[{bar.bar(pct)}] {int(pct * 100):3d}%  "
                    f"ionic {ionic}/{nsw}  scf {scf}  ⏱ {bar.elapsed()}"
                )
            bar.render(content)

        try:
            if proc.stdout:
                for raw in iter(proc.stdout.readline, b""):
                    if stop.is_set():
                        break
                    ln = raw.decode(errors="replace").rstrip()
                    if ln.strip().startswith(("DAV:", "RMM:")):
                        try:
                            scf = int(ln.split()[1])
                        except (IndexError, ValueError):
                            pass
                    elif ln.strip() and ln.strip()[0].isdigit() and "F=" in ln:
                        try:
                            ionic = int(ln.split()[0])
                            scf = 0
                        except (IndexError, ValueError):
                            pass
                    elif "Total:" in ln:
                        m = re.search(r"Total:\s*(\d+)/\s*(\d+)", ln)
                        if m:
                            disp_cur, disp_tot = int(m.group(1)), int(m.group(2))
                    elif "NSW" in ln and "=" in ln:
                        m = re.search(r"NSW\s*=\s*(\d+)", ln)
                        if m:
                            nsw = max(1, int(m.group(1)))
                    _render()
        except (OSError, ValueError):
            pass
        finally:
            ProgressBar.clear()
