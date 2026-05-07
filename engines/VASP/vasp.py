"""
engines/VASP/vasp.py  —  VASP Engine Implementation.
═════════════════════════════════════════════════════
Registered as "vasp" via @register_engine decorator.
Elastic strategy: internal IBRION=6 via ElasticCapable.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import config as _cfg
import core_physics as _phys
from core_physics import Crystal
from engines.engine import (
    BaseEngine,
    ElasticCapable,
    EngineResult,
    WatchdogCapable,
    read_tail,
    register_engine,
)
from engines.VASP.POSCAR_INCAR import (
    parse_outcar,
    write_engine_params,
    write_vca_poscar,
)

_KR_CONVERGENCE = "vasp_no_convergence"


def _parse_ibrion6_tensor(text: str) -> np.ndarray | None:
    """Parse VASP IBRION=6 elastic tensor from OUTCAR text."""
    m = re.search(
        r"TOTAL ELASTIC MODULI \(kBar\)\s+Direction[^\n]+\n[^\n]+\n(.*?)(?:\n\s*\n|---)",
        text,
        re.DOTALL,
    )
    if not m:
        return None
    rows = []
    for line in m.group(1).splitlines():
        parts = line.split()
        if len(parts) >= 7:
            try:
                rows.append([float(x) for x in parts[1:7]])
            except ValueError:
                pass
    if len(rows) >= 6:
        return np.array(rows[:6]) / 10.0  # kBar to GPa
    return None


@register_engine("vasp")
class VaspEngine(BaseEngine):
    """VASP DFT engine with native IBRION=6 elastic constants."""

    name = "vasp"
    output_suffix = "OUTCAR"
    subdir_name = _cfg.VASP_SUBDIR
    SUPPORTED_MODES = frozenset({"vca", "sqs", "direct"})
    _cleanup_globs = getattr(_cfg, "VASP_CLEANUP_GLOBS", [])

    def __init__(
        self,
        engine_cmd: str,
        potcar_dir: str | Path,
        vaspkit_cmd: str | None = None,
        cutoff: int = getattr(_cfg, "ENCUT_SOFT", 400),
        spin: bool = False,
        smearing: float = getattr(_cfg, "SMEARING_VCA", 0.2),
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
    ) -> tuple["VaspEngine", str]:
        import ui

        cpu = os.cpu_count() or 4
        ui.section("Engine execution (VASP)")

        # 1. Binary
        bin_path = (
            override_cmd
            if override_cmd
            else cls.find_resource(_cfg.VASP_SEARCH_PATHS, must_be_executable=True)
        )

        while not bin_path:
            print("  ✗  vasp_std binary not found automatically.")
            ans = ui.ask_str("  Path to vasp_std (or 'skip'): ").strip()
            if ans.lower() == "skip":
                bin_path = ""
                break
            bin_path = os.path.expanduser(ans)

        cores = getattr(args, "cores", None)
        if cores is not None:
            n = max(1, cores)
            print(f"  MPI processes : {n}  (from --cores)")
        else:
            n_procs = ui.ask_str(f"  MPI processes [{cpu}]: ", str(cpu))
            try:
                n = max(1, int(n_procs))
            except ValueError:
                n = cpu

        cmd = (
            f"mpirun -n {n} {bin_path}"
            if n > 1 and bin_path and "mpi" not in bin_path
            else (bin_path or "")
        )

        # 2. POTCAR Dir
        potcar_dir = cls.find_resource(_cfg.POTCAR_SEARCH_PATHS, is_dir=True)
        while not potcar_dir:
            ans = ui.ask_str("  PAW_PBE dir not found. Path to POTCARs: ").strip()
            if not ans:
                break
            potcar_dir = os.path.expanduser(ans)

        # 3. VASPkit
        vk = cls.find_resource(_cfg.VASPKIT_SEARCH_PATHS, must_be_executable=True)

        print(f"  Command : {cmd}")
        print(f"  POTCARs : {potcar_dir or 'Not found'}")
        print(f"  VASPkit : {vk or 'Not found (using internal parser)'}")

        # 4. Parameters
        species_list = list(dict.fromkeys(crystal.species))
        param_src = src.with_suffix(".vasp_param")

        cutoff = getattr(_cfg, "ENCUT_SOFT", 400)
        spin = False
        smearing = getattr(_cfg, "SMEARING_VCA", 0.2)

        if not param_src.exists():
            schema = cls.get_wizard_schema(crystal, is_vca=True)
            ui.section("Parameter setup (VASP)")
            answers = ui.render_wizard(schema)

            cutoff = int(answers["cutoff"])
            spin = bool(answers["spin"])
            smearing = float(answers["smearing"])

            data = {"cutoff": cutoff, "spin": spin, "smearing": smearing}
            param_src.write_text(json.dumps(data), encoding="utf-8")
            print(f"\n  ✓ Written: {param_src.name}")
        else:
            print(f"  .vasp_param : {param_src.name}  (found)")
            try:
                p = json.loads(param_src.read_text())
                cutoff = p.get("cutoff", cutoff)
                spin = p.get("spin", spin)
                smearing = p.get("smearing", smearing)
            except Exception:
                pass

        engine = cls(
            engine_cmd=cmd,
            potcar_dir=potcar_dir or "",
            vaspkit_cmd=vk,
            cutoff=cutoff,
            spin=spin,
            smearing=smearing,
        )
        return engine, cmd

    # ── Wizard Schema ─────────────────────────────────────────────────────────

    @classmethod
    def get_wizard_schema(cls, crystal: Crystal, is_vca: bool) -> list[dict]:
        species = list(dict.fromkeys(crystal.species)) if crystal else []
        has_hard = any(
            _cfg.ELEMENTS.get(s.capitalize(), {}).get("hard", False) for s in species
        )
        has_mag = any(
            _cfg.ELEMENTS.get(s.capitalize(), {}).get("mag", False) for s in species
        )
        def_cut = (
            getattr(_cfg, "ENCUT_HARD", 520)
            if has_hard
            else getattr(_cfg, "ENCUT_SOFT", 400)
        )
        def_smear = (
            getattr(_cfg, "SMEARING_VCA", 0.2)
            if is_vca
            else getattr(_cfg, "SMEARING_SINGLE", 0.1)
        )

        return [
            {
                "key": "cutoff",
                "label": "1/3  Plane-wave cutoff energy (ENCUT, eV)",
                "type": "int",
                "default": def_cut,
                "help": f"{'⚠ Hard elements detected: ' + str(def_cut) + '+ eV required.' if has_hard else '400-500 eV is a good default.'}",
            },
            {
                "key": "spin",
                "label": "2/3  Spin polarization (ISPIN)",
                "type": "bool",
                "default": has_mag,
                "help": f"{'⚠ Magnetic elements detected — ISPIN=2 recommended.' if has_mag else 'Non-magnetic — no is correct.'}",
            },
            {
                "key": "smearing",
                "label": "3/3  Fermi smearing width (SIGMA, eV)",
                "type": "float",
                "default": def_smear,
                "help": "0.10-0.20 eV helps convergence for VCA",
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
        ord_elems, vca_weights = write_vca_poscar(
            dest_dir / "POSCAR", scaled, template_element, target_mix
        )

        zval_map = self._build_potcar(dest_dir, ord_elems)

        poscar_text = (dest_dir / "POSCAR").read_text(encoding="utf-8")
        counts = [int(c) for c in poscar_text.splitlines()[6].split()]
        nelect = sum(
            zval_map.get(el, 0.0) * count * weight
            for el, count, weight in zip(ord_elems, counts, vca_weights)
        )

        ncore = int(multiprocessing.cpu_count() ** 0.5)
        write_engine_params(
            dest_dir / "INCAR",
            task_type="GeometryOptimization",
            xc=getattr(_cfg, "XC_DEFAULT", "PBE"),
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
        """Evaluate VASP OUTCAR tail for critical failures."""
        if (
            "ZBRENT: fatal error" in log_tail
            or "EDDDAV: Call to ZHEGV failed" in log_tail
        ):
            return _KR_CONVERGENCE
        return None

    # ── ElasticCapable ────────────────────────────────────────────────────────

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
            return {
                "_elastic_error": "CONTCAR not found. Geometry optimization failed?"
            }

        shutil.copy2(contcar, step_dir / "POSCAR")

        incar_text = (step_dir / "INCAR").read_text(encoding="utf-8", errors="replace")
        vca_weights: list[float] = []
        m_vca = re.search(r"VCA\s*=\s*(.*)", incar_text)
        if m_vca:
            vca_weights = [float(w) for w in m_vca.group(1).split()]

        m_nelect = re.search(r"NELECT\s*=\s*([\d.]+)", incar_text)
        nelect = float(m_nelect.group(1)) if m_nelect else None

        write_engine_params(
            step_dir / "INCAR",
            task_type="ElasticIBRION6",
            xc=getattr(_cfg, "XC_DEFAULT", "PBE"),
            cutoff=self.cutoff,
            spin=self.spin,
            smearing=getattr(_cfg, "SMEARING_SINGLE", 0.1),
            vca_weights=vca_weights,
            nelect=nelect,
            ncore=0,
        )

        try:
            subprocess.run(
                self.engine_cmd,
                shell=True,
                cwd=step_dir,
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as e:
            return {
                "_elastic_error": f"VASP crash: {e.stderr.decode(errors='replace')}"
            }
        except OSError as e:
            return {"_elastic_error": f"OS Error running VASP: {e}"}

        outcar = step_dir / "OUTCAR"
        if not outcar.exists():
            return {"_elastic_error": "OUTCAR not found."}

        text = read_tail(outcar, 2 * 1024 * 1024)
        C_GPa = _parse_ibrion6_tensor(text)

        if C_GPa is None or C_GPa.shape != (6, 6):
            return {"_elastic_error": "Elastic tensor not found in OUTCAR."}

        c11, c12, c44 = float(C_GPa[0, 0]), float(C_GPa[0, 1]), float(C_GPa[3, 3])

        result: dict[str, str] = {}
        props = _phys.cubic_vrh(c11, c12, c44, density=density_gcm3, vol=volume_ang3)

        if not props.get("born_stable", False):
            return {"_elastic_error": "Born stability violated."}

        result.update({k: f"{v:.4f}" for k, v in props.items() if isinstance(v, float)})
        result.update({"C11": f"{c11:.4f}", "C12": f"{c12:.4f}", "C44": f"{c44:.4f}"})
        result["elastic_source"] = "VASP-IBRION6"
        result["elastic_n_points"] = "1"
        result["elastic_R2_min"] = "N/A"
        result["elastic_wall_time_s"] = f"{time.monotonic() - t0:.0f}"

        self.cleanup(step_dir)
        return result

    # ── POTCAR builder ────────────────────────────────────────────────────────

    def _build_potcar(self, dest_dir: Path, ord_elems: list[str]) -> dict[str, float]:
        if self.vaspkit_cmd:
            try:
                proc = subprocess.run(
                    f"echo '103' | {self.vaspkit_cmd}",
                    shell=True,
                    cwd=dest_dir,
                    capture_output=True,
                    text=True,
                    timeout=60,
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
        potcar_content = []
        for el in ord_elems:
            sub_dir = getattr(_cfg, "POTCAR_PREFERRED", {}).get(
                el.capitalize(), el.capitalize()
            )
            p_path = self.potcar_dir / sub_dir / "POTCAR"
            if not p_path.exists():
                p_path = self.potcar_dir / el.capitalize() / "POTCAR"
                if not p_path.exists():
                    raise FileNotFoundError(
                        f"POTCAR not found for {el} in {self.potcar_dir}"
                    )
            text = p_path.read_text(encoding="utf-8", errors="replace")
            potcar_content.append(text)
            m = re.search(r"ZVAL\s*=\s*([\d.]+)", text)
            zval_map[el] = float(m.group(1)) if m else 0.0

        (dest_dir / "POTCAR").write_text("".join(potcar_content), encoding="utf-8")
        return zval_map

    # ── Progress Monitor ──────────────────────────────────────────────────────

    def progress_monitor(self, proc: subprocess.Popen, stop: threading.Event) -> None:
        _BAR = 22
        t0 = time.time()
        ionic, scf, nsw = 0, 0, 200
        disp_cur, disp_tot = 0, 0

        def _render() -> None:
            width = shutil.get_terminal_size().columns or 80
            el = time.time() - t0
            mm, ss = divmod(int(el), 60)

            if disp_tot > 0:
                pct = min(disp_cur / disp_tot, 1.0)
                filled = int(pct * _BAR)
                bar = "█" * filled + "░" * (_BAR - filled)
                line = f"  │  [{bar}] {int(pct * 100):3d}%  displ {disp_cur}/{disp_tot}  scf {scf}  ⏱ {mm:02d}:{ss:02d}"
            else:
                pct = min(ionic / max(nsw, 1), 1.0)
                filled = int(pct * _BAR)
                bar = "█" * filled + "░" * (_BAR - filled)
                line = f"  │  [{bar}] {int(pct * 100):3d}%  ionic {ionic}/{nsw}  scf {scf}  ⏱ {mm:02d}:{ss:02d}"

            if len(line) > width - 1:
                line = line[: width - 4] + "..."
            print(f"\r{line.ljust(width - 1)}", end="", flush=True)

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
            print(
                "\r" + " " * (shutil.get_terminal_size().columns or 80),
                end="\r",
                flush=True,
            )
