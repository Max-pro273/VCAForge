"""
engines/VASP/vasp.py  —  VASP Engine Implementation.
═════════════════════════════════════════════════════
Registered as "vasp" via @register_engine decorator.

Elastic strategy: internal (run_internal_elastic present).
Orchestrator detects this and delegates the full elastic workflow.
"""

from __future__ import annotations

import glob
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
from engines.engine import EngineResult, find_engine_binary, register_engine

from engines.VASP.POSCAR_INCAR import (
    parse_ibrion6_tensor,
    parse_outcar,
    write_engine_params,
    write_vca_poscar,
)

# ─────────────────────────────────────────────────────────────────────────────
# Interactive Setup Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_path(paths: list[str]) -> str | None:
    """Safely expands ~ and evaluates globs to find the first matching path."""
    for p in paths:
        expanded = os.path.expanduser(p)
        # If it's a simple command like "vasp_std" with no slashes, use shutil
        if "/" not in expanded and not expanded.startswith("~"):
            if w := shutil.which(expanded):
                return w
        # Otherwise, evaluate as a glob
        hits = sorted(glob.glob(expanded))
        if hits:
            # Check if the found hit is an executable file or a directory (for POTCARs)
            for hit in reversed(hits): # Prefer the last sorted hit (often highest version)
                if Path(hit).is_file() and os.access(hit, os.X_OK):
                    return hit
                elif Path(hit).is_dir():
                    return hit
    return None

def _cmd_is_valid(cmd: str) -> bool:
    """Checks if the actual binary in the command string exists and is executable."""
    if not cmd:
        return False

    # Extract the actual binary.
    # E.g., from "mpirun -n 4 /path/to/vasp_std", we want the last part,
    # or just the single word if it's "vasp_std".
    parts = cmd.split()

    # Simple heuristic: if 'mpirun' or 'mpiexec' is used, the real binary is usually at the end.
    # Otherwise, assume the first word is the binary.
    binary_str = parts[-1] if parts[0] in ("mpirun", "mpiexec") else parts[0]

    binary = os.path.expanduser(binary_str)

    if Path(binary).is_absolute() or "/" in binary:
        return Path(binary).is_file() and os.access(binary, os.X_OK)
    return shutil.which(binary) is not None

def _wizard_vasp_cmd(override: str | None) -> tuple[str, Path, str | None]:
    import ui
    cpu = os.cpu_count() or 4

    bin_path = _find_path(_cfg.VASP_SEARCH_PATHS)

    ui.section("Engine execution (VASP)")
    print(f"  Machine : {cpu} logical cores")

    while not bin_path or not _cmd_is_valid(bin_path):
        print("  ✗  vasp_std binary not found automatically.")
        ans = ui.ask_str("  Path to vasp_std (or 'skip' for prepare-only): ").strip()
        if ans.lower() == "skip":
            return "", Path(""), None
        bin_path = os.path.expanduser(ans)

    n_procs = ui.ask_str(f"  MPI processes [{cpu}]: ", str(cpu))

    # Only wrap in mpirun if they actually want more than 1 proc AND didn't provide a full mpirun command themselves
    try:
        n = max(1, int(n_procs))
        cmd = f"mpirun -n {n} {bin_path}" if n > 1 and "mpi" not in bin_path else bin_path
    except ValueError:
        cmd = bin_path # Fallback if they typed something weird

    # Allow full command override from CLI args
    if override:
        cmd = override

    potcar_dir = _find_path(_cfg.POTCAR_SEARCH_PATHS)
    while not potcar_dir or not Path(potcar_dir).is_dir():
        ans = ui.ask_str("  PAW_PBE dir not found. Path to POTCARs: ").strip()
        potcar_dir = os.path.expanduser(ans)

    vk = _find_path(_cfg.VASPKIT_SEARCH_PATHS)

    print(f"  Command : {cmd}")
    print(f"  POTCARs : {potcar_dir}  ✓")
    print(f"  VASPkit : {vk or 'Not found (using internal parser)'}")

    return cmd, Path(potcar_dir), vk

def _wizard_param(src_file: Path, crystal: Crystal, species_list: list[str], is_vca: bool) -> Path:
    import ui
    param_path = src_file.with_suffix(".vasp_param")
    if param_path.exists():
        print(f"  .vasp_param : {param_path.name}  (found)")
        return param_path

    schema = VaspEngine.get_wizard_schema(crystal, is_vca)
    ui.section("Parameter setup (VASP)")
    print("  No .vasp_param found — answer quick questions to configure physics.")
    print("  Press Enter to accept the default shown in [brackets].\n")

    answers = ui.render_wizard(schema)

    data = {
        "cutoff": int(answers["cutoff"]),
        "spin": bool(answers["spin"]),
        "smearing": float(answers["smearing"]),
    }
    param_path.write_text(json.dumps(data), encoding="utf-8")
    print(f"\n  ✓ Written: {param_path.name}")
    return param_path

# ─────────────────────────────────────────────────────────────────────────────
# Engine Class
# ─────────────────────────────────────────────────────────────────────────────

@register_engine("vasp")
class VaspEngine:
    """VASP DFT engine with native IBRION=6 elastic constants."""

    name: str = "vasp"
    output_suffix: str = "OUTCAR"
    subdir_name: str = _cfg.VASP_SUBDIR
    _cleanup_globs = _cfg.VASP_CLEANUP_GLOBS

    def __init__(
        self,
        vasp_cmd: str,
        potcar_dir: str | Path,
        vaspkit_cmd: str | None = None,
        param_src: str | Path | None = None,
        cutoff: int = _cfg.ENCUT_SOFT,
        spin: bool = False,
        smearing: float = _cfg.SMEARING_VCA,
    ) -> None:
        self.vasp_cmd = str(vasp_cmd)
        self.potcar_dir = Path(potcar_dir) if potcar_dir else Path("")
        self.vaspkit_cmd = str(vaspkit_cmd) if vaspkit_cmd else None

        self.cutoff = int(cutoff)
        self.spin = bool(spin)
        self.smearing = float(smearing)

        if param_src:
            ps = Path(param_src)
            if ps.exists():
                try:
                    p = json.loads(ps.read_text())
                    self.cutoff = p.get("cutoff", self.cutoff)
                    self.spin = p.get("spin", self.spin)
                    self.smearing = p.get("smearing", self.smearing)
                except Exception:
                    pass

    @classmethod
    def setup_interactive(
        cls,
        src: Path,
        crystal: Crystal,
        override_cmd: str | None = None,
    ) -> tuple["VaspEngine", str]:
        """Factory called by main.py — runs wizards and returns (engine, cmd)."""
        species_list = list(dict.fromkeys(crystal.species))
        vasp_cmd, potcar_dir, vaspkit_cmd = _wizard_vasp_cmd(override_cmd)
        param_src = _wizard_param(src, crystal, species_list, is_vca=True)

        engine = cls(
            vasp_cmd=vasp_cmd,
            potcar_dir=potcar_dir,
            vaspkit_cmd=vaspkit_cmd,
            param_src=param_src
        )
        return engine, vasp_cmd

    def cleanup(self, step_dir: Path) -> None:
        """Remove heavy output files matching _cleanup_globs."""
        for pat in self._cleanup_globs:
            for f in step_dir.glob(pat):
                try:
                    f.unlink()
                except OSError:
                    pass

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

            # 1. POSCAR (Delegates to POSCAR_INCAR.py pure functions)
            ord_elems, vca_weights = write_vca_poscar(
                dest_dir / "POSCAR", crystal, tmpl_elem, target_mix, vegard=True
            )

            # 2. POTCAR (Raises explicit error if fails)
            zval_map = self._build_potcar(dest_dir, ord_elems)

            # 3. NELECT
            poscar_text = (dest_dir / "POSCAR").read_text(encoding="utf-8")
            counts = [int(c) for c in poscar_text.splitlines()[6].split()]
            nelect = sum(
                zval_map.get(el, 0.0) * count * weight
                for el, count, weight in zip(ord_elems, counts, vca_weights)
            )

            # 4. INCAR (KPOINTS is no longer generated; KSPACING is used in INCAR)
            ncore = int(multiprocessing.cpu_count() ** 0.5)
            write_engine_params(
                dest_dir / "INCAR",
                task_type="GeometryOptimization",
                xc=_cfg.XC_DEFAULT,
                cutoff=self.cutoff,
                spin=self.spin,
                smearing=self.smearing,
                vca_weights=vca_weights,
                nelect=nelect,
                ncore=ncore,
            )

    def parse_output(self, output_file: Path) -> EngineResult:
        return parse_outcar(output_file)

    def progress_monitor(self, proc: subprocess.Popen, stop: threading.Event) -> None:
        _BAR = 22
        t0 = time.time()
        ionic, scf, nsw = 0, 0, 200
        disp_cur, disp_tot = 0, 0

        def _render() -> None:
            el = time.time() - t0
            mm, ss = divmod(int(el), 60)
            if disp_tot > 0:
                pct = min(disp_cur / disp_tot, 1.0)
                filled = int(pct * _BAR)
                bar = "█" * filled + "░" * (_BAR - filled)
                print(f"\r  │  [{bar}] {int(pct*100):3d}%  displ {disp_cur}/{disp_tot}  scf {scf}  ⏱ {mm:02d}:{ss:02d}    ", end="", flush=True)
            else:
                pct = min(ionic / max(nsw, 1), 1.0)
                filled = int(pct * _BAR)
                bar = "█" * filled + "░" * (_BAR - filled)
                print(f"\r  │  [{bar}] {int(pct*100):3d}%  ionic {ionic}/{nsw}  scf {scf}  ⏱ {mm:02d}:{ss:02d}    ", end="", flush=True)

        try:
            if proc.stdout:
                for raw in iter(proc.stdout.readline, b""):
                    if stop.is_set(): break
                    ln = raw.decode(errors="replace").rstrip()

                    if ln.strip().startswith(("DAV:", "RMM:")):
                        try: scf = int(ln.split()[1])
                        except (IndexError, ValueError): pass
                    elif ln.strip() and ln.strip()[0].isdigit() and "F=" in ln:
                        try:
                            ionic = int(ln.split()[0])
                            scf = 0
                        except (IndexError, ValueError): pass
                    elif "Total:" in ln:
                        m = re.search(r"Total:\s*(\d+)/\s*(\d+)", ln)
                        if m: disp_cur, disp_tot = int(m.group(1)), int(m.group(2))
                    elif "NSW" in ln and "=" in ln:
                        m = re.search(r"NSW\s*=\s*(\d+)", ln)
                        if m: nsw = max(1, int(m.group(1)))

                    _render()
        except (OSError, ValueError): pass
        finally:
            print("\r" + " " * 90, end="\r", flush=True)

    @classmethod
    def get_wizard_schema(cls, crystal: Crystal, is_vca: bool) -> list[dict]:
        species = list(dict.fromkeys(crystal.species)) if crystal else []
        has_hard = any(s.capitalize() in _cfg.HARD_ELEMENTS for s in species)
        has_mag = any(s.capitalize() in _cfg.MAGNETIC_ELEMENTS for s in species)
        def_cut = _cfg.ENCUT_HARD if has_hard else _cfg.ENCUT_SOFT
        def_smear = _cfg.SMEARING_VCA if is_vca else _cfg.SMEARING_SINGLE

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

    def run_internal_elastic(
        self,
        step_dir: Path,
        seed: str,
        x: float,
        species: list[tuple[str, float]],
        nonmetal: str | None,
        density_gcm3: float | None,
        volume_ang3: float | None,
    ) -> dict[str, str]:
        """
        VASP native IBRION=6 elastic constants workflow.
        Parses OUTCAR directly, computes properties via core_physics,
        and formats exactly like CASTEP for unified CSV output.
        """
        import time
        import shutil
        import re
        import subprocess
        import numpy as np
        import core_physics as _phys

        t0 = time.monotonic()

        contcar = step_dir / "CONTCAR"
        if not contcar.exists():
            return {"_elastic_error": "CONTCAR not found. Geometry optimization failed?"}

        shutil.copy2(contcar, step_dir / "POSCAR")

        # Extract VCA parameters from existing INCAR safely
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
            xc=_cfg.XC_DEFAULT,
            cutoff=self.cutoff,
            spin=self.spin,
            smearing=_cfg.SMEARING_SINGLE,
            vca_weights=vca_weights,
            nelect=nelect,
            ncore=0,
        )

        # Isolated Subprocess Execution
        try:
            subprocess.run(self.vasp_cmd, shell=True, cwd=step_dir, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            return {"_elastic_error": f"VASP crash: {e.stderr.decode(errors='replace')}"}
        except OSError as e:
            return {"_elastic_error": f"OS Error running VASP: {e}"}

        # Parse the IBRION=6 elastic tensor directly from OUTCAR
        outcar = step_dir / "OUTCAR"
        if not outcar.exists():
            return {"_elastic_error": "OUTCAR not found."}

        text = outcar.read_text(encoding="utf-8", errors="replace")

        # VASP native IBRION=6 tensor block
        m = re.search(r"TOTAL ELASTIC MODULI \(kBar\)\s+Direction[^\n]+\n[^\n]+\n(.*?)(?:\n\s*\n|---)", text, re.DOTALL)
        if not m:
            return {"_elastic_error": "Elastic tensor not found in OUTCAR."}

        rows = []
        for line in m.group(1).splitlines():
            parts = line.split()
            if len(parts) >= 7:  # Example: "XX 11.1 12.2 ..."
                try:
                    rows.append([float(val) for val in parts[1:7]])
                except ValueError:
                    pass

        if len(rows) < 6:
            return {"_elastic_error": "Incomplete elastic tensor in OUTCAR."}

        C_GPa = np.array(rows[:6]) / 10.0  # Convert kBar to GPa

        # Map 6x6 tensor to cubic parameters for core_physics
        c11 = float(C_GPa[0, 0])
        c12 = float(C_GPa[0, 1])
        c44 = float(C_GPa[3, 3])

        result: dict[str, str] = {}

        # Delegate to core_physics exactly like CASTEP
        props = _phys.cubic_vrh(c11, c12, c44, density=density_gcm3, vol=volume_ang3)

        if props:
            result.update({k: f"{v:.4f}" for k, v in props.items()})
            # Ensure C11, C12, C44 are explicitly added for CSV alignment
            result.update({"C11": f"{c11:.4f}", "C12": f"{c12:.4f}", "C44": f"{c44:.4f}"})
            result["elastic_source"] = "VASP-IBRION6"
            result["elastic_n_points"] = "1"  # IBRION=6 executes everything in one run
            result["elastic_R2_min"] = "N/A"  # No linear regression applied here
        else:
            return {"_elastic_error": "Born stability violated."}

        result["elastic_wall_time_s"] = f"{time.monotonic() - t0:.0f}"

        # Safe cleanup call (checking if it exists to avoid crash)
        if hasattr(self, "cleanup"):
            self.cleanup(step_dir)

        return result

    def _build_potcar(self, dest_dir: Path, ord_elems: list[str]) -> dict[str, float]:
        if self.vaspkit_cmd:
            try:
                proc = subprocess.run(
                    f"echo '103' | {self.vaspkit_cmd}",
                    shell=True, cwd=dest_dir,
                    capture_output=True, text=True, timeout=60,
                )
                if proc.returncode == 0 and (dest_dir / "POTCAR").exists():
                    return _extract_zval(dest_dir / "POTCAR", ord_elems)
            except (OSError, subprocess.TimeoutExpired):
                pass

        # Fallback to manual merge
        return _merge_potcars(dest_dir, ord_elems, self.potcar_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def _extract_zval(potcar: Path, elements: list[str]) -> dict[str, float]:
    text = potcar.read_text(encoding="utf-8", errors="replace")
    zvals = re.findall(r"ZVAL\s*=\s*([\d.]+)", text)
    return {el: float(z) for el, z in zip(elements, zvals)}

def _merge_potcars(dest: Path, elements: list[str], potcar_dir: Path) -> dict[str, float]:
    zval_map: dict[str, float] = {}
    potcar_content = []

    for el in elements:
        sub_dir = _cfg.POTCAR_PREFERRED.get(el.capitalize(), el.capitalize())
        p_path = potcar_dir / sub_dir / "POTCAR"

        if not p_path.exists():
            p_path = potcar_dir / el.capitalize() / "POTCAR"
            if not p_path.exists():
                raise FileNotFoundError(f"POTCAR not found for {el} in {potcar_dir}")

        text = p_path.read_text(encoding="utf-8", errors="replace")
        potcar_content.append(text)

        m = re.search(r"ZVAL\s*=\s*([\d.]+)", text)
        zval_map[el] = float(m.group(1)) if m else 0.0

    (dest / "POTCAR").write_text("".join(potcar_content), encoding="utf-8")
    return zval_map
