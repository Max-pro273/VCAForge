"""
engines/MLIP/mlip.py  —  MLIP engine for VCAForge.
═══════════════════════════════════════════════════════════════════════════════
Implements InProcessCapable and InProcessElasticCapable.
Uses model_registry.py for lazy loading of ML calculators.
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

import config as _cfg
from core_physics import (
    Crystal,
    universal_vrh,
    fit_cij_universal,
    generate_strain_steps,
    lattice_to_abc,
)
from engines.engine import (
    BaseEngine,
    EngineResult,
    InProcessCapable,
    InProcessElasticCapable,
    WatchdogCapable,
    register_engine,
)
from engines.MLIP.model_registry import load_model, all_models

if TYPE_CHECKING:
    from runstate import Step, RunState

log = logging.getLogger(__name__)


class _StopCallback:
    """Check stop_event every ASE step."""
    def __init__(self, stop_event: threading.Event | None):
        self.stop_event = stop_event
        self.step = 0

    def __call__(self):
        self.step += 1
        if self.stop_event and self.stop_event.is_set():
            raise InterruptedError("VCAForge stop signal received")


@register_engine("mlip")
class MlipEngine(
    BaseEngine,
    InProcessCapable,
    InProcessElasticCapable,
    WatchdogCapable,
):
    """Machine Learning Interatomic Potential engine."""

    name            = "mlip"
    output_suffix   = ".mlip.json"
    subdir_name     = _cfg.MLIP_SUBDIR
    SUPPORTED_MODES = frozenset({"sqs", "direct"})

    def __init__(
        self,
        model_name: str = _cfg.MLIP_DEFAULT_BACKEND,
        device: str = _cfg.MLIP_DEVICE,
        fmax: float = _cfg.MLIP_FMAX,
        relax_steps: int = _cfg.MLIP_RELAX_STEPS,
    ) -> None:
        self.model_name   = model_name
        self.device       = device
        self.fmax         = float(fmax)
        self.relax_steps  = int(relax_steps)
        self._calc        = None

    def _get_calc(self):
        if self._calc is None:
            self._calc = load_model(self.model_name, self.device)
        return self._calc

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name":  self.model_name,
            "device":      self.device,
            "fmax":        self.fmax,
            "relax_steps": self.relax_steps,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MlipEngine":
        return cls(
            model_name   = d.get("model_name", _cfg.MLIP_DEFAULT_BACKEND),
            device       = d.get("device", _cfg.MLIP_DEVICE),
            fmax         = float(d.get("fmax", _cfg.MLIP_FMAX)),
            relax_steps  = int(d.get("relax_steps", _cfg.MLIP_RELAX_STEPS)),
        )

    @classmethod
    def setup_interactive(
        cls, src: Path, crystal: Crystal, override_cmd: str | None, args: argparse.Namespace
    ) -> tuple["MlipEngine", str]:
        # Dynamic import to avoid module-level UI dependency
        import ui
        ui.section("Engine setup (MLIP)")

        model_name = getattr(args, "model", None) or _cfg.MLIP_DEFAULT_BACKEND
        device = getattr(args, "device", None) or _cfg.MLIP_DEVICE
        fmax = float(getattr(args, "mlip_fmax", None) or _cfg.MLIP_FMAX)

        print(f"  ✓  Model   : {model_name}")
        print(f"  ✓  Device  : {device}")
        print(f"  ✓  Fmax    : {fmax} eV/Å")

        return cls(model_name=model_name, device=device, fmax=fmax), ""

    @classmethod
    def get_wizard_schema(cls, crystal: Crystal, is_vca: bool) -> list[dict[str, Any]]:
        models = all_models()
        return [
            {
                "key": "model_name", "type": "choice",
                "label": "MLIP Model",
                "options": [m["key"] for m in models],
                "option_labels": {m["key"]: f"{m['label']} [{m['library']}]" for m in models},
                "default": _cfg.MLIP_DEFAULT_BACKEND,
            },
            {
                "key": "device", "type": "choice",
                "label": "Compute Device",
                "options": ["cpu", "cuda", "mps", "rocm"],
                "default": _cfg.MLIP_DEVICE,
            },
            {
                "key": "fmax", "type": "float",
                "label": "Force Convergence (eV/Å)",
                "default": _cfg.MLIP_FMAX,
            },
        ]

    def write_input(self, dest_dir: Path, seed: str, crystal: Crystal) -> None:
        # MLIP is in-process, but we write a sentinel for resume detection
        (dest_dir / f"{seed}.mlip_input.json").write_text(
            json.dumps(self.to_dict(), indent=2)
        )

    def parse_output(self, output_file: Path) -> EngineResult:
        if not output_file.exists():
            return EngineResult(warning="Output not found")
        data = json.loads(output_file.read_text())
        return EngineResult(
            energy_ev=data.get("energy_ev"),
            volume_ang3=data.get("volume_ang3"),
            density_gcm3=data.get("density_gcm3"),
            run_time_s=data.get("run_time_s"),
            extra_data=data.get("extra_data", {}),
            warning=data.get("warning"),
        )

    def run_in_process(
        self, step_dir: Path, seed: str, crystal: Crystal, stop_event: threading.Event
    ) -> EngineResult:
        import ase.optimize
        from ase.filters import FrechetCellFilter

        t0 = time.monotonic()
        atoms = crystal.to_ase()
        atoms.calc = self._get_calc()

        # FrechetCellFilter allows simultaneous relaxation of positions and cell
        ecf = FrechetCellFilter(atoms)
        opt = ase.optimize.LBFGS(ecf, logfile=str(step_dir / f"{seed}.mlip_opt.log"))

        cb = _StopCallback(stop_event)
        opt.attach(cb)

        warning = None
        try:
            opt.run(fmax=self.fmax, steps=self.relax_steps)
        except InterruptedError:
            warning = "Interrupted"
        except Exception as e:
            warning = str(e)
            log.exception("MLIP optimization failed")

        # Results
        try:
            energy = float(atoms.get_potential_energy())
            volume = float(atoms.get_volume())
            density = self._get_density(atoms)
            abc = lattice_to_abc(atoms.get_cell())

            extra_data = {
                "a_ang": abc[0], "b_ang": abc[1], "c_ang": abc[2],
                "alpha_deg": abc[3], "beta_deg": abc[4], "gamma_deg": abc[5],
                "n_steps": cb.step,
            }

            np.save(step_dir / "relaxed_cell.npy", atoms.get_cell()[:])
            np.save(step_dir / "relaxed_pos.npy", atoms.get_scaled_positions())
            np.save(step_dir / "relaxed_nums.npy", atoms.get_atomic_numbers())
        except Exception as e:
            energy = volume = density = None
            extra_data = {"n_steps": cb.step, "error": str(e)}
            warning = warning or str(e)

        res = EngineResult(
            energy_ev=energy,
            volume_ang3=volume,
            density_gcm3=density,
            run_time_s=time.monotonic() - t0,
            extra_data=extra_data,
            warning=warning,
        )

        # Write results for persistence/resume
        (step_dir / f"{seed}{self.output_suffix}").write_text(
            json.dumps({**asdict(res), "extra_data": extra_data})
        )
        # Update POSCAR for potential next steps/manual inspection
        import ase.io
        ase.io.write(str(step_dir / "POSCAR"), atoms, format="vasp")

        return res

    def run_elastic_in_process(
        self,
        crystal: Crystal,
        density_gcm3: float | None,
        volume_ang3: float | None,
        stop_event: threading.Event,
    ) -> dict[str, str]:
        import ase.optimize
        import torch
        from ase.eos import EquationOfState

        t0 = time.monotonic()
        # For elastic constants, float64 is critical to overcome numerical noise
        orig_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            # Re-load model with float64 to ensure weights and buffers are high-precision
            calc = load_model(self.model_name, self.device, default_dtype="float64")
            atoms_base = crystal.to_ase()
            atoms_base.calc = calc

            # 1. Independent EOS check for Bulk Modulus (Theory cross-check)
            # This is much more robust than stress-strain fit for unstable/noisy phases.
            v_list, e_list = [], []
            for scale in np.linspace(0.98, 1.02, 7):
                at_eos = atoms_base.copy()
                at_eos.set_cell(atoms_base.cell * scale, scale_atoms=True)
                at_eos.calc = calc
                # Re-relax internal for each volume step
                try:
                    opt_eos = ase.optimize.BFGS(at_eos, logfile=None)
                    opt_eos.run(fmax=0.005, steps=50)
                    v_list.append(at_eos.get_volume())
                    e_list.append(at_eos.get_potential_energy())
                except Exception:
                    continue
            
            b_eos_gpa = 0.0
            if len(v_list) >= 4:
                try:
                    eos = EquationOfState(v_list, e_list)
                    _, _, B_ev_a3 = eos.fit()
                    # 1 eV/A^3 = 160.21766208 GPa
                    b_eos_gpa = B_ev_a3 * 160.21766208
                except Exception as exc:
                    log.warning(f"EOS fit failed: {exc}")

            # 2. Full Cij via Stress-Strain
            # We use a slightly larger max_strain (0.01) for MLIPs to overcome remaining noise
            strain_steps = generate_strain_steps(crystal, max_strain=0.01, n_steps=3)
            stresses, strains = [], []

            for ss in strain_steps:
                if stop_event.is_set():
                    return {"_elastic_error": "Interrupted"}

                atoms = atoms_base.copy()
                eps = ss.strain_voigt
                # Deformation gradient F = I + ε_tensor
                F = np.eye(3) + np.array([
                    [eps[0],     eps[5] / 2, eps[4] / 2],
                    [eps[5] / 2, eps[1],     eps[3] / 2],
                    [eps[4] / 2, eps[3] / 2, eps[2]]
                ])
                atoms.set_cell(atoms_base.get_cell() @ F.T, scale_atoms=True)
                atoms.calc = calc

                # Re-relax internal coordinates (BFGS is more reliable for small shifts)
                try:
                    opt = ase.optimize.BFGS(atoms, logfile=None)
                    opt.run(fmax=0.001, steps=100)
                    
                    from ase.units import GPa as ase_GPa
                    stress = atoms.get_stress(voigt=True) / ase_GPa
                    stresses.append(stress)
                    strains.append(eps)
                except Exception as e:
                    log.error(f"Strain step failed: {e}")
                    continue

            if len(stresses) < 3:
                return {"_elastic_error": "Not enough successful strain steps"}

            result = fit_cij_universal(
                stresses, strains, density_gcm3=density_gcm3,
                n_atoms=crystal.num_atoms, volume_ang3=volume_ang3
            )

            if "_elastic_error" in result:
                    return result

            # Calibrated Vickers hardness for MLIP (corrects systematic softening)
            try:
                B = float(result.get("B_Hill_GPa", 0.0))
                G = float(result.get("G_Hill_GPa", 0.0))
                if B > 0 and G > 0:
                    # Formula: results['VCAForge_Calibrated'] = max(-53.18 + 0.2149 * B + 0.1976 * G, 0.0)
                    val = -53.18 + 0.2149 * B + 0.1976 * G
                    result["H_Vickers_VCAForge_Calibrated"] = f"{max(val, 0.0):.4f}"
            except (ValueError, TypeError):
                pass

            if b_eos_gpa > 0:
                result["B_EOS_GPa"] = f"{b_eos_gpa:.4f}"
            
            result["elastic_source"] = f"MLIP-{self.model_name.upper()}"
            result["elastic_wall_time_s"] = f"{time.monotonic() - t0:.1f}"
            return result
        finally:
            torch.set_default_dtype(orig_dtype)
        return result

    def _get_density(self, atoms):
        from ase.data import atomic_masses
        mass = sum(atomic_masses[z] for z in atoms.numbers)
        vol_cm3 = atoms.get_volume() * 1e-24
        return (mass / 6.022e23) / vol_cm3

    def check_health(self, log_tail: str) -> str | None:
        return None

    def progress_monitor(self, proc: Any, stop: threading.Event, cwd: Path) -> None:
        # Simple waiting message for in-process
        while not stop.is_set():
            time.sleep(1)

    def parse_extra_outputs(self, step_dir: Path, seed: str) -> dict[str, Any]:
        return {}
