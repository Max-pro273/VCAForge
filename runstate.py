"""
runstate.py  —  Data model for VCAForge runs.
═══════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import config
import core_physics

# ─────────────────────────────────────────────────────────────────────────────
# Status constants
# ─────────────────────────────────────────────────────────────────────────────

PENDING = "pending"
RUNNING = "running"
DONE = "done"
SKIPPED = "skipped"
FAILED = "failed"


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Step:
    idx: int
    concentration: float
    status: str = PENDING
    step_dir: str = ""
    started_at: str = ""
    finished_at: str = ""
    rc: str = ""
    parsed: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        base = {
            "step": self.idx,
            "concentration": self.concentration,
            "status": self.status,
            "step_dir": self.step_dir,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "rc": self.rc,
        }
        base.update(self.parsed)
        return base

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Step:
        reserved = {
            "step", "concentration", "status", "step_dir",
            "started_at", "finished_at", "rc", "parsed",
        }
        parsed = dict(d.get("parsed") or {})
        parsed.update({k: v for k, v in d.items() if k not in reserved})
        return cls(
            idx=int(d.get("step", 0)),
            concentration=float(d.get("concentration", 0.0)),
            status=d.get("status", PENDING),
            step_dir=d.get("step_dir", ""),
            started_at=d.get("started_at", ""),
            finished_at=d.get("finished_at", ""),
            rc=d.get("rc", ""),
            parsed=parsed,
        )


@dataclass
class RunState:
    version: str
    seed: str
    proj_dir: Path
    template_element: str
    species: list[tuple[str, float]]
    engine_cmd: str
    c_start: float
    c_end: float
    n_steps: int
    created_at: str
    single_mode: bool = False
    nonmetal: str = ""
    nonmetal_occ: float = 1.0
    run_elastic: bool = False
    engine_kwargs: dict[str, Any] = field(default_factory=dict)
    crystal_mode: str = "vca"
    crystal_kwargs: dict[str, Any] = field(default_factory=dict)
    engine_name: str = ""
    steps: list[Step] = field(default_factory=list)

    @property
    def n_done(self) -> int:
        return sum(1 for s in self.steps if s.status == DONE)

    @property
    def n_failed(self) -> int:
        return sum(1 for s in self.steps if s.status == FAILED)

    def system_label(self) -> str:
        if self.single_mode or not self.species:
            return self.seed
        sp = self.species
        nm = self.nonmetal
        base = self.template_element
        if not base and sp:
            base = sp[0][0]

        if len(sp) == 2 and sum(f for e, f in sp if e != base) == 1.0:
            other = next(e for e, f in sp if e != base)
            metal = f"{base}(1-x){other}(x)"
        else:
            inner_parts = []
            # self.species might not be sorted alphabetically, sort it for consistent brackets
            for e, f in sorted(sp, key=lambda kv: kv[0]):
                # Keep elements even with 0.0 fraction so the system dimensionality is preserved
                inner_parts.append(f"{e}{f:.4f}")
            inner = "".join(inner_parts)
            metal = f"{base}(1-x)[{inner}](x)"
        return f"{metal}{nm}" if nm else metal

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["proj_dir"] = str(self.proj_dir)
        d["steps"] = [s.to_dict() for s in self.steps]
        return d

    @classmethod
    def from_json(cls, d: dict[str, Any], proj_dir: Path) -> RunState:
        return cls(
            version=d.get("version", "?"),
            seed=d["seed"],
            proj_dir=proj_dir,
            template_element=d.get("template_element", ""),
            species=d.get("species", []),
            engine_cmd=d.get("engine_cmd", ""),
            c_start=d.get("c_start", 0.0),
            c_end=d.get("c_end", 1.0),
            n_steps=d.get("n_steps", 0),
            created_at=d.get("created_at", ""),
            single_mode=d.get("single_mode", False),
            nonmetal=d.get("nonmetal", ""),
            nonmetal_occ=d.get("nonmetal_occ", 1.0),
            run_elastic=d.get("run_elastic", False),
            engine_kwargs=d.get("engine_kwargs", {}),
            crystal_mode=d.get("crystal_mode", "vca"),
            crystal_kwargs=d.get("crystal_kwargs", {}),
            engine_name=d.get("engine_name", ""),
            steps=[Step.from_dict(s) for s in d.get("steps", [])],
        )


class StateIO:
    _CSV_FIXED = [
        "step", "concentration", "status",
        "started_at", "finished_at",
        "wall_time_s", "elastic_wall_time_s", "total_wall_time_s",
    ]

    _CSV_PREFERRED_ORDER = [
        "concentration", "VEC",
        "a_opt_ang", "b_opt_ang", "c_opt_ang", "a_prim_ang",
        "alpha", "beta", "gamma", "volume_ang3", "density_gcm3",
        "energy_ev", "free_energy_ev", "energy_0k_ev",
        "enthalpy_eV", "dH_mix_meV_per_fu",
        "fermi_ev", "mag_moment",
        "C11", "C22", "C33", "C44", "C55", "C66",
        "C12", "C13", "C14", "C15", "C16",
        "C23", "C24", "C25", "C26",
        "C34", "C35", "C36",
        "C45", "C46", "C56",
        "B_Voigt_GPa", "B_Reuss_GPa", "B_Hill_GPa", "B_EOS_GPa",
        "G_Voigt_GPa", "G_Reuss_GPa", "G_Hill_GPa",
        "E_GPa", "nu", "Zener_A", "Pugh_ratio",
        "Cauchy_pressure_GPa", "C_prime_GPa",
        "H_Vickers_Chen_GPa", "H_Vickers_Tian_GPa",
        "H_Vickers_VCAForge_Calibrated",
        "born_stable",
        "v_longitudinal_ms", "v_transverse_ms", "v_mean_ms", "T_Debye_K",
        "residual_pressure_GPa", "fmax_ev_ang",
        "geom_converged", "kill_reason", "warnings",
        "strategy",
        "sqs_supercell", "sqs_n_template_sites", "sqs_objective",
        "sqs_iterations", "sqs_max_deviation", "sqs_is_exact",
        "vca_vegard_k",
        "elastic_source", "elastic_n_points", "elastic_R2_min", "elastic_R2_global",
        "elastic_quality_note", "elastic_wall_time_s",
        "peak_mem_mb", "step_dir", "rc",
    ]

    @staticmethod
    def _now() -> str:
        return datetime.now().isoformat(timespec="seconds")

    @staticmethod
    def _state_path(proj_dir: Path) -> Path:
        return proj_dir / config.STATE_FILE

    @staticmethod
    def save_run(state: RunState) -> None:
        """Atomically persist state to JSON (write-then-rename)."""
        dst = StateIO._state_path(state.proj_dir)
        tmp = dst.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(state.to_json(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp.replace(dst)

    @staticmethod
    def load_run(proj_dir: Path) -> RunState | None:
        f = StateIO._state_path(proj_dir)
        if not f.exists():
            return None
        try:
            return RunState.from_json(
                json.loads(f.read_text(encoding="utf-8")), proj_dir
            )
        except (json.JSONDecodeError, KeyError):
            return None

    @staticmethod
    def new_run(
        *,
        seed: str,
        proj_dir: Path,
        template_element: str,
        species: list[tuple[str, float]],
        engine_cmd: str,
        engine_name: str = "",
        c_start: float,
        c_end: float,
        n_steps: int,
        points: list[float] | None = None,
        single_mode: bool = False,
        nonmetal: str = "",
        nonmetal_occ: float = 1.0,
        run_elastic: bool = False,
        engine_kwargs: dict[str, Any] | None = None,
        crystal_mode: str = "vca",
        crystal_kwargs: dict[str, Any] | None = None,
    ) -> RunState:
        proj_dir.mkdir(parents=True, exist_ok=True)

        if single_mode:
            step_list = [Step(idx=0, concentration=0.0)]
        elif points:
            step_list = [
                Step(idx=i, concentration=round(p, 10))
                for i, p in enumerate(points)
            ]
        else:
            d = (c_end - c_start) / max(n_steps, 1)
            step_list = [
                Step(
                    idx=i,
                    concentration=round(
                        c_end if i == n_steps else c_start + i * d, 10
                    ),
                )
                for i in range(n_steps + 1)
            ]

        state = RunState(
            version=config.VERSION,
            seed=seed,
            proj_dir=proj_dir,
            template_element=template_element,
            species=species,
            engine_cmd=engine_cmd,
            c_start=c_start,
            c_end=c_end,
            n_steps=n_steps,
            created_at=StateIO._now(),
            single_mode=single_mode,
            nonmetal=nonmetal,
            nonmetal_occ=nonmetal_occ,
            run_elastic=run_elastic,
            engine_kwargs=engine_kwargs or {},
            crystal_mode=crystal_mode,
            crystal_kwargs=crystal_kwargs or {},
            engine_name=engine_name,
            steps=step_list,
        )
        StateIO.save_run(state)
        return state

    @staticmethod
    def write_csv(state: RunState) -> Path:
        core_physics._maybe_decorate_dh_mix(state)

        all_keys: set[str] = set()
        for s in state.steps:
            all_keys.update(s.parsed.keys())

        ordered: list[str] = []
        seen: set[str] = set(StateIO._CSV_FIXED)
        for k in StateIO._CSV_PREFERRED_ORDER:
            if k not in seen and (k in all_keys or k in StateIO._CSV_FIXED):
                ordered.append(k)
                seen.add(k)
        for s in state.steps:
            for k in s.parsed:
                if k not in seen:
                    ordered.append(k)
                    seen.add(k)

        all_fields = StateIO._CSV_FIXED + ordered
        out = state.proj_dir / config.CSV_FILE

        with out.open("w", newline="", encoding="utf-8") as f:
            f.write(f"# VCAForge v{config.VERSION}\n")
            f.write(f"# System  : {state.system_label()}\n")
            f.write(f"# Seed    : {state.seed}\n")
            f.write(f"# Mode    : {state.crystal_mode}\n")
            f.write(f"# Updated : {StateIO._now()}\n")
            f.write("#\n")
            w = csv.DictWriter(
                f, fieldnames=all_fields, extrasaction="ignore", restval="N/A"
            )
            w.writeheader()
            w.writerows(s.to_dict() for s in state.steps)
        return out
