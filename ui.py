"""
ui.py  —  VCAForge console UI.
═══════════════════════════════
Pure presentation: prompts, wizards, step boxes, summary table.
"""

from __future__ import annotations

import shutil
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import config as _cfg
import core_physics as phys
from orchestrator import (
    DONE,
    FAILED,
    PENDING,
    SKIPPED,
    ExecResult,
    RunState,
    Step,
    mixing_enthalpy,
)


def _tw(fallback: int = 88) -> int:
    try:
        return min(shutil.get_terminal_size().columns, 120)
    except Exception:
        return fallback


def section(title: str) -> None:
    w = _tw()
    s = f"── {title} "
    print(f"\n{s}{'─' * max(2, w - len(s))}")


def fmt_time(s: float) -> str:
    s = int(s)
    return f"{s}s" if s < 60 else f"{s // 60}m {s % 60}s"


def ask_yes_no(question: str, default: bool | None = None) -> bool:
    hint = {True: "[Y/n]", False: "[y/N]", None: "[y/n]"}[default]
    while True:
        raw = input(f"  {question} {hint}: ").strip().lower()
        if raw == "" and default is not None:
            return default
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False


def ask_float(prompt: str, low: float = 0.0, high: float = 1.0) -> float:
    while True:
        try:
            v = float(input(f"  {prompt}").strip())
            if low <= v <= high:
                return v
            print(f"  Must be in [{low:.4g}, {high:.4g}].")
        except ValueError:
            print("  Not a valid number.")


def ask_int(prompt: str, low: int = 1) -> int:
    while True:
        try:
            v = int(input(f"  {prompt}").strip())
            if v >= low:
                return v
            print(f"  Must be >= {low}.")
        except ValueError:
            print("  Not a valid integer.")


def ask_str(prompt: str, default: str = "") -> str:
    raw = input(f"  {prompt}").strip()
    return raw if raw else default


def ask_choice(options: list[str], default: str) -> str:
    while True:
        raw = input(f"  [{default}]: ").strip()
        if not raw:
            return default
        for o in options:
            if o.lower() == raw.lower():
                return o
        print(f"  '{raw}' not valid.  Options: {', '.join(options)}")


def render_wizard(schema: list[dict]) -> dict[str, Any]:
    answers: dict[str, Any] = {}
    for field in schema:
        key = field["key"]
        label = field["label"]
        ftype = field["type"]
        default = field["default"]
        help_text = field.get("help", "")
        options = field.get("options", [])

        print(f"\n  ┌ {label}")
        if help_text:
            for ln in help_text.splitlines():
                print(f"  │  {ln}")
        print("  └")

        if ftype == "choice":
            answers[key] = ask_choice(options, str(default))
        elif ftype == "int":
            raw = input(f"  [{default}]: ").strip()
            try:
                answers[key] = int(float(raw)) if raw else int(default)
            except ValueError:
                answers[key] = int(default)
        elif ftype == "float":
            raw = input(f"  [{default:.2f}]: ").strip()
            try:
                answers[key] = float(raw) if raw else float(default)
            except ValueError:
                answers[key] = float(default)
        elif ftype == "bool":
            def_str = "yes" if default else "no"
            answers[key] = ask_choice(["yes", "no"], def_str) == "yes"
        else:
            raw = input(f"  [{default}]: ").strip()
            answers[key] = raw if raw else default

    return answers


@dataclass
class WizardResult:
    template_element: str
    target_mix: dict[str, float]
    single_mode: bool
    c_start: float
    c_end: float
    n_steps: int
    nonmetal: str
    run_elastic: bool


def wizard_mode(
    crystal,
    cli_elements: list[str] | None,
    cli_range: tuple[float, float, int] | None,
    args: Any = None,
) -> WizardResult:
    all_species = list(dict.fromkeys(crystal.species))
    nonmetals = [
        s
        for s in all_species
        if s in {"C", "N", "O", "B", "H", "P", "S", "F", "Cl", "Si"}
    ]
    metals = [
        s
        for s in all_species
        if s not in {"C", "N", "O", "B", "H", "P", "S", "F", "Cl", "Si"}
    ]
    nonmetal = nonmetals[0] if nonmetals else ""
    default_tmpl = metals[0] if metals else (all_species[0] if all_species else "X")

    section("Species / mode")

    if cli_elements is None and len(all_species) > 1:
        print(f"\n  Structure elements: {', '.join(all_species)}")
        print(
            f"  ┌ Which element in the original structure should be replaced by the VCA mixture?"
        )
        print(f"  │  Options: {', '.join(all_species)}")
        print("  └")
        template = ask_choice(all_species, default_tmpl)
    else:
        template = default_tmpl

    template_label = f"{template}{nonmetal}" if nonmetal else template
    if cli_elements is not None:
        raw_elems = [e.capitalize() for e in cli_elements]
    else:
        prompt = (
            f"Enter elements for the VCA mix separated by space (e.g. 'Nb V'),\n"
            f"replace a single element (e.g. 'Nb'),\n"
            f"or press Enter to run the pure template [{template_label}]:"
        )
        raw = input(f"\n{textwrap.indent(prompt, '  ')}\n  > ").strip()
        raw_elems = [e.capitalize() for e in raw.split()] if raw else []

    if not raw_elems:
        print(f"\n  · Mode: single compound  ({template_label}  pure template)")
        run_elastic = ask_yes_no("Run elastic constants after GeomOpt?", default=False)
        return WizardResult(
            template, {template: 1.0}, True, 0.0, 0.0, 0, nonmetal, run_elastic
        )

    if len(raw_elems) == 1:
        compound = f"{raw_elems[0]}{nonmetal}"
        print(f"\n  · Mode: single compound  ({compound})")
        run_elastic = ask_yes_no("Run elastic constants after GeomOpt?", default=False)
        return WizardResult(
            template, {raw_elems[0]: 1.0}, True, 0.0, 0.0, 0, nonmetal, run_elastic
        )

    elems = raw_elems
    if len(elems) == 2:
        target_mix_x1 = {elems[0]: 0.0, elems[1]: 1.0}
        a, b = elems[0], elems[1]
    else:
        target_mix_x1, a, b = _ask_ternary_fracs(elems)

    print(f"\n  · Mode: VCA sweep  {a}(1-x) → {b}(x)  on {template} sublattice")

    if cli_range:
        c_start, c_end, n_steps = cli_range
    else:
        section("Concentration range")
        c_start = ask_float("Start [0–1]: ")
        c_end = ask_float("End   [0–1]: ")
        n_steps = ask_int("Intervals (e.g. 8): ")

    run_elastic = ask_yes_no(
        "\n  Run elastic constants after each GeomOpt?", default=False
    )
    return WizardResult(
        template, target_mix_x1, False, c_start, c_end, n_steps, nonmetal, run_elastic
    )


def _ask_ternary_fracs(elems: list[str]) -> tuple[dict[str, float], str, str]:
    import sys as _sys

    new_elems = elems[1:]
    print(
        f"\n  {len(elems)} elements: {', '.join(elems)}\n  '{elems[0]}' phased out (→ 0 as x → 1).\n  Assign fractions for {len(new_elems)} elements at x=1:"
    )
    fracs: dict[str, float] = {elems[0]: 0.0}
    remaining = 1.0
    for i, e in enumerate(new_elems):
        if i == len(new_elems) - 1:
            f = round(remaining, 10)
            if f < 1e-6:
                print(f"  ✗  No fraction left for '{e}'.  Reduce earlier allocations.")
                _sys.exit(1)
            print(f"  Fraction of {e} at x=1  (auto-remainder): {f:.4f}")
        else:
            n_after = len(new_elems) - i - 1
            hi = max(0.0, round(remaining - 1e-4 * n_after, 10))
            if hi < 1e-6:
                print(f"  ✗  No fraction for '{e}'.  Reduce earlier allocations.")
                _sys.exit(1)
            f = ask_float(
                f"  Fraction of {e} at x=1  (remaining: {remaining:.4f}, max {hi:.4f}): ",
                0.0,
                hi,
            )
        fracs[e] = f
        remaining = round(remaining - f, 10)

    total_new = sum(v for k, v in fracs.items() if k != elems[0])
    if abs(total_new - 1.0) > 1e-3 and total_new > 1e-9:
        scale = 1.0 / total_new
        fracs = {
            k: (0.0 if k == elems[0] else round(v * scale, 10))
            for k, v in fracs.items()
        }
    parts = "  ".join(f"{e}={fracs[e]:.4f}" for e in new_elems)
    print(f"\n  ✓  Mix at x=1: {parts}")
    return fracs, elems[0], elems[-1]


def print_step_header(step: Step, total: int, state: RunState) -> None:
    x = step.concentration
    width = _tw(88)

    cmd = (
        state.engine_cmd.replace("{seed}", state.seed)
        if state.engine_cmd
        else "(Local MLIP Exec)"
    )
    if len(cmd) > width - 10:
        cmd = cmd[: width - 13] + "..."

    if len(state.species) == 2:
        sp = f"  {state.species[0][0]}={1 - x:.4f}  {state.species[1][0]}={x:.4f}"
    else:
        sp = "  " + "  ".join(
            f"{e}={round(f * x if i else 1 - x, 4)}"
            for i, (e, f) in enumerate(state.species)
        )

    vec_str = ""
    try:
        fracs = [(state.species[0][0], 1.0 - x)] + [
            (e, f * x) for e, f in state.species[1:]
        ]
        vec = phys.vec_for_system(fracs, state.nonmetal or None)
        vec_str = f"  VEC={vec:.2f}"
    except Exception:
        pass

    print(f"\n  ┌─ {step.idx}/{total - 1}  x={x:.4f}{sp}{vec_str}\n  │  $ {cmd}")


def print_step_result(
    result: ExecResult, step: Step, proj_dir: Path, seed: str, run_elastic: bool
) -> None:
    sym = "│ " if run_elastic else "└─"
    if result.skipped:
        print(f"  {sym} ⊘ Skipped")
        return

    if step.status == DONE:
        conv = "✓" if step.geom_converged == "yes" else "⚠ not converged"
        t = fmt_time(float(step.wall_time_s)) if step.wall_time_s else "—"
        a = f"a={step.a_opt_ang} Å"
        print(
            f"  {sym} {conv} Geometry Optimization   ({t})  [{a}  H={step.enthalpy_eV} eV]"
        )
        return

    print(f"  │  ✗  FAILED  (rc={step.rc})")
    for ln in [l for l in result.stderr_tail if l.strip() and "PMIX" not in l][-5:]:
        print(f"  │     {ln}")
    print("  └─ ✗ Step failed.")


def print_elastic_result(data: dict[str, Any], elapsed: float) -> None:
    t = fmt_time(elapsed) if elapsed else "—"
    b = data.get("B_Hill_GPa", "—")
    g = data.get("G_Hill_GPa", "—")
    e = data.get("E_GPa", "—")
    c11 = data.get("C11", "—")
    c12 = data.get("C12", "—")
    c44 = data.get("C44", "—")
    r2 = data.get("elastic_R2_min", "")
    src = data.get("elastic_source", "")

    tag = "  [Vegard]" if "Vegard" in src else ""
    r2s = f"  R²={r2}" if r2 and r2 != "N/A" else ""
    hv = data.get("H_Vickers_GPa", "")
    hvs = f"  Hv={hv} GPa" if hv else ""

    print(f"  │  ✓ Elastic Tensors ({t}){tag}  [B={b}  G={g}  E={e} GPa]{r2s}{hvs}")
    print(f"  │     C11={c11}  C12={c12}  C44={c44} GPa")
    if note := data.get("elastic_quality_note", ""):
        print(f"  │  ⚠  {note}")
    print("  └─ ✓ Step completed.")


def print_elastic_error(msg: str) -> None:
    print(f"  │  ⚠  Elastic failed: {msg}\n  └─ ✗ Elastic step failed.")


def print_summary(state: RunState) -> None:
    steps = state.steps
    W = min(_tw(), 92)
    sp = state.species
    sp_label = (
        f"{sp[0][0]}(1-x){sp[1][0]}(x)"
        if len(sp) == 2
        else " + ".join(
            f"{e}({f:.0%})" if i else f"{e}(1-x)" for i, (e, f) in enumerate(sp)
        )
    )

    print(f"\n{'═' * W}\n  {sp_label}  —  {state.proj_dir.name}")
    print(
        f"  {'#':>4}  {'x':>7}  {'Status':<8}  {'H (eV)':>16}  {'a (Å)':>8}  {'B (GPa)':>7}  conv"
    )
    print(
        f"  {'─' * 4}  {'─' * 7}  {'─' * 8}  {'─' * 16}  {'─' * 8}  {'─' * 7}  {'─' * 4}"
    )

    _ICON = {DONE: "✓", SKIPPED: "⊘", FAILED: "✗", PENDING: "·"}
    for s in steps:
        flag = " ⚠" if s.geom_converged == "no" and s.status == DONE else ""
        B = s.parsed.get("B_Hill_GPa", s.parsed.get("B_lbfgs_GPa", "—")) or "—"
        print(
            f"  {_ICON.get(s.status, '?')}{s.idx:>3}  {s.concentration:>7.4f}  {s.status:<8}  {(s.enthalpy_eV or '—'):>16}  {(s.a_opt_ang or '—'):>8}  {str(B):>7}  {(s.geom_converged or '—')}{flag}"
        )

    counts = {
        st: sum(1 for s in steps if s.status == st)
        for st in (DONE, SKIPPED, FAILED, PENDING)
    }
    print(
        f"{'═' * W}\n  ✓ {counts[DONE]}  ⊘ {counts[SKIPPED]}  ✗ {counts[FAILED]}  · {counts[PENDING]}"
    )


def print_smart_retry(failed_steps: list[Step], state: RunState) -> list[Step]:
    """Smart retry logic is primarily for CASTEP (.param editing). Skips for MLIP/VASP."""
    try:
        from orchestrator import patch_for_recovery
    except ImportError:
        return []

    section("Sweep completed — failure analysis")
    n_ok = sum(1 for s in state.steps if s.status != FAILED)
    print(f"  ✓  {n_ok} step(s) succeeded\n  ✗  {len(failed_steps)} step(s) failed\n")

    patchable: list[Step] = []
    for s in failed_steps:
        reason = s.parsed.get("kill_reason", "unknown")
        print(f"  ✗  x={s.concentration:.4f}  —  {reason}")
        if reason != "unknown":
            patchable.append(s)
        print()

    if not patchable or not ask_yes_no(
        f"Auto-patch .param and retry {len(patchable)} fixable step(s)?", default=True
    ):
        return []

    for s in patchable:
        ok = patch_for_recovery(
            state.proj_dir / s.step_dir,
            state.seed,
            s.parsed.get("kill_reason", "unknown"),
        )
        print(
            f"  {'✓' if ok else '⚠'}  {'Patched' if ok else 'Failed to patch'} {s.step_dir}"
        )
        s.status = "pending"
        s.rc = ""
    return patchable
