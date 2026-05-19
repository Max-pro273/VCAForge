"""
ui.py  —  VCAForge console UI.
═══════════════════════════════════════════════════════════════
Pure presentation: prompts, wizards, step boxes, summary table.
NO physics, NO engine logic, NO state mutation.

Wizard accepts both interactive and CLI input. CLI flags ALWAYS skip
the interactive prompt for their field:
    --template Ti
    --composition "Nb:0.5 Ti:0.25 Zr:0.25"
    --points 0.0,0.3,0.5,0.7,1.0
    --x 0.3                       (single-point shortcut)
    --range 0.0 1.0 8             (start end intervals)
    --single                      (pure template)
    --elastic / --no-elastic
"""

from __future__ import annotations

import shutil
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import config as _cfg
from core_physics import Crystal
from orchestrator import ExecResult
from runstate import (
    DONE,
    FAILED,
    PENDING,
    SKIPPED,
    RunState,
    Step,
)


# ─────────────────────────────────────────────────────────────────────────────
# Layout
# ─────────────────────────────────────────────────────────────────────────────


def _tw(fallback: int = 88) -> int:
    return min(shutil.get_terminal_size((fallback, 24)).columns, 120)


def section(title: str) -> None:
    w = _tw()
    s = f"── {title} "
    print(f"{s}{'─' * max(2, w - len(s))}")


def fmt_time(s: float) -> str:
    s = int(s)
    return f"{s}s" if s < 60 else f"{s // 60}m {s % 60}s"


# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────


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
    opts = "/".join(options)
    while True:
        raw = input(f"  [{default}] ({opts}): ").strip()
        if not raw:
            return default
        for o in options:
            if o.lower() == raw.lower():
                return o
        print(f"  '{raw}' not valid. Options: {opts}")


def render_wizard(schema: list[dict]) -> dict[str, Any]:
    """Render an engine-supplied wizard schema (CASTEP/VASP parameter setup)."""
    answers: dict[str, Any] = {}
    for fs in schema:
        key = fs["key"]
        ftype = fs["type"]
        default = fs["default"]
        print(f"┌ {fs['label']}")
        if fs.get("help"):
            for ln in fs["help"].splitlines():
                print(f"  │  {ln}")
        print("  └")

        if ftype == "choice":
            answers[key] = ask_choice(fs.get("options", []), str(default))
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
            answers[key] = (
                ask_choice(["yes", "no"], "yes" if default else "no") == "yes"
            )
        else:
            raw = input(f"  [{default}]: ").strip()
            answers[key] = raw if raw else default
    return answers


# ─────────────────────────────────────────────────────────────────────────────
# Composition / points parsing
# ─────────────────────────────────────────────────────────────────────────────


def parse_composition(spec: str) -> dict[str, float]:
    """Parse 'Nb:0.5 Ti:0.25 Zr:0.25' (':' or '=' separator; ',' or ' ' between).

    Auto-normalises if the sum isn't 1.0. Raises ValueError on bad syntax.
    """
    if not spec or not spec.strip():
        raise ValueError("Empty composition spec.")
    normalized = spec.replace("=", ":").replace(",", " ").replace(";", " ")
    result: dict[str, float] = {}
    for tok in (t for t in normalized.split() if t):
        if ":" not in tok:
            raise ValueError(
                f"Bad token {tok!r}: expected 'Element:fraction' (e.g. Nb:0.5)."
            )
        el, frac = tok.split(":", 1)
        el = el.strip().capitalize()
        if not el.isalpha():
            raise ValueError(f"Bad element symbol {el!r} in token {tok!r}.")
        try:
            f = float(frac.strip())
        except ValueError as exc:
            raise ValueError(
                f"Bad fraction {frac!r} in token {tok!r}: {exc}"
            ) from exc
        if f < 0:
            raise ValueError(f"Negative fraction {f} for {el}.")
        result[el] = result.get(el, 0.0) + f
    total = sum(result.values())
    if total <= 1e-12:
        raise ValueError(f"Composition sums to zero: {spec!r}")
    if abs(total - 1.0) > 1e-6:
        result = {k: v / total for k, v in result.items()}
    return result


def parse_points(spec: str) -> list[float]:
    """Parse '0.0,0.3,0.5' or '0.0 0.3 0.5'."""
    if not spec.strip():
        raise ValueError("Empty points spec.")
    points: list[float] = []
    for tok in spec.replace(",", " ").split():
        try:
            v = float(tok)
        except ValueError as exc:
            raise ValueError(f"Bad value {tok!r}: {exc}") from exc
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Point {v} outside [0,1].")
        points.append(round(v, 10))
    if not points:
        raise ValueError(f"No points parsed from {spec!r}")
    return points


# ─────────────────────────────────────────────────────────────────────────────
# WizardResult
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class WizardResult:
    """Composition + sweep description. If `points` is non-empty it is the
    source of truth — c_start/c_end/n_steps are derived for compatibility."""

    template_element: str
    target_mix: dict[str, float]
    single_mode: bool
    c_start: float
    c_end: float
    n_steps: int
    nonmetal: str
    run_elastic: bool
    points: list[float] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Wizard
# ─────────────────────────────────────────────────────────────────────────────


def _split_metals_nonmetals(species: list[str]) -> tuple[list[str], list[str]]:
    nonmetals = [
        s for s in species
        if _cfg.ELEMENTS.get(s.capitalize(), {}).get("nonmetal", False)
    ]
    metals = [s for s in species if s not in nonmetals]
    return metals, nonmetals


def wizard_mode(crystal: Crystal, args: Any) -> WizardResult:
    """Interactive (and CLI-fed) composition / sweep wizard.

    Each field is resolved in this priority order:
      1. CLI flag        (skips interactive prompt for that field)
      2. Interactive prompt
    """
    all_species = list(dict.fromkeys(crystal.species))
    metals, nonmetals = _split_metals_nonmetals(all_species)
    nonmetal = nonmetals[0] if nonmetals else ""
    default_tmpl = (
        metals[0] if metals else (all_species[0] if all_species else "X")
    )

    template = _resolve_template(args, all_species, default_tmpl)

    # Single-compound shortcut (no composition, no sweep).
    if getattr(args, "single", False):
        print(f"· Mode: single compound ({template}{nonmetal})")
        return WizardResult(
            template_element=template,
            target_mix={template: 1.0},
            single_mode=True,
            c_start=0.0, c_end=0.0, n_steps=0,
            nonmetal=nonmetal,
            run_elastic=_resolve_run_elastic(args, default=False),
        )

    target_mix_at_x1 = _resolve_composition(args, template, nonmetal)

    if target_mix_at_x1 is None:
        # User declined → pure template, no sweep.
        print(f"· Mode: single compound ({template}{nonmetal} pure)")
        return WizardResult(
            template_element=template,
            target_mix={template: 1.0},
            single_mode=True,
            c_start=0.0, c_end=0.0, n_steps=0,
            nonmetal=nonmetal,
            run_elastic=_resolve_run_elastic(args, default=False),
        )

    nonzero = {e: f for e, f in target_mix_at_x1.items() if f > 1e-9}
    if len(target_mix_at_x1) == 1:
        sole = next(iter(nonzero)) if nonzero else next(iter(target_mix_at_x1))
        print(f"· Mode: single compound ({sole}{nonmetal})")
        return WizardResult(
            template_element=template,
            target_mix=target_mix_at_x1,
            single_mode=True,
            c_start=0.0, c_end=0.0, n_steps=0,
            nonmetal=nonmetal,
            run_elastic=_resolve_run_elastic(args, default=False),
        )

    print(
        f"· Mode: sweep  {template}(1-x) → mixture(x)  "
        f"on {template} sublattice"
    )
    # Ensure mixture output displays properly even if some are 0.0
    print("    Mixture at x=1: "
          + "  ".join(f"{e}={f:.4f}" for e, f in target_mix_at_x1.items()))

    points = _resolve_concentration_points(args)
    run_elastic = _resolve_run_elastic(args, default=False)

    c_start, c_end = (min(points), max(points)) if points else (0.0, 1.0)
    return WizardResult(
        template_element=template,
        target_mix=target_mix_at_x1,
        single_mode=False,
        c_start=c_start, c_end=c_end,
        n_steps=max(0, len(points) - 1),
        nonmetal=nonmetal,
        run_elastic=run_elastic,
        points=points,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Wizard sub-resolvers
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_template(
    args: Any, all_species: list[str], default: str,
) -> str:
    if getattr(args, "template", None):
        tmpl = str(args.template).capitalize()
        if tmpl not in all_species:
            sys.exit(
                f"  ERROR: --template {tmpl!r} not in structure species "
                f"({', '.join(all_species)})"
            )
        section("Template element")
        print(f"  Using --template = {tmpl}")
        return tmpl

    section("Template element")
    if len(all_species) == 1:
        print(f"  Only one element: {all_species[0]}")
        return all_species[0]
    print(f"  Structure elements: {', '.join(all_species)}")
    print("  ┌ Which element should be replaced by the mixture?")
    print(f"  │  Options: {', '.join(all_species)}")
    print("  └")
    return ask_choice(all_species, default)


def _resolve_composition(
    args: Any, template: str, nonmetal: str,
) -> dict[str, float] | None:
    """Resolve the mixture at x=1. Returns None for pure-template runs."""
    if getattr(args, "composition", None):
        try:
            comp = args.composition
            # --composition uses nargs="+" so argparse gives a list of tokens.
            # Join them back into a single string for parse_composition().
            if isinstance(comp, list):
                comp = " ".join(comp)
            return parse_composition(comp)
        except ValueError as exc:
            sys.exit(f"  ERROR: --composition: {exc}")

    if getattr(args, "species", None):
        elems = [e.capitalize() for e in args.species]
        if len(elems) == 1:
            return {elems[0]: 1.0}
        if len(elems) == 2:
            return {elems[0]: 0.0, elems[1]: 1.0}
        # ≥3 species without explicit composition → must ask.

    section("Composition")
    print(textwrap.indent(
        "Specify the mixture at x=1 (the composition reached as x → 1).\n"
        "Formats accepted:\n"
        "  · Explicit fractions  :  Nb:0.5 Zr:0.25 Ti:0.25\n"
        "                          (':' or '=', whitespace or comma between)\n"
        "  · Binary symbols      :  Nb Zr      → Nb(1-x) → Zr(x)\n"
        f"Press Enter or type '0' for a pure {template} run.",
        "  ",
    ))
    while True:
        raw = input("composition > ").strip()
        if not raw or raw == "0":
            return None
        if ":" in raw or "=" in raw:
            try:
                return parse_composition(raw)
            except ValueError as exc:
                print(f"  ✗ {exc}")
                continue
        elems = [e.capitalize() for e in raw.split()]
        if not elems:
            continue
        if len(elems) == 1:
            return {elems[0]: 1.0}
        if len(elems) == 2:
            return {elems[0]: 0.0, elems[1]: 1.0}
        print(
            f"  ✗ {len(elems)} bare symbols. For ternary+, use explicit "
            "fractions (e.g. 'Nb:0.5 Ti:0.25 Zr:0.25')."
        )


def _resolve_concentration_points(args: Any) -> list[float]:
    if getattr(args, "points", None):
        try:
            return sorted(set(parse_points(args.points)))
        except ValueError as exc:
            sys.exit(f"  ERROR: --points: {exc}")

    if getattr(args, "x", None) is not None:
        v = float(args.x)
        if not 0.0 <= v <= 1.0:
            sys.exit(f"  ERROR: --x must be in [0,1], got {v}")
        return [round(v, 10)]

    if getattr(args, "range", None):
        x0, x1, n = args.range
        if n < 1:
            sys.exit(f"  ERROR: --range N must be >= 1, got {n}")
        step = (x1 - x0) / n
        return [round(x0 + i * step, 10) for i in range(n + 1)]

    section("Concentration")
    print(textwrap.indent(
        "Formats accepted:\n"
        "  · Single point :  0.3\n"
        "  · List         :  0.0, 0.3, 0.5, 0.7, 1.0\n"
        "  · Range form   :  0.0 - 1.0 / 8     (start - end / N intervals)",
        "  ",
    ))
    while True:
        raw = input("concentration > ").strip()
        if not raw:
            print("  ✗ Empty. Type '0.3' for one point.")
            continue
        try:
            return sorted(set(_parse_concentration_freeform(raw)))
        except ValueError as exc:
            print(f"  ✗ {exc}")


def _parse_concentration_freeform(raw: str) -> list[float]:
    """One of: single value, comma/space list, 'a - b / n' range form."""
    if "/" in raw:
        left, n_str = raw.rsplit("/", 1)
        n = int(n_str.strip())
        if n < 1:
            raise ValueError(f"Range N must be >= 1, got {n}")
        if "-" not in left:
            raise ValueError(f"Range form needs 'a - b / n', got: {raw!r}")
        # Split only on the first '-' to avoid clobbering scientific notation.
        marker = "@MINUS@"
        protected = left.replace("e-", "e" + marker).replace("E-", "E" + marker)
        parts = protected.split("-", 1)
        if len(parts) != 2:
            raise ValueError(f"Range form needs 'a - b / n', got: {raw!r}")
        x0 = float(parts[0].replace(marker, "-"))
        x1 = float(parts[1].replace(marker, "-"))
        if not (0.0 <= x0 <= 1.0 and 0.0 <= x1 <= 1.0):
            raise ValueError(f"Range endpoints must be in [0,1]: {x0}, {x1}")
        step = (x1 - x0) / n
        return [round(x0 + i * step, 10) for i in range(n + 1)]

    return parse_points(raw)


def _resolve_run_elastic(args: Any, default: bool) -> bool:
    if getattr(args, "elastic", False):
        return True
    if getattr(args, "no_elastic", False):
        return False
    return ask_yes_no("Run elastic constants after each GeomOpt?", default=default)


# ─────────────────────────────────────────────────────────────────────────────
# Step renderers
# ─────────────────────────────────────────────────────────────────────────────


def _vec_for_step(state: RunState, x: float) -> float | None:
    fracs = [(state.species[0][0], 1.0 - x)] + [
        (e, f * x) for e, f in state.species[1:]
    ]
    metal_vec = sum(
        frac * _cfg.ELEMENTS.get(elem.capitalize(), {}).get("val", 0)
        for elem, frac in fracs
    )
    nm = state.nonmetal
    nm_vec = _cfg.ELEMENTS.get(nm.capitalize(), {}).get("val", 0) if nm else 0
    return metal_vec + nm_vec


def print_step_header(step: Step, total: int, state: RunState) -> None:
    x = step.concentration
    width = _tw(88)
    cmd = (
        state.engine_cmd.replace("{seed}", state.seed)
        if state.engine_cmd
        else "(in-process engine)"
    )
    if len(cmd) > width - 10:
        cmd = cmd[: width - 13] + "..."

    if len(state.species) == 2:
        sp = (
            f"  {state.species[0][0]}={1 - x:.4f}  "
            f"{state.species[1][0]}={x:.4f}"
        )
    else:
        sp = "  " + "  ".join(
            f"{e}={round(f * x if i else 1 - x, 4)}"
            for i, (e, f) in enumerate(state.species)
        )

    vec = _vec_for_step(state, x)
    vec_str = f"  VEC={vec:.2f}" if vec is not None else ""
    print(f"┌─ {step.idx + 1}/{total}  x={x:.4f}{sp}{vec_str}\n│  $ {cmd}")


def print_step_result(
    result: ExecResult, step: Step, proj_dir: Path,
    seed: str, run_elastic: bool,
) -> None:
    sym = "│ " if run_elastic else "└─"
    if result.skipped:
        print(f"  {sym} ⊘ Skipped")
        return
    if step.status == DONE:
        conv = (
            "✓" if step.parsed.get("geom_converged") == "yes"
            else "⚠ not converged"
        )
        wt = step.parsed.get("wall_time_s")
        t = fmt_time(float(wt)) if wt else "—"
        a = f"a={step.parsed.get('a_opt_ang', '—')} Å"
        H = step.parsed.get("enthalpy_eV") or step.parsed.get("energy_ev", "—")
        print(
            f"  {sym} {conv} Geometry Optimization   ({t})  "
            f"[{a}  H={H} eV]"
        )
        return

    print(f"  │  ✗  FAILED  (rc={step.rc})")
    for ln in [
        l for l in result.stderr_tail if l.strip() and "PMIX" not in l
    ][-5:]:
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
    hv = data.get("H_Vickers_GPa", "")
    r2s = f"  R²={r2}" if r2 and r2 != "N/A" else ""
    hvs = f"  Hv={hv} GPa" if hv else ""

    print(f"  │  ✓ Elastic Tensors ({t})  [B={b}  G={g}  E={e} GPa]{r2s}{hvs}")
    print(f"  │     C11={c11}  C12={c12}  C44={c44} GPa")
    if note := data.get("elastic_quality_note", ""):
        print(f"  │  ⚠  {note}")
    print("  └─ ✓ Step completed.")


def print_elastic_error(msg: str) -> None:
    print(f"  │  ⚠  Elastic failed: {msg}\n  └─ ✗ Elastic step failed.")


# ─────────────────────────────────────────────────────────────────────────────
# Summary + retry
# ─────────────────────────────────────────────────────────────────────────────


def print_summary(state: RunState) -> None:
    steps = state.steps
    W = min(_tw(), 92)
    sp = state.species
    sp_label = (
        f"{sp[0][0]}(1-x){sp[1][0]}(x)"
        if len(sp) == 2
        else " + ".join(
            f"{e}({f:.0%})" if i else f"{e}(1-x)"
            for i, (e, f) in enumerate(sp)
        )
    )

    print(f"{'═' * W}\n"
          f"  {sp_label}  —  {state.proj_dir.name}")
    print(
        f"  {'#':>4}  {'x':>7}  {'Status':<8}  {'H (eV)':>16}"
        f"  {'a (Å)':>8}  {'B (GPa)':>7}  conv"
    )
    print(
        f"  {'─' * 4}  {'─' * 7}  {'─' * 8}  {'─' * 16}"
        f"  {'─' * 8}  {'─' * 7}  {'─' * 4}"
    )

    icons = {DONE: "✓", SKIPPED: "⊘", FAILED: "✗", PENDING: "·"}
    for s in steps:
        gc = s.parsed.get("geom_converged", "")
        flag = " ⚠" if gc == "no" and s.status == DONE else ""
        B = s.parsed.get("B_Hill_GPa") or s.parsed.get("B_lbfgs_GPa") or "—"
        H = s.parsed.get("enthalpy_eV") or s.parsed.get("energy_ev") or "—"
        a = s.parsed.get("a_opt_ang") or "—"
        print(
            f"  {icons.get(s.status, '?')}{s.idx:>3}  "
            f"{s.concentration:>7.4f}  {s.status:<8}  "
            f"{H:>16}  {a:>8}  {B:>7}  {(gc or '—')}{flag}"
        )

    counts = {
        st: sum(1 for s in steps if s.status == st)
        for st in (DONE, SKIPPED, FAILED, PENDING)
    }
    print(
        f"{'═' * W}\n"
        f"✓ {counts[DONE]}  ⊘ {counts[SKIPPED]}  "
        f"✗ {counts[FAILED]}  · {counts[PENDING]}"
    )


def print_retry_plan(failed_steps: list[Step]) -> None:
    section("Sweep completed — failure analysis")
    print(f"  ✗  {len(failed_steps)} step(s) failed:")
    for s in failed_steps:
        reason = s.parsed.get("kill_reason", "unknown")
        print(f"  ✗  x={s.concentration:.4f}  —  {reason}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# SQS pre-flight reporting
# ─────────────────────────────────────────────────────────────────────────────


def print_sqs_plan(
    plan, target_fractions: dict[str, float],
    template_element: str, base_atom_count: int,
    exact_extent: int | None = None,
) -> None:
    """Show the user what cell size SQS will use before any work starts."""
    section(f"SQS plan — substituting on {template_element} sublattice")
    print(
        "  Requested mixture : "
        + "  ".join(f"{e}={f:.4f}" for e, f in target_fractions.items())
    )
    print(
        f"  Chosen supercell  : "
        f"{plan.extent[0]}×{plan.extent[1]}×{plan.extent[2]}"
    )
    print(f"  Template sites    : {plan.n_template_sites}")
    print(
        f"  Total atoms       : ~{base_atom_count * (plan.extent[0] ** 3)}"
    )
    print(
        "  Site counts       : "
        + "  ".join(f"{e}={n}" for e, n in plan.integer_counts.items())
    )
    print(
        "  Achieved fractions: "
        + "  ".join(
            f"{e}={f:.4f}" for e, f in plan.achieved_fractions.items()
        )
    )
    if plan.is_exact:
        print("  ✓ Exact representation (deviation = 0)")
    else:
        print(f"  ⚠ Approximate — max deviation = {plan.max_deviation:.4f}")
        if exact_extent is not None:
            print(
                f"     For exact representation use --sqs-max {exact_extent} "
                f"(≈{base_atom_count * (exact_extent ** 3)} atoms)"
            )
    print()
