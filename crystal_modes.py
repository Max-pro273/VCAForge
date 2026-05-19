"""
crystal_modes.py  —  Crystal preparation strategies (VCA / SQS / Direct).
═══════════════════════════════════════════════════════════════════════════
Engine-agnostic. Each strategy receives a primitive Crystal plus a
composition spec and returns a PreparedCrystal that the engine writes
verbatim. Engines never know which mode produced the Crystal.

Engines declare what they accept via BaseEngine.SUPPORTED_MODES.
Per-mode advisories (e.g. VASP+VCA being non-physical) live in
BaseEngine.MODE_WARNINGS and are surfaced once, in main.py, before the
sweep starts.

Modes
─────
    vca     — Vegard-scaled primitive cell + fractional target_mix.
              Engines render fractional occupancy on disk
              (CASTEP MIXTURE:, VASP VCA= tag).
    sqs     — Quasirandom supercell with integer site occupations,
              built via sqsgenerator 0.5.x in `split` sublattice mode.
              Non-substituted sublattices (e.g. C in TiC, O in oxides)
              stay pinned.
    direct  — Pass-through. Single species, no mixing, no supercell.

SQS specifics
─────────────
* Library: sqsgenerator >= 0.5.0 (`pip install sqsgenerator`).
* Crystal is passed inline (lattice + coords + species) — no
  POSCAR/CIF round-trip via disk into sqsgenerator.
* Output is persisted to <step_dir>/sqs/ for reproducibility:
    sqs/sqsconfig.json   — exact input to sqsgenerator.optimize()
    sqs/sqs.poscar       — best structure as POSCAR
    sqs/sqsresult.json   — best objective, supercell, fractions
* User supplies a max supercell extent (e.g. --sqs-max 3 → up to 3×3×3).
  Strategy picks the SMALLEST supercell whose template-sublattice site
  count makes requested fractions exactly representable as integers.
  If no exact fit fits within max_extent, the closest cell is used
  with the achieved-vs-requested deviation logged for the user.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

import config
from core_physics import Crystal


# ─────────────────────────────────────────────────────────────────────────────
# Result envelope
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PreparedCrystal:
    """Output of a CrystalModeStrategy. The engine consumes this object."""

    crystal: Crystal
    metadata: dict[str, str] = field(default_factory=dict)
    artifacts: tuple[Path, ...] = ()


# ─────────────────────────────────────────────────────────────────────────────
# Strategy protocol
# ─────────────────────────────────────────────────────────────────────────────


@runtime_checkable
class CrystalModeStrategy(Protocol):
    """A pure function: composition spec → PreparedCrystal."""

    name: str

    def prepare(
        self,
        base_crystal: Crystal,
        template_element: str,
        target_mix: dict[str, float],
        x: float,
        step_dir: Path,
        verbose: bool = True,
    ) -> PreparedCrystal: ...


# ─────────────────────────────────────────────────────────────────────────────
# VCA
# ─────────────────────────────────────────────────────────────────────────────


class VCAStrategy:
    """Virtual Crystal Approximation: average potentials on a single site."""

    name = "vca"

    def prepare(
        self,
        base_crystal: Crystal,
        template_element: str,
        target_mix: dict[str, float],
        x: float,
        step_dir: Path,
        verbose: bool = True,
    ) -> PreparedCrystal:
        # 1. Vegard scaling
        eps = 1e-9
        nonzero_mix = {e: f for e, f in target_mix.items() if f > eps}
        
        r_template = config.ELEMENTS.get(template_element.capitalize(), {}).get("rad", 0.0)
        
        scaled_lattice = base_crystal.lattice.copy()
        scale_factor = 1.0

        if r_template > 1e-6 and nonzero_mix:
            total_new_frac = sum(nonzero_mix.values())
            template_remaining = max(0.0, 1.0 - total_new_frac)
            
            r_mix = r_template * template_remaining + sum(
                config.ELEMENTS.get(e.capitalize(), {}).get("rad", 0.0) * f
                for e, f in nonzero_mix.items()
            )
            
            if r_mix > 1e-6:
                scale_factor = r_mix / r_template
                scaled_lattice = base_crystal.lattice * scale_factor

        # 2. Create new sites with fractional occupancy
        new_sites = []
        template_element_lower = template_element.casefold()
        for site in base_crystal.sites:
            current_species = next(iter(site.keys()))
            if current_species.casefold() == template_element_lower:
                new_sites.append(target_mix)
            else:
                new_sites.append(site)
        
        vca_crystal = Crystal(
            lattice=scaled_lattice,
            frac_coords=base_crystal.frac_coords.copy(),
            sites=new_sites
        )

        return PreparedCrystal(
            crystal=vca_crystal,
            metadata={
                "strategy": "vca",
                "vca_vegard_k": f"{scale_factor:.6f}",
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
# Direct
# ─────────────────────────────────────────────────────────────────────────────


class DirectStrategy:
    """No mixing. Used for pure end-members and single-compound runs."""

    name = "direct"

    def prepare(
        self,
        base_crystal: Crystal,
        template_element: str,
        target_mix: dict[str, float],
        x: float,
        step_dir: Path,
        verbose: bool = True,
    ) -> PreparedCrystal:
        nonzero = {e: f for e, f in target_mix.items() if f > 1e-9}
        if len(nonzero) != 1:
            raise ValueError(
                f"Direct mode requires exactly one species on the template "
                f"sublattice, got: {nonzero}. Use VCA or SQS for mixtures."
            )
        sole = next(iter(nonzero))
        
        new_sites = []
        template_element_lower = template_element.casefold()
        for site in base_crystal.sites:
            # Assuming single species in base_crystal sites
            current_species = next(iter(site.keys()))
            if current_species.casefold() == template_element_lower:
                new_sites.append({sole: 1.0})
            else:
                new_sites.append(site)

        return PreparedCrystal(
            crystal=Crystal(
                lattice=base_crystal.lattice.copy(),
                frac_coords=base_crystal.frac_coords.copy(),
                sites=new_sites,
            ),
            metadata={"strategy": "direct"},
        )


# ─────────────────────────────────────────────────────────────────────────────
# SQS supercell math
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SupercellPlan:
    """Chosen supercell + the rounding it implies.

    dims is the actual (nx, ny, nz) triple — NOT forced to be cubic.
    The anisotropic search (from sqs.py) finds rectangular cells that
    reduce anisotropy penalty before growing volume, giving better
    SQS quality in fewer atoms.
    """

    dims: tuple[int, int, int]
    n_template_sites: int
    integer_counts: dict[str, int]
    achieved_fractions: dict[str, float]
    max_deviation: float
    is_exact: bool

    # Back-compat: plan.extent used in ui.print_sqs_plan and main.py
    @property
    def extent(self) -> tuple[int, int, int]:
        return self.dims


def plan_supercell(
    target_fractions: dict[str, float],
    n_template_primitive: int,
    max_extent: int,
) -> SupercellPlan:
    """Find the smallest anisotropic (nx, ny, nz) supercell with minimum
    concentration error.

    Search strategy (from sqs.py):
      1. Enumerate ALL (nx, ny, nz) with each dimension in [1, max_extent].
      2. Sort by (anisotropy = max−min, volume = nx*ny*nz) — prefer cubic,
         then small — so we try the most efficient cells first.
      3. Return the first cell where max_deviation < 1e-9 (exact fit),
         or the cell with globally minimum max_deviation if no exact fit.

    This is strictly better than the previous cubic-only search (k,k,k):
    for Ti₀.₃₃Nb₀.₆₇ with 1 Ti per primitive cell, a 2×1×1 supercell
    gives 2 sites → {Ti:1, Nb:1} (exact, 33%), whereas 1×1×1 is wrong
    and 2×2×2 is 8 sites (over-kill). The anisotropic search finds 2×1×1
    immediately.
    """
    if max_extent < 1:
        raise ValueError(f"max_extent must be >= 1, got {max_extent}")

    eps = 1e-9
    active = {e: f for e, f in target_fractions.items() if f > eps}
    if not active:
        raise ValueError(
            f"plan_supercell: no significant components in {target_fractions}"
        )
    total = sum(active.values())
    if abs(total - 1.0) > 1e-6:
        active = {e: f / total for e, f in active.items()}

    # Build candidate list: (anisotropy, volume, (nx, ny, nz))
    options: list[tuple[int, int, tuple[int, int, int]]] = []
    for nx in range(1, max_extent + 1):
        for ny in range(1, max_extent + 1):
            for nz in range(1, max_extent + 1):
                vol  = nx * ny * nz
                anis = max(nx, ny, nz) - min(nx, ny, nz)
                options.append((anis, vol, (nx, ny, nz)))
    # Priority: 1. cubicity (low anis), 2. size (low vol)
    options.sort(key=lambda t: (t[0], t[1]))

    best: SupercellPlan | None = None
    for _, _, dims in options:
        n_sites = n_template_primitive * dims[0] * dims[1] * dims[2]
        counts  = _largest_remainder_rounding(active, n_sites)
        achieved = {e: c / n_sites for e, c in counts.items()}
        max_dev  = max(abs(achieved[e] - active[e]) for e in active)

        plan = SupercellPlan(
            dims=dims,
            n_template_sites=n_sites,
            integer_counts=counts,
            achieved_fractions=achieved,
            max_deviation=max_dev,
            is_exact=max_dev < eps,
        )
        if plan.is_exact:
            return plan
        if best is None or max_dev < best.max_deviation - eps:
            best = plan

    assert best is not None
    return best


def exact_supercell_extent(
    target_fractions: dict[str, float],
    n_template_primitive: int,
    *,
    limit_denominator: int = 1000,
    max_search: int = 50,
) -> int:
    """Return the cubic extent k such that k**3 * n_template_primitive is
    exactly divisible by the LCM of denominators of `target_fractions`.

    This is the smallest CUBIC cell that admits zero-rounding.
    Use plan_supercell() to find a potentially smaller ANISOTROPIC cell.
    The return value is used only by ui.print_sqs_plan to print a hint
    ("use --sqs-max N for exact representation").
    """
    eps = 1e-9
    active = {e: f for e, f in target_fractions.items() if f > eps}
    if not active:
        return 1
    fr = {
        e: Fraction(f).limit_denominator(limit_denominator)
        for e, f in active.items()
    }
    denoms = [f.denominator for f in fr.values()]
    lcm = math.lcm(*denoms) if denoms else 1

    for k in range(1, max_search + 1):
        if (k ** 3 * n_template_primitive) % lcm == 0:
            return k
    return max_search


def _largest_remainder_rounding(
    fractions: dict[str, float], n_sites: int
) -> dict[str, int]:
    """Convert fractions → integer counts that sum to exactly n_sites."""
    scaled = {e: f * n_sites for e, f in fractions.items()}
    floors = {e: int(math.floor(v)) for e, v in scaled.items()}
    deficit = n_sites - sum(floors.values())
    remainders = sorted(
        ((scaled[e] - floors[e], e) for e in fractions),
        reverse=True,
    )
    for _, e in remainders[: max(0, deficit)]:
        floors[e] += 1
    return floors


# ─────────────────────────────────────────────────────────────────────────────
# SQS strategy — sqsgenerator 0.5.x API
# ─────────────────────────────────────────────────────────────────────────────


class SQSStrategy:
    """Quasirandom supercell via sqsgenerator's `split` sublattice mode.

    Anisotropic supercell search: tries all (nx, ny, nz) combinations up to
    max_extent in each dimension, sorted by (anisotropy, volume). This matches
    the sqs.py standalone script behaviour and finds smaller cells for
    non-symmetric compositions (e.g. Ti₀.₃₃Nb₀.₆₇ → 2×1×1 not 2×2×2).

    n_threads: passed to sqsgenerator as thread_config=[n] (parallel MC).
    Defaults to config.SQS_THREADS (set to 0 to let sqsgenerator auto-pick).
    """

    name = "sqs"

    def __init__(
        self,
        max_extent: int | None = None,
        iterations: int | None = None,
        shell_weights: dict[int, float] | None = None,
        n_threads: int | None = None,
    ) -> None:
        self.max_extent    = max_extent   or config.SQS_MAX_EXTENT
        self.iterations    = iterations   or config.SQS_ITERATIONS
        self.shell_weights = shell_weights or config.SQS_SHELL_WEIGHTS
        # 0 = let sqsgenerator choose automatically (uses all available cores)
        self.n_threads     = n_threads if n_threads is not None else config.SQS_THREADS

    # Planning helper for main.py pre-flight reporting.
    @staticmethod
    def plan(
        base_crystal: Crystal,
        template_element: str,
        target_fractions: dict[str, float],
        max_extent: int,
    ) -> SupercellPlan:
        template_element_lower = template_element.casefold()
        n_template_primitive = sum(
            1 for site in base_crystal.sites
            if next(iter(site.keys())).casefold() == template_element_lower
        )
        if n_template_primitive == 0:
            raise ValueError(
                f"SQS plan: template element '{template_element}' not present "
                f"in crystal species {sorted(set(base_crystal.species))}"
            )
        return plan_supercell(
            target_fractions, n_template_primitive, max_extent
        )

    def prepare(
        self,
        base_crystal: Crystal,
        template_element: str,
        target_mix: dict[str, float],
        x: float,
        step_dir: Path,
        verbose: bool = True,
    ) -> PreparedCrystal:
        try:
            from sqsgenerator import StructureFormat, optimize, parse_config  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "SQS mode requires sqsgenerator >= 0.5.0. "
                "Install: pip install sqsgenerator"
            ) from exc

        plan = self.plan(
            base_crystal, template_element, target_mix, self.max_extent
        )

        # ── Console summary (matches sqs.py output) ───────────────────────────
        if verbose:
            nx, ny, nz = plan.dims
            print(
                f"\n  ── SQS supercell x={x:.4f} ──────────────────────────────\n"
                f"  Dimensions    : {nx}×{ny}×{nz}\n"
                f"  Template sites: {plan.n_template_sites}  "
                f"({'exact' if plan.is_exact else f'error {plan.max_deviation*100:.4f}%'})\n"
                "  Distribution  :"
            )
            for el, count in plan.integer_counts.items():
                ach = plan.achieved_fractions.get(el, 0.0) * 100
                tgt = target_mix.get(el, 0.0) * 100
                print(f"    {el:2s}: {count:4d}  actual {ach:6.2f}%  target {tgt:6.2f}%")
            print("  ─────────────────────────────────────────────────────────")
        # ─────────────────────────────────────────────────────────────────────

        if len(plan.integer_counts) == 1:
            from ase import Atoms
            from ase.build import make_supercell
            import numpy as np

            # Bypass sqsgenerator optimization for 100% pure substitutions
            el = next(iter(plan.integer_counts.keys()))
            symbols = [max(s, key=s.get).capitalize() for s in base_crystal.sites]
            atoms = Atoms(
                symbols=symbols, 
                scaled_positions=base_crystal.frac_coords, 
                cell=base_crystal.lattice, 
                pbc=True
            )
            P = np.diag(plan.dims)
            super_atoms = make_supercell(atoms, P)

            syms = np.array(super_atoms.get_chemical_symbols())
            syms[syms == template_element.capitalize()] = el.capitalize()
            super_atoms.set_chemical_symbols(syms)

            sites = [{str(s).capitalize(): 1.0} for s in super_atoms.get_chemical_symbols()]
            crystal = Crystal(
                lattice=np.asarray(super_atoms.cell),
                frac_coords=super_atoms.get_scaled_positions(),
                sites=sites
            )

            sqs_dir = step_dir / config.SQS_SUBDIR
            sqs_dir.mkdir(parents=True, exist_ok=True)
            
            return PreparedCrystal(
                crystal=crystal,
                metadata={
                    "strategy":             "sqs_bypass_pure",
                    "sqs_supercell":        "x".join(str(n) for n in plan.dims),
                    "sqs_n_template_sites": str(plan.n_template_sites),
                    "sqs_objective":        "0.000000",
                    "sqs_iterations":       "0",
                    "sqs_max_deviation":    "0.000000",
                    "sqs_is_exact":         "yes",
                },
                artifacts=(),
            )

        cfg_dict = self._build_sqsgen_config(
            base_crystal, template_element, plan
        )

        # parse_config returns SqsConfiguration OR ParseError — it does
        # NOT raise. (Root cause of the AttributeError users hit: passing
        # a ParseError into optimize() crashes inside .copy().)
        #
        # CAVEAT: sqsgenerator 0.5.6 has a refcount bug — keeping a
        # ParseError object around can cause segfaults at interpreter
        # shutdown. We extract its fields immediately and drop the ref.
        parsed = parse_config(cfg_dict)
        if type(parsed).__name__ == "ParseError":
            err_msg   = str(parsed.msg)
            err_param = str(getattr(parsed, "parameter", "?"))
            err_key   = str(getattr(parsed, "key", "?"))
            del parsed
            cfg_repr = dict(cfg_dict)
            cfg_repr["shell_weights"] = {
                str(k): float(v) for k, v in cfg_dict["shell_weights"].items()
            }
            raise RuntimeError(
                "sqsgenerator rejected the SQS configuration.\n"
                f"  message  : {err_msg}\n"
                f"  parameter: {err_param}\n"
                f"  key      : {err_key}\n"
                f"  config   : {json.dumps(cfg_repr, default=str)[:600]}"
            )

        # Persist the input config BEFORE running. JSON cannot serialise
        # int dict keys, so convert shell_weights → str-keyed for the file.
        sqs_dir = step_dir / config.SQS_SUBDIR
        sqs_dir.mkdir(parents=True, exist_ok=True)
        cfg_for_disk = dict(cfg_dict)
        cfg_for_disk["shell_weights"] = {
            str(k): float(v) for k, v in cfg_dict["shell_weights"].items()
        }
        cfg_path = sqs_dir / "sqsconfig.json"
        cfg_path.write_text(
            json.dumps(cfg_for_disk, indent=2, default=str), encoding="utf-8"
        )

        pack = optimize(parsed)
        best = pack.best()
        sqsgen_struct = best.structure()
        crystal = self._sqsgen_to_crystal(sqsgen_struct)

        # Persist outputs
        poscar_text = sqsgen_struct.dump(StructureFormat.poscar)
        poscar_path = sqs_dir / "sqs.poscar"
        poscar_path.write_text(poscar_text, encoding="utf-8")

        result_path = sqs_dir / "sqsresult.json"
        result_path.write_text(
            json.dumps(
                {
                    "objective":           float(best.objective),
                    "n_template_sites":    plan.n_template_sites,
                    "supercell":           list(plan.dims),
                    "integer_counts":      plan.integer_counts,
                    "achieved_fractions":  plan.achieved_fractions,
                    "requested_fractions": target_mix,
                    "max_deviation":       plan.max_deviation,
                    "is_exact":            plan.is_exact,
                    "symbols":             list(sqsgen_struct.symbols),
                    "n_threads":           self.n_threads,
                },
                indent=2, default=str,
            ),
            encoding="utf-8",
        )
        print(
            f"  ✓ SQS x={x:.4f} done.  "
            f"Objective (penalty) = {float(best.objective):.6f}"
        )

        return PreparedCrystal(
            crystal=crystal,
            metadata={
                "strategy":             "sqs",
                "sqs_supercell":        "x".join(str(n) for n in plan.dims),
                "sqs_n_template_sites": str(plan.n_template_sites),
                "sqs_objective":        f"{float(best.objective):.6f}",
                "sqs_iterations":       str(self.iterations),
                "sqs_max_deviation":    f"{plan.max_deviation:.6f}",
                "sqs_is_exact":         "yes" if plan.is_exact else "no",
            },
            artifacts=(cfg_path, poscar_path, result_path),
        )

    def _build_sqsgen_config(
        self,
        base_crystal: Crystal,
        template_element: str,
        plan: SupercellPlan,
    ) -> dict:
        """Build JSON-clean dict for sqsgenerator 0.5.x.

        Critical 0.5.x details:
          • sublattice_mode is a STRING ('split' / 'interact').
          • shell_weights keys are INTEGER (int) when calling parse_config()
            from Python; they become strings only in the JSON file on disk.
          • composition is a LIST of dicts, each with 'sites' + per-element
            integer counts that sum to the sublattice site count exactly.
          • supercell is a LIST [nx, ny, nz] — anisotropic dims from plan.dims.
          • thread_config=[n_threads] enables parallel Monte-Carlo; 0 = auto.
        """
        species_list = [
            max(site, key=site.get).capitalize() for site in base_crystal.sites
        ]
        cfg: dict = {
            "structure": {
                "lattice":   base_crystal.lattice.tolist(),
                "coords":    base_crystal.frac_coords.tolist(),
                "species":   species_list,
                "supercell": list(plan.dims),   # (nx, ny, nz) — may be anisotropic
            },
            "iterations": int(self.iterations),
            "shell_weights": {
                int(k): float(v) for k, v in self.shell_weights.items()
            },
            "sublattice_mode": "split",
            "composition": [
                {
                    "sites": template_element.capitalize(),
                    **{el.capitalize(): int(n)
                       for el, n in plan.integer_counts.items() if n > 0},
                }
            ],
        }
        # thread_config is optional; sqsgenerator ignores the key if absent.
        # A value of 0 lets sqsgenerator choose automatically.
        if self.n_threads != 0:
            cfg["thread_config"] = [int(self.n_threads)]
        return cfg

    @staticmethod
    def _sqsgen_to_crystal(struct) -> Crystal:
        """Convert sqsgenerator Structure → VCAForge Crystal.

        sqsgenerator 0.5.x Structure attributes:
            .lattice       (3, 3) ndarray-compatible
            .frac_coords   (N, 3) ndarray-compatible
            .symbols       list[str] of atomic symbols
        """
        sites = [{str(s).capitalize(): 1.0} for s in struct.symbols]
        return Crystal(
            lattice=np.asarray(struct.lattice, dtype=float),
            frac_coords=np.asarray(struct.frac_coords, dtype=float),
            sites=sites,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────


MODES: dict[str, CrystalModeStrategy] = {
    "vca": VCAStrategy(),
    "direct": DirectStrategy(),
    "sqs": SQSStrategy(),
}


def get_strategy(name: str) -> CrystalModeStrategy:
    if name not in MODES:
        raise ValueError(
            f"Unknown crystal mode: {name!r}. Available: {sorted(MODES)}"
        )
    return MODES[name]


def register_strategy(name: str, strategy: CrystalModeStrategy) -> None:
    """Override a registered strategy. Used by main.py to install a
    per-run SQSStrategy with the user's --sqs-max."""
    MODES[name] = strategy
