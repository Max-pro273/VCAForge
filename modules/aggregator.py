"""
aggregator.py  —  Collects per-run vca_results.csv files into a master CSV.
═══════════════════════════════════════════════════════════════════════════
After many active-learning iterations you may have dozens of project
directories, each with its own vca_results.csv. ResultAggregator scans
a base directory recursively, filters by SystemSpec, deduplicates, and
writes a single master CSV with the same column conventions as VCAForge.

The master CSV is:
  - Compatible with DataIngestor (so subsequent navigator runs can use it).
  - Sorted by composition (canonical x-vector ordering).
  - Annotated with provenance: source_dir, iteration, suggested_by.
  - VCAForge-style header: '# System :', '# Generated:', '# Sources:'.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

import config
from navigator import (
    SystemSpec,
    parse_system_from_csv,
    parse_bracket_header_with_fractions,
)

log = logging.getLogger("vcaforge.aggregator")


def _cfg(name: str, default: Any) -> Any:
    return getattr(config, name, default)


def _master_suffix() -> str:
    return _cfg("NAVIGATOR_MASTER_CSV_SUFFIX", "_master.csv")


def _csv_filename() -> str:
    return _cfg("CSV_FILE", "vca_results.csv")


@dataclass
class AggregationStats:
    """Summary of an aggregation pass."""
    output_path: Path
    n_source_files: int = 0
    n_source_files_matched: int = 0
    n_rows_total: int = 0
    n_rows_after_dedup: int = 0
    duplicates_resolved: int = 0


class ResultAggregator:
    """Walks a directory tree, collects all vca_results.csv files for one
    SystemSpec, and writes a master CSV.

    Deduplication policy
    ────────────────────
    Two rows are duplicates if they (a) match SystemSpec exactly and (b) have
    the same composition vector within `dedupe_tol` (Euclidean) AND both have
    status="done". When duplicated:
      - "best" (default): keep the row with the better target value
        (max for maximize-mode, min for minimize-mode).
      - "mean": collapse duplicates into a single row whose numeric columns
        are the MEAN of the duplicates and whose `n_repeats` column counts
        the number of merged rows. This is the statistician's choice — it
        averages out DFT noise and gives every measurement equal weight.
      - "all": keep all rows, just sort. Useful for noise estimation.
    """

    _DEDUP_POLICIES: frozenset[str] = frozenset({"best", "mean", "all"})

    def __init__(
        self,
        target: str | None = None,
        mode: str = "maximize",
        dedupe_tol: float = 1e-3,
        dedupe_policy: str = "best",   # "best" | "mean" | "all"
    ) -> None:
        self.target = target or _cfg("NAVIGATOR_TARGET", "H_Vickers_GPa")
        self.mode = mode.lower()
        self.dedupe_tol = float(dedupe_tol)
        if dedupe_policy not in self._DEDUP_POLICIES:
            raise ValueError(
                f"dedupe_policy must be one of {sorted(self._DEDUP_POLICIES)}, "
                f"got {dedupe_policy!r}"
            )
        self.dedupe_policy = dedupe_policy

    # ── Public API ─────────────────────────────────────────────────────────

    def aggregate(
        self,
        base_dir: Path,
        system: SystemSpec,
        output_path: Path | None = None,
        include_subsystems: bool = True,
    ) -> Path:
        """Collect all CSVs under `base_dir` matching `system` into one master.

        Parameters
        ──────────
        include_subsystems : bool, default True
            When True, also include CSVs from systems whose metal set is a
            SUBSET of `system.metals` and whose nonmetal matches. For example,
            asking for 'Ti-Zr-Nb-C' will pull in Ti-Nb-C, Zr-Nb-C, Ti-Zr-C,
            Ti-C, Nb-C, and Zr-C runs as well — these points lie on the edges
            and faces of the larger simplex and are valuable training data.
            Subsystem rows are padded with x_M=0 for the metals they don't
            contain (mathematically correct: those compositions ARE on the
            boundary of the larger simplex).

        Returns the output Path. Raises FileNotFoundError if no matching
        files were found.
        """
        base_dir = Path(base_dir).expanduser().resolve()
        if not base_dir.exists():
            raise FileNotFoundError(f"Aggregator base dir not found: {base_dir}")
        if output_path is None:
            output_path = base_dir / f"{system.label()}{_master_suffix()}"
        else:
            output_path = Path(output_path).expanduser().resolve()

        files = self._discover_csvs(base_dir, exclude=output_path)
        if not files:
            raise FileNotFoundError(
                f"No '{_csv_filename()}' files found under {base_dir}."
            )

        target_metal_set = frozenset(system.metals)
        matched_files: list[Path] = []
        frames: list[pd.DataFrame] = []
        for f in files:
            try:
                sys_in_csv = parse_system_from_csv(f)
            except Exception as exc:  # noqa: BLE001
                # Hardening: never let a malformed header crash the whole run
                log.warning("Could not parse system header of %s: %s", f, exc)
                continue
            if sys_in_csv is None:
                continue
            # Match policy: equal OR (subsystem mode) metal set is a subset
            # AND nonmetal matches. Subsets-only enforcement keeps Ni-B-C OUT
            # of a Ti-Zr-Nb-C aggregation (different chemistry).
            csv_metals = frozenset(sys_in_csv.metals)
            if sys_in_csv == system:
                pass   # exact match — always include
            elif (include_subsystems
                  and csv_metals.issubset(target_metal_set)
                  and sys_in_csv.nonmetal == system.nonmetal):
                pass   # subsystem — include if nonmetal matches
            else:
                continue
            try:
                df = pd.read_csv(f, comment="#")
            except Exception as exc:  # noqa: BLE001
                log.warning("Skipping unreadable CSV %s: %s", f, exc)
                continue
            if df.empty:
                continue
            # Reconstruct per-metal x_M columns IN THE SUBSYSTEM's own column
            # names first, then expand to the target system's columns by
            # filling missing metals with 0.
            df = self._reconstruct_xm_columns(df, f, sys_in_csv)
            df = self._project_to_target_system(df, sys_in_csv, system)
            df = self._normalise_for_master(df, f, base_dir, system)
            df["origin_system"] = sys_in_csv.label()   # provenance for subsystems
            frames.append(df)
            matched_files.append(f)

        if not frames:
            raise FileNotFoundError(
                f"No CSVs under {base_dir} match system '{system.label()}'."
            )

        merged = pd.concat(frames, ignore_index=True, sort=False)

        # Ensure composition columns exist (so downstream tools can sort/dedup)
        merged = self._ensure_composition_cols(merged, system)

        # Deduplicate
        before = len(merged)
        merged = self._deduplicate(merged, system)
        dups = before - len(merged)

        # Sort by composition
        sort_cols = [c for c in system.column_names() if c in merged.columns]
        if sort_cols:
            merged = merged.sort_values(sort_cols, kind="stable").reset_index(drop=True)

        # Write
        self._write_master(merged, output_path, system, len(matched_files))

        log.info(
            "Aggregated %d CSV(s), %d row(s) → %s (deduped %d)",
            len(matched_files), len(merged), output_path, dups,
        )
        return output_path

    def aggregate_all_systems(
        self, base_dir: Path, include_subsystems: bool = True,
    ) -> dict[str, Path]:
        """Convenience: aggregate one master CSV per discovered MAXIMAL system.

        Parameters
        ──────────
        include_subsystems : bool, default True
            When True, only MAXIMAL systems get their own master CSV.
            A system S is "maximal" if no other discovered system S'
            (with the same nonmetal) has S.metals ⊊ S'.metals. The
            non-maximal systems become subsystem rows in the maximal
            CSV. This avoids producing N redundant masters when the
            data is naturally ordered by inclusion (e.g. Zr-Nb-C ⊂
            Ti-Zr-Nb-C ⊂ Ti-Zr-Nb-V-C).

            When False, every distinct system gets its own master CSV
            (legacy behaviour).

        Returns {system_label: output_path}.
        """
        base_dir = Path(base_dir).expanduser().resolve()
        if not base_dir.exists():
            raise FileNotFoundError(f"Aggregator base dir not found: {base_dir}")

        files = self._discover_csvs(base_dir, exclude=None)
        seen: dict[str, SystemSpec] = {}
        for f in files:
            try:
                s = parse_system_from_csv(f)
            except Exception as exc:  # noqa: BLE001
                log.warning("Skip %s: %s", f, exc)
                continue
            if s is not None:
                seen.setdefault(s.label(), s)

        if include_subsystems:
            target_systems = self._maximal_systems(list(seen.values()))
        else:
            target_systems = list(seen.values())

        results: dict[str, Path] = {}
        for s in target_systems:
            try:
                out = self.aggregate(
                    base_dir, s, include_subsystems=include_subsystems,
                )
                results[s.label()] = out
            except FileNotFoundError:
                continue
        return results

    @staticmethod
    def _maximal_systems(systems: list[SystemSpec]) -> list[SystemSpec]:
        """Return only the systems that are NOT proper subsets of another
        discovered system with the same nonmetal.

        Example input:  [Zr-Nb-C, Ti-Nb-C, Ti-Zr-Nb-C, Mn-Mo]
        Output:         [Ti-Zr-Nb-C, Mn-Mo]
                        (Zr-Nb-C and Ti-Nb-C are dropped because
                         they are proper subsets of Ti-Zr-Nb-C.)
        """
        out: list[SystemSpec] = []
        for s in systems:
            s_metals = frozenset(s.metals)
            dominated = False
            for other in systems:
                if other is s:
                    continue
                if other.nonmetal != s.nonmetal:
                    continue
                if s_metals < frozenset(other.metals):    # proper subset
                    dominated = True
                    break
            if not dominated:
                out.append(s)
        return out

    # ── Internals ─────────────────────────────────────────────────────────

    def _discover_csvs(self, base_dir: Path, exclude: Path | None) -> list[Path]:
        csv_name = _csv_filename()
        files = sorted(base_dir.rglob(csv_name))
        # Don't include any pre-existing master CSVs
        suffix = _master_suffix()
        files = [f for f in files if not f.name.endswith(suffix)]
        if exclude is not None:
            ex = exclude.resolve()
            files = [f for f in files if f.resolve() != ex]
        return files

    def _project_to_target_system(
        self, df: pd.DataFrame, csv_system: SystemSpec, target_system: SystemSpec,
    ) -> pd.DataFrame:
        """Project a subsystem's rows into the target system's column space.

        For metals in target_system.metals that are NOT in csv_system.metals,
        a column 'x_<metal>' is added with all zeros. This is mathematically
        correct: a row from Zr-Nb-C with x_Zr=0.5, x_Nb=0.5 represents a
        composition on the edge of the larger Ti-Zr-Nb-C simplex where x_Ti=0.
        """
        if csv_system == target_system:
            return df
        df = df.copy()
        for m in target_system.metals:
            col = f"x_{m}"
            if col not in df.columns:
                df[col] = 0.0
        return df

    def _reconstruct_xm_columns(
        self, df: pd.DataFrame, src: Path, system: SystemSpec,
    ) -> pd.DataFrame:
        """Add x_M_i columns to a single file's frame.

        Resolution order matches DataIngestor:
          1. If the frame already has all x_M columns → keep them as-is.
          2. If the source CSV is bracketed-VCA → use its inner fractions
             together with the 'concentration' column to compute per-metal x.
          3. If binary + 'concentration' → x_a = 1 - c, x_b = c.
          4. Otherwise leave the frame unchanged.
        """
        cols = system.column_names()
        if all(c in df.columns for c in cols):
            return df
        if "concentration" not in df.columns:
            return df

        df = df.copy()
        c = pd.to_numeric(df["concentration"], errors="coerce")

        # Try bracketed-VCA reconstruction
        bracket_info = parse_bracket_header_with_fractions(src)
        if bracket_info is not None:
            # Defensive: must be 2-tuple. Older versions returned 3-tuples.
            if not isinstance(bracket_info, tuple) or len(bracket_info) < 2:
                log.warning(
                    "parse_bracket_header_with_fractions returned %r — version mismatch? "
                    "Skipping bracket reconstruction for %s.", bracket_info, src,
                )
                return df
            bracket_fracs = bracket_info[1]
            for m, col in zip(system.metals, cols):
                frac = float(bracket_fracs.get(m, 0.0))
                if frac >= 0.999:   # pure-metal sentinel (frac==1.0)
                    df[col] = (1.0 - c)
                else:
                    df[col] = c * frac
            return df

        # Binary fallback
        if system.is_binary():
            df[cols[0]] = 1.0 - c
            df[cols[1]] = c
        return df

    def _normalise_for_master(
        self, df: pd.DataFrame, src: Path, base_dir: Path, system: SystemSpec,
    ) -> pd.DataFrame:
        """Add provenance columns; keep all original columns intact."""
        df = df.copy()

        # Provenance: source directory (relative to base) and a guess at iteration
        try:
            rel = src.parent.relative_to(base_dir)
            df["source_dir"] = str(rel)
        except ValueError:
            df["source_dir"] = str(src.parent)

        # Iteration: best-effort. If a directory is named "..._iter_N", parse N.
        # Otherwise leave 0 (sweep) — the loop driver can overwrite it explicitly.
        if "iteration" not in df.columns:
            df["iteration"] = self._guess_iteration(src.parent.name)
        if "suggested_by" not in df.columns:
            # If the parent run was a navigator iteration we'd ideally know,
            # but for general aggregation we assume "sweep" unless overridden.
            df["suggested_by"] = "sweep"

        return df

    @staticmethod
    def _guess_iteration(dir_name: str) -> int:
        """Extract iteration number from a directory name like 'foo_iter_3'."""
        import re
        m = re.search(r"iter[_-]?(\d+)", dir_name, re.IGNORECASE)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                pass
        return 0

    def _ensure_composition_cols(
        self, df: pd.DataFrame, system: SystemSpec,
    ) -> pd.DataFrame:
        """Ensure x_M_i columns exist; back-fill from 'concentration' for binary."""
        cols = system.column_names()
        if all(c in df.columns for c in cols):
            return df
        if system.is_binary() and "concentration" in df.columns:
            c = pd.to_numeric(df["concentration"], errors="coerce")
            df = df.copy()
            df[cols[0]] = 1.0 - c
            df[cols[1]] = c
            return df
        # Else: leave as-is; downstream tools will see partial data
        return df

    def _deduplicate(self, df: pd.DataFrame, system: SystemSpec) -> pd.DataFrame:
        if self.dedupe_policy == "all":
            return df
        cols = [c for c in system.column_names() if c in df.columns]
        if not cols or self.target not in df.columns:
            return df

        # Only deduplicate among status=done rows
        if "status" in df.columns:
            done_mask = df["status"].fillna("").astype(str).str.lower() == "done"
        else:
            done_mask = pd.Series([True] * len(df), index=df.index)

        done = df[done_mask].copy()
        rest = df[~done_mask]

        # Round compositions to dedupe_tol precision for grouping
        round_decimals = max(0, int(round(-np.log10(self.dedupe_tol))))
        keys = done[cols].round(round_decimals)
        done["_dedupe_key"] = list(map(tuple, keys.itertuples(index=False, name=None)))

        if self.dedupe_policy == "mean":
            # Mean-merge: for each duplicate group, average the numeric columns
            # and concatenate string-valued columns. Preserves a count via n_repeats.
            numeric_cols = [c for c in done.columns
                            if c not in {"_dedupe_key"} and pd.api.types.is_numeric_dtype(done[c])]
            string_cols = [c for c in done.columns
                           if c not in {"_dedupe_key"} and c not in numeric_cols]

            agg_rows: list[dict] = []
            for _key, group in done.groupby("_dedupe_key", sort=False):
                row: dict = {}
                for c in numeric_cols:
                    row[c] = float(pd.to_numeric(group[c], errors="coerce").mean(skipna=True))
                for c in string_cols:
                    vals = group[c].fillna("").astype(str).unique()
                    # If all the same → keep the value; else join with '|'
                    row[c] = vals[0] if len(vals) == 1 else "|".join(v for v in vals if v)
                row["n_repeats"] = int(len(group))
                agg_rows.append(row)
            kept = pd.DataFrame(agg_rows)
        else:
            # "best": pick row with highest (or lowest) target value per group
            target_vals = pd.to_numeric(done[self.target], errors="coerce")
            kept_idx: list[int] = []
            for _key, group in done.groupby("_dedupe_key", sort=False):
                ys = target_vals.loc[group.index]
                valid = group.index[ys.notna()]
                if len(valid) == 0:
                    kept_idx.append(group.index[0])
                    continue
                if self.mode == "minimize":
                    pick = ys.loc[valid].idxmin()
                else:
                    pick = ys.loc[valid].idxmax()
                kept_idx.append(pick)
            kept = done.loc[kept_idx].drop(columns=["_dedupe_key"])

        out = pd.concat([kept, rest], ignore_index=True, sort=False)
        return out

    def _write_master(
        self,
        df: pd.DataFrame,
        output_path: Path,
        system: SystemSpec,
        n_sources: int,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Keep one rolling backup of the previous master (if any) so users
        # can recover from a regression in dedup logic or compare runs.
        if output_path.exists():
            try:
                bak = output_path.with_suffix(output_path.suffix + ".bak")
                bak.write_bytes(output_path.read_bytes())
            except OSError as exc:
                log.warning("Could not create backup %s: %s", bak, exc)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metals_kv = ",".join(system.metals)
        nm_kv = system.nonmetal or ""
        header = (
            f"# VCAForge Navigator Master CSV\n"
            f"# System  : {system.label()}  metals={metals_kv}  nonmetal={nm_kv}\n"
            f"# Generated: {timestamp}\n"
            f"# Sources: {n_sources} directories\n"
            f"# Dedup   : policy={self.dedupe_policy}  tol={self.dedupe_tol}\n"
            f"#\n"
        )
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(header)
            df.to_csv(f, index=False)
