# VCAForge

**Automated VCA concentration sweeps for CASTEP 25.**  
From a single `.cell` or `.cif` file to a full Ti(1-x)Nb(x) sweep with ΔH_mix table — in one command.

---

## Why this tool

Most CASTEP workflows require manually editing `.cell` files, copying folders, and tracking results in a spreadsheet. `vca_tool` replaces all of that:

- **Runs entirely from one command.** No scripting, no template editing.
- **Adaptive physics, not hardcoded defaults.** `nextra_bands` is calculated per concentration step using Vegard-weighted interpolation — pure endpoints get fewer bands, VCA midpoints get more. Cutoff energy, spin polarisation, and nextra are auto-detected from the elements in your cell.
- **Crash-safe by design.** Every step is checkpointed to `vca_state.json` before CASTEP starts. Power cut mid-run? Resume with `--resume` and continue from where you left off. No results are ever lost.
- **Ctrl+C skips, not quits.** Press Ctrl+C during a running step — it terminates that step gracefully, marks it `skipped`, and moves on. All completed steps stay intact.
- **Works with any cell.** TiC ceramics, NbMo alloys, ternary systems. The tool warns you about physics issues (magnetic elements, cross-sublattice mixing) but never blocks you — you decide whether to proceed.
- **Dynamic CSV output.** Columns adapt to what CASTEP actually computed. Run an `ElasticConstants` task later and the elastic moduli appear in the same CSV automatically — no hardcoded field list.

---

## Setup

**Requirements:** Python 3.10+, CASTEP 25, `cif2cell` (optional, for `.cif` input).

```bash
pip install -r requirements.txt
```

No further configuration needed. The tool detects your CASTEP binary at first run and saves the command. If the binary moves, it asks you to fix it on the next resume.

To use a non-default CASTEP binary once:
```bash
python main.py NbMo.cell --castep-cmd 'mpirun -n 4 /path/to/castep.mpi {seed}'
```
Or change permanently in `castep_io.py` on line 65:
```python
CASTEP_ENGINE = EngineConfig(
    name="CASTEP",
    default_bin=("castep.mpi"), # <- here
    cmd_template="mpirun -n {ncores} {bin} {seed}",
    input_suffix=".cell",
    output_suffix=".castep",
)
```

---

## Quick start

```bash
# Interactive wizard — asks species, range, CASTEP command
python main.py TiC.cif

# Full VCA sweep, 8 intervals, no prompts
python main.py TiC.cell --species Ti Nb --range 0 1 8 
# (0 - start, 1 - finish, 8 - intervals)

# Single compound (two ways)
python main.py TiC.cell --single
python main.py TiC.cell --species Ti Ti

# Resume after crash or Ctrl+C
python main.py TiC.cell --resume

# Pause before each step to review
python main.py TiC.cif --interactive

# Keep large output files (.check, .bands)
python main.py NbMo.cell --keep-all
```

For the full flag reference:
```bash
python main.py --help
```

---

## What gets created

```
TiC_Mar17_21-29/
├── original_TiC.cell       ← your original file, never modified
├── original_TiC.param      ← your original param, never modified
├── vca_state.json           ← crash-safe checkpoint
├── vca_results.csv          ← live results table, updated after every step
├── run.log
├── x0.0000/
│   ├── TiC.cell             ← VCA cell at x=0 (pure Ti)
│   ├── TiC.param            ← nextra_bands patched for this concentration
│   ├── TiC.castep
│   └── TiC-out.cell
├── x0.3333/
│   └── ...                  ← nextra_bands interpolated for x=0.33
└── x1.0000/
    └── ...                  ← pure Nb endpoint
```

The source `.cell` and `.param` in your working directory are **never modified**.

---

## Interactive wizard

When run without `--species`, the tool asks:

1. **Which element to substitute** (must exist in the `.cell`)
2. **Replace with what** — enter the same element twice to run a single-compound calculation instead of a sweep
3. **Concentration range** — start, end, number of intervals
4. **Task, XC functional, cutoff energy, spin** — with smart defaults pre-filled based on detected elements
5. **MPI process count** and CASTEP binary path

Physics warnings (magnetic elements, cross-sublattice mixing) are shown as `⚠` notes — you always decide whether to continue.

## Module structure

The tool is four files with clear separation of concerns:

| File | Responsibility |
|------|---------------|
| `main.py` | Entry point, argument parsing, run loop |
| `ui.py` | All console interaction — wizards, prompts, result display |
| `castep_io.py` | File I/O — reads `.cell`, writes VCA cells, generates `.param`, parses `.castep` output |
| `workflow.py` | Process management — state JSON, CASTEP subprocess, CSV export, ΔH_mix |

**Adding a new engine** (e.g. VASP): create a new `vasp_io.py` following the same interface as `castep_io.py` (`write_cell`, `write_param`, `parse_log`), and swap the import in `workflow.py`. The UI and state management stay unchanged.

---

## Tested on

- Ryzen 5 4500U | CachyOS x86_64
- CASTEP 25.12, GFortran 15.2.1, OpenMPI
- Python 3.11 / 3.14
- `.cell` files generated by `cif2cell` from Materials Project CIFs
