# VCAForge

**Automated VCA concentration sweeps for CASTEP 25.**  
From a single `.cell` or `.cif` file to a full Ti(1-x)Nb(x) sweep with elastic constants — in one command.

---

## Why this tool

Most CASTEP workflows require manually editing `.cell` files, copying folders, and tracking results in a spreadsheet. VCAForge replaces all of that:

- **One command.** No scripting, no template editing.
- **Adaptive physics.** `nextra_bands`, cutoff energy, and spin polarisation are auto-detected and adjusted per concentration step — pure endpoints get fewer bands, VCA midpoints with high VEC get more.
- **VEC Stability Predictor.** Before the run starts, the tool calculates the Valence Electron Count (VEC) for every planned composition and warns you if any step is likely to crash CASTEP or yield C44 < 0 (Born instability). You choose to skip or proceed.
- **Crash-safe.** Every step is checkpointed to `vca_state.json` before CASTEP starts. Power cut mid-run? `--resume` picks up exactly where you left off.
- **Ctrl+C skips, not quits.** Terminates the current step gracefully, marks it `skipped`, and continues. All completed steps stay intact.
- **Elastic constants built-in.** The `--elastic` flag runs a full finite-strain Cij workflow after geometry optimisation — no external scripts needed, pure Python + numpy.

---

## Setup

**Requirements:** Python 3.10+, CASTEP 25, numpy. `cif2cell` is optional for `.cif` input.

```bash
pip install -r requirements.txt
```

No further configuration needed. The tool detects your CASTEP binary at first run. To use a custom path:

```bash
python main.py NbMo.cell --castep-cmd 'mpirun -n 4 /path/to/castep.mpi {seed}'
```

Or set it permanently in `castep_io.py`:

```python
CASTEP_ENGINE = EngineConfig(
    name="CASTEP",
    default_bin="castep.mpi",
    cmd_template="mpirun -n {ncores} {bin} {seed}",
    ...
)
```

---

## Quick start

```bash
# Interactive wizard — asks species, range, CASTEP command
python main.py TiC.cif

# Full VCA sweep, 8 intervals
python main.py TiC.cell --species Ti Nb --range 0 1 8

# With elastic constants after each GeomOpt
python main.py TiC.cell --species Ti Nb --range 0 1 8 --elastic

# Single compound (two equivalent ways)
python main.py TiC.cell --single
python main.py TiC.cell --species Ti Ti

# Resume after crash
python main.py TiC.cell --resume

# Pause before each step
python main.py TiC.cif --interactive

# Keep large output files (.check, .bands)
python main.py NbMo.cell --keep-all
```

---

## VEC Stability Predictor

Before starting a sweep with `--elastic`, the tool calculates the **Valence Electron Count** for every planned composition and prints a forecast table:

```
── VEC Stability Forecast ──────────────────────────────────────
     x      VEC   Status
──────   ──────   ──────────────────────────────────────────────
0.0000     8.00   ● Stable — SCF converges quickly
0.2500     8.25   ◑ Yellow zone — SCF may need extra iterations
0.5000     8.50   ○ RED ZONE — Born instability likely (C44 → 0)
```

If any step exceeds VEC = 8.4, you get an interactive warning:

```
┌─────────────────────────────────────────────────────────────────────┐
│  ⚠  WARNING: High Valence Electron Concentration (VEC > 8.4)        │
│     detected for x ≥ 0.5000  (VEC = 8.50).                          │
│                                                                     │
│  [1]  Skip unstable steps (run only x ≤ 0.3750)  ← Recommended      │
│  [2]  Proceed anyway  (may hang for hours or crash)                 │
└─────────────────────────────────────────────────────────────────────┘
```

Option `[1]` filters out the dangerous steps. Elastic constants for skipped intermediate x are filled in automatically via **Vegard linear interpolation** from the pure endpoints (x=0 and x=1), which are always stable.

The VEC = 8.4 threshold is based on the Band Jahn-Teller instability observed in Ti(1-x)V(x)C and Ti(1-x)Nb(x)C systems (Mei et al. 2014). For VEC > 8.4, C44 approaches zero and CASTEP either fails to converge or returns a mechanically unstable tensor.

---

## Elastic constants workflow

The `--elastic` flag activates finite-strain Cij calculation after each successful geometry optimisation. CASTEP's built-in `task: ElasticConstants` cannot be used for VCA systems (it requires DFPT, which is incompatible with MIXTURE atoms and ultrasoft pseudopotentials). VCAForge works around this entirely within Python.

**How it works:**

1. **Strain generation.** The relaxed cell is deformed in 6 directions (for cubic symmetry) by ±0.3%. For each deformation, only the LATTICE_CART vectors change — POSITIONS_FRAC are invariant under homogeneous strain, so MIXTURE/VCA tags survive untouched.

2. **CASTEP SinglePoint × 6.** Each strained cell is run independently. The `.param` file is automatically configured with:
   - `task: SinglePoint`
   - `calculate_stress: true` (mandatory — the stress tensor is what we fit)
   - `elec_energy_tol: 1.0e-7 eV` (tighter than GeomOpt's 1e-5, elastic constants are sensitive to SCF precision)
   - `finite_basis_corr: 1` (basis set correction applied once — saves ~60% time vs mode 2)
   - `nextra_bands` adaptive: endpoints = 10, VCA = 15 + ⌊|VEC − 8| × 20⌋

3. **Cij fitting.** From the 6 stress tensors σ and their corresponding strain vectors ε, the elastic constants are fitted by OLS regression (with intercept, to handle any residual pressure after GeomOpt):

   ```
   C11 = ∂σ₁₁/∂ε₁₁     (axial response to axial strain)
   C12 = ∂σ₂₂/∂ε₁₁     (transverse response to axial strain)
   C44 = ∂σ₂₃/∂(2ε₂₃)  (shear response to shear strain)
   ```

4. **Polycrystalline averages.** From C11, C12, C44 the Voigt-Reuss-Hill bulk modulus B, shear modulus G, Young's modulus E, and Poisson ratio ν are derived analytically.

5. **Born stability check.** If the fitted tensor violates C11 > 0, C44 > 0, C11 > |C12|, or C11 + 2·C12 > 0, the result is flagged as unstable and written to CSV as N/A — no Python traceback, no crash.

**What the step output looks like:**

```
┌─ 2/8  x=0.2500  Ti=0.75  Nb=0.25  VEC=8.25◑
│  $ mpirun -n 6 castep.mpi TiC
│  ▶ Geometry Optimization …
│  [████████████░░░░░░░░░░] 55%  geo 14/100  scf 6  ⏱ 00:38
│  ▶ Geometry Optimization … ✓  (1m 12s)  [a=3.061Å  H=-2022.1 eV]
│  ▶ Elastic Tensors (6 strains,  nextra_bands=20,  VEC=8.25) …
│  ✓ Elastic Tensors (4m 8s)  [B=267 G=182 E=450 GPa]  R²=0.9990
│     C11=518  C12=141  C44=175 GPa
└─ ✓ Elastic step completed.
```

If elastic calculation fails (e.g. Born instability for high VEC):
```
│  ⚠  Elasticity failed: Born stability violated: C11=120  C12=400  C44=-15 GPa
└─ ✗ Elastic step failed.
```

**Vegard interpolation fallback.** For VCA intermediate compositions where CASTEP fails (common for VEC > 8.4 due to ghost states in the ultrasoft overlap matrix), elastic constants are estimated by linear interpolation between the pure x=0 and x=1 endpoints. This is the elastic analogue of Vegard's law and is a well-established approximation for isostructural solid solutions — error vs SQS supercell calculations is typically < 5% for d-metal carbides.

---

## Project structure

```
TiVC_Mar17_21-29/
├── original_TiC.cell        ← your original file, never modified
├── original_TiC.param       ← your original param, never modified
├── vca_state.json            ← crash-safe checkpoint
├── vca_results.csv           ← live results, updated after every step
├── run.log
├── x0.0000/
│   ├── TiC.cell              ← VCA cell at x=0 (pure Ti)
│   ├── TiC.param             ← nextra_bands=10 (pure endpoint)
│   ├── TiC.castep
│   ├── TiC-out.cell
│   └── TiC.cijdat            ← strain metadata (kept for diagnostics)
├── x0.2500/
│   ├── TiC.param             ← nextra_bands=20 (VCA, VEC=8.25)
│   └── ...
└── x1.0000/
    ├── TiC.param             ← nextra_bands=10 (pure endpoint)
    └── ...
```

Strained cell files (`*_cij__*`) are deleted automatically after fitting to keep the directory clean. Pass `--keep-all` to retain them along with `.check` and `.bands` files.

---

## CSV output

`vca_results.csv` is written after every step and contains:

| Column | Description |
|--------|-------------|
| `x` | Concentration of species B |
| `status` | `done` / `failed` / `skipped` |
| `enthalpy_eV` | Final enthalpy from GeomOpt |
| `a_opt_ang` | Optimised lattice parameter a |
| `geom_converged` | `yes` / `no` |
| `wall_time_s` | Total CASTEP wall time |
| `C11`, `C12`, `C44` | Elastic constants (GPa) |
| `B_Hill_GPa` | VRH bulk modulus |
| `G_Hill_GPa` | VRH shear modulus |
| `E_GPa` | Young's modulus |
| `nu` | Poisson ratio |
| `Zener_A` | Zener anisotropy index (1 = isotropic) |
| `elastic_source` | `CASTEP` or `Vegard_interpolation` |
| `elastic_R2_min` | Minimum R² of the stress-strain fit |

---

## Module structure

| File | Responsibility |
|------|---------------|
| `main.py` | Entry point, argument parsing, run loop, elastic orchestration |
| `ui.py` | All console output — wizards, VEC predictor, step boxes, summary table |
| `castep_io.py` | File I/O — reads `.cell`, writes VCA cells, generates `.param`, parses `.castep` |
| `workflow.py` | Process management — state JSON, CASTEP subprocess with progress bar, CSV export |
| `elastic_workflow.py` | Finite-strain elastic workflow — cell straining, SinglePoint runs, result dispatch |
| `core/elasticity.py` | Pure numpy elastic math — strain patterns, Cij OLS fitting, VRH averages, VEC |

---

## Physics notes

**Why VCA and not SQS?**  
Virtual Crystal Approximation treats the alloy as a single atom with fractional nuclear charge Z = (1-x)·Z_A + x·Z_B. It is fast (same unit cell as the pure compound) but ignores chemical disorder. For properties that are smooth in x — lattice parameter, bulk modulus — VCA is accurate to a few percent. For properties sensitive to local chemistry (formation enthalpy, surface energy), SQS supercells are more appropriate.

**Why can't CASTEP ElasticConstants task be used for VCA?**  
CASTEP's ElasticConstants uses DFPT (density-functional perturbation theory), which requires computing the response of the electronic density to an applied strain. The strain field response formalism is not implemented for MIXTURE atoms with ultrasoft pseudopotentials. The finite-strain workaround used here avoids DFPT entirely — it just runs SinglePoint calculations at displaced geometries.

**Why does the elastic fit use an intercept?**  
After GeomOpt, the residual stress is theoretically zero but practically ~0.01–0.1 GPa due to finite basis set and k-point errors. A through-origin fit (σ = C·ε) gives misleading R² values when this residual is non-zero. The OLS fit with intercept (σ = C·ε + σ₀) gives the correct slope C regardless of σ₀ and a proper R² based on variance around the fit line.

---

## Tested on

- Ryzen 5 4500U | CachyOS x86_64
- CASTEP 25.12, GFortran 15.2.1, OpenMPI
- Python 3.11 / 3.14
- `.cell` files generated by `cif2cell` from Materials Project CIFs
