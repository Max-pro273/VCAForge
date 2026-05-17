"""
engines/MLIP/model_registry.py  —  Universal MLIP model catalogue.
═══════════════════════════════════════════════════════════════════════════════
Single source of truth for every supported ML interatomic potential.

Design rules
────────────
• One @register_mlip_model("key") per model variant.
• Loader must accept (variant: str, device: str) → ASE Calculator.
• ImportError inside a loader → model listed as "unavailable" in the wizard.
  The engine never crashes at import time; libraries are loaded lazily.
• Device strings: "cpu" | "cuda" | "cuda:N" | "mps" | "rocm" | "hip"
  ROCm/HIP aliases route to the same CUDA code-path for libraries that
  support it (MACE, MatGL). For libraries that do not (older CHGNet),
  the loader warns and falls back to CPU.
• All models that ship universal pre-trained weights are listed here.
  Local/custom checkpoints are handled by the "local" family at the bottom.
• Install commands use `uv pip install` — avoids venv conflicts and resolves
  deps faster. CPU and GPU variants are separate so the user gets a minimal
  install on CPU-only machines.

UI grouping (shown in wizard schema)
────────────
  Library      Models                           Install group
  ──────────── ──────────────────────────────── ─────────────────────────────
  MACE         mace-mp-0  mace-mp-0b  mace-off  mace-cpu  |  mace-gpu
               mace-anicc
  CHGNet       chgnet                           chgnet-cpu | chgnet-gpu
  MatGL        m3gnet  tensornet  chgnet-matgl  matgl-cpu  | matgl-gpu
  SevenNet     7net-0  7net-mf-ompa             sevenn-cpu | sevenn-gpu
  ORB          orb-v2  orb-v3                   orb-cpu    | orb-gpu
  FairChem     uma-sm  uma-md  esen-sm           fairchem   (GPU only)
  NequIP       nequip-local                     nequip
  DeepMD       deepmd-local                     deepmd-kit
  Custom       local-ase  local-torch            (no install)
"""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import Any, Callable

import config as _cfg

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Registry core
# ─────────────────────────────────────────────────────────────────────────────

# key → {"loader": Callable, "meta": dict}
_REGISTRY: dict[str, dict[str, Any]] = {}


def register_mlip_model(
    key: str,
    *,
    label: str,
    library: str,
    description: str,
    install_cpu: str,
    install_gpu: str,
    supports_stress: bool = True,
    universal: bool = True,
    needs_file: bool = False,
) -> Callable:
    """Decorator that registers a model loader with its UI metadata.

    Args:
        key:             Short identifier used in --model CLI flag and config.
        label:           Human-readable name shown in wizard.
        library:         Parent library name for grouping (MACE / CHGNet / …).
        description:     One-line description of training set / use case.
        install_cpu:     `uv pip install` command for CPU-only setup.
        install_gpu:     `uv pip install` command for GPU setup.
        supports_stress: Whether this calculator returns a stress tensor
                         (needed for elastic constants and cell relaxation).
        universal:       True = pre-trained universal potential.
                         False = requires a local checkpoint file.
        needs_file:      True = loader expects variant to be a file path.
    """
    def wrapper(fn: Callable) -> Callable:
        _REGISTRY[key] = {
            "loader": fn,
            "meta": {
                "key": key,
                "label": label,
                "library": library,
                "description": description,
                "install_cpu": install_cpu,
                "install_gpu": install_gpu,
                "supports_stress": supports_stress,
                "universal": universal,
                "needs_file": needs_file,
            },
        }
        return fn
    return wrapper


def load_model(key: str, device: str = "cpu", **kwargs) -> Any:
    """Instantiate a calculator by registry key.

    Raises:
        KeyError:      Unknown model key.
        ImportError:   Library not installed — caller shows install hint.
        RuntimeError:  Loader-specific error (bad checkpoint, bad device).
    """
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise KeyError(
            f"Unknown MLIP model: {key!r}. Available: {available}"
        )
    device = _normalise_device(device)
    return _REGISTRY[key]["loader"](key, device, **kwargs)


def all_models() -> list[dict[str, Any]]:
    """Return all registered model metadata dicts, sorted by library+key."""
    return sorted(
        (v["meta"] for v in _REGISTRY.values()),
        key=lambda m: (m["library"], m["key"]),
    )


def available_models() -> list[dict[str, Any]]:
    """Return only models whose library is importable right now."""
    result = []
    for key, entry in _REGISTRY.items():
        meta = entry["meta"]
        try:
            # Probe-load: check library importable without downloading weights.
            _probe_import(meta["library"])
            result.append(meta)
        except ImportError:
            pass
    return sorted(result, key=lambda m: (m["library"], m["key"]))


def install_hint(key: str, device: str = "cpu") -> str:
    """Return the uv install command for a given model + device combo."""
    if key not in _REGISTRY:
        return ""
    meta = _REGISTRY[key]["meta"]
    is_gpu = device not in ("cpu",)
    cmd = meta["install_gpu"] if is_gpu else meta["install_cpu"]
    return f"uv pip install {cmd}"


# ─────────────────────────────────────────────────────────────────────────────
# Device normalisation
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_device(device: str) -> str:
    """Map user-facing device strings to what each library expects.

    ROCm note: PyTorch with ROCm builds exposes AMD GPUs as "cuda" devices.
    Older RX 470/480 cards (gfx803) need ROCm 5.x + HSA_OVERRIDE_GFX_VERSION.
    We do NOT set HSA_OVERRIDE_GFX_VERSION here — the user must set it in their
    environment. We just accept "rocm" / "hip" as aliases for "cuda" and warn.
    """
    d = device.lower().strip()
    if d in ("rocm", "hip", "amd"):
        log.warning(
            "Device alias '%s' mapped to 'cuda' (PyTorch ROCm path). "
            "For old AMD GCN cards (RX 470/480, gfx803) set: "
            "HSA_OVERRIDE_GFX_VERSION=9.0.0 in your environment before "
            "running VCAForge. Install PyTorch ROCm wheel separately: "
            "uv pip install torch --index-url https://download.pytorch.org/whl/rocm6.2",
            device,
        )
        return "cuda"
    if d == "mps":
        # Apple Silicon — only MACE and some MatGL models support this natively.
        return "mps"
    return d


def _probe_import(library: str) -> None:
    """Raise ImportError if the library's top-level package is missing."""
    probe = {
        "MACE": "mace",
        "CHGNet": "chgnet",
        "MatGL": "matgl",
        "SevenNet": "sevenn",
        "ORB": "orb_models",
        "FairChem": "fairchem.core",
        "NequIP": "nequip",
        "DeepMD": "deepmd.calculator",
        "Custom": None,  # no probe needed
    }.get(library)
    if probe:
        __import__(probe)


# ─────────────────────────────────────────────────────────────────────────────
# ── MACE ─────────────────────────────────────────────────────────────────────
# Universal potentials from the Cambridge group.
# mace-mp-0: trained on Materials Project (89 elements).
# mace-mp-0b: updated weights, better for oxides and halides.
# mace-off: organic force field variant, NOT for periodic inorganic solids.
# mace-anicc: ANI-style carbon/nitrogen/oxygen/hydrogen chemistry.
#
# CPU install:  uv pip install mace-torch --extra-index-url https://download.pytorch.org/whl/cpu
# GPU install:  uv pip install mace-torch  (picks up CUDA torch automatically)
# MPS (Apple):  uv pip install mace-torch  (MPS backend via torch)
# ROCm (AMD):   install PyTorch ROCm wheel first, then mace-torch
# ─────────────────────────────────────────────────────────────────────────────

def _mace_loader(key: str, device: str, **kwargs) -> Any:
    from mace.calculators import mace_mp, mace_off, mace_anicc  # noqa: PLC0415
    model_map = {
        "mace-mp-0":    ("medium",   mace_mp),
        "mace-mp-0b":   ("medium-v2",mace_mp),
        "mace-mp-0b2":  ("medium-v2-2024-09-26", mace_mp),
        "mace-mp-0b2-large": ("large-0b2", mace_mp),             # ДОДАНО
        "mace-mp-0b3":  ("medium-0b3", mace_mp),                 # ДОДАНО
        "mace-mpa-0":   ("medium-mpa-0", mace_mp),               # ДОДАНО
        "mace-off-sm":  ("small",    mace_off),
        "mace-off-md":  ("medium",   mace_off),
        "mace-anicc":   ("ANI-cc-pol-2025-01-21", mace_anicc),
    }
    variant, factory = model_map[key]
    # mace_mp / mace_off accept device= directly; mps is supported since mace 0.3.6
    dtype = kwargs.pop("default_dtype", "float32")
    return factory(model=variant, device=device, default_dtype=dtype, **kwargs)

@register_mlip_model(
    "mace-mp-0",
    label="MACE-MP-0 (medium)",
    library="MACE",
    description="Universal potential, 89 elements, Materials Project training set. "
                "Best general choice for inorganic solids.",
    install_cpu="mace-torch --extra-index-url https://download.pytorch.org/whl/cpu",
    install_gpu="mace-torch",
    supports_stress=True,
    universal=True,
)
def _reg_mace_mp0(key: str, device: str, **kwargs): return _mace_loader(key, device, **kwargs)


@register_mlip_model(
    "mace-mp-0b",
    label="MACE-MP-0b (medium-v2)",
    library="MACE",
    description="Updated MACE-MP weights. Improved accuracy for oxides and halides.",
    install_cpu="mace-torch --extra-index-url https://download.pytorch.org/whl/cpu",
    install_gpu="mace-torch",
    supports_stress=True,
    universal=True,
)
def _reg_mace_mp0b(key: str, device: str, **kwargs): return _mace_loader(key, device, **kwargs)


@register_mlip_model(
    "mace-mp-0b2",
    label="MACE-MP-0b2 (medium-v2-2024)",
    library="MACE",
    description="Latest MACE-MP release (Sep 2024). Best accuracy across the periodic table.",
    install_cpu="mace-torch --extra-index-url https://download.pytorch.org/whl/cpu",
    install_gpu="mace-torch",
    supports_stress=True,
    universal=True,
)
def _reg_mace_mp0b2(key: str, device: str, **kwargs): return _mace_loader(key, device, **kwargs)


@register_mlip_model(
    "mace-off-sm",
    label="MACE-OFF small (organic)",
    library="MACE",
    description="Organic force field. For molecules and polymers — NOT for periodic metals.",
    install_cpu="mace-torch --extra-index-url https://download.pytorch.org/whl/cpu",
    install_gpu="mace-torch",
    supports_stress=False,
    universal=True,
)
def _reg_mace_off_sm(key: str, device: str, **kwargs): return _mace_loader(key, device, **kwargs)


@register_mlip_model(
    "mace-off-md",
    label="MACE-OFF medium (organic)",
    library="MACE",
    description="Higher-accuracy organic force field. Slow but transferable.",
    install_cpu="mace-torch --extra-index-url https://download.pytorch.org/whl/cpu",
    install_gpu="mace-torch",
    supports_stress=False,
    universal=True,
)
def _reg_mace_off_md(key: str, device: str, **kwargs): return _mace_loader(key, device, **kwargs)

@register_mlip_model(
    "mace-mp-0b2-large",
    label="MACE-MP-0b2 (large)",
    library="MACE",
    description="Large variant of MACE-MP-0b2 (5.7M params). Highest accuracy, slower inference.",
    install_cpu="mace-torch --extra-index-url https://download.pytorch.org/whl/cpu",
    install_gpu="mace-torch",
    supports_stress=True,
    universal=True,
)
def _reg_mace_mp0b2_large(key: str, device: str, **kwargs): return _mace_loader(key, device, **kwargs)

@register_mlip_model(
    "mace-mp-0b3",
    label="MACE-MP-0b3 (medium)",
    library="MACE",
    description="Latest MACE-MP release. Fixed phonon instabilities present in 0b2.",
    install_cpu="mace-torch --extra-index-url https://download.pytorch.org/whl/cpu",
    install_gpu="mace-torch",
    supports_stress=True,
    universal=True,
)
def _reg_mace_mp0b3(key: str, device: str, **kwargs): return _mace_loader(key, device, **kwargs)

@register_mlip_model(
    "mace-mpa-0",
    label="MACE-MPA-0 (Alexandria)",
    library="MACE",
    description="Trained on MPTrj + sAlex. Superior high-pressure stability and robustness for SQS/alloys.",
    install_cpu="mace-torch --extra-index-url https://download.pytorch.org/whl/cpu",
    install_gpu="mace-torch",
    supports_stress=True,
    universal=True,
)
def _reg_mace_mpa0(key: str, device: str, **kwargs): return _mace_loader(key, device, **kwargs)

# ─────────────────────────────────────────────────────────────────────────────
# ── CHGNet ───────────────────────────────────────────────────────────────────
# Crystal Hamiltonian Graph Neural Network (Deng et al. 2023).
# Trained on Materials Project trajectories. Returns energy, forces,
# stress, and magnetic moments. Supports spin-polarised calculations.
#
# GPU note: chgnet uses PyTorch internally. Any CUDA/ROCm device works.
# CPU install:  uv pip install chgnet
# GPU install:  uv pip install chgnet  (inherits whatever torch is installed)
# ─────────────────────────────────────────────────────────────────────────────

@register_mlip_model(
    "chgnet",
    label="CHGNet v0.3.0",
    library="CHGNet",
    description="Graph NN with charge equilibration. Returns magnetic moments. "
                "Good for magnetic and oxide systems.",
    install_cpu="chgnet",
    install_gpu="chgnet",
    supports_stress=True,
    universal=True,
)
def _reg_chgnet(key: str, device: str, **kwargs) -> Any:
    from chgnet.model import CHGNet  # noqa: PLC0415
    from chgnet.model.dynamics import CHGNetCalculator  # noqa: PLC0415
    model = CHGNet.load()
    # CHGNet ≥0.3 maps "cuda" / "cpu" / "mps"; "rocm" alias already normalised.
    return CHGNetCalculator(model=model, use_device=device)


# ─────────────────────────────────────────────────────────────────────────────
# ── MatGL ────────────────────────────────────────────────────────────────────
# Materials Graph Library (Ong group). Wraps M3GNet, TensorNet, and CHGNet
# in a unified ASE-compatible interface.
#
# m3gnet:       Original M3GNet (2022). Fast, ~89 elements.
# tensornet:    TensorNet-MatPES (2024). Best MatGL accuracy.
# chgnet-matgl: CHGNet weights loaded through MatGL (alternative to above).
#
# CPU install:  uv pip install matgl dgl -f https://data.dgl.ai/wheels/torch-2.4/repo.html
# GPU install:  uv pip install matgl dgl-cu124 -f https://data.dgl.ai/wheels/torch-2.4/repo.html
# DGL note: DGL provides the graph backend. The CUDA variant must match your
# installed CUDA toolkit version (cu118/cu121/cu124).
# ─────────────────────────────────────────────────────────────────────────────

def _matgl_loader(key: str, device: str, **kwargs) -> Any:
    import matgl  # noqa: PLC0415
    from matgl.ext.ase import PESCalculator  # noqa: PLC0415

    model_names = {
        "m3gnet":       "M3GNet-MP-2021.2.8-PES",
        "tensornet":    "TensorNet-MatPES-PBE-v2024.12",
        "chgnet-matgl": "CHGNet-MPtrj-2024.2.13-PES",
    }
    pot = matgl.load_model(model_names[key])
    # PESCalculator wraps stress output; stress_weight=1.0 means full stress.
    calc = PESCalculator(potential=pot, stress_weight=1.0)
    # MatGL uses DGL which manages device internally via torch.
    import torch  # noqa: PLC0415
    calc.device = torch.device(device)
    return calc


@register_mlip_model(
    "m3gnet",
    label="M3GNet-MP (2021)",
    library="MatGL",
    description="Fast graph NN for 89 elements. Good baseline. Superseded by TensorNet.",
    install_cpu="matgl dgl -f https://data.dgl.ai/wheels/torch-2.4/repo.html",
    install_gpu="matgl dgl-cu124 -f https://data.dgl.ai/wheels/torch-2.4/repo.html",
    supports_stress=True,
    universal=True,
)
def _reg_m3gnet(key: str, device: str, **kwargs): return _matgl_loader(key, device, **kwargs)


@register_mlip_model(
    "tensornet",
    label="TensorNet-MatPES (2024)",
    library="MatGL",
    description="State-of-the-art MatGL model. High accuracy for solids. Recommended over M3GNet.",
    install_cpu="matgl dgl -f https://data.dgl.ai/wheels/torch-2.4/repo.html",
    install_gpu="matgl dgl-cu124 -f https://data.dgl.ai/wheels/torch-2.4/repo.html",
    supports_stress=True,
    universal=True,
)
def _reg_tensornet(key: str, device: str, **kwargs): return _matgl_loader(key, device, **kwargs)


@register_mlip_model(
    "chgnet-matgl",
    label="CHGNet via MatGL",
    library="MatGL",
    description="CHGNet weights loaded through MatGL. Stress output is more reliable than the standalone chgnet package.",
    install_cpu="matgl dgl -f https://data.dgl.ai/wheels/torch-2.4/repo.html",
    install_gpu="matgl dgl-cu124 -f https://data.dgl.ai/wheels/torch-2.4/repo.html",
    supports_stress=True,
    universal=True,
)
def _reg_chgnet_matgl(key: str, device: str, **kwargs): return _matgl_loader(key, device, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# ── SevenNet ─────────────────────────────────────────────────────────────────
# 7net = SevenNet (Batatia group, Seoul). Equivariant GNN.
# 7net-0: lightweight universal potential trained on MPF.2021.2.8.
# 7net-mf-ompa: multi-fidelity, trained on OQMD + MPF + Alexandria.
#               Best SevenNet accuracy — recommended for hard materials.
#
# CPU install:  uv pip install sevenn
# GPU install:  uv pip install sevenn  (uses installed CUDA torch)
# ─────────────────────────────────────────────────────────────────────────────

def _sevenn_loader(key: str, device: str, **kwargs) -> Any:
    from sevenn.sevennet_calculator import SevenNetCalculator  # noqa: PLC0415
    model_map = {
        "7net-0":       "7net-0",
        "7net-mf-ompa": "7net-mf-ompa",
    }
    return SevenNetCalculator(model_map[key], device=device)


@register_mlip_model(
    "7net-0",
    label="SevenNet-0 (universal)",
    library="SevenNet",
    description="Equivariant GNN, 89 elements, MPF training set. Fast inference.",
    install_cpu="sevenn",
    install_gpu="sevenn",
    supports_stress=True,
    universal=True,
)
def _reg_7net0(key: str, device: str, **kwargs): return _sevenn_loader(key, device, **kwargs)


@register_mlip_model(
    "7net-mf-ompa",
    label="SevenNet-MF-OMPA (multi-fidelity)",
    library="SevenNet",
    description="Multi-fidelity SevenNet on OQMD + MPF + Alexandria. Best for refractory alloys.",
    install_cpu="sevenn",
    install_gpu="sevenn",
    supports_stress=True,
    universal=True,
)
def _reg_7net_mf(key: str, device: str, **kwargs): return _sevenn_loader(key, device, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# ── ORB ──────────────────────────────────────────────────────────────────────
# Orbital Materials ORB models. Very fast inference, good for high-throughput.
# orb-v2: 2024 release. orb-v3: 2025, improved coverage.
#
# CPU install:  uv pip install orb-models --extra-index-url https://download.pytorch.org/whl/cpu
# GPU install:  uv pip install orb-models
# Note: orb-models requires torch ≥ 2.3. No MPS support as of orb-v3.
# ─────────────────────────────────────────────────────────────────────────────

def _orb_loader(key: str, device: str, **kwargs) -> Any:
    import orb_models  # noqa: PLC0415
    from orb_models.forcefield import pretrained  # noqa: PLC0415
    from orb_models.forcefield.calculator import ORBCalculator  # noqa: PLC0415
    import torch  # noqa: PLC0415

    model_map = {
        "orb-v2": pretrained.orb_v2,
        "orb-v3": pretrained.orb_v3,
    }
    dev = torch.device(device)
    model = model_map[key](device=dev)
    return ORBCalculator(model, device=dev)


@register_mlip_model(
    "orb-v2",
    label="ORB v2",
    library="ORB",
    description="Very fast universal potential from Orbital Materials. Good throughput/accuracy trade-off.",
    install_cpu="orb-models --extra-index-url https://download.pytorch.org/whl/cpu",
    install_gpu="orb-models",
    supports_stress=True,
    universal=True,
)
def _reg_orb_v2(key: str, device: str, **kwargs): return _orb_loader(key, device, **kwargs)


@register_mlip_model(
    "orb-v3",
    label="ORB v3 (2025)",
    library="ORB",
    description="ORB 2025 release. Improved coverage and force accuracy over v2.",
    install_cpu="orb-models --extra-index-url https://download.pytorch.org/whl/cpu",
    install_gpu="orb-models",
    supports_stress=True,
    universal=True,
)
def _reg_orb_v3(key: str, device: str, **kwargs): return _orb_loader(key, device, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# ── FairChem (Meta) ──────────────────────────────────────────────────────────
# Meta's Open Catalyst / Universal Models suite.
# uma-sm / uma-md: Universal Models for Atoms, small / medium.
# esen-sm:         eSEN architecture, small variant.
#
# GPU ONLY — FairChem does not officially support CPU inference.
# Install: uv pip install fairchem-core
# Note: requires CUDA ≥ 11.7. Not compatible with ROCm as of 2025-05.
# ─────────────────────────────────────────────────────────────────────────────

def _fairchem_loader(key: str, device: str, **kwargs) -> Any:
    from fairchem.core import FAIRChemCalculator  # noqa: PLC0415
    model_map = {
        "uma-sm":  "uma-s-1",
        "uma-md":  "uma-m-1",
        "esen-sm": "eSEN-30M-OC20",
    }
    if device == "cpu":
        warnings.warn(
            "FairChem models are GPU-optimised. CPU inference is very slow "
            "and may fail for large cells. Consider using MACE or ORB on CPU.",
            stacklevel=3,
        )
    return FAIRChemCalculator(model_map[key], device=device)


@register_mlip_model(
    "uma-sm",
    label="UMA small (Meta)",
    library="FairChem",
    description="Universal Model for Atoms — small. Fast, high-coverage, 89+ elements.",
    install_cpu="fairchem-core",
    install_gpu="fairchem-core",
    supports_stress=True,
    universal=True,
)
def _reg_uma_sm(key: str, device: str, **kwargs): return _fairchem_loader(key, device, **kwargs)


@register_mlip_model(
    "uma-md",
    label="UMA medium (Meta)",
    library="FairChem",
    description="Universal Model for Atoms — medium. Best FairChem accuracy.",
    install_cpu="fairchem-core",
    install_gpu="fairchem-core",
    supports_stress=True,
    universal=True,
)
def _reg_uma_md(key: str, device: str, **kwargs): return _fairchem_loader(key, device, **kwargs)


@register_mlip_model(
    "esen-sm",
    label="eSEN small (Meta, OC20)",
    library="FairChem",
    description="eSEN-30M trained on OC20. Best for surface and catalysis applications.",
    install_cpu="fairchem-core",
    install_gpu="fairchem-core",
    supports_stress=True,
    universal=True,
)
def _reg_esen_sm(key: str, device: str, **kwargs): return _fairchem_loader(key, device, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# ── NequIP / Allegro ─────────────────────────────────────────────────────────
# Equivariant NNs from Batzner / Musaelian (Kozinsky group, Harvard).
# Requires a locally trained checkpoint — no universal pre-trained weights
# are shipped with the package.
#
# Install: uv pip install nequip
# File:    --mlip-file path/to/deployed.pth
# ─────────────────────────────────────────────────────────────────────────────

@register_mlip_model(
    "nequip-local",
    label="NequIP (local checkpoint)",
    library="NequIP",
    description="Equivariant NN. Requires a locally trained deployed.pth checkpoint. "
                "Excellent accuracy for fine-tuned systems.",
    install_cpu="nequip",
    install_gpu="nequip",
    supports_stress=True,
    universal=False,
    needs_file=True,
)
def _reg_nequip(key: str, device: str) -> Any:
    # Actual file path injected at load_model() call time via variant=path.
    # The MlipEngine passes the --mlip-file argument as variant.
    raise RuntimeError(
        "NequIP loader called without a file path. "
        "Use --mlip-file path/to/deployed.pth"
    )


# ─────────────────────────────────────────────────────────────────────────────
# ── DeepMD-kit ───────────────────────────────────────────────────────────────
# Deep Potential MD — widely used in ab-initio MD workflows.
# Requires a locally trained frozen.pb or model.pt checkpoint.
#
# CPU install:  uv pip install deepmd-kit
# GPU install:  uv pip install deepmd-kit[gpu]
# File:         --mlip-file path/to/frozen.pb
# ─────────────────────────────────────────────────────────────────────────────

@register_mlip_model(
    "deepmd-local",
    label="DeepMD (local checkpoint)",
    library="DeepMD",
    description="Deep Potential MD. Requires a locally trained frozen.pb or model.pt checkpoint.",
    install_cpu="deepmd-kit",
    install_gpu="deepmd-kit[gpu]",
    supports_stress=True,
    universal=False,
    needs_file=True,
)
def _reg_deepmd(key: str, device: str) -> Any:
    raise RuntimeError(
        "DeepMD loader called without a file path. "
        "Use --mlip-file path/to/frozen.pb"
    )


# ─────────────────────────────────────────────────────────────────────────────
# ── Custom / local ───────────────────────────────────────────────────────────
# Generic fallback: any ASE-compatible calculator loaded from a Python file
# that exposes a `get_calculator(device)` function, or any TorchScript model.
# ─────────────────────────────────────────────────────────────────────────────

@register_mlip_model(
    "local-ase",
    label="Custom ASE calculator (Python file)",
    library="Custom",
    description="Load an ASE calculator from a Python file that exposes "
                "get_calculator(device: str) -> Calculator. Pass path via --mlip-file.",
    install_cpu="",
    install_gpu="",
    supports_stress=True,
    universal=False,
    needs_file=True,
)
def _reg_local_ase(key: str, device: str) -> Any:
    raise RuntimeError(
        "local-ase called without a file path. Use --mlip-file path/to/calc.py"
    )


@register_mlip_model(
    "local-torch",
    label="Custom TorchScript model",
    library="Custom",
    description="Load a TorchScript .pt model with torch.jit.load. "
                "Model must implement forward(pos, cell, species) → (energy, forces, stress). "
                "Pass path via --mlip-file.",
    install_cpu="",
    install_gpu="",
    supports_stress=True,
    universal=False,
    needs_file=True,
)
def _reg_local_torch(key: str, device: str) -> Any:
    raise RuntimeError(
        "local-torch called without a file path. Use --mlip-file path/to/model.pt"
    )


# ─────────────────────────────────────────────────────────────────────────────
# File-based loaders (called by MlipEngine when needs_file=True)
# ─────────────────────────────────────────────────────────────────────────────

def load_file_model(key: str, file_path: Path, device: str = "cpu") -> Any:
    """Load a file-based model (NequIP, DeepMD, local-ase, local-torch).

    Unlike load_model(), this accepts a file path as the primary argument.
    Called by MlipEngine when meta["needs_file"] is True.
    """
    device = _normalise_device(device)
    if key == "nequip-local":
        from nequip.ase import NequIPCalculator  # noqa: PLC0415
        return NequIPCalculator.from_deployed_model(
            str(file_path), device=device, species_to_type_name=None,
        )
    if key == "deepmd-local":
        from deepmd.calculator import DP  # noqa: PLC0415
        return DP(model=str(file_path))
    if key == "local-ase":
        import importlib.util  # noqa: PLC0415
        spec = importlib.util.spec_from_file_location("_user_calc", file_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if not hasattr(mod, "get_calculator"):
            raise RuntimeError(
                f"{file_path} must define get_calculator(device: str) -> ASE Calculator"
            )
        return mod.get_calculator(device)
    if key == "local-torch":
        import torch  # noqa: PLC0415
        from ase.calculators.calculator import Calculator  # noqa: PLC0415
        # Wrap TorchScript model in a minimal ASE Calculator shim.
        model = torch.jit.load(str(file_path), map_location=device)
        return _TorchScriptCalculator(model, device)
    raise KeyError(f"No file loader for model key: {key!r}")


class _TorchScriptCalculator:
    """Minimal ASE Calculator shim for a TorchScript model.

    The model must implement:
      forward(positions: Tensor[N,3],
              cell: Tensor[3,3],
              numbers: Tensor[N]) -> (energy: Tensor[], forces: Tensor[N,3],
                                       stress: Tensor[3,3])
    """
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, model: Any, device: str) -> None:
        import torch  # noqa: PLC0415
        self._model = model
        self._device = torch.device(device)
        self.results: dict[str, Any] = {}

    def calculate(self, atoms: Any, properties=None, system_changes=None) -> None:
        import torch  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
        pos   = torch.tensor(atoms.positions,     dtype=torch.float64, device=self._device)
        cell  = torch.tensor(np.array(atoms.cell), dtype=torch.float64, device=self._device)
        nums  = torch.tensor(atoms.numbers,        dtype=torch.long,   device=self._device)
        with torch.no_grad():
            energy, forces, stress = self._model(pos, cell, nums)
        self.results = {
            "energy": float(energy.cpu()),
            "forces": forces.cpu().numpy(),
            "stress": stress.cpu().numpy().flatten(),
        }
