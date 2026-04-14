"""
castep/castep.py  —  Public API re-export shim.

All logic lives in:
  castep/io.py      — cell I/O, .param generation, output parsing
  castep/elastic.py — finite-strain elastic-constants workflow

This module re-exports every symbol that the rest of VCAForge imports so
that existing ``from castep.castep import …`` call-sites need no changes.
"""

from castep.elastic import run_elastic
from castep.io import (
    CifResult,
    atom_count,
    count_atoms,
    inject_ncp,
    is_conventional_cell,
    nextra_for_step,
    parse_elastic_file,
    parse_output,
    patch_nextra,
    read_species,
    read_stress,
    reduce_cif,
    smart_defaults,
    sp_param_content as _sp_param,  # kept for any internal references
    write_geomopt_param,
    write_primitive_cell,
    write_singlepoint_param,
    write_vca_cell,
    _parse_lattice,
)

__all__ = [
    # Cell I/O
    "read_species",
    "atom_count",
    "is_conventional_cell",
    "write_vca_cell",
    "inject_ncp",
    "write_primitive_cell",
    # Param helpers
    "smart_defaults",
    "write_geomopt_param",
    "write_singlepoint_param",
    "patch_nextra",
    "nextra_for_step",
    # Output parsing
    "parse_output",
    "parse_elastic_file",
    "read_stress",
    "count_atoms",
    # CIF reduction
    "CifResult",
    "reduce_cif",
    # Elastic workflow
    "run_elastic",
    # Private (used by main.py helpers)
    "_parse_lattice",
]
