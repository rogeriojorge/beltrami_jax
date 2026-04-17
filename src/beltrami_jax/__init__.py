from .operators import (
    assemble_operator,
    assemble_rhs,
    magnetic_energy,
    magnetic_helicity,
    relative_residual_norm,
    residual,
    residual_norm,
)
from .reference import load_packaged_reference, load_spec_text_dump
from .solver import solve_from_components, solve_operator, solve_parameter_scan
from .types import BeltramiLinearSystem, SolveResult, SpecLinearSystemReference

__all__ = [
    "BeltramiLinearSystem",
    "SolveResult",
    "SpecLinearSystemReference",
    "assemble_operator",
    "assemble_rhs",
    "load_packaged_reference",
    "load_spec_text_dump",
    "magnetic_energy",
    "magnetic_helicity",
    "relative_residual_norm",
    "residual",
    "residual_norm",
    "solve_from_components",
    "solve_operator",
    "solve_parameter_scan",
]

__version__ = "0.1.0"
