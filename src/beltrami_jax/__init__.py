from .benchmark import benchmark_parameter_scan, benchmark_solve
from .diagnostics import compare_against_reference, compute_solve_diagnostics
from .operators import (
    assemble_operator,
    assemble_rhs,
    magnetic_energy,
    magnetic_helicity,
    relative_residual_norm,
    residual,
    residual_norm,
)
from .reference import list_packaged_references, load_packaged_reference, load_spec_text_dump
from .solver import solve_from_components, solve_operator, solve_parameter_scan
from .types import (
    BeltramiLinearSystem,
    ParameterScanBenchmark,
    ReferenceComparison,
    SolveBenchmark,
    SolveDiagnostics,
    SolveResult,
    SpecLinearSystemReference,
)

__all__ = [
    "BeltramiLinearSystem",
    "ParameterScanBenchmark",
    "ReferenceComparison",
    "SolveBenchmark",
    "SolveDiagnostics",
    "SolveResult",
    "SpecLinearSystemReference",
    "assemble_operator",
    "assemble_rhs",
    "benchmark_parameter_scan",
    "benchmark_solve",
    "compare_against_reference",
    "compute_solve_diagnostics",
    "list_packaged_references",
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
