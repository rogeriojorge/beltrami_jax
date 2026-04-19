from .benchmark import benchmark_parameter_scan, benchmark_solve
from .diagnostics import compare_against_reference, compute_solve_diagnostics
from .geometry import (
    assemble_fourier_beltrami_system,
    basis_values,
    build_fourier_mode_basis,
    collocation_grid,
    shift_mu,
    torus_coordinates,
)
from .io import load_problem_json, load_saved_solution, save_nonlinear_solution, save_problem_json
from .iterative import gmres_solve
from .nonlinear import solve_helicity_constrained_equilibrium
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
    BeltramiProblem,
    BeltramiLinearSystem,
    FourierBeltramiGeometry,
    FourierModeBasis,
    GeometryAssemblyResult,
    IterativeSolveResult,
    NonlinearSolveResult,
    ParameterScanBenchmark,
    ReferenceComparison,
    SolveBenchmark,
    SolveDiagnostics,
    SolveResult,
    SpecLinearSystemReference,
)

__all__ = [
    "BeltramiProblem",
    "BeltramiLinearSystem",
    "FourierBeltramiGeometry",
    "FourierModeBasis",
    "GeometryAssemblyResult",
    "IterativeSolveResult",
    "NonlinearSolveResult",
    "ParameterScanBenchmark",
    "ReferenceComparison",
    "SolveBenchmark",
    "SolveDiagnostics",
    "SolveResult",
    "SpecLinearSystemReference",
    "assemble_fourier_beltrami_system",
    "assemble_operator",
    "assemble_rhs",
    "basis_values",
    "benchmark_parameter_scan",
    "benchmark_solve",
    "build_fourier_mode_basis",
    "collocation_grid",
    "compare_against_reference",
    "compute_solve_diagnostics",
    "gmres_solve",
    "list_packaged_references",
    "load_problem_json",
    "load_packaged_reference",
    "load_saved_solution",
    "load_spec_text_dump",
    "magnetic_energy",
    "magnetic_helicity",
    "relative_residual_norm",
    "residual",
    "residual_norm",
    "save_nonlinear_solution",
    "save_problem_json",
    "shift_mu",
    "solve_helicity_constrained_equilibrium",
    "solve_from_components",
    "solve_operator",
    "solve_parameter_scan",
    "torus_coordinates",
]

__version__ = "0.1.0"
