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
from .spectre_input import SpectreInputSummary, load_spectre_input_toml
from .spectre_io import (
    SpectreH5Reference,
    SpectreVectorPotential,
    SpectreVectorPotentialComparison,
    compare_vector_potentials,
    load_spectre_reference_h5,
    load_spectre_vector_potential_h5,
    load_spectre_vector_potential_npz,
    save_spectre_vector_potential_npz,
)
from .spectre_validation import (
    PackagedSpectreCase,
    list_packaged_spectre_cases,
    load_all_packaged_spectre_cases,
    load_packaged_spectre_case,
    packaged_spectre_case_paths,
)
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
    "PackagedSpectreCase",
    "ReferenceComparison",
    "SolveBenchmark",
    "SolveDiagnostics",
    "SolveResult",
    "SpecLinearSystemReference",
    "SpectreH5Reference",
    "SpectreInputSummary",
    "SpectreVectorPotential",
    "SpectreVectorPotentialComparison",
    "assemble_fourier_beltrami_system",
    "assemble_operator",
    "assemble_rhs",
    "basis_values",
    "benchmark_parameter_scan",
    "benchmark_solve",
    "build_fourier_mode_basis",
    "collocation_grid",
    "compare_against_reference",
    "compare_vector_potentials",
    "compute_solve_diagnostics",
    "gmres_solve",
    "list_packaged_references",
    "list_packaged_spectre_cases",
    "load_problem_json",
    "load_packaged_reference",
    "load_packaged_spectre_case",
    "load_all_packaged_spectre_cases",
    "load_saved_solution",
    "load_spectre_reference_h5",
    "load_spectre_input_toml",
    "load_spectre_vector_potential_h5",
    "load_spectre_vector_potential_npz",
    "load_spec_text_dump",
    "magnetic_energy",
    "magnetic_helicity",
    "packaged_spectre_case_paths",
    "relative_residual_norm",
    "residual",
    "residual_norm",
    "save_nonlinear_solution",
    "save_problem_json",
    "save_spectre_vector_potential_npz",
    "shift_mu",
    "solve_helicity_constrained_equilibrium",
    "solve_from_components",
    "solve_operator",
    "solve_parameter_scan",
    "torus_coordinates",
]

__version__ = "0.1.0"
