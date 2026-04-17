from __future__ import annotations

import numpy as np

from .types import ReferenceComparison, SolveDiagnostics, SolveResult, SpecLinearSystemReference


def _scaled_relative_error(actual: np.ndarray, expected: np.ndarray, *, matrix: bool = False) -> float:
    ord_value = "fro" if matrix else 2
    numerator = float(np.linalg.norm(actual - expected, ord=ord_value))
    denominator = float(np.linalg.norm(expected, ord=ord_value))
    if denominator <= 0.0:
        return numerator
    return numerator / denominator


def compute_solve_diagnostics(
    result: SolveResult,
    *,
    include_condition_number: bool = False,
) -> SolveDiagnostics:
    """Compute dense linear-algebra diagnostics for a solved system."""
    operator = np.asarray(result.operator)
    rhs = np.asarray(result.rhs)
    solution = np.asarray(result.solution)
    residual = np.asarray(result.residual)

    operator_fro_norm = float(np.linalg.norm(operator, ord="fro"))
    rhs_l2_norm = float(np.linalg.norm(rhs, ord=2))
    solution_l2_norm = float(np.linalg.norm(solution, ord=2))
    residual_l2_norm = float(np.linalg.norm(residual, ord=2))
    symmetry_numerator = float(np.linalg.norm(operator - operator.T, ord="fro"))
    symmetry_defect = symmetry_numerator / max(operator_fro_norm, 1e-30)
    amplification_factor = solution_l2_norm / max(rhs_l2_norm, 1e-30)
    condition_number_2 = float(np.linalg.cond(operator)) if include_condition_number else None

    return SolveDiagnostics(
        label=result.system.label,
        size=result.system.size,
        is_vacuum=result.system.is_vacuum,
        operator_fro_norm=operator_fro_norm,
        rhs_l2_norm=rhs_l2_norm,
        solution_l2_norm=solution_l2_norm,
        max_abs_solution=float(np.max(np.abs(solution))),
        max_abs_residual=float(np.max(np.abs(residual))),
        residual_l2_norm=residual_l2_norm,
        relative_residual_norm=float(np.asarray(result.relative_residual_norm)),
        symmetry_defect=symmetry_defect,
        amplification_factor=amplification_factor,
        condition_number_2=condition_number_2,
    )


def compare_against_reference(
    reference: SpecLinearSystemReference,
    result: SolveResult,
) -> ReferenceComparison:
    """Measure agreement between a solved system and a dumped SPEC reference."""
    operator = np.asarray(result.operator)
    rhs = np.asarray(result.rhs)
    solution = np.asarray(result.solution)

    return ReferenceComparison(
        label=reference.system.label,
        size=reference.system.size,
        volume_index=reference.volume_index,
        operator_relative_error=_scaled_relative_error(operator, np.asarray(reference.matrix), matrix=True),
        rhs_relative_error=_scaled_relative_error(rhs, np.asarray(reference.rhs)),
        solution_relative_error=_scaled_relative_error(solution, np.asarray(reference.expected_solution)),
        max_abs_solution_error=float(np.max(np.abs(solution - np.asarray(reference.expected_solution)))),
    )
