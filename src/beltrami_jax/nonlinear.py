from __future__ import annotations

import jax.numpy as jnp

from .geometry import assemble_fourier_beltrami_system, shift_mu
from .solver import solve_from_components
from .types import BeltramiProblem, NonlinearSolveResult


def solve_helicity_constrained_equilibrium(
    problem: BeltramiProblem,
    *,
    verbose: bool = False,
) -> NonlinearSolveResult:
    """Run a secant-style outer loop to match a target magnetic helicity."""
    if problem.max_iterations <= 0:
        raise ValueError("problem.max_iterations must be positive")

    assembly = assemble_fourier_beltrami_system(
        problem.geometry,
        problem.basis,
        mu=problem.initial_mu,
        psi=problem.psi,
        is_vacuum=problem.is_vacuum,
        vacuum_strength=problem.vacuum_strength,
        label=problem.label or problem.geometry.label or "beltrami_problem",
    )

    mu_history: list[float] = []
    helicity_history: list[float] = []
    residual_history: list[float] = []

    def evaluate(mu_value: float):
        shifted = shift_mu(assembly, mu_value, label=assembly.system.label)
        result = solve_from_components(
            shifted.system,
            method=problem.solver,
            tolerance=problem.tolerance,
            max_iterations=max(shifted.system.size, 8),
            verbose=verbose,
        )
        helicity = float(result.magnetic_helicity)
        residual = helicity - problem.target_helicity
        mu_history.append(mu_value)
        helicity_history.append(helicity)
        residual_history.append(residual)
        if verbose:
            print(
                "[beltrami_jax] "
                f"outer_iteration={len(mu_history):02d} mu={mu_value:.8e} "
                f"helicity={helicity:.8e} target={problem.target_helicity:.8e} "
                f"constraint_residual={residual:.8e}"
            )
        return shifted, result, residual

    delta = 1.0e-2 if problem.initial_mu == 0.0 else 5.0e-2 * abs(problem.initial_mu)
    left_assembly, left_result, left_residual = evaluate(problem.initial_mu)
    if abs(left_residual) <= problem.tolerance:
        return NonlinearSolveResult(
            problem=problem,
            assembly=left_assembly,
            solve=left_result,
            converged=True,
            iterations=1,
            mu_history=jnp.asarray(mu_history, dtype=jnp.float64),
            helicity_history=jnp.asarray(helicity_history, dtype=jnp.float64),
            constraint_residual_history=jnp.asarray(residual_history, dtype=jnp.float64),
        )

    right_assembly, right_result, right_residual = evaluate(problem.initial_mu + delta)
    converged = abs(right_residual) <= problem.tolerance

    while len(mu_history) < problem.max_iterations and not converged:
        denominator = right_residual - left_residual
        if abs(denominator) < 1.0e-14:
            candidate_mu = mu_history[-1] + delta
        else:
            candidate_mu = mu_history[-1] - right_residual * (mu_history[-1] - mu_history[-2]) / denominator

        candidate_assembly, candidate_result, candidate_residual = evaluate(candidate_mu)
        left_assembly = right_assembly
        left_result = right_result
        left_residual = right_residual
        right_assembly = candidate_assembly
        right_result = candidate_result
        right_residual = candidate_residual
        converged = abs(candidate_residual) <= problem.tolerance

    return NonlinearSolveResult(
        problem=problem,
        assembly=right_assembly,
        solve=right_result,
        converged=converged,
        iterations=len(mu_history),
        mu_history=jnp.asarray(mu_history, dtype=jnp.float64),
        helicity_history=jnp.asarray(helicity_history, dtype=jnp.float64),
        constraint_residual_history=jnp.asarray(residual_history, dtype=jnp.float64),
    )
