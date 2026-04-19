from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
from jax import Array

from .types import IterativeSolveResult


def gmres_solve(
    operator: Array | Callable[[Array], Array],
    rhs: Array,
    *,
    x0: Array | None = None,
    tolerance: float = 1.0e-10,
    max_iterations: int | None = None,
) -> IterativeSolveResult:
    """Solve ``A x = b`` with a compact GMRES implementation."""
    rhs = jnp.asarray(rhs, dtype=jnp.float64)
    size = int(rhs.shape[0])
    if size == 0:
        raise ValueError("rhs must be non-empty")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")

    max_iterations = size if max_iterations is None else int(max_iterations)
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")

    if callable(operator):
        matvec = lambda vector: jnp.asarray(operator(vector), dtype=jnp.float64)
    else:
        matrix = jnp.asarray(operator, dtype=jnp.float64)
        matvec = lambda vector: matrix @ vector

    x = jnp.zeros_like(rhs) if x0 is None else jnp.asarray(x0, dtype=jnp.float64)
    residual0 = rhs - matvec(x)
    rhs_norm = jnp.maximum(jnp.linalg.norm(rhs, ord=2), jnp.asarray(1.0e-30, dtype=rhs.dtype))
    beta = jnp.linalg.norm(residual0, ord=2)
    if float(beta) <= tolerance * float(rhs_norm):
        return IterativeSolveResult(
            solution=x,
            residual=residual0,
            residual_norm=beta,
            relative_residual_norm=beta / rhs_norm,
            iterations=0,
            converged=True,
        )

    vectors = jnp.zeros((size, max_iterations + 1), dtype=jnp.float64)
    hessenberg = jnp.zeros((max_iterations + 1, max_iterations), dtype=jnp.float64)
    vectors = vectors.at[:, 0].set(residual0 / beta)
    e1 = jnp.zeros((max_iterations + 1,), dtype=jnp.float64).at[0].set(beta)

    best_solution = x
    best_residual = residual0
    best_relative = beta / rhs_norm
    converged = False
    iterations = 0

    for iteration in range(max_iterations):
        w = matvec(vectors[:, iteration])
        for basis_index in range(iteration + 1):
            h_value = jnp.vdot(vectors[:, basis_index], w)
            hessenberg = hessenberg.at[basis_index, iteration].set(h_value)
            w = w - h_value * vectors[:, basis_index]

        next_norm = jnp.linalg.norm(w, ord=2)
        hessenberg = hessenberg.at[iteration + 1, iteration].set(next_norm)
        if iteration + 1 < max_iterations and float(next_norm) > 0.0:
            vectors = vectors.at[:, iteration + 1].set(w / next_norm)

        least_squares = jnp.linalg.lstsq(
            hessenberg[: iteration + 2, : iteration + 1],
            e1[: iteration + 2],
            rcond=None,
        )[0]
        candidate = x + vectors[:, : iteration + 1] @ least_squares
        candidate_residual = rhs - matvec(candidate)
        candidate_norm = jnp.linalg.norm(candidate_residual, ord=2)
        candidate_relative = candidate_norm / rhs_norm

        best_solution = candidate
        best_residual = candidate_residual
        best_relative = candidate_relative
        iterations = iteration + 1

        if float(candidate_relative) <= tolerance:
            converged = True
            break

    return IterativeSolveResult(
        solution=best_solution,
        residual=best_residual,
        residual_norm=jnp.linalg.norm(best_residual, ord=2),
        relative_residual_norm=best_relative,
        iterations=iterations,
        converged=converged,
    )
