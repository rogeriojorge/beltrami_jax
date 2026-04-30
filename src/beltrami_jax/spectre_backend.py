from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from functools import partial
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike


@dataclass(frozen=True)
class SpectreBackendSolve:
    """Minimal SPECTRE-facing Beltrami solve result."""

    solution: Array
    derivative_solutions: Array
    residual: Array
    derivative_residuals: Array
    residual_norm: Array
    relative_residual_norm: Array
    derivative_residual_norms: Array
    derivative_relative_residual_norms: Array
    magnetic_energy_integral: Array
    magnetic_helicity_integral: Array


@dataclass(frozen=True)
class SpectreBackendBatchSolve:
    """Batched SPECTRE-facing Beltrami solve result for equal-size systems."""

    solutions: Array
    derivative_solutions: Array
    residuals: Array
    derivative_residuals: Array
    residual_norms: Array
    relative_residual_norms: Array
    derivative_residual_norms: Array
    derivative_relative_residual_norms: Array
    magnetic_energy_integrals: Array
    magnetic_helicity_integrals: Array


@dataclass(frozen=True)
class SpectreBackendTiming:
    """Timing summary for the small SPECTRE backend adapter."""

    label: str
    size: int
    batch_size: int
    repeats: int
    compile_and_solve_seconds: float
    steady_state_seconds: float
    per_system_seconds: float


@partial(jax.jit, static_argnames=("is_vacuum", "include_d_mg_in_rhs"))
def _solve_spectre_assembled_jax(
    d_ma: Array,
    d_md: Array,
    d_mb: Array,
    d_mg: Array,
    mu: Array,
    psi: Array,
    *,
    is_vacuum: bool,
    include_d_mg_in_rhs: bool,
) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array, Array]:
    operator = d_ma if is_vacuum else d_ma - mu * d_md
    rhs = -(d_mb @ psi)
    if include_d_mg_in_rhs:
        rhs = rhs - d_mg
    solution = jnp.linalg.solve(operator, rhs)
    residual = operator @ solution - rhs
    residual_norm = jnp.linalg.norm(residual, ord=2)
    rhs_norm = jnp.maximum(jnp.linalg.norm(rhs, ord=2), jnp.asarray(1.0e-30, dtype=rhs.dtype))

    first_derivative_rhs = jnp.where(
        is_vacuum,
        -d_mb[:, 0],
        jnp.where(include_d_mg_in_rhs, -d_mb[:, 0], d_md @ solution),
    )
    second_derivative_rhs = -d_mb[:, 1]
    derivative_rhs = jnp.stack((first_derivative_rhs, second_derivative_rhs), axis=1)
    derivative_solutions = jnp.linalg.solve(operator, derivative_rhs).T
    derivative_residuals = (operator @ derivative_solutions.T - derivative_rhs).T
    derivative_residual_norms = jnp.linalg.norm(derivative_residuals, ord=2, axis=1)
    derivative_rhs_norms = jnp.maximum(jnp.linalg.norm(derivative_rhs.T, ord=2, axis=1), jnp.asarray(1.0e-30, dtype=rhs.dtype))
    magnetic_energy_integral = 0.5 * (solution @ (d_ma @ solution)) + solution @ (d_mb @ psi)
    magnetic_helicity_integral = 0.5 * (solution @ (d_md @ solution))
    return (
        solution,
        derivative_solutions,
        residual,
        derivative_residuals,
        residual_norm,
        residual_norm / rhs_norm,
        derivative_residual_norms,
        derivative_residual_norms / derivative_rhs_norms,
        magnetic_energy_integral,
        magnetic_helicity_integral,
    )


@partial(jax.jit, static_argnames=("is_vacuum", "include_d_mg_in_rhs"))
def _solve_spectre_assembled_batch_jax(
    d_ma: Array,
    d_md: Array,
    d_mb: Array,
    d_mg: Array,
    mu: Array,
    psi: Array,
    *,
    is_vacuum: bool,
    include_d_mg_in_rhs: bool,
) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array, Array]:
    operators = d_ma if is_vacuum else d_ma - mu[:, None, None] * d_md
    rhs = -jnp.einsum("bij,bj->bi", d_mb, psi)
    if include_d_mg_in_rhs:
        rhs = rhs - d_mg
    solutions = jax.vmap(jnp.linalg.solve)(operators, rhs)
    residuals = jnp.einsum("bij,bj->bi", operators, solutions) - rhs
    residual_norms = jnp.linalg.norm(residuals, ord=2, axis=1)
    rhs_norms = jnp.maximum(jnp.linalg.norm(rhs, ord=2, axis=1), jnp.asarray(1.0e-30, dtype=rhs.dtype))
    first_derivative_rhs = jnp.where(
        is_vacuum,
        -d_mb[:, :, 0],
        jnp.where(
            include_d_mg_in_rhs,
            -d_mb[:, :, 0],
            jnp.einsum("bij,bj->bi", d_md, solutions),
        ),
    )
    second_derivative_rhs = -d_mb[:, :, 1]
    derivative_rhs = jnp.stack((first_derivative_rhs, second_derivative_rhs), axis=1)
    derivative_solutions = jax.vmap(lambda operator, local_rhs: jnp.linalg.solve(operator, local_rhs.T).T)(
        operators,
        derivative_rhs,
    )
    derivative_residuals = jnp.einsum("bij,bkj->bki", operators, derivative_solutions) - derivative_rhs
    derivative_residual_norms = jnp.linalg.norm(derivative_residuals, ord=2, axis=2)
    derivative_rhs_norms = jnp.maximum(
        jnp.linalg.norm(derivative_rhs, ord=2, axis=2),
        jnp.asarray(1.0e-30, dtype=rhs.dtype),
    )
    magnetic_energy_integrals = 0.5 * jnp.einsum("bi,bij,bj->b", solutions, d_ma, solutions) + jnp.einsum(
        "bi,bij,bj->b",
        solutions,
        d_mb,
        psi,
    )
    magnetic_helicity_integrals = 0.5 * jnp.einsum("bi,bij,bj->b", solutions, d_md, solutions)
    return (
        solutions,
        derivative_solutions,
        residuals,
        derivative_residuals,
        residual_norms,
        residual_norms / rhs_norms,
        derivative_residual_norms,
        derivative_residual_norms / derivative_rhs_norms,
        magnetic_energy_integrals,
        magnetic_helicity_integrals,
    )


def _as_vector_d_mg(
    d_mg: ArrayLike | None,
    *,
    size: int,
    include_d_mg_in_rhs: bool,
) -> Array:
    if d_mg is None:
        if include_d_mg_in_rhs:
            raise ValueError("d_mg is required when include_d_mg_in_rhs=True")
        return jnp.zeros((size,), dtype=jnp.float64)
    return jnp.asarray(d_mg, dtype=jnp.float64)


def _as_batch_d_mg(
    d_mg: ArrayLike | None,
    *,
    batch_size: int,
    size: int,
    include_d_mg_in_rhs: bool,
) -> Array:
    if d_mg is None:
        if include_d_mg_in_rhs:
            raise ValueError("d_mg is required when include_d_mg_in_rhs=True")
        return jnp.zeros((batch_size, size), dtype=jnp.float64)
    return jnp.asarray(d_mg, dtype=jnp.float64)


def solve_spectre_assembled(
    *,
    d_ma: ArrayLike,
    d_md: ArrayLike,
    d_mb: ArrayLike,
    mu: ArrayLike,
    psi: ArrayLike,
    d_mg: ArrayLike | None = None,
    is_vacuum: bool = False,
    include_d_mg_in_rhs: bool | None = None,
) -> SpectreBackendSolve:
    """Solve one SPECTRE-assembled Beltrami linear system.

    This is the smallest intended SPECTRE integration boundary: SPECTRE keeps
    ownership of geometry and matrix assembly, then passes the assembled
    Beltrami blocks to this JIT-backed solve kernel.
    """
    if include_d_mg_in_rhs is None:
        include_d_mg_in_rhs = is_vacuum
    d_ma_j = jnp.asarray(d_ma, dtype=jnp.float64)
    d_md_j = jnp.asarray(d_md, dtype=jnp.float64)
    d_mb_j = jnp.asarray(d_mb, dtype=jnp.float64)
    d_mg_j = _as_vector_d_mg(
        d_mg,
        size=int(d_ma_j.shape[0]),
        include_d_mg_in_rhs=include_d_mg_in_rhs,
    )
    (
        solution,
        derivative_solutions,
        residual,
        derivative_residuals,
        residual_norm,
        relative_residual_norm,
        derivative_residual_norms,
        derivative_relative_residual_norms,
        magnetic_energy_integral,
        magnetic_helicity_integral,
    ) = _solve_spectre_assembled_jax(
        d_ma_j,
        d_md_j,
        d_mb_j,
        d_mg_j,
        jnp.asarray(mu, dtype=jnp.float64),
        jnp.asarray(psi, dtype=jnp.float64),
        is_vacuum=is_vacuum,
        include_d_mg_in_rhs=include_d_mg_in_rhs,
    )
    return SpectreBackendSolve(
        solution=solution,
        derivative_solutions=derivative_solutions,
        residual=residual,
        derivative_residuals=derivative_residuals,
        residual_norm=residual_norm,
        relative_residual_norm=relative_residual_norm,
        derivative_residual_norms=derivative_residual_norms,
        derivative_relative_residual_norms=derivative_relative_residual_norms,
        magnetic_energy_integral=magnetic_energy_integral,
        magnetic_helicity_integral=magnetic_helicity_integral,
    )


def solve_spectre_assembled_numpy(
    **kwargs,
) -> dict[str, np.ndarray | float]:
    """NumPy-returning wrapper intended for a thin SPECTRE Python adapter."""
    result = solve_spectre_assembled(**kwargs)
    solution = np.asarray(result.solution)
    residual = np.asarray(result.residual)
    return {
        "solution": solution,
        "derivative_solutions": np.asarray(result.derivative_solutions),
        "residual": residual,
        "derivative_residuals": np.asarray(result.derivative_residuals),
        "residual_norm": float(np.asarray(result.residual_norm)),
        "relative_residual_norm": float(np.asarray(result.relative_residual_norm)),
        "derivative_residual_norms": np.asarray(result.derivative_residual_norms),
        "derivative_relative_residual_norms": np.asarray(result.derivative_relative_residual_norms),
        "magnetic_energy_integral": float(np.asarray(result.magnetic_energy_integral)),
        "magnetic_helicity_integral": float(np.asarray(result.magnetic_helicity_integral)),
    }


def solve_spectre_assembled_batch(
    *,
    d_ma: ArrayLike,
    d_md: ArrayLike,
    d_mb: ArrayLike,
    mu: ArrayLike,
    psi: ArrayLike,
    d_mg: ArrayLike | None = None,
    is_vacuum: bool = False,
    include_d_mg_in_rhs: bool | None = None,
) -> SpectreBackendBatchSolve:
    """Solve a batch of equal-size SPECTRE-assembled systems.

    Batching is optional, but it is useful when SPECTRE has multiple volumes
    with the same active degree-of-freedom count and branch flags.
    """
    if include_d_mg_in_rhs is None:
        include_d_mg_in_rhs = is_vacuum
    d_ma_j = jnp.asarray(d_ma, dtype=jnp.float64)
    d_md_j = jnp.asarray(d_md, dtype=jnp.float64)
    d_mb_j = jnp.asarray(d_mb, dtype=jnp.float64)
    batch_size = int(d_ma_j.shape[0])
    size = int(d_ma_j.shape[1])
    d_mg_j = _as_batch_d_mg(
        d_mg,
        batch_size=batch_size,
        size=size,
        include_d_mg_in_rhs=include_d_mg_in_rhs,
    )
    (
        solutions,
        derivative_solutions,
        residuals,
        derivative_residuals,
        residual_norms,
        relative_residual_norms,
        derivative_residual_norms,
        derivative_relative_residual_norms,
        magnetic_energy_integrals,
        magnetic_helicity_integrals,
    ) = _solve_spectre_assembled_batch_jax(
        d_ma_j,
        d_md_j,
        d_mb_j,
        d_mg_j,
        jnp.asarray(mu, dtype=jnp.float64),
        jnp.asarray(psi, dtype=jnp.float64),
        is_vacuum=is_vacuum,
        include_d_mg_in_rhs=include_d_mg_in_rhs,
    )
    return SpectreBackendBatchSolve(
        solutions=solutions,
        derivative_solutions=derivative_solutions,
        residuals=residuals,
        derivative_residuals=derivative_residuals,
        residual_norms=residual_norms,
        relative_residual_norms=relative_residual_norms,
        derivative_residual_norms=derivative_residual_norms,
        derivative_relative_residual_norms=derivative_relative_residual_norms,
        magnetic_energy_integrals=magnetic_energy_integrals,
        magnetic_helicity_integrals=magnetic_helicity_integrals,
    )


def _block_until_ready(value: object) -> None:
    if is_dataclass(value) and not isinstance(value, type):
        leaves = (getattr(value, field.name) for field in fields(value))
    else:
        leaves = jax.tree_util.tree_leaves(value)
    for leaf in leaves:
        block_until_ready = getattr(leaf, "block_until_ready", None)
        if block_until_ready is not None:
            block_until_ready()


def _time_backend_call(fn) -> float:
    start = perf_counter()
    value = fn()
    _block_until_ready(value)
    return perf_counter() - start


def benchmark_spectre_backend(
    *,
    label: str,
    size: int,
    batch_size: int,
    solve_fn,
    repeats: int = 3,
) -> SpectreBackendTiming:
    """Benchmark a SPECTRE backend solve callable without imposing a runtime budget."""
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    compile_and_solve_seconds = _time_backend_call(solve_fn)
    steady_state_samples = [_time_backend_call(solve_fn) for _ in range(repeats)]
    steady_state_seconds = float(np.mean(steady_state_samples))
    return SpectreBackendTiming(
        label=label,
        size=size,
        batch_size=batch_size,
        repeats=repeats,
        compile_and_solve_seconds=compile_and_solve_seconds,
        steady_state_seconds=steady_state_seconds,
        per_system_seconds=steady_state_seconds / batch_size,
    )
