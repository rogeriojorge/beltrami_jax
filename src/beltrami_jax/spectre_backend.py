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
    residual: Array
    residual_norm: Array
    relative_residual_norm: Array


@dataclass(frozen=True)
class SpectreBackendBatchSolve:
    """Batched SPECTRE-facing Beltrami solve result for equal-size systems."""

    solutions: Array
    residuals: Array
    residual_norms: Array
    relative_residual_norms: Array


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
) -> tuple[Array, Array, Array, Array]:
    operator = d_ma if is_vacuum else d_ma - mu * d_md
    rhs = -(d_mb @ psi)
    if include_d_mg_in_rhs:
        rhs = rhs - d_mg
    solution = jnp.linalg.solve(operator, rhs)
    residual = operator @ solution - rhs
    residual_norm = jnp.linalg.norm(residual, ord=2)
    rhs_norm = jnp.maximum(jnp.linalg.norm(rhs, ord=2), jnp.asarray(1.0e-30, dtype=rhs.dtype))
    return solution, residual, residual_norm, residual_norm / rhs_norm


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
) -> tuple[Array, Array, Array, Array]:
    operators = d_ma if is_vacuum else d_ma - mu[:, None, None] * d_md
    rhs = -jnp.einsum("bij,bj->bi", d_mb, psi)
    if include_d_mg_in_rhs:
        rhs = rhs - d_mg
    solutions = jax.vmap(jnp.linalg.solve)(operators, rhs)
    residuals = jnp.einsum("bij,bj->bi", operators, solutions) - rhs
    residual_norms = jnp.linalg.norm(residuals, ord=2, axis=1)
    rhs_norms = jnp.maximum(jnp.linalg.norm(rhs, ord=2, axis=1), jnp.asarray(1.0e-30, dtype=rhs.dtype))
    return solutions, residuals, residual_norms, residual_norms / rhs_norms


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
    solution, residual, residual_norm, relative_residual_norm = _solve_spectre_assembled_jax(
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
        residual=residual,
        residual_norm=residual_norm,
        relative_residual_norm=relative_residual_norm,
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
        "residual": residual,
        "residual_norm": float(np.asarray(result.residual_norm)),
        "relative_residual_norm": float(np.asarray(result.relative_residual_norm)),
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
    solutions, residuals, residual_norms, relative_residual_norms = _solve_spectre_assembled_batch_jax(
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
        residuals=residuals,
        residual_norms=residual_norms,
        relative_residual_norms=relative_residual_norms,
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
