from __future__ import annotations

import time
from typing import Iterable

import jax
import numpy as np

from .reference import SpecLinearSystemReference
from .solver import solve_from_components, solve_parameter_scan
from .types import ParameterScanBenchmark, SolveBenchmark


def _block_tree(value: object) -> None:
    for leaf in jax.tree_util.tree_leaves(value):
        block_until_ready = getattr(leaf, "block_until_ready", None)
        if block_until_ready is not None:
            block_until_ready()


def _time_call(fn) -> float:
    start = time.perf_counter()
    value = fn()
    _block_tree(value)
    return time.perf_counter() - start


def benchmark_solve(
    reference: SpecLinearSystemReference,
    *,
    repeats: int = 3,
) -> SolveBenchmark:
    """Benchmark the dense solve for one packaged SPEC reference."""
    if repeats < 1:
        raise ValueError("repeats must be at least 1")

    compile_and_solve_seconds = _time_call(lambda: solve_from_components(reference.system))
    steady_state_samples = [_time_call(lambda: solve_from_components(reference.system)) for _ in range(repeats)]
    return SolveBenchmark(
        label=reference.system.label,
        size=reference.system.size,
        repeats=repeats,
        compile_and_solve_seconds=compile_and_solve_seconds,
        steady_state_seconds=float(np.mean(steady_state_samples)),
    )


def benchmark_parameter_scan(
    reference: SpecLinearSystemReference,
    *,
    batch_sizes: Iterable[int] = (1, 4, 8, 16),
    repeats: int = 3,
    relative_span: float = 0.05,
) -> list[ParameterScanBenchmark]:
    """Benchmark batched `mu` scans around one non-vacuum reference."""
    if reference.system.is_vacuum:
        raise ValueError("parameter-scan benchmarks require a plasma-region reference")
    if repeats < 1:
        raise ValueError("repeats must be at least 1")

    mu0 = float(reference.system.mu)
    psi = np.asarray(reference.system.psi)
    results: list[ParameterScanBenchmark] = []
    for batch_size in batch_sizes:
        mu_values = np.linspace(mu0 - relative_span, mu0 + relative_span, batch_size, dtype=np.float64)
        psi_values = np.repeat(psi[None, :], batch_size, axis=0)
        compile_and_solve_seconds = _time_call(
            lambda: solve_parameter_scan(
                reference.system.d_ma,
                reference.system.d_md,
                reference.system.d_mb,
                mu_values,
                psi_values,
            )
        )
        steady_state_samples = [
            _time_call(
                lambda: solve_parameter_scan(
                    reference.system.d_ma,
                    reference.system.d_md,
                    reference.system.d_mb,
                    mu_values,
                    psi_values,
                )
            )
            for _ in range(repeats)
        ]
        steady_state_seconds = float(np.mean(steady_state_samples))
        results.append(
            ParameterScanBenchmark(
                label=reference.system.label,
                size=reference.system.size,
                batch_size=batch_size,
                repeats=repeats,
                compile_and_solve_seconds=compile_and_solve_seconds,
                steady_state_seconds=steady_state_seconds,
                per_system_seconds=steady_state_seconds / batch_size,
            )
        )
    return results
