from __future__ import annotations

import numpy as np
import pytest

from beltrami_jax import (
    benchmark_spectre_backend,
    load_all_packaged_spectre_linear_systems,
    load_packaged_spectre_linear_system,
    solve_spectre_assembled,
    solve_spectre_assembled_batch,
    solve_spectre_assembled_numpy,
)


def _backend_kwargs(fixture):
    system = fixture.system
    return {
        "d_ma": system.d_ma,
        "d_md": system.d_md,
        "d_mb": system.d_mb,
        "d_mg": system.d_mg,
        "mu": system.mu,
        "psi": system.psi,
        "is_vacuum": system.is_vacuum,
        "include_d_mg_in_rhs": system.include_d_mg_in_rhs,
    }


def test_spectre_backend_matches_all_packaged_spectre_linear_solutions() -> None:
    for fixture in load_all_packaged_spectre_linear_systems():
        result = solve_spectre_assembled(**_backend_kwargs(fixture))
        expected = np.asarray(fixture.expected_solution)
        relative_error = np.linalg.norm(np.asarray(result.solution) - expected) / max(np.linalg.norm(expected), 1e-300)
        assert relative_error < 3e-12
        assert float(result.relative_residual_norm) < 3e-12


def test_spectre_backend_numpy_wrapper_is_adapter_friendly() -> None:
    fixture = load_packaged_spectre_linear_system("G2V32L1Fi/lvol1")
    result = solve_spectre_assembled_numpy(**_backend_kwargs(fixture))

    assert isinstance(result["solution"], np.ndarray)
    assert isinstance(result["residual"], np.ndarray)
    assert isinstance(result["residual_norm"], float)
    assert isinstance(result["relative_residual_norm"], float)
    np.testing.assert_allclose(result["solution"], np.asarray(fixture.expected_solution), rtol=3e-12, atol=3e-12)


def test_spectre_backend_batches_equal_size_plasma_volumes() -> None:
    fixtures = [
        load_packaged_spectre_linear_system("G2V32L1Fi/lvol2"),
        load_packaged_spectre_linear_system("G2V32L1Fi/lvol3"),
        load_packaged_spectre_linear_system("G2V32L1Fi/lvol4"),
    ]
    result = solve_spectre_assembled_batch(
        d_ma=np.stack([np.asarray(fixture.system.d_ma) for fixture in fixtures]),
        d_md=np.stack([np.asarray(fixture.system.d_md) for fixture in fixtures]),
        d_mb=np.stack([np.asarray(fixture.system.d_mb) for fixture in fixtures]),
        mu=np.asarray([float(fixture.system.mu) for fixture in fixtures]),
        psi=np.stack([np.asarray(fixture.system.psi) for fixture in fixtures]),
        is_vacuum=False,
        include_d_mg_in_rhs=False,
    )

    for row, fixture in enumerate(fixtures):
        expected = np.asarray(fixture.expected_solution)
        relative_error = np.linalg.norm(np.asarray(result.solutions[row]) - expected) / max(np.linalg.norm(expected), 1e-300)
        assert relative_error < 3e-12
    assert np.max(np.asarray(result.relative_residual_norms)) < 3e-12


def test_spectre_backend_supports_no_dmg_plasma_default() -> None:
    result = solve_spectre_assembled(
        d_ma=np.eye(2),
        d_md=np.zeros((2, 2)),
        d_mb=np.eye(2),
        mu=0.0,
        psi=np.asarray([1.0, -2.0]),
    )
    np.testing.assert_allclose(np.asarray(result.solution), np.asarray([-1.0, 2.0]))


def test_spectre_backend_batches_source_branch() -> None:
    result = solve_spectre_assembled_batch(
        d_ma=np.stack([np.eye(2), 2.0 * np.eye(2)]),
        d_md=np.zeros((2, 2, 2)),
        d_mb=np.stack([np.eye(2), np.eye(2)]),
        d_mg=np.asarray([[0.5, -0.5], [1.0, 2.0]]),
        mu=np.asarray([0.0, 0.0]),
        psi=np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        is_vacuum=False,
        include_d_mg_in_rhs=True,
    )
    np.testing.assert_allclose(np.asarray(result.solutions[0]), np.asarray([-1.5, -1.5]))
    np.testing.assert_allclose(np.asarray(result.solutions[1]), np.asarray([-2.0, -3.0]))


def test_spectre_backend_requires_dmg_for_source_branch() -> None:
    fixture = load_packaged_spectre_linear_system("G3V8L3Free/lvol9")
    kwargs = _backend_kwargs(fixture)
    kwargs["d_mg"] = None
    with pytest.raises(ValueError, match="d_mg is required"):
        solve_spectre_assembled(**kwargs)


def test_spectre_backend_batch_requires_dmg_for_source_branch() -> None:
    with pytest.raises(ValueError, match="d_mg is required"):
        solve_spectre_assembled_batch(
            d_ma=np.eye(2)[None, :, :],
            d_md=np.zeros((1, 2, 2)),
            d_mb=np.eye(2)[None, :, :],
            mu=np.asarray([0.0]),
            psi=np.asarray([[1.0, 2.0]]),
            d_mg=None,
            include_d_mg_in_rhs=True,
        )


def test_spectre_backend_timing_helper_reports_positive_times() -> None:
    fixture = load_packaged_spectre_linear_system("G3V8L3Free/lvol7")
    timing = benchmark_spectre_backend(
        label=fixture.name,
        size=fixture.n_dof,
        batch_size=1,
        solve_fn=lambda: solve_spectre_assembled(**_backend_kwargs(fixture)),
        repeats=1,
    )

    assert timing.label == fixture.name
    assert timing.size == fixture.n_dof
    assert timing.batch_size == 1
    assert timing.repeats == 1
    assert timing.compile_and_solve_seconds > 0.0
    assert timing.steady_state_seconds > 0.0
    assert timing.per_system_seconds == timing.steady_state_seconds


def test_spectre_backend_timing_helper_rejects_invalid_repeats() -> None:
    with pytest.raises(ValueError, match="repeats"):
        benchmark_spectre_backend(
            label="invalid",
            size=1,
            batch_size=1,
            solve_fn=lambda: None,
            repeats=0,
        )
