from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from beltrami_jax.benchmark import benchmark_parameter_scan, benchmark_solve
from beltrami_jax.diagnostics import compare_against_reference, compute_solve_diagnostics
from beltrami_jax.operators import magnetic_energy, relative_residual_norm
from beltrami_jax.reference import load_packaged_reference
from beltrami_jax.solver import solve_from_components, solve_parameter_scan
from beltrami_jax.types import BeltramiLinearSystem


@pytest.mark.parametrize("name", ["g3v01l0fi_lvol1", "g1v03l0fi_lvol2", "g3v02l0fr_lu_lvol3", "g3v02l1fi_lvol1"])
def test_spec_fixture_solution_matches_dump(name: str) -> None:
    reference = load_packaged_reference(name)
    result = solve_from_components(reference.system)
    np.testing.assert_allclose(np.asarray(result.solution), np.asarray(reference.expected_solution), rtol=1e-11, atol=1e-11)
    assert float(result.relative_residual_norm) < 1e-11


def test_residual_helper_matches_result() -> None:
    reference = load_packaged_reference()
    result = solve_from_components(reference.system)
    rel = relative_residual_norm(result.solution, reference.system)
    np.testing.assert_allclose(np.asarray(rel), np.asarray(result.relative_residual_norm), rtol=1e-13, atol=1e-13)


def test_solved_energy_is_differentiable_in_mu() -> None:
    reference = load_packaged_reference()

    def solved_energy(mu_value: float) -> jax.Array:
        system = BeltramiLinearSystem.from_arraylike(
            d_ma=reference.system.d_ma,
            d_md=reference.system.d_md,
            d_mb=reference.system.d_mb,
            mu=mu_value,
            psi=reference.system.psi,
            label="grad-check",
        )
        result = solve_from_components(system)
        return magnetic_energy(result.solution, system)

    gradient = jax.grad(solved_energy)(float(reference.system.mu))
    assert jnp.isfinite(gradient)


def test_parameter_scan_matches_scalar_solve_at_center() -> None:
    reference = load_packaged_reference()
    mu0 = float(reference.system.mu)
    mu_values = jnp.asarray([mu0 - 0.02, mu0, mu0 + 0.02], dtype=jnp.float64)
    psi_values = jnp.repeat(reference.system.psi[None, :], repeats=3, axis=0)
    batched = solve_parameter_scan(reference.system.d_ma, reference.system.d_md, reference.system.d_mb, mu_values, psi_values)
    scalar = solve_from_components(reference.system)
    np.testing.assert_allclose(np.asarray(batched[1]), np.asarray(scalar.solution), rtol=1e-11, atol=1e-11)


def test_vacuum_rhs_path_uses_dmg() -> None:
    system = BeltramiLinearSystem.from_arraylike(
        d_ma=[[4.0, 1.0], [1.0, 3.0]],
        d_md=[[0.0, 0.0], [0.0, 0.0]],
        d_mb=[[1.0, 0.0], [0.0, 1.0]],
        d_mg=[0.5, -0.5],
        mu=0.0,
        psi=[1.0, 2.0],
        is_vacuum=True,
        label="vacuum-sanity",
    )
    result = solve_from_components(system)
    expected_rhs = np.array([-1.5, -1.5])
    np.testing.assert_allclose(np.asarray(result.rhs), expected_rhs)
    np.testing.assert_allclose(np.asarray(result.operator @ result.solution), expected_rhs, rtol=1e-12, atol=1e-12)


def test_verbose_solve_reports_progress(capsys: pytest.CaptureFixture[str]) -> None:
    reference = load_packaged_reference()
    result = solve_from_components(reference.system, verbose=True)
    captured = capsys.readouterr()
    assert "[beltrami_jax] solving" in captured.out
    assert "operator_fro_norm=" in captured.out
    assert "rhs_norm=" in captured.out
    assert "solution_norm=" in captured.out
    assert "residual_norm=" in captured.out
    assert "relative_residual_norm=" in captured.out
    assert float(result.relative_residual_norm) < 1e-11


def test_compute_solve_diagnostics_reports_finite_metrics() -> None:
    reference = load_packaged_reference("g3v02l1fi_lvol1")
    result = solve_from_components(reference.system)
    diagnostics = compute_solve_diagnostics(result, include_condition_number=True)
    assert diagnostics.label == reference.system.label
    assert diagnostics.size == reference.system.size
    assert diagnostics.operator_fro_norm > 0.0
    assert diagnostics.rhs_l2_norm > 0.0
    assert diagnostics.solution_l2_norm > 0.0
    assert diagnostics.residual_l2_norm < 1e-8
    assert diagnostics.relative_residual_norm < 1e-10
    assert diagnostics.condition_number_2 is not None
    assert np.isfinite(diagnostics.condition_number_2)


def test_compare_against_reference_is_machine_precision() -> None:
    reference = load_packaged_reference("g3v02l1fi_lvol1")
    result = solve_from_components(reference.system)
    comparison = compare_against_reference(reference, result)
    assert comparison.size == reference.system.size
    assert comparison.operator_relative_error < 1e-12
    assert comparison.rhs_relative_error < 1e-12
    assert comparison.solution_relative_error < 1e-10
    assert comparison.max_abs_solution_error < 1e-10


def test_benchmark_helpers_return_positive_timings() -> None:
    reference = load_packaged_reference("g1v03l0fi_lvol2")
    solve_benchmark = benchmark_solve(reference, repeats=1)
    assert solve_benchmark.size == reference.system.size
    assert solve_benchmark.compile_and_solve_seconds > 0.0
    assert solve_benchmark.steady_state_seconds > 0.0

    scan_benchmarks = benchmark_parameter_scan(reference, batch_sizes=(1, 2), repeats=1, relative_span=0.01)
    assert [item.batch_size for item in scan_benchmarks] == [1, 2]
    for item in scan_benchmarks:
        assert item.size == reference.system.size
        assert item.compile_and_solve_seconds > 0.0
        assert item.steady_state_seconds > 0.0
        assert item.per_system_seconds > 0.0


def test_benchmark_helpers_reject_invalid_inputs() -> None:
    reference = load_packaged_reference("g1v03l0fi_lvol2")
    vacuum_reference = load_packaged_reference("g3v02l0fr_lu_lvol3")

    with pytest.raises(ValueError, match="repeats"):
        benchmark_solve(reference, repeats=0)

    with pytest.raises(ValueError, match="plasma-region"):
        benchmark_parameter_scan(vacuum_reference, repeats=1)

    with pytest.raises(ValueError, match="repeats"):
        benchmark_parameter_scan(reference, repeats=0)
