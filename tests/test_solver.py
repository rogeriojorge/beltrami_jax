from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from beltrami_jax.operators import magnetic_energy, relative_residual_norm
from beltrami_jax.reference import load_packaged_reference
from beltrami_jax.solver import solve_from_components, solve_parameter_scan
from beltrami_jax.types import BeltramiLinearSystem


def test_spec_fixture_solution_matches_dump() -> None:
    reference = load_packaged_reference()
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
    assert "residual_norm=" in captured.out
    assert "relative_residual_norm=" in captured.out
    assert float(result.relative_residual_norm) < 1e-11
