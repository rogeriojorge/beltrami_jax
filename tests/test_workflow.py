from __future__ import annotations

import json

from dataclasses import replace

import numpy as np

from beltrami_jax.geometry import (
    assemble_fourier_beltrami_system,
    build_fourier_mode_basis,
    collocation_grid,
    torus_coordinates,
)
from beltrami_jax.io import load_problem_json, load_saved_solution, save_nonlinear_solution, save_problem_json
from beltrami_jax.iterative import gmres_solve
from beltrami_jax.nonlinear import solve_helicity_constrained_equilibrium
from beltrami_jax.operators import assemble_operator, assemble_rhs
from beltrami_jax.solver import solve_from_components
from beltrami_jax.types import BeltramiProblem, FourierBeltramiGeometry


def _example_problem(*, is_vacuum: bool = False) -> BeltramiProblem:
    geometry = FourierBeltramiGeometry(
        major_radius=3.1,
        minor_radius=0.9,
        elongation=1.25,
        triangularity=0.12,
        field_periods=2,
        radial_points=6,
        poloidal_points=12,
        toroidal_points=8,
        mass_shift=0.8,
        label="test_geometry",
    )
    basis = build_fourier_mode_basis(max_radial_order=1, max_poloidal_mode=2, max_toroidal_mode=1, label="test_basis")
    return BeltramiProblem.from_arraylike(
        geometry=geometry,
        basis=basis,
        psi=[0.07, -0.02],
        target_helicity=0.0065 if not is_vacuum else 0.0015,
        initial_mu=0.04,
        is_vacuum=is_vacuum,
        vacuum_strength=0.025 if is_vacuum else 0.0,
        solver="dense",
        tolerance=1.0e-9,
        max_iterations=6,
        label="test_problem",
    )


def test_fourier_geometry_assembly_shapes_and_symmetry() -> None:
    problem = _example_problem()
    assembly = assemble_fourier_beltrami_system(
        problem.geometry,
        problem.basis,
        mu=problem.initial_mu,
        psi=problem.psi,
        label=problem.label,
    )

    assert assembly.system.size == problem.basis.size
    assert assembly.system.d_ma.shape == (problem.basis.size, problem.basis.size)
    assert assembly.system.d_md.shape == (problem.basis.size, problem.basis.size)
    assert assembly.system.d_mb.shape == (problem.basis.size, 2)
    np.testing.assert_allclose(np.asarray(assembly.system.d_ma), np.asarray(assembly.system.d_ma).T, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(np.asarray(assembly.system.d_md), np.asarray(assembly.system.d_md).T, rtol=1e-10, atol=1e-10)


def test_collocation_grid_and_coordinates_are_finite() -> None:
    problem = _example_problem()
    radial, theta, zeta = collocation_grid(problem.geometry)
    major, vertical, toroidal = torus_coordinates(problem.geometry, radial, theta, zeta)
    assert radial.shape == (problem.geometry.radial_points,)
    assert theta.shape == (problem.geometry.poloidal_points,)
    assert zeta.shape == (problem.geometry.toroidal_points,)
    assert np.all(np.isfinite(np.asarray(major)))
    assert np.all(np.isfinite(np.asarray(vertical)))
    assert np.all(np.isfinite(np.asarray(toroidal)))


def test_problem_json_roundtrip(tmp_path) -> None:
    problem = _example_problem()
    path = save_problem_json(tmp_path / "problem.json", problem)
    restored = load_problem_json(path)
    assert restored.geometry.label == problem.geometry.label
    assert restored.basis.label == problem.basis.label
    np.testing.assert_allclose(np.asarray(restored.psi), np.asarray(problem.psi))
    assert restored.target_helicity == problem.target_helicity

    payload = json.loads(path.read_text())
    assert payload["geometry"]["major_radius"] == problem.geometry.major_radius


def test_gmres_matches_dense_geometry_solve() -> None:
    problem = _example_problem()
    assembly = assemble_fourier_beltrami_system(problem.geometry, problem.basis, mu=problem.initial_mu, psi=problem.psi, label=problem.label)
    operator = assemble_operator(assembly.system)
    rhs = assemble_rhs(assembly.system)
    dense = np.linalg.solve(np.asarray(operator), np.asarray(rhs))
    gmres = gmres_solve(
        lambda vector: operator @ vector,
        rhs,
        tolerance=1.0e-10,
        max_iterations=assembly.system.size,
    )
    np.testing.assert_allclose(np.asarray(gmres.solution), dense, rtol=1e-8, atol=1e-8)
    assert gmres.converged


def test_outer_helicity_loop_converges_and_can_be_saved(tmp_path) -> None:
    problem = _example_problem()
    target_assembly = assemble_fourier_beltrami_system(problem.geometry, problem.basis, mu=0.06, psi=problem.psi, label=problem.label)
    target_result = solve_from_components(target_assembly.system)
    problem = replace(problem, target_helicity=float(target_result.magnetic_helicity), initial_mu=0.03)
    result = solve_helicity_constrained_equilibrium(problem)
    assert result.iterations >= 1
    assert result.mu_history.shape[0] == result.helicity_history.shape[0]
    assert abs(float(result.helicity_history[-1]) - problem.target_helicity) <= 5.0e-6

    bundle_path = save_nonlinear_solution(tmp_path / "bundle.npz", result)
    loaded = load_saved_solution(bundle_path)
    np.testing.assert_allclose(np.asarray(loaded["solution"]), np.asarray(result.solve.solution))
    np.testing.assert_allclose(np.asarray(loaded["mu_history"]), np.asarray(result.mu_history))


def test_vacuum_geometry_produces_dmg() -> None:
    problem = _example_problem(is_vacuum=True)
    assembly = assemble_fourier_beltrami_system(
        problem.geometry,
        problem.basis,
        mu=problem.initial_mu,
        psi=problem.psi,
        is_vacuum=True,
        vacuum_strength=problem.vacuum_strength,
        label=problem.label,
    )
    assert assembly.system.is_vacuum is True
    assert assembly.system.d_mg is not None
    assert np.linalg.norm(np.asarray(assembly.system.d_mg)) > 0.0
