from __future__ import annotations

import numpy as np
import pytest

from beltrami_jax import (
    SpectreInputSummary,
    SpectreConstraintDiagnostics,
    SpectreConstraintTargets,
    SpectrePlasmaCurrentDiagnostic,
    SpectreRotationalTransformDiagnostic,
    build_spectre_dof_layout,
    evaluate_spectre_local_constraints,
    evaluate_spectre_constraints,
    evaluate_spectre_helicity_constraint,
    load_all_packaged_spectre_linear_systems,
    load_packaged_spectre_linear_system,
    solve_spectre_assembled,
    solve_spectre_beltrami_branch,
    solve_spectre_beltrami_branch_numpy,
    spectre_branch_unknowns,
    spectre_constraint_dof_count,
    spectre_local_unknown_count,
)


def _branch_kwargs(fixture):
    system = fixture.system
    return {
        "d_ma": system.d_ma,
        "d_md": system.d_md,
        "d_mb": system.d_mb,
        "d_mg": system.d_mg,
        "mu": system.mu,
        "psi": system.psi,
        "lconstraint": fixture.lconstraint,
        "is_vacuum": fixture.is_vacuum,
        "coordinate_singularity": fixture.coordinate_singularity,
    }


def test_spectre_branch_solve_matches_packaged_primary_solutions() -> None:
    for fixture in load_all_packaged_spectre_linear_systems():
        result = solve_spectre_beltrami_branch(**_branch_kwargs(fixture))
        expected = np.asarray(fixture.expected_solution)
        relative_error = np.linalg.norm(np.asarray(result.solution) - expected) / max(np.linalg.norm(expected), 1e-300)
        assert relative_error < 3e-12
        assert float(result.relative_residual_norms[0]) < 1e-11
        np.testing.assert_allclose(np.asarray(result.operator), np.asarray(fixture.matrix), rtol=3e-12, atol=3e-12)
        np.testing.assert_allclose(np.asarray(result.rhs), np.asarray(fixture.rhs), rtol=3e-12, atol=3e-12)


def test_spectre_branch_numpy_wrapper_is_adapter_friendly() -> None:
    fixture = load_packaged_spectre_linear_system("G2V32L1Fi/lvol2")
    result = solve_spectre_beltrami_branch_numpy(**_branch_kwargs(fixture))

    assert isinstance(result["solution"], np.ndarray)
    assert isinstance(result["solutions"], np.ndarray)
    assert isinstance(result["magnetic_energy"], float)
    assert isinstance(result["magnetic_helicity"], float)
    assert result["branch_unknowns"] == ("mu", "dpflux")
    np.testing.assert_allclose(result["solution"], np.asarray(fixture.expected_solution), rtol=3e-12, atol=3e-12)


def test_spectre_branch_derivative_rhs_formulas() -> None:
    plasma = load_packaged_spectre_linear_system("G2V32L1Fi/lvol2")
    plasma_result = solve_spectre_beltrami_branch(**_branch_kwargs(plasma))
    np.testing.assert_allclose(
        np.asarray(plasma_result.derivative_rhs[0]),
        np.asarray(plasma.system.d_md @ plasma_result.solution),
        rtol=3e-12,
        atol=3e-12,
    )
    np.testing.assert_allclose(
        np.asarray(plasma_result.derivative_rhs[1]),
        -np.asarray(plasma.system.d_mb[:, 1]),
        rtol=3e-12,
        atol=3e-12,
    )

    vacuum = load_packaged_spectre_linear_system("G3V8L3Free/lvol9")
    vacuum_result = solve_spectre_beltrami_branch(**_branch_kwargs(vacuum))
    assert vacuum_result.branch_unknowns == ("dtflux", "dpflux")
    np.testing.assert_allclose(np.asarray(vacuum_result.derivative_rhs[0]), -np.asarray(vacuum.system.d_mb[:, 0]))
    np.testing.assert_allclose(np.asarray(vacuum_result.derivative_rhs[1]), -np.asarray(vacuum.system.d_mb[:, 1]))


def test_spectre_branch_coordinate_singularity_current_branch() -> None:
    result = solve_spectre_beltrami_branch(
        d_ma=np.eye(2),
        d_md=np.zeros((2, 2)),
        d_mb=np.eye(2),
        d_mg=np.asarray([0.25, -0.5]),
        mu=0.0,
        psi=np.asarray([1.0, 2.0]),
        lconstraint=-2,
        is_vacuum=False,
        coordinate_singularity=True,
    )

    assert result.branch_unknowns == ("dtflux",)
    np.testing.assert_allclose(np.asarray(result.solution), np.asarray([-1.25, -1.5]))
    np.testing.assert_allclose(np.asarray(result.derivative_rhs[0]), np.asarray([-1.0, -0.0]))
    np.testing.assert_allclose(np.asarray(result.derivative_rhs[1]), np.asarray([0.0, 0.0]))


def test_spectre_branch_requires_dmg_when_branch_uses_source() -> None:
    with pytest.raises(ValueError, match="d_mg is required"):
        solve_spectre_beltrami_branch(
            d_ma=np.eye(2),
            d_md=np.zeros((2, 2)),
            d_mb=np.eye(2),
            mu=0.0,
            psi=np.asarray([1.0, 2.0]),
            d_mg=None,
            lconstraint=-2,
            is_vacuum=False,
            coordinate_singularity=True,
        )


@pytest.mark.parametrize(
    ("lconstraint", "is_vacuum", "coordinate_singularity", "expected_count", "expected_unknowns"),
    [
        (-2, False, True, 1, ("dtflux",)),
        (-1, False, False, 0, ()),
        (0, False, False, 0, ()),
        (1, False, True, 1, ("mu",)),
        (1, False, False, 2, ("mu", "dpflux")),
        (2, False, False, 1, ("mu",)),
        (3, False, False, 0, ()),
        (-1, True, False, 0, ()),
        (0, True, False, 2, ("dtflux", "dpflux")),
        (1, True, False, 2, ("dtflux", "dpflux")),
        (2, True, False, 2, ("dtflux", "dpflux")),
        (3, True, False, 0, ()),
    ],
)
def test_spectre_constraint_dof_table(
    lconstraint: int,
    is_vacuum: bool,
    coordinate_singularity: bool,
    expected_count: int,
    expected_unknowns: tuple[str, ...],
) -> None:
    assert (
        spectre_constraint_dof_count(
            lconstraint=lconstraint,
            is_vacuum=is_vacuum,
            coordinate_singularity=coordinate_singularity,
        )
        == expected_count
    )
    assert (
        spectre_branch_unknowns(
            lconstraint=lconstraint,
            is_vacuum=is_vacuum,
            coordinate_singularity=coordinate_singularity,
        )
        == expected_unknowns
    )


def test_evaluate_spectre_constraints_matches_branch_formulas() -> None:
    transform = np.asarray([[1.2, 0.3, 0.4], [1.7, 0.5, 0.6]])
    current = np.asarray([[7.0, 0.1, 0.2], [11.0, 0.3, 0.4]])

    current_branch = evaluate_spectre_constraints(
        SpectreConstraintTargets(lconstraint=-2, is_vacuum=False, coordinate_singularity=True, curpol=10.0),
        SpectreConstraintDiagnostics(plasma_current=current),
    )
    np.testing.assert_allclose(np.asarray(current_branch.residual), np.asarray([1.0]))
    np.testing.assert_allclose(np.asarray(current_branch.jacobian), np.asarray([[0.3]]))

    vacuum_current = evaluate_spectre_constraints(
        SpectreConstraintTargets(lconstraint=0, is_vacuum=True, curtor=6.5, curpol=10.0),
        SpectreConstraintDiagnostics(plasma_current=current),
    )
    np.testing.assert_allclose(np.asarray(vacuum_current.residual), np.asarray([0.5, 1.0]))
    np.testing.assert_allclose(np.asarray(vacuum_current.jacobian), current[:, 1:3])

    plasma_iota = evaluate_spectre_constraints(
        SpectreConstraintTargets(lconstraint=1, is_vacuum=False, iota_inner=1.0, iota_outer=1.5),
        SpectreConstraintDiagnostics(rotational_transform=transform),
    )
    np.testing.assert_allclose(np.asarray(plasma_iota.residual), np.asarray([0.2, 0.2]))
    np.testing.assert_allclose(np.asarray(plasma_iota.jacobian), transform[:, 1:3])

    coordinate_iota = evaluate_spectre_constraints(
        SpectreConstraintTargets(lconstraint=1, is_vacuum=False, coordinate_singularity=True, iota_outer=1.5),
        SpectreConstraintDiagnostics(rotational_transform=transform),
    )
    np.testing.assert_allclose(np.asarray(coordinate_iota.residual), np.asarray([0.2]))
    np.testing.assert_allclose(np.asarray(coordinate_iota.jacobian), np.asarray([[0.5]]))

    vacuum_iota_current = evaluate_spectre_constraints(
        SpectreConstraintTargets(lconstraint=1, is_vacuum=True, iota_inner=1.0, curpol=10.0),
        SpectreConstraintDiagnostics(rotational_transform=transform, plasma_current=current),
    )
    np.testing.assert_allclose(np.asarray(vacuum_iota_current.residual), np.asarray([0.2, 1.0]))
    np.testing.assert_allclose(np.asarray(vacuum_iota_current.jacobian), np.asarray([[0.3, 0.4], [0.3, 0.4]]))

    helicity = evaluate_spectre_constraints(
        SpectreConstraintTargets(lconstraint=2, is_vacuum=False, helicity=8.0),
        SpectreConstraintDiagnostics(helicity=np.asarray(8.25), helicity_derivatives=np.asarray([0.75, 0.0])),
    )
    np.testing.assert_allclose(np.asarray(helicity.residual), np.asarray([0.25]))
    np.testing.assert_allclose(np.asarray(helicity.jacobian), np.asarray([[0.75]]))


def test_evaluate_spectre_constraints_zero_and_missing_diagnostics() -> None:
    zero = evaluate_spectre_constraints(
        SpectreConstraintTargets(lconstraint=3, is_vacuum=False),
        SpectreConstraintDiagnostics(),
    )
    assert zero.unknowns == ()
    assert zero.residual.shape == (0,)
    assert zero.jacobian.shape == (0, 0)

    with pytest.raises(ValueError, match="rotational_transform"):
        evaluate_spectre_constraints(
            SpectreConstraintTargets(lconstraint=1, is_vacuum=False, iota_outer=1.0),
            SpectreConstraintDiagnostics(),
        )

    with pytest.raises(ValueError, match="unsupported"):
        spectre_constraint_dof_count(lconstraint=99, is_vacuum=False)


def _summary_for_local_constraints(*, lconstraint: int, nvol: int = 2, free_boundary: bool = False) -> SpectreInputSummary:
    packed = nvol + 1 if free_boundary else nvol
    return SpectreInputSummary(
        source="synthetic_local_constraints",
        physics={
            "igeometry": 2,
            "nfp": 1,
            "nvol": nvol,
            "mpol": 0,
            "ntor": 0,
            "lrad": [1] * packed,
            "lfreebound": free_boundary,
            "tflux": np.linspace(0.5, 1.0, packed).tolist(),
            "pflux": np.linspace(0.1, 0.2, packed).tolist(),
            "mu": [0.0] * nvol,
            "iota": [0.7, 0.8, 0.9, 1.0][: packed + 1],
            "oita": [0.65, 0.75, 0.85, 0.95][: packed + 1],
            "helicity": [0.2] * packed,
            "curtor": 6.5,
            "curpol": 10.0,
            "lconstraint": lconstraint,
        },
        numeric={},
        global_options={},
        local={},
        diagnostics={},
        rbc={},
        zbs={},
        rbs={},
        zbc={},
    )


def _dummy_backend_solve():
    return solve_spectre_assembled(
        d_ma=np.eye(2),
        d_md=0.25 * np.eye(2),
        d_mb=np.eye(2),
        mu=0.1,
        psi=np.asarray([1.0, -0.5]),
    )


def test_evaluate_spectre_local_constraint_branches_from_toml_targets() -> None:
    current = SpectrePlasmaCurrentDiagnostic(
        lvol=1,
        innout=0,
        currents=np.asarray([7.0, 11.0]),
        derivative_currents=np.asarray([[0.1, 0.2], [0.3, 0.4]]),
    )
    transform = SpectreRotationalTransformDiagnostic(
        iota=np.asarray([0.9, 1.1]),
        derivative_iota=np.asarray([[0.5, 0.6], [0.7, 0.8]]),
    )
    solve = _dummy_backend_solve()

    current_summary = _summary_for_local_constraints(lconstraint=-2, nvol=1)
    axis_map = build_spectre_dof_layout(current_summary).volume_maps[0]
    axis_eval = evaluate_spectre_local_constraints(
        current_summary,
        lvol=1,
        volume_map=axis_map,
        solve=solve,
        currents=current,
    )
    np.testing.assert_allclose(np.asarray(axis_eval.residual), np.asarray([1.0]))
    np.testing.assert_allclose(np.asarray(axis_eval.jacobian), np.asarray([[0.3]]))
    assert spectre_local_unknown_count(current_summary, axis_map) == 1

    vacuum_summary = _summary_for_local_constraints(lconstraint=0, nvol=1, free_boundary=True)
    vacuum_map = build_spectre_dof_layout(vacuum_summary).volume_maps[1]
    vacuum_eval = evaluate_spectre_local_constraints(
        vacuum_summary,
        lvol=2,
        volume_map=vacuum_map,
        solve=solve,
        currents=current,
    )
    np.testing.assert_allclose(np.asarray(vacuum_eval.residual), np.asarray([0.5, 1.0]))
    np.testing.assert_allclose(np.asarray(vacuum_eval.jacobian), np.asarray([[0.1, 0.2], [0.3, 0.4]]))

    iota_summary = _summary_for_local_constraints(lconstraint=1, nvol=2)
    bulk_map = build_spectre_dof_layout(iota_summary).volume_maps[1]
    iota_eval = evaluate_spectre_local_constraints(
        iota_summary,
        lvol=2,
        volume_map=bulk_map,
        solve=solve,
        transform=transform,
    )
    np.testing.assert_allclose(np.asarray(iota_eval.residual), np.asarray([0.25, 0.3]))
    np.testing.assert_allclose(np.asarray(iota_eval.jacobian), np.asarray([[0.5, 0.6], [0.7, 0.8]]))

    vacuum_iota_summary = _summary_for_local_constraints(lconstraint=1, nvol=1, free_boundary=True)
    vacuum_iota_map = build_spectre_dof_layout(vacuum_iota_summary).volume_maps[1]
    vacuum_iota_eval = evaluate_spectre_local_constraints(
        vacuum_iota_summary,
        lvol=2,
        volume_map=vacuum_iota_map,
        solve=solve,
        transform=transform,
        currents=current,
    )
    np.testing.assert_allclose(np.asarray(vacuum_iota_eval.residual), np.asarray([0.25, 1.0]))
    np.testing.assert_allclose(np.asarray(vacuum_iota_eval.jacobian), np.asarray([[0.5, 0.6], [0.3, 0.4]]))


def test_evaluate_spectre_helicity_constraint_from_backend_integral() -> None:
    summary = _summary_for_local_constraints(lconstraint=2, nvol=1)
    volume_map = build_spectre_dof_layout(summary).volume_maps[0]
    solve = _dummy_backend_solve()
    evaluation = evaluate_spectre_helicity_constraint(
        summary,
        lvol=1,
        volume_map=volume_map,
        solve=solve,
        d_md=0.25 * np.eye(2),
    )
    assert evaluation.unknown_count == 1
    assert evaluation.jacobian.shape == (1, 1)

    with pytest.raises(ValueError, match="plasma Lconstraint=2"):
        evaluate_spectre_helicity_constraint(
            _summary_for_local_constraints(lconstraint=3, nvol=1),
            lvol=1,
            volume_map=volume_map,
            solve=solve,
            d_md=0.25 * np.eye(2),
        )
