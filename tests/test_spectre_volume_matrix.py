from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from beltrami_jax import (
    SpectreGlobalConstraintEvaluation,
    SpectreInputSummary,
    SpectreMetricIntegrals,
    assemble_spectre_matrix_ad,
    assemble_spectre_matrix_ad_from_input,
    assemble_spectre_volume_matrices_from_input,
    build_spectre_dof_layout,
    build_spectre_interface_geometry,
    chebyshev_basis,
    compute_spectre_btheta_mean,
    compute_spectre_plasma_current,
    compute_spectre_rotational_transform,
    compare_vector_potentials,
    load_packaged_spectre_case,
    load_packaged_spectre_linear_system,
    solve_spectre_assembled,
    solve_spectre_toml,
    solve_spectre_volumes_from_input,
    solve_spectre_volume_from_input,
    spectre_effective_current_profiles,
    spectre_lconstraint3_mu,
    spectre_normalized_fluxes,
    spectre_volume_flux_vector,
    zernike_axis_basis,
)
from beltrami_jax.spectre_solve import (
    _apply_lconstraint3_correction,
    _lconstraint3_total_pflux,
    _lconstraint3_unknowns,
)


def _synthetic_metric_integrals(*, lrad: int, mode_count: int, enforce_stellarator_symmetry: bool) -> SpectreMetricIntegrals:
    shape = (lrad + 1, lrad + 1, mode_count, mode_count)
    base = np.arange(np.prod(shape), dtype=np.float64).reshape(shape) / 100.0

    def tensor(offset: float) -> np.ndarray:
        return base + offset

    return SpectreMetricIntegrals(
        d_toocc=tensor(0.1),
        d_toocs=tensor(0.2),
        d_toosc=tensor(0.3),
        d_tooss=tensor(0.4),
        ttsscc=tensor(1.1),
        ttsscs=tensor(1.2),
        ttsssc=tensor(1.3),
        ttssss=tensor(1.4),
        tdstcc=tensor(2.1),
        tdstcs=tensor(2.2),
        tdstsc=tensor(2.3),
        tdstss=tensor(2.4),
        tdszcc=tensor(3.1),
        tdszcs=tensor(3.2),
        tdszsc=tensor(3.3),
        tdszss=tensor(3.4),
        ddttcc=tensor(4.1),
        ddttcs=tensor(4.2),
        ddttsc=tensor(4.3),
        ddttss=tensor(4.4),
        ddtzcc=tensor(5.1),
        ddtzcs=tensor(5.2),
        ddtzsc=tensor(5.3),
        ddtzss=tensor(5.4),
        ddzzcc=tensor(6.1),
        ddzzcs=tensor(6.2),
        ddzzsc=tensor(6.3),
        ddzzss=tensor(6.4),
        lrad=lrad,
        mpol=1,
        nfp=1,
        lvol=1,
        nt=8,
        nz=8,
        coordinate_singularity=False,
        enforce_stellarator_symmetry=enforce_stellarator_symmetry,
    )


def _synthetic_lconstraint3_summary(
    *,
    source: str = "synthetic_lconstraint3",
    igeometry: int = 3,
    nvol: int = 2,
    lrad: list[int] | None = None,
    tflux: list[float] | None = None,
    pflux: list[float] | None = None,
    ivolume: list[float] | None = None,
    isurf: list[float] | None = None,
    lfreebound: bool = False,
    curtor: float = 0.0,
    phiedge: float = 1.0,
) -> SpectreInputSummary:
    lrad = [1] * nvol if lrad is None else lrad
    packed = len(lrad)
    tflux = [float(i + 1) for i in range(packed)] if tflux is None else tflux
    pflux = [0.25 * float(i + 1) for i in range(packed)] if pflux is None else pflux
    ivolume = [float(i + 1) for i in range(packed)] if ivolume is None else ivolume
    isurf = [0.1 * float(i + 1) for i in range(max(packed - 1, 0))] if isurf is None else isurf
    return SpectreInputSummary(
        source=source,
        physics={
            "igeometry": igeometry,
            "nfp": 1,
            "nvol": nvol,
            "mpol": 0,
            "ntor": 0,
            "lrad": lrad,
            "tflux": tflux,
            "pflux": pflux,
            "mu": [0.0] * nvol,
            "helicity": [0.0] * packed,
            "lconstraint": 3,
            "ivolume": ivolume,
            "isurf": isurf,
            "lfreebound": lfreebound,
            "curtor": curtor,
            "phiedge": phiedge,
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


def test_radial_basis_matches_spectre_endpoint_recombination() -> None:
    cheby_left = np.asarray(chebyshev_basis(-1.0, 4))
    np.testing.assert_allclose(cheby_left[:, 0], np.asarray([1.0, 0.0, 0.0, 0.0, 0.0]))

    axis = np.asarray(zernike_axis_basis(4, 2))
    assert axis[2, 2] == 1.0 / 3.0
    assert axis[4, 2] == -3.0 / 5.0


def test_linitialize_cylindrical_interfaces_are_generated_like_spectre() -> None:
    case = load_packaged_spectre_case("G2V32L1Fi")
    geometry = build_spectre_interface_geometry(case.input_summary)

    assert geometry.interface_count == case.input_summary.packed_volume_count
    assert float(geometry.rbc[0, 0]) == 0.0
    assert float(geometry.rbc[-1, 0]) == 1.0

    normalized_tflux = np.asarray(case.input_summary.fluxes["tflux"]) / case.input_summary.fluxes["tflux"][3]
    np.testing.assert_allclose(
        np.asarray(geometry.rbc[1:4, 0]),
        np.sqrt(normalized_tflux[:3]),
    )


def test_matrix_ad_matches_spectre_for_cylindrical_axis_and_bulk_volumes() -> None:
    case = load_packaged_spectre_case("G2V32L1Fi")
    for lvol in (1, 2):
        assembled = assemble_spectre_matrix_ad_from_input(case.input_summary, lvol=lvol)
        fixture = load_packaged_spectre_linear_system(case_label=case.label, volume_index=lvol)
        np.testing.assert_allclose(np.asarray(assembled.d_ma), np.asarray(fixture.system.d_ma), rtol=2e-13, atol=8e-11)
        np.testing.assert_allclose(np.asarray(assembled.d_md), np.asarray(fixture.system.d_md), rtol=2e-13, atol=8e-13)


def test_matrix_ad_matches_spectre_for_free_boundary_bulk_and_vacuum_volumes() -> None:
    case = load_packaged_spectre_case("G3V8L3Free")
    for lvol in (1, 2, 9):
        assembled = assemble_spectre_matrix_ad_from_input(case.input_summary, lvol=lvol)
        fixture = load_packaged_spectre_linear_system(case_label=case.label, volume_index=lvol)
        np.testing.assert_allclose(np.asarray(assembled.d_ma), np.asarray(fixture.system.d_ma), rtol=2e-13, atol=8e-9)
        np.testing.assert_allclose(np.asarray(assembled.d_md), np.asarray(fixture.system.d_md), rtol=2e-13, atol=8e-13)


def test_matrix_ad_matches_spectre_for_toroidal_generated_interfaces() -> None:
    case = load_packaged_spectre_case("G3V3L3Fi")
    geometry = build_spectre_interface_geometry(case.input_summary)

    assert float(geometry.rbc[0, 0]) > 0.0
    for lvol in (1, 2):
        assembled = assemble_spectre_matrix_ad_from_input(case.input_summary, lvol=lvol)
        fixture = load_packaged_spectre_linear_system(case_label=case.label, volume_index=lvol)
        np.testing.assert_allclose(np.asarray(assembled.d_ma), np.asarray(fixture.system.d_ma), rtol=2e-13, atol=8e-11)
        np.testing.assert_allclose(np.asarray(assembled.d_md), np.asarray(fixture.system.d_md), rtol=2e-13, atol=8e-13)


def test_matrix_ad_non_stellarator_symmetric_branch_fills_odd_components() -> None:
    summary = SpectreInputSummary(
        source="synthetic_matrix_ad_non_stellsym",
        physics={
            "igeometry": 1,
            "nfp": 1,
            "nvol": 1,
            "mpol": 1,
            "ntor": 1,
            "lrad": [1],
            "enforce_stell_sym": False,
            "tflux": [1.0],
            "pflux": [0.0],
            "mu": [0.0],
            "helicity": [0.0],
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
    volume_map = build_spectre_dof_layout(summary).volume_maps[0]
    integrals = _synthetic_metric_integrals(
        lrad=volume_map.block.lrad,
        mode_count=volume_map.mode_count,
        enforce_stellarator_symmetry=False,
    )

    assembled = assemble_spectre_matrix_ad(volume_map, integrals)

    ii = 4  # SPECTRE internal mode (m=1, n=1)
    jj = 2  # SPECTRE internal mode (m=1, n=-1)
    ll = pp = 1
    mi = mj = 1
    ni = 1
    nj = -1
    ll1 = pp1 = 1
    expected_wzoto = 2 * (
        (ni * mj) * integrals.ttsscc[pp1, ll1, jj, ii]
        + mj * integrals.tdszcs[pp1, ll1, jj, ii]
        - ni * integrals.tdstcs[ll1, pp1, ii, jj]
        - integrals.ddtzss[pp1, ll1, jj, ii]
    )
    expected_hzoto = -integrals.d_tooss[pp1, ll1, jj, ii] + integrals.d_tooss[ll1, pp1, ii, jj]
    expected_wtozo = 2 * (
        (nj * mi) * integrals.ttsscc[pp1, ll1, jj, ii]
        - nj * integrals.tdstcs[pp1, ll1, jj, ii]
        + mi * integrals.tdszcs[ll1, pp1, ii, jj]
        - integrals.ddtzss[pp1, ll1, jj, ii]
    )

    row_ato = volume_map.ato[ll, ii] - 1
    row_azo = volume_map.azo[ll, ii] - 1
    col_azo = volume_map.azo[pp, jj] - 1
    col_ato = volume_map.ato[pp, jj] - 1
    assert row_ato >= 0 and row_azo >= 0 and col_azo >= 0 and col_ato >= 0
    np.testing.assert_allclose(assembled.d_ma[row_ato, col_azo], expected_wzoto)
    np.testing.assert_allclose(assembled.d_md[row_ato, col_azo], expected_hzoto)
    np.testing.assert_allclose(assembled.d_ma[row_azo, col_ato], expected_wtozo)
    assert np.count_nonzero(np.asarray(assembled.d_ma)) > volume_map.lagrange_multiplier_count
    assert np.count_nonzero(np.asarray(assembled.d_md)) > 0


def test_spectre_volume_flux_vector_matches_released_fixture_metadata() -> None:
    for label in ("G2V32L1Fi", "G3V3L3Fi", "G3V8L3Free"):
        case = load_packaged_spectre_case(label)
        first_volume = load_packaged_spectre_linear_system(case_label=label, volume_index=1)
        np.testing.assert_allclose(
            np.asarray(spectre_volume_flux_vector(case.input_summary, lvol=1)),
            np.asarray(first_volume.system.psi),
            rtol=3e-13,
            atol=3e-13,
        )

        second_lvol = min(2, case.input_summary.packed_volume_count)
        second_volume = load_packaged_spectre_linear_system(case_label=label, volume_index=second_lvol)
        np.testing.assert_allclose(
            np.asarray(spectre_volume_flux_vector(case.input_summary, lvol=second_lvol))[0],
            np.asarray(second_volume.system.psi)[0],
            rtol=3e-13,
            atol=3e-13,
        )


def test_spectre_lconstraint3_mu_matches_released_fixture_metadata() -> None:
    for label in ("G3V3L3Fi", "G3V8L3Free"):
        case = load_packaged_spectre_case(label)
        ivolume, isurf = spectre_effective_current_profiles(case.input_summary)
        assert ivolume.shape == (case.input_summary.packed_volume_count,)
        assert isurf.shape == (case.input_summary.packed_volume_count - 1,)
        for lvol in range(1, case.input_summary.nvol + 1):
            fixture = load_packaged_spectre_linear_system(case_label=label, volume_index=lvol)
            np.testing.assert_allclose(
                float(spectre_lconstraint3_mu(case.input_summary, lvol=lvol)),
                float(fixture.system.mu),
                rtol=3e-13,
                atol=3e-13,
            )


def test_solve_spectre_volume_from_input_produces_vector_potential_coefficients() -> None:
    case = load_packaged_spectre_case("G3V3L3Fi")
    lvol = 2
    fixture = load_packaged_spectre_linear_system(case_label=case.label, volume_index=lvol)
    result = solve_spectre_volume_from_input(
        case.input_summary,
        lvol=lvol,
        mu=fixture.system.mu,
        psi=fixture.system.psi,
    )

    np.testing.assert_allclose(np.asarray(result.solution), np.asarray(fixture.expected_solution), rtol=3e-12, atol=5e-11)
    reference_block = case.layout.split_vector_potential(case.reference.vector_potential)[lvol - 1]
    for name in ("ate", "aze", "ato", "azo"):
        np.testing.assert_allclose(
            getattr(result.vector_potential, name),
            getattr(reference_block, name),
            rtol=3e-12,
            atol=5e-11,
        )
    assert float(result.relative_residual_norm) < 1.0e-12
    assert len(result.derivative_vector_potentials) == 2
    assert result.constraint is None


def test_spectre_plasma_current_diagnostic_uses_solution_derivatives() -> None:
    case = load_packaged_spectre_case("G3V3L3Fi")
    lvol = 2
    fixture = load_packaged_spectre_linear_system(case_label=case.label, volume_index=lvol)
    result = solve_spectre_volume_from_input(
        case.input_summary,
        lvol=lvol,
        mu=fixture.system.mu,
        psi=fixture.system.psi,
    )

    currents = compute_spectre_plasma_current(
        case.input_summary,
        lvol=lvol,
        vector_potential=result.vector_potential,
        derivative_vector_potentials=result.derivative_vector_potentials,
    )

    np.testing.assert_allclose(np.asarray(currents.currents), np.asarray([-3.34899608e-05, 2.13891175e01]), rtol=2e-9)
    np.testing.assert_allclose(
        np.asarray(currents.derivative_currents),
        np.asarray([[-0.12509176, 1.00983483], [0.03711546, -0.30117631]]),
        rtol=2e-8,
        atol=2e-8,
    )

    radial_currents = compute_spectre_plasma_current(
        case.input_summary,
        lvol=lvol,
        vector_potential=result.vector_potential,
        derivative_vector_potentials=result.derivative_vector_potentials,
        include_radial_field=True,
    )
    assert radial_currents.derivative_currents.shape == (2, 2)
    assert np.all(np.isfinite(np.asarray(radial_currents.currents)))

    with pytest.raises(ValueError, match="outside"):
        compute_spectre_plasma_current(case.input_summary, lvol=99, vector_potential=result.vector_potential)


def test_spectre_btheta_mean_diagnostic_feeds_lconstraint3_global_system() -> None:
    case = load_packaged_spectre_case("G3V3L3Fi")
    lvol = 2
    fixture = load_packaged_spectre_linear_system(case_label=case.label, volume_index=lvol)
    result = solve_spectre_volume_from_input(
        case.input_summary,
        lvol=lvol,
        mu=fixture.system.mu,
        psi=fixture.system.psi,
    )
    diagnostic = compute_spectre_btheta_mean(
        case.input_summary,
        lvol=lvol,
        innout=0,
        vector_potential=result.vector_potential,
        derivative_vector_potentials=result.derivative_vector_potentials,
    )

    assert diagnostic.derivative_btheta.shape == (2,)
    assert np.all(np.isfinite(np.asarray(diagnostic.derivative_btheta)))
    np.testing.assert_allclose(float(diagnostic.btheta), -5.330092802070351e-06, rtol=2e-9, atol=2e-13)

    with pytest.raises(ValueError, match="innout"):
        compute_spectre_btheta_mean(case.input_summary, lvol=lvol, innout=2, vector_potential=result.vector_potential)
    with pytest.raises(ValueError, match="outside"):
        compute_spectre_btheta_mean(case.input_summary, lvol=99, innout=0, vector_potential=result.vector_potential)


def test_spectre_rotational_transform_diagnostic_matches_lconstraint1_targets() -> None:
    case = load_packaged_spectre_case("G2V32L1Fi")
    dof_layout = build_spectre_dof_layout(case.input_summary)
    bulk_result = None

    for lvol in range(1, case.input_summary.packed_volume_count + 1):
        fixture = load_packaged_spectre_linear_system(case_label=case.label, volume_index=lvol)
        result = solve_spectre_volume_from_input(
            case.input_summary,
            lvol=lvol,
            mu=fixture.system.mu,
            psi=fixture.system.psi,
        )
        if lvol == 2:
            bulk_result = result
        transform = compute_spectre_rotational_transform(
            case.input_summary,
            lvol=lvol,
            vector_potential=result.vector_potential,
            derivative_vector_potentials=result.derivative_vector_potentials,
            volume_map=dof_layout.volume_maps[lvol - 1],
        )

        assert transform.derivative_iota.shape == (2, 2)
        if lvol == 1:
            np.testing.assert_allclose(np.asarray(transform.iota[1]), case.input_summary.constraints["iota"][1])
        else:
            np.testing.assert_allclose(np.asarray(transform.iota[0]), case.input_summary.constraints["oita"][lvol - 1])
            np.testing.assert_allclose(np.asarray(transform.iota[1]), case.input_summary.constraints["iota"][lvol])
        assert np.all(np.isfinite(np.asarray(transform.derivative_iota)))

    assert bulk_result is not None
    no_derivative = compute_spectre_rotational_transform(
        case.input_summary,
        lvol=2,
        vector_potential=bulk_result.vector_potential,
        volume_map=dof_layout.volume_maps[1],
    )
    assert no_derivative.derivative_iota.shape == (2, 0)

    with pytest.raises(ValueError, match="outside"):
        compute_spectre_rotational_transform(case.input_summary, lvol=99, vector_potential=bulk_result.vector_potential)

    unsupported_summary = replace(case.input_summary, numeric={**case.input_summary.numeric, "lsparse": 1})
    with pytest.raises(NotImplementedError, match="Lsparse"):
        compute_spectre_rotational_transform(unsupported_summary, lvol=2, vector_potential=bulk_result.vector_potential)


def test_spectre_lconstraint2_local_helicity_solve_is_satisfied_at_reference_state() -> None:
    case = load_packaged_spectre_case("G3V3L2Fi_stability")
    lvol = 2
    fixture = load_packaged_spectre_linear_system(case_label=case.label, volume_index=lvol)

    result = solve_spectre_volume_from_input(
        case.input_summary,
        lvol=lvol,
        mu=fixture.system.mu,
        psi=fixture.system.psi,
        solve_local_constraints=True,
    )

    assert result.constraint is not None
    assert result.constraint.unknown_count == 1
    assert float(result.constraint.residual_norm) < 1.0e-12
    np.testing.assert_allclose(float(result.mu), float(fixture.system.mu), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(result.solution), np.asarray(fixture.expected_solution), rtol=3e-12, atol=5e-11)


def test_spectre_zero_unknown_local_constraint_solve_returns_constraint_record() -> None:
    case = load_packaged_spectre_case("G3V3L3Fi")
    lvol = 2
    fixture = load_packaged_spectre_linear_system(case_label=case.label, volume_index=lvol)

    result = solve_spectre_volume_from_input(
        case.input_summary,
        lvol=lvol,
        mu=fixture.system.mu,
        psi=fixture.system.psi,
        solve_local_constraints=True,
    )

    assert result.constraint is not None
    assert result.constraint.unknown_count == 0
    assert float(result.constraint.residual_norm) == 0.0
    np.testing.assert_allclose(np.asarray(result.solution), np.asarray(fixture.expected_solution), rtol=3e-12, atol=5e-11)


def test_spectre_lconstraint1_local_transform_solve_matches_spectre_reference_state() -> None:
    case = load_packaged_spectre_case("G2V32L1Fi")
    lvol = 2
    fixture = load_packaged_spectre_linear_system(case_label=case.label, volume_index=lvol)

    result = solve_spectre_volume_from_input(
        case.input_summary,
        lvol=lvol,
        solve_local_constraints=True,
        max_constraint_iterations=8,
    )

    assert result.constraint is not None
    assert result.constraint.unknown_count == 2
    assert float(result.constraint.residual_norm) < 2.0e-12
    np.testing.assert_allclose(float(result.mu), float(fixture.system.mu), rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(np.asarray(result.psi), np.asarray(fixture.system.psi), rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(np.asarray(result.solution), np.asarray(fixture.expected_solution), rtol=3e-11, atol=5e-10)


def test_spectre_lconstraint1_axis_transform_solve_matches_spectre_reference_state() -> None:
    case = load_packaged_spectre_case("G2V32L1Fi")
    lvol = 1
    fixture = load_packaged_spectre_linear_system(case_label=case.label, volume_index=lvol)

    result = solve_spectre_volume_from_input(
        case.input_summary,
        lvol=lvol,
        solve_local_constraints=True,
    )

    assert result.constraint is not None
    assert result.constraint.unknown_count == 1
    assert float(result.constraint.residual_norm) < 2.0e-12
    np.testing.assert_allclose(float(result.mu), float(fixture.system.mu), rtol=4e-11, atol=4e-11)
    np.testing.assert_allclose(np.asarray(result.psi), np.asarray(fixture.system.psi), rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(np.asarray(result.solution), np.asarray(fixture.expected_solution), rtol=4e-11, atol=5e-10)


def test_spectre_local_constraint_solve_validates_iteration_count() -> None:
    case = load_packaged_spectre_case("G3V3L2Fi_stability")
    with pytest.raises(ValueError, match="max_constraint_iterations"):
        solve_spectre_volume_from_input(
            case.input_summary,
            lvol=2,
            solve_local_constraints=True,
            max_constraint_iterations=0,
        )


def test_toml_assembled_volume_matrices_solve_and_unpack_reference_coefficients() -> None:
    case = load_packaged_spectre_case("G2V32L1Fi")
    lvol = 2
    matrices = assemble_spectre_volume_matrices_from_input(case.input_summary, lvol=lvol)
    fixture = load_packaged_spectre_linear_system(case_label=case.label, volume_index=lvol)
    result = solve_spectre_assembled(
        d_ma=matrices.d_ma,
        d_md=matrices.d_md,
        d_mb=matrices.d_mb,
        d_mg=matrices.d_mg,
        mu=fixture.system.mu,
        psi=fixture.system.psi,
        is_vacuum=fixture.system.is_vacuum,
        include_d_mg_in_rhs=fixture.system.include_d_mg_in_rhs,
    )
    np.testing.assert_allclose(np.asarray(result.solution), np.asarray(fixture.expected_solution), rtol=3e-12, atol=5e-11)

    dof_layout = build_spectre_dof_layout(case.input_summary)
    solved_block = dof_layout.volume_maps[lvol - 1].unpack_solution(np.asarray(result.solution))
    reference_block = case.layout.split_vector_potential(case.reference.vector_potential)[lvol - 1]
    for name in ("ate", "aze", "ato", "azo"):
        np.testing.assert_allclose(getattr(solved_block, name), getattr(reference_block, name), rtol=3e-12, atol=5e-11)


def test_toml_multi_volume_solve_returns_full_vector_potential_block() -> None:
    case = load_packaged_spectre_case("G3V3L3Fi")
    mu_by_volume = {}
    psi_by_volume = {}
    for lvol in range(1, case.input_summary.packed_volume_count + 1):
        fixture = load_packaged_spectre_linear_system(case_label=case.label, volume_index=lvol)
        mu_by_volume[lvol] = fixture.system.mu
        psi_by_volume[lvol] = fixture.system.psi

    result = solve_spectre_volumes_from_input(case.input_summary, mu=mu_by_volume, psi=psi_by_volume)

    assert len(result.volume_solves) == case.input_summary.packed_volume_count
    assert result.vector_potential.shape == case.reference.vector_potential.shape
    assert float(result.max_relative_residual_norm) < 2.0e-12
    assert result.residual_norms.shape == (case.input_summary.packed_volume_count,)
    assert result.relative_residual_norms.shape == (case.input_summary.packed_volume_count,)
    assert set(result.component_norms()) == {"ate", "aze", "ato", "azo"}
    for name in ("ate", "aze", "ato", "azo"):
        np.testing.assert_allclose(
            getattr(result.vector_potential, name),
            getattr(case.reference.vector_potential, name),
            rtol=3e-12,
            atol=8e-11,
        )


def test_toml_multi_volume_lconstraint3_global_solve_matches_reference_without_state_injection() -> None:
    case = load_packaged_spectre_case("G3V3L3Fi")

    result = solve_spectre_volumes_from_input(case.input_summary, solve_local_constraints=True)

    assert result.global_constraint is not None
    assert result.global_constraint.unknowns == ("dpflux_lvol2", "dpflux_lvol3")
    assert float(result.global_constraint.initial_residual_norm) > 1.0e-4
    assert float(result.global_constraint.residual_norm) < 1.0e-13
    for volume_solve in result.volume_solves:
        fixture = load_packaged_spectre_linear_system(case_label=case.label, volume_index=volume_solve.lvol)
        np.testing.assert_allclose(float(volume_solve.mu), float(fixture.system.mu), rtol=3e-13, atol=3e-13)
        np.testing.assert_allclose(np.asarray(volume_solve.psi), np.asarray(fixture.system.psi), rtol=3e-12, atol=3e-12)
    comparison = compare_vector_potentials(result.vector_potential, case.reference.vector_potential, label="G3V3L3Fi")
    assert comparison.global_relative_error < 2.0e-13

    with pytest.raises(ValueError, match="all packed volumes"):
        solve_spectre_volumes_from_input(case.input_summary, volumes=(1, 2), solve_local_constraints=True)
    with pytest.raises(ValueError, match="do not pass overrides"):
        solve_spectre_volumes_from_input(case.input_summary, mu={1: 0.0}, solve_local_constraints=True)


def test_spectre_solve_helpers_cover_empty_and_invalid_volume_selections() -> None:
    case = load_packaged_spectre_case("G3V3L3Fi")

    empty = solve_spectre_volumes_from_input(case.input_summary, volumes=(), verbose=True)
    assert empty.vector_potential.shape == (0, 0)
    assert float(empty.max_relative_residual_norm) == 0.0

    empty_from_toml = solve_spectre_toml(case.input_summary.source, volumes=())
    assert empty_from_toml.vector_potential.shape == (0, 0)

    with pytest.raises(ValueError, match="volumes"):
        solve_spectre_volumes_from_input(case.input_summary, volumes=(99,))
    with pytest.raises(ValueError, match="outside"):
        solve_spectre_volume_from_input(case.input_summary, lvol=99)
    with pytest.raises(ValueError, match="outside"):
        spectre_volume_flux_vector(case.input_summary, lvol=99)


def test_spectre_lconstraint3_helper_branches_and_validation_errors() -> None:
    empty_constraint = SpectreGlobalConstraintEvaluation(
        lconstraint=3,
        unknowns=(),
        initial_residual=np.zeros((0,), dtype=np.float64),
        jacobian=np.zeros((0, 0), dtype=np.float64),
        correction=np.zeros((0,), dtype=np.float64),
        final_residual=np.zeros((0,), dtype=np.float64),
    )
    assert float(empty_constraint.initial_residual_norm) == 0.0
    assert float(empty_constraint.residual_norm) == 0.0

    base_summary = _synthetic_lconstraint3_summary()
    non_lconstraint3 = replace(base_summary, physics={**base_summary.physics, "lconstraint": 0})
    ivolume, isurf = spectre_effective_current_profiles(non_lconstraint3)
    np.testing.assert_allclose(ivolume, np.asarray(non_lconstraint3.physics["ivolume"], dtype=np.float64))
    np.testing.assert_allclose(isurf, np.asarray(non_lconstraint3.physics["isurf"], dtype=np.float64))
    with pytest.raises(ValueError, match="only applies"):
        spectre_lconstraint3_mu(non_lconstraint3, lvol=1)

    with pytest.raises(ValueError, match="ivolume length"):
        spectre_effective_current_profiles(_synthetic_lconstraint3_summary(ivolume=[1.0]))
    with pytest.raises(ValueError, match="isurf"):
        spectre_effective_current_profiles(_synthetic_lconstraint3_summary(isurf=[]))

    free = _synthetic_lconstraint3_summary(
        nvol=2,
        lrad=[1, 1, 1],
        tflux=[1.0, 2.0, 3.0],
        pflux=[0.1, 0.2, 0.3],
        ivolume=[1.0, 2.0, 999.0],
        isurf=[3.0, 5.0],
        lfreebound=True,
        curtor=20.0,
    )
    ivolume, isurf = spectre_effective_current_profiles(free)
    np.testing.assert_allclose(ivolume, np.asarray([2.0, 4.0, 4.0]))
    np.testing.assert_allclose(isurf, np.asarray([6.0, 10.0]))
    assert _lconstraint3_unknowns(free) == ("dpflux_lvol2", "dpflux_lvol3", "dtflux_lvol3")
    assert _lconstraint3_total_pflux(free) == 0.0

    with pytest.raises(ValueError, match="nonzero curtor"):
        spectre_effective_current_profiles(
            _synthetic_lconstraint3_summary(
                nvol=2,
                lrad=[1, 1, 1],
                ivolume=[0.0, 0.0, 0.0],
                isurf=[0.0, 0.0],
                lfreebound=True,
                curtor=1.0,
            )
        )
    with pytest.raises(ValueError, match="zero curtor"):
        spectre_effective_current_profiles(
            _synthetic_lconstraint3_summary(
                nvol=2,
                lrad=[1, 1, 1],
                ivolume=[1.0, 2.0, 3.0],
                isurf=[0.0, 0.0],
                lfreebound=True,
                curtor=0.0,
            )
        )

    cylindrical = _synthetic_lconstraint3_summary(
        source="synthetic_cylindrical_lconstraint3",
        igeometry=1,
        nvol=2,
        tflux=[1.0, 2.0],
        pflux=[1.0, 99.0],
        ivolume=[1.0, 3.0],
        isurf=[0.5],
        phiedge=2.0,
    )
    _, pflux = spectre_normalized_fluxes(cylindrical)
    np.testing.assert_allclose(pflux, np.asarray([0.5, 0.0]))
    assert _lconstraint3_unknowns(cylindrical) == ("dpflux_lvol2", "dpflux_lvol1")
    np.testing.assert_allclose(
        _lconstraint3_total_pflux(cylindrical),
        99.0 / 2.0 * 2.0 / (2.0 * np.pi),
    )
    assert float(spectre_lconstraint3_mu(cylindrical, lvol=3)) == 0.0

    with pytest.raises(ValueError, match="zero toroidal-flux interval"):
        spectre_lconstraint3_mu(
            _synthetic_lconstraint3_summary(tflux=[0.0, 1.0], ivolume=[1.0, 2.0], isurf=[0.1]),
            lvol=1,
        )
    with pytest.raises(ValueError, match="nonzero"):
        _lconstraint3_total_pflux(_synthetic_lconstraint3_summary(igeometry=1, tflux=[1.0, 0.0]))


def test_spectre_lconstraint3_correction_updates_coupled_fluxes() -> None:
    free = _synthetic_lconstraint3_summary(nvol=2, lrad=[1, 1, 1], lfreebound=True)
    free_solves = (
        SimpleNamespace(lvol=1, psi=np.asarray([1.0, 10.0])),
        SimpleNamespace(lvol=2, psi=np.asarray([2.0, 20.0])),
        SimpleNamespace(lvol=3, psi=np.asarray([3.0, 30.0])),
    )
    updated_free = _apply_lconstraint3_correction(free, free_solves, np.asarray([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(np.asarray(updated_free[1]), np.asarray([1.0, 10.0]))
    np.testing.assert_allclose(np.asarray(updated_free[2]), np.asarray([2.0, 19.0]))
    np.testing.assert_allclose(np.asarray(updated_free[3]), np.asarray([0.0, 28.0]))

    cylindrical = _synthetic_lconstraint3_summary(igeometry=1, nvol=2)
    cylindrical_solves = (
        SimpleNamespace(lvol=1, psi=np.asarray([1.0, 10.0])),
        SimpleNamespace(lvol=2, psi=np.asarray([2.0, 20.0])),
    )
    updated_cylindrical = _apply_lconstraint3_correction(cylindrical, cylindrical_solves, np.asarray([2.0, 3.0]))
    np.testing.assert_allclose(np.asarray(updated_cylindrical[1]), np.asarray([1.0, 7.0]))
    np.testing.assert_allclose(np.asarray(updated_cylindrical[2]), np.asarray([2.0, 18.0]))


def test_spectre_flux_helpers_validate_bad_summary_data() -> None:
    bad_lengths = SpectreInputSummary(
        source="bad_lengths",
        physics={
            "igeometry": 1,
            "nfp": 1,
            "nvol": 2,
            "mpol": 0,
            "ntor": 0,
            "lrad": [1, 1],
            "tflux": [1.0],
            "pflux": [0.0, 0.0],
            "mu": [0.0, 0.0],
            "helicity": [0.0, 0.0],
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
    with pytest.raises(ValueError, match="tflux and pflux"):
        spectre_normalized_fluxes(bad_lengths)

    zero_reference = SpectreInputSummary(
        source="zero_reference",
        physics={
            "igeometry": 1,
            "nfp": 1,
            "nvol": 2,
            "mpol": 0,
            "ntor": 0,
            "lrad": [1, 1],
            "tflux": [1.0, 0.0],
            "pflux": [0.0, 0.0],
            "mu": [0.0, 0.0],
            "helicity": [0.0, 0.0],
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
    with pytest.raises(ValueError, match="nonzero"):
        spectre_normalized_fluxes(zero_reference)
