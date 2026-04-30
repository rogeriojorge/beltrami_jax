from __future__ import annotations

import numpy as np

from beltrami_jax import (
    SpectreInputSummary,
    SpectreMetricIntegrals,
    assemble_spectre_matrix_ad,
    assemble_spectre_matrix_ad_from_input,
    assemble_spectre_volume_matrices_from_input,
    build_spectre_dof_layout,
    build_spectre_interface_geometry,
    chebyshev_basis,
    load_packaged_spectre_case,
    load_packaged_spectre_linear_system,
    solve_spectre_assembled,
    solve_spectre_volumes_from_input,
    solve_spectre_volume_from_input,
    spectre_volume_flux_vector,
    zernike_axis_basis,
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
    for name in ("ate", "aze", "ato", "azo"):
        np.testing.assert_allclose(
            getattr(result.vector_potential, name),
            getattr(case.reference.vector_potential, name),
            rtol=3e-12,
            atol=8e-11,
        )
