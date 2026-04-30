from __future__ import annotations

import numpy as np

from beltrami_jax import (
    SpectreBoundaryNormalField,
    SpectreInputSummary,
    assemble_spectre_matrix_bg,
    assemble_spectre_matrix_bg_from_input,
    build_spectre_boundary_normal_field,
    build_spectre_dof_layout,
    load_all_packaged_spectre_cases,
    load_packaged_spectre_case,
    load_packaged_spectre_linear_system,
)


def _fixture_normal_field(volume_map, fixture) -> SpectreBoundaryNormalField:
    d_mg = np.asarray(fixture.system.d_mg)
    ivns = np.zeros((volume_map.mode_count,), dtype=np.float64)
    ibns = np.zeros((volume_map.mode_count,), dtype=np.float64)
    ivnc = np.zeros((volume_map.mode_count,), dtype=np.float64)
    ibnc = np.zeros((volume_map.mode_count,), dtype=np.float64)

    for mode_index, one_based_id in enumerate(volume_map.lme):
        if one_based_id > 0:
            ivns[mode_index] = -d_mg[one_based_id - 1]
    for mode_index, one_based_id in enumerate(volume_map.lmf):
        if one_based_id > 0:
            ivnc[mode_index] = -d_mg[one_based_id - 1]

    return SpectreBoundaryNormalField(
        ivns=ivns,
        ibns=ibns,
        ivnc=ivnc,
        ibnc=ibnc,
    )


def test_matrix_bg_from_fixed_boundary_input_matches_spectre_fixtures() -> None:
    for case in load_all_packaged_spectre_cases():
        if case.input_summary.is_free_boundary:
            continue

        field = build_spectre_boundary_normal_field(case.input_summary)
        assert np.count_nonzero(np.asarray(field.ivns)) == 0
        assert np.count_nonzero(np.asarray(field.ibns)) == 0

        dof_layout = build_spectre_dof_layout(case.input_summary)
        for volume_map in dof_layout.volume_maps:
            assembled = assemble_spectre_matrix_bg(volume_map, field)
            fixture = load_packaged_spectre_linear_system(
                case_label=case.label,
                volume_index=volume_map.block.index + 1,
            )
            np.testing.assert_allclose(np.asarray(assembled.d_mb), np.asarray(fixture.system.d_mb), atol=0.0, rtol=0.0)
            np.testing.assert_allclose(np.asarray(assembled.d_mg), np.asarray(fixture.system.d_mg), atol=0.0, rtol=0.0)


def test_matrix_bg_from_input_builds_free_boundary_initial_source() -> None:
    case = load_packaged_spectre_case("G3V8L3Free")
    summary = case.input_summary
    field = build_spectre_boundary_normal_field(summary)

    expected_ivns = np.asarray([0.0, -0.03794877659624626, -0.03922424672833572, 0.009308639007382408, -0.004028451409448938])
    expected_ibns = np.asarray([0.0, 0.0321685156629137, -0.00571503189066238, -0.0005739913236142605, 0.0008204157785357483])
    np.testing.assert_allclose(np.asarray(field.ivns), expected_ivns, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(field.ibns), expected_ibns, rtol=0.0, atol=0.0)

    assembled = assemble_spectre_matrix_bg_from_input(summary, lvol=9)
    fixture = load_packaged_spectre_linear_system("G3V8L3Free/lvol9")
    np.testing.assert_allclose(np.asarray(assembled.d_mb), np.asarray(fixture.system.d_mb), atol=0.0, rtol=0.0)

    # The packaged free-boundary linear fixture stores SPECTRE's post-Picard
    # normal field.  TOML-only assembly intentionally gives the initial source.
    assert not np.allclose(np.asarray(assembled.d_mg), np.asarray(fixture.system.d_mg))
    dof_layout = build_spectre_dof_layout(summary)
    lme = dof_layout.volume_maps[8].lme
    for mode_index, one_based_id in enumerate(lme):
        if one_based_id > 0:
            assert assembled.d_mg[one_based_id - 1] == -(field.ivns[mode_index] + field.ibns[mode_index])


def test_matrix_bg_accepts_updated_normal_field_for_exact_fixture_parity() -> None:
    for case in load_all_packaged_spectre_cases():
        dof_layout = build_spectre_dof_layout(case.input_summary)
        for volume_map in dof_layout.volume_maps:
            fixture = load_packaged_spectre_linear_system(
                case_label=case.label,
                volume_index=volume_map.block.index + 1,
            )
            field = _fixture_normal_field(volume_map, fixture)
            assembled = assemble_spectre_matrix_bg(volume_map, field)
            np.testing.assert_allclose(np.asarray(assembled.d_mb), np.asarray(fixture.system.d_mb), atol=0.0, rtol=0.0)
            np.testing.assert_allclose(np.asarray(assembled.d_mg), np.asarray(fixture.system.d_mg), atol=0.0, rtol=0.0)


def test_matrix_bg_non_stellarator_symmetric_branches_and_lbnszero() -> None:
    summary = SpectreInputSummary(
        source="synthetic_matrix_bg",
        physics={
            "igeometry": 1,
            "nfp": 1,
            "nvol": 1,
            "mpol": 1,
            "ntor": 1,
            "lrad": [2],
            "enforce_stell_sym": False,
            "lbdybnzero": False,
            "tflux": [0.1],
            "pflux": [0.0],
            "mu": [0.2],
            "helicity": [0.0],
            "vns": {"(0, 1)": 2.0, "(0, -1)": 0.5, "(1, 0)": 3.0, "(-1, 0)": 1.2},
            "bns": {"(0, 1)": 5.0, "(0, -1)": 1.0, "(1, 0)": -0.25},
            "vnc": {"(0, 1)": 7.0, "(0, -1)": 3.0, "(1, 0)": 0.75},
            "bnc": {"(0, 1)": 11.0, "(0, -1)": 13.0, "(1, 0)": -2.0},
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
    dof_layout = build_spectre_dof_layout(summary)
    volume_map = dof_layout.volume_maps[0]
    field = build_spectre_boundary_normal_field(summary)
    assembled = assemble_spectre_matrix_bg(volume_map, field)

    mode_01 = 1
    lme_id = volume_map.lme[mode_01] - 1
    lmf_id = volume_map.lmf[mode_01] - 1
    assert lme_id >= 0
    assert lmf_id >= 0
    assert assembled.d_mg[lme_id] == -((2.0 - 0.5) + (5.0 - 1.0))
    assert assembled.d_mg[lmf_id] == -((7.0 + 3.0) + (11.0 + 13.0))

    lbnszero_summary = SpectreInputSummary(
        source="synthetic_matrix_bg_lbnszero",
        physics={**summary.physics, "lbnszero": True},
        numeric={},
        global_options={},
        local={},
        diagnostics={},
        rbc={},
        zbs={},
        rbs={},
        zbc={},
    )
    lbnszero_field = build_spectre_boundary_normal_field(lbnszero_summary)
    np.testing.assert_allclose(np.asarray(lbnszero_field.ibns), np.zeros((volume_map.mode_count,)))
    np.testing.assert_allclose(np.asarray(lbnszero_field.ibnc), np.zeros((volume_map.mode_count,)))


def test_matrix_bg_rejects_invalid_volume_and_normal_field_shape() -> None:
    case = load_packaged_spectre_case("G2V32L1Fi")
    with np.testing.assert_raises(ValueError):
        assemble_spectre_matrix_bg_from_input(case.input_summary, lvol=99)

    dof_layout = build_spectre_dof_layout(case.input_summary)
    bad_field = SpectreBoundaryNormalField(
        ivns=np.zeros((1,)),
        ibns=np.zeros((1,)),
        ivnc=np.zeros((1,)),
        ibnc=np.zeros((1,)),
    )
    with np.testing.assert_raises(ValueError):
        assemble_spectre_matrix_bg(dof_layout.volume_maps[0], bad_field)
