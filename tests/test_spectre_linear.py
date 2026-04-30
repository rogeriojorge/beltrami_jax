from __future__ import annotations

import numpy as np
import pytest

from beltrami_jax import (
    DEFAULT_PACKAGED_SPECTRE_LINEAR_SYSTEM,
    assemble_operator,
    assemble_rhs,
    compare_against_reference,
    list_packaged_spectre_linear_cases,
    list_packaged_spectre_linear_systems,
    load_all_packaged_spectre_linear_systems,
    load_packaged_spectre_linear_system,
    solve_from_components,
)


EXPECTED_LINEAR_CASE_COUNTS = {
    "G2V32L1Fi": 4,
    "G3V3L2Fi_stability": 3,
    "G3V3L3Fi": 3,
    "G3V8L3Free": 9,
}


def test_packaged_spectre_linear_listing_is_complete() -> None:
    assert list_packaged_spectre_linear_cases() == tuple(sorted(EXPECTED_LINEAR_CASE_COUNTS))
    systems = list_packaged_spectre_linear_systems()
    assert len(systems) == sum(EXPECTED_LINEAR_CASE_COUNTS.values())
    assert DEFAULT_PACKAGED_SPECTRE_LINEAR_SYSTEM in systems


def test_packaged_spectre_linear_case_filter() -> None:
    systems = list_packaged_spectre_linear_systems("G2V32L1Fi")
    assert systems == ("G2V32L1Fi/lvol1", "G2V32L1Fi/lvol2", "G2V32L1Fi/lvol3", "G2V32L1Fi/lvol4")
    loaded = load_all_packaged_spectre_linear_systems("G2V32L1Fi")
    assert tuple(item.name for item in loaded) == systems


def test_packaged_spectre_linear_loader_metadata() -> None:
    fixture = load_packaged_spectre_linear_system("G3V8L3Free/lvol9")
    assert fixture.case_label == "G3V8L3Free"
    assert fixture.volume_index == 9
    assert fixture.nvol == 8
    assert fixture.mvol == 9
    assert fixture.lconstraint == 3
    assert fixture.is_vacuum is True
    assert fixture.system.is_vacuum is True
    assert fixture.include_d_mg_in_rhs is True
    assert fixture.system.include_d_mg_in_rhs is True
    assert fixture.system.size == fixture.n_dof
    assert fixture.matrix.shape == (fixture.n_dof, fixture.n_dof)
    assert fixture.rhs.shape == (fixture.n_dof,)
    assert fixture.expected_solution.shape == (fixture.n_dof,)
    assert fixture.relative_residual_norm < 2e-12


def test_packaged_spectre_linear_reassembles_spectre_operator_and_rhs() -> None:
    for fixture in load_all_packaged_spectre_linear_systems():
        np.testing.assert_allclose(
            np.asarray(assemble_operator(fixture.system)),
            np.asarray(fixture.matrix),
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            np.asarray(assemble_rhs(fixture.system)),
            np.asarray(fixture.rhs),
            rtol=0.0,
            atol=0.0,
        )


def test_packaged_spectre_linear_solution_matches_spectre_solve() -> None:
    worst_solution_error = 0.0
    for fixture in load_all_packaged_spectre_linear_systems():
        result = solve_from_components(fixture.system)
        comparison = compare_against_reference(fixture.reference, result)
        worst_solution_error = max(worst_solution_error, comparison.solution_relative_error)
        assert comparison.operator_relative_error == 0.0
        assert comparison.rhs_relative_error == 0.0
        assert comparison.solution_relative_error < 3e-12
        assert comparison.max_abs_solution_error < 5e-11
        assert float(result.relative_residual_norm) < 3e-12
    assert worst_solution_error > 0.0


def test_packaged_spectre_linear_loader_accepts_filename_stem_and_explicit_args() -> None:
    by_stem = load_packaged_spectre_linear_system("G2V32L1Fi_lvol2")
    by_explicit = load_packaged_spectre_linear_system(case_label="G2V32L1Fi", volume_index=2)
    assert by_stem.name == by_explicit.name == "G2V32L1Fi/lvol2"
    np.testing.assert_allclose(np.asarray(by_stem.expected_solution), np.asarray(by_explicit.expected_solution))


def test_packaged_spectre_linear_rejects_unknown_case() -> None:
    with pytest.raises(ValueError, match="unknown packaged SPECTRE linear case"):
        list_packaged_spectre_linear_systems("missing")
    with pytest.raises(ValueError, match="unknown packaged SPECTRE linear system"):
        load_packaged_spectre_linear_system("G2V32L1Fi/lvol999")
    with pytest.raises(ValueError, match="expected volume label"):
        load_packaged_spectre_linear_system("G2V32L1Fi/volume1")
