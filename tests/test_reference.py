from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from beltrami_jax.operators import assemble_operator, assemble_rhs
from beltrami_jax.reference import list_packaged_references, load_packaged_reference, load_spec_text_dump
from beltrami_jax.diagnostics import compare_against_reference
from beltrami_jax.solver import solve_from_components


PACKAGED_FIXTURE_METADATA = {
    "g1v03l0fi_lvol2": {"size": 51, "volume_index": 2, "is_vacuum": False},
    "g3v01l0fi_lvol1": {"size": 361, "volume_index": 1, "is_vacuum": False},
    "g3v02l0fr_lu_lvol3": {"size": 1548, "volume_index": 3, "is_vacuum": True},
    "g3v02l1fi_lvol1": {"size": 361, "volume_index": 1, "is_vacuum": False},
}


def test_packaged_reference_listing_is_complete() -> None:
    assert list_packaged_references() == tuple(sorted(PACKAGED_FIXTURE_METADATA))


def test_packaged_reference_shapes() -> None:
    for name, metadata in PACKAGED_FIXTURE_METADATA.items():
        reference = load_packaged_reference(name)
        assert reference.system.size == metadata["size"]
        assert reference.matrix.shape == (metadata["size"], metadata["size"])
        assert reference.rhs.shape == (metadata["size"],)
        assert reference.expected_solution.shape == (metadata["size"],)
        assert reference.volume_index == metadata["volume_index"]
        assert reference.system.is_vacuum is metadata["is_vacuum"]


def test_packaged_reference_reassembles_spec_operator() -> None:
    for name in PACKAGED_FIXTURE_METADATA:
        reference = load_packaged_reference(name)
        np.testing.assert_allclose(np.asarray(assemble_operator(reference.system)), np.asarray(reference.matrix), rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(np.asarray(assemble_rhs(reference.system)), np.asarray(reference.rhs), rtol=1e-13, atol=1e-13)


def test_load_spec_text_dump_reads_dotted_prefix_with_vacuum_metadata(tmp_path: Path) -> None:
    prefix = tmp_path / "fixture.dump.lvol1"
    (tmp_path / "fixture.dump.lvol1.meta.txt").write_text(
        "\n".join(
            [
                "lvol 3",
                "nn 2",
                "mu 0.0",
                "psi_t 0.5",
                "psi_p -0.25",
                "is_vacuum 1",
            ]
        )
        + "\n"
    )
    np.savetxt(tmp_path / "fixture.dump.lvol1.dma.txt", np.array([[4.0, 1.0], [1.0, 3.0]]))
    np.savetxt(tmp_path / "fixture.dump.lvol1.dmd.txt", np.array([[2.0, 0.0], [0.0, 1.0]]))
    np.savetxt(tmp_path / "fixture.dump.lvol1.dmb.txt", np.array([[1.0, 2.0], [3.0, 4.0]]))
    np.savetxt(tmp_path / "fixture.dump.lvol1.dmg.txt", np.array([0.5, -0.25]))
    np.savetxt(tmp_path / "fixture.dump.lvol1.matrix.txt", np.array([[4.0, 1.0], [1.0, 3.0]]))
    np.savetxt(tmp_path / "fixture.dump.lvol1.rhs.txt", np.array([-0.5, -0.25]))
    np.savetxt(tmp_path / "fixture.dump.lvol1.solution.txt", np.array([0.1, -0.2]))

    reference = load_spec_text_dump(prefix)

    assert reference.volume_index == 3
    assert reference.source == str(prefix)
    assert reference.system.is_vacuum is True
    np.testing.assert_allclose(np.asarray(reference.system.mu), np.array(0.0))
    np.testing.assert_allclose(np.asarray(reference.system.psi), np.array([0.5, -0.25]))
    np.testing.assert_allclose(np.asarray(reference.system.d_mg), np.array([0.5, -0.25]))


def test_load_spec_text_dump_reads_legacy_dotted_prefix_without_vacuum_fields(tmp_path: Path) -> None:
    prefix = tmp_path / "fixture.dump.lvol1"
    (tmp_path / "fixture.dump.lvol1.meta.txt").write_text(
        "\n".join(
            [
                "lvol 3",
                "nn 2",
                "mu 1.25",
                "psi_t 0.5",
                "psi_p -0.25",
            ]
        )
        + "\n"
    )
    np.savetxt(tmp_path / "fixture.dump.lvol1.dma.txt", np.array([[4.0, 1.0], [1.0, 3.0]]))
    np.savetxt(tmp_path / "fixture.dump.lvol1.dmd.txt", np.array([[2.0, 0.0], [0.0, 1.0]]))
    np.savetxt(tmp_path / "fixture.dump.lvol1.dmb.txt", np.array([[1.0, 2.0], [3.0, 4.0]]))
    np.savetxt(tmp_path / "fixture.dump.lvol1.matrix.txt", np.array([[1.5, 1.0], [1.0, 1.75]]))
    np.savetxt(tmp_path / "fixture.dump.lvol1.rhs.txt", np.array([0.0, -0.5]))
    np.savetxt(tmp_path / "fixture.dump.lvol1.solution.txt", np.array([0.3076923076923077, -0.46153846153846156]))

    reference = load_spec_text_dump(prefix)

    assert reference.volume_index == 3
    assert reference.source == str(prefix)
    assert reference.system.is_vacuum is False
    assert reference.system.d_mg is None
    np.testing.assert_allclose(np.asarray(reference.system.mu), np.array(1.25))
    np.testing.assert_allclose(np.asarray(reference.system.psi), np.array([0.5, -0.25]))
    np.testing.assert_allclose(np.asarray(reference.system.d_ma), np.array([[4.0, 1.0], [1.0, 3.0]]))
    np.testing.assert_allclose(np.asarray(reference.system.d_md), np.array([[2.0, 0.0], [0.0, 1.0]]))
    np.testing.assert_allclose(np.asarray(reference.system.d_mb), np.array([[1.0, 2.0], [3.0, 4.0]]))
    np.testing.assert_allclose(np.asarray(assemble_operator(reference.system)), np.asarray(reference.matrix), rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(np.asarray(assemble_rhs(reference.system)), np.asarray(reference.rhs), rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(np.asarray(reference.matrix), np.array([[1.5, 1.0], [1.0, 1.75]]))
    np.testing.assert_allclose(np.asarray(reference.rhs), np.array([0.0, -0.5]))
    np.testing.assert_allclose(np.asarray(reference.expected_solution), np.array([0.3076923076923077, -0.46153846153846156]))


def test_load_spec_text_dump_requires_dmg_for_vacuum_case(tmp_path: Path) -> None:
    prefix = tmp_path / "fixture.dump.lvol2"
    (tmp_path / "fixture.dump.lvol2.meta.txt").write_text(
        "\n".join(
            [
                "lvol 2",
                "nn 2",
                "mu 0.0",
                "psi_t 0.5",
                "psi_p 0.25",
                "is_vacuum 1",
            ]
        )
        + "\n"
    )
    np.savetxt(tmp_path / "fixture.dump.lvol2.dma.txt", np.eye(2))
    np.savetxt(tmp_path / "fixture.dump.lvol2.dmd.txt", np.eye(2))
    np.savetxt(tmp_path / "fixture.dump.lvol2.dmb.txt", np.ones((2, 2)))
    np.savetxt(tmp_path / "fixture.dump.lvol2.matrix.txt", np.eye(2))
    np.savetxt(tmp_path / "fixture.dump.lvol2.rhs.txt", np.zeros(2))
    np.savetxt(tmp_path / "fixture.dump.lvol2.solution.txt", np.zeros(2))

    with pytest.raises(FileNotFoundError, match="dmg.txt"):
        load_spec_text_dump(prefix)


def test_reference_comparison_handles_zero_expected_norm() -> None:
    reference = load_packaged_reference("g1v03l0fi_lvol2")
    result = solve_from_components(reference.system)
    zero_reference = type(reference)(
        system=reference.system,
        matrix=reference.matrix * 0.0,
        rhs=reference.rhs * 0.0,
        expected_solution=reference.expected_solution * 0.0,
        volume_index=reference.volume_index,
        source=reference.source,
    )
    comparison = compare_against_reference(zero_reference, result)
    assert comparison.operator_relative_error > 0.0
    assert comparison.rhs_relative_error > 0.0
    assert comparison.solution_relative_error > 0.0
