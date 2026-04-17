from __future__ import annotations

import numpy as np
from pathlib import Path

from beltrami_jax.operators import assemble_operator, assemble_rhs
from beltrami_jax.reference import load_packaged_reference, load_spec_text_dump


def test_packaged_reference_shapes() -> None:
    reference = load_packaged_reference()
    assert reference.system.size == 361
    assert reference.matrix.shape == (361, 361)
    assert reference.rhs.shape == (361,)
    assert reference.expected_solution.shape == (361,)


def test_packaged_reference_reassembles_spec_operator() -> None:
    reference = load_packaged_reference()
    np.testing.assert_allclose(np.asarray(assemble_operator(reference.system)), np.asarray(reference.matrix), rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(np.asarray(assemble_rhs(reference.system)), np.asarray(reference.rhs), rtol=1e-13, atol=1e-13)


def test_load_spec_text_dump_reads_dotted_prefix(tmp_path: Path) -> None:
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
    np.savetxt(tmp_path / "fixture.dump.lvol1.rhs.txt", np.array([0.0, 1.0]))
    np.savetxt(tmp_path / "fixture.dump.lvol1.solution.txt", np.array([2.0, -1.0]))

    reference = load_spec_text_dump(prefix)

    assert reference.volume_index == 3
    assert reference.source == str(prefix)
    np.testing.assert_allclose(np.asarray(reference.system.mu), np.array(1.25))
    np.testing.assert_allclose(np.asarray(reference.system.psi), np.array([0.5, -0.25]))
    np.testing.assert_allclose(np.asarray(reference.system.d_ma), np.array([[4.0, 1.0], [1.0, 3.0]]))
    np.testing.assert_allclose(np.asarray(reference.system.d_md), np.array([[2.0, 0.0], [0.0, 1.0]]))
    np.testing.assert_allclose(np.asarray(reference.system.d_mb), np.array([[1.0, 2.0], [3.0, 4.0]]))
    np.testing.assert_allclose(np.asarray(reference.matrix), np.array([[1.5, 1.0], [1.0, 1.75]]))
    np.testing.assert_allclose(np.asarray(reference.rhs), np.array([0.0, 1.0]))
    np.testing.assert_allclose(np.asarray(reference.expected_solution), np.array([2.0, -1.0]))
