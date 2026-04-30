from __future__ import annotations

import numpy as np
import pytest

from beltrami_jax import (
    SpectreVectorPotential,
    build_spectre_beltrami_layout,
    build_spectre_beltrami_layout_for_vector_potential,
    load_all_packaged_spectre_cases,
)

pytest.importorskip("h5py")


def test_packaged_spectre_layouts_match_vector_potential_shapes():
    for case in load_all_packaged_spectre_cases():
        layout = case.layout
        assert layout.shape == case.reference.vector_potential.shape
        assert layout.radial_size == case.input_summary.radial_size
        assert len(layout.blocks) == case.input_summary.packed_volume_count
        assert len(layout.plasma_blocks) == case.input_summary.nvol

        pieces = layout.split_vector_potential(case.reference.vector_potential)
        assert len(pieces) == len(layout.blocks)
        for block, piece in zip(layout.blocks, pieces, strict=True):
            assert piece.radial_size == block.width
            np.testing.assert_allclose(piece.ate, case.reference.vector_potential.ate[block.radial_slice])

        exterior = layout.exterior_block
        if case.input_summary.is_free_boundary:
            assert exterior is not None
            assert exterior.label == "exterior"
        else:
            assert exterior is None


def test_spectre_layout_rejects_shape_mismatch():
    case = load_all_packaged_spectre_cases()[0]
    layout = build_spectre_beltrami_layout(case.input_summary, mode_count=case.vector_potential_shape[1])
    bad = SpectreVectorPotential(
        ate=np.zeros((layout.radial_size + 1, layout.mode_count)),
        aze=np.zeros((layout.radial_size + 1, layout.mode_count)),
        ato=np.zeros((layout.radial_size + 1, layout.mode_count)),
        azo=np.zeros((layout.radial_size + 1, layout.mode_count)),
    )
    with pytest.raises(ValueError, match="does not match SPECTRE layout"):
        layout.validate_vector_potential(bad)


def test_spectre_layout_rejects_invalid_mode_count():
    case = load_all_packaged_spectre_cases()[0]
    with pytest.raises(ValueError, match="mode_count must be positive"):
        build_spectre_beltrami_layout(case.input_summary, mode_count=0)


def test_spectre_layout_summary_is_json_ready():
    case = load_all_packaged_spectre_cases()[0]
    layout = build_spectre_beltrami_layout_for_vector_potential(
        case.input_summary,
        case.reference.vector_potential,
    )
    summary = layout.as_dict()
    assert summary["shape"] == list(case.vector_potential_shape)
    assert summary["blocks"][0]["label"] == "volume_1"
