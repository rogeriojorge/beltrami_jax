from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from beltrami_jax import (
    SpectreInputSummary,
    build_spectre_dof_layout,
    build_spectre_dof_layout_for_vector_potential,
    compare_vector_potentials,
    load_all_packaged_spectre_cases,
    spectre_fourier_modes,
    spectre_mode_count,
)

pytest.importorskip("h5py")


def test_spectre_mode_order_matches_gi00ab():
    case = load_all_packaged_spectre_cases()[0]
    summary = case.input_summary
    modes = spectre_fourier_modes(summary)
    assert spectre_mode_count(summary) == case.vector_potential_shape[1]
    assert modes == ((0, 0), (0, 1), (1, -1), (1, 0), (1, 1), (2, -1), (2, 0), (2, 1))


def test_packaged_spectre_vector_potentials_round_trip_through_dof_maps():
    for case in load_all_packaged_spectre_cases():
        dof_layout = build_spectre_dof_layout_for_vector_potential(
            case.input_summary,
            case.reference.vector_potential,
        )
        assert dof_layout.mode_count == case.vector_potential_shape[1]
        assert len(dof_layout.volume_maps) == case.input_summary.packed_volume_count
        assert sum(block.width for block in case.layout.blocks) == case.vector_potential_shape[0]

        for volume_map in dof_layout.volume_maps:
            volume_map.validate_contiguous_ids()
            assert volume_map.solution_size == (
                volume_map.coefficient_dof_count + volume_map.lagrange_multiplier_count
            )
            if case.input_summary.enforce_stellarator_symmetry:
                assert not np.any(volume_map.ato)
                assert not np.any(volume_map.azo)

        solutions = dof_layout.pack_vector_potential(case.reference.vector_potential)
        reconstructed = dof_layout.unpack_solutions(solutions, source=case.label)
        comparison = compare_vector_potentials(
            reconstructed,
            case.reference.vector_potential,
            label=f"{case.label} pack/unpack",
        )
        assert comparison.global_max_abs_error == 0.0
        assert comparison.global_relative_error == 0.0


def test_coordinate_singularity_axis_recombination_matches_spectre_rules():
    case = load_all_packaged_spectre_cases()[1]
    dof_layout = build_spectre_dof_layout_for_vector_potential(
        case.input_summary,
        case.reference.vector_potential,
    )
    first_volume = dof_layout.volume_maps[0]
    second_volume = dof_layout.volume_maps[1]

    assert first_volume.coordinate_singularity
    assert not second_volume.coordinate_singularity

    # SPECTRE's Zernike axis recombination removes Ate for (m,ll)=(0,0)
    # and (m,ll)=(1,1), while Aze still receives a solution-vector id.
    assert first_volume.ate[0, 0] == 0
    assert first_volume.aze[0, 0] > 0
    mode_m1 = int(np.flatnonzero(first_volume.poloidal_modes == 1)[0])
    assert first_volume.ate[1, mode_m1] == 0
    assert first_volume.aze[1, mode_m1] > 0

    # Non-singular regions use Chebyshev-like radial rows and remove ll=0.
    assert not np.any(second_volume.ate[0, :])
    assert not np.any(second_volume.aze[0, :])
    assert np.any(second_volume.ate[1:, :])
    assert np.any(second_volume.aze[1:, :])


def test_non_stellarator_symmetric_maps_include_odd_components():
    summary = SpectreInputSummary(
        source="synthetic_non_stellsym",
        physics={
            "igeometry": 3,
            "nfp": 1,
            "nvol": 1,
            "mpol": 1,
            "ntor": 1,
            "lrad": [3],
            "enforce_stell_sym": False,
            "tflux": [0.1],
            "pflux": [0.0],
            "mu": [0.2],
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
    dof_layout = build_spectre_dof_layout(summary)
    volume_map = dof_layout.volume_maps[0]

    assert not volume_map.enforce_stellarator_symmetry
    assert volume_map.coordinate_singularity
    assert not np.any(volume_map.ato[:, 0])
    assert not np.any(volume_map.azo[:, 0])
    assert np.any(volume_map.ato[:, 1:])
    assert np.any(volume_map.azo[:, 1:])
    assert np.any(volume_map.lmf[1:])


def test_spectre_dof_layout_jax_pack_unpack_is_differentiable():
    case = load_all_packaged_spectre_cases()[0]
    dof_layout = build_spectre_dof_layout_for_vector_potential(
        case.input_summary,
        case.reference.vector_potential,
    )
    components = {
        name: jnp.asarray(array)
        for name, array in case.reference.vector_potential.components().items()
    }

    packed = dof_layout.pack_vector_potential_jax(components)
    unpacked = dof_layout.unpack_solutions_jax(packed)
    for name, reference in components.items():
        np.testing.assert_allclose(np.asarray(unpacked[name]), np.asarray(reference), atol=0.0, rtol=0.0)

    def objective(scale):
        scaled = {name: scale * array for name, array in components.items()}
        solutions = dof_layout.pack_vector_potential_jax(scaled)
        return sum(jnp.sum(solution**2) for solution in solutions)

    gradient = jax.grad(objective)(1.0)
    assert np.isfinite(float(gradient))
    assert float(gradient) > 0.0


def test_spectre_dof_layout_rejects_wrong_mode_count():
    case = load_all_packaged_spectre_cases()[0]
    with pytest.raises(ValueError, match="does not match SPECTRE mode count"):
        build_spectre_dof_layout(case.input_summary, mode_count=case.vector_potential_shape[1] + 1)
