from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from beltrami_jax import (
    build_spectre_interface_geometry,
    evaluate_spectre_volume_coordinates,
    interpolate_spectre_volume_geometry,
    load_packaged_spectre_case,
)


def test_spectre_interface_geometry_uses_allrzrz_and_free_boundary_wall() -> None:
    case = load_packaged_spectre_case("G3V8L3Free")
    geometry = build_spectre_interface_geometry(case.input_summary)

    assert geometry.interface_count == case.input_summary.packed_volume_count
    assert geometry.mode_count == case.vector_potential_shape[1]
    assert geometry.igeometry == 3

    outer_mode_zero = case.input_summary.rbc[(0, 0)]
    wall_mode_zero = float(case.input_summary.physics["rwc"]["(0, 0)"])
    assert np.isclose(float(geometry.rbc[-2, 0]), outer_mode_zero)
    assert np.isclose(float(geometry.rbc[-1, 0]), wall_mode_zero)


def test_spectre_volume_interpolation_matches_interface_rows() -> None:
    case = load_packaged_spectre_case("G3V8L3Free")
    geometry = build_spectre_interface_geometry(case.input_summary)

    axis = interpolate_spectre_volume_geometry(geometry, lvol=1, s=-1.0)
    first_interface = interpolate_spectre_volume_geometry(geometry, lvol=1, s=1.0)
    exterior_wall = interpolate_spectre_volume_geometry(geometry, lvol=geometry.interface_count, s=1.0)

    np.testing.assert_allclose(np.asarray(axis.rbc), np.asarray(geometry.rbc[0]))
    np.testing.assert_allclose(np.asarray(first_interface.rbc), np.asarray(geometry.rbc[1]))
    np.testing.assert_allclose(np.asarray(first_interface.zbs), np.asarray(geometry.zbs[1]))
    np.testing.assert_allclose(np.asarray(exterior_wall.rbc), np.asarray(geometry.rbc[-1]))
    assert first_interface.coordinate_singularity
    assert not exterior_wall.coordinate_singularity


def test_spectre_coordinates_are_finite_and_metric_is_symmetric() -> None:
    case = load_packaged_spectre_case("G3V8L3Free")
    geometry = build_spectre_interface_geometry(case.input_summary)
    volume = interpolate_spectre_volume_geometry(geometry, lvol=2, s=0.25)
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, 8, endpoint=False)
    zeta = jnp.linspace(0.0, 2.0 * jnp.pi, 5, endpoint=False)

    grid = evaluate_spectre_volume_coordinates(volume, theta=theta, zeta=zeta)

    assert grid.r.shape == (8, 5)
    assert grid.z.shape == (8, 5)
    assert grid.metric.shape == (8, 5, 3, 3)
    assert np.all(np.isfinite(np.asarray(grid.jacobian)))
    assert np.all(np.isfinite(np.asarray(grid.metric)))
    np.testing.assert_allclose(np.asarray(grid.metric), np.swapaxes(np.asarray(grid.metric), -1, -2))


def test_spectre_geometry_is_differentiable_in_radial_coordinate() -> None:
    case = load_packaged_spectre_case("G3V8L3Free")
    geometry = build_spectre_interface_geometry(case.input_summary)
    theta = jnp.asarray([0.1, 1.7])
    zeta = jnp.asarray([0.0])

    def mean_radius(s):
        volume = interpolate_spectre_volume_geometry(geometry, lvol=2, s=s)
        return jnp.mean(evaluate_spectre_volume_coordinates(volume, theta=theta, zeta=zeta).r)

    derivative = jax.grad(mean_radius)(0.25)
    assert np.isfinite(float(derivative))


def test_spectre_geometry_rejects_invalid_volume() -> None:
    case = load_packaged_spectre_case("G3V8L3Free")
    geometry = build_spectre_interface_geometry(case.input_summary)

    try:
        interpolate_spectre_volume_geometry(geometry, lvol=0, s=0.0)
    except ValueError as exc:
        assert "outside" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected invalid lvol to raise")
