from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
from jax import Array

from .types import (
    BeltramiLinearSystem,
    FourierBeltramiGeometry,
    FourierModeBasis,
    GeometryAssemblyResult,
)


def build_fourier_mode_basis(
    *,
    max_radial_order: int,
    max_poloidal_mode: int,
    max_toroidal_mode: int,
    include_sine: bool = True,
    label: str = "",
) -> FourierModeBasis:
    """Build a compact cosine/sine Fourier basis for internal assembly."""
    if max_radial_order < 0 or max_poloidal_mode < 0 or max_toroidal_mode < 0:
        raise ValueError("mode limits must be non-negative")

    radial_orders: list[int] = []
    poloidal_modes: list[int] = []
    toroidal_modes: list[int] = []
    families: list[int] = []

    for radial_order in range(max_radial_order + 1):
        for m in range(max_poloidal_mode + 1):
            for n in range(-max_toroidal_mode, max_toroidal_mode + 1):
                radial_orders.append(radial_order)
                poloidal_modes.append(m)
                toroidal_modes.append(n)
                families.append(0)

                if include_sine and (m != 0 or n != 0):
                    radial_orders.append(radial_order)
                    poloidal_modes.append(m)
                    toroidal_modes.append(n)
                    families.append(1)

    return FourierModeBasis.from_arraylike(
        radial_orders=radial_orders,
        poloidal_modes=poloidal_modes,
        toroidal_modes=toroidal_modes,
        families=families,
        label=label or f"r{max_radial_order}_m{max_poloidal_mode}_n{max_toroidal_mode}",
    )


def collocation_grid(geometry: FourierBeltramiGeometry) -> tuple[Array, Array, Array]:
    """Return the radial, poloidal, and toroidal collocation coordinates."""
    radial = jnp.linspace(0.0, 1.0, geometry.radial_points, dtype=jnp.float64)
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, geometry.poloidal_points, endpoint=False, dtype=jnp.float64)
    zeta = jnp.linspace(
        0.0,
        2.0 * jnp.pi / geometry.field_periods,
        geometry.toroidal_points,
        endpoint=False,
        dtype=jnp.float64,
    )
    return radial, theta, zeta


def torus_coordinates(
    geometry: FourierBeltramiGeometry,
    radial: Array,
    theta: Array,
    zeta: Array,
) -> tuple[Array, Array, Array]:
    """Map collocation coordinates to a simple shaped torus."""
    radial_grid, theta_grid, zeta_grid = jnp.meshgrid(radial, theta, zeta, indexing="ij")
    radial_shape = radial_grid * geometry.minor_radius
    shaping = jnp.cos(theta_grid) + geometry.triangularity * jnp.sin(theta_grid) ** 2
    major = geometry.major_radius + radial_shape * shaping
    vertical = radial_shape * geometry.elongation * jnp.sin(theta_grid)
    return major, vertical, zeta_grid


def basis_values(
    basis: FourierModeBasis,
    geometry: FourierBeltramiGeometry,
    radial: Array,
    theta: Array,
    zeta: Array,
) -> tuple[Array, Array, Array, Array]:
    """Evaluate basis functions and derivatives on the collocation grid."""
    radial_grid, theta_grid, zeta_grid = jnp.meshgrid(radial, theta, zeta, indexing="ij")
    radial_regularized = jnp.sqrt(radial_grid**2 + geometry.axis_regularization**2)

    radial_orders = basis.radial_orders[:, None, None, None]
    poloidal_modes = basis.poloidal_modes[:, None, None, None]
    toroidal_modes = basis.toroidal_modes[:, None, None, None]
    families = basis.families[:, None, None, None]

    radial_power = radial_orders + poloidal_modes
    phase = poloidal_modes * theta_grid[None, :, :, :] - toroidal_modes * geometry.field_periods * zeta_grid[None, :, :, :]
    trig_primary = jnp.where(families == 0, jnp.cos(phase), jnp.sin(phase))
    trig_dual = jnp.where(families == 0, -jnp.sin(phase), jnp.cos(phase))

    radial_factor = radial_regularized[None, :, :, :] ** radial_power
    values = radial_factor * trig_primary

    safe_radial = jnp.where(radial_power == 0, 1.0, radial_regularized[None, :, :, :])
    radial_derivative = jnp.where(
        radial_power == 0,
        0.0,
        radial_power * radial_factor * radial_grid[None, :, :, :] / (safe_radial**2),
    )
    dtheta = poloidal_modes * radial_factor * trig_dual
    dzeta = -toroidal_modes * geometry.field_periods * radial_factor * trig_dual
    return values, radial_derivative, dtheta, dzeta


def assemble_fourier_beltrami_system(
    geometry: FourierBeltramiGeometry,
    basis: FourierModeBasis,
    *,
    mu: float,
    psi: tuple[float, float] | Array,
    is_vacuum: bool = False,
    vacuum_strength: float = 0.0,
    label: str = "",
) -> GeometryAssemblyResult:
    """Assemble a SPEC-style Beltrami system from an internal Fourier geometry."""
    radial, theta, zeta = collocation_grid(geometry)
    major_radius, _, _ = torus_coordinates(geometry, radial, theta, zeta)
    values, dradial, dtheta, dzeta = basis_values(basis, geometry, radial, theta, zeta)

    radial_grid, _, zeta_grid = jnp.meshgrid(radial, theta, zeta, indexing="ij")
    radial_regularized = jnp.sqrt(radial_grid**2 + geometry.axis_regularization**2)

    dr = 1.0 / max(geometry.radial_points - 1, 1)
    dtheta_value = 2.0 * jnp.pi / geometry.poloidal_points
    dzeta_value = (2.0 * jnp.pi / geometry.field_periods) / geometry.toroidal_points
    jacobian = (
        geometry.minor_radius**2
        * geometry.elongation
        * radial_regularized
        * jnp.maximum(major_radius, geometry.axis_regularization)
    )
    weights = jacobian * dr * dtheta_value * dzeta_value

    g_rr = 1.0 / geometry.minor_radius**2
    g_tt = 1.0 / (geometry.minor_radius**2 * radial_regularized**2)
    g_zz = 1.0 / jnp.maximum(major_radius**2, geometry.axis_regularization**2)
    integrand = (
        g_rr * dradial[:, None, :, :, :] * dradial[None, :, :, :, :]
        + g_tt[None, :, :, :] * dtheta[:, None, :, :, :] * dtheta[None, :, :, :, :]
        + g_zz[None, :, :, :] * dzeta[:, None, :, :, :] * dzeta[None, :, :, :, :]
        + geometry.mass_shift * values[:, None, :, :, :] * values[None, :, :, :, :]
    )

    stabilization = geometry.axis_regularization * jnp.eye(basis.size, dtype=jnp.float64)
    d_ma = jnp.einsum("rtn,ijrtn->ij", weights, integrand) + stabilization
    d_md = jnp.einsum("rtn,irtn,jrtn->ij", weights, values, values) + 0.1 * stabilization

    flux_modes = jnp.stack(
        [
            jnp.ones_like(radial_grid),
            radial_grid * (1.0 + 0.25 * jnp.cos(geometry.field_periods * zeta_grid)),
        ],
        axis=0,
    )
    d_mb = jnp.einsum("rtn,frtn,irtn->if", weights, flux_modes, values)

    d_mg = None
    if is_vacuum:
        vacuum_pattern = vacuum_strength * (1.0 + radial_grid) * jnp.cos(theta[None, :, None] - geometry.field_periods * zeta_grid)
        d_mg = jnp.einsum("rtn,irtn,rtn->i", weights, values, vacuum_pattern)

    system = BeltramiLinearSystem.from_arraylike(
        d_ma=d_ma,
        d_md=d_md,
        d_mb=d_mb,
        d_mg=d_mg,
        mu=mu,
        psi=psi,
        is_vacuum=is_vacuum,
        label=label or geometry.label or basis.label or "fourier_geometry_system",
    )
    return GeometryAssemblyResult(
        geometry=geometry,
        basis=basis,
        system=system,
        radial_grid=radial,
        theta_grid=theta,
        zeta_grid=zeta,
    )


def shift_mu(assembly: GeometryAssemblyResult, mu: float, *, label: str | None = None) -> GeometryAssemblyResult:
    """Reuse assembled matrices while changing the Beltrami multiplier."""
    return replace(
        assembly,
        system=BeltramiLinearSystem.from_arraylike(
            d_ma=assembly.system.d_ma,
            d_md=assembly.system.d_md,
            d_mb=assembly.system.d_mb,
            d_mg=assembly.system.d_mg,
            mu=mu,
            psi=assembly.system.psi,
            is_vacuum=assembly.system.is_vacuum,
            label=label or assembly.system.label,
        ),
    )
