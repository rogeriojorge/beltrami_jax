"""JAX-native SPECTRE Beltrami field diagnostics.

The routines here port the algebraic diagnostics that sit immediately after the
linear Beltrami solve in SPECTRE.  They intentionally operate on the same
``Ate/Aze/Ato/Azo`` coefficient blocks used by SPECTRE so they can be validated
independently of the nonlinear force loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .spectre_geometry import (
    SpectreInterfaceGeometry,
    build_spectre_interface_geometry,
    evaluate_spectre_volume_coordinates,
    interpolate_spectre_volume_geometry,
)
from .spectre_input import SpectreInputSummary
from .spectre_io import SpectreVectorPotential
from .spectre_pack import SpectreVolumeDofMap, build_spectre_dof_layout
from .spectre_radial import chebyshev_basis, spectre_default_angular_grid, zernike_basis

_PI2 = 2.0 * np.pi


@dataclass(frozen=True)
class SpectrePlasmaCurrentDiagnostic:
    """SPECTRE plasma/linking current diagnostic on one volume interface.

    ``currents`` stores ``[toroidal_plasma_current, poloidal_linking_current]``.
    ``derivative_currents[:, 0]`` and ``derivative_currents[:, 1]`` match the
    first and second derivative solution vectors returned by
    :func:`beltrami_jax.solve_spectre_assembled`.
    """

    lvol: int
    innout: int
    currents: Array
    derivative_currents: Array

    @property
    def toroidal_current(self) -> Array:
        """SPECTRE ``It``/toroidal plasma current."""

        return self.currents[0]

    @property
    def poloidal_current(self) -> Array:
        """SPECTRE ``Gp``/poloidal linking current."""

        return self.currents[1]


def _endpoint_basis(
    *,
    lrad: int,
    mpol: int,
    coordinate_singularity: bool,
    innout: int,
    derivative: bool,
) -> Array:
    if innout not in (0, 1):
        raise ValueError(f"innout must be 0 or 1, got {innout}")
    if coordinate_singularity:
        radial_coordinate = float(innout)
        basis = zernike_basis(radial_coordinate, lrad=lrad, mpol=mpol)
        values = basis[:, :, 1 if derivative else 0]
        return 0.5 * values if derivative else values
    local_s = -1.0 if innout == 0 else 1.0
    values_1d = chebyshev_basis(local_s, lrad)[:, 1 if derivative else 0]
    return jnp.repeat(values_1d[:, None], mpol + 1, axis=1)


def _mode_endpoint_coefficients(
    component: ArrayLike,
    basis: Array,
    poloidal_modes: Array,
) -> Array:
    component_j = jnp.asarray(component, dtype=jnp.float64)
    weights = basis[:, poloidal_modes.astype(jnp.int32)]
    return jnp.sum(component_j * weights, axis=0)


def _evaluate_series(
    even_cos: Array,
    odd_sin: Array,
    *,
    poloidal_modes: Array,
    toroidal_modes: Array,
    theta: Array,
    zeta: Array,
) -> Array:
    phase = poloidal_modes[:, None, None] * theta[None, :, None] - toroidal_modes[:, None, None] * zeta[None, None, :]
    return jnp.sum(even_cos[:, None, None] * jnp.cos(phase) + odd_sin[:, None, None] * jnp.sin(phase), axis=0)


def _boundary_fields(
    vector_potential: SpectreVectorPotential,
    *,
    volume_map: SpectreVolumeDofMap,
    theta: Array,
    zeta: Array,
    innout: int,
    include_radial_field: bool,
) -> tuple[Array, Array, Array]:
    mpol = int(np.max(volume_map.poloidal_modes)) if volume_map.poloidal_modes.size else 0
    poloidal_modes = jnp.asarray(volume_map.poloidal_modes, dtype=jnp.int32)
    toroidal_modes = jnp.asarray(volume_map.toroidal_modes, dtype=jnp.int32)
    derivative_basis = _endpoint_basis(
        lrad=volume_map.block.lrad,
        mpol=mpol,
        coordinate_singularity=volume_map.coordinate_singularity,
        innout=innout,
        derivative=True,
    )
    ate_s = _mode_endpoint_coefficients(vector_potential.ate, derivative_basis, poloidal_modes)
    aze_s = _mode_endpoint_coefficients(vector_potential.aze, derivative_basis, poloidal_modes)
    ato_s = _mode_endpoint_coefficients(vector_potential.ato, derivative_basis, poloidal_modes)
    azo_s = _mode_endpoint_coefficients(vector_potential.azo, derivative_basis, poloidal_modes)

    bsupz = _evaluate_series(
        ate_s,
        ato_s,
        poloidal_modes=poloidal_modes,
        toroidal_modes=toroidal_modes,
        theta=theta,
        zeta=zeta,
    )
    bsupt = _evaluate_series(
        aze_s,
        azo_s,
        poloidal_modes=poloidal_modes,
        toroidal_modes=toroidal_modes,
        theta=theta,
        zeta=zeta,
    )

    if not include_radial_field:
        return jnp.zeros_like(bsupz), bsupt, bsupz

    value_basis = _endpoint_basis(
        lrad=volume_map.block.lrad,
        mpol=mpol,
        coordinate_singularity=volume_map.coordinate_singularity,
        innout=innout,
        derivative=False,
    )
    ate = _mode_endpoint_coefficients(vector_potential.ate, value_basis, poloidal_modes)
    aze = _mode_endpoint_coefficients(vector_potential.aze, value_basis, poloidal_modes)
    ato = _mode_endpoint_coefficients(vector_potential.ato, value_basis, poloidal_modes)
    azo = _mode_endpoint_coefficients(vector_potential.azo, value_basis, poloidal_modes)

    bsups_cos = toroidal_modes.astype(jnp.float64) * ato + poloidal_modes.astype(jnp.float64) * azo
    bsups_sin = -poloidal_modes.astype(jnp.float64) * aze - toroidal_modes.astype(jnp.float64) * ate
    bsups = _evaluate_series(
        bsups_cos,
        bsups_sin,
        poloidal_modes=poloidal_modes,
        toroidal_modes=toroidal_modes,
        theta=theta,
        zeta=zeta,
    )
    return bsups, bsupt, bsupz


def _plasma_current_from_vector_potential(
    vector_potential: SpectreVectorPotential,
    *,
    summary: SpectreInputSummary,
    geometry: SpectreInterfaceGeometry,
    volume_map: SpectreVolumeDofMap,
    lvol: int,
    innout: int,
    theta: Array,
    zeta: Array,
    include_radial_field: bool,
) -> Array:
    volume = interpolate_spectre_volume_geometry(geometry, lvol=lvol, s=2.0 * float(innout) - 1.0)
    grid = evaluate_spectre_volume_coordinates(volume, theta=theta, zeta=zeta)
    bsups, bsupt, bsupz = _boundary_fields(
        vector_potential,
        volume_map=volume_map,
        theta=theta,
        zeta=zeta,
        innout=innout,
        include_radial_field=include_radial_field,
    )
    metric = grid.metric
    inverse_jacobian = grid.inverse_jacobian
    current_toroidal_integrand = (-bsupt * metric[..., 1, 1] + bsupz * metric[..., 1, 2]) * inverse_jacobian
    current_poloidal_integrand = (-bsupt * metric[..., 1, 2] + bsupz * metric[..., 2, 2]) * inverse_jacobian
    if include_radial_field:
        current_toroidal_integrand = current_toroidal_integrand + bsups * metric[..., 1, 0] * inverse_jacobian
        current_poloidal_integrand = current_poloidal_integrand + bsups * metric[..., 0, 2] * inverse_jacobian
    return _PI2 * jnp.asarray(
        (
            jnp.mean(current_toroidal_integrand),
            jnp.mean(current_poloidal_integrand),
        ),
        dtype=jnp.float64,
    )


def compute_spectre_plasma_current(
    summary: SpectreInputSummary,
    *,
    lvol: int,
    vector_potential: SpectreVectorPotential,
    derivative_vector_potentials: tuple[SpectreVectorPotential, ...] = (),
    volume_map: SpectreVolumeDofMap | None = None,
    geometry: SpectreInterfaceGeometry | None = None,
    innout: int = 0,
    nt: int | None = None,
    nz: int | None = None,
    include_radial_field: bool = False,
) -> SpectrePlasmaCurrentDiagnostic:
    """Compute SPECTRE ``compute_plasma_current`` currents on one interface.

    The default branch matches the common SPECTRE call with ``innout=0`` and no
    boundary-normal radial field.  Set ``innout=1`` and
    ``include_radial_field=True`` for the ``Lconstraint=-2`` branch where SPECTRE
    evaluates the outer face and includes ``B^s`` unless ``Lbdybnzero`` is set.
    """

    if lvol < 1 or lvol > summary.packed_volume_count:
        raise ValueError(f"lvol={lvol} outside 1..{summary.packed_volume_count}")
    if volume_map is None:
        volume_map = build_spectre_dof_layout(summary).volume_maps[lvol - 1]
    if geometry is None:
        geometry = build_spectre_interface_geometry(summary)
    if nt is None or nz is None:
        default_nt, default_nz = spectre_default_angular_grid(summary)
        nt = default_nt if nt is None else nt
        nz = default_nz if nz is None else nz
    theta = jnp.arange(int(nt), dtype=jnp.float64) * (_PI2 / int(nt))
    zeta = jnp.arange(int(nz), dtype=jnp.float64) * (_PI2 / (summary.nfp * int(nz)))

    currents = _plasma_current_from_vector_potential(
        vector_potential,
        summary=summary,
        geometry=geometry,
        volume_map=volume_map,
        lvol=lvol,
        innout=innout,
        theta=theta,
        zeta=zeta,
        include_radial_field=include_radial_field,
    )
    derivative_currents = jnp.stack(
        [
            _plasma_current_from_vector_potential(
                derivative_vector_potential,
                summary=summary,
                geometry=geometry,
                volume_map=volume_map,
                lvol=lvol,
                innout=innout,
                theta=theta,
                zeta=zeta,
                include_radial_field=include_radial_field,
            )
            for derivative_vector_potential in derivative_vector_potentials
        ],
        axis=1,
    ) if derivative_vector_potentials else jnp.zeros((2, 0), dtype=jnp.float64)
    return SpectrePlasmaCurrentDiagnostic(
        lvol=lvol,
        innout=innout,
        currents=currents,
        derivative_currents=derivative_currents,
    )
