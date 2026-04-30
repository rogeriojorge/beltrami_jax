"""SPECTRE radial basis and quadrature helpers.

The routines in this module mirror the public numerical pieces used by
SPECTRE's ``base_functions_mod.F90`` and ``preset_mod.F90`` when constructing
Beltrami matrix integrals.  They are small, deterministic, and JAX-compatible,
so higher-level geometry and matrix assembly can be differentiated end to end.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from .spectre_input import SpectreInputSummary

Array = jnp.ndarray


@dataclass(frozen=True)
class SpectreRadialQuadrature:
    """Gauss-Legendre quadrature on SPECTRE's local radial coordinate."""

    abscissae: Array
    weights: Array

    @property
    def size(self) -> int:
        """Number of quadrature points."""

        return int(self.abscissae.shape[0])


@dataclass(frozen=True)
class SpectreRadialBoundaryBasis:
    """Endpoint basis values used by SPECTRE Lagrange-multiplier rows."""

    values: Array
    axis_values: Array


def spectre_default_angular_grid(summary: SpectreInputSummary) -> tuple[int, int]:
    """Return SPECTRE's default ``Nt, Nz`` matrix-integration grid.

    SPECTRE sets ``Nt = max(Ndiscrete * 4 * Mpol, 1)`` and
    ``Nz = max(Ndiscrete * 4 * Ntor, 1)`` during internal-array
    initialization.
    """

    ndiscrete = int(summary.numeric.get("ndiscrete", 2))
    return max(ndiscrete * 4 * summary.mpol, 1), max(ndiscrete * 4 * summary.ntor, 1)


def spectre_default_quadrature_size(summary: SpectreInputSummary, *, lvol: int) -> int:
    """Return SPECTRE's default ``Iquad(lvol)`` value.

    ``lvol`` is one-based.  Positive ``numeric.nquad`` is used directly.
    Non-positive values follow SPECTRE's resolution heuristic, including the
    coordinate-singularity branch for the first non-Cartesian volume.
    """

    if lvol < 1 or lvol > summary.packed_volume_count:
        raise ValueError(f"lvol={lvol} outside 1..{summary.packed_volume_count}")
    nquad = int(summary.numeric.get("nquad", -1))
    if nquad > 0:
        return nquad
    lrad = summary.lrad[lvol - 1]
    coordinate_singularity = summary.igeometry != 1 and lvol == 1
    if coordinate_singularity:
        return summary.mpol + 2 * lrad - nquad
    return 2 * lrad - nquad


def gauss_legendre_quadrature(size: int) -> SpectreRadialQuadrature:
    """Build Gauss-Legendre abscissae and weights on ``[-1, 1]``."""

    if size <= 0:
        raise ValueError(f"quadrature size must be positive, got {size}")
    abscissae, weights = np.polynomial.legendre.leggauss(size)
    return SpectreRadialQuadrature(
        abscissae=jnp.asarray(abscissae, dtype=jnp.float64),
        weights=jnp.asarray(weights, dtype=jnp.float64),
    )


def chebyshev_basis(s: object, lrad: int) -> Array:
    """Evaluate SPECTRE's recombined/scaled Chebyshev basis.

    The returned array has shape ``(lrad + 1, 2)``.  The last axis stores
    ``[T_l, dT_l/ds]`` after SPECTRE's endpoint recombination and ``1/(l+1)``
    scaling.
    """

    if lrad < 1:
        raise ValueError("SPECTRE Chebyshev basis requires lrad >= 1")
    s_j = jnp.asarray(s, dtype=jnp.float64)
    rows = []
    rows.append(jnp.asarray([1.0, 0.0], dtype=jnp.float64))
    rows.append(jnp.asarray([s_j, 1.0], dtype=jnp.float64))
    for ll in range(2, lrad + 1):
        previous = rows[ll - 1]
        before_previous = rows[ll - 2]
        value = 2.0 * s_j * previous[0] - before_previous[0]
        derivative = 2.0 * previous[0] + 2.0 * s_j * previous[1] - before_previous[1]
        rows.append(jnp.asarray([value, derivative], dtype=jnp.float64))

    output = jnp.stack(rows)
    recombination = jnp.asarray([0.0] + [-( -1.0) ** ll for ll in range(1, lrad + 1)], dtype=jnp.float64)
    output = output.at[:, 0].add(recombination)
    scale = jnp.arange(1, lrad + 2, dtype=jnp.float64)[:, None]
    return output / scale


def zernike_basis(r: object, *, lrad: int, mpol: int) -> Array:
    """Evaluate SPECTRE's recombined/scaled Zernike radial basis.

    The returned array has shape ``(lrad + 1, mpol + 1, 2)``.  The last axis
    stores ``[R_n^m, dR_n^m/dr]``.  Matrix assembly multiplies the derivative
    by ``1/2`` separately because SPECTRE's local coordinate is
    ``s = 2r - 1``.
    """

    if lrad < 0 or mpol < 0:
        raise ValueError("lrad and mpol must be non-negative")
    r_j = jnp.asarray(r, dtype=jnp.float64)
    zernike = jnp.zeros((lrad + 1, mpol + 1, 2), dtype=jnp.float64)
    rm = jnp.asarray(1.0, dtype=jnp.float64)
    rm1 = jnp.asarray(0.0, dtype=jnp.float64)

    for m in range(mpol + 1):
        if lrad >= m:
            zernike = zernike.at[m, m, :].set(jnp.asarray([rm, float(m) * rm1], dtype=jnp.float64))
        if lrad >= m + 2:
            value = float(m + 2) * rm * r_j**2 - float(m + 1) * rm
            derivative = float((m + 2) ** 2) * rm * r_j - float((m + 1) * m) * rm1
            zernike = zernike.at[m + 2, m, :].set(jnp.asarray([value, derivative], dtype=jnp.float64))

        for n in range(m + 4, lrad + 1, 2):
            factor1 = float(n) / float(n**2 - m**2)
            factor2 = float(4 * (n - 1))
            factor3 = float((n - 2 + m) ** 2) / float(n - 2) + float((n - m) ** 2) / float(n)
            factor4 = float((n - 2) ** 2 - m**2) / float(n - 2)
            previous = zernike[n - 2, m]
            before_previous = zernike[n - 4, m]
            value = factor1 * ((factor2 * r_j**2 - factor3) * previous[0] - factor4 * before_previous[0])
            derivative = factor1 * (
                2.0 * factor2 * r_j * previous[0]
                + (factor2 * r_j**2 - factor3) * previous[1]
                - factor4 * before_previous[1]
            )
            zernike = zernike.at[n, m, :].set(jnp.asarray([value, derivative], dtype=jnp.float64))

        rm1 = rm
        rm = rm * r_j

    for n in range(2, lrad + 1, 2):
        zernike = zernike.at[n, 0, 0].add(-(-1.0) ** (n // 2))
    if mpol >= 1:
        for n in range(3, lrad + 1, 2):
            coefficient = -(-1.0) ** ((n - 1) // 2) * float((n + 1) // 2)
            zernike = zernike.at[n, 1, 0].add(coefficient * r_j)
            zernike = zernike.at[n, 1, 1].add(coefficient)

    for m in range(mpol + 1):
        for n in range(m, lrad + 1, 2):
            zernike = zernike.at[n, m, :].divide(float(n + 1))
    return zernike


def zernike_axis_basis(lrad: int, mpol: int) -> Array:
    """Evaluate SPECTRE ``get_zernike_rm(0)`` for axis multipliers."""

    r_j = jnp.asarray(0.0, dtype=jnp.float64)
    zernike = jnp.zeros((lrad + 1, mpol + 1), dtype=jnp.float64)
    for m in range(mpol + 1):
        if lrad >= m:
            zernike = zernike.at[m, m].set(1.0)
        if lrad >= m + 2:
            zernike = zernike.at[m + 2, m].set(float(m + 2) * r_j**2 - float(m + 1))
        for n in range(m + 4, lrad + 1, 2):
            factor1 = float(n) / float(n**2 - m**2)
            factor2 = float(4 * (n - 1))
            factor3 = float((n - 2 + m) ** 2) / float(n - 2) + float((n - m) ** 2) / float(n)
            factor4 = float((n - 2) ** 2 - m**2) / float(n - 2)
            value = factor1 * ((factor2 * r_j**2 - factor3) * zernike[n - 2, m] - factor4 * zernike[n - 4, m])
            zernike = zernike.at[n, m].set(value)

    for n in range(2, lrad + 1, 2):
        zernike = zernike.at[n, 0].add(-(-1.0) ** (n // 2))
    if mpol >= 1:
        for n in range(3, lrad + 1, 2):
            zernike = zernike.at[n, 1].add(-(-1.0) ** ((n - 1) // 2) * float((n + 1) // 2))

    for m in range(mpol + 1):
        for n in range(m, lrad + 1, 2):
            zernike = zernike.at[n, m].divide(float(n + 1))
    return zernike


def spectre_boundary_radial_basis(
    *,
    lrad: int,
    mpol: int,
    coordinate_singularity: bool,
) -> SpectreRadialBoundaryBasis:
    """Return endpoint radial basis values used in SPECTRE ``matrix``."""

    if coordinate_singularity:
        values = jnp.stack(
            (
                zernike_basis(0.0, lrad=lrad, mpol=mpol)[:, :, 0],
                zernike_basis(1.0, lrad=lrad, mpol=mpol)[:, :, 0],
            ),
            axis=-1,
        )
        axis_values = zernike_axis_basis(lrad, mpol)
    else:
        left = chebyshev_basis(-1.0, lrad)[:, 0]
        right = chebyshev_basis(1.0, lrad)[:, 0]
        values_1d = jnp.stack((left, right), axis=-1)
        values = jnp.repeat(values_1d[:, None, :], mpol + 1, axis=1)
        axis_values = jnp.repeat(left[:, None], mpol + 1, axis=1)
    return SpectreRadialBoundaryBasis(values=values, axis_values=axis_values)


def spectre_radial_basis_at_quadrature(
    *,
    lrad: int,
    mpol: int,
    quadrature: SpectreRadialQuadrature,
    coordinate_singularity: bool,
) -> Array:
    """Evaluate SPECTRE radial basis on all quadrature points."""

    rows = []
    for s in quadrature.abscissae:
        if coordinate_singularity:
            rows.append(zernike_basis(0.5 * (s + 1.0), lrad=lrad, mpol=mpol))
        else:
            cheby = chebyshev_basis(s, lrad)
            rows.append(jnp.repeat(cheby[:, None, :], mpol + 1, axis=1))
    return jnp.stack(rows, axis=-1)
