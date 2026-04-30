"""JAX-native SPECTRE Beltrami metric-integral assembly.

This module is the direct-integration counterpart of SPECTRE
``chebyshev_mod.F90::volume_integrate_chebyshev``.  It computes the radial and
angular integral tensors consumed by ``matrices_mod.F90::matrix`` without using
SPECTRE or SPEC.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from .spectre_geometry import (
    SpectreInterfaceGeometry,
    build_spectre_interface_geometry,
    evaluate_spectre_volume_coordinates,
    interpolate_spectre_volume_geometry,
)
from .spectre_input import SpectreInputSummary
from .spectre_pack import spectre_fourier_modes
from .spectre_radial import (
    SpectreRadialQuadrature,
    gauss_legendre_quadrature,
    spectre_default_angular_grid,
    spectre_default_quadrature_size,
    spectre_radial_basis_at_quadrature,
)

Array = jnp.ndarray

_PI2 = 2.0 * np.pi


@dataclass(frozen=True)
class SpectreMetricIntegrals:
    """Metric integral tensors used by SPECTRE matrix assembly.

    Arrays have shape ``(lrad + 1, lrad + 1, mn, mn)``.  For
    coordinate-singularity volumes, only the compressed radial indices used by
    SPECTRE are populated.
    """

    d_toocc: Array
    d_toocs: Array
    d_toosc: Array
    d_tooss: Array
    ttsscc: Array
    ttsscs: Array
    ttsssc: Array
    ttssss: Array
    tdstcc: Array
    tdstcs: Array
    tdstsc: Array
    tdstss: Array
    tdszcc: Array
    tdszcs: Array
    tdszsc: Array
    tdszss: Array
    ddttcc: Array
    ddttcs: Array
    ddttsc: Array
    ddttss: Array
    ddtzcc: Array
    ddtzcs: Array
    ddtzsc: Array
    ddtzss: Array
    ddzzcc: Array
    ddzzcs: Array
    ddzzsc: Array
    ddzzss: Array
    lrad: int
    mpol: int
    nfp: int
    lvol: int
    nt: int
    nz: int
    coordinate_singularity: bool
    enforce_stellarator_symmetry: bool

    @property
    def mode_count(self) -> int:
        """Number of SPECTRE Fourier modes represented by each tensor."""

        return int(self.d_toocc.shape[2])

    @property
    def radial_size(self) -> int:
        """Stored radial tensor size."""

        return int(self.d_toocc.shape[0])


def _active_radial_basis(
    basis: Array,
    *,
    lrad: int,
    poloidal_modes: Array,
    derivative: bool,
    coordinate_singularity: bool,
) -> Array:
    """Return basis values indexed by SPECTRE's compressed radial index."""

    mode_count = int(poloidal_modes.shape[0])
    output = jnp.zeros((lrad + 1, mode_count, basis.shape[-1]), dtype=jnp.float64)
    derivative_index = 1 if derivative else 0
    for mode_index, m_value in enumerate(np.asarray(poloidal_modes, dtype=np.int64)):
        m = int(m_value)
        for ll in range(lrad + 1):
            if coordinate_singularity:
                if ll < m or (ll + m) % 2 != 0:
                    continue
                ll1 = (ll - (ll % 2)) // 2
                factor = 0.5 if derivative else 1.0
            else:
                ll1 = ll
                factor = 1.0
            output = output.at[ll1, mode_index, :].set(factor * basis[ll, m, derivative_index, :])
    return output


def _phase_trig(
    *,
    poloidal_modes: Array,
    toroidal_modes: Array,
    theta: Array,
    zeta: Array,
) -> tuple[Array, Array]:
    phase = poloidal_modes[:, None, None] * theta[None, :, None] - toroidal_modes[:, None, None] * zeta[None, None, :]
    return jnp.cos(phase), jnp.sin(phase)


def _angular_products(cos_phase: Array, sin_phase: Array, metric_component: Array, angle_weight: float) -> dict[str, Array]:
    weighted_metric = metric_component * angle_weight
    return {
        "cc": jnp.einsum("iab,ab,jab->ij", cos_phase, weighted_metric, cos_phase),
        "cs": jnp.einsum("iab,ab,jab->ij", cos_phase, weighted_metric, sin_phase),
        "sc": jnp.einsum("iab,ab,jab->ij", sin_phase, weighted_metric, cos_phase),
        "ss": jnp.einsum("iab,ab,jab->ij", sin_phase, weighted_metric, sin_phase),
    }


def _radial_angular_contract(left: Array, right: Array, angular_by_quad: Array, weights: Array) -> Array:
    weighted = angular_by_quad * weights[:, None, None]
    return jnp.einsum("liq,pjq,qij->lpij", left, right, weighted)


def assemble_spectre_metric_integrals(
    *,
    geometry: SpectreInterfaceGeometry,
    lvol: int,
    lrad: int,
    quadrature: SpectreRadialQuadrature,
    nt: int,
    nz: int,
) -> SpectreMetricIntegrals:
    """Assemble SPECTRE metric integrals for one volume.

    The implementation evaluates SPECTRE's geometry on the same uniform
    angular grid used by its FFT path and performs the angular products
    directly.  This avoids reproducing SPECTRE's intermediate double-angle
    coefficient tables while preserving the same mathematical integrals.
    """

    if nt <= 0 or nz <= 0:
        raise ValueError(f"nt and nz must be positive, got nt={nt}, nz={nz}")
    if lrad < 0:
        raise ValueError(f"lrad must be non-negative, got {lrad}")

    poloidal_modes = geometry.poloidal_modes.astype(jnp.int32)
    toroidal_modes = geometry.toroidal_modes.astype(jnp.int32)
    coordinate_singularity = bool(geometry.igeometry != 1 and lvol == 1)
    theta = jnp.arange(nt, dtype=jnp.float64) * (_PI2 / nt)
    zeta = jnp.arange(nz, dtype=jnp.float64) * (_PI2 / (geometry.nfp * nz))
    cos_phase, sin_phase = _phase_trig(
        poloidal_modes=poloidal_modes,
        toroidal_modes=toroidal_modes,
        theta=theta,
        zeta=zeta,
    )
    angle_weight = (_PI2 * _PI2 / geometry.nfp) / float(nt * nz)

    basis = spectre_radial_basis_at_quadrature(
        lrad=lrad,
        mpol=int(jnp.max(poloidal_modes)) if poloidal_modes.size else 0,
        quadrature=quadrature,
        coordinate_singularity=coordinate_singularity,
    )
    value_basis = _active_radial_basis(
        basis,
        lrad=lrad,
        poloidal_modes=poloidal_modes,
        derivative=False,
        coordinate_singularity=coordinate_singularity,
    )
    derivative_basis = _active_radial_basis(
        basis,
        lrad=lrad,
        poloidal_modes=poloidal_modes,
        derivative=True,
        coordinate_singularity=coordinate_singularity,
    )

    angular_identity = []
    angular_gss = []
    angular_gst = []
    angular_gsz = []
    angular_gtt = []
    angular_gtz = []
    angular_gzz = []
    for s in quadrature.abscissae:
        volume = interpolate_spectre_volume_geometry(geometry, lvol=lvol, s=s)
        grid = evaluate_spectre_volume_coordinates(volume, theta=theta, zeta=zeta)
        metric_bar = grid.metric * grid.inverse_jacobian[..., None, None]
        ones = jnp.ones_like(grid.jacobian)
        angular_identity.append(_angular_products(cos_phase, sin_phase, ones, angle_weight))
        angular_gss.append(_angular_products(cos_phase, sin_phase, metric_bar[..., 0, 0], angle_weight))
        angular_gst.append(_angular_products(cos_phase, sin_phase, metric_bar[..., 0, 1], angle_weight))
        angular_gsz.append(_angular_products(cos_phase, sin_phase, metric_bar[..., 0, 2], angle_weight))
        angular_gtt.append(_angular_products(cos_phase, sin_phase, metric_bar[..., 1, 1], angle_weight))
        angular_gtz.append(_angular_products(cos_phase, sin_phase, metric_bar[..., 1, 2], angle_weight))
        angular_gzz.append(_angular_products(cos_phase, sin_phase, metric_bar[..., 2, 2], angle_weight))

    def stack(name: str, entries: list[dict[str, Array]]) -> Array:
        return jnp.stack([entry[name] for entry in entries], axis=0)

    zeros = jnp.zeros((lrad + 1, lrad + 1, poloidal_modes.size, poloidal_modes.size), dtype=jnp.float64)
    weights = quadrature.weights

    d_toocc = _radial_angular_contract(derivative_basis, value_basis, stack("cc", angular_identity), weights)
    ttssss = _radial_angular_contract(value_basis, value_basis, stack("ss", angular_gss), weights)
    tdstsc = _radial_angular_contract(value_basis, derivative_basis, stack("sc", angular_gst), weights)
    tdszsc = _radial_angular_contract(value_basis, derivative_basis, stack("sc", angular_gsz), weights)
    ddttcc = _radial_angular_contract(derivative_basis, derivative_basis, stack("cc", angular_gtt), weights)
    ddtzcc = _radial_angular_contract(derivative_basis, derivative_basis, stack("cc", angular_gtz), weights)
    ddzzcc = _radial_angular_contract(derivative_basis, derivative_basis, stack("cc", angular_gzz), weights)

    if geometry.enforce_stellarator_symmetry:
        d_toocs = d_toosc = d_tooss = zeros
        ttsscc = ttsscs = ttsssc = zeros
        tdstcc = tdstcs = tdstss = zeros
        tdszcc = tdszcs = tdszss = zeros
        ddttcs = ddttsc = ddttss = zeros
        ddtzcs = ddtzsc = ddtzss = zeros
        ddzzcs = ddzzsc = ddzzss = zeros
    else:
        d_toocs = _radial_angular_contract(derivative_basis, value_basis, stack("cs", angular_identity), weights)
        d_toosc = _radial_angular_contract(derivative_basis, value_basis, stack("sc", angular_identity), weights)
        d_tooss = _radial_angular_contract(derivative_basis, value_basis, stack("ss", angular_identity), weights)
        ttsscc = _radial_angular_contract(value_basis, value_basis, stack("cc", angular_gss), weights)
        ttsscs = _radial_angular_contract(value_basis, value_basis, stack("cs", angular_gss), weights)
        ttsssc = _radial_angular_contract(value_basis, value_basis, stack("sc", angular_gss), weights)
        tdstcc = _radial_angular_contract(value_basis, derivative_basis, stack("cc", angular_gst), weights)
        tdstcs = _radial_angular_contract(value_basis, derivative_basis, stack("cs", angular_gst), weights)
        tdstss = _radial_angular_contract(value_basis, derivative_basis, stack("ss", angular_gst), weights)
        tdszcc = _radial_angular_contract(value_basis, derivative_basis, stack("cc", angular_gsz), weights)
        tdszcs = _radial_angular_contract(value_basis, derivative_basis, stack("cs", angular_gsz), weights)
        tdszss = _radial_angular_contract(value_basis, derivative_basis, stack("ss", angular_gsz), weights)
        ddttcs = _radial_angular_contract(derivative_basis, derivative_basis, stack("cs", angular_gtt), weights)
        ddttsc = _radial_angular_contract(derivative_basis, derivative_basis, stack("sc", angular_gtt), weights)
        ddttss = _radial_angular_contract(derivative_basis, derivative_basis, stack("ss", angular_gtt), weights)
        ddtzcs = _radial_angular_contract(derivative_basis, derivative_basis, stack("cs", angular_gtz), weights)
        ddtzsc = _radial_angular_contract(derivative_basis, derivative_basis, stack("sc", angular_gtz), weights)
        ddtzss = _radial_angular_contract(derivative_basis, derivative_basis, stack("ss", angular_gtz), weights)
        ddzzcs = _radial_angular_contract(derivative_basis, derivative_basis, stack("cs", angular_gzz), weights)
        ddzzsc = _radial_angular_contract(derivative_basis, derivative_basis, stack("sc", angular_gzz), weights)
        ddzzss = _radial_angular_contract(derivative_basis, derivative_basis, stack("ss", angular_gzz), weights)

    return SpectreMetricIntegrals(
        d_toocc=d_toocc,
        d_toocs=d_toocs,
        d_toosc=d_toosc,
        d_tooss=d_tooss,
        ttsscc=ttsscc,
        ttsscs=ttsscs,
        ttsssc=ttsssc,
        ttssss=ttssss,
        tdstcc=tdstcc,
        tdstcs=tdstcs,
        tdstsc=tdstsc,
        tdstss=tdstss,
        tdszcc=tdszcc,
        tdszcs=tdszcs,
        tdszsc=tdszsc,
        tdszss=tdszss,
        ddttcc=ddttcc,
        ddttcs=ddttcs,
        ddttsc=ddttsc,
        ddttss=ddttss,
        ddtzcc=ddtzcc,
        ddtzcs=ddtzcs,
        ddtzsc=ddtzsc,
        ddtzss=ddtzss,
        ddzzcc=ddzzcc,
        ddzzcs=ddzzcs,
        ddzzsc=ddzzsc,
        ddzzss=ddzzss,
        lrad=lrad,
        mpol=int(jnp.max(poloidal_modes)) if poloidal_modes.size else 0,
        nfp=geometry.nfp,
        lvol=lvol,
        nt=nt,
        nz=nz,
        coordinate_singularity=coordinate_singularity,
        enforce_stellarator_symmetry=geometry.enforce_stellarator_symmetry,
    )


def assemble_spectre_metric_integrals_from_input(
    summary: SpectreInputSummary,
    *,
    lvol: int,
    geometry: SpectreInterfaceGeometry | None = None,
    quadrature_size: int | None = None,
    nt: int | None = None,
    nz: int | None = None,
) -> SpectreMetricIntegrals:
    """Assemble metric integrals from a SPECTRE TOML/interface summary."""

    if lvol < 1 or lvol > summary.packed_volume_count:
        raise ValueError(f"lvol={lvol} outside 1..{summary.packed_volume_count}")
    geometry = build_spectre_interface_geometry(summary) if geometry is None else geometry
    if lvol > geometry.interface_count:
        raise ValueError(
            f"lvol={lvol} requires interface geometry through row {lvol}; "
            f"only {geometry.interface_count} interface rows are available"
        )
    if quadrature_size is None:
        quadrature_size = spectre_default_quadrature_size(summary, lvol=lvol)
    if nt is None or nz is None:
        default_nt, default_nz = spectre_default_angular_grid(summary)
        nt = default_nt if nt is None else nt
        nz = default_nz if nz is None else nz
    quadrature = gauss_legendre_quadrature(quadrature_size)
    return assemble_spectre_metric_integrals(
        geometry=geometry,
        lvol=lvol,
        lrad=summary.lrad[lvol - 1],
        quadrature=quadrature,
        nt=nt,
        nz=nz,
    )


def spectre_integral_mode_labels(summary: SpectreInputSummary) -> tuple[tuple[int, int], ...]:
    """Return mode labels for assembled metric-integral tensors."""

    return spectre_fourier_modes(summary)
