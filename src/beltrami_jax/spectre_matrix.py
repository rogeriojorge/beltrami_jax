"""SPECTRE Beltrami matrix-assembly pieces.

This module ports the boundary-source part of SPECTRE's
``matrices_mod.F90::matrixBG``.  It assembles the flux-coupling matrix ``dMB``
and boundary-normal-field source vector ``dMG`` from the SPECTRE packed
degree-of-freedom maps, without asking SPECTRE or SPEC to do that assembly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import jax.numpy as jnp

from .spectre_input import SpectreInputSummary
from .spectre_pack import SpectreVolumeDofMap, build_spectre_dof_layout, spectre_fourier_modes

Array = jnp.ndarray


@dataclass(frozen=True)
class SpectreBoundaryNormalField:
    """Boundary-normal-field coefficients in SPECTRE internal mode order.

    The arrays have shape ``(mn,)`` and correspond to SPECTRE's initialized
    ``iVns``, ``iBns``, ``iVnc``, and ``iBnc`` arrays after the input tables
    have been recombined into the internal ``gi00ab`` mode order.  The
    ``ns`` arrays are the stellarator-symmetric sine-parity contribution and
    the ``nc`` arrays are the non-stellarator-symmetric cosine-parity
    contribution.
    """

    ivns: Array
    ibns: Array
    ivnc: Array
    ibnc: Array

    @property
    def mode_count(self) -> int:
        """Number of SPECTRE Fourier modes."""

        return int(self.ivns.shape[0])

    def validate(self, mode_count: int) -> None:
        """Validate that all normal-field arrays match ``mode_count``."""

        expected = (mode_count,)
        for name in ("ivns", "ibns", "ivnc", "ibnc"):
            value = getattr(self, name)
            if value.shape != expected:
                raise ValueError(f"{name} shape {value.shape} does not match expected {expected}")


@dataclass(frozen=True)
class SpectreMatrixBG:
    """Boundary matrix/source pair assembled by SPECTRE ``matrixBG``."""

    d_mb: Array
    d_mg: Array

    @property
    def solution_size(self) -> int:
        """Per-volume SPECTRE solution-vector length."""

        return int(self.d_mg.shape[0])


def _parse_mode_key(key: Any) -> tuple[int, int]:
    text = str(key).strip()
    if not (text.startswith("(") and text.endswith(")")):
        raise ValueError(f"invalid SPECTRE Fourier mode key: {key!r}")
    parts = [part.strip() for part in text[1:-1].split(",")]
    if len(parts) != 2:
        raise ValueError(f"invalid SPECTRE Fourier mode key: {key!r}")
    return int(parts[0]), int(parts[1])


def _parse_mode_table(raw_table: Any) -> dict[tuple[int, int], float]:
    if raw_table is None:
        return {}
    if not isinstance(raw_table, Mapping):
        raise ValueError("SPECTRE normal-field tables must be mappings")
    return {_parse_mode_key(key): float(value) for key, value in raw_table.items()}


def _table_value(table: Mapping[tuple[int, int], float], m: int, n: int) -> float:
    return float(table.get((m, n), 0.0))


def _logical_toroidal_mode(internal_n: int, nfp: int) -> int:
    if nfp == 0:
        raise ValueError("SPECTRE nfp must be nonzero")
    if internal_n % nfp != 0:
        raise ValueError(f"internal toroidal mode {internal_n} is not divisible by nfp={nfp}")
    return internal_n // nfp


def build_spectre_boundary_normal_field(summary: SpectreInputSummary) -> SpectreBoundaryNormalField:
    """Build SPECTRE ``iVns/iBns/iVnc/iBnc`` arrays from a TOML summary.

    The recombination follows SPECTRE ``preset_mod.F90`` for the default angle
    convention:

    ``iVns(m,n) = Vns(m,n) - Vns(-m,-n)``

    ``iBns(m,n) = Bns(m,n) - Bns(-m,-n)``

    ``iVnc(m,n) = Vnc(m,n) + Vnc(-m,-n)``

    ``iBnc(m,n) = Bnc(m,n) + Bnc(-m,-n)``

    Fixed-boundary inputs keep these arrays at zero unless
    ``physics.lbdybnzero = false``.  Free-boundary inputs always use the normal
    field tables, matching SPECTRE's initialization before any Picard
    free-boundary update changes ``iBns/iBnc``.
    """

    modes = spectre_fourier_modes(summary)
    vns = _parse_mode_table(summary.physics.get("vns"))
    bns = _parse_mode_table(summary.physics.get("bns"))
    vnc = _parse_mode_table(summary.physics.get("vnc"))
    bnc = _parse_mode_table(summary.physics.get("bnc"))
    use_boundary_normal_field = summary.is_free_boundary or not bool(summary.physics.get("lbdybnzero", True))
    lbnszero = bool(summary.physics.get("lbnszero", False))
    enforce_symmetry = summary.enforce_stellarator_symmetry

    ivns: list[float] = []
    ibns: list[float] = []
    ivnc: list[float] = []
    ibnc: list[float] = []

    for m, internal_n in modes:
        n = _logical_toroidal_mode(internal_n, summary.nfp)
        if m == 0 and n == 0:
            ns_vacuum = 0.0
            ns_plasma = 0.0
            if summary.is_free_boundary and not enforce_symmetry:
                nc_vacuum = _table_value(vnc, 0, 0)
                nc_plasma = _table_value(bnc, 0, 0)
            else:
                nc_vacuum = 0.0
                nc_plasma = 0.0
        elif use_boundary_normal_field:
            ns_vacuum = _table_value(vns, m, n) - _table_value(vns, -m, -n)
            ns_plasma = _table_value(bns, m, n) - _table_value(bns, -m, -n)
            if enforce_symmetry:
                nc_vacuum = 0.0
                nc_plasma = 0.0
            else:
                nc_vacuum = _table_value(vnc, m, n) + _table_value(vnc, -m, -n)
                nc_plasma = _table_value(bnc, m, n) + _table_value(bnc, -m, -n)
        else:
            ns_vacuum = 0.0
            ns_plasma = 0.0
            nc_vacuum = 0.0
            nc_plasma = 0.0

        if lbnszero:
            ns_plasma = 0.0
            nc_plasma = 0.0

        ivns.append(ns_vacuum)
        ibns.append(ns_plasma)
        ivnc.append(nc_vacuum)
        ibnc.append(nc_plasma)

    return SpectreBoundaryNormalField(
        ivns=jnp.asarray(ivns, dtype=jnp.float64),
        ibns=jnp.asarray(ibns, dtype=jnp.float64),
        ivnc=jnp.asarray(ivnc, dtype=jnp.float64),
        ibnc=jnp.asarray(ibnc, dtype=jnp.float64),
    )


def spectre_boundary_normal_field_from_dmg(
    volume_map: SpectreVolumeDofMap,
    d_mg: object,
) -> SpectreBoundaryNormalField:
    """Recover a boundary-normal-field source equivalent to SPECTRE ``dMG``.

    SPECTRE's ``matrixBG`` uses only the sums ``iVns + iBns`` and
    ``iVnc + iBnc`` when scattering the free-boundary source into ``dMG``.
    Released validation fixtures and SPECTRE runtime seams can therefore
    provide either the live ``iV/iB`` arrays or an already exported ``dMG``
    vector.  This helper reconstructs an equivalent source state from ``dMG``
    for one packed volume by storing the combined source in the vacuum
    component and leaving the plasma component zero.

    The reconstruction is intended for validation and adapter seams.  It does
    not claim to separate the physical vacuum and plasma contributions after
    SPECTRE's free-boundary Picard update has mixed them.
    """

    source = jnp.asarray(d_mg, dtype=jnp.float64)
    expected = (volume_map.solution_size,)
    if source.shape != expected:
        raise ValueError(f"d_mg shape {source.shape} does not match expected {expected}")

    ivns = jnp.zeros((volume_map.mode_count,), dtype=jnp.float64)
    ibns = jnp.zeros((volume_map.mode_count,), dtype=jnp.float64)
    ivnc = jnp.zeros((volume_map.mode_count,), dtype=jnp.float64)
    ibnc = jnp.zeros((volume_map.mode_count,), dtype=jnp.float64)

    lme_ids = jnp.asarray(volume_map.lme, dtype=jnp.int32) - 1
    lme_mask = lme_ids >= 0
    lme_safe = jnp.where(lme_mask, lme_ids, 0)
    ivns = jnp.where(lme_mask, -source[lme_safe], ivns)

    lmf_ids = jnp.asarray(volume_map.lmf, dtype=jnp.int32) - 1
    lmf_mask = lmf_ids >= 0
    lmf_safe = jnp.where(lmf_mask, lmf_ids, 0)
    ivnc = jnp.where(lmf_mask, -source[lmf_safe], ivnc)

    return SpectreBoundaryNormalField(ivns=ivns, ibns=ibns, ivnc=ivnc, ibnc=ibnc)


def _scatter_vector(ids: object, values: Array, size: int) -> Array:
    ids_j = jnp.asarray(ids, dtype=jnp.int32) - 1
    mask = ids_j >= 0
    safe_ids = jnp.where(mask, ids_j, 0)
    safe_values = jnp.where(mask, values, 0.0)
    return jnp.zeros((size,), dtype=jnp.float64).at[safe_ids].add(safe_values)


def _scatter_column(ids: object, values: Array, *, size: int, column: int) -> Array:
    vector = _scatter_vector(ids, values, size)
    return jnp.zeros((size, 2), dtype=jnp.float64).at[:, column].set(vector)


def assemble_spectre_matrix_bg(
    volume_map: SpectreVolumeDofMap,
    normal_field: SpectreBoundaryNormalField,
) -> SpectreMatrixBG:
    """Assemble SPECTRE ``matrixBG`` for one packed volume map.

    Parameters
    ----------
    volume_map:
        One per-volume SPECTRE degree-of-freedom map from
        :func:`beltrami_jax.build_spectre_dof_layout`.
    normal_field:
        Boundary-normal-field arrays in SPECTRE internal mode order.  Use
        :func:`build_spectre_boundary_normal_field` to construct them from a
        SPECTRE TOML input, or pass updated free-boundary arrays from a coupled
        free-boundary iteration.

    Returns
    -------
    SpectreMatrixBG
        JAX arrays ``d_mb`` with shape ``(NAdof, 2)`` and ``d_mg`` with shape
        ``(NAdof,)``.
    """

    normal_field.validate(volume_map.mode_count)
    size = volume_map.solution_size

    lmg = jnp.asarray(volume_map.lmg, dtype=jnp.int32)
    lmh = jnp.asarray(volume_map.lmh, dtype=jnp.int32)
    d_mb = _scatter_column(lmg[:1], jnp.asarray([-1.0], dtype=jnp.float64), size=size, column=0)
    d_mb = d_mb + _scatter_column(lmh[:1], jnp.asarray([1.0], dtype=jnp.float64), size=size, column=1)

    lme_values = -(normal_field.ivns + normal_field.ibns)
    d_mg = _scatter_vector(volume_map.lme, lme_values, size)
    if not volume_map.enforce_stellarator_symmetry:
        lmf_values = -(normal_field.ivnc + normal_field.ibnc)
        d_mg = d_mg + _scatter_vector(volume_map.lmf, lmf_values, size)

    return SpectreMatrixBG(d_mb=d_mb, d_mg=d_mg)


def assemble_spectre_matrix_bg_from_input(
    summary: SpectreInputSummary,
    lvol: int,
    *,
    normal_field: SpectreBoundaryNormalField | None = None,
) -> SpectreMatrixBG:
    """Assemble ``dMB/dMG`` for one one-based SPECTRE volume from TOML data."""

    dof_layout = build_spectre_dof_layout(summary)
    if lvol < 1 or lvol > len(dof_layout.volume_maps):
        raise ValueError(f"lvol={lvol} outside 1..{len(dof_layout.volume_maps)}")
    field = normal_field if normal_field is not None else build_spectre_boundary_normal_field(summary)
    return assemble_spectre_matrix_bg(dof_layout.volume_maps[lvol - 1], field)
