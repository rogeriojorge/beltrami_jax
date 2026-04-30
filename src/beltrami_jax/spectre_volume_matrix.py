"""SPECTRE ``dMA/dMD`` Beltrami matrix assembly in JAX.

This module ports the contraction layer of
``matrices_mod.F90::matrix``.  It consumes metric-integral tensors assembled by
``spectre_integrals`` and the SPECTRE degree-of-freedom maps from
``spectre_pack`` to produce the volume matrices that SPECTRE uses in the local
Beltrami solve.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from .spectre_input import SpectreInputSummary
from .spectre_integrals import SpectreMetricIntegrals, assemble_spectre_metric_integrals_from_input
from .spectre_matrix import (
    SpectreBoundaryNormalField,
    assemble_spectre_matrix_bg,
    build_spectre_boundary_normal_field,
)
from .spectre_pack import SpectreVolumeDofMap, build_spectre_dof_layout
from .spectre_radial import spectre_boundary_radial_basis

Array = jnp.ndarray


@dataclass(frozen=True)
class SpectreMatrixAD:
    """Volume matrix pair assembled by SPECTRE ``matrix``."""

    d_ma: Array
    d_md: Array

    @property
    def solution_size(self) -> int:
        """Per-volume SPECTRE solution-vector length."""

        return int(self.d_ma.shape[0])


@dataclass(frozen=True)
class SpectreVolumeMatrices:
    """Complete local SPECTRE matrix ingredients for one volume."""

    d_ma: Array
    d_md: Array
    d_mb: Array
    d_mg: Array

    @property
    def solution_size(self) -> int:
        """Per-volume SPECTRE solution-vector length."""

        return int(self.d_ma.shape[0])


def _active_radial_index(*, ll: int, m: int, coordinate_singularity: bool) -> int | None:
    if not coordinate_singularity:
        return ll
    if ll < m or (ll + m) % 2 != 0:
        return None
    return (ll - (ll % 2)) // 2


def _set_entry(matrix: Array, row_id: int, column_id: int, value: Array | float) -> Array:
    if row_id <= 0 or column_id <= 0:
        return matrix
    return matrix.at[row_id - 1, column_id - 1].set(value)


def _append_entry(rows: list[int], columns: list[int], values: list[Array | float], row_id: int, column_id: int, value: Array | float) -> None:
    if row_id <= 0 or column_id <= 0:
        return
    rows.append(row_id - 1)
    columns.append(column_id - 1)
    values.append(value)


def _scatter_entries(size: int, rows: list[int], columns: list[int], values: list[Array | float]) -> Array:
    matrix = jnp.zeros((size, size), dtype=jnp.float64)
    if not rows:
        return matrix
    return matrix.at[
        jnp.asarray(rows, dtype=jnp.int32),
        jnp.asarray(columns, dtype=jnp.int32),
    ].set(jnp.stack([jnp.asarray(value, dtype=jnp.float64) for value in values]))


def _validate_integrals(volume_map: SpectreVolumeDofMap, integrals: SpectreMetricIntegrals) -> None:
    if volume_map.mode_count != integrals.mode_count:
        raise ValueError(
            f"mode-count mismatch: volume map has {volume_map.mode_count}, "
            f"integrals have {integrals.mode_count}"
        )
    if volume_map.block.lrad != integrals.lrad:
        raise ValueError(
            f"lrad mismatch: volume map has {volume_map.block.lrad}, integrals have {integrals.lrad}"
        )
    if volume_map.coordinate_singularity != integrals.coordinate_singularity:
        raise ValueError("coordinate-singularity flag mismatch between volume map and integrals")


def _assemble_stellarator_symmetric(
    volume_map: SpectreVolumeDofMap,
    integrals: SpectreMetricIntegrals,
    d_ma: Array,
    d_md: Array,
) -> tuple[Array, Array]:
    modes_m = np.asarray(volume_map.poloidal_modes, dtype=np.int64)
    modes_n = np.asarray(volume_map.toroidal_modes, dtype=np.int64)
    lrad = volume_map.block.lrad
    ma_rows: list[int] = []
    ma_columns: list[int] = []
    ma_values: list[Array | float] = []
    md_rows: list[int] = []
    md_columns: list[int] = []
    md_values: list[Array | float] = []

    for ii in range(volume_map.mode_count):
        mi = int(modes_m[ii])
        ni = int(modes_n[ii])
        for jj in range(volume_map.mode_count):
            mj = int(modes_m[jj])
            nj = int(modes_n[jj])
            mimj = mi * mj
            minj = mi * nj
            nimj = ni * mj
            ninj = ni * nj
            for ll in range(lrad + 1):
                ll1 = _active_radial_index(
                    ll=ll,
                    m=mi,
                    coordinate_singularity=volume_map.coordinate_singularity,
                )
                if ll1 is None:
                    continue
                for pp in range(lrad + 1):
                    pp1 = _active_radial_index(
                        ll=pp,
                        m=mj,
                        coordinate_singularity=volume_map.coordinate_singularity,
                    )
                    if pp1 is None:
                        continue

                    wtete = (
                        2 * ninj * integrals.ttssss[ll1, pp1, ii, jj]
                        - 2 * ni * integrals.tdszsc[ll1, pp1, ii, jj]
                        - 2 * nj * integrals.tdszsc[pp1, ll1, jj, ii]
                        + 2 * integrals.ddzzcc[ll1, pp1, ii, jj]
                    )
                    wzete = (
                        2 * nimj * integrals.ttssss[ll1, pp1, ii, jj]
                        + 2 * ni * integrals.tdstsc[ll1, pp1, ii, jj]
                        - 2 * mj * integrals.tdszsc[pp1, ll1, jj, ii]
                        - 2 * integrals.ddtzcc[pp1, ll1, jj, ii]
                    )
                    wteze = (
                        2 * minj * integrals.ttssss[ll1, pp1, ii, jj]
                        + 2 * nj * integrals.tdstsc[pp1, ll1, jj, ii]
                        - 2 * mi * integrals.tdszsc[ll1, pp1, ii, jj]
                        - 2 * integrals.ddtzcc[ll1, pp1, ii, jj]
                    )
                    wzeze = (
                        2 * mimj * integrals.ttssss[ll1, pp1, ii, jj]
                        + 2 * mi * integrals.tdstsc[ll1, pp1, ii, jj]
                        + 2 * mj * integrals.tdstsc[pp1, ll1, jj, ii]
                        + 2 * integrals.ddttcc[ll1, pp1, ii, jj]
                    )
                    hzete = -integrals.d_toocc[pp1, ll1, jj, ii] + integrals.d_toocc[ll1, pp1, ii, jj]
                    hteze = integrals.d_toocc[pp1, ll1, jj, ii] - integrals.d_toocc[ll1, pp1, ii, jj]

                    ate_i = int(volume_map.ate[ll, ii])
                    aze_i = int(volume_map.aze[ll, ii])
                    ate_j = int(volume_map.ate[pp, jj])
                    aze_j = int(volume_map.aze[pp, jj])
                    _append_entry(ma_rows, ma_columns, ma_values, ate_i, ate_j, wtete)
                    _append_entry(ma_rows, ma_columns, ma_values, ate_i, aze_j, wzete)
                    _append_entry(md_rows, md_columns, md_values, ate_i, aze_j, hzete)
                    _append_entry(ma_rows, ma_columns, ma_values, aze_i, ate_j, wteze)
                    _append_entry(md_rows, md_columns, md_values, aze_i, ate_j, hteze)
                    _append_entry(ma_rows, ma_columns, ma_values, aze_i, aze_j, wzeze)
    size = volume_map.solution_size
    return d_ma + _scatter_entries(size, ma_rows, ma_columns, ma_values), d_md + _scatter_entries(size, md_rows, md_columns, md_values)


def _assemble_non_stellarator_symmetric(
    volume_map: SpectreVolumeDofMap,
    integrals: SpectreMetricIntegrals,
    d_ma: Array,
    d_md: Array,
) -> tuple[Array, Array]:
    modes_m = np.asarray(volume_map.poloidal_modes, dtype=np.int64)
    modes_n = np.asarray(volume_map.toroidal_modes, dtype=np.int64)
    lrad = volume_map.block.lrad

    for ii in range(volume_map.mode_count):
        mi = int(modes_m[ii])
        ni = int(modes_n[ii])
        for jj in range(volume_map.mode_count):
            mj = int(modes_m[jj])
            nj = int(modes_n[jj])
            mjmi = mi * mj
            mjni = ni * mj
            njmi = mi * nj
            njni = ni * nj
            for ll in range(lrad + 1):
                ll1 = _active_radial_index(
                    ll=ll,
                    m=mi,
                    coordinate_singularity=volume_map.coordinate_singularity,
                )
                if ll1 is None:
                    continue
                for pp in range(lrad + 1):
                    pp1 = _active_radial_index(
                        ll=pp,
                        m=mj,
                        coordinate_singularity=volume_map.coordinate_singularity,
                    )
                    if pp1 is None:
                        continue

                    wtete = 2 * (
                        njni * integrals.ttssss[pp1, ll1, jj, ii]
                        - nj * integrals.tdszsc[pp1, ll1, jj, ii]
                        - ni * integrals.tdszsc[ll1, pp1, ii, jj]
                        + integrals.ddzzcc[pp1, ll1, jj, ii]
                    )
                    wtote = 2 * (
                        -njni * integrals.ttsscs[pp1, ll1, jj, ii]
                        + nj * integrals.tdszcc[pp1, ll1, jj, ii]
                        - ni * integrals.tdszss[ll1, pp1, ii, jj]
                        + integrals.ddzzsc[pp1, ll1, jj, ii]
                    )
                    wzete = 2 * (
                        mjni * integrals.ttssss[pp1, ll1, jj, ii]
                        - mj * integrals.tdszsc[pp1, ll1, jj, ii]
                        + ni * integrals.tdstsc[ll1, pp1, ii, jj]
                        - integrals.ddtzcc[pp1, ll1, jj, ii]
                    )
                    wzote = 2 * (
                        -mjni * integrals.ttsscs[pp1, ll1, jj, ii]
                        + mj * integrals.tdszcc[pp1, ll1, jj, ii]
                        + ni * integrals.tdstss[ll1, pp1, ii, jj]
                        - integrals.ddtzsc[pp1, ll1, jj, ii]
                    )
                    wteto = 2 * (
                        -njni * integrals.ttsssc[pp1, ll1, jj, ii]
                        - nj * integrals.tdszss[pp1, ll1, jj, ii]
                        + ni * integrals.tdszcc[ll1, pp1, ii, jj]
                        + integrals.ddzzcs[pp1, ll1, jj, ii]
                    )
                    wtoto = 2 * (
                        njni * integrals.ttsscc[pp1, ll1, jj, ii]
                        + nj * integrals.tdszcs[pp1, ll1, jj, ii]
                        + ni * integrals.tdszcs[ll1, pp1, ii, jj]
                        + integrals.ddzzss[pp1, ll1, jj, ii]
                    )
                    wzeto = 2 * (
                        -mjni * integrals.ttsssc[pp1, ll1, jj, ii]
                        - mj * integrals.tdszss[pp1, ll1, jj, ii]
                        - ni * integrals.tdstcc[ll1, pp1, ii, jj]
                        - integrals.ddtzcs[pp1, ll1, jj, ii]
                    )
                    wzoto = 2 * (
                        mjni * integrals.ttsscc[pp1, ll1, jj, ii]
                        + mj * integrals.tdszcs[pp1, ll1, jj, ii]
                        - ni * integrals.tdstcs[ll1, pp1, ii, jj]
                        - integrals.ddtzss[pp1, ll1, jj, ii]
                    )
                    wteze = 2 * (
                        njmi * integrals.ttssss[pp1, ll1, jj, ii]
                        + nj * integrals.tdstsc[pp1, ll1, jj, ii]
                        - mi * integrals.tdszsc[ll1, pp1, ii, jj]
                        - integrals.ddtzcc[pp1, ll1, jj, ii]
                    )
                    wtoze = 2 * (
                        -njmi * integrals.ttsscs[pp1, ll1, jj, ii]
                        - nj * integrals.tdstcc[pp1, ll1, jj, ii]
                        - mi * integrals.tdszss[ll1, pp1, ii, jj]
                        - integrals.ddtzsc[pp1, ll1, jj, ii]
                    )
                    wzeze = 2 * (
                        mjmi * integrals.ttssss[pp1, ll1, jj, ii]
                        + mj * integrals.tdstsc[pp1, ll1, jj, ii]
                        + mi * integrals.tdstsc[ll1, pp1, ii, jj]
                        + integrals.ddttcc[pp1, ll1, jj, ii]
                    )
                    wzoze = 2 * (
                        -mjmi * integrals.ttsscs[pp1, ll1, jj, ii]
                        - mj * integrals.tdstcc[pp1, ll1, jj, ii]
                        + mi * integrals.tdstss[ll1, pp1, ii, jj]
                        + integrals.ddttsc[pp1, ll1, jj, ii]
                    )
                    wtezo = 2 * (
                        -njmi * integrals.ttsssc[pp1, ll1, jj, ii]
                        + nj * integrals.tdstss[pp1, ll1, jj, ii]
                        + mi * integrals.tdszcc[ll1, pp1, ii, jj]
                        - integrals.ddtzcs[pp1, ll1, jj, ii]
                    )
                    wtozo = 2 * (
                        njmi * integrals.ttsscc[pp1, ll1, jj, ii]
                        - nj * integrals.tdstcs[pp1, ll1, jj, ii]
                        + mi * integrals.tdszcs[ll1, pp1, ii, jj]
                        - integrals.ddtzss[pp1, ll1, jj, ii]
                    )
                    wzezo = 2 * (
                        -mjmi * integrals.ttsssc[pp1, ll1, jj, ii]
                        + mj * integrals.tdstss[pp1, ll1, jj, ii]
                        - mi * integrals.tdstcc[ll1, pp1, ii, jj]
                        + integrals.ddttcs[pp1, ll1, jj, ii]
                    )
                    wzozo = 2 * (
                        mjmi * integrals.ttsscc[pp1, ll1, jj, ii]
                        - mj * integrals.tdstcs[pp1, ll1, jj, ii]
                        - mi * integrals.tdstcs[ll1, pp1, ii, jj]
                        + integrals.ddttss[pp1, ll1, jj, ii]
                    )

                    hzete = -integrals.d_toocc[pp1, ll1, jj, ii] + integrals.d_toocc[ll1, pp1, ii, jj]
                    hzote = -integrals.d_toosc[pp1, ll1, jj, ii] + integrals.d_toocs[ll1, pp1, ii, jj]
                    hzeto = -integrals.d_toocs[pp1, ll1, jj, ii] + integrals.d_toosc[ll1, pp1, ii, jj]
                    hzoto = -integrals.d_tooss[pp1, ll1, jj, ii] + integrals.d_tooss[ll1, pp1, ii, jj]
                    hteze = integrals.d_toocc[pp1, ll1, jj, ii] - integrals.d_toocc[ll1, pp1, ii, jj]
                    htoze = integrals.d_toosc[pp1, ll1, jj, ii] - integrals.d_toocs[ll1, pp1, ii, jj]
                    htezo = integrals.d_toocs[pp1, ll1, jj, ii] - integrals.d_toosc[ll1, pp1, ii, jj]
                    htozo = integrals.d_tooss[pp1, ll1, jj, ii] - integrals.d_tooss[ll1, pp1, ii, jj]

                    ids_i = {
                        "ate": int(volume_map.ate[ll, ii]),
                        "ato": int(volume_map.ato[ll, ii]),
                        "aze": int(volume_map.aze[ll, ii]),
                        "azo": int(volume_map.azo[ll, ii]),
                    }
                    ids_j = {
                        "ate": int(volume_map.ate[pp, jj]),
                        "ato": int(volume_map.ato[pp, jj]),
                        "aze": int(volume_map.aze[pp, jj]),
                        "azo": int(volume_map.azo[pp, jj]),
                    }
                    entries = (
                        ("ate", "ate", wtete, 0.0),
                        ("ate", "ato", wtote, 0.0),
                        ("ate", "aze", wzete, hzete),
                        ("ate", "azo", wzote, hzote),
                        ("ato", "ate", wteto, 0.0),
                        ("ato", "ato", wtoto, 0.0),
                        ("ato", "aze", wzeto, hzeto),
                        ("ato", "azo", wzoto, hzoto),
                        ("aze", "ate", wteze, hteze),
                        ("aze", "ato", wtoze, htoze),
                        ("aze", "aze", wzeze, 0.0),
                        ("aze", "azo", wzoze, 0.0),
                        ("azo", "ate", wtezo, htezo),
                        ("azo", "ato", wtozo, htozo),
                        ("azo", "aze", wzezo, 0.0),
                        ("azo", "azo", wzozo, 0.0),
                    )
                    for row_name, column_name, ma_value, md_value in entries:
                        d_ma = _set_entry(d_ma, ids_i[row_name], ids_j[column_name], ma_value)
                        d_md = _set_entry(d_md, ids_i[row_name], ids_j[column_name], md_value)
    return d_ma, d_md


def _assemble_lagrange_rows(volume_map: SpectreVolumeDofMap, d_ma: Array) -> Array:
    lrad = volume_map.block.lrad
    mpol = int(np.max(volume_map.poloidal_modes)) if volume_map.poloidal_modes.size else 0
    basis = spectre_boundary_radial_basis(
        lrad=lrad,
        mpol=mpol,
        coordinate_singularity=volume_map.coordinate_singularity,
    )
    modes_m = np.asarray(volume_map.poloidal_modes, dtype=np.int64)
    modes_n = np.asarray(volume_map.toroidal_modes, dtype=np.int64)
    rows: list[int] = []
    columns: list[int] = []
    values: list[Array | float] = []

    for ii in range(volume_map.mode_count):
        mi = int(modes_m[ii])
        ni = int(modes_n[ii])
        kk = 1 if (volume_map.coordinate_singularity and ii == 0) else 0
        for ll in range(lrad + 1):
            axis_value = basis.axis_values[ll, mi]
            endpoint_value = basis.values[ll, mi, kk]
            outer_derivative_value = basis.values[ll, mi, 1]

            _append_entry(rows, columns, values, int(volume_map.ate[ll, ii]), int(volume_map.lma[ii]), axis_value)
            _append_entry(rows, columns, values, int(volume_map.aze[ll, ii]), int(volume_map.lmb[ii]), endpoint_value)
            if ii > 0:
                _append_entry(rows, columns, values, int(volume_map.ate[ll, ii]), int(volume_map.lme[ii]), -ni * outer_derivative_value)
                _append_entry(rows, columns, values, int(volume_map.aze[ll, ii]), int(volume_map.lme[ii]), -mi * outer_derivative_value)
                if not volume_map.enforce_stellarator_symmetry:
                    _append_entry(rows, columns, values, int(volume_map.ato[ll, ii]), int(volume_map.lmc[ii]), axis_value)
                    _append_entry(rows, columns, values, int(volume_map.azo[ll, ii]), int(volume_map.lmd[ii]), basis.values[ll, mi, 0])
                    _append_entry(rows, columns, values, int(volume_map.ato[ll, ii]), int(volume_map.lmf[ii]), ni * outer_derivative_value)
                    _append_entry(rows, columns, values, int(volume_map.azo[ll, ii]), int(volume_map.lmf[ii]), mi * outer_derivative_value)
            else:
                _append_entry(rows, columns, values, int(volume_map.ate[ll, ii]), int(volume_map.lmg[ii]), outer_derivative_value)
                _append_entry(rows, columns, values, int(volume_map.aze[ll, ii]), int(volume_map.lmh[ii]), outer_derivative_value)

        for pp in range(lrad + 1):
            axis_value = basis.axis_values[pp, mi]
            endpoint_value = basis.values[pp, mi, kk]
            outer_derivative_value = basis.values[pp, mi, 1]

            _append_entry(rows, columns, values, int(volume_map.lma[ii]), int(volume_map.ate[pp, ii]), axis_value)
            _append_entry(rows, columns, values, int(volume_map.lmb[ii]), int(volume_map.aze[pp, ii]), endpoint_value)
            if ii > 0:
                _append_entry(rows, columns, values, int(volume_map.lme[ii]), int(volume_map.ate[pp, ii]), -ni * outer_derivative_value)
                _append_entry(rows, columns, values, int(volume_map.lme[ii]), int(volume_map.aze[pp, ii]), -mi * outer_derivative_value)
                if not volume_map.enforce_stellarator_symmetry:
                    _append_entry(rows, columns, values, int(volume_map.lmc[ii]), int(volume_map.ato[pp, ii]), axis_value)
                    _append_entry(rows, columns, values, int(volume_map.lmd[ii]), int(volume_map.azo[pp, ii]), basis.values[pp, mi, 0])
                    _append_entry(rows, columns, values, int(volume_map.lmf[ii]), int(volume_map.ato[pp, ii]), ni * outer_derivative_value)
                    _append_entry(rows, columns, values, int(volume_map.lmf[ii]), int(volume_map.azo[pp, ii]), mi * outer_derivative_value)
            else:
                _append_entry(rows, columns, values, int(volume_map.lmg[ii]), int(volume_map.ate[pp, ii]), outer_derivative_value)
                _append_entry(rows, columns, values, int(volume_map.lmh[ii]), int(volume_map.aze[pp, ii]), outer_derivative_value)
    return d_ma + _scatter_entries(volume_map.solution_size, rows, columns, values)


def assemble_spectre_matrix_ad(
    volume_map: SpectreVolumeDofMap,
    integrals: SpectreMetricIntegrals,
) -> SpectreMatrixAD:
    """Assemble SPECTRE ``dMA`` and ``dMD`` for one volume."""

    _validate_integrals(volume_map, integrals)
    size = volume_map.solution_size
    d_ma = jnp.zeros((size, size), dtype=jnp.float64)
    d_md = jnp.zeros((size, size), dtype=jnp.float64)

    if volume_map.enforce_stellarator_symmetry:
        d_ma, d_md = _assemble_stellarator_symmetric(volume_map, integrals, d_ma, d_md)
    else:
        d_ma, d_md = _assemble_non_stellarator_symmetric(volume_map, integrals, d_ma, d_md)
    d_ma = _assemble_lagrange_rows(volume_map, d_ma)
    return SpectreMatrixAD(d_ma=d_ma, d_md=d_md)


def assemble_spectre_matrix_ad_from_input(
    summary: SpectreInputSummary,
    *,
    lvol: int,
    quadrature_size: int | None = None,
    nt: int | None = None,
    nz: int | None = None,
) -> SpectreMatrixAD:
    """Assemble SPECTRE ``dMA/dMD`` directly from TOML/interface geometry."""

    dof_layout = build_spectre_dof_layout(summary)
    if lvol < 1 or lvol > len(dof_layout.volume_maps):
        raise ValueError(f"lvol={lvol} outside 1..{len(dof_layout.volume_maps)}")
    integrals = assemble_spectre_metric_integrals_from_input(
        summary,
        lvol=lvol,
        quadrature_size=quadrature_size,
        nt=nt,
        nz=nz,
    )
    return assemble_spectre_matrix_ad(dof_layout.volume_maps[lvol - 1], integrals)


def assemble_spectre_volume_matrices_from_input(
    summary: SpectreInputSummary,
    *,
    lvol: int,
    normal_field: SpectreBoundaryNormalField | None = None,
    quadrature_size: int | None = None,
    nt: int | None = None,
    nz: int | None = None,
) -> SpectreVolumeMatrices:
    """Assemble ``dMA/dMD/dMB/dMG`` from a SPECTRE input summary."""

    dof_layout = build_spectre_dof_layout(summary)
    if lvol < 1 or lvol > len(dof_layout.volume_maps):
        raise ValueError(f"lvol={lvol} outside 1..{len(dof_layout.volume_maps)}")
    volume_map = dof_layout.volume_maps[lvol - 1]
    integrals = assemble_spectre_metric_integrals_from_input(
        summary,
        lvol=lvol,
        quadrature_size=quadrature_size,
        nt=nt,
        nz=nz,
    )
    ad = assemble_spectre_matrix_ad(volume_map, integrals)
    field = build_spectre_boundary_normal_field(summary) if normal_field is None else normal_field
    bg = assemble_spectre_matrix_bg(volume_map, field)
    return SpectreVolumeMatrices(d_ma=ad.d_ma, d_md=ad.d_md, d_mb=bg.d_mb, d_mg=bg.d_mg)
