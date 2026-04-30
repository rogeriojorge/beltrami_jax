"""User-facing SPECTRE TOML-to-vector-potential solve helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .spectre_backend import SpectreBackendSolve, solve_spectre_assembled
from .spectre_input import SpectreInputSummary, load_spectre_input_toml
from .spectre_io import SpectreVectorPotential
from .spectre_matrix import SpectreBoundaryNormalField
from .spectre_pack import build_spectre_dof_layout
from .spectre_volume_matrix import SpectreVolumeMatrices, assemble_spectre_volume_matrices_from_input

_PI2 = 2.0 * np.pi


@dataclass(frozen=True)
class SpectreVolumeSolve:
    """One SPECTRE-compatible volume solve assembled and solved in JAX."""

    lvol: int
    matrices: SpectreVolumeMatrices
    solve: SpectreBackendSolve
    vector_potential: SpectreVectorPotential
    mu: Array
    psi: Array
    is_vacuum: bool
    include_d_mg_in_rhs: bool

    @property
    def solution(self) -> Array:
        """Packed SPECTRE solution vector for this volume."""

        return self.solve.solution

    @property
    def residual_norm(self) -> Array:
        """Absolute residual norm of the assembled linear solve."""

        return self.solve.residual_norm

    @property
    def relative_residual_norm(self) -> Array:
        """Relative residual norm of the assembled linear solve."""

        return self.solve.relative_residual_norm


@dataclass(frozen=True)
class SpectreMultiVolumeSolve:
    """All packed SPECTRE volumes solved from one input summary.

    This is the highest-level SPECTRE-facing linear entry point currently
    provided by :mod:`beltrami_jax`: it starts from SPECTRE TOML/interface
    geometry, assembles the local Beltrami matrices in JAX, solves each packed
    volume, and concatenates the unpacked SPECTRE vector-potential coefficient
    blocks into full ``Ate/Aze/Ato/Azo`` arrays.
    """

    summary: SpectreInputSummary
    volume_solves: tuple[SpectreVolumeSolve, ...]
    vector_potential: SpectreVectorPotential

    @property
    def residual_norms(self) -> Array:
        """Absolute residual norms for all packed volume solves."""

        return jnp.asarray([solve.residual_norm for solve in self.volume_solves], dtype=jnp.float64)

    @property
    def relative_residual_norms(self) -> Array:
        """Relative residual norms for all packed volume solves."""

        return jnp.asarray([solve.relative_residual_norm for solve in self.volume_solves], dtype=jnp.float64)

    @property
    def max_relative_residual_norm(self) -> Array:
        """Maximum relative residual norm across all packed volumes."""

        if not self.volume_solves:
            return jnp.asarray(0.0, dtype=jnp.float64)
        return jnp.max(self.relative_residual_norms)

    def component_norms(self) -> dict[str, float]:
        """Return Euclidean norms for the concatenated vector-potential arrays."""

        return self.vector_potential.component_norms()


def spectre_normalized_fluxes(summary: SpectreInputSummary) -> tuple[np.ndarray, np.ndarray]:
    """Return SPECTRE-normalized cumulative toroidal and poloidal flux arrays.

    SPECTRE normalizes cumulative fluxes by ``tflux[Nvol]`` during input
    processing, then converts per-volume flux differences to physical units by
    multiplying by ``phiedge / (2*pi)``.
    """

    tflux = np.asarray(summary.fluxes["tflux"], dtype=np.float64)
    pflux = np.asarray(summary.fluxes["pflux"], dtype=np.float64)
    if tflux.size != summary.packed_volume_count or pflux.size != summary.packed_volume_count:
        raise ValueError("tflux and pflux must match the packed SPECTRE volume count")
    reference = tflux[summary.nvol - 1]
    if reference == 0.0:
        raise ValueError("SPECTRE tflux(Nvol) must be nonzero")
    tflux = tflux / reference
    pflux = pflux / reference
    if int(summary.constraints["lconstraint"]) == 3 and summary.igeometry == 1:
        pflux = pflux.copy()
        pflux[-1] = 0.0
    return tflux, pflux


def spectre_volume_flux_vector(summary: SpectreInputSummary, *, lvol: int) -> Array:
    """Return SPECTRE's two-component ``(dtflux, dpflux)`` vector for a volume."""

    if lvol < 1 or lvol > summary.packed_volume_count:
        raise ValueError(f"lvol={lvol} outside 1..{summary.packed_volume_count}")
    tflux, pflux = spectre_normalized_fluxes(summary)
    index = lvol - 1
    if index == 0:
        dtflux = tflux[0]
        dpflux = pflux[0] if summary.igeometry == 1 else 0.0
    else:
        dtflux = tflux[index] - tflux[index - 1]
        dpflux = pflux[index] - pflux[index - 1]
    scale = float(summary.physics.get("phiedge", 1.0)) / _PI2
    return jnp.asarray([dtflux * scale, dpflux * scale], dtype=jnp.float64)


def _volume_mu(summary: SpectreInputSummary, *, lvol: int, mu: ArrayLike | None) -> Array:
    if mu is not None:
        return jnp.asarray(mu, dtype=jnp.float64)
    if lvol > summary.nvol:
        return jnp.asarray(0.0, dtype=jnp.float64)
    values = summary.constraints["mu"]
    if lvol > len(values):
        raise ValueError(f"mu for lvol={lvol} is not present in SPECTRE input")
    return jnp.asarray(values[lvol - 1], dtype=jnp.float64)


def solve_spectre_volume_from_input(
    summary: SpectreInputSummary,
    *,
    lvol: int,
    mu: ArrayLike | None = None,
    psi: ArrayLike | None = None,
    normal_field: SpectreBoundaryNormalField | None = None,
    quadrature_size: int | None = None,
    nt: int | None = None,
    nz: int | None = None,
    verbose: bool = False,
) -> SpectreVolumeSolve:
    """Assemble, solve, and unpack one SPECTRE Beltrami volume from TOML data.

    By default, ``mu`` and ``psi`` are taken from the normalized SPECTRE input
    state. Pass explicit values when validating against a post-constraint
    SPECTRE state whose local nonlinear solve has updated ``mu`` or the flux
    increments.
    """

    dof_layout = build_spectre_dof_layout(summary)
    if lvol < 1 or lvol > len(dof_layout.volume_maps):
        raise ValueError(f"lvol={lvol} outside 1..{len(dof_layout.volume_maps)}")
    volume_map = dof_layout.volume_maps[lvol - 1]
    matrices = assemble_spectre_volume_matrices_from_input(
        summary,
        lvol=lvol,
        normal_field=normal_field,
        quadrature_size=quadrature_size,
        nt=nt,
        nz=nz,
    )
    mu_value = _volume_mu(summary, lvol=lvol, mu=mu)
    psi_value = spectre_volume_flux_vector(summary, lvol=lvol) if psi is None else jnp.asarray(psi, dtype=jnp.float64)
    lconstraint = int(summary.constraints["lconstraint"])
    include_d_mg_in_rhs = bool(volume_map.vacuum_region or (volume_map.coordinate_singularity and lconstraint == -2))
    if verbose:
        print(
            "[beltrami_jax] SPECTRE volume solve "
            f"lvol={lvol} size={volume_map.solution_size} vacuum={volume_map.vacuum_region} "
            f"mu={float(mu_value):.12e} psi=({float(psi_value[0]):.12e}, {float(psi_value[1]):.12e})"
        )
    solve = solve_spectre_assembled(
        d_ma=matrices.d_ma,
        d_md=matrices.d_md,
        d_mb=matrices.d_mb,
        d_mg=matrices.d_mg,
        mu=mu_value,
        psi=psi_value,
        is_vacuum=volume_map.vacuum_region,
        include_d_mg_in_rhs=include_d_mg_in_rhs,
    )
    vector_potential = volume_map.unpack_solution(np.asarray(solve.solution), source=f"{summary.source}:lvol{lvol}")
    if verbose:
        print(
            "[beltrami_jax] SPECTRE volume solve complete "
            f"residual={float(solve.residual_norm):.3e} relative={float(solve.relative_residual_norm):.3e}"
        )
    return SpectreVolumeSolve(
        lvol=lvol,
        matrices=matrices,
        solve=solve,
        vector_potential=vector_potential,
        mu=mu_value,
        psi=psi_value,
        is_vacuum=volume_map.vacuum_region,
        include_d_mg_in_rhs=include_d_mg_in_rhs,
    )


def _lookup_volume_override(
    values: Mapping[int, ArrayLike] | ArrayLike | None,
    *,
    lvol: int,
) -> ArrayLike | None:
    if values is None:
        return None
    if isinstance(values, Mapping):
        return values.get(lvol)
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        return array
    if lvol > array.shape[0]:
        return None
    return array[lvol - 1]


def _lookup_normal_field(
    values: Mapping[int, SpectreBoundaryNormalField] | SpectreBoundaryNormalField | None,
    *,
    lvol: int,
) -> SpectreBoundaryNormalField | None:
    if values is None:
        return None
    if isinstance(values, Mapping):
        return values.get(lvol)
    return values


def _concat_vector_potentials(
    volume_solves: tuple[SpectreVolumeSolve, ...],
    *,
    source: str,
) -> SpectreVectorPotential:
    if not volume_solves:
        empty = np.zeros((0, 0), dtype=np.float64)
        return SpectreVectorPotential(ate=empty, aze=empty, ato=empty, azo=empty, source=source)
    return SpectreVectorPotential(
        ate=np.concatenate([solve.vector_potential.ate for solve in volume_solves], axis=0),
        aze=np.concatenate([solve.vector_potential.aze for solve in volume_solves], axis=0),
        ato=np.concatenate([solve.vector_potential.ato for solve in volume_solves], axis=0),
        azo=np.concatenate([solve.vector_potential.azo for solve in volume_solves], axis=0),
        source=source,
    )


def solve_spectre_volumes_from_input(
    summary: SpectreInputSummary,
    *,
    volumes: tuple[int, ...] | None = None,
    mu: Mapping[int, ArrayLike] | ArrayLike | None = None,
    psi: Mapping[int, ArrayLike] | ArrayLike | None = None,
    normal_field: Mapping[int, SpectreBoundaryNormalField] | SpectreBoundaryNormalField | None = None,
    quadrature_size: int | None = None,
    nt: int | None = None,
    nz: int | None = None,
    verbose: bool = False,
) -> SpectreMultiVolumeSolve:
    """Solve all selected SPECTRE volumes from TOML/interface geometry.

    Parameters
    ----------
    summary:
        Parsed SPECTRE TOML input summary.
    volumes:
        Optional one-based packed-volume indices. The default solves every
        packed radial block, including a free-boundary vacuum exterior block
        when present in ``physics.lrad``.
    mu, psi:
        Optional per-volume overrides. These may be dictionaries keyed by
        one-based volume, arrays indexed in packed-volume order, or scalars.
        Overrides are needed for exact comparison to a post-constraint SPECTRE
        state until the transform/current nonlinear update loop is fully
        ported.
    normal_field:
        Optional SPECTRE boundary-normal-field source. A single value applies
        to every volume; a dictionary may provide one source per volume.
    """

    selected = tuple(range(1, summary.packed_volume_count + 1)) if volumes is None else tuple(int(v) for v in volumes)
    if any(lvol < 1 or lvol > summary.packed_volume_count for lvol in selected):
        raise ValueError(f"volumes must be within 1..{summary.packed_volume_count}; got {selected}")
    volume_solves: list[SpectreVolumeSolve] = []
    if verbose:
        print(
            "[beltrami_jax] SPECTRE multi-volume solve "
            f"source={summary.source} volumes={selected} geometry={summary.igeometry}"
        )
    for lvol in selected:
        volume_solves.append(
            solve_spectre_volume_from_input(
                summary,
                lvol=lvol,
                mu=_lookup_volume_override(mu, lvol=lvol),
                psi=_lookup_volume_override(psi, lvol=lvol),
                normal_field=_lookup_normal_field(normal_field, lvol=lvol),
                quadrature_size=quadrature_size,
                nt=nt,
                nz=nz,
                verbose=verbose,
            )
        )
    solves_tuple = tuple(volume_solves)
    vector_potential = _concat_vector_potentials(solves_tuple, source=summary.source)
    if verbose:
        print(
            "[beltrami_jax] SPECTRE multi-volume solve complete "
            f"max_relative_residual={float(jnp.max(jnp.asarray([s.relative_residual_norm for s in solves_tuple]))):.3e}"
            if solves_tuple
            else "[beltrami_jax] SPECTRE multi-volume solve complete no volumes"
        )
    return SpectreMultiVolumeSolve(
        summary=summary,
        volume_solves=solves_tuple,
        vector_potential=vector_potential,
    )


def solve_spectre_toml(
    path: str | Path,
    **kwargs,
) -> SpectreMultiVolumeSolve:
    """Load a SPECTRE TOML file and solve selected packed volumes in JAX."""

    return solve_spectre_volumes_from_input(load_spectre_input_toml(path), **kwargs)
