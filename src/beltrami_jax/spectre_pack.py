"""SPECTRE-compatible Beltrami degree-of-freedom packing.

This module mirrors the public parts of SPECTRE's ``gi00ab``,
``lregion``, ``initialize_internal_arrays``, and ``packab`` logic.  It does
not assemble the Beltrami matrices; it defines the exact integer maps that
move between per-volume solution vectors and the packed ``Ate/Aze/Ato/Azo``
coefficient arrays used by SPECTRE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import jax.numpy as jnp
import numpy as np

from .spectre_input import SpectreInputSummary
from .spectre_io import COMPONENT_NAMES, SpectreVectorPotential
from .spectre_layout import (
    SpectreBeltramiLayout,
    SpectreVolumeBlock,
    build_spectre_beltrami_layout,
    build_spectre_beltrami_layout_for_vector_potential,
)


COEFFICIENT_MAP_NAMES = COMPONENT_NAMES
LAGRANGE_MAP_NAMES = ("lma", "lmb", "lmc", "lmd", "lme", "lmf", "lmg", "lmh")


def spectre_mode_count(input_summary: SpectreInputSummary) -> int:
    """Return SPECTRE's internal Fourier mode count ``mn``.

    SPECTRE computes this as ``1 + Ntor + Mpol * (2 * Ntor + 1)`` before
    constructing both interface-geometry and vector-potential modes.
    """

    return 1 + input_summary.ntor + input_summary.mpol * (2 * input_summary.ntor + 1)


def spectre_fourier_modes(input_summary: SpectreInputSummary) -> tuple[tuple[int, int], ...]:
    """Return SPECTRE ``(m, n)`` Fourier modes in internal ``gi00ab`` order.

    The returned toroidal mode includes SPECTRE's field-period factor, matching
    the Fortran ``in`` array. For example, an input with ``nfp=5`` and logical
    toroidal mode ``n=2`` is represented internally as ``in=10``.
    """

    modes: list[tuple[int, int]] = []
    for n in range(input_summary.ntor + 1):
        modes.append((0, n * input_summary.nfp))
    for m in range(1, input_summary.mpol + 1):
        for n in range(-input_summary.ntor, input_summary.ntor + 1):
            modes.append((m, n * input_summary.nfp))
    return tuple(modes)


def spectre_region_flags(
    input_summary: SpectreInputSummary,
    block: SpectreVolumeBlock,
) -> dict[str, bool]:
    """Return SPECTRE ``lregion`` flags for one packed volume block."""

    coordinate_singularity = input_summary.igeometry != 1 and block.index == 0
    plasma_region = block.index < input_summary.nvol
    return {
        "coordinate_singularity": coordinate_singularity,
        "plasma_region": plasma_region,
        "vacuum_region": not plasma_region,
    }


@dataclass(frozen=True)
class SpectreVolumeDofMap:
    """Integer index maps for one SPECTRE Beltrami volume.

    The coefficient maps have shape ``(Lrad + 1, mn)`` and store SPECTRE's
    one-based solution-vector ids. A zero entry means the corresponding
    coefficient is constrained to zero by symmetry, gauge, or coordinate-axis
    regularity. Lagrange-multiplier maps have shape ``(mn,)``.
    """

    block: SpectreVolumeBlock
    poloidal_modes: np.ndarray
    toroidal_modes: np.ndarray
    ate: np.ndarray
    aze: np.ndarray
    ato: np.ndarray
    azo: np.ndarray
    lma: np.ndarray
    lmb: np.ndarray
    lmc: np.ndarray
    lmd: np.ndarray
    lme: np.ndarray
    lmf: np.ndarray
    lmg: np.ndarray
    lmh: np.ndarray
    coordinate_singularity: bool
    plasma_region: bool
    enforce_stellarator_symmetry: bool

    @property
    def mode_count(self) -> int:
        """Number of Fourier modes in this volume."""

        return int(self.poloidal_modes.size)

    @property
    def radial_size(self) -> int:
        """Number of radial rows in this volume."""

        return self.block.width

    @property
    def vacuum_region(self) -> bool:
        """Whether this block is SPECTRE's vacuum/exterior region."""

        return not self.plasma_region

    @property
    def solution_size(self) -> int:
        """Length of the SPECTRE solution vector for this volume."""

        maximum = 0
        for array in (*self.coefficient_maps().values(), *self.lagrange_multiplier_maps().values()):
            if array.size:
                maximum = max(maximum, int(np.max(array)))
        return maximum

    @property
    def coefficient_dof_count(self) -> int:
        """Number of coefficient entries represented in the solution vector."""

        return sum(int(np.count_nonzero(array)) for array in self.coefficient_maps().values())

    @property
    def lagrange_multiplier_count(self) -> int:
        """Number of SPECTRE Lagrange-multiplier entries for this volume."""

        return sum(int(np.count_nonzero(array)) for array in self.lagrange_multiplier_maps().values())

    def coefficient_maps(self) -> dict[str, np.ndarray]:
        """Return ``Ate/Aze/Ato/Azo`` index maps."""

        return {name: getattr(self, name) for name in COEFFICIENT_MAP_NAMES}

    def lagrange_multiplier_maps(self) -> dict[str, np.ndarray]:
        """Return ``Lma`` through ``Lmh`` index maps."""

        return {name: getattr(self, name) for name in LAGRANGE_MAP_NAMES}

    def positive_ids(self) -> np.ndarray:
        """Return all positive ids used by coefficients and multipliers."""

        arrays = [array.ravel() for array in self.coefficient_maps().values()]
        arrays.extend(array.ravel() for array in self.lagrange_multiplier_maps().values())
        if not arrays:
            return np.zeros((0,), dtype=np.int64)
        ids = np.concatenate(arrays)
        return ids[ids > 0]

    def validate_contiguous_ids(self) -> None:
        """Validate that SPECTRE ids are unique and contiguous from one."""

        ids = np.sort(self.positive_ids())
        expected = np.arange(1, self.solution_size + 1, dtype=ids.dtype)
        if ids.size != expected.size or not np.array_equal(ids, expected):
            raise ValueError(
                f"{self.block.label} has non-contiguous SPECTRE ids: "
                f"found {ids.tolist()}, expected {expected.tolist()}"
            )

    def validate_vector_potential(self, vector_potential: SpectreVectorPotential) -> None:
        """Validate that a per-volume vector-potential block matches this map."""

        expected_shape = (self.radial_size, self.mode_count)
        if vector_potential.shape != expected_shape:
            raise ValueError(
                f"{self.block.label} vector-potential shape {vector_potential.shape} "
                f"does not match expected {expected_shape}"
            )

    def pack_vector_potential(self, vector_potential: SpectreVectorPotential) -> np.ndarray:
        """Pack one per-volume coefficient block into a SPECTRE solution vector.

        Lagrange multiplier entries are initialized to zero. This mirrors
        ``packab('P', ...)``, which only packs vector-potential coefficients.
        """

        self.validate_vector_potential(vector_potential)
        solution = np.zeros((self.solution_size,), dtype=np.float64)
        for name, index_map in self.coefficient_maps().items():
            _pack_component_numpy(solution, getattr(vector_potential, name), index_map)
        return solution

    def unpack_solution(
        self,
        solution: np.ndarray,
        *,
        source: str = "",
    ) -> SpectreVectorPotential:
        """Unpack a SPECTRE solution vector into one coefficient block."""

        vector = np.asarray(solution, dtype=np.float64)
        if vector.shape != (self.solution_size,):
            raise ValueError(
                f"{self.block.label} solution shape {vector.shape} does not match "
                f"expected {(self.solution_size,)}"
            )
        arrays = {
            name: _unpack_component_numpy(vector, index_map)
            for name, index_map in self.coefficient_maps().items()
        }
        return SpectreVectorPotential(
            **arrays,
            source=source or self.block.label,
            layout="spectre_volume",
        )

    def pack_vector_potential_jax(self, components: Mapping[str, object]) -> jnp.ndarray:
        """Differentiably pack one coefficient block using JAX scatter operations."""

        solution = jnp.zeros((self.solution_size,), dtype=jnp.float64)
        for name, index_map in self.coefficient_maps().items():
            component = jnp.asarray(components[name], dtype=jnp.float64)
            if component.shape != (self.radial_size, self.mode_count):
                raise ValueError(
                    f"{self.block.label} component {name} has shape {component.shape}; "
                    f"expected {(self.radial_size, self.mode_count)}"
                )
            solution = _pack_component_jax(solution, component, index_map)
        return solution

    def unpack_solution_jax(self, solution: object) -> dict[str, jnp.ndarray]:
        """Differentiably unpack one SPECTRE solution vector using JAX gather operations."""

        vector = jnp.asarray(solution, dtype=jnp.float64)
        if vector.shape != (self.solution_size,):
            raise ValueError(
                f"{self.block.label} solution shape {vector.shape} does not match "
                f"expected {(self.solution_size,)}"
            )
        return {
            name: _unpack_component_jax(vector, index_map)
            for name, index_map in self.coefficient_maps().items()
        }

    def as_dict(self) -> dict[str, object]:
        """Return a compact JSON-serializable map summary."""

        return {
            "block": self.block.as_dict(),
            "coordinate_singularity": self.coordinate_singularity,
            "plasma_region": self.plasma_region,
            "vacuum_region": self.vacuum_region,
            "enforce_stellarator_symmetry": self.enforce_stellarator_symmetry,
            "solution_size": self.solution_size,
            "coefficient_dof_count": self.coefficient_dof_count,
            "lagrange_multiplier_count": self.lagrange_multiplier_count,
        }


@dataclass(frozen=True)
class SpectreBeltramiDofLayout:
    """SPECTRE Beltrami solution-vector layout across all packed volumes."""

    layout: SpectreBeltramiLayout
    poloidal_modes: np.ndarray
    toroidal_modes: np.ndarray
    volume_maps: tuple[SpectreVolumeDofMap, ...]

    @property
    def mode_count(self) -> int:
        """Number of Fourier modes."""

        return int(self.poloidal_modes.size)

    @property
    def solution_sizes(self) -> tuple[int, ...]:
        """Per-volume SPECTRE solution-vector lengths."""

        return tuple(volume_map.solution_size for volume_map in self.volume_maps)

    @property
    def total_solution_size(self) -> int:
        """Sum of per-volume SPECTRE solution-vector lengths."""

        return sum(self.solution_sizes)

    def validate_contiguous_ids(self) -> None:
        """Validate every per-volume id map."""

        for volume_map in self.volume_maps:
            volume_map.validate_contiguous_ids()

    def pack_vector_potential(self, vector_potential: SpectreVectorPotential) -> tuple[np.ndarray, ...]:
        """Pack a full SPECTRE vector potential into per-volume solution vectors."""

        self.layout.validate_vector_potential(vector_potential)
        blocks = self.layout.split_vector_potential(vector_potential)
        return tuple(
            volume_map.pack_vector_potential(block)
            for volume_map, block in zip(self.volume_maps, blocks, strict=True)
        )

    def unpack_solutions(
        self,
        solutions: tuple[np.ndarray, ...] | list[np.ndarray],
        *,
        source: str = "",
    ) -> SpectreVectorPotential:
        """Unpack per-volume solution vectors into a full vector-potential state."""

        if len(solutions) != len(self.volume_maps):
            raise ValueError(
                f"expected {len(self.volume_maps)} solution vectors, got {len(solutions)}"
            )
        pieces = [
            volume_map.unpack_solution(solution, source=source or volume_map.block.label)
            for volume_map, solution in zip(self.volume_maps, solutions, strict=True)
        ]
        arrays = {
            name: np.concatenate([getattr(piece, name) for piece in pieces], axis=0)
            for name in COEFFICIENT_MAP_NAMES
        }
        return SpectreVectorPotential(
            **arrays,
            source=source or self.layout.input_summary.source,
            layout="spectre_packed",
        )

    def pack_vector_potential_jax(self, components: Mapping[str, object]) -> tuple[jnp.ndarray, ...]:
        """Differentiably pack full coefficient arrays into per-volume solutions."""

        for name in COEFFICIENT_MAP_NAMES:
            component = jnp.asarray(components[name], dtype=jnp.float64)
            if component.shape != self.layout.shape:
                raise ValueError(
                    f"component {name} has shape {component.shape}; expected {self.layout.shape}"
                )

        packed: list[jnp.ndarray] = []
        for volume_map in self.volume_maps:
            block_components = {
                name: jnp.asarray(components[name], dtype=jnp.float64)[volume_map.block.radial_slice, :]
                for name in COEFFICIENT_MAP_NAMES
            }
            packed.append(volume_map.pack_vector_potential_jax(block_components))
        return tuple(packed)

    def unpack_solutions_jax(self, solutions: tuple[object, ...] | list[object]) -> dict[str, jnp.ndarray]:
        """Differentiably unpack per-volume solutions into full coefficient arrays."""

        if len(solutions) != len(self.volume_maps):
            raise ValueError(
                f"expected {len(self.volume_maps)} solution vectors, got {len(solutions)}"
            )
        pieces = [
            volume_map.unpack_solution_jax(solution)
            for volume_map, solution in zip(self.volume_maps, solutions, strict=True)
        ]
        return {
            name: jnp.concatenate([piece[name] for piece in pieces], axis=0)
            for name in COEFFICIENT_MAP_NAMES
        }

    def as_dict(self) -> dict[str, object]:
        """Return a compact JSON-serializable layout summary."""

        return {
            "layout": self.layout.as_dict(),
            "modes": [
                {"m": int(m), "n": int(n)}
                for m, n in zip(self.poloidal_modes, self.toroidal_modes, strict=True)
            ],
            "solution_sizes": list(self.solution_sizes),
            "total_solution_size": self.total_solution_size,
            "volume_maps": [volume_map.as_dict() for volume_map in self.volume_maps],
        }


def build_spectre_dof_layout(
    input_summary: SpectreInputSummary,
    *,
    mode_count: int | None = None,
) -> SpectreBeltramiDofLayout:
    """Build SPECTRE-compatible coefficient and multiplier id maps."""

    modes = spectre_fourier_modes(input_summary)
    expected_mode_count = len(modes)
    if mode_count is None:
        mode_count = expected_mode_count
    if int(mode_count) != expected_mode_count:
        raise ValueError(
            f"mode_count {mode_count} does not match SPECTRE mode count "
            f"{expected_mode_count} from mpol={input_summary.mpol}, ntor={input_summary.ntor}"
        )

    layout = build_spectre_beltrami_layout(input_summary, mode_count=mode_count)
    poloidal_modes = np.asarray([mode[0] for mode in modes], dtype=np.int64)
    toroidal_modes = np.asarray([mode[1] for mode in modes], dtype=np.int64)
    volume_maps = tuple(
        _build_volume_dof_map(
            input_summary,
            block,
            poloidal_modes=poloidal_modes,
            toroidal_modes=toroidal_modes,
        )
        for block in layout.blocks
    )
    dof_layout = SpectreBeltramiDofLayout(
        layout=layout,
        poloidal_modes=poloidal_modes,
        toroidal_modes=toroidal_modes,
        volume_maps=volume_maps,
    )
    dof_layout.validate_contiguous_ids()
    return dof_layout


def build_spectre_dof_layout_for_vector_potential(
    input_summary: SpectreInputSummary,
    vector_potential: SpectreVectorPotential,
) -> SpectreBeltramiDofLayout:
    """Build and validate SPECTRE id maps against an existing coefficient state."""

    layout = build_spectre_beltrami_layout_for_vector_potential(input_summary, vector_potential)
    dof_layout = build_spectre_dof_layout(
        input_summary,
        mode_count=vector_potential.mode_count,
    )
    if dof_layout.layout.shape != layout.shape:
        raise ValueError("internal SPECTRE layout mismatch")
    return dof_layout


def _build_volume_dof_map(
    input_summary: SpectreInputSummary,
    block: SpectreVolumeBlock,
    *,
    poloidal_modes: np.ndarray,
    toroidal_modes: np.ndarray,
) -> SpectreVolumeDofMap:
    radial_shape = (block.width, poloidal_modes.size)
    mode_shape = (poloidal_modes.size,)
    coefficient_maps = {
        name: np.zeros(radial_shape, dtype=np.int64) for name in COEFFICIENT_MAP_NAMES
    }
    lagrange_maps = {
        name: np.zeros(mode_shape, dtype=np.int64) for name in LAGRANGE_MAP_NAMES
    }

    flags = spectre_region_flags(input_summary, block)
    enforce_symmetry = input_summary.enforce_stellarator_symmetry
    idof = 0

    if flags["coordinate_singularity"]:
        for mode_index, m in enumerate(poloidal_modes):
            for ll in range(block.lrad + 1):
                zernike_active = ll >= m and (m + ll) % 2 == 0
                if not zernike_active:
                    continue
                recombined_axis_basis = (ll == 0 and m == 0) or (ll == 1 and m == 1)
                if not recombined_axis_basis:
                    idof += 1
                    coefficient_maps["ate"][ll, mode_index] = idof
                idof += 1
                coefficient_maps["aze"][ll, mode_index] = idof
                if not enforce_symmetry and mode_index > 0:
                    if not recombined_axis_basis:
                        idof += 1
                        coefficient_maps["ato"][ll, mode_index] = idof
                    idof += 1
                    coefficient_maps["azo"][ll, mode_index] = idof

        for mode_index, m in enumerate(poloidal_modes):
            if m not in (0, 1):
                idof += 1
                lagrange_maps["lma"][mode_index] = idof
            if m == 0:
                idof += 1
                lagrange_maps["lmb"][mode_index] = idof
            if mode_index > 0:
                idof += 1
                lagrange_maps["lme"][mode_index] = idof
            if mode_index == 0:
                idof += 1
                lagrange_maps["lmg"][mode_index] = idof
            if not enforce_symmetry:
                if m not in (0, 1):
                    idof += 1
                    lagrange_maps["lmc"][mode_index] = idof
                if mode_index > 0:
                    idof += 1
                    lagrange_maps["lmf"][mode_index] = idof
                if mode_index > 0 and m == 0:
                    idof += 1
                    lagrange_maps["lmd"][mode_index] = idof
    else:
        for mode_index in range(poloidal_modes.size):
            for ll in range(1, block.lrad + 1):
                idof += 1
                coefficient_maps["ate"][ll, mode_index] = idof
                idof += 1
                coefficient_maps["aze"][ll, mode_index] = idof
                if mode_index > 0 and not enforce_symmetry:
                    idof += 1
                    coefficient_maps["ato"][ll, mode_index] = idof
                    idof += 1
                    coefficient_maps["azo"][ll, mode_index] = idof

        for mode_index in range(poloidal_modes.size):
            if mode_index > 0:
                idof += 1
                lagrange_maps["lme"][mode_index] = idof
            if mode_index > 0 and not enforce_symmetry:
                idof += 1
                lagrange_maps["lmf"][mode_index] = idof
            if mode_index == 0:
                idof += 1
                lagrange_maps["lmg"][mode_index] = idof
                idof += 1
                lagrange_maps["lmh"][mode_index] = idof

    return SpectreVolumeDofMap(
        block=block,
        poloidal_modes=poloidal_modes,
        toroidal_modes=toroidal_modes,
        coordinate_singularity=flags["coordinate_singularity"],
        plasma_region=flags["plasma_region"],
        enforce_stellarator_symmetry=enforce_symmetry,
        **coefficient_maps,
        **lagrange_maps,
    )


def _pack_component_numpy(solution: np.ndarray, component: np.ndarray, index_map: np.ndarray) -> None:
    ids = index_map.ravel()
    values = np.asarray(component, dtype=np.float64).ravel()
    mask = ids > 0
    solution[ids[mask] - 1] = values[mask]


def _unpack_component_numpy(solution: np.ndarray, index_map: np.ndarray) -> np.ndarray:
    output = np.zeros(index_map.shape, dtype=np.float64)
    mask = index_map > 0
    output[mask] = solution[index_map[mask] - 1]
    return output


def _pack_component_jax(solution: jnp.ndarray, component: jnp.ndarray, index_map: np.ndarray) -> jnp.ndarray:
    ids = jnp.asarray(index_map.ravel(), dtype=jnp.int32) - 1
    values = component.reshape((-1,))
    mask = ids >= 0
    safe_ids = jnp.where(mask, ids, 0)
    safe_values = jnp.where(mask, values, 0.0)
    return solution.at[safe_ids].add(safe_values)


def _unpack_component_jax(solution: jnp.ndarray, index_map: np.ndarray) -> jnp.ndarray:
    ids = jnp.asarray(index_map, dtype=jnp.int32) - 1
    mask = ids >= 0
    return jnp.where(mask, solution[jnp.maximum(ids, 0)], 0.0)
