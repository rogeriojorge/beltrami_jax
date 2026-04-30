"""SPECTRE/SPEC HDF5 vector-potential IO and comparison helpers.

The routines in this module intentionally keep :mod:`h5py` as an optional
dependency. Importing :mod:`beltrami_jax` only requires JAX; reading SPECTRE
HDF5 output requires installing ``beltrami_jax[validation]`` or ``h5py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

COMPONENT_NAMES = ("ate", "aze", "ato", "azo")
H5_COMPONENT_PATHS = {
    "ate": "vector_potential/Ate",
    "aze": "vector_potential/Aze",
    "ato": "vector_potential/Ato",
    "azo": "vector_potential/Azo",
}


@dataclass(frozen=True)
class SpectreVectorPotential:
    """Packed SPECTRE vector-potential coefficient arrays.

    Parameters
    ----------
    ate, aze, ato, azo:
        Arrays with shape ``(sum(Lrad + 1), mn)``. SPECTRE HDF5 files store the
        same data as ``(mn, sum(Lrad + 1))``; :func:`load_spectre_vector_potential_h5`
        transposes by default so Python-side comparisons use the same radial-first
        layout exposed by SPECTRE's ``get_vec_pot_flat`` helper.
    source:
        Human-readable source path or label.
    layout:
        Name of the in-memory layout. The public loaders use ``"radial_mode"``.
    """

    ate: np.ndarray
    aze: np.ndarray
    ato: np.ndarray
    azo: np.ndarray
    source: str = ""
    layout: str = "radial_mode"

    def __post_init__(self) -> None:
        arrays = {}
        for name in COMPONENT_NAMES:
            array = np.asarray(getattr(self, name), dtype=np.float64)
            if array.ndim != 2:
                raise ValueError(f"{name} must be a 2D array, got shape {array.shape}")
            arrays[name] = array
            object.__setattr__(self, name, array)

        shapes = {name: array.shape for name, array in arrays.items()}
        if len(set(shapes.values())) != 1:
            raise ValueError(f"all vector-potential components must have the same shape: {shapes}")

    @property
    def shape(self) -> tuple[int, int]:
        """Common ``(radial_size, mode_count)`` shape for all components."""

        return self.ate.shape

    @property
    def radial_size(self) -> int:
        """Number of packed radial coefficients across all volumes."""

        return self.shape[0]

    @property
    def mode_count(self) -> int:
        """Number of Fourier modes per radial row."""

        return self.shape[1]

    def components(self) -> dict[str, np.ndarray]:
        """Return components as a name-to-array mapping."""

        return {name: getattr(self, name) for name in COMPONENT_NAMES}

    def stack_components(self) -> np.ndarray:
        """Return a stacked array with shape ``(4, radial_size, mode_count)``."""

        return np.stack([getattr(self, name) for name in COMPONENT_NAMES], axis=0)

    def component_norms(self) -> dict[str, float]:
        """Return Euclidean norms for each vector-potential component."""

        return {name: float(np.linalg.norm(array)) for name, array in self.components().items()}

    def split_by_lrad(self, lrad: Iterable[int]) -> tuple["SpectreVectorPotential", ...]:
        """Split packed arrays into one :class:`SpectreVectorPotential` per volume.

        SPECTRE packs ``Lrad[volume] + 1`` radial rows for each volume. This helper
        is useful when a comparison needs to isolate a single plasma or vacuum
        region from the flat HDF5 export.
        """

        lrad_values = [int(value) for value in lrad]
        widths = [value + 1 for value in lrad_values]
        if any(width <= 0 for width in widths):
            raise ValueError(f"all Lrad entries must be non-negative, got {lrad_values}")
        if sum(widths) != self.radial_size:
            raise ValueError(
                "Lrad does not match vector-potential radial size: "
                f"sum(Lrad + 1)={sum(widths)} but radial_size={self.radial_size}"
            )

        volumes: list[SpectreVectorPotential] = []
        start = 0
        for index, width in enumerate(widths, start=1):
            stop = start + width
            volumes.append(
                SpectreVectorPotential(
                    ate=self.ate[start:stop, :],
                    aze=self.aze[start:stop, :],
                    ato=self.ato[start:stop, :],
                    azo=self.azo[start:stop, :],
                    source=f"{self.source}:volume{index}" if self.source else f"volume{index}",
                    layout=self.layout,
                )
            )
            start = stop
        return tuple(volumes)


@dataclass(frozen=True)
class SpectreH5Reference:
    """SPECTRE reference quantities loaded from an HDF5 file."""

    vector_potential: SpectreVectorPotential
    force_final: np.ndarray | None = None
    force_final_grad: np.ndarray | None = None
    source: str = ""


@dataclass(frozen=True)
class SpectreVectorPotentialComparison:
    """Coefficient-level comparison between two SPECTRE vector-potential states."""

    label: str
    shape: tuple[int, int]
    component_relative_errors: dict[str, float]
    component_max_abs_errors: dict[str, float]
    global_relative_error: float
    global_max_abs_error: float

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable comparison summary."""

        return {
            "label": self.label,
            "shape": list(self.shape),
            "component_relative_errors": dict(self.component_relative_errors),
            "component_max_abs_errors": dict(self.component_max_abs_errors),
            "global_relative_error": self.global_relative_error,
            "global_max_abs_error": self.global_max_abs_error,
        }


def _require_h5py():
    try:
        import h5py  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised only without optional dependency
        raise ImportError(
            "Reading SPECTRE HDF5 files requires h5py. Install with "
            "`python -m pip install -e '.[validation]'` or add h5py to the environment."
        ) from exc
    return h5py


def _read_dataset(handle, dataset_path: str, *, required: bool) -> np.ndarray | None:
    if dataset_path not in handle:
        if required:
            raise KeyError(f"missing required HDF5 dataset: {dataset_path}")
        return None
    return np.asarray(handle[dataset_path][()], dtype=np.float64)


def _relative_error(candidate: np.ndarray, reference: np.ndarray) -> float:
    difference_norm = float(np.linalg.norm(candidate - reference))
    reference_norm = float(np.linalg.norm(reference))
    if reference_norm == 0.0:
        return difference_norm
    return difference_norm / reference_norm


def load_spectre_vector_potential_h5(
    path: str | Path,
    *,
    transpose: bool = True,
) -> SpectreVectorPotential:
    """Load ``vector_potential/Ate/Aze/Ato/Azo`` from a SPECTRE HDF5 file.

    Parameters
    ----------
    path:
        SPECTRE HDF5 file path.
    transpose:
        SPECTRE stores HDF5 datasets as ``(mn, radial_size)``. The default
        transposes them to ``(radial_size, mn)`` to match the flat arrays returned
        by SPECTRE's Python wrapper and used by :class:`SpectreVectorPotential`.
    """

    h5py = _require_h5py()
    input_path = Path(path)
    arrays: dict[str, np.ndarray] = {}
    with h5py.File(input_path, "r") as handle:
        for name, dataset_path in H5_COMPONENT_PATHS.items():
            array = _read_dataset(handle, dataset_path, required=True)
            if array is None:  # for type checkers
                raise KeyError(dataset_path)
            arrays[name] = array.T if transpose else array

    return SpectreVectorPotential(source=str(input_path), **arrays)


def load_spectre_reference_h5(
    path: str | Path,
    *,
    transpose_vector_potential: bool = True,
) -> SpectreH5Reference:
    """Load vector-potential coefficients and optional force arrays from SPECTRE HDF5."""

    h5py = _require_h5py()
    input_path = Path(path)
    vector_potential = load_spectre_vector_potential_h5(
        input_path, transpose=transpose_vector_potential
    )
    with h5py.File(input_path, "r") as handle:
        force_final = _read_dataset(handle, "output/force_final", required=False)
        force_final_grad = _read_dataset(handle, "output/force_final_grad", required=False)
    return SpectreH5Reference(
        vector_potential=vector_potential,
        force_final=force_final,
        force_final_grad=force_final_grad,
        source=str(input_path),
    )


def save_spectre_vector_potential_npz(
    path: str | Path,
    vector_potential: SpectreVectorPotential,
    **metadata: object,
) -> None:
    """Write a SPECTRE vector-potential state to a compressed NumPy archive."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "ate": vector_potential.ate,
        "aze": vector_potential.aze,
        "ato": vector_potential.ato,
        "azo": vector_potential.azo,
        "source": np.asarray(vector_potential.source),
        "layout": np.asarray(vector_potential.layout),
    }
    payload.update(metadata)
    np.savez_compressed(output_path, **payload)


def load_spectre_vector_potential_npz(path: str | Path) -> SpectreVectorPotential:
    """Load a vector-potential archive written by :func:`save_spectre_vector_potential_npz`."""

    input_path = Path(path)
    with np.load(input_path, allow_pickle=False) as data:
        source = str(data["source"].item()) if "source" in data else str(input_path)
        layout = str(data["layout"].item()) if "layout" in data else "radial_mode"
        return SpectreVectorPotential(
            ate=data["ate"],
            aze=data["aze"],
            ato=data["ato"],
            azo=data["azo"],
            source=source,
            layout=layout,
        )


def compare_vector_potentials(
    candidate: SpectreVectorPotential,
    reference: SpectreVectorPotential,
    *,
    label: str = "",
) -> SpectreVectorPotentialComparison:
    """Compare two vector-potential states component by component."""

    if candidate.shape != reference.shape:
        raise ValueError(
            "candidate and reference vector potentials have incompatible shapes: "
            f"{candidate.shape} vs {reference.shape}"
        )

    relative_errors: dict[str, float] = {}
    max_abs_errors: dict[str, float] = {}
    for name in COMPONENT_NAMES:
        candidate_array = getattr(candidate, name)
        reference_array = getattr(reference, name)
        relative_errors[name] = _relative_error(candidate_array, reference_array)
        max_abs_errors[name] = float(np.max(np.abs(candidate_array - reference_array)))

    candidate_stack = candidate.stack_components()
    reference_stack = reference.stack_components()
    return SpectreVectorPotentialComparison(
        label=label,
        shape=candidate.shape,
        component_relative_errors=relative_errors,
        component_max_abs_errors=max_abs_errors,
        global_relative_error=_relative_error(candidate_stack, reference_stack),
        global_max_abs_error=float(np.max(np.abs(candidate_stack - reference_stack))),
    )
