"""Packaged SPECTRE compare-case validation fixtures."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from .spectre_input import SpectreInputSummary, load_spectre_input_toml
from .spectre_io import (
    SpectreH5Reference,
    SpectreVectorPotential,
    SpectreVectorPotentialComparison,
    compare_vector_potentials,
    load_spectre_reference_h5,
    load_spectre_vector_potential_npz,
)

PACKAGED_SPECTRE_CASES = (
    "G2V32L1Fi",
    "G3V3L3Fi",
    "G3V3L2Fi_stability",
    "G3V8L3Free",
)


@dataclass(frozen=True)
class PackagedSpectreCase:
    """A loaded public SPECTRE compare case packaged with ``beltrami_jax``."""

    label: str
    input_summary: SpectreInputSummary
    reference: SpectreH5Reference
    candidate: SpectreVectorPotential
    comparison: SpectreVectorPotentialComparison

    @property
    def vector_potential_shape(self) -> tuple[int, int]:
        """Common ``(radial_size, mode_count)`` coefficient shape for this case."""

        return self.reference.vector_potential.shape


def list_packaged_spectre_cases() -> tuple[str, ...]:
    """Return labels for the packaged SPECTRE compare cases."""

    return PACKAGED_SPECTRE_CASES


def _case_resource(label: str):
    if label not in PACKAGED_SPECTRE_CASES:
        known = ", ".join(PACKAGED_SPECTRE_CASES)
        raise KeyError(f"unknown packaged SPECTRE case {label!r}; known cases: {known}")
    return resources.files("beltrami_jax.data").joinpath("spectre_compare", label)


def packaged_spectre_case_paths(label: str) -> dict[str, Path]:
    """Return real filesystem paths for a packaged SPECTRE case.

    Wheels are normally unpacked on disk, but this helper still uses
    :func:`importlib.resources.as_file` internally in the higher-level loader so
    zipped import contexts remain safe.
    """

    root = _case_resource(label)
    return {
        "input_toml": Path(str(root.joinpath("input.toml"))),
        "reference_h5": Path(str(root.joinpath("reference.h5"))),
        "candidate_npz": Path(str(root.joinpath("fresh_spectre_export.npz"))),
    }


def load_packaged_spectre_case(label: str) -> PackagedSpectreCase:
    """Load a packaged SPECTRE input, reference HDF5, and fresh coefficient export."""

    root = _case_resource(label)
    with ExitStack() as stack:
        input_toml = stack.enter_context(resources.as_file(root.joinpath("input.toml")))
        reference_h5 = stack.enter_context(resources.as_file(root.joinpath("reference.h5")))
        candidate_npz = stack.enter_context(resources.as_file(root.joinpath("fresh_spectre_export.npz")))
        input_summary = load_spectre_input_toml(input_toml)
        reference = load_spectre_reference_h5(reference_h5)
        candidate = load_spectre_vector_potential_npz(candidate_npz)

    comparison = compare_vector_potentials(candidate, reference.vector_potential, label=label)
    return PackagedSpectreCase(
        label=label,
        input_summary=input_summary,
        reference=reference,
        candidate=candidate,
        comparison=comparison,
    )


def load_all_packaged_spectre_cases() -> tuple[PackagedSpectreCase, ...]:
    """Load every packaged SPECTRE compare case."""

    return tuple(load_packaged_spectre_case(label) for label in PACKAGED_SPECTRE_CASES)
