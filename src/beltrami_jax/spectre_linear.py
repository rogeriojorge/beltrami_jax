from __future__ import annotations

from dataclasses import dataclass
from importlib import resources

import jax.numpy as jnp
import numpy as np

from .types import BeltramiLinearSystem, SpecLinearSystemReference


DEFAULT_PACKAGED_SPECTRE_LINEAR_SYSTEM = "G2V32L1Fi/lvol1"


@dataclass(frozen=True)
class PackagedSpectreLinearSystem:
    """SPECTRE-exported per-volume Beltrami linear solve fixture.

    The fixture stores the assembled SPECTRE matrices ``dMA``, ``dMD``,
    ``dMB``, and ``dMG``, the dense operator and right-hand side passed to
    SPECTRE's Beltrami solver, and SPECTRE's solved vector-potential degrees of
    freedom for one volume.
    """

    case_label: str
    volume_index: int
    reference: SpecLinearSystemReference
    n_dof: int
    lrad: int
    mn: int
    nvol: int
    mvol: int
    mpol: int
    ntor: int
    nfp: int
    lconstraint: int
    is_vacuum: bool
    coordinate_singularity: bool
    include_d_mg_in_rhs: bool
    iflag: int
    residual_norm: float
    relative_residual_norm: float
    source: str

    @property
    def system(self) -> BeltramiLinearSystem:
        return self.reference.system

    @property
    def matrix(self):
        return self.reference.matrix

    @property
    def rhs(self):
        return self.reference.rhs

    @property
    def expected_solution(self):
        return self.reference.expected_solution

    @property
    def name(self) -> str:
        return f"{self.case_label}/lvol{self.volume_index}"


def _linear_root():
    return resources.files("beltrami_jax.data").joinpath("spectre_linear")


def _parse_system_name(name: str) -> tuple[str, int]:
    if "/" in name:
        case_label, volume_label = name.split("/", maxsplit=1)
    else:
        stem = name.removesuffix(".npz")
        case_label, volume_label = stem.rsplit("_lvol", maxsplit=1)
        volume_label = f"lvol{volume_label}"
    if not volume_label.startswith("lvol"):
        raise ValueError(f"expected volume label like 'lvol1', got {volume_label!r}")
    return case_label, int(volume_label.removeprefix("lvol"))


def list_packaged_spectre_linear_cases() -> tuple[str, ...]:
    """List SPECTRE cases with packaged per-volume linear-system fixtures."""
    root = _linear_root()
    if not root.is_dir():
        return ()
    return tuple(sorted(entry.name for entry in root.iterdir() if entry.is_dir()))


def list_packaged_spectre_linear_systems(case_label: str | None = None) -> tuple[str, ...]:
    """List packaged SPECTRE linear systems as ``case/lvolN`` names."""
    root = _linear_root()
    case_labels = (case_label,) if case_label is not None else list_packaged_spectre_linear_cases()
    names: list[str] = []
    for label in case_labels:
        case_dir = root.joinpath(label)
        if not case_dir.is_dir():
            raise ValueError(f"unknown packaged SPECTRE linear case: {label}")
        entries = [entry for entry in case_dir.iterdir() if entry.name.endswith(".npz")]
        for entry in sorted(entries, key=lambda item: _parse_system_name(item.name)[1]):
            _, volume_index = _parse_system_name(entry.name)
            names.append(f"{label}/lvol{volume_index}")
    return tuple(names)


def load_packaged_spectre_linear_system(
    name: str = DEFAULT_PACKAGED_SPECTRE_LINEAR_SYSTEM,
    *,
    case_label: str | None = None,
    volume_index: int | None = None,
) -> PackagedSpectreLinearSystem:
    """Load a packaged SPECTRE Beltrami linear-system fixture.

    Parameters
    ----------
    name:
        Name returned by :func:`list_packaged_spectre_linear_systems`, for
        example ``"G2V32L1Fi/lvol1"``. A filename stem such as
        ``"G2V32L1Fi_lvol1"`` is also accepted.
    case_label, volume_index:
        Explicit case and one-based volume. If provided, these override
        ``name``.
    """
    if case_label is None or volume_index is None:
        parsed_case_label, parsed_volume_index = _parse_system_name(name)
        case_label = parsed_case_label if case_label is None else case_label
        volume_index = parsed_volume_index if volume_index is None else volume_index

    filename = f"{case_label}_lvol{volume_index}.npz"
    fixture = _linear_root().joinpath(case_label, filename)
    if not fixture.is_file():
        raise ValueError(f"unknown packaged SPECTRE linear system: {case_label}/lvol{volume_index}")

    with resources.as_file(fixture) as fixture_path:
        data = np.load(fixture_path, allow_pickle=False)
        include_d_mg = (
            bool(int(data["include_d_mg_in_rhs"]))
            if "include_d_mg_in_rhs" in data.files
            else bool(int(data["is_vacuum"]))
        )
        label = str(data["label"].item())
        system = BeltramiLinearSystem.from_arraylike(
            d_ma=data["d_ma"],
            d_md=data["d_md"],
            d_mb=data["d_mb"],
            d_mg=data["d_mg"],
            mu=data["mu"],
            psi=data["psi"],
            is_vacuum=bool(int(data["is_vacuum"])),
            include_d_mg_in_rhs=include_d_mg,
            label=label,
        )
        reference = SpecLinearSystemReference(
            system=system,
            matrix=jnp.asarray(data["matrix"], dtype=jnp.float64),
            rhs=jnp.asarray(data["rhs"], dtype=jnp.float64),
            expected_solution=jnp.asarray(data["solution"], dtype=jnp.float64),
            volume_index=int(data["volume_index"]),
            source=str(data["source"].item()),
        )
        return PackagedSpectreLinearSystem(
            case_label=str(data["case_label"].item()),
            volume_index=int(data["volume_index"]),
            reference=reference,
            n_dof=int(data["n_dof"]),
            lrad=int(data["lrad"]),
            mn=int(data["mn"]),
            nvol=int(data["nvol"]),
            mvol=int(data["mvol"]),
            mpol=int(data["mpol"]),
            ntor=int(data["ntor"]),
            nfp=int(data["nfp"]),
            lconstraint=int(data["lconstraint"]),
            is_vacuum=bool(int(data["is_vacuum"])),
            coordinate_singularity=bool(int(data["coordinate_singularity"])),
            include_d_mg_in_rhs=include_d_mg,
            iflag=int(data["iflag"]),
            residual_norm=float(data["residual_norm"]),
            relative_residual_norm=float(data["relative_residual_norm"]),
            source=str(data["source"].item()),
        )


def load_all_packaged_spectre_linear_systems(
    case_label: str | None = None,
) -> tuple[PackagedSpectreLinearSystem, ...]:
    """Load all packaged SPECTRE linear fixtures, optionally for one case."""
    return tuple(
        load_packaged_spectre_linear_system(name)
        for name in list_packaged_spectre_linear_systems(case_label)
    )
