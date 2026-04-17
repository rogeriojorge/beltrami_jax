from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from .types import BeltramiLinearSystem, SpecLinearSystemReference


@dataclass(frozen=True)
class SpecDumpMetadata:
    volume_index: int
    size: int
    mu: float
    psi_t: float
    psi_p: float
    is_vacuum: bool


def _parse_metadata(path: Path) -> SpecDumpMetadata:
    values: dict[str, float] = {}
    for line in path.read_text().splitlines():
        key, value = line.split(maxsplit=1)
        values[key] = float(value)
    return SpecDumpMetadata(
        volume_index=int(values["lvol"]),
        size=int(values["nn"]),
        mu=float(values["mu"]),
        psi_t=float(values["psi_t"]),
        psi_p=float(values["psi_p"]),
        is_vacuum=bool(int(values.get("is_vacuum", 0))),
    )


def load_spec_text_dump(prefix: str | Path) -> SpecLinearSystemReference:
    """Load a SPEC text dump produced by `tools/build_spec_fixture.py`."""
    prefix = Path(prefix)
    metadata = _parse_metadata(Path(f"{prefix}.meta.txt"))
    d_mg_path = Path(f"{prefix}.dmg.txt")
    d_ma = np.loadtxt(Path(f"{prefix}.dma.txt"))
    d_md = np.loadtxt(Path(f"{prefix}.dmd.txt"))
    d_mb = np.loadtxt(Path(f"{prefix}.dmb.txt"))
    d_mg = np.loadtxt(d_mg_path) if d_mg_path.exists() else None
    matrix = np.loadtxt(Path(f"{prefix}.matrix.txt"))
    rhs = np.loadtxt(Path(f"{prefix}.rhs.txt"))
    solution = np.loadtxt(Path(f"{prefix}.solution.txt"))
    if metadata.is_vacuum and d_mg is None:
        raise FileNotFoundError(f"Vacuum SPEC dump is missing {d_mg_path.name}")

    system = BeltramiLinearSystem.from_arraylike(
        d_ma=d_ma,
        d_md=d_md,
        d_mb=d_mb,
        d_mg=d_mg,
        mu=metadata.mu,
        psi=np.array([metadata.psi_t, metadata.psi_p], dtype=np.float64),
        is_vacuum=metadata.is_vacuum,
        label=f"SPEC lvol={metadata.volume_index}",
    )
    return SpecLinearSystemReference(
        system=system,
        matrix=jnp.asarray(matrix, dtype=jnp.float64),
        rhs=jnp.asarray(rhs, dtype=jnp.float64),
        expected_solution=jnp.asarray(solution, dtype=jnp.float64),
        volume_index=metadata.volume_index,
        source=str(prefix),
    )


DEFAULT_PACKAGED_REFERENCE = "g3v01l0fi_lvol1"


def list_packaged_references() -> tuple[str, ...]:
    """List packaged SPEC regression fixtures bundled with beltrami_jax."""
    package_dir = resources.files("beltrami_jax.data")
    return tuple(sorted(entry.name[:-4] for entry in package_dir.iterdir() if entry.name.endswith(".npz")))


def load_packaged_reference(name: str = DEFAULT_PACKAGED_REFERENCE) -> SpecLinearSystemReference:
    """Load the packaged SPEC regression fixture."""
    with resources.as_file(resources.files("beltrami_jax.data").joinpath(f"{name}.npz")) as fixture_path:
        data = np.load(fixture_path)
        d_mg = data["d_mg"] if "d_mg" in data.files else None
        is_vacuum = bool(int(data["is_vacuum"])) if "is_vacuum" in data.files else False
        system = BeltramiLinearSystem.from_arraylike(
            d_ma=data["d_ma"],
            d_md=data["d_md"],
            d_mb=data["d_mb"],
            d_mg=d_mg,
            mu=data["mu"],
            psi=data["psi"],
            is_vacuum=is_vacuum,
            label=str(data["label"].item()),
        )
        return SpecLinearSystemReference(
            system=system,
            matrix=jnp.asarray(data["matrix"], dtype=jnp.float64),
            rhs=jnp.asarray(data["rhs"], dtype=jnp.float64),
            expected_solution=jnp.asarray(data["solution"], dtype=jnp.float64),
            volume_index=int(data["volume_index"]),
            source=str(data["source"].item()),
        )
