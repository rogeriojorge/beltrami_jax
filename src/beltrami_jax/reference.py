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
    )


def load_spec_text_dump(prefix: str | Path) -> SpecLinearSystemReference:
    """Load a SPEC text dump produced by `tools/build_spec_fixture.py`."""
    prefix = Path(prefix)
    metadata = _parse_metadata(Path(f"{prefix}.meta.txt"))
    d_ma = np.loadtxt(Path(f"{prefix}.dma.txt"))
    d_md = np.loadtxt(Path(f"{prefix}.dmd.txt"))
    d_mb = np.loadtxt(Path(f"{prefix}.dmb.txt"))
    matrix = np.loadtxt(Path(f"{prefix}.matrix.txt"))
    rhs = np.loadtxt(Path(f"{prefix}.rhs.txt"))
    solution = np.loadtxt(Path(f"{prefix}.solution.txt"))

    system = BeltramiLinearSystem.from_arraylike(
        d_ma=d_ma,
        d_md=d_md,
        d_mb=d_mb,
        mu=metadata.mu,
        psi=np.array([metadata.psi_t, metadata.psi_p], dtype=np.float64),
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


def load_packaged_reference(name: str = "g3v01l0fi_lvol1") -> SpecLinearSystemReference:
    """Load the packaged SPEC regression fixture."""
    with resources.as_file(resources.files("beltrami_jax.data").joinpath(f"{name}.npz")) as fixture_path:
        data = np.load(fixture_path)
        system = BeltramiLinearSystem.from_arraylike(
            d_ma=data["d_ma"],
            d_md=data["d_md"],
            d_mb=data["d_mb"],
            mu=data["mu"],
            psi=data["psi"],
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
