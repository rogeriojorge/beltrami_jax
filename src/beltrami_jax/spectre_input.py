"""SPECTRE TOML input summaries.

This module does not pretend to reproduce SPECTRE's full field solve. It provides
the user-facing input contract needed for that work: load a SPECTRE TOML file,
normalize the Fourier boundary tables, and expose geometry/resolution/constraint
metadata in a form that the future JAX assembly can consume and tests can assert.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

ModeTable = dict[tuple[int, int], float]


@dataclass(frozen=True)
class SpectreInputSummary:
    """Normalized summary of a SPECTRE TOML input file.

    Attributes mirror SPECTRE's top-level TOML sections. The nested Fourier
    tables ``rbc``, ``zbs``, ``rbs``, and ``zbc`` are normalized from string keys
    like ``"(1, 0)"`` to integer mode tuples ``(m, n)``.
    """

    source: str
    physics: Mapping[str, Any]
    numeric: Mapping[str, Any]
    global_options: Mapping[str, Any]
    local: Mapping[str, Any]
    diagnostics: Mapping[str, Any]
    rbc: ModeTable
    zbs: ModeTable
    rbs: ModeTable
    zbc: ModeTable

    @property
    def nvol(self) -> int:
        """Number of relaxed volumes in the SPECTRE input."""

        return int(self.physics.get("nvol", 0))

    @property
    def nfp(self) -> int:
        """Number of field periods."""

        return int(self.physics.get("nfp", 1))

    @property
    def igeometry(self) -> int:
        """SPECTRE geometry flag."""

        return int(self.physics.get("igeometry", 0))

    @property
    def mpol(self) -> int:
        """Maximum poloidal mode in the SPECTRE input."""

        return int(self.physics.get("mpol", 0))

    @property
    def ntor(self) -> int:
        """Maximum toroidal mode in the SPECTRE input."""

        return int(self.physics.get("ntor", 0))

    @property
    def lrad(self) -> tuple[int, ...]:
        """Radial polynomial orders per volume."""

        return tuple(int(value) for value in self.physics.get("lrad", ()))

    @property
    def radial_size(self) -> int:
        """Packed SPECTRE vector-potential radial rows, ``sum(Lrad + 1)``."""

        return sum(value + 1 for value in self.lrad)

    @property
    def packed_volume_count(self) -> int:
        """Number of packed radial blocks represented by ``Lrad``.

        Fixed-boundary cases commonly use ``nvol`` blocks. Free-boundary cases can
        include one additional exterior/vacuum block, so this can be ``nvol + 1``.
        """

        return len(self.lrad)

    @property
    def is_free_boundary(self) -> bool:
        """Whether the input requests SPECTRE's free-boundary workflow."""

        return bool(self.physics.get("lfreebound", False))

    @property
    def free_boundary_iterations(self) -> int:
        """SPECTRE free-boundary update count from the ``[global]`` section."""

        return int(self.global_options.get("mfreeits", 0))

    @property
    def enforce_stellarator_symmetry(self) -> bool:
        """Whether stellarator symmetry is enforced in the SPECTRE input."""

        return bool(self.physics.get("enforce_stell_sym", True))

    @property
    def fluxes(self) -> dict[str, tuple[float, ...]]:
        """Toroidal and poloidal flux arrays by volume/interface."""

        return {
            "tflux": tuple(float(value) for value in self.physics.get("tflux", ())),
            "pflux": tuple(float(value) for value in self.physics.get("pflux", ())),
        }

    @property
    def constraints(self) -> dict[str, object]:
        """Constraint metadata used by the Beltrami solve and outer loop."""

        return {
            "lconstraint": int(self.physics.get("lconstraint", 0)),
            "mu": tuple(float(value) for value in self.physics.get("mu", ())),
            "iota": tuple(float(value) for value in self.physics.get("iota", ())),
            "helicity": tuple(float(value) for value in self.physics.get("helicity", ())),
        }

    def boundary_tables(self) -> dict[str, ModeTable]:
        """Return normalized Fourier boundary coefficient tables."""

        return {"rbc": self.rbc, "zbs": self.zbs, "rbs": self.rbs, "zbc": self.zbc}

    def validate_for_beltrami_contract(self) -> None:
        """Validate fields required by a SPECTRE-compatible Beltrami backend."""

        if self.nvol <= 0:
            raise ValueError("SPECTRE input must define physics.nvol > 0")
        if self.packed_volume_count not in (self.nvol, self.nvol + 1):
            raise ValueError(
                f"physics.lrad length {self.packed_volume_count} must be nvol or nvol + 1; "
                f"nvol={self.nvol}"
            )
        for key in ("tflux", "pflux"):
            values = self.physics.get(key, ())
            if len(values) != self.packed_volume_count:
                raise ValueError(
                    f"physics.{key} length {len(values)} does not match packed volume count "
                    f"{self.packed_volume_count}"
                )
        for key in ("mu",):
            values = self.physics.get(key, ())
            if len(values) != self.nvol:
                raise ValueError(f"physics.{key} length {len(values)} does not match nvol {self.nvol}")
        helicity = self.physics.get("helicity", ())
        if len(helicity) not in (self.nvol, self.packed_volume_count):
            raise ValueError(
                f"physics.helicity length {len(helicity)} must match nvol {self.nvol} "
                f"or packed volume count {self.packed_volume_count}"
            )

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable compact summary."""

        return {
            "source": self.source,
            "igeometry": self.igeometry,
            "nfp": self.nfp,
            "nvol": self.nvol,
            "mpol": self.mpol,
            "ntor": self.ntor,
            "lrad": list(self.lrad),
            "radial_size": self.radial_size,
            "packed_volume_count": self.packed_volume_count,
            "is_free_boundary": self.is_free_boundary,
            "free_boundary_iterations": self.free_boundary_iterations,
            "enforce_stellarator_symmetry": self.enforce_stellarator_symmetry,
            "fluxes": {name: list(values) for name, values in self.fluxes.items()},
            "constraints": {
                name: list(value) if isinstance(value, tuple) else value
                for name, value in self.constraints.items()
            },
            "boundary_mode_counts": {
                name: len(table) for name, table in self.boundary_tables().items()
            },
        }


def _load_toml(path: Path) -> Mapping[str, Any]:
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10 only when tomli is absent
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ModuleNotFoundError as exc:
            raise ImportError(
                "Reading SPECTRE TOML on Python 3.10 requires tomli. Install with "
                "`python -m pip install -e '.[validation]'` or `python -m pip install tomli`."
            ) from exc

    with path.open("rb") as stream:
        return tomllib.load(stream)


def _parse_mode_key(key: str) -> tuple[int, int]:
    stripped = key.strip()
    if not (stripped.startswith("(") and stripped.endswith(")")):
        raise ValueError(f"invalid SPECTRE Fourier mode key: {key!r}")
    values = [part.strip() for part in stripped[1:-1].split(",")]
    if len(values) != 2:
        raise ValueError(f"invalid SPECTRE Fourier mode key: {key!r}")
    return int(values[0]), int(values[1])


def _parse_mode_table(table: Mapping[str, Any] | None) -> ModeTable:
    if not table:
        return {}
    return {_parse_mode_key(key): float(value) for key, value in table.items()}


def load_spectre_input_toml(path: str | Path) -> SpectreInputSummary:
    """Load and normalize a SPECTRE TOML input file."""

    input_path = Path(path)
    data = _load_toml(input_path)
    physics = dict(data.get("physics", {}))
    summary = SpectreInputSummary(
        source=str(input_path),
        physics=physics,
        numeric=dict(data.get("numeric", {})),
        global_options=dict(data.get("global", {})),
        local=dict(data.get("local", {})),
        diagnostics=dict(data.get("diagnostics", {})),
        rbc=_parse_mode_table(physics.get("rbc")),
        zbs=_parse_mode_table(physics.get("zbs")),
        rbs=_parse_mode_table(physics.get("rbs")),
        zbc=_parse_mode_table(physics.get("zbc")),
    )
    summary.validate_for_beltrami_contract()
    return summary
