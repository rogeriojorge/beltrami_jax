"""SPECTRE Beltrami coefficient layout helpers."""

from __future__ import annotations

from dataclasses import dataclass

from .spectre_input import SpectreInputSummary
from .spectre_io import SpectreVectorPotential


@dataclass(frozen=True)
class SpectreVolumeBlock:
    """One packed radial block in a SPECTRE vector-potential coefficient array."""

    index: int
    label: str
    lrad: int
    start: int
    stop: int
    is_exterior: bool = False

    @property
    def width(self) -> int:
        """Number of packed radial rows in this block."""

        return self.stop - self.start

    @property
    def radial_slice(self) -> slice:
        """Slice selecting this block from a radial-first coefficient array."""

        return slice(self.start, self.stop)

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable layout summary."""

        return {
            "index": self.index,
            "label": self.label,
            "lrad": self.lrad,
            "start": self.start,
            "stop": self.stop,
            "width": self.width,
            "is_exterior": self.is_exterior,
        }


@dataclass(frozen=True)
class SpectreBeltramiLayout:
    """Packed SPECTRE vector-potential layout for one input file."""

    input_summary: SpectreInputSummary
    mode_count: int
    blocks: tuple[SpectreVolumeBlock, ...]

    @property
    def radial_size(self) -> int:
        """Total packed radial rows across all blocks."""

        return sum(block.width for block in self.blocks)

    @property
    def shape(self) -> tuple[int, int]:
        """Expected vector-potential shape ``(radial_size, mode_count)``."""

        return self.radial_size, self.mode_count

    @property
    def plasma_blocks(self) -> tuple[SpectreVolumeBlock, ...]:
        """Blocks corresponding to physical relaxed volumes."""

        return tuple(block for block in self.blocks if not block.is_exterior)

    @property
    def exterior_block(self) -> SpectreVolumeBlock | None:
        """Optional free-boundary exterior/vacuum block."""

        exterior = tuple(block for block in self.blocks if block.is_exterior)
        if not exterior:
            return None
        if len(exterior) != 1:
            raise ValueError(f"expected at most one exterior block, found {len(exterior)}")
        return exterior[0]

    def validate_vector_potential(self, vector_potential: SpectreVectorPotential) -> None:
        """Validate that a vector-potential state matches this layout."""

        if vector_potential.shape != self.shape:
            raise ValueError(
                "vector-potential shape does not match SPECTRE layout: "
                f"{vector_potential.shape} vs {self.shape}"
            )

    def split_vector_potential(
        self, vector_potential: SpectreVectorPotential
    ) -> tuple[SpectreVectorPotential, ...]:
        """Split a vector-potential state according to this layout."""

        self.validate_vector_potential(vector_potential)
        pieces: list[SpectreVectorPotential] = []
        for block in self.blocks:
            radial_slice = block.radial_slice
            source = (
                f"{vector_potential.source}:{block.label}"
                if vector_potential.source
                else block.label
            )
            pieces.append(
                SpectreVectorPotential(
                    ate=vector_potential.ate[radial_slice, :],
                    aze=vector_potential.aze[radial_slice, :],
                    ato=vector_potential.ato[radial_slice, :],
                    azo=vector_potential.azo[radial_slice, :],
                    source=source,
                    layout=vector_potential.layout,
                )
            )
        return tuple(pieces)

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable layout summary."""

        return {
            "source": self.input_summary.source,
            "nvol": self.input_summary.nvol,
            "packed_volume_count": self.input_summary.packed_volume_count,
            "mode_count": self.mode_count,
            "radial_size": self.radial_size,
            "shape": list(self.shape),
            "blocks": [block.as_dict() for block in self.blocks],
        }


def build_spectre_beltrami_layout(
    input_summary: SpectreInputSummary,
    *,
    mode_count: int,
) -> SpectreBeltramiLayout:
    """Build the packed vector-potential layout implied by a SPECTRE input."""

    if mode_count <= 0:
        raise ValueError(f"mode_count must be positive, got {mode_count}")

    blocks: list[SpectreVolumeBlock] = []
    start = 0
    for index, lrad in enumerate(input_summary.lrad):
        width = lrad + 1
        stop = start + width
        is_exterior = index >= input_summary.nvol
        label = "exterior" if is_exterior else f"volume_{index + 1}"
        blocks.append(
            SpectreVolumeBlock(
                index=index,
                label=label,
                lrad=lrad,
                start=start,
                stop=stop,
                is_exterior=is_exterior,
            )
        )
        start = stop

    layout = SpectreBeltramiLayout(
        input_summary=input_summary,
        mode_count=int(mode_count),
        blocks=tuple(blocks),
    )
    if layout.radial_size != input_summary.radial_size:
        raise ValueError(
            f"layout radial size {layout.radial_size} does not match input summary "
            f"{input_summary.radial_size}"
        )
    return layout


def build_spectre_beltrami_layout_for_vector_potential(
    input_summary: SpectreInputSummary,
    vector_potential: SpectreVectorPotential,
) -> SpectreBeltramiLayout:
    """Build and validate a SPECTRE layout from an existing vector-potential state."""

    layout = build_spectre_beltrami_layout(input_summary, mode_count=vector_potential.mode_count)
    layout.validate_vector_potential(vector_potential)
    return layout
