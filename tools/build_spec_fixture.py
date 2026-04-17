from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from beltrami_jax.reference import load_spec_text_dump


def build_fixture(prefix: Path, output: Path) -> None:
    reference = load_spec_text_dump(prefix)
    output.parent.mkdir(parents=True, exist_ok=True)
    fixture_kwargs = dict(
        d_ma=np.asarray(reference.system.d_ma),
        d_md=np.asarray(reference.system.d_md),
        d_mb=np.asarray(reference.system.d_mb),
        mu=np.asarray(reference.system.mu),
        psi=np.asarray(reference.system.psi),
        matrix=np.asarray(reference.matrix),
        rhs=np.asarray(reference.rhs),
        solution=np.asarray(reference.expected_solution),
        volume_index=np.asarray(reference.volume_index),
        label=np.asarray(reference.system.label),
        source=np.asarray(reference.source),
    )
    if reference.system.d_mg is not None:
        fixture_kwargs["d_mg"] = np.asarray(reference.system.d_mg)
    fixture_kwargs["is_vacuum"] = np.asarray(int(reference.system.is_vacuum), dtype=np.int8)
    np.savez_compressed(output, **fixture_kwargs)
    print(f"Wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a SPEC text dump into a compressed beltrami_jax fixture.")
    parser.add_argument("prefix", type=Path, help="Dump prefix, e.g. /path/to/G3V01L0Fi.dump.lvol1")
    parser.add_argument("output", type=Path, help="Output .npz path")
    args = parser.parse_args()
    build_fixture(args.prefix, args.output)


if __name__ == "__main__":
    main()
