#!/usr/bin/env python
"""Generate reviewer-facing SPECTRE vector-potential parity plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from beltrami_jax import (
    compare_vector_potentials,
    load_spectre_reference_h5,
    load_spectre_vector_potential_npz,
)
from beltrami_jax.spectre_io import COMPONENT_NAMES

DEFAULT_CASES = (
    "G2V32L1Fi",
    "G3V3L3Fi",
    "G3V3L2Fi_stability",
    "G3V8L3Free",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spectre-root",
        type=Path,
        default=Path("/Users/rogerio/local/spectre"),
        help="Local SPECTRE checkout containing tests/compare references",
    )
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=Path("examples/_generated/spectre_vecpot_exports"),
        help="Directory containing exported SPECTRE candidate .npz files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/_static/spectre_vecpot_parity.png"),
        help="Output plot path",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("docs/_static/spectre_vecpot_parity_summary.json"),
        help="Output JSON summary path",
    )
    return parser.parse_args()


def _flatten(vector_potential):
    return np.concatenate([getattr(vector_potential, name).ravel() for name in COMPONENT_NAMES])


def main() -> None:
    args = parse_args()
    cases = []
    for label in DEFAULT_CASES:
        candidate_path = args.candidate_dir / f"{label}.npz"
        reference_path = args.spectre_root / "tests" / "compare" / label / "reference.h5"
        if candidate_path.exists() and reference_path.exists():
            cases.append((label, candidate_path, reference_path))
        else:
            print(
                f"[spectre-validation] skipping {label}: "
                f"candidate={candidate_path.exists()} reference={reference_path.exists()}"
            )

    if not cases:
        raise FileNotFoundError(
            "No SPECTRE validation cases found. First run tools/export_spectre_vecpot_npz.py "
            "for at least one case, or pass paths matching the default generated directory."
        )

    print(f"[spectre-validation] generating parity panel for {len(cases)} cases")
    rows = []
    candidate_all = []
    reference_all = []
    labels = []
    component_errors = {name: [] for name in COMPONENT_NAMES}
    max_abs_errors = {name: [] for name in COMPONENT_NAMES}

    for label, candidate_path, reference_path in cases:
        candidate = load_spectre_vector_potential_npz(candidate_path)
        reference = load_spectre_reference_h5(reference_path).vector_potential
        comparison = compare_vector_potentials(candidate, reference, label=label)
        rows.append(
            {
                "case": label,
                "candidate": str(candidate_path),
                "reference": str(reference_path),
                **comparison.as_dict(),
            }
        )
        labels.append(label)
        candidate_all.append(_flatten(candidate))
        reference_all.append(_flatten(reference))
        for name in COMPONENT_NAMES:
            component_errors[name].append(comparison.component_relative_errors[name])
            max_abs_errors[name].append(comparison.component_max_abs_errors[name])
        print(
            f"[spectre-validation] {label}: global_rel={comparison.global_relative_error:.3e}, "
            f"global_max_abs={comparison.global_max_abs_error:.3e}"
        )

    candidate_flat = np.concatenate(candidate_all)
    reference_flat = np.concatenate(reference_all)
    difference = candidate_flat - reference_flat
    nonzero = np.maximum(np.abs(candidate_flat), np.abs(reference_flat)) > 0
    parity_scale = float(np.max(np.abs(np.concatenate([candidate_flat, reference_flat]))))
    if parity_scale == 0.0:
        parity_scale = 1.0

    max_global = max(row["global_relative_error"] for row in rows)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.6), constrained_layout=True)
    fig.suptitle(
        "SPECTRE vector-potential coefficient parity\n"
        f"four public compare cases; worst global relative error = {max_global:.2e}",
        fontsize=15,
        fontweight="bold",
    )

    ax = axes[0, 0]
    ax.scatter(reference_flat[nonzero], candidate_flat[nonzero], s=8, alpha=0.5, color="#214E8A")
    diagonal = np.array([-1.05 * parity_scale, 1.05 * parity_scale])
    ax.plot(diagonal, diagonal, color="#D1495B", linewidth=1.5, label="exact parity")
    ax.set_xlim(diagonal)
    ax.set_ylim(diagonal)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("reference HDF5 coefficients")
    ax.set_ylabel("fresh SPECTRE export")
    ax.set_title("Coefficient-by-coefficient parity")
    ax.ticklabel_format(axis="both", style="sci", scilimits=(-2, 2))
    ax.legend(frameon=True)

    ax = axes[0, 1]
    abs_diff = np.abs(difference)
    abs_diff = abs_diff[abs_diff > 0]
    if abs_diff.size:
        lower = max(float(np.min(abs_diff)) * 0.8, 1.0e-19)
        upper = max(float(np.max(abs_diff)) * 1.2, lower * 10.0)
        bins = np.logspace(np.log10(lower), np.log10(upper), 32)
        ax.hist(abs_diff, bins=bins, color="#2A9D8F", edgecolor="white")
        ax.set_xscale("log")
    else:
        ax.bar(["exact"], [1], color="#2A9D8F")
    ax.set_xlabel("|candidate - reference|")
    ax.set_ylabel("coefficient count")
    ax.set_title("Absolute coefficient differences")

    ax = axes[1, 0]
    x = np.arange(len(labels))
    width = 0.18
    for offset, name in enumerate(COMPONENT_NAMES):
        ax.bar(
            x + (offset - 1.5) * width,
            np.maximum(component_errors[name], 1.0e-18),
            width=width,
            label=name.upper(),
        )
    ax.set_yscale("log")
    ax.set_ylim(5.0e-18, 2.0e-13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("relative 2-norm error")
    ax.set_title("Component relative errors")
    ax.legend(ncol=2)

    ax = axes[1, 1]
    for offset, name in enumerate(COMPONENT_NAMES):
        ax.bar(
            x + (offset - 1.5) * width,
            np.maximum(max_abs_errors[name], 1.0e-18),
            width=width,
            label=name.upper(),
        )
    ax.set_yscale("log")
    ax.set_ylim(5.0e-19, 3.0e-15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("max absolute error")
    ax.set_title("Worst coefficient mismatch")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220)
    plt.close(fig)
    args.summary.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"[spectre-validation] wrote {args.output}")
    print(f"[spectre-validation] wrote {args.summary}")


if __name__ == "__main__":
    main()
