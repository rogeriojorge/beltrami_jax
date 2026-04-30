#!/usr/bin/env python
"""Generate reviewer-facing SPECTRE linear-solve parity plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from beltrami_jax import (
    assemble_operator,
    assemble_rhs,
    load_all_packaged_spectre_linear_systems,
    solve_from_components,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/_static/spectre_linear_parity.png"),
        help="Output plot path",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("docs/_static/spectre_linear_parity_summary.json"),
        help="Output JSON summary path",
    )
    return parser.parse_args()


def _relative_norm(actual: np.ndarray, expected: np.ndarray, *, matrix: bool = False) -> float:
    ord_value = "fro" if matrix else 2
    denominator = float(np.linalg.norm(expected, ord=ord_value))
    numerator = float(np.linalg.norm(actual - expected, ord=ord_value))
    return numerator if denominator == 0.0 else numerator / denominator


def main() -> None:
    args = parse_args()
    fixtures = load_all_packaged_spectre_linear_systems()
    rows = []
    expected_all = []
    solved_all = []

    for fixture in fixtures:
        result = solve_from_components(fixture.system)
        operator = np.asarray(assemble_operator(fixture.system))
        rhs = np.asarray(assemble_rhs(fixture.system))
        solved = np.asarray(result.solution)
        expected = np.asarray(fixture.expected_solution)
        row = {
            "case": fixture.case_label,
            "name": fixture.name,
            "volume_index": fixture.volume_index,
            "n_dof": fixture.n_dof,
            "is_vacuum": fixture.is_vacuum,
            "coordinate_singularity": fixture.coordinate_singularity,
            "operator_relative_error": _relative_norm(operator, np.asarray(fixture.matrix), matrix=True),
            "rhs_relative_error": _relative_norm(rhs, np.asarray(fixture.rhs)),
            "solution_relative_error": _relative_norm(solved, expected),
            "max_abs_solution_error": float(np.max(np.abs(solved - expected))),
            "jax_relative_residual_norm": float(result.relative_residual_norm),
            "spectre_relative_residual_norm": fixture.relative_residual_norm,
        }
        rows.append(row)
        expected_all.append(expected)
        solved_all.append(solved)
        print(
            f"[spectre-linear-validation] {fixture.name}: "
            f"solution_rel={row['solution_relative_error']:.3e}, "
            f"residual={row['jax_relative_residual_norm']:.3e}"
        )

    expected_flat = np.concatenate(expected_all)
    solved_flat = np.concatenate(solved_all)
    difference = solved_flat - expected_flat
    max_solution_relative_error = max(row["solution_relative_error"] for row in rows)
    max_residual = max(row["jax_relative_residual_norm"] for row in rows)
    max_scale = float(np.max(np.abs(np.concatenate([expected_flat, solved_flat]))))
    if max_scale == 0.0:
        max_scale = 1.0

    labels = [row["name"].replace("/", "\n") for row in rows]
    x = np.arange(len(rows))
    solution_errors = np.asarray([max(row["solution_relative_error"], 1e-18) for row in rows])
    residuals = np.asarray([max(row["jax_relative_residual_norm"], 1e-18) for row in rows])
    sizes = np.asarray([row["n_dof"] for row in rows])
    colors = ["#1B4965" if not row["is_vacuum"] else "#CA6702" for row in rows]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.8), constrained_layout=True)
    fig.suptitle(
        "SPECTRE Beltrami linear-system parity\n"
        f"19 released-case volume solves; worst solution relative error = {max_solution_relative_error:.2e}",
        fontsize=15,
        fontweight="bold",
    )

    ax = axes[0, 0]
    ax.scatter(expected_flat, solved_flat, s=9, alpha=0.55, color="#214E8A", linewidths=0)
    diagonal = np.array([-1.05 * max_scale, 1.05 * max_scale])
    ax.plot(diagonal, diagonal, color="#D1495B", linewidth=1.5, label="exact parity")
    ax.set_xlim(diagonal)
    ax.set_ylim(diagonal)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("SPECTRE solved degrees of freedom")
    ax.set_ylabel("JAX solved degrees of freedom")
    ax.set_title("Solution-vector parity")
    ax.ticklabel_format(axis="both", style="sci", scilimits=(-2, 2))
    ax.legend(frameon=True)

    ax = axes[0, 1]
    abs_diff = np.abs(difference)
    abs_diff = abs_diff[abs_diff > 0]
    if abs_diff.size:
        lower = max(float(np.min(abs_diff)) * 0.8, 1.0e-19)
        upper = max(float(np.max(abs_diff)) * 1.2, lower * 10.0)
        bins = np.logspace(np.log10(lower), np.log10(upper), 36)
        ax.hist(abs_diff, bins=bins, color="#2A9D8F", edgecolor="white")
        ax.set_xscale("log")
    else:
        ax.bar(["exact"], [1], color="#2A9D8F")
    ax.set_xlabel("|JAX solution - SPECTRE solution|")
    ax.set_ylabel("degree-of-freedom count")
    ax.set_title("Absolute solution differences")

    ax = axes[1, 0]
    ax.bar(x, solution_errors, color=colors)
    ax.axhline(1.0e-12, color="#D1495B", linewidth=1.2, linestyle="--", label="1e-12")
    ax.set_yscale("log")
    ax.set_ylim(5.0e-18, 5.0e-12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("relative 2-norm error")
    ax.set_title("Per-volume solution agreement")
    ax.legend(frameon=True)

    ax = axes[1, 1]
    ax.scatter(sizes, residuals, c=colors, s=64, alpha=0.9)
    ax.axhline(max_residual, color="#5F0F40", linewidth=1.1, linestyle=":", label=f"max = {max_residual:.1e}")
    ax.set_yscale("log")
    ax.set_xlabel("active SPECTRE degrees of freedom")
    ax.set_ylabel("relative residual norm")
    ax.set_title("JAX solve residuals")
    ax.legend(frameon=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220)
    plt.close(fig)
    args.summary.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"[spectre-linear-validation] wrote {args.output}")
    print(f"[spectre-linear-validation] wrote {args.summary}")
    print(f"[spectre-linear-validation] worst solution relative error: {max_solution_relative_error:.3e}")
    print(f"[spectre-linear-validation] worst JAX residual: {max_residual:.3e}")


if __name__ == "__main__":
    main()
