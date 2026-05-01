#!/usr/bin/env python
"""Regenerate SPECTRE backend-seam validation assets.

This script requires a local SPECTRE checkout with the experimental
``beltrami_jax`` seam. It runs SPECTRE's Fortran Beltrami path and the
``force_real(..., beltrami_backend="jax", solve_local_constraints=True)`` path
on compact public compare cases, then writes the static JSON/PNG assets used in
the docs.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CASES = (
    {
        "case": "G3V3L2Fi_stability",
        "input": "tests/compare/G3V3L2Fi_stability/input.toml",
        "branch": "local helicity, Lconstraint=2",
        "superseded_relative_force_error": 1.2510234941523128e-12,
    },
    {
        "case": "G3V3L3Fi",
        "input": "tests/compare/G3V3L3Fi/input.toml",
        "branch": "fixed-boundary global current, Lconstraint=3",
        "superseded_relative_force_error": 1.6675304372497828e-3,
    },
    {
        "case": "G2V32L1Fi",
        "input": "tests/compare/G2V32L1Fi/input.toml",
        "branch": "rotational transform, Lconstraint=1",
        "superseded_relative_force_error": 2.4055140334128033e-2,
    },
)


def _component_relative_error(actual: tuple[np.ndarray, ...], expected) -> float:
    reference = (
        np.asarray(expected.ate, dtype=np.float64),
        np.asarray(expected.aze, dtype=np.float64),
        np.asarray(expected.ato, dtype=np.float64),
        np.asarray(expected.azo, dtype=np.float64),
    )
    errors = []
    for got, ref in zip(actual, reference, strict=True):
        scale = max(float(np.linalg.norm(ref)), 1.0)
        errors.append(float(np.linalg.norm(np.asarray(got) - ref) / scale))
    return max(errors)


def _run_case(case: dict[str, object], *, spectre_root: Path) -> dict[str, object]:
    sys.path.insert(0, str(spectre_root))
    from spectre import SPECTRE, force_real, get_vec_pot_flat, get_xinit_specwrap

    input_path = str(spectre_root / str(case["input"]))
    fortran_obj = SPECTRE.from_input_file(input_path, verbose=False)
    jax_obj = SPECTRE.from_input_file(input_path, verbose=False)
    xin = get_xinit_specwrap(fortran_obj)

    force_fortran = force_real(xin, fortran_obj, beltrami_backend="fortran")
    force_jax = force_real(
        xin,
        jax_obj,
        beltrami_backend="jax",
        solve_local_constraints=True,
        jax_backend_options={"verbose": False},
    )

    difference = np.asarray(force_jax) - np.asarray(force_fortran)
    relative_force_error = float(np.linalg.norm(difference) / max(float(np.linalg.norm(force_fortran)), 1.0))
    solution = getattr(jax_obj, "_beltrami_jax_solution", None)
    if solution is None:
        raise RuntimeError(f"SPECTRE did not store _beltrami_jax_solution for {case['case']}")

    local_constraint_residuals = [
        float(volume.constraint.residual_norm)
        for volume in solution.volume_solves
        if volume.constraint is not None
    ]
    global_constraint_final = (
        float(solution.global_constraint.residual_norm)
        if solution.global_constraint is not None
        else None
    )

    return {
        "case": case["case"],
        "branch": case["branch"],
        "input": str(case["input"]),
        "force_shape": list(np.asarray(force_fortran).shape),
        "force_norm_fortran": float(np.linalg.norm(force_fortran)),
        "force_norm_jax": float(np.linalg.norm(force_jax)),
        "relative_force_error": relative_force_error,
        "max_abs_force_error": float(np.max(np.abs(difference))),
        "injection_relative_error": _component_relative_error(get_vec_pot_flat(jax_obj), solution.vector_potential),
        "max_linear_relative_residual": float(solution.max_relative_residual_norm),
        "max_local_constraint_residual": max(local_constraint_residuals) if local_constraint_residuals else None,
        "global_constraint_initial": (
            float(solution.global_constraint.initial_residual_norm)
            if solution.global_constraint is not None
            else None
        ),
        "global_constraint_final": global_constraint_final,
        "superseded_relative_force_error": case["superseded_relative_force_error"],
    }


def _plot_summary(summary: dict[str, object], output: Path) -> None:
    cases = summary["force_backend_switch"]
    labels = [case["case"] for case in cases]
    force_errors = np.asarray([case["relative_force_error"] for case in cases], dtype=np.float64)
    max_abs = np.asarray([case["max_abs_force_error"] for case in cases], dtype=np.float64)
    linear_residuals = np.asarray([case["max_linear_relative_residual"] for case in cases], dtype=np.float64)
    constraint_residuals = np.asarray(
        [
            case["global_constraint_final"]
            if case["global_constraint_final"] is not None
            else case["max_local_constraint_residual"]
            for case in cases
        ],
        dtype=np.float64,
    )
    superseded = np.asarray([case["superseded_relative_force_error"] for case in cases], dtype=np.float64)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)
    color = "#1f6f5b"
    accent = "#c45a31"
    x = np.arange(len(labels))

    axes[0, 0].bar(x, force_errors, color=color)
    axes[0, 0].axhline(1.0e-12, color="#555555", linestyle="--", linewidth=1.0, label="1e-12")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("SPECTRE force seam")
    axes[0, 0].set_ylabel("relative force error")
    axes[0, 0].set_xticks(x, labels, rotation=18, ha="right")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].bar(x, max_abs, color="#315c8c")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("Pointwise force agreement")
    axes[0, 1].set_ylabel("max absolute force error")
    axes[0, 1].set_xticks(x, labels, rotation=18, ha="right")

    width = 0.36
    axes[1, 0].bar(x - width / 2, linear_residuals, width=width, color="#7a8f2a", label="linear residual")
    axes[1, 0].bar(x + width / 2, np.maximum(constraint_residuals, 1.0e-18), width=width, color="#704c8f", label="constraint residual")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("Internal beltrami_jax closure")
    axes[1, 0].set_ylabel("residual norm")
    axes[1, 0].set_xticks(x, labels, rotation=18, ha="right")
    axes[1, 0].legend(frameon=False)

    axes[1, 1].scatter(superseded, force_errors, s=90, color=accent)
    for label, old, new in zip(labels, superseded, force_errors, strict=True):
        axes[1, 1].annotate(label, (old, new), xytext=(5, 5), textcoords="offset points", fontsize=8)
    lims = [min(np.min(force_errors), np.min(superseded)) * 0.3, max(np.max(force_errors), np.max(superseded)) * 3.0]
    axes[1, 1].plot(lims, lims, color="#555555", linestyle="--", linewidth=1.0)
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlim(lims)
    axes[1, 1].set_ylim(lims)
    axes[1, 1].set_title("Superseded seam errors closed")
    axes[1, 1].set_xlabel("previous relative force error")
    axes[1, 1].set_ylabel("current relative force error")

    fig.suptitle("SPECTRE Fortran force diagnostic after beltrami_jax coefficient injection", fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spectre-root", default="/Users/rogerio/local/spectre")
    parser.add_argument("--output-dir", default="docs/_static")
    args = parser.parse_args()

    spectre_root = Path(args.spectre_root).resolve()
    output_dir = Path(args.output_dir)
    results = [_run_case(case, spectre_root=spectre_root) for case in CASES]
    summary = {
        "generated": date.today().isoformat(),
        "spectre_root": str(spectre_root),
        "beltrami_backend": "force_real(..., beltrami_backend='jax', solve_local_constraints=True)",
        "force_backend_switch": results,
        "max_relative_force_error": max(result["relative_force_error"] for result in results),
        "max_injection_relative_error": max(result["injection_relative_error"] for result in results),
    }
    summary_path = output_dir / "spectre_backend_seam_runtime_summary.json"
    figure_path = output_dir / "spectre_backend_seam_runtime.png"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    _plot_summary(summary, figure_path)
    print(f"wrote {summary_path}")
    print(f"wrote {figure_path}")
    for result in results:
        print(
            f"{result['case']}: force={result['relative_force_error']:.3e} "
            f"injection={result['injection_relative_error']:.3e}"
        )


if __name__ == "__main__":
    main()
