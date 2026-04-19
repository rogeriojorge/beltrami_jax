from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from beltrami_jax import (
    compare_against_reference,
    compute_solve_diagnostics,
    gmres_solve,
    load_packaged_reference,
    solve_from_components,
)


FIXTURE_NAME = "g1v03l0fi_lvol2"
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "examples" / "_generated" / "solve_spec_fixture"
SUMMARY_PATH = OUTPUT_DIR / f"{FIXTURE_NAME}_summary.json"
FIGURE_PATH = OUTPUT_DIR / f"{FIXTURE_NAME}_panel.png"


def _to_builtin(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
reference = load_packaged_reference(FIXTURE_NAME)
dense_result = solve_from_components(reference.system, method="dense", verbose=True)
gmres_result = solve_from_components(reference.system, method="gmres", tolerance=1.0e-12, max_iterations=reference.system.size, verbose=True)
matrix_free_gmres = gmres_solve(lambda vector: dense_result.operator @ vector, dense_result.rhs, tolerance=1.0e-12, max_iterations=reference.system.size)

dense_diagnostics = compute_solve_diagnostics(dense_result, include_condition_number=True)
dense_comparison = compare_against_reference(reference, dense_result)
gmres_comparison = compare_against_reference(
    reference,
    solve_from_components(reference.system, method="gmres", tolerance=1.0e-12, max_iterations=reference.system.size),
)

summary = {
    "fixture_name": FIXTURE_NAME,
    "volume_index": reference.volume_index,
    "size": reference.system.size,
    "dense_diagnostics": {key: _to_builtin(value) for key, value in dense_diagnostics.__dict__.items()},
    "dense_comparison": {key: _to_builtin(value) for key, value in dense_comparison.__dict__.items()},
    "gmres_comparison": {key: _to_builtin(value) for key, value in gmres_comparison.__dict__.items()},
    "matrix_free_gmres_iterations": matrix_free_gmres.iterations,
    "matrix_free_gmres_relative_residual_norm": float(matrix_free_gmres.relative_residual_norm),
}
SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")

coefficients_dense = np.asarray(dense_result.solution)
coefficients_spec = np.asarray(reference.expected_solution)
coefficients_gmres = np.asarray(gmres_result.solution)
coefficient_error = np.abs(coefficients_dense - coefficients_spec)

figure, axes = plt.subplots(1, 3, figsize=(14, 4.2))
axes[0].semilogy(np.abs(coefficients_spec) + 1.0e-18, label="SPEC", linewidth=2.0)
axes[0].semilogy(np.abs(coefficients_dense) + 1.0e-18, "--", label="dense JAX")
axes[0].semilogy(np.abs(coefficients_gmres) + 1.0e-18, ":", label="GMRES")
axes[0].set_xlabel("packed coefficient")
axes[0].set_ylabel(r"$|a_i|$")
axes[0].set_title("Coefficient spectrum")
axes[0].legend()

axes[1].semilogy(coefficient_error + 1.0e-22, color="tab:red")
axes[1].set_xlabel("packed coefficient")
axes[1].set_ylabel(r"$|a_i^{\mathrm{JAX}}-a_i^{\mathrm{SPEC}}|$")
axes[1].set_title("Dense agreement error")

axes[2].bar(
    ["operator", "rhs", "solution", "gmres"],
    [
        dense_comparison.operator_relative_error,
        dense_comparison.rhs_relative_error,
        dense_comparison.solution_relative_error,
        gmres_comparison.solution_relative_error,
    ],
    color=["#4c72b0", "#55a868", "#c44e52", "#8172b2"],
)
axes[2].set_yscale("log")
axes[2].set_ylabel("relative error")
axes[2].set_title("SPEC agreement metrics")

figure.suptitle(f"SPEC validation workflow: {FIXTURE_NAME}", fontsize=13)
figure.tight_layout()
figure.savefig(FIGURE_PATH, dpi=200)
plt.close(figure)

print(f"[beltrami_jax] fixture={FIXTURE_NAME} size={reference.system.size} lvol={reference.volume_index}")
print(f"[beltrami_jax] dense_solution_relative_error={dense_comparison.solution_relative_error:.8e}")
print(f"[beltrami_jax] gmres_solution_relative_error={gmres_comparison.solution_relative_error:.8e}")
print(f"[beltrami_jax] matrix_free_gmres_iterations={matrix_free_gmres.iterations}")
print(f"[beltrami_jax] wrote {SUMMARY_PATH}")
print(f"[beltrami_jax] wrote {FIGURE_PATH}")
