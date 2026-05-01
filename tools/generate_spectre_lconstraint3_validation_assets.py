#!/usr/bin/env python
"""Generate SPECTRE Lconstraint=3 global-current validation plots."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from beltrami_jax import (
    compare_vector_potentials,
    load_packaged_spectre_case,
    load_packaged_spectre_linear_system,
    solve_spectre_volumes_from_input,
    spectre_volume_flux_vector,
)

CASE = "G3V3L3Fi"
ROOT = Path(__file__).resolve().parents[1]
FIGURE_PATH = ROOT / "docs" / "_static" / "spectre_lconstraint3_global.png"
SUMMARY_PATH = ROOT / "docs" / "_static" / "spectre_lconstraint3_global_summary.json"

print("[beltrami_jax] generating SPECTRE Lconstraint=3 validation assets")
case = load_packaged_spectre_case(CASE)

print(f"[beltrami_jax] solving {CASE} from TOML initial state with JAX global constraint")
result = solve_spectre_volumes_from_input(case.input_summary, solve_local_constraints=True, verbose=True)
if result.global_constraint is None:
    raise RuntimeError("expected Lconstraint=3 global constraint record")

comparison = compare_vector_potentials(
    result.vector_potential,
    case.reference.vector_potential,
    label=CASE,
)

volume_labels = [f"vol {solve.lvol}" for solve in result.volume_solves]
initial_dpflux = []
final_dpflux = []
spectre_dpflux = []
mu_errors = []
psi_errors = []
linear_residuals = []
rows = []

for volume_solve in result.volume_solves:
    fixture = load_packaged_spectre_linear_system(case_label=case.label, volume_index=volume_solve.lvol)
    initial_psi = np.asarray(spectre_volume_flux_vector(case.input_summary, lvol=volume_solve.lvol), dtype=np.float64)
    final_psi = np.asarray(volume_solve.psi, dtype=np.float64)
    reference_psi = np.asarray(fixture.system.psi, dtype=np.float64)
    initial_dpflux.append(float(initial_psi[1]))
    final_dpflux.append(float(final_psi[1]))
    spectre_dpflux.append(float(reference_psi[1]))
    mu_error = abs(float(volume_solve.mu) - float(fixture.system.mu))
    psi_error = float(np.linalg.norm(final_psi - reference_psi))
    mu_errors.append(max(mu_error, 1.0e-18))
    psi_errors.append(max(psi_error, 1.0e-18))
    linear_residuals.append(max(float(volume_solve.relative_residual_norm), 1.0e-18))
    rows.append(
        {
            "lvol": volume_solve.lvol,
            "initial_psi": initial_psi.tolist(),
            "jax_psi": final_psi.tolist(),
            "spectre_psi": reference_psi.tolist(),
            "jax_mu": float(volume_solve.mu),
            "spectre_mu": float(fixture.system.mu),
            "mu_abs_error": mu_error,
            "psi_l2_error": psi_error,
            "linear_relative_residual_norm": float(volume_solve.relative_residual_norm),
        }
    )

initial_residual = np.asarray(result.global_constraint.initial_residual, dtype=np.float64)
final_residual = np.asarray(result.global_constraint.final_residual, dtype=np.float64)
correction = np.asarray(result.global_constraint.correction, dtype=np.float64)
max_final_residual = float(np.max(np.abs(final_residual)))
max_state_error = float(max(max(mu_errors), max(psi_errors)))

plt.style.use("seaborn-v0_8-whitegrid")
figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.4), constrained_layout=True)
figure.suptitle(
    f"{CASE}: JAX-native SPECTRE Lconstraint=3 global-current closure",
    fontsize=14,
    fontweight="bold",
)

axis = axes[0, 0]
x = np.arange(len(volume_labels))
width = 0.26
axis.bar(x - width, initial_dpflux, width=width, label="TOML initial", color="#8E9AAF")
axis.bar(x, final_dpflux, width=width, label="JAX global", color="#2A9D8F")
axis.scatter(x + width, spectre_dpflux, marker="x", s=70, label="SPECTRE fixture", color="#D1495B", zorder=3)
axis.set_xticks(x, volume_labels)
axis.set_ylabel("dpflux")
axis.set_title("Global constraint updates the coupled poloidal fluxes")
axis.legend(frameon=True)

axis = axes[0, 1]
constraint_labels = list(result.global_constraint.unknowns)
axis.plot(constraint_labels, np.abs(initial_residual), marker="o", label="before Newton", color="#CA6702")
axis.plot(constraint_labels, np.maximum(np.abs(final_residual), 1.0e-18), marker="s", label="after Newton", color="#2A9D8F")
axis.set_yscale("log")
axis.set_ylabel("constraint absolute residual")
axis.set_title(f"Global residual closure; final max = {max_final_residual:.2e}")
axis.tick_params(axis="x", rotation=20)
axis.legend(frameon=True)

axis = axes[1, 0]
axis.bar(volume_labels, mu_errors, color="#1B4965", label="mu")
axis.bar(volume_labels, psi_errors, bottom=mu_errors, color="#E9C46A", label="psi")
axis.set_yscale("log")
axis.set_ylabel("absolute state error")
axis.set_title(f"Post-constraint state parity; worst = {max_state_error:.2e}")
axis.legend(frameon=True)

axis = axes[1, 1]
bars = {
    "coefficients": max(comparison.global_relative_error, 1.0e-18),
    "linear residual": max(float(np.max(linear_residuals)), 1.0e-18),
}
axis.bar(list(bars), list(bars.values()), color=["#4C956C", "#264653"])
axis.set_yscale("log")
axis.set_ylabel("relative error")
axis.set_title(f"Ate/Aze/Ato/Azo parity = {comparison.global_relative_error:.2e}")

summary = {
    "case": CASE,
    "unknowns": list(result.global_constraint.unknowns),
    "initial_residual": initial_residual.tolist(),
    "final_residual": final_residual.tolist(),
    "correction": correction.tolist(),
    "initial_residual_norm": float(result.global_constraint.initial_residual_norm),
    "final_residual_norm": float(result.global_constraint.residual_norm),
    "coefficient_global_relative_error": comparison.global_relative_error,
    "coefficient_global_max_abs_error": comparison.global_max_abs_error,
    "max_state_error": max_state_error,
    "volumes": rows,
}
FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(FIGURE_PATH, dpi=220)
plt.close(figure)
SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

print(f"[beltrami_jax] initial_residual_norm={summary['initial_residual_norm']:.3e}")
print(f"[beltrami_jax] final_residual_norm={summary['final_residual_norm']:.3e}")
print(f"[beltrami_jax] coefficient_global_relative_error={comparison.global_relative_error:.3e}")
print(f"[beltrami_jax] max_state_error={max_state_error:.3e}")
print(f"[beltrami_jax] wrote {FIGURE_PATH}")
print(f"[beltrami_jax] wrote {SUMMARY_PATH}")
