#!/usr/bin/env python
"""Generate free-boundary SPECTRE Lconstraint=3 validation plots."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from beltrami_jax import (
    build_spectre_dof_layout,
    compare_vector_potentials,
    load_packaged_spectre_case,
    load_packaged_spectre_linear_system,
    solve_spectre_volumes_from_input,
    spectre_boundary_normal_field_from_dmg,
    spectre_volume_flux_vector,
)

CASE = "G3V8L3Free"
ROOT = Path(__file__).resolve().parents[1]
FIGURE_PATH = ROOT / "docs" / "_static" / "spectre_lconstraint3_free_boundary.png"
SUMMARY_PATH = ROOT / "docs" / "_static" / "spectre_lconstraint3_free_boundary_summary.json"


print("[beltrami_jax] generating free-boundary SPECTRE Lconstraint=3 validation assets")
case = load_packaged_spectre_case(CASE)
layout = build_spectre_dof_layout(case.input_summary)

normal_field_by_volume = {}
for volume_map in layout.volume_maps:
    lvol = volume_map.block.index + 1
    fixture = load_packaged_spectre_linear_system(case_label=CASE, volume_index=lvol)
    normal_field_by_volume[lvol] = spectre_boundary_normal_field_from_dmg(volume_map, fixture.system.d_mg)

print(f"[beltrami_jax] solving {CASE} from TOML/interface geometry with updated normal-field state")
result = solve_spectre_volumes_from_input(
    case.input_summary,
    solve_local_constraints=True,
    normal_field=normal_field_by_volume,
    verbose=True,
)
if result.global_constraint is None:
    raise RuntimeError("expected Lconstraint=3 global constraint record")

comparison = compare_vector_potentials(
    result.vector_potential,
    case.reference.vector_potential,
    label=CASE,
)

initial_residual = np.asarray(result.global_constraint.initial_residual, dtype=np.float64)
final_residual = np.asarray(result.global_constraint.final_residual, dtype=np.float64)
correction = np.asarray(result.global_constraint.correction, dtype=np.float64)
unknowns = list(result.global_constraint.unknowns)
volume_labels = [f"vol {solve.lvol}" for solve in result.volume_solves]

state_rows = []
mu_errors = []
psi_errors = []
linear_residuals = []
initial_dpflux = []
final_dpflux = []
spectre_dpflux = []
initial_dtflux = []
final_dtflux = []
spectre_dtflux = []

for volume_solve in result.volume_solves:
    fixture = load_packaged_spectre_linear_system(case_label=CASE, volume_index=volume_solve.lvol)
    initial_psi = np.asarray(spectre_volume_flux_vector(case.input_summary, lvol=volume_solve.lvol), dtype=np.float64)
    final_psi = np.asarray(volume_solve.psi, dtype=np.float64)
    reference_psi = np.asarray(fixture.system.psi, dtype=np.float64)
    initial_dtflux.append(float(initial_psi[0]))
    final_dtflux.append(float(final_psi[0]))
    spectre_dtflux.append(float(reference_psi[0]))
    initial_dpflux.append(float(initial_psi[1]))
    final_dpflux.append(float(final_psi[1]))
    spectre_dpflux.append(float(reference_psi[1]))
    mu_error = abs(float(volume_solve.mu) - float(fixture.system.mu))
    psi_error = float(np.linalg.norm(final_psi - reference_psi))
    mu_errors.append(max(mu_error, 1.0e-18))
    psi_errors.append(max(psi_error, 1.0e-18))
    linear_residuals.append(max(float(volume_solve.relative_residual_norm), 1.0e-18))
    state_rows.append(
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

component_errors = comparison.component_relative_errors
component_names = list(component_errors)
component_values = [max(component_errors[name], 1.0e-18) for name in component_names]
max_final_residual = float(result.global_constraint.residual_norm)
max_state_error = float(max(max(mu_errors), max(psi_errors)))

plt.style.use("seaborn-v0_8-whitegrid")
figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.7), constrained_layout=True)
figure.suptitle(
    f"{CASE}: free-boundary JAX Beltrami solve with SPECTRE-updated normal-field state",
    fontsize=14,
    fontweight="bold",
)

axis = axes[0, 0]
x = np.arange(len(volume_labels))
axis.plot(x, initial_dpflux, marker="o", color="#8E9AAF", label="initial dpflux")
axis.plot(x, final_dpflux, marker="s", color="#2A9D8F", label="JAX dpflux")
axis.scatter(x, spectre_dpflux, marker="x", s=70, color="#D1495B", label="SPECTRE dpflux", zorder=4)
axis.plot(x, initial_dtflux, marker="o", linestyle="--", color="#A78A7F", label="initial dtflux")
axis.plot(x, final_dtflux, marker="s", linestyle="--", color="#1D4E89", label="JAX dtflux")
axis.scatter(x, spectre_dtflux, marker="+", s=90, color="#6D597A", label="SPECTRE dtflux", zorder=4)
axis.set_xticks(x, volume_labels, rotation=20, ha="right")
axis.set_ylabel("flux increment")
axis.set_title("Free-boundary global unknowns update plasma and vacuum flux")
axis.legend(frameon=True, fontsize=8, ncols=2)

axis = axes[0, 1]
axis.plot(unknowns, np.abs(initial_residual), marker="o", label="before Newton", color="#CA6702")
axis.plot(unknowns, np.maximum(np.abs(final_residual), 1.0e-18), marker="s", label="after Newton", color="#2A9D8F")
axis.set_yscale("log")
axis.set_ylabel("constraint absolute residual")
axis.set_title(f"Global-current residual closure; final max = {max_final_residual:.2e}")
axis.tick_params(axis="x", rotation=25)
axis.legend(frameon=True)

axis = axes[1, 0]
width = 0.36
axis.bar(x - width / 2, mu_errors, width=width, color="#1B4965", label="mu")
axis.bar(x + width / 2, psi_errors, width=width, color="#E9C46A", label="psi")
axis.set_yscale("log")
axis.set_xticks(x, volume_labels, rotation=20, ha="right")
axis.set_ylabel("absolute state error")
axis.set_title(f"Post-correction state parity; worst = {max_state_error:.2e}")
axis.legend(frameon=True)

axis = axes[1, 1]
bars = {
    **dict(zip(component_names, component_values, strict=True)),
    "global": max(comparison.global_relative_error, 1.0e-18),
    "linear residual": max(float(np.max(linear_residuals)), 1.0e-18),
}
axis.bar(list(bars), list(bars.values()), color=["#4C956C", "#5F8D4E", "#88A47C", "#C7BCA1", "#264653", "#315C8C"])
axis.set_yscale("log")
axis.set_ylabel("relative error")
axis.set_title(f"Ate/Aze/Ato/Azo parity = {comparison.global_relative_error:.2e}")
axis.tick_params(axis="x", rotation=20)

summary = {
    "case": CASE,
    "normal_field_source": "released SPECTRE post-Picard dMG fixtures reconstructed as equivalent iV/iB source",
    "unknowns": unknowns,
    "initial_residual": initial_residual.tolist(),
    "final_residual": final_residual.tolist(),
    "correction": correction.tolist(),
    "initial_residual_norm": float(result.global_constraint.initial_residual_norm),
    "final_residual_norm": max_final_residual,
    "coefficient_global_relative_error": comparison.global_relative_error,
    "coefficient_global_max_abs_error": comparison.global_max_abs_error,
    "component_relative_errors": comparison.component_relative_errors,
    "component_max_abs_errors": comparison.component_max_abs_errors,
    "max_state_error": max_state_error,
    "max_linear_relative_residual": float(np.max(linear_residuals)),
    "volumes": state_rows,
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
