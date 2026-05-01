#!/usr/bin/env python
"""Generate SPECTRE Lconstraint=1 rotational-transform validation plots."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from beltrami_jax import (
    compare_vector_potentials,
    compute_spectre_rotational_transform,
    load_packaged_spectre_case,
    load_packaged_spectre_linear_system,
    solve_spectre_volume_from_input,
)

CASE = "G2V32L1Fi"
ROOT = Path(__file__).resolve().parents[1]
FIGURE_PATH = ROOT / "docs" / "_static" / "spectre_lconstraint1_transform.png"
SUMMARY_PATH = ROOT / "docs" / "_static" / "spectre_lconstraint1_transform_summary.json"


def _targets(summary, lvol: int) -> tuple[list[str], list[float]]:
    names: list[str] = []
    values: list[float] = []
    if lvol == 1:
        names.append(f"vol {lvol} outer")
        values.append(float(summary.constraints["iota"][lvol]))
    else:
        names.extend((f"vol {lvol} inner", f"vol {lvol} outer"))
        values.extend(
            (
                float(summary.constraints["oita"][lvol - 1]),
                float(summary.constraints["iota"][lvol]),
            )
        )
    return names, values


print("[beltrami_jax] generating SPECTRE Lconstraint=1 validation assets")
case = load_packaged_spectre_case(CASE)
reference_blocks = case.layout.split_vector_potential(case.reference.vector_potential)

rows = []
labels = []
faces = []
targets = []
computed = []
residuals = []
state_errors = []
coefficient_errors = []

for lvol in range(1, case.input_summary.packed_volume_count + 1):
    print(f"[beltrami_jax] solving {CASE} volume {lvol} with JAX local transform Newton")
    fixture = load_packaged_spectre_linear_system(case_label=case.label, volume_index=lvol)
    result = solve_spectre_volume_from_input(
        case.input_summary,
        lvol=lvol,
        solve_local_constraints=True,
        max_constraint_iterations=32,
    )
    transform = compute_spectre_rotational_transform(
        case.input_summary,
        lvol=lvol,
        vector_potential=result.vector_potential,
        derivative_vector_potentials=result.derivative_vector_potentials,
    )
    target_names, target_values = _targets(case.input_summary, lvol)
    computed_values = [float(transform.iota[1])] if lvol == 1 else [float(transform.iota[0]), float(transform.iota[1])]
    block_comparison = compare_vector_potentials(
        result.vector_potential,
        reference_blocks[lvol - 1],
        label=f"{CASE}/lvol{lvol}",
    )
    mu_error = abs(float(result.mu) - float(fixture.system.mu))
    psi_error = float(np.linalg.norm(np.asarray(result.psi) - np.asarray(fixture.system.psi)))
    state_error = max(mu_error, psi_error)
    residual_norm = float(result.constraint.residual_norm) if result.constraint is not None else 0.0

    labels.extend(target_names)
    faces.extend(("outer",) if lvol == 1 else ("inner", "outer"))
    targets.extend(target_values)
    computed.extend(computed_values)
    residuals.append(max(residual_norm, 1.0e-18))
    state_errors.append(max(state_error, 1.0e-18))
    coefficient_errors.append(max(block_comparison.global_relative_error, 1.0e-18))
    rows.append(
        {
            "case": CASE,
            "lvol": lvol,
            "targets": dict(zip(target_names, target_values)),
            "computed_iota": dict(zip(target_names, computed_values)),
            "local_constraint_residual_norm": residual_norm,
            "mu": float(result.mu),
            "spectre_mu": float(fixture.system.mu),
            "psi": np.asarray(result.psi).tolist(),
            "spectre_psi": np.asarray(fixture.system.psi).tolist(),
            "state_max_abs_error": state_error,
            "coefficient_global_relative_error": block_comparison.global_relative_error,
            "linear_relative_residual_norm": float(result.relative_residual_norm),
        }
    )

targets_np = np.asarray(targets)
computed_np = np.asarray(computed)
max_iota_error = float(np.max(np.abs(computed_np - targets_np)))
max_constraint_residual = float(np.max(residuals))
max_state_error = float(np.max(state_errors))
max_coefficient_error = float(np.max(coefficient_errors))

plt.style.use("seaborn-v0_8-whitegrid")
figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.4), constrained_layout=True)
figure.suptitle(
    f"{CASE}: JAX-native SPECTRE Lconstraint=1 rotational-transform closure",
    fontsize=14,
    fontweight="bold",
)

axis = axes[0, 0]
faces_np = np.asarray(faces)
for face, marker, color in (("inner", "^", "#2A9D8F"), ("outer", "o", "#1B4965")):
    selection = faces_np == face
    if np.any(selection):
        axis.scatter(
            targets_np[selection],
            computed_np[selection],
            s=68,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            label=f"{face} interface",
            zorder=3,
        )
scale = np.asarray([float(np.min(targets_np)) * 0.995, float(np.max(targets_np)) * 1.005])
axis.plot(scale, scale, color="#D1495B", linewidth=1.4, label="exact target")
axis.set_xlabel("SPECTRE target iota")
axis.set_ylabel("JAX computed iota")
axis.set_title(f"Interface transform parity; max |delta iota| = {max_iota_error:.2e}")
axis.legend(frameon=True)

axis = axes[0, 1]
volume_labels = [f"vol {row['lvol']}" for row in rows]
axis.bar(volume_labels, residuals, color="#2A9D8F")
axis.set_yscale("log")
axis.set_ylabel("local residual infinity norm")
axis.set_title(f"Newton closure; worst residual = {max_constraint_residual:.2e}")

axis = axes[1, 0]
axis.bar(volume_labels, state_errors, color="#CA6702")
axis.set_yscale("log")
axis.set_ylabel("max(|delta mu|, ||delta psi||_2)")
axis.set_title(f"Post-constraint state parity; worst = {max_state_error:.2e}")

axis = axes[1, 1]
axis.bar(volume_labels, coefficient_errors, color="#4C956C")
axis.set_yscale("log")
axis.set_ylabel("relative coefficient error")
axis.set_title(f"Ate/Aze/Ato/Azo parity; worst = {max_coefficient_error:.2e}")

summary = {
    "case": CASE,
    "max_iota_abs_error": max_iota_error,
    "max_constraint_residual_norm": max_constraint_residual,
    "max_state_error": max_state_error,
    "max_coefficient_global_relative_error": max_coefficient_error,
    "volumes": rows,
}
FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(FIGURE_PATH, dpi=220)
plt.close(figure)
SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

print(f"[beltrami_jax] max_iota_abs_error={max_iota_error:.3e}")
print(f"[beltrami_jax] max_constraint_residual_norm={max_constraint_residual:.3e}")
print(f"[beltrami_jax] max_state_error={max_state_error:.3e}")
print(f"[beltrami_jax] max_coefficient_global_relative_error={max_coefficient_error:.3e}")
print(f"[beltrami_jax] wrote {FIGURE_PATH}")
print(f"[beltrami_jax] wrote {SUMMARY_PATH}")
