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
    save_spectre_vector_potential_npz,
    solve_spectre_volumes_from_input,
)


CASE = "G3V3L3Fi"
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "examples" / "_generated" / "spectre_toml_full_solve"
SUMMARY_PATH = OUTPUT_DIR / "spectre_toml_full_solve_summary.json"
COEFFICIENT_PATH = OUTPUT_DIR / "jax_vector_potential.npz"
FIGURE_PATH = OUTPUT_DIR / "spectre_toml_full_solve.png"
STATIC_FIGURE_PATH = ROOT / "docs" / "_static" / "spectre_toml_full_solve.png"
STATIC_SUMMARY_PATH = ROOT / "docs" / "_static" / "spectre_toml_full_solve_summary.json"


def _post_constraint_state(case):
    mu_by_volume = {}
    psi_by_volume = {}
    for lvol in range(1, case.input_summary.packed_volume_count + 1):
        fixture = load_packaged_spectre_linear_system(case_label=case.label, volume_index=lvol)
        mu_by_volume[lvol] = fixture.system.mu
        psi_by_volume[lvol] = fixture.system.psi
    return mu_by_volume, psi_by_volume


print("[beltrami_jax] running full SPECTRE TOML-to-coefficients solve example")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STATIC_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)

case = load_packaged_spectre_case(CASE)
mu, psi = _post_constraint_state(case)
result = solve_spectre_volumes_from_input(case.input_summary, mu=mu, psi=psi, verbose=True)
comparison = compare_vector_potentials(
    result.vector_potential,
    case.reference.vector_potential,
    label=f"{CASE}: TOML assembly -> JAX solve -> Ate/Aze/Ato/Azo",
)
save_spectre_vector_potential_npz(COEFFICIENT_PATH, result.vector_potential)

relative_residuals = np.asarray(result.relative_residual_norms)
component_errors = comparison.component_relative_errors
summary = {
    "case": CASE,
    "source": case.input_summary.source,
    "packed_volume_count": case.input_summary.packed_volume_count,
    "shape": result.vector_potential.shape,
    "global_relative_error": comparison.global_relative_error,
    "global_max_abs_error": comparison.global_max_abs_error,
    "component_relative_errors": component_errors,
    "relative_residual_norms": relative_residuals.tolist(),
    "max_relative_residual_norm": float(np.max(relative_residuals)),
    "note": "Post-constraint SPECTRE mu/psi values are injected here only to validate the linear assembly/solve lane.",
}
SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
STATIC_SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.4), constrained_layout=True)
component_names = ("ate", "aze", "ato", "azo")
colors = ("#1B4965", "#2A9D8F", "#CA6702", "#D1495B")
for name, color in zip(component_names, colors):
    reference = getattr(case.reference.vector_potential, name).ravel()
    candidate = getattr(result.vector_potential, name).ravel()
    axes[0].scatter(reference, candidate, s=11, alpha=0.55, color=color, label=name.upper())
scale = max(
    float(np.max(np.abs(case.reference.vector_potential.stack_components()))),
    float(np.max(np.abs(result.vector_potential.stack_components()))),
    1.0e-14,
)
line = np.asarray([-1.05 * scale, 1.05 * scale])
axes[0].plot(line, line, color="black", linewidth=1.2)
axes[0].set_title("full coefficient parity")
axes[0].set_xlabel("SPECTRE reference")
axes[0].set_ylabel("JAX TOML solve")
axes[0].set_aspect("equal", adjustable="box")
axes[0].legend(frameon=False, fontsize=8)

axes[1].bar(component_names, [component_errors[name] for name in component_names], color=colors)
axes[1].set_yscale("log")
axes[1].set_title("component relative error")
axes[1].set_ylabel("||A_jax - A_ref|| / ||A_ref||")

axes[2].bar(
    [f"vol {index}" for index in range(1, len(relative_residuals) + 1)],
    np.maximum(relative_residuals, 1.0e-18),
    color="#4C956C",
)
axes[2].set_yscale("log")
axes[2].set_title("linear residuals")
axes[2].set_ylabel("relative residual norm")

figure.suptitle(f"{CASE}: SPECTRE TOML/interface geometry assembled and solved in JAX", fontsize=12)
figure.savefig(FIGURE_PATH, dpi=180)
figure.savefig(STATIC_FIGURE_PATH, dpi=180)
plt.close(figure)

print(f"[beltrami_jax] case={CASE} volumes={case.input_summary.packed_volume_count}")
print(f"[beltrami_jax] global_relative_error={comparison.global_relative_error:.3e}")
print(f"[beltrami_jax] max_relative_residual={float(np.max(relative_residuals)):.3e}")
print(f"[beltrami_jax] wrote {COEFFICIENT_PATH}")
print(f"[beltrami_jax] wrote {SUMMARY_PATH}")
print(f"[beltrami_jax] wrote {FIGURE_PATH}")
