from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from beltrami_jax import (
    benchmark_spectre_backend,
    load_packaged_spectre_linear_system,
    solve_spectre_assembled_batch,
    solve_spectre_assembled_numpy,
)


SINGLE_SYSTEM = "G3V8L3Free/lvol7"
BATCH_SYSTEMS = ("G2V32L1Fi/lvol2", "G2V32L1Fi/lvol3", "G2V32L1Fi/lvol4")
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "examples" / "_generated" / "spectre_backend_dropin"
SUMMARY_PATH = OUTPUT_DIR / "spectre_backend_summary.json"
FIGURE_PATH = OUTPUT_DIR / "spectre_backend_panel.png"


def _backend_kwargs(fixture):
    return {
        "d_ma": fixture.system.d_ma,
        "d_md": fixture.system.d_md,
        "d_mb": fixture.system.d_mb,
        "d_mg": fixture.system.d_mg,
        "mu": fixture.system.mu,
        "psi": fixture.system.psi,
        "is_vacuum": fixture.system.is_vacuum,
        "include_d_mg_in_rhs": fixture.system.include_d_mg_in_rhs,
    }


print("[beltrami_jax] running SPECTRE assembled-matrix backend drop-in example")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

single_fixture = load_packaged_spectre_linear_system(SINGLE_SYSTEM)
single_result = solve_spectre_assembled_numpy(**_backend_kwargs(single_fixture))
single_expected = np.asarray(single_fixture.expected_solution)
single_error = np.asarray(single_result["solution"]) - single_expected
single_relative_error = float(np.linalg.norm(single_error) / max(np.linalg.norm(single_expected), 1e-300))

timing = benchmark_spectre_backend(
    label=single_fixture.name,
    size=single_fixture.n_dof,
    batch_size=1,
    solve_fn=lambda: solve_spectre_assembled_numpy(**_backend_kwargs(single_fixture)),
    repeats=2,
)

batch_fixtures = [load_packaged_spectre_linear_system(name) for name in BATCH_SYSTEMS]
batch_result = solve_spectre_assembled_batch(
    d_ma=np.stack([np.asarray(fixture.system.d_ma) for fixture in batch_fixtures]),
    d_md=np.stack([np.asarray(fixture.system.d_md) for fixture in batch_fixtures]),
    d_mb=np.stack([np.asarray(fixture.system.d_mb) for fixture in batch_fixtures]),
    mu=np.asarray([float(fixture.system.mu) for fixture in batch_fixtures]),
    psi=np.stack([np.asarray(fixture.system.psi) for fixture in batch_fixtures]),
    is_vacuum=False,
    include_d_mg_in_rhs=False,
)
batch_relative_errors = []
for row, fixture in enumerate(batch_fixtures):
    expected = np.asarray(fixture.expected_solution)
    error = np.asarray(batch_result.solutions[row]) - expected
    batch_relative_errors.append(float(np.linalg.norm(error) / max(np.linalg.norm(expected), 1e-300)))

summary = {
    "single_system": single_fixture.name,
    "single_size": single_fixture.n_dof,
    "single_relative_solution_error": single_relative_error,
    "single_relative_residual_norm": single_result["relative_residual_norm"],
    "timing": timing.__dict__,
    "batch_systems": [fixture.name for fixture in batch_fixtures],
    "batch_relative_solution_errors": batch_relative_errors,
    "batch_relative_residual_norms": np.asarray(batch_result.relative_residual_norms).tolist(),
    "spectre_side_change": "one optional backend flag plus one Python adapter call after SPECTRE assembly",
}
SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.2), constrained_layout=True)
axes[0].scatter(single_expected, single_result["solution"], s=18, color="#214E8A", alpha=0.75)
scale = float(np.max(np.abs(np.concatenate([single_expected, single_result["solution"]]))))
line = np.asarray([-1.05 * scale, 1.05 * scale])
axes[0].plot(line, line, color="#D1495B", linewidth=1.5)
axes[0].set_title("Single SPECTRE volume")
axes[0].set_xlabel("SPECTRE solution")
axes[0].set_ylabel("JAX backend solution")
axes[0].set_aspect("equal", adjustable="box")

axes[1].bar(range(len(batch_relative_errors)), np.maximum(batch_relative_errors, 1e-18), color="#2A9D8F")
axes[1].set_yscale("log")
axes[1].set_xticks(range(len(batch_relative_errors)))
axes[1].set_xticklabels([fixture.name.split("/")[-1] for fixture in batch_fixtures])
axes[1].set_title("Equal-size batched volumes")
axes[1].set_ylabel("relative solution error")

axes[2].bar(
    ["compile+solve", "steady-state"],
    [timing.compile_and_solve_seconds, timing.steady_state_seconds],
    color=["#CA6702", "#1B4965"],
)
axes[2].set_title("Adapter timing")
axes[2].set_ylabel("seconds")

figure.savefig(FIGURE_PATH, dpi=180)
plt.close(figure)

print(f"[beltrami_jax] single_system={single_fixture.name}")
print(f"[beltrami_jax] single_relative_solution_error={single_relative_error:.3e}")
print(f"[beltrami_jax] single_relative_residual_norm={single_result['relative_residual_norm']:.3e}")
print(f"[beltrami_jax] batch_max_relative_solution_error={max(batch_relative_errors):.3e}")
print(f"[beltrami_jax] wrote {SUMMARY_PATH}")
print(f"[beltrami_jax] wrote {FIGURE_PATH}")
