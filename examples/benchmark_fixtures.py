from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from beltrami_jax import (
    BeltramiLinearSystem,
    BeltramiProblem,
    FourierBeltramiGeometry,
    assemble_fourier_beltrami_system,
    benchmark_parameter_scan,
    benchmark_solve,
    build_fourier_mode_basis,
    compute_solve_diagnostics,
    save_nonlinear_solution,
    solve_from_components,
    solve_helicity_constrained_equilibrium,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "examples" / "_generated" / "benchmark_fixtures"
BUNDLE_PATH = OUTPUT_DIR / "vacuum_gmres_bundle.npz"
FIGURE_PATH = OUTPUT_DIR / "vacuum_gmres_panel.png"

GEOMETRY = FourierBeltramiGeometry(
    major_radius=4.2,
    minor_radius=0.85,
    elongation=1.1,
    triangularity=0.1,
    field_periods=3,
    radial_points=7,
    poloidal_points=16,
    toroidal_points=12,
    mass_shift=0.65,
    label="vacuum_qh_geometry",
)
BASIS = build_fourier_mode_basis(max_radial_order=1, max_poloidal_mode=2, max_toroidal_mode=1, label="vacuum_qh_basis")
REFERENCE_MU = -0.25
TARGET_HELICITY = float(
    solve_from_components(
        assemble_fourier_beltrami_system(
            GEOMETRY,
            BASIS,
            mu=REFERENCE_MU,
            psi=[0.05, 0.025],
            is_vacuum=True,
            vacuum_strength=0.045,
            label="vacuum_reference",
        ).system,
        method="gmres",
        tolerance=1.0e-10,
    ).magnetic_helicity
)
PROBLEM = BeltramiProblem.from_arraylike(
    geometry=GEOMETRY,
    basis=BASIS,
    psi=[0.05, 0.025],
    target_helicity=TARGET_HELICITY,
    initial_mu=0.0,
    is_vacuum=True,
    vacuum_strength=0.045,
    solver="gmres",
    tolerance=1.0e-10,
    max_iterations=8,
    label="vacuum_gmres_problem",
)


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
nonlinear_result = solve_helicity_constrained_equilibrium(PROBLEM, verbose=True)
save_nonlinear_solution(BUNDLE_PATH, nonlinear_result)
diagnostics = compute_solve_diagnostics(nonlinear_result.solve, include_condition_number=True)
solve_benchmark = benchmark_solve(
    type(
        "ReferenceLike",
        (),
        {
            "system": nonlinear_result.assembly.system,
            "source": "assembled",
            "volume_index": 0,
            "matrix": nonlinear_result.solve.operator,
            "rhs": nonlinear_result.solve.rhs,
            "expected_solution": nonlinear_result.solve.solution,
        },
    )(),
    repeats=2,
)
scan_benchmarks = benchmark_parameter_scan(
    type(
        "ReferenceLike",
        (),
        {
            "system": BeltramiLinearSystem.from_arraylike(
                d_ma=nonlinear_result.assembly.system.d_ma,
                d_md=nonlinear_result.assembly.system.d_md,
                d_mb=nonlinear_result.assembly.system.d_mb,
                mu=0.02,
                psi=nonlinear_result.assembly.system.psi,
                label="benchmark_plasma_projection",
            ),
            "source": "assembled",
            "volume_index": 0,
            "matrix": nonlinear_result.solve.operator,
            "rhs": nonlinear_result.solve.rhs,
            "expected_solution": nonlinear_result.solve.solution,
        },
    )(),
    batch_sizes=(1, 4, 8),
    repeats=1,
    relative_span=0.015,
)

figure, axes = plt.subplots(1, 3, figsize=(14, 4.2))
axes[0].plot(np.asarray(nonlinear_result.mu_history), np.asarray(nonlinear_result.constraint_residual_history), marker="o")
axes[0].set_yscale("symlog", linthresh=1.0e-12)
axes[0].set_xlabel(r"$\mu$")
axes[0].set_ylabel("constraint residual")
axes[0].set_title("Vacuum outer loop")

axes[1].bar(
    ["compile+solve", "steady"],
    [solve_benchmark.compile_and_solve_seconds, solve_benchmark.steady_state_seconds],
    color=["#4c72b0", "#55a868"],
)
axes[1].set_ylabel("seconds")
axes[1].set_title("Dense solve timing")

axes[2].plot([item.batch_size for item in scan_benchmarks], [item.per_system_seconds for item in scan_benchmarks], marker="o", color="#c44e52")
axes[2].set_xlabel("batch size")
axes[2].set_ylabel("seconds / system")
axes[2].set_title("Vectorized scan throughput")

figure.tight_layout()
figure.savefig(FIGURE_PATH, dpi=200)
plt.close(figure)

print(f"[beltrami_jax] vacuum_problem_converged={nonlinear_result.converged}")
print(f"[beltrami_jax] vacuum_solution_method={nonlinear_result.solve.method}")
print(f"[beltrami_jax] vacuum_condition_number={diagnostics.condition_number_2:.8e}")
print(f"[beltrami_jax] benchmark_compile_and_solve={solve_benchmark.compile_and_solve_seconds:.6f}")
print(f"[beltrami_jax] wrote {BUNDLE_PATH}")
print(f"[beltrami_jax] wrote {FIGURE_PATH}")
