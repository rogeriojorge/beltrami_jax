from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from beltrami_jax import (
    BeltramiProblem,
    FourierBeltramiGeometry,
    assemble_fourier_beltrami_system,
    build_fourier_mode_basis,
    magnetic_energy,
    save_problem_json,
    solve_from_components,
    solve_helicity_constrained_equilibrium,
    solve_parameter_scan,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "examples" / "_generated" / "parameter_scan"
INPUT_PATH = OUTPUT_DIR / "tokamak_problem.json"
SCAN_PATH = OUTPUT_DIR / "tokamak_scan.npz"
FIGURE_PATH = OUTPUT_DIR / "tokamak_scan.png"

GEOMETRY = FourierBeltramiGeometry(
    major_radius=3.7,
    minor_radius=1.15,
    elongation=1.45,
    triangularity=0.22,
    field_periods=2,
    radial_points=8,
    poloidal_points=20,
    toroidal_points=12,
    mass_shift=0.75,
    label="qa_like_geometry",
)
BASIS = build_fourier_mode_basis(max_radial_order=1, max_poloidal_mode=2, max_toroidal_mode=1, label="qa_like_basis")
PSI = np.array([0.12, -0.03])
REFERENCE_MU = -2.45
TARGET_HELICITY = float(
    solve_from_components(
        assemble_fourier_beltrami_system(GEOMETRY, BASIS, mu=REFERENCE_MU, psi=PSI, label="qa_like_reference").system
    ).magnetic_helicity
)
PROBLEM = BeltramiProblem.from_arraylike(
    geometry=GEOMETRY,
    basis=BASIS,
    psi=PSI,
    target_helicity=TARGET_HELICITY,
    initial_mu=0.05,
    solver="dense",
    tolerance=1.0e-10,
    max_iterations=12,
    label="qa_like_equilibrium",
)


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
save_problem_json(INPUT_PATH, PROBLEM)
nonlinear_result = solve_helicity_constrained_equilibrium(PROBLEM, verbose=True)
assembled = assemble_fourier_beltrami_system(
    GEOMETRY,
    BASIS,
    mu=float(nonlinear_result.assembly.system.mu),
    psi=PSI,
    label="scan_base_system",
)

mu0 = float(nonlinear_result.assembly.system.mu)
mu_values = np.linspace(mu0 - 0.04, mu0 + 0.04, 9)
psi_values = np.repeat(np.asarray(assembled.system.psi)[None, :], repeats=len(mu_values), axis=0)
solutions = solve_parameter_scan(assembled.system.d_ma, assembled.system.d_md, assembled.system.d_mb, mu_values, psi_values)
energies = np.asarray(
    [
        magnetic_energy(
            solution,
            assembled.system.__class__.from_arraylike(
                d_ma=assembled.system.d_ma,
                d_md=assembled.system.d_md,
                d_mb=assembled.system.d_mb,
                mu=mu_value,
                psi=assembled.system.psi,
                label=assembled.system.label,
            ),
        )
        for mu_value, solution in zip(mu_values, np.asarray(solutions), strict=True)
    ]
)
helicity_history = np.asarray(nonlinear_result.helicity_history)

np.savez_compressed(
    SCAN_PATH,
    mu_values=mu_values,
    energies=energies,
    solutions=np.asarray(solutions),
    outer_mu_history=np.asarray(nonlinear_result.mu_history),
    outer_helicity_history=helicity_history,
    outer_constraint_residual_history=np.asarray(nonlinear_result.constraint_residual_history),
)

figure, axes = plt.subplots(1, 3, figsize=(14, 4.2))
axes[0].plot(mu_values, energies, marker="o", color="#4c72b0")
axes[0].axvline(mu0, linestyle="--", color="black", linewidth=1.0)
axes[0].set_xlabel(r"$\mu$")
axes[0].set_ylabel("magnetic energy")
axes[0].set_title("Vectorized parameter scan")

axes[1].plot(np.asarray(nonlinear_result.mu_history), helicity_history, marker="o", color="#55a868")
axes[1].axhline(TARGET_HELICITY, linestyle="--", color="black", linewidth=1.0)
axes[1].set_xlabel("outer iteration mu")
axes[1].set_ylabel("magnetic helicity")
axes[1].set_title("Outer constraint loop")

axes[2].semilogy(np.abs(np.asarray(solutions[solutions.shape[0] // 2])) + 1.0e-18, color="#c44e52")
axes[2].set_xlabel("packed coefficient")
axes[2].set_ylabel(r"$|a_i|$")
axes[2].set_title("Converged coefficient spectrum")

figure.suptitle("Geometry-defined Beltrami workflow", fontsize=13)
figure.tight_layout()
figure.savefig(FIGURE_PATH, dpi=200)
plt.close(figure)

print(f"[beltrami_jax] geometry={GEOMETRY.label} basis_size={BASIS.size}")
print(f"[beltrami_jax] converged_mu={mu0:.8e} target_helicity={TARGET_HELICITY:.8e}")
print(f"[beltrami_jax] scan_energy_min={energies.min():.8e} scan_energy_max={energies.max():.8e}")
print(f"[beltrami_jax] wrote {INPUT_PATH}")
print(f"[beltrami_jax] wrote {SCAN_PATH}")
print(f"[beltrami_jax] wrote {FIGURE_PATH}")
