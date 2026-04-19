from __future__ import annotations

import json
from pathlib import Path

import jax
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from beltrami_jax import (
    BeltramiLinearSystem,
    BeltramiProblem,
    FourierBeltramiGeometry,
    assemble_fourier_beltrami_system,
    build_fourier_mode_basis,
    magnetic_energy,
    save_problem_json,
    solve_from_components,
    solve_helicity_constrained_equilibrium,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "examples" / "_generated" / "autodiff_mu"
INPUT_PATH = OUTPUT_DIR / "autodiff_problem.json"
REPORT_PATH = OUTPUT_DIR / "autodiff_report.json"
FIGURE_PATH = OUTPUT_DIR / "autodiff_gradient_check.png"

GEOMETRY = FourierBeltramiGeometry(
    major_radius=2.9,
    minor_radius=0.95,
    elongation=1.2,
    triangularity=0.15,
    field_periods=1,
    radial_points=7,
    poloidal_points=18,
    toroidal_points=10,
    mass_shift=0.9,
    label="tokamak_like_geometry",
)
BASIS = build_fourier_mode_basis(max_radial_order=1, max_poloidal_mode=2, max_toroidal_mode=1, label="tokamak_like_basis")
PSI = np.array([0.08, 0.01])
REFERENCE_MU = -0.85
TARGET_HELICITY = float(
    solve_from_components(
        assemble_fourier_beltrami_system(GEOMETRY, BASIS, mu=REFERENCE_MU, psi=PSI, label="autodiff_reference").system
    ).magnetic_helicity
)
PROBLEM = BeltramiProblem.from_arraylike(
    geometry=GEOMETRY,
    basis=BASIS,
    psi=PSI,
    target_helicity=TARGET_HELICITY,
    initial_mu=0.02,
    solver="dense",
    tolerance=1.0e-10,
    max_iterations=8,
    label="autodiff_problem",
)


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
save_problem_json(INPUT_PATH, PROBLEM)
nonlinear_result = solve_helicity_constrained_equilibrium(PROBLEM, verbose=True)
assembly = assemble_fourier_beltrami_system(
    GEOMETRY,
    BASIS,
    mu=float(nonlinear_result.assembly.system.mu),
    psi=PSI,
    label="autodiff_base_system",
)


def solved_energy(mu_value: float) -> jax.Array:
    system = BeltramiLinearSystem.from_arraylike(
        d_ma=assembly.system.d_ma,
        d_md=assembly.system.d_md,
        d_mb=assembly.system.d_mb,
        mu=mu_value,
        psi=assembly.system.psi,
        label=assembly.system.label,
    )
    result = solve_from_components(system)
    return magnetic_energy(result.solution, system)


mu0 = float(nonlinear_result.assembly.system.mu)
energy0 = float(solved_energy(mu0))
gradient0 = float(jax.grad(solved_energy)(mu0))
delta = 1.0e-4
finite_difference = float((solved_energy(mu0 + delta) - solved_energy(mu0 - delta)) / (2.0 * delta))
mu_values = np.linspace(mu0 - 0.05, mu0 + 0.05, 13)
energy_values = np.asarray([float(solved_energy(mu_value)) for mu_value in mu_values])
linearized = energy0 + gradient0 * (mu_values - mu0)

REPORT_PATH.write_text(
    json.dumps(
        {
            "mu0": mu0,
            "energy0": energy0,
            "gradient0": gradient0,
            "finite_difference": finite_difference,
            "absolute_gradient_error": abs(gradient0 - finite_difference),
        },
        indent=2,
    )
    + "\n"
)

figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].plot(mu_values, energy_values, marker="o", label="solved energy")
axes[0].plot(mu_values, linearized, "--", label="linearization")
axes[0].axvline(mu0, linestyle=":", color="black")
axes[0].set_xlabel(r"$\mu$")
axes[0].set_ylabel("magnetic energy")
axes[0].set_title("Autodiff around converged equilibrium")
axes[0].legend()

axes[1].bar(
    ["autodiff", "finite difference"],
    [gradient0, finite_difference],
    color=["#4c72b0", "#55a868"],
)
axes[1].set_ylabel(r"$dE/d\mu$")
axes[1].set_title("Gradient check")

figure.tight_layout()
figure.savefig(FIGURE_PATH, dpi=200)
plt.close(figure)

print(f"[beltrami_jax] autodiff_mu0={mu0:.8e}")
print(f"[beltrami_jax] energy(mu0)={energy0:.8e}")
print(f"[beltrami_jax] denergy_dmu={gradient0:.8e}")
print(f"[beltrami_jax] finite_difference={finite_difference:.8e}")
print(f"[beltrami_jax] wrote {INPUT_PATH}")
print(f"[beltrami_jax] wrote {REPORT_PATH}")
print(f"[beltrami_jax] wrote {FIGURE_PATH}")
