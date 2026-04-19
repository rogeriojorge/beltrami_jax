from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from .types import BeltramiProblem, FourierBeltramiGeometry, FourierModeBasis, NonlinearSolveResult


def save_problem_json(path: str | Path, problem: BeltramiProblem) -> Path:
    """Serialize a high-level Beltrami problem to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "geometry": {
            "major_radius": problem.geometry.major_radius,
            "minor_radius": problem.geometry.minor_radius,
            "elongation": problem.geometry.elongation,
            "triangularity": problem.geometry.triangularity,
            "field_periods": problem.geometry.field_periods,
            "radial_points": problem.geometry.radial_points,
            "poloidal_points": problem.geometry.poloidal_points,
            "toroidal_points": problem.geometry.toroidal_points,
            "axis_regularization": problem.geometry.axis_regularization,
            "mass_shift": problem.geometry.mass_shift,
            "label": problem.geometry.label,
        },
        "basis": {
            "radial_orders": np.asarray(problem.basis.radial_orders).tolist(),
            "poloidal_modes": np.asarray(problem.basis.poloidal_modes).tolist(),
            "toroidal_modes": np.asarray(problem.basis.toroidal_modes).tolist(),
            "families": np.asarray(problem.basis.families).tolist(),
            "label": problem.basis.label,
        },
        "psi": np.asarray(problem.psi).tolist(),
        "target_helicity": problem.target_helicity,
        "initial_mu": problem.initial_mu,
        "is_vacuum": problem.is_vacuum,
        "vacuum_strength": problem.vacuum_strength,
        "solver": problem.solver,
        "tolerance": problem.tolerance,
        "max_iterations": problem.max_iterations,
        "label": problem.label,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def load_problem_json(path: str | Path) -> BeltramiProblem:
    """Load a JSON-serialized high-level Beltrami problem."""
    payload = json.loads(Path(path).read_text())
    geometry = FourierBeltramiGeometry(**payload["geometry"])
    basis = FourierModeBasis.from_arraylike(**payload["basis"])
    return BeltramiProblem.from_arraylike(
        geometry=geometry,
        basis=basis,
        psi=payload["psi"],
        target_helicity=payload["target_helicity"],
        initial_mu=payload["initial_mu"],
        is_vacuum=payload["is_vacuum"],
        vacuum_strength=payload["vacuum_strength"],
        solver=payload["solver"],
        tolerance=payload["tolerance"],
        max_iterations=payload["max_iterations"],
        label=payload["label"],
    )


def save_nonlinear_solution(path: str | Path, result: NonlinearSolveResult) -> Path:
    """Write the assembled system, solved state, and outer-loop history to NPZ."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        d_ma=np.asarray(result.assembly.system.d_ma),
        d_md=np.asarray(result.assembly.system.d_md),
        d_mb=np.asarray(result.assembly.system.d_mb),
        d_mg=np.asarray([]) if result.assembly.system.d_mg is None else np.asarray(result.assembly.system.d_mg),
        has_d_mg=result.assembly.system.d_mg is not None,
        mu=float(result.assembly.system.mu),
        psi=np.asarray(result.assembly.system.psi),
        solution=np.asarray(result.solve.solution),
        operator=np.asarray(result.solve.operator),
        rhs=np.asarray(result.solve.rhs),
        residual=np.asarray(result.solve.residual),
        magnetic_energy=float(result.solve.magnetic_energy),
        magnetic_helicity=float(result.solve.magnetic_helicity),
        mu_history=np.asarray(result.mu_history),
        helicity_history=np.asarray(result.helicity_history),
        constraint_residual_history=np.asarray(result.constraint_residual_history),
        converged=result.converged,
        iterations=result.iterations,
    )
    return path


def load_saved_solution(path: str | Path) -> dict[str, jnp.ndarray]:
    """Load an exported nonlinear-solve bundle."""
    with np.load(Path(path), allow_pickle=True) as payload:
        loaded = {key: jnp.asarray(payload[key]) for key in payload.files}
    if "has_d_mg" in loaded and not bool(loaded["has_d_mg"]):
        loaded["d_mg"] = jnp.asarray([])
    return loaded
