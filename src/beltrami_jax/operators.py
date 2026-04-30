from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from .types import BeltramiLinearSystem


def assemble_operator(system: BeltramiLinearSystem) -> Array:
    """Assemble the SPEC-style operator M = A - mu D."""
    if system.is_vacuum:
        return system.d_ma
    return system.d_ma - system.mu * system.d_md


def assemble_rhs(system: BeltramiLinearSystem) -> Array:
    """Assemble the right-hand side used by SPEC's linear solve stage."""
    rhs = -(system.d_mb @ system.psi)
    if system.include_d_mg_in_rhs and system.d_mg is not None:
        rhs = rhs - system.d_mg
    return rhs


def magnetic_energy(solution: Array, system: BeltramiLinearSystem) -> Array:
    """Quadratic magnetic-energy functional used in SPEC's linear stage."""
    return 0.5 * solution @ (system.d_ma @ solution) + solution @ (system.d_mb @ system.psi)


def magnetic_helicity(solution: Array, system: BeltramiLinearSystem) -> Array:
    """Quadratic helicity functional used in SPEC's linear stage."""
    return 0.5 * solution @ (system.d_md @ solution)


def residual(solution: Array, system: BeltramiLinearSystem) -> Array:
    """Linear residual M a - rhs."""
    return assemble_operator(system) @ solution - assemble_rhs(system)


def residual_norm(solution: Array, system: BeltramiLinearSystem) -> Array:
    return jnp.linalg.norm(residual(solution, system), ord=2)


def relative_residual_norm(solution: Array, system: BeltramiLinearSystem) -> Array:
    rhs = assemble_rhs(system)
    scale = jnp.maximum(jnp.linalg.norm(rhs, ord=2), jnp.asarray(1e-30, dtype=rhs.dtype))
    return residual_norm(solution, system) / scale
