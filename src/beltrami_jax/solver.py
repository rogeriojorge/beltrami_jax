from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from .operators import (
    assemble_operator,
    assemble_rhs,
    magnetic_energy,
    magnetic_helicity,
)
from .types import BeltramiLinearSystem, SolveResult

jax.config.update("jax_enable_x64", True)


@jax.jit
def solve_operator(operator: Array, rhs: Array) -> tuple[Array, Array]:
    """Solve M x = rhs and return both the solution and residual."""
    solution = jnp.linalg.solve(operator, rhs)
    return solution, operator @ solution - rhs


def solve_from_components(
    system: BeltramiLinearSystem,
    *,
    verbose: bool = False,
) -> SolveResult:
    """Assemble and solve a SPEC-style linear Beltrami system."""
    if verbose:
        print(f"[beltrami_jax] solving {system.label or 'beltrami system'}")
        print(f"[beltrami_jax] size={system.size} mu={float(system.mu):.8e} psi={tuple(map(float, system.psi))}")

    operator = assemble_operator(system)
    rhs = assemble_rhs(system)
    solution, linear_residual = solve_operator(operator, rhs)

    residual_norm = jnp.linalg.norm(linear_residual, ord=2)
    rhs_norm = jnp.maximum(jnp.linalg.norm(rhs, ord=2), jnp.asarray(1e-30, dtype=rhs.dtype))
    operator_norm = jnp.linalg.norm(operator, ord="fro")
    solution_norm = jnp.linalg.norm(solution, ord=2)
    relative = residual_norm / rhs_norm

    if verbose:
        print(f"[beltrami_jax] operator_fro_norm={float(operator_norm):.8e}")
        print(f"[beltrami_jax] rhs_norm={float(rhs_norm):.8e}")
        print(f"[beltrami_jax] solution_norm={float(solution_norm):.8e}")
        print(f"[beltrami_jax] residual_norm={float(residual_norm):.8e}")
        print(f"[beltrami_jax] relative_residual_norm={float(relative):.8e}")

    return SolveResult(
        system=system,
        operator=operator,
        rhs=rhs,
        solution=solution,
        residual=linear_residual,
        residual_norm=residual_norm,
        relative_residual_norm=relative,
        magnetic_energy=magnetic_energy(solution, system),
        magnetic_helicity=magnetic_helicity(solution, system),
    )


@partial(jax.jit, static_argnames=())
def solve_parameter_scan(
    d_ma: ArrayLike,
    d_md: ArrayLike,
    d_mb: ArrayLike,
    mu_values: ArrayLike,
    psi_values: ArrayLike,
) -> Array:
    """Vectorized solve over a batch of `mu` and `psi` values."""
    d_ma = jnp.asarray(d_ma, dtype=jnp.float64)
    d_md = jnp.asarray(d_md, dtype=jnp.float64)
    d_mb = jnp.asarray(d_mb, dtype=jnp.float64)
    mu_values = jnp.asarray(mu_values, dtype=jnp.float64)
    psi_values = jnp.asarray(psi_values, dtype=jnp.float64)

    operators = d_ma[None, :, :] - mu_values[:, None, None] * d_md[None, :, :]
    rhs = -jnp.einsum("ij,bj->bi", d_mb, psi_values)
    return jax.vmap(jnp.linalg.solve)(operators, rhs)
