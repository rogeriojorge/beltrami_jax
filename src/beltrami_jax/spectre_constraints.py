from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike


SUPPORTED_LCONSTRAINTS = (-2, -1, 0, 1, 2, 3)


@dataclass(frozen=True)
class SpectreBeltramiBranchSolve:
    """SPECTRE ``solve_beltrami_system`` branch solve result.

    ``solutions[0]`` is the vector-potential solution. Active derivative rows
    follow SPECTRE's branch-specific unknown ordering: ``("mu", "dpflux")``
    for plasma, ``("dtflux", "dpflux")`` for vacuum, and ``("dtflux",)``
    for the coordinate-singularity ``Lconstraint == -2`` current branch.
    Branches with only one active derivative keep a trailing zero row so the
    array shape remains stable for JIT compilation.
    """

    solution: Array
    solutions: Array
    operator: Array
    rhs: Array
    derivative_rhs: Array
    residuals: Array
    residual_norms: Array
    relative_residual_norms: Array
    magnetic_energy: Array
    magnetic_helicity: Array
    helicity_derivatives: Array
    branch_unknowns: tuple[str, ...]


@dataclass(frozen=True)
class SpectreConstraintTargets:
    """Targets used by SPECTRE's local Beltrami constraint residuals."""

    lconstraint: int
    is_vacuum: bool
    coordinate_singularity: bool = False
    iota_inner: float = 0.0
    iota_outer: float = 0.0
    curtor: float = 0.0
    curpol: float = 0.0
    helicity: float = 0.0


@dataclass(frozen=True)
class SpectreConstraintDiagnostics:
    """Diagnostic values and derivatives consumed by constraint residuals.

    Arrays use columns ``(value, derivative_1, derivative_2)`` in the same
    branch-unknown ordering returned by :func:`spectre_branch_unknowns`.
    ``rotational_transform`` rows are ``(inner, outer)``. ``plasma_current``
    rows are ``(toroidal, poloidal)``.
    """

    rotational_transform: Array | None = None
    plasma_current: Array | None = None
    helicity: Array | None = None
    helicity_derivatives: Array | None = None


@dataclass(frozen=True)
class SpectreConstraintEvaluation:
    """Constraint residual and Jacobian in SPECTRE branch ordering."""

    residual: Array
    jacobian: Array
    unknowns: tuple[str, ...]


def _validate_lconstraint(lconstraint: int) -> None:
    if int(lconstraint) not in SUPPORTED_LCONSTRAINTS:
        raise ValueError(f"unsupported SPECTRE Lconstraint={lconstraint}; expected one of {SUPPORTED_LCONSTRAINTS}")


def spectre_constraint_dof_count(
    *,
    lconstraint: int,
    is_vacuum: bool,
    coordinate_singularity: bool = False,
) -> int:
    """Return SPECTRE's local nonlinear unknown count for one volume.

    This mirrors the ``Nxdof`` selection in SPECTRE
    ``construct_beltrami_field``. A zero return value means SPECTRE calls the
    linear Beltrami solve once without a local nonlinear iteration.
    """

    _validate_lconstraint(lconstraint)
    if not is_vacuum:
        if lconstraint == -2:
            return 1
        if lconstraint in (-1, 0, 3):
            return 0
        if lconstraint == 1:
            return 1 if coordinate_singularity else 2
        if lconstraint == 2:
            return 1
    else:
        if lconstraint in (-1, 3):
            return 0
        if lconstraint in (0, 1, 2):
            return 2
    raise AssertionError("unreachable SPECTRE constraint branch")


def spectre_branch_unknowns(
    *,
    lconstraint: int,
    is_vacuum: bool,
    coordinate_singularity: bool = False,
) -> tuple[str, ...]:
    """Return SPECTRE's local unknown names for one Beltrami branch."""

    count = spectre_constraint_dof_count(
        lconstraint=lconstraint,
        is_vacuum=is_vacuum,
        coordinate_singularity=coordinate_singularity,
    )
    if count == 0:
        return ()
    if lconstraint == -2 and not is_vacuum:
        return ("dtflux",)
    if is_vacuum:
        return ("dtflux", "dpflux")[:count]
    return ("mu", "dpflux")[:count]


def _as_required_vector(
    value: ArrayLike | None,
    *,
    size: int,
    required: bool,
) -> Array:
    if value is None:
        if required:
            raise ValueError("d_mg is required for this SPECTRE branch")
        return jnp.zeros((size,), dtype=jnp.float64)
    return jnp.asarray(value, dtype=jnp.float64)


def _branch_uses_d_mg(
    *,
    lconstraint: int,
    is_vacuum: bool,
    coordinate_singularity: bool,
) -> bool:
    return bool(is_vacuum or (coordinate_singularity and lconstraint == -2))


def _derivative_names_for_linear_solve(
    *,
    lconstraint: int,
    is_vacuum: bool,
    coordinate_singularity: bool,
) -> tuple[str, ...]:
    if coordinate_singularity and lconstraint == -2 and not is_vacuum:
        return ("dtflux",)
    if is_vacuum:
        return ("dtflux", "dpflux")
    return ("mu", "dpflux")


@partial(jax.jit, static_argnames=("lconstraint", "is_vacuum", "coordinate_singularity"))
def _solve_spectre_branch_jax(
    d_ma: Array,
    d_md: Array,
    d_mb: Array,
    d_mg: Array,
    mu: Array,
    psi: Array,
    *,
    lconstraint: int,
    is_vacuum: bool,
    coordinate_singularity: bool,
) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array]:
    operator = d_ma if is_vacuum else d_ma - mu * d_md
    use_d_mg = _branch_uses_d_mg(
        lconstraint=lconstraint,
        is_vacuum=is_vacuum,
        coordinate_singularity=coordinate_singularity,
    )
    rhs = -(d_mb @ psi)
    if use_d_mg:
        rhs = rhs - d_mg

    solution = jnp.linalg.solve(operator, rhs)
    tpsi = jnp.asarray([1.0, 0.0], dtype=jnp.float64)
    ppsi = jnp.asarray([0.0, 1.0], dtype=jnp.float64)

    if coordinate_singularity and lconstraint == -2 and not is_vacuum:
        derivative_rhs = jnp.stack((-(d_mb @ tpsi), jnp.zeros_like(rhs)))
    elif is_vacuum:
        derivative_rhs = jnp.stack((-(d_mb @ tpsi), -(d_mb @ ppsi)))
    else:
        derivative_rhs = jnp.stack((d_md @ solution, -(d_mb @ ppsi)))

    derivative_solutions = jax.vmap(lambda one_rhs: jnp.linalg.solve(operator, one_rhs))(derivative_rhs)
    solutions = jnp.concatenate((solution[None, :], derivative_solutions), axis=0)
    all_rhs = jnp.concatenate((rhs[None, :], derivative_rhs), axis=0)
    residuals = jax.vmap(lambda one_solution, one_rhs: operator @ one_solution - one_rhs)(solutions, all_rhs)
    residual_norms = jnp.linalg.norm(residuals, ord=2, axis=1)
    rhs_norms = jnp.maximum(jnp.linalg.norm(all_rhs, ord=2, axis=1), jnp.asarray(1.0e-30, dtype=jnp.float64))
    magnetic_energy = 0.5 * solution @ (d_ma @ solution) + solution @ (d_mb @ psi)
    magnetic_helicity = 0.5 * solution @ (d_md @ solution)
    helicity_derivatives = jnp.asarray(
        [
            0.5 * derivative_solutions[0] @ (d_md @ solution) + 0.5 * solution @ (d_md @ derivative_solutions[0]),
            0.5 * derivative_solutions[1] @ (d_md @ solution) + 0.5 * solution @ (d_md @ derivative_solutions[1]),
        ],
        dtype=jnp.float64,
    )
    return (
        solution,
        solutions,
        operator,
        rhs,
        derivative_rhs,
        residuals,
        residual_norms,
        residual_norms / rhs_norms,
        jnp.asarray([magnetic_energy, magnetic_helicity, helicity_derivatives[0], helicity_derivatives[1]]),
    )


def solve_spectre_beltrami_branch(
    *,
    d_ma: ArrayLike,
    d_md: ArrayLike,
    d_mb: ArrayLike,
    mu: ArrayLike,
    psi: ArrayLike,
    d_mg: ArrayLike | None = None,
    lconstraint: int,
    is_vacuum: bool,
    coordinate_singularity: bool = False,
) -> SpectreBeltramiBranchSolve:
    """Solve SPECTRE's local Beltrami branch including derivative RHS solves.

    This ports the linear-algebra portion of SPECTRE
    ``solve_beltrami_system``. It does not compute rotational transforms or
    plasma currents; those diagnostics are handled by
    :func:`evaluate_spectre_constraints` once supplied by a geometry/field
    backend.
    """

    _validate_lconstraint(lconstraint)
    d_ma_j = jnp.asarray(d_ma, dtype=jnp.float64)
    d_md_j = jnp.asarray(d_md, dtype=jnp.float64)
    d_mb_j = jnp.asarray(d_mb, dtype=jnp.float64)
    d_mg_j = _as_required_vector(
        d_mg,
        size=int(d_ma_j.shape[0]),
        required=_branch_uses_d_mg(
            lconstraint=lconstraint,
            is_vacuum=is_vacuum,
            coordinate_singularity=coordinate_singularity,
        ),
    )
    (
        solution,
        solutions,
        operator,
        rhs,
        derivative_rhs,
        residuals,
        residual_norms,
        relative_residual_norms,
        scalar_outputs,
    ) = _solve_spectre_branch_jax(
        d_ma_j,
        d_md_j,
        d_mb_j,
        d_mg_j,
        jnp.asarray(mu, dtype=jnp.float64),
        jnp.asarray(psi, dtype=jnp.float64),
        lconstraint=int(lconstraint),
        is_vacuum=bool(is_vacuum),
        coordinate_singularity=bool(coordinate_singularity),
    )
    return SpectreBeltramiBranchSolve(
        solution=solution,
        solutions=solutions,
        operator=operator,
        rhs=rhs,
        derivative_rhs=derivative_rhs,
        residuals=residuals,
        residual_norms=residual_norms,
        relative_residual_norms=relative_residual_norms,
        magnetic_energy=scalar_outputs[0],
        magnetic_helicity=scalar_outputs[1],
        helicity_derivatives=scalar_outputs[2:4],
        branch_unknowns=_derivative_names_for_linear_solve(
            lconstraint=lconstraint,
            is_vacuum=is_vacuum,
            coordinate_singularity=coordinate_singularity,
        ),
    )


def solve_spectre_beltrami_branch_numpy(**kwargs) -> dict[str, np.ndarray | float | tuple[str, ...]]:
    """NumPy-returning wrapper for SPECTRE branch-solve integration tests."""

    result = solve_spectre_beltrami_branch(**kwargs)
    return {
        "solution": np.asarray(result.solution),
        "solutions": np.asarray(result.solutions),
        "rhs": np.asarray(result.rhs),
        "derivative_rhs": np.asarray(result.derivative_rhs),
        "residuals": np.asarray(result.residuals),
        "residual_norms": np.asarray(result.residual_norms),
        "relative_residual_norms": np.asarray(result.relative_residual_norms),
        "magnetic_energy": float(np.asarray(result.magnetic_energy)),
        "magnetic_helicity": float(np.asarray(result.magnetic_helicity)),
        "helicity_derivatives": np.asarray(result.helicity_derivatives),
        "branch_unknowns": result.branch_unknowns,
    }


def _diagnostic_array(value: ArrayLike | None, *, name: str) -> Array:
    if value is None:
        raise ValueError(f"{name} diagnostics are required for this Lconstraint branch")
    array = jnp.asarray(value, dtype=jnp.float64)
    if array.shape != (2, 3):
        raise ValueError(f"{name} diagnostics must have shape (2, 3); got {array.shape}")
    return array


def _helicity_value(value: ArrayLike | None) -> Array:
    if value is None:
        raise ValueError("helicity diagnostic is required for Lconstraint=2")
    return jnp.asarray(value, dtype=jnp.float64)


def _helicity_derivatives(value: ArrayLike | None) -> Array:
    if value is None:
        raise ValueError("helicity derivatives are required for Lconstraint=2")
    array = jnp.asarray(value, dtype=jnp.float64)
    if array.shape[0] < 1:
        raise ValueError("helicity derivatives must contain at least one derivative")
    return array


def evaluate_spectre_constraints(
    targets: SpectreConstraintTargets,
    diagnostics: SpectreConstraintDiagnostics,
) -> SpectreConstraintEvaluation:
    """Evaluate SPECTRE ``Lconstraint`` residuals and Jacobians.

    The formulas mirror the branch table in SPECTRE ``solve_beltrami_system``.
    Expensive field diagnostics such as rotational transform and plasma current
    are injected as arrays so this layer is independent of the future
    JAX-native geometry/field evaluator.
    """

    lconstraint = int(targets.lconstraint)
    _validate_lconstraint(lconstraint)
    unknowns = spectre_branch_unknowns(
        lconstraint=lconstraint,
        is_vacuum=targets.is_vacuum,
        coordinate_singularity=targets.coordinate_singularity,
    )
    ndof = len(unknowns)
    residual = jnp.zeros((ndof,), dtype=jnp.float64)
    jacobian = jnp.zeros((ndof, ndof), dtype=jnp.float64)

    if ndof == 0 or lconstraint in (-1, 3):
        return SpectreConstraintEvaluation(residual=residual, jacobian=jacobian, unknowns=unknowns)

    if lconstraint == -2:
        current = _diagnostic_array(diagnostics.plasma_current, name="plasma_current")
        residual = residual.at[0].set(current[1, 0] - targets.curpol)
        jacobian = jacobian.at[0, 0].set(current[1, 1])
    elif lconstraint == 0:
        if targets.is_vacuum:
            current = _diagnostic_array(diagnostics.plasma_current, name="plasma_current")
            residual = jnp.asarray([current[0, 0] - targets.curtor, current[1, 0] - targets.curpol])
            jacobian = current[:, 1:3]
    elif lconstraint == 1:
        transform = _diagnostic_array(diagnostics.rotational_transform, name="rotational_transform")
        if not targets.is_vacuum:
            if targets.coordinate_singularity:
                residual = residual.at[0].set(transform[1, 0] - targets.iota_outer)
                jacobian = jacobian.at[0, 0].set(transform[1, 1])
            else:
                residual = jnp.asarray(
                    [transform[0, 0] - targets.iota_inner, transform[1, 0] - targets.iota_outer]
                )
                jacobian = transform[:, 1:3]
        else:
            current = _diagnostic_array(diagnostics.plasma_current, name="plasma_current")
            residual = jnp.asarray([transform[0, 0] - targets.iota_inner, current[1, 0] - targets.curpol])
            jacobian = jnp.asarray(
                [
                    [transform[0, 1], transform[0, 2]],
                    [current[1, 1], current[1, 2]],
                ],
                dtype=jnp.float64,
            )
    elif lconstraint == 2:
        helicity = _helicity_value(diagnostics.helicity)
        helicity_derivatives = _helicity_derivatives(diagnostics.helicity_derivatives)
        residual = residual.at[0].set(helicity - targets.helicity)
        jacobian = jacobian.at[0, 0].set(helicity_derivatives[0])

    return SpectreConstraintEvaluation(residual=residual, jacobian=jacobian, unknowns=unknowns)
