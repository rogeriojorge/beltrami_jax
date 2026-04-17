from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


@dataclass(frozen=True)
class BeltramiLinearSystem:
    """SPEC-style linear Beltrami system components.

    The naming follows SPEC's `matrix.f90` / `mp00ac.f90` convention:

    - `d_ma` corresponds to the quadratic magnetic-energy matrix A.
    - `d_md` corresponds to the quadratic helicity matrix D.
    - `d_mb` maps the flux vector psi to the right-hand side.
    - `d_mg` is an additive vacuum forcing term, only used for vacuum regions.
    """

    d_ma: Array
    d_md: Array
    d_mb: Array
    mu: Array
    psi: Array
    d_mg: Array | None = None
    is_vacuum: bool = False
    label: str = ""

    @property
    def size(self) -> int:
        return int(self.d_ma.shape[0])

    @classmethod
    def from_arraylike(
        cls,
        *,
        d_ma: ArrayLike,
        d_md: ArrayLike,
        d_mb: ArrayLike,
        mu: ArrayLike,
        psi: ArrayLike,
        d_mg: ArrayLike | None = None,
        is_vacuum: bool = False,
        label: str = "",
    ) -> "BeltramiLinearSystem":
        return cls(
            d_ma=jnp.asarray(d_ma, dtype=jnp.float64),
            d_md=jnp.asarray(d_md, dtype=jnp.float64),
            d_mb=jnp.asarray(d_mb, dtype=jnp.float64),
            mu=jnp.asarray(mu, dtype=jnp.float64),
            psi=jnp.asarray(psi, dtype=jnp.float64),
            d_mg=None if d_mg is None else jnp.asarray(d_mg, dtype=jnp.float64),
            is_vacuum=is_vacuum,
            label=label,
        )


@dataclass(frozen=True)
class SpecLinearSystemReference:
    """Reference system exported from a SPEC run."""

    system: BeltramiLinearSystem
    matrix: Array
    rhs: Array
    expected_solution: Array
    volume_index: int
    source: str


@dataclass(frozen=True)
class SolveResult:
    """Outputs of a linear Beltrami solve."""

    system: BeltramiLinearSystem
    operator: Array
    rhs: Array
    solution: Array
    residual: Array
    residual_norm: Array
    relative_residual_norm: Array
    magnetic_energy: Array
    magnetic_helicity: Array


@dataclass(frozen=True)
class SolveDiagnostics:
    """Post-solve diagnostic summary for a dense Beltrami system."""

    label: str
    size: int
    is_vacuum: bool
    operator_fro_norm: float
    rhs_l2_norm: float
    solution_l2_norm: float
    max_abs_solution: float
    max_abs_residual: float
    residual_l2_norm: float
    relative_residual_norm: float
    symmetry_defect: float
    amplification_factor: float
    condition_number_2: float | None = None


@dataclass(frozen=True)
class ReferenceComparison:
    """Agreement metrics between a JAX solve and a dumped SPEC reference."""

    label: str
    size: int
    volume_index: int
    operator_relative_error: float
    rhs_relative_error: float
    solution_relative_error: float
    max_abs_solution_error: float


@dataclass(frozen=True)
class SolveBenchmark:
    """Timing summary for solving a packed Beltrami system."""

    label: str
    size: int
    repeats: int
    compile_and_solve_seconds: float
    steady_state_seconds: float


@dataclass(frozen=True)
class ParameterScanBenchmark:
    """Timing summary for a batched `solve_parameter_scan` execution."""

    label: str
    size: int
    batch_size: int
    repeats: int
    compile_and_solve_seconds: float
    steady_state_seconds: float
    per_system_seconds: float
