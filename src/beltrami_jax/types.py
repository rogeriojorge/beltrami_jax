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
class FourierBeltramiGeometry:
    """Large-aspect-ratio Fourier geometry used for internal assembly.

    The goal is not to replicate every SPEC branch symbol-for-symbol, but to
    provide a fully internal geometry/integral assembly path that produces the
    same style of discrete Beltrami system:

        (A - mu D) a = -B psi - g_vac
    """

    major_radius: float
    minor_radius: float
    elongation: float = 1.0
    triangularity: float = 0.0
    field_periods: int = 1
    radial_points: int = 10
    poloidal_points: int = 32
    toroidal_points: int = 24
    axis_regularization: float = 1.0e-3
    mass_shift: float = 1.0
    label: str = ""


@dataclass(frozen=True)
class FourierModeBasis:
    """Packed spectral basis used by the internal geometry-driven assembly."""

    radial_orders: Array
    poloidal_modes: Array
    toroidal_modes: Array
    families: Array
    label: str = ""

    @property
    def size(self) -> int:
        return int(self.radial_orders.shape[0])

    @classmethod
    def from_arraylike(
        cls,
        *,
        radial_orders: ArrayLike,
        poloidal_modes: ArrayLike,
        toroidal_modes: ArrayLike,
        families: ArrayLike,
        label: str = "",
    ) -> "FourierModeBasis":
        return cls(
            radial_orders=jnp.asarray(radial_orders, dtype=jnp.int32),
            poloidal_modes=jnp.asarray(poloidal_modes, dtype=jnp.int32),
            toroidal_modes=jnp.asarray(toroidal_modes, dtype=jnp.int32),
            families=jnp.asarray(families, dtype=jnp.int32),
            label=label,
        )


@dataclass(frozen=True)
class GeometryAssemblyResult:
    """Output of the internal geometry and integral assembly stage."""

    geometry: FourierBeltramiGeometry
    basis: FourierModeBasis
    system: BeltramiLinearSystem
    radial_grid: Array
    theta_grid: Array
    zeta_grid: Array


@dataclass(frozen=True)
class BeltramiProblem:
    """High-level input definition for an internally assembled Beltrami solve."""

    geometry: FourierBeltramiGeometry
    basis: FourierModeBasis
    psi: Array
    target_helicity: float
    initial_mu: float = 0.0
    is_vacuum: bool = False
    vacuum_strength: float = 0.0
    solver: str = "dense"
    tolerance: float = 1.0e-10
    max_iterations: int = 12
    label: str = ""

    @classmethod
    def from_arraylike(
        cls,
        *,
        geometry: FourierBeltramiGeometry,
        basis: FourierModeBasis,
        psi: ArrayLike,
        target_helicity: float,
        initial_mu: float = 0.0,
        is_vacuum: bool = False,
        vacuum_strength: float = 0.0,
        solver: str = "dense",
        tolerance: float = 1.0e-10,
        max_iterations: int = 12,
        label: str = "",
    ) -> "BeltramiProblem":
        return cls(
            geometry=geometry,
            basis=basis,
            psi=jnp.asarray(psi, dtype=jnp.float64),
            target_helicity=float(target_helicity),
            initial_mu=float(initial_mu),
            is_vacuum=is_vacuum,
            vacuum_strength=float(vacuum_strength),
            solver=solver,
            tolerance=float(tolerance),
            max_iterations=int(max_iterations),
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
    method: str = "dense"
    iterations: int | None = None


@dataclass(frozen=True)
class IterativeSolveResult:
    """Outputs of a matrix-free/dense Krylov solve."""

    solution: Array
    residual: Array
    residual_norm: Array
    relative_residual_norm: Array
    iterations: int
    converged: bool


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


@dataclass(frozen=True)
class NonlinearSolveResult:
    """Outputs of the outer helicity-constrained Beltrami update."""

    problem: BeltramiProblem
    assembly: GeometryAssemblyResult
    solve: SolveResult
    converged: bool
    iterations: int
    mu_history: Array
    helicity_history: Array
    constraint_residual_history: Array
