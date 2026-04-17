# Theory and Numerics

## Beltrami fields

A Beltrami field satisfies

$$
\nabla \times \mathbf{B} = \mu \mathbf{B},
$$

with scalar multiplier $\mu$. In a relaxed plasma region this implies

$$
\mathbf{J} \times \mathbf{B} = \mathbf{0},
$$

so the field is force free inside that region. In Taylor relaxation theory, the relaxed state minimizes magnetic energy while preserving appropriate helicity and flux constraints.

## Role inside MRxMHD

In multi-region relaxed magnetohydrodynamics (MRxMHD), the plasma is partitioned into regions separated by ideal interfaces. Each region relaxes internally to a Beltrami state, while the interfaces preserve additional constraints. This allows equilibria that are not restricted to globally nested flux surfaces and is one of the reasons SPEC is useful for strongly 3D stellarator and shaped tokamak configurations.

## Discrete linear system in SPEC

The part of SPEC currently reproduced here is the dense linear solve described in `mp00ac.f90`. The operator is

$$
\mathbf{M} = \mathbf{A} - \mu \mathbf{D},
$$

where:

- $\mathbf{A}$ is the quadratic magnetic-energy matrix
- $\mathbf{D}$ is the quadratic helicity matrix
- $\mu$ is the Beltrami multiplier for the region

The right-hand side depends on the branch.

### Plasma region

For the current non-singular plasma branch implemented in `beltrami_jax`,

$$
\mathbf{r} = -\mathbf{B}\boldsymbol{\psi},
$$

with $\boldsymbol{\psi} = (\Delta\psi_t, \Delta\psi_p)^T$.

### Vacuum region

For the vacuum branch,

$$
\mathbf{r} = -\mathbf{G} - \mathbf{B}\boldsymbol{\psi}.
$$

The operator itself is reduced to

$$
\mathbf{M} = \mathbf{A},
$$

which is represented in the package by setting `is_vacuum=True`.

## Quadratic diagnostics

After solving for the coefficient vector $\mathbf{a}$, the code evaluates the same quadratic diagnostics that SPEC uses in its linear stage:

magnetic energy
: $$
  W(\mathbf{a}) =
  \frac{1}{2}\mathbf{a}^T \mathbf{A}\mathbf{a}
  + \mathbf{a}^T \mathbf{B}\boldsymbol{\psi}
  $$

magnetic helicity
: $$
  K(\mathbf{a}) =
  \frac{1}{2}\mathbf{a}^T \mathbf{D}\mathbf{a}
  $$

linear residual
: $$
  \mathbf{e} = \mathbf{M}\mathbf{a} - \mathbf{r}
  $$

relative residual norm
: $$
  \frac{\lVert \mathbf{e} \rVert_2}
       {\max(\lVert \mathbf{r} \rVert_2, 10^{-30})}
  $$

## Numerical implementation in JAX

The current dense solve path uses

$$
\mathbf{a} = \operatorname{solve}(\mathbf{M}, \mathbf{r}),
$$

implemented through `jax.numpy.linalg.solve`.

This choice has three advantages for the current stage of the project:

1. It matches the dense linear-regression use case against dumped SPEC systems.
2. It remains differentiable with respect to matrix entries and parameters such as `mu`.
3. It is simple to batch with `jax.vmap`.

## JAX transformations used

The implemented package already relies on three important JAX features.

### `jit`

`solve_operator` and `solve_parameter_scan` are JIT compiled. This reduces repeated solve overhead once shapes are fixed and is the first step toward a performance-oriented interface.

### `vmap`

`solve_parameter_scan` builds a batch of operators

$$
\mathbf{M}_b = \mathbf{A} - \mu_b \mathbf{D}
$$

and solves them in parallel across a batch index.

### `grad`

Because the solve is written in JAX primitives, scalar objectives depending on the solved state can be differentiated. The repository's example and tests differentiate magnetic energy with respect to `mu`.

## What is not yet modeled

The current theory-to-code map stops at the dense linear operator level. It does not yet reconstruct:

- geometry integrals from `ma00aa.f90`
- outer nonlinear constraint updates from `ma02aa.f90`
- matrix-free or GMRES branches from `mp00ac.f90`
- coordinate-singularity-specific terms involving additional matrices such as `dME`

That boundary is intentional. The current package solves the most reusable and testable part first.
