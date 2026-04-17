# Overview

## Problem statement

The SPEC/SPECTRE Beltrami stage solves a linear system for the packed coefficients of the magnetic vector potential in one relaxed region. In the notation used by SPEC's `matrix.f90` and `mp00ac.f90`,

$$
\mathbf{M} = \mathbf{A} - \mu \mathbf{D}.
$$

For plasma regions,

$$
\mathbf{r} = -\mathbf{B}\boldsymbol{\psi},
$$

while for vacuum regions the discrete forcing becomes

$$
\mathbf{r} = -\mathbf{G} - \mathbf{B}\boldsymbol{\psi}.
$$

The solution vector $\mathbf{a}$ contains packed vector-potential coefficients. Once solved, SPEC also evaluates quadratic diagnostics such as magnetic energy and helicity.

## Implemented model in `beltrami_jax`

The current package starts at the point where the geometry-dependent matrices are already available. That is the correct boundary for the first JAX port because it isolates the linear algebra that SPECTRE can reuse without first reimplementing the entire geometry/integration machinery of SPEC.

The package currently provides:

- `BeltramiLinearSystem`
  - holds the dense matrices and metadata required for the solve
- `assemble_operator`
  - constructs $\mathbf{M}$
- `assemble_rhs`
  - constructs $\mathbf{r}$ for plasma and vacuum branches
- `solve_from_components`
  - assembles, solves, and returns diagnostics
- `solve_parameter_scan`
  - vectorized batched dense solves over varying `mu` and `psi`
- `load_spec_text_dump`
  - loads raw text exports from a SPEC run
- `load_packaged_reference`
  - loads a committed `.npz` regression fixture

## What is inside a `BeltramiLinearSystem`

The dataclass stores:

- `d_ma`
  - the dense quadratic magnetic-energy matrix $\mathbf{A}$
- `d_md`
  - the dense quadratic helicity matrix $\mathbf{D}$
- `d_mb`
  - the dense flux-to-right-hand-side operator $\mathbf{B}$
- `mu`
  - the Beltrami multiplier for the region
- `psi`
  - the two-component flux vector used by the linear stage
- `d_mg`
  - optional vacuum forcing vector $\mathbf{G}$
- `is_vacuum`
  - controls whether the operator uses the vacuum branch

## Design choices

The current implementation makes a few deliberate choices.

### Dense first

SPEC supports more than one solve path, including iterative solvers. The initial JAX port keeps the model dense and exact because that is the cleanest path for validating operator assembly against dumped SPEC matrices.

### X64 enabled

`jax_enable_x64` is switched on so the comparison against SPEC's dense double-precision linear algebra is meaningful.

### Verbose examples

The example scripts print progress and diagnostics. This is intentional. The goal is to make debugging and validation transparent instead of having scripts appear to hang silently during JIT compilation or dense factorization.

## Current limitations

This package does not yet perform:

- geometry/integral assembly from boundary/interface information
- nonlinear constraint updates
- rotational-transform or current-constraint iterations
- sparse scaling strategies for larger systems

Those topics are documented in {doc}`limitations`.
