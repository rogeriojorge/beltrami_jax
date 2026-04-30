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

The package currently supports three complementary workflows:

- dumped SPEC regression, where the geometry-dependent matrices are loaded from a SPEC export
- internal Fourier-geometry prototype assembly, where `beltrami_jax` builds matrices for a shaped large-aspect-ratio torus and then runs the same solve machinery
- released SPECTRE validation, where TOML/HDF5 coefficient fixtures and per-volume matrix/RHS/solution fixtures are loaded from packaged public compare cases

The SPEC and SPECTRE fixture workflows are the current parity-oriented paths. The internal geometry workflow is a development and demonstration path. It is not yet a replacement for SPECTRE's full 3D interface-geometry assembly.

The package currently provides:

- `BeltramiLinearSystem`
  - holds the dense matrices and metadata required for the solve
- `FourierBeltramiGeometry`
  - parameterizes a shaped large-aspect-ratio torus used for prototype internal assembly
- `build_fourier_mode_basis`
  - constructs a packed cosine/sine basis with axis-safe radial powers
- `assemble_fourier_beltrami_system`
  - assembles internal `A`, `D`, `B`, and optional vacuum forcing terms
- `assemble_operator`
  - constructs $\mathbf{M}$
- `assemble_rhs`
  - constructs $\mathbf{r}$ for plasma and vacuum branches
- `solve_from_components`
  - assembles, solves, and returns diagnostics
- `gmres_solve`
  - provides a Krylov solve for dense or matrix-free operators
- `BeltramiProblem`
  - wraps geometry, basis, fluxes, and nonlinear target data
- `solve_helicity_constrained_equilibrium`
  - performs the outer nonlinear `mu` update to match a target helicity
- `solve_parameter_scan`
  - vectorized batched dense solves over varying `mu` and `psi`
- `load_spec_text_dump`
  - loads raw text exports from a SPEC run
- `load_packaged_reference`
  - loads a committed `.npz` regression fixture
- `load_spectre_input_toml`
  - loads SPECTRE TOML metadata into a normalized input summary
- `load_spectre_reference_h5`, `compare_vector_potentials`
  - load and compare SPECTRE HDF5 `Ate`, `Aze`, `Ato`, and `Azo` coefficient arrays
- `build_spectre_beltrami_layout_for_vector_potential`
  - maps SPECTRE `Lrad` metadata onto packed volume/exterior coefficient slices
- `build_spectre_dof_layout_for_vector_potential`
  - builds SPECTRE-compatible per-volume solution-vector maps for `Ate`, `Aze`, `Ato`, and `Azo`
- `load_packaged_spectre_case`
  - loads packaged public SPECTRE compare cases for reproducible coefficient-target tests
- `load_packaged_spectre_linear_system`
  - loads packaged released-SPECTRE matrix/RHS/solution fixtures for linear-solve parity tests
- `save_problem_json`, `load_problem_json`, `save_nonlinear_solution`
  - handle user-facing input and output files for standalone workflows

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
  - optional forcing vector $\mathbf{G}$ used by vacuum and selected SPECTRE coordinate-constraint branches
- `is_vacuum`
  - controls whether the operator uses the vacuum branch
- `include_d_mg_in_rhs`
  - controls whether `d_mg` is subtracted from the RHS source term

## Design choices

The current implementation makes a few deliberate choices.

### Dense plus GMRES

The package keeps the dense direct solve because it is the cleanest path for exact regression against dumped SPEC matrices, and it now also exposes a compact GMRES implementation for Krylov and matrix-free workflows.

### X64 enabled

`jax_enable_x64` is switched on so the comparison against SPEC's dense double-precision linear algebra is meaningful.

### Verbose examples

The example scripts print progress and diagnostics. This is intentional. The goal is to make debugging and validation transparent instead of having scripts appear to hang silently during JIT compilation or dense factorization.

## Current limitations

This package now performs prototype internal geometry assembly, Krylov solves, and a helicity-constrained outer loop, but it still does not cover:

- every SPEC/SPECTRE branch and auxiliary matrix path
- SPECTRE's exact interface-Fourier geometry input and integral assembly
- JAX-native HDF5 vector-potential coefficient generation matching `Ate`, `Aze`, `Ato`, and `Azo`
- JAX-native generation of the released SPECTRE linear-system matrices from TOML/interface geometry
- full sparse production scaling
- the broader equilibrium and constraint machinery beyond the supported Beltrami workflow

Those topics are documented in {doc}`limitations`.
