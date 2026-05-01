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

The package currently supports four complementary workflows:

- dumped SPEC regression, where the geometry-dependent matrices are loaded from a SPEC export
- internal Fourier-geometry prototype assembly, where `beltrami_jax` builds matrices for a shaped large-aspect-ratio torus and then runs the same solve machinery
- released SPECTRE validation, where TOML/HDF5 coefficient fixtures and per-volume matrix/RHS/solution fixtures are loaded from packaged public compare cases
- SPECTRE interface-geometry and matrix assembly, where TOML Fourier interfaces, free-boundary wall tables, radial bases, metric integrals, and SPECTRE volume matrices are assembled in JAX

The SPEC and SPECTRE fixture workflows are the current parity-oriented paths. The internal geometry workflow is a development and demonstration path. The SPECTRE path is now the intended replacement route for the Beltrami matrix/linear-solve stage, with fixed-boundary and free-boundary global `Lconstraint=3` now covered for the public validation cases. Remaining work is focused on the virtual-casing/free-boundary normal-field update, broader diagnostic branches, and production scaling.

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
- `solve_spectre_assembled_numpy`
  - provides the minimal NumPy-returning SPECTRE adapter for already assembled Beltrami matrices, including derivative solves and energy/helicity integrals
- `solve_spectre_beltrami_branch`
  - ports SPECTRE's local branch solve, including derivative right-hand sides used by constraint Jacobians
- `evaluate_spectre_constraints`
  - evaluates the `Lconstraint` residual/Jacobian branch table using SPECTRE-style transform/current diagnostics
- `build_spectre_interface_geometry`, `interpolate_spectre_volume_geometry`, `evaluate_spectre_volume_coordinates`
  - provide the first JAX-native SPECTRE interface-geometry layer: Fourier interfaces, volume interpolation, coordinates, Jacobian, and metric tensor
- `build_spectre_boundary_normal_field`, `assemble_spectre_matrix_bg`, `assemble_spectre_matrix_bg_from_input`
  - port SPECTRE `matrixBG`, producing `dMB` and `dMG` from packed maps plus TOML or updated normal-field arrays
- `chebyshev_basis`, `zernike_basis`, `assemble_spectre_metric_integrals_from_input`, `assemble_spectre_matrix_ad_from_input`
  - port the SPECTRE radial basis, metric-integral, and `dMA/dMD` matrix-contraction path for the packaged cylindrical, toroidal, free-boundary, and vacuum branches
- `assemble_spectre_volume_matrices_from_input`
  - assembles `dMA`, `dMD`, `dMB`, and `dMG` for one SPECTRE volume directly from TOML/interface geometry when the branch is supported
- `solve_spectre_volume_from_input`
  - assembles one SPECTRE volume, solves it, unpacks directly to `Ate/Aze/Ato/Azo`, and exposes derivative vector-potential blocks
- `solve_spectre_volumes_from_input`, `solve_spectre_toml`
  - solve selected or all packed SPECTRE volumes from TOML/interface geometry and concatenate a full `Ate/Aze/Ato/Azo` coefficient block
- `compute_spectre_plasma_current`
  - evaluates SPECTRE-style toroidal and poloidal current diagnostics from solved coefficient blocks
- `compute_spectre_rotational_transform`
  - evaluates SPECTRE-style straight-field-line rotational transform diagnostics from solved coefficient blocks for validated stellarator-symmetric Fourier branches
- `compute_spectre_btheta_mean`
  - evaluates the SPECTRE `lbpol` mean covariant `B_theta` diagnostic used by global-current constraints
- `evaluate_spectre_helicity_constraint`, `evaluate_spectre_local_constraints`
  - evaluate local SPECTRE constraint residuals/Jacobians from TOML targets and JAX diagnostics
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

This package now performs prototype internal geometry assembly, Krylov solves, a helicity-constrained outer loop, SPECTRE matrix assembly, SPECTRE current diagnostics, SPECTRE `B_theta` diagnostics, SPECTRE rotational-transform diagnostics for validated stellarator-symmetric Fourier cases, selected local SPECTRE constraint updates, and fixed-boundary/free-boundary global `Lconstraint=3`, but it still does not cover:

- every SPEC/SPECTRE branch and auxiliary matrix path
- the virtual-casing/free-boundary normal-field Picard update without live SPECTRE normal-field metadata
- non-stellarator-symmetric SPECTRE rotational-transform/current diagnostic branches
- broader non-stellarator-symmetric and high-resolution fixture coverage
- full sparse production scaling
- the broader equilibrium and constraint machinery beyond the supported Beltrami workflow

Those topics are documented in {doc}`limitations`.
