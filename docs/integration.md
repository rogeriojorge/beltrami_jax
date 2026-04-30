# Integration Guide

## Purpose

`beltrami_jax` is meant to be usable from more than one upstream environment:

- direct regression against SPEC dumps
- geometry-driven Python workflows
- future SPECTRE integration
- optimization, inverse-design, and analysis loops that need autodiff through the Beltrami solve

The package therefore exposes multiple entry points rather than forcing every caller through a single API.

## Integration levels

### Level 1: dumped SPEC systems

Use this when another code can already export SPEC-style dense arrays.

```python
from beltrami_jax import load_spec_text_dump, solve_from_components

reference = load_spec_text_dump("/path/to/run.dump.lvol1")
result = solve_from_components(reference.system, method="dense", verbose=True)
```

This is the most direct way to validate agreement against legacy Fortran output.

### Level 2: direct component injection

Use this when another code has already assembled matrices and flux operators but does not use SPEC dump files.

```python
from beltrami_jax import BeltramiLinearSystem, solve_from_components

system = BeltramiLinearSystem.from_arraylike(
    d_ma=d_ma,
    d_md=d_md,
    d_mb=d_mb,
    mu=mu,
    psi=psi,
    d_mg=d_mg,
    is_vacuum=is_vacuum,
    label="external_code_region",
)
result = solve_from_components(system, method="gmres", tolerance=1.0e-10)
```

This is the natural entry point for C++, Fortran, or Python codes that already know how to assemble their own discrete operators.

### Level 3: internal prototype geometry assembly

Use this when the upstream code wants `beltrami_jax` to handle the current prototype assembly itself. This path uses `FourierBeltramiGeometry`, which is a shaped large-aspect-ratio torus model. It is not yet SPECTRE's full arbitrary 3D interface-Fourier geometry model.

```python
from beltrami_jax import (
    FourierBeltramiGeometry,
    build_fourier_mode_basis,
    assemble_fourier_beltrami_system,
    solve_from_components,
)

geometry = FourierBeltramiGeometry(
    major_radius=3.0,
    minor_radius=1.0,
    elongation=1.25,
    triangularity=0.15,
    field_periods=2,
)
basis = build_fourier_mode_basis(max_radial_order=1, max_poloidal_mode=2, max_toroidal_mode=1)
assembly = assemble_fourier_beltrami_system(geometry, basis, mu=0.04, psi=(0.1, 0.0))
result = solve_from_components(assembly.system)
```

### Level 4: nonlinear helicity-constrained workflow

Use this when the upstream code wants a compact end-to-end Beltrami solve with outer `mu` updates.

```python
from beltrami_jax import (
    BeltramiProblem,
    FourierBeltramiGeometry,
    build_fourier_mode_basis,
    solve_helicity_constrained_equilibrium,
)

geometry = FourierBeltramiGeometry(major_radius=3.0, minor_radius=1.0)
basis = build_fourier_mode_basis(max_radial_order=1, max_poloidal_mode=2, max_toroidal_mode=1)
problem = BeltramiProblem.from_arraylike(
    geometry=geometry,
    basis=basis,
    psi=(0.1, 0.0),
    target_helicity=0.05,
    initial_mu=0.02,
    solver="gmres",
)
result = solve_helicity_constrained_equilibrium(problem)
```

## How this maps to different codes

### SPEC

Current use:

- export linear systems
- validate exact operator, RHS, and solution agreement

Main entry points:

- `load_spec_text_dump`
- `load_packaged_reference`
- `solve_from_components`

### SPECTRE

Current supported use:

- read and validate SPECTRE TOML input metadata
- read SPECTRE HDF5 vector-potential coefficients
- compare fresh SPECTRE coefficient exports against `reference.h5`
- evaluate SPECTRE `allrzrz`/wall interface geometry in JAX, including volume interpolation, Jacobian, and metric tensor
- assemble SPECTRE `matrixBG` boundary terms `dMB/dMG` from packed maps plus TOML or updated normal-field arrays
- assemble SPECTRE `dMA/dMD/dMB/dMG` volume matrices directly from TOML/interface geometry for the packaged validated branches
- solve one SPECTRE volume from TOML/interface geometry and unpack directly to `Ate/Aze/Ato/Azo` with `solve_spectre_volume_from_input`
- solve all packed SPECTRE volumes from TOML/interface geometry and return one full vector-potential block with `solve_spectre_volumes_from_input`
- load packaged public SPECTRE compare cases for reproducible CI validation
- reconstruct SPECTRE's internal Fourier mode order and packed radial block layout
- pack and unpack SPECTRE-compatible per-volume solution vectors to/from `Ate`, `Aze`, `Ato`, and `Azo`
- load released SPECTRE per-volume Beltrami linear systems and solve them with JAX
- call a narrow JIT-backed backend adapter for already assembled SPECTRE Beltrami matrices
- solve SPECTRE local Beltrami branches including derivative right-hand sides
- evaluate the local `Lconstraint` residual/Jacobian table once transform/current diagnostics are supplied
- use the comparison tooling as the target contract for the future JAX-native backend

Intended future use:

- replace or simplify legacy Fortran-interface Beltrami solve components
- keep the Beltrami kernel differentiable and easier to test
- exchange SPECTRE interface geometry, basis metadata, fluxes, constraints, and branch flags

Current safe entry points:

- `BeltramiLinearSystem`
- `solve_from_components`
- `load_spectre_input_toml`
- `load_spectre_reference_h5`
- `load_spectre_vector_potential_npz`
- `compare_vector_potentials`
- `build_spectre_beltrami_layout_for_vector_potential`
- `build_spectre_dof_layout_for_vector_potential`
- `build_spectre_interface_geometry`
- `interpolate_spectre_volume_geometry`
- `evaluate_spectre_volume_coordinates`
- `build_spectre_boundary_normal_field`
- `assemble_spectre_matrix_bg`
- `assemble_spectre_matrix_bg_from_input`
- `assemble_spectre_matrix_ad_from_input`
- `assemble_spectre_volume_matrices_from_input`
- `spectre_volume_flux_vector`
- `solve_spectre_volume_from_input`
- `solve_spectre_volumes_from_input`
- `solve_spectre_toml`
- `spectre_fourier_modes`
- `list_packaged_spectre_cases`
- `load_packaged_spectre_case`
- `list_packaged_spectre_linear_systems`
- `load_packaged_spectre_linear_system`
- `solve_spectre_assembled`
- `solve_spectre_assembled_numpy`
- `solve_spectre_assembled_batch`
- `solve_spectre_beltrami_branch`
- `evaluate_spectre_constraints`

Prototype-only entry points:

- `BeltramiProblem`
- `assemble_fourier_beltrami_system`
- `solve_helicity_constrained_equilibrium`

These prototype entry points are not yet the final SPECTRE backend API.

### Optimization or inverse-design codes

Use:

- `solve_from_components`
- `solve_parameter_scan`
- `magnetic_energy`
- `magnetic_helicity`
- JAX autodiff with `grad`, `jit`, and `vmap`

The examples in `examples/autodiff_mu.py` and `examples/parameter_scan.py` show the intended style.

## Output contracts

The main solve outputs include:

- operator and RHS arrays
- solved coefficients
- magnetic energy
- magnetic helicity
- residual norms
- solve method and iteration count

For nonlinear runs, the output also includes:

- `mu_history`
- `helicity_history`
- `constraint_residual_history`

These are the fields most likely to matter to a caller that needs monitoring, diagnostics, or coupling to a larger equilibrium loop.

## File-based exchange

For file-based workflows, the package provides:

- `save_problem_json`
- `load_problem_json`
- `save_nonlinear_solution`
- `load_saved_solution`

These helpers are useful when one code prepares the problem and another code performs the solve or postprocessing.

## SPECTRE coefficient-validation workflow

Export fresh SPECTRE coefficients from the SPECTRE environment:

```bash
OMP_NUM_THREADS=1 DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib \
  /Users/rogerio/local/spectre/.venv/bin/python tools/export_spectre_vecpot_npz.py \
  /Users/rogerio/local/spectre/tests/compare/G2V32L1Fi/input.toml \
  examples/_generated/spectre_vecpot_exports/G2V32L1Fi.npz
```

Compare and plot from the `beltrami_jax` environment:

```bash
PYTHONPATH=src ./.venv/bin/python tools/generate_spectre_validation_assets.py --use-packaged
```

This is intentionally separated from the JAX-native assembly work. It validates
the coefficient layout, HDF5 orientation, free-boundary update convention, and
comparison metrics that the replacement backend must reproduce.

## SPECTRE TOML-to-coefficients workflow

The highest-level linear SPECTRE path currently starts from a SPECTRE TOML
summary and returns a full SPECTRE-compatible vector-potential coefficient
block:

```python
from beltrami_jax import load_spectre_input_toml, solve_spectre_volumes_from_input

summary = load_spectre_input_toml("/path/to/input.toml")
result = solve_spectre_volumes_from_input(summary, verbose=True)

print(result.vector_potential.ate.shape)
print(float(result.max_relative_residual_norm))
```

For exact validation against a completed SPECTRE run, pass the post-constraint
branch state:

```python
from beltrami_jax import load_packaged_spectre_case, load_packaged_spectre_linear_system

case = load_packaged_spectre_case("G3V3L3Fi")
mu = {}
psi = {}
for lvol in range(1, case.input_summary.packed_volume_count + 1):
    fixture = load_packaged_spectre_linear_system(case_label=case.label, volume_index=lvol)
    mu[lvol] = fixture.system.mu
    psi[lvol] = fixture.system.psi

result = solve_spectre_volumes_from_input(case.input_summary, mu=mu, psi=psi)
```

The explicit `mu`/`psi` injection is a validation bridge, not the desired final
SPECTRE contract. The remaining backend work is to compute the transform/current
diagnostics and nonlinear updates directly in JAX so this call can reproduce the
post-constraint state from TOML alone.

## SPECTRE pack/unpack workflow

SPECTRE stores one vector-potential coefficient array per component, but the
Beltrami solve itself uses one solution vector per packed volume. The integer
maps are built in SPECTRE's `preset_mod.F90` and used by `packab`. The
corresponding `beltrami_jax` API is:

```python
from beltrami_jax import (
    build_spectre_dof_layout_for_vector_potential,
    load_packaged_spectre_case,
)

case = load_packaged_spectre_case("G3V3L3Fi")
dof_layout = build_spectre_dof_layout_for_vector_potential(
    case.input_summary,
    case.reference.vector_potential,
)
solutions = dof_layout.pack_vector_potential(case.reference.vector_potential)
roundtrip = dof_layout.unpack_solutions(solutions)
print(dof_layout.solution_sizes)
print(roundtrip.shape)
```

For differentiable coupling, use `pack_vector_potential_jax` and
`unpack_solutions_jax` with dictionaries containing `ate`, `aze`, `ato`, and
`azo` arrays. These methods use JAX scatter/gather operations, so scalar
objectives downstream of the packed solution vectors remain compatible with
`jax.grad`.

## SPECTRE `matrixBG` boundary assembly workflow

SPECTRE's `matrices_mod.F90::matrixBG` constructs the flux-coupling matrix
`dMB` and the boundary-normal-field source vector `dMG`. This low-dimensional
piece is now available without calling SPECTRE:

```python
from beltrami_jax import (
    assemble_spectre_matrix_bg_from_input,
    load_spectre_input_toml,
)

summary = load_spectre_input_toml("input.toml")
matrix_bg = assemble_spectre_matrix_bg_from_input(summary, lvol=1)
print(matrix_bg.d_mb.shape, matrix_bg.d_mg.shape)
```

For free-boundary calculations, SPECTRE can update `iBns/iBnc` during the
Picard boundary-normal-field iteration. The TOML-only helper reconstructs the
initial source. For exact post-update parity, pass the live updated arrays:

```python
from beltrami_jax import (
    SpectreBoundaryNormalField,
    assemble_spectre_matrix_bg,
    build_spectre_dof_layout,
)

dof_layout = build_spectre_dof_layout(summary)
normal_field = SpectreBoundaryNormalField(
    ivns=updated_ivns,
    ibns=updated_ibns,
    ivnc=updated_ivnc,
    ibnc=updated_ibnc,
)
matrix_bg = assemble_spectre_matrix_bg(dof_layout.volume_maps[lvol - 1], normal_field)
```

This closes the `dMB/dMG` assembly lane.

## SPECTRE `dMA/dMD` volume-matrix assembly workflow

The next SPECTRE Fortran-removal ingredient is also available for the
validated branches: radial basis/quadrature, metric-integral assembly, and the
`matrices_mod.F90::matrix` contraction into `dMA/dMD`.

```python
from beltrami_jax import (
    assemble_spectre_matrix_ad_from_input,
    assemble_spectre_volume_matrices_from_input,
    load_spectre_input_toml,
)

summary = load_spectre_input_toml("input.toml")

# dMA/dMD only
matrix_ad = assemble_spectre_matrix_ad_from_input(summary, lvol=2)
print(matrix_ad.d_ma.shape, matrix_ad.d_md.shape)

# dMA/dMD plus matrixBG's dMB/dMG
matrices = assemble_spectre_volume_matrices_from_input(summary, lvol=2)
print(matrices.d_ma.shape, matrices.d_mb.shape)
```

The current exact-parity tests cover:

- cylindrical `Igeometry=2` coordinate-singularity and bulk volumes generated from `Linitialize=1` TOML data
- toroidal `Igeometry=3` generated-interface volumes using SPECTRE-compatible centroid `rzaxis` axis initialization
- explicit-interface free-boundary non-axis volumes from `physics.allrzrz`
- explicit-interface free-boundary coordinate-axis volumes where SPECTRE recomputes the axis during geometry unpacking
- the free-boundary exterior/vacuum volume
- a TOML-driven solve/unpack path where JAX-assembled matrices reproduce a packaged SPECTRE `Ate/Aze/Ato/Azo` block when supplied the same solved branch `mu` and flux vector used by SPECTRE

The important remaining backend gaps are no longer the packaged `dMA/dMD`
matrix branches. They are the field diagnostics and nonlinear constraint
updates needed to obtain final SPECTRE `mu`/flux values without injecting them
from a SPECTRE run, plus broader non-stellarator-symmetric fixture coverage.

## SPECTRE TOML-to-coefficients volume solve

For a supported branch, `solve_spectre_volume_from_input` performs the local
replacement workflow: read a `SpectreInputSummary`, assemble all four matrix
ingredients, solve the local linear system, and unpack the result to
SPECTRE-compatible `Ate/Aze/Ato/Azo` arrays.

```python
from beltrami_jax import load_spectre_input_toml, solve_spectre_volume_from_input

summary = load_spectre_input_toml("input.toml")
result = solve_spectre_volume_from_input(summary, lvol=2, verbose=True)

print(result.solution.shape)
print(result.vector_potential.ate.shape)
print(float(result.relative_residual_norm))
```

For exact validation against a completed SPECTRE run, pass the post-constraint
`mu` and flux vector used by SPECTRE:

```python
result = solve_spectre_volume_from_input(
    summary,
    lvol=2,
    mu=spectre_mu,
    psi=(spectre_dtflux, spectre_dpflux),
)
```

Without those overrides, the helper uses the normalized TOML initial state. That
is the right user-facing default, but it will not reproduce a post-Newton
SPECTRE fixture until the JAX transform/current diagnostics and nonlinear
constraint loop are complete.

## SPECTRE linear-system validation workflow

When SPECTRE has already assembled `dMA`, `dMD`, `dMB`, and `dMG`, the current
JAX solve path can reproduce SPECTRE's dense Beltrami solve directly:

```python
from beltrami_jax import (
    load_packaged_spectre_linear_system,
    solve_from_components,
)

fixture = load_packaged_spectre_linear_system("G3V8L3Free/lvol9")
result = solve_from_components(fixture.system, verbose=True)

print(fixture.matrix.shape)
print(fixture.relative_residual_norm)
print(float(result.relative_residual_norm))
```

This workflow validates the linear algebra and branch-specific RHS assembly
after SPECTRE's Fortran geometry assembly has run. For the branches listed
above, the upstream construction of `dMA`, `dMD`, `dMB`, and `dMG` can now be
done by `beltrami_jax`; the remaining integration lane is to compute
transform/current diagnostics from solved JAX fields and drive the nonlinear
constraint updates without asking SPECTRE for final `mu`/flux metadata.

## SPECTRE interface-geometry workflow

The first JAX-native SPECTRE assembly ingredient is the coordinate evaluator.
It consumes SPECTRE TOML metadata and returns the same type of real-space
geometry quantities used by SPECTRE's `compute_coordinates`/metric path:

```python
from beltrami_jax import (
    build_spectre_interface_geometry,
    evaluate_spectre_volume_coordinates,
    interpolate_spectre_volume_geometry,
    load_packaged_spectre_case,
)

case = load_packaged_spectre_case("G3V8L3Free")
geometry = build_spectre_interface_geometry(case.input_summary)
volume = interpolate_spectre_volume_geometry(geometry, lvol=2, s=0.0)
grid = evaluate_spectre_volume_coordinates(volume, theta=[0.0, 1.0], zeta=[0.0])

print(geometry.interface_count)
print(grid.jacobian)
print(grid.metric.shape)
```

Implemented at this layer:

- SPECTRE internal Fourier mode order.
- Axis plus `physics.allrzrz.interface_*` rows.
- Free-boundary wall rows from `rwc/zws/rws/zwc`.
- Coordinate-singularity interpolation for `Igeometry == 2` and `Igeometry == 3`.
- Linear interpolation between neighboring interfaces for non-axis volumes.
- First derivatives, Jacobian, inverse Jacobian, and covariant metric tensor.

This geometry layer now feeds the SPECTRE radial/metric-integral and
`dMA/dMD` contraction path described above for the validated branches.

## SPECTRE branch/constraint workflow

The SPECTRE local solve has branch-specific derivative right-hand sides and an
outer `Lconstraint` residual table. These are now represented explicitly:

```python
from beltrami_jax import (
    SpectreConstraintDiagnostics,
    SpectreConstraintTargets,
    evaluate_spectre_constraints,
    load_packaged_spectre_linear_system,
    solve_spectre_beltrami_branch,
)

fixture = load_packaged_spectre_linear_system("G2V32L1Fi/lvol2")
branch = solve_spectre_beltrami_branch(
    d_ma=fixture.system.d_ma,
    d_md=fixture.system.d_md,
    d_mb=fixture.system.d_mb,
    d_mg=fixture.system.d_mg,
    mu=fixture.system.mu,
    psi=fixture.system.psi,
    lconstraint=fixture.lconstraint,
    is_vacuum=fixture.is_vacuum,
    coordinate_singularity=fixture.coordinate_singularity,
)

constraints = evaluate_spectre_constraints(
    SpectreConstraintTargets(lconstraint=1, is_vacuum=False, iota_inner=0.9, iota_outer=0.8),
    SpectreConstraintDiagnostics(rotational_transform=[[0.91, 0.1, 0.2], [0.82, 0.3, 0.4]]),
)

print(branch.branch_unknowns)
print(constraints.residual)
print(constraints.jacobian)
```

This makes the branch contract testable before the geometry-derived field
diagnostics are complete.

## Minimal SPECTRE backend adapter

The smallest safe SPECTRE-side change is a thin optional adapter after SPECTRE
has already assembled `dMA`, `dMD`, `dMB`, and `dMG`. It should not change
SPECTRE's geometry representation, quadrature, matrix assembly, or default
Fortran backend.

The adapter call looks like this on the Python side:

```python
from beltrami_jax import solve_spectre_assembled_numpy

result = solve_spectre_assembled_numpy(
    d_ma=d_ma,
    d_md=d_md,
    d_mb=d_mb,
    d_mg=d_mg,
    mu=mu,
    psi=(delta_psi_t, delta_psi_p),
    is_vacuum=is_vacuum,
    include_d_mg_in_rhs=is_vacuum or coordinate_singularity_current_constraint,
)
solution = result["solution"]
relative_residual_norm = result["relative_residual_norm"]
```

Recommended SPECTRE-side patch size:

- Add one experimental option such as `beltrami_backend = "fortran" | "jax"`, with `"fortran"` as the default.
- Add one Python helper that converts the already assembled Fortran arrays to NumPy views/copies and calls `solve_spectre_assembled_numpy`.
- Copy the returned solution vector into the same SPECTRE solution slot that `solve_beltrami_system` fills today.
- Keep all existing Fortran branches, diagnostics, and tests active as the fallback path.

Performance notes:

- The core solve is JIT compiled by shape and branch flags.
- Repeated calls with the same active degree-of-freedom count reuse the compiled solve.
- `solve_spectre_assembled_batch` can solve equal-size volumes together when SPECTRE has multiple volumes with the same branch flags.
- No absolute runtime threshold is hard-coded in CI because GitHub-hosted CPU timing is noisy; tests validate parity, residuals, batching, NumPy adapter behavior, and timing-helper behavior.

## Current boundary

The integration boundary is strong enough to ship for the supported assembled-system, prototype internal-geometry, SPECTRE coefficient-validation, SPECTRE solution-vector packing, SPECTRE interface-geometry evaluation, SPECTRE matrix assembly, TOML-driven per-volume coefficient solves, and SPECTRE local branch/constraint models, but it is still not a full SPECTRE backend. The main remaining work is field diagnostics that produce transform/current constraints directly from solved JAX fields and the nonlinear update loop that removes final `mu`/flux injection.

See the root-level `SPECTRE_MIGRATION_PLAN.md` for the current SPECTRE replacement plan.
