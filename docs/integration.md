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
- load packaged public SPECTRE compare cases for reproducible CI validation
- reconstruct SPECTRE's internal Fourier mode order and packed radial block layout
- pack and unpack SPECTRE-compatible per-volume solution vectors to/from `Ate`, `Aze`, `Ato`, and `Azo`
- load released SPECTRE per-volume Beltrami linear systems and solve them with JAX
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
- `spectre_fourier_modes`
- `list_packaged_spectre_cases`
- `load_packaged_spectre_case`
- `list_packaged_spectre_linear_systems`
- `load_packaged_spectre_linear_system`

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
after SPECTRE's Fortran geometry assembly has run. It does not yet remove
SPECTRE's Fortran assembly path. The final integration lane is to replace the
upstream construction of `dMA`, `dMD`, `dMB`, and `dMG` with JAX-native
SPECTRE interface-geometry assembly, then unpack the solved vectors through the
existing SPECTRE `Ate/Aze/Ato/Azo` maps.

## Current boundary

The integration boundary is strong enough to ship for the supported assembled-system, prototype internal-geometry, SPECTRE coefficient-validation, and SPECTRE solution-vector packing models, but it is still not a full SPECTRE backend. The main remaining work is JAX-native SPECTRE interface-geometry assembly and full branch-specific constraint logic.

See the root-level `SPECTRE_MIGRATION_PLAN.md` for the current SPECTRE replacement plan.
