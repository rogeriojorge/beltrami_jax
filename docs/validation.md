# Validation

## Validation philosophy

The package is validated against real dense systems exported from SPEC rather than only against synthetic toy matrices. This keeps the implementation anchored to the actual Fortran interface that motivated the JAX port.

## Current reference fixtures

The committed fixtures currently cover four dense systems exported from local SPEC runs:

- `g3v01l0fi_lvol1`
  - fixed-boundary toroidal plasma region
  - `lvol = 1`
  - matrix dimension `361`
  - `mu = 0.0`
- `g1v03l0fi_lvol2`
  - fixed-boundary cylindrical plasma region
  - `lvol = 2`
  - matrix dimension `51`
  - `mu = 1.0e-1`
- `g3v02l1fi_lvol1`
  - fixed-boundary 3D plasma region
  - `lvol = 1`
  - matrix dimension `361`
  - `mu = 1.8189908612531447e-4`
- `g3v02l0fr_lu_lvol3`
  - free-boundary toroidal vacuum region
  - `lvol = 3`
  - matrix dimension `1548`
  - `mu = 0.0`
  - `is_vacuum = 1`

The committed compressed fixtures live at:

- `src/beltrami_jax/data/g3v01l0fi_lvol1.npz`
- `src/beltrami_jax/data/g1v03l0fi_lvol2.npz`
- `src/beltrami_jax/data/g3v02l1fi_lvol1.npz`
- `src/beltrami_jax/data/g3v02l0fr_lu_lvol3.npz`

## Validation figures

The repository now commits regenerated publication-style validation assets:

![Validation panel](_static/validation_panel.png)

![Benchmark panel](_static/benchmark_panel.png)

The SPECTRE-facing validation tools also generate a coefficient-level HDF5 parity panel:

![SPECTRE vector-potential parity](_static/spectre_vecpot_parity.png)

The released SPECTRE linear-system fixtures generate a per-volume matrix/RHS/solution parity panel:

![SPECTRE linear-system parity](_static/spectre_linear_parity.png)

The SPECTRE interface-geometry evaluator generates a geometry/Jacobian/metric probe:

![SPECTRE geometry probe](_static/spectre_geometry_probe.png)

These figures summarize:

- coefficient-level agreement between SPEC and `beltrami_jax`
- operator, RHS, solution, and residual error metrics
- condition numbers, symmetry defects, and solution amplification
- autodiff agreement along a `mu` scan
- steady-state dense solve timings
- batched parameter-scan throughput
- SPECTRE `Ate`, `Aze`, `Ato`, and `Azo` HDF5 coefficient parity for public SPECTRE compare cases
- SPECTRE `dMA`, `dMD`, `dMB`, `dMG`, matrix, RHS, and solved degree-of-freedom parity for released SPECTRE volume solves
- SPECTRE interface geometry, Jacobian, and metric tensor evaluation from packaged TOML input
- SPECTRE `matrixBG` `dMB/dMG` boundary assembly parity for fixed-boundary fixtures and exact updated-normal-field source parity for all packaged fixtures

Release-gate example outputs generated from the current source tree:

![SPEC workflow panel](_static/spec_fixture_spectrum.png)

![Geometry workflow panel](_static/parameter_scan.png)

![Autodiff panel](_static/autodiff_gradient_check.png)

![Vacuum GMRES panel](_static/vacuum_gmres_panel.png)

## How the fixture was generated

The local SPEC checkout was instrumented with a temporary dump hook controlled by the environment variable:

```text
SPEC_DUMP_LINEAR_SYSTEM
```

Running SPEC with that variable set writes:

- `.meta.txt`
- `.dma.txt`
- `.dmd.txt`
- `.dmb.txt`
- `.dmg.txt`
- `.matrix.txt`
- `.rhs.txt`
- `.solution.txt`

Those files are then packaged with:

```bash
PYTHONPATH=src ./.venv/bin/python tools/build_spec_fixture.py \
  /path/to/G3V01L0Fi.dump.lvol1 \
  src/beltrami_jax/data/g3v01l0fi_lvol1.npz
```

## Checks currently performed

The test suite verifies:

### Fixture consistency

- the packaged arrays have the expected shapes
- the loaded metadata matches the expected fixture source

### Operator reconstruction

- `assemble_operator(system)` reproduces the dumped SPEC matrix exactly within floating-point tolerance
- `assemble_rhs(system)` reproduces the dumped SPEC right-hand side

### Solution regression

- the JAX dense solution matches the dumped SPEC solution
- the relative residual norm is near machine precision

### Autodiff

- magnetic energy computed from the solved state is differentiable with respect to `mu`
- the autodiff tangent is compared against a vmapped energy scan on the 3D plasma fixture

### Vectorization

- a batched `mu` scan reproduces the scalar solution at the reference `mu`
- batched parameter-scan timings are tracked as part of the committed benchmark figure

### Vacuum branch

- the vacuum right-hand-side path including `d_mg` behaves as expected on both a synthetic system and a dumped SPEC vacuum fixture

### Example smoke tests

- all example scripts execute successfully and print progress messages

### SPECTRE HDF5 coefficient validation

- `load_spectre_input_toml` normalizes SPECTRE TOML metadata including geometry flags, resolution, `Lrad`, flux arrays, constraints, free-boundary options, and Fourier boundary tables.
- `load_spectre_reference_h5` reads `vector_potential/Ate`, `Aze`, `Ato`, and `Azo` from SPECTRE reference files and transposes from SPECTRE HDF5 layout to radial-first Python layout.
- `compare_vector_potentials` reports component-wise and global relative errors and max absolute coefficient differences.
- `build_spectre_beltrami_layout_for_vector_potential` validates packed radial slices and identifies free-boundary exterior blocks from `Lrad`.
- `build_spectre_dof_layout_for_vector_potential` reconstructs SPECTRE's internal `gi00ab` mode order, `lregion` branch flags, `Ate/Aze/Ato/Azo` coefficient id maps, and `Lma` through `Lmh` multiplier id maps.
- Packaged SPECTRE vector potentials round-trip exactly through `pack_vector_potential` and `unpack_solutions`.
- The JAX pack/unpack methods are covered by an autodiff test that differentiates through packed per-volume solution vectors.
- `tools/export_spectre_vecpot_npz.py` runs from a SPECTRE environment and exports fresh coefficients from `spectre.get_vec_pot_flat`.
- The four public SPECTRE compare cases are packaged under `src/beltrami_jax/data/spectre_compare/`.
- `tools/generate_spectre_validation_assets.py --use-packaged` compares those packaged fresh exports against packaged SPECTRE `reference.h5` files and writes the committed parity figure.
- Omitting `--use-packaged` compares against a local SPECTRE checkout and local fresh exports when those are present.
- `solve_spectre_assembled_numpy` is tested as the thin adapter that SPECTRE can call once it has already assembled one Beltrami linear system.
- `solve_spectre_assembled_batch` is tested on equal-size SPECTRE plasma volumes so repeated same-shape solves have a vectorized path.
- `build_spectre_interface_geometry` parses SPECTRE `allrzrz` interface rows and free-boundary wall rows from TOML input into internal Fourier mode order.
- `interpolate_spectre_volume_geometry` is tested for coordinate-singularity and non-axis interpolation consistency.
- `evaluate_spectre_volume_coordinates` is tested for finite Jacobian/metric values, metric symmetry, and JAX autodiff through the radial interpolation coordinate.
- `build_spectre_boundary_normal_field` reconstructs SPECTRE's initialized `iVns/iBns/iVnc/iBnc` normal-field arrays from TOML tables.
- `assemble_spectre_matrix_bg` and `assemble_spectre_matrix_bg_from_input` reproduce SPECTRE `matrixBG` `dMB/dMG` for fixed-boundary public fixtures; for the free-boundary fixture the TOML-only path is tested as the initial source and the updated-normal-field path reproduces the packaged post-Picard fixture exactly.

Current public SPECTRE compare-case results:

- `G2V32L1Fi`: global relative coefficient error `3.30e-15`
- `G3V3L3Fi`: global relative coefficient error `1.51e-14`
- `G3V3L2Fi_stability`: global relative coefficient error `1.52e-14`
- `G3V8L3Free`: global relative coefficient error `2.79e-15`

Current SPECTRE pack/unpack checks:

- SPECTRE Fourier modes match `gi00ab` ordering, including the field-period-scaled toroidal `in` array.
- Positive coefficient and multiplier ids are unique and contiguous for every packed volume.
- Coordinate-singularity axis recombination removes the same `Ate/Ato` rows for `(m,ll)=(0,0)` and `(m,ll)=(1,1)` that SPECTRE removes in `preset_mod.F90`.
- Free-boundary exterior blocks are included in the packed layout.
- Non-stellarator-symmetric synthetic metadata exercises odd `Ato/Azo` maps even though the current public SPECTRE fixtures are stellarator-symmetric.

Programmatic access:

```python
from beltrami_jax import list_packaged_spectre_cases, load_packaged_spectre_case

for label in list_packaged_spectre_cases():
    case = load_packaged_spectre_case(label)
    print(label, case.comparison.global_relative_error)
```

### SPECTRE linear-system validation

The SPECTRE linear fixtures are exported from the released SPECTRE wrapper after
SPECTRE has assembled one volume's Beltrami system and called
`solve_beltrami_system`. Each fixture stores:

- `d_ma`, `d_md`, `d_mb`, and `d_mg`
- the dense SPECTRE matrix and RHS used by the solve
- the SPECTRE solved vector-potential degree-of-freedom vector
- volume metadata, branch flags, and residual norms

The package currently commits 19 per-volume systems from:

- `G2V32L1Fi`: four plasma volumes
- `G3V3L3Fi`: three plasma volumes
- `G3V3L2Fi_stability`: three plasma volumes
- `G3V8L3Free`: eight plasma volumes plus the vacuum exterior block

Current released-SPECTRE linear parity:

- operator relative error: exactly `0.0` for all packaged fixtures
- RHS relative error: exactly `0.0` for all packaged fixtures
- worst solution relative error: `1.59e-15`
- worst JAX relative residual norm: `2.56e-12`
- backend adapter solution parity: below `3e-12` for all packaged fixtures
- branch-solve solution parity: below `3e-12` for all packaged fixtures
- branch-solve primary residuals: below `1e-11` for all packaged fixtures, including the ill-conditioned compact free-boundary axis volume
- `matrixBG` fixed-boundary `dMB/dMG` parity: exact for all packaged fixed-boundary fixtures
- `matrixBG` updated-normal-field `dMB/dMG` parity: exact for all 19 packaged released-SPECTRE fixtures

Programmatic access:

```python
from beltrami_jax import (
    list_packaged_spectre_linear_systems,
    load_packaged_spectre_linear_system,
    solve_from_components,
)

for name in list_packaged_spectre_linear_systems():
    fixture = load_packaged_spectre_linear_system(name)
    result = solve_from_components(fixture.system)
    print(name, fixture.n_dof, float(result.relative_residual_norm))
```

### SPECTRE branch and constraint validation

The SPECTRE branch tests cover the local pieces ported from
`construct_beltrami_field` and `solve_beltrami_system`:

- `spectre_constraint_dof_count` matches SPECTRE's local `Nxdof` branch table for `Lconstraint = -2, -1, 0, 1, 2, 3`.
- `solve_spectre_beltrami_branch` reproduces all 19 packaged released-SPECTRE primary solution vectors.
- plasma, vacuum, and coordinate-singularity-current derivative right-hand-side formulas are checked explicitly.
- `evaluate_spectre_constraints` checks residual/Jacobian formulas for current, rotational-transform, helicity, no-iteration, and global-constraint branches using injected diagnostic arrays.

This validates the branch contract before the JAX-native field diagnostic layer is complete.

## Coverage target

The repository enforces a coverage threshold in `pyproject.toml`:

- required line coverage: at least 90%
- current release-gate result: `96 passed` with `93.23%` line coverage

## Known validation gaps

The current validation is strong for the implemented regression and internal-workflow stages, but still incomplete in project terms.

Remaining validation work includes:

- comparisons against later SPECTRE integration points beyond exported matrix/RHS/solution fixtures
- JAX-native generation of SPECTRE HDF5 vector-potential coefficients `vector_potential/Ate`, `Aze`, `Ato`, and `Azo`
- JAX-native assembly of SPECTRE `dMA` and `dMD` from the new geometry/metric layer
- JAX-native transform/current diagnostics from solved fields rather than injected diagnostic arrays
- broader 3D fixture coverage closer to anticipated SPECTRE use cases
- end-to-end SPECTRE fork integration once matrix assembly and diagnostics are complete

## Why exact dense regression matters

The most important current risk is accidentally drifting away from the exact discrete system that SPEC solves. Exact dense regression is therefore the right first milestone. Once that is stable, broader performance-oriented changes can be judged against a known-correct baseline.
