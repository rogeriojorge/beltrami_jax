# SPECTRE Migration and User-Facing Parity Plan

Date: 2026-04-30

This document captures the current `beltrami_jax` state, the SPECTRE release assessment, the gaps raised by collaborator feedback, and the plan to make `beltrami_jax` a credible replacement for the SPEC/SPECTRE Beltrami backend.

It is intentionally explicit. If work is restarted from scratch, this file should be enough to reconstruct the goals, local setup, validation targets, source-code map, and next decisions.

## 1. Executive Summary

The current `beltrami_jax` repository has a useful tested kernel, but it is not yet a full SPECTRE Beltrami replacement.

What currently works:

- It solves SPEC-style dense Beltrami systems of the form `M a = r`, with `M = A - mu D`.
- It supports plasma and vacuum RHS branches using `d_ma`, `d_md`, `d_mb`, and optional `d_mg`.
- It has packaged fixtures from instrumented SPEC text dumps and verifies operator, RHS, and packed-solution agreement for those dumped systems.
- It has an internal geometry-driven prototype with a shaped large-aspect-ratio torus, basis construction, dense/GMRES solves, simple helicity-target iteration, examples, plots, docs, CI, and coverage.
- It now reads SPECTRE TOML input summaries and SPECTRE HDF5 vector-potential coefficients.
- It now compares fresh SPECTRE `get_vec_pot_flat` exports against public SPECTRE `reference.h5` files and generates a reviewer-facing parity figure.
- It now reconstructs SPECTRE's Fourier mode order, per-volume coefficient id maps, and `packab`-compatible solution-vector layout for `Ate/Aze/Ato/Azo`.
- It now packages released-SPECTRE per-volume Beltrami linear systems and validates JAX matrix/RHS/solution parity for 19 volume solves.
- It now exposes a narrow SPECTRE backend adapter for already assembled `dMA/dMD/dMB/dMG` arrays, so the first SPECTRE-side experiment can be a small optional branch rather than a Fortran rewrite.
- It now assembles SPECTRE `dMA/dMD/dMB/dMG` directly from TOML/interface geometry for the packaged cylindrical, toroidal, free-boundary, and vacuum branches.
- It now solves a SPECTRE volume from TOML/interface geometry and unpacks directly to `Ate/Aze/Ato/Azo` when supplied the same post-constraint `mu`/flux state used by SPECTRE.

What is not yet done:

- It does not yet compute SPECTRE rotational-transform/current diagnostics directly from the solved JAX fields.
- It does not yet run the full nonlinear local/global constraint loop that updates `mu`, `dtflux`, and `dpflux` without SPECTRE metadata injection.
- It does not yet cover enough non-stellarator-symmetric and high-resolution fixtures to claim full SPECTRE backend replacement.
- The docs have been tightened, but must continue to avoid claiming SPECTRE backend replacement before JAX-native coefficient parity exists.

Recommended immediate direction:

- Treat `beltrami_jax` as a strict SPECTRE Beltrami backend project, not as a loosely similar Beltrami solver.
- Next implement JAX-native transform/current diagnostics and feed them into the existing `Lconstraint` branch layer.
- Only after post-constraint coefficient parity is established without injected SPECTRE metadata should we attempt the SPECTRE fork replacement branch.

## 2. Direct Answers to the Collaborator's Email

### 2.1 What is the input to the code?

Current answer:

- `beltrami_jax` has three input modes today.
- Mode 1 is the regression/developer mode: load packaged fixtures or text dumps containing already assembled SPEC-style matrices and vectors: `d_ma`, `d_md`, `d_mb`, optional `d_mg`, `mu`, `psi`, RHS, matrix, and expected solution.
- Mode 2 is the internal geometry prototype: create a `FourierBeltramiGeometry`, build a `FourierModeBasis`, assemble a toy/SPEC-style linear system internally, and solve it.
- Mode 3 is the SPECTRE TOML/interface-geometry mode: load `SpectreInputSummary`, assemble SPECTRE-compatible volume matrices, solve one volume with `solve_spectre_volume_from_input`, and unpack to `Ate/Aze/Ato/Azo`.

Important correction:

- The current geometry-driven mode is not equivalent to SPECTRE or SPEC arbitrary 3D interface input.
- It uses a shaped large-aspect-ratio torus parameterization in `src/beltrami_jax/geometry.py`.
- The user-facing SPECTRE target now exists for per-volume linear solves, but the final nonlinear diagnostic loop still needs to be ported before it can reproduce post-constraint SPECTRE runs from TOML alone.

Final target input contract:

- `BeltramiFromSpectreInput`: parse SPECTRE TOML or a SPECTRE object/state.
- `SpectreInterfaceGeometry`: hold `Igeometry`, `Nfp`, `Mvol`, `Mpol`, `Ntor`, `Lrad`, `im`, `in`, interface `R/Z` Fourier coefficients, fluxes, current/helicity/iota constraints, stellarator symmetry flags, free-boundary flags, and numerical resolution.
- `BeltramiLinearSystem`: remain as a low-level escape hatch for already assembled matrices.
- Fixtures should remain available for tests, but they should not be the primary user-facing workflow.

### 2.2 Have vector potential coefficients been compared to SPEC HDF5 output?

Current `beltrami_jax` answer:

- Yes for the SPECTRE IO/validation contract and for validated per-volume JAX-assembled solves when supplied the post-constraint SPECTRE branch state.
- `beltrami_jax` now loads SPECTRE `reference.h5` vector-potential datasets and compares them to fresh SPECTRE exports from `spectre.get_vec_pot_flat`.
- The JAX-native SPECTRE path can now assemble and unpack per-volume coefficients; the remaining blocker is computing the final post-constraint branch state and field diagnostics without SPECTRE injection.
- Existing dense validation compares to linear systems dumped from an instrumented local SPEC build and now also to released SPECTRE per-volume linear-system exports: operator, RHS, and solved degree-of-freedom vector.

SPECTRE assessment:

- The released SPECTRE repo already has this validation pattern in `tests/compare/test_compare_to_spec.py`.
- It compares:
  - `output/force_final`
  - `output/force_final_grad`
  - `vector_potential/Ate`
  - `vector_potential/Aze`
  - `vector_potential/Ato`
  - `vector_potential/Azo`
- The Python helper `spectre.get_vec_pot_flat(test)` obtains the current solution coefficients from the Fortran wrapper.

Manual SPECTRE validation run on 2026-04-30:

- Local SPECTRE commit: `08e358a`.
- After local build/runtime patches listed below, all four shipped comparison cases were manually checked against `reference.h5`.
- Relative vector-potential coefficient errors:
  - `G2V32L1Fi` cylinder: `Ate 1.27e-15`, `Aze 4.70e-15`, `Ato 0`, `Azo 0`.
  - `G3V3L3Fi` rotating ellipse: `Ate 5.68e-15`, `Aze 4.83e-14`, `Ato 0`, `Azo 0`.
  - `G3V3L2Fi_stability`: `Ate 5.69e-15`, `Aze 5.07e-14`, `Ato 0`, `Azo 0`.
  - `G3V8L3Free` free-boundary tokamak: `Ate 2.78e-15`, `Aze 3.20e-15`, `Ato 0`, `Azo 0`.
- Force-mode relative errors in the same manual checks were `1.1e-13` to `3.2e-12`.

Completed `beltrami_jax` work:

- Added `src/beltrami_jax/spectre_io.py` with HDF5/NPZ loaders and vector-potential comparison diagnostics.
- Added `src/beltrami_jax/spectre_input.py` with SPECTRE TOML summaries for geometry, resolution, flux, constraints, free-boundary settings, and boundary Fourier tables.
- Added tests for the SPECTRE TOML and HDF5 IO layers.
- Added `examples/validate_spectre_vector_potential.py`.
- Added `tools/export_spectre_vecpot_npz.py` and `tools/generate_spectre_validation_assets.py`.
- Added packaged public SPECTRE compare cases under `src/beltrami_jax/data/spectre_compare/` so coefficient parity is reproducible without a local SPECTRE checkout.
- Added `src/beltrami_jax/spectre_layout.py` to turn SPECTRE `Lrad` metadata into packed volume/exterior slices.
- Added `src/beltrami_jax/spectre_pack.py` to mirror SPECTRE `gi00ab`, `lregion`, `preset_mod.F90`, and `packab` degree-of-freedom maps.
- Generated `docs/_static/spectre_vecpot_parity.png`, showing worst global relative coefficient error `1.52e-14` across four public SPECTRE compare cases.
- Added `tools/export_spectre_linear_system_npz.py`, `src/beltrami_jax/spectre_linear.py`, packaged fixtures under `src/beltrami_jax/data/spectre_linear/`, and `docs/_static/spectre_linear_parity.png`.
- Current released-SPECTRE linear parity covers 19 volume solves with exact matrix/RHS reconstruction and worst solution relative error `1.59e-15`.
- Added `src/beltrami_jax/spectre_backend.py` with JIT-backed `solve_spectre_assembled`, NumPy-returning `solve_spectre_assembled_numpy`, equal-size batched solves, and a lightweight timing helper.

Required remaining `beltrami_jax` work:

- Add tests that compare JAX-native vector-potential coefficients to SPECTRE/SPEC `.h5` datasets.
- Add exact JAX-native geometry/integral assembly that produces the per-volume solution vectors consumed by the new pack/unpack maps.

### 2.3 Does "large aspect-ratio" mean only large-aspect-ratio tokamaks?

Current answer:

- The phrase currently applies only to the internal prototype geometry in `FourierBeltramiGeometry`.
- It does not describe the intended final scope of `beltrami_jax`.
- The current code should not be advertised as accepting arbitrary 3D SPECTRE geometry yet.

Final target:

- Support the geometries SPECTRE supports through its own interface Fourier representation:
  - cylindrical, toroidal, and full 3D geometries;
  - stellarator-symmetric and non-stellarator-symmetric cases;
  - fixed-boundary and free-boundary workflows;
  - coordinate-singularity and non-singular volume branches.
- Keep the current large-aspect-ratio geometry only as a standalone educational/prototype example unless it is replaced by exact SPECTRE assembly.

## 3. Current `beltrami_jax` Assessment

### 3.1 Repository

- Local path: `/Users/rogerio/local/beltrami_jax`.
- Remote target: `https://github.com/rogeriojorge/beltrami_jax`.
- Current branch: `main`.
- Current commit at assessment start: `fc120af`.
- Working tree at assessment start: clean.

### 3.2 Package State

Important source files:

- `src/beltrami_jax/types.py`
  - Defines `BeltramiLinearSystem`, `FourierBeltramiGeometry`, `FourierModeBasis`, `BeltramiProblem`, `SpecLinearSystemReference`, result dataclasses, diagnostics, and benchmark records.
- `src/beltrami_jax/operators.py`
  - Implements `assemble_operator`, `assemble_rhs`, magnetic energy, helicity, residuals.
  - This is the most SPEC-like part of the current code.
- `src/beltrami_jax/solver.py`
  - Dense JAX solve and batched parameter scan.
  - Uses `jax_enable_x64`.
  - `solve_operator` and `solve_parameter_scan` are JIT compiled.
- `src/beltrami_jax/iterative.py`
  - Compact GMRES implementation accepting a dense matrix or callable matvec.
- `src/beltrami_jax/geometry.py`
  - Internal shaped-torus prototype assembly.
  - Builds collocation grid, simple torus coordinates, basis functions, and approximate `d_ma`, `d_md`, `d_mb`, optional `d_mg`.
  - This is not exact SPECTRE/SPEC `matrices_mod.F90` parity.
- `src/beltrami_jax/nonlinear.py`
  - Secant-style helicity target solve for `mu`.
  - This is not full SPECTRE `Lconstraint` branch parity.
- `src/beltrami_jax/reference.py`
  - Loads SPEC text dumps and packaged `.npz` fixtures.
  - Does not load `.h5` vector-potential datasets yet.
- `src/beltrami_jax/io.py`
  - JSON problem save/load and `.npz` nonlinear-solution save/load.
- `tools/build_spec_fixture.py`
  - Converts SPEC text dumps into packaged fixtures.
- `tools/generate_validation_assets.py`
  - Generates validation and benchmark figures from current fixtures.

### 3.3 Tests

Current test files:

- `tests/test_solver.py`
  - SPEC fixture solution parity.
  - residual helper parity.
  - autodiff of solved energy with respect to `mu`.
  - parameter scan parity.
  - vacuum RHS.
  - GMRES vs dense.
  - verbose output.
  - diagnostics and benchmarks.
- `tests/test_reference.py`
  - fixture listing, shapes, loaded metadata.
  - operator/RHS reassembly from fixtures.
  - text dump loading with dotted prefixes and vacuum metadata.
- `tests/test_workflow.py`
  - internal geometry assembly shape/symmetry.
  - collocation and coordinate sanity.
  - JSON roundtrip.
  - matrix-free GMRES.
  - simple outer helicity loop.
  - vacuum geometry `d_mg`.
- `tests/test_examples.py`
  - smoke tests for all example scripts.

Current gap:

- Tests now cover SPECTRE TOML summaries and SPECTRE HDF5 vector-potential IO/comparison.
- Tests now validate exact SPECTRE packing/unpacking of `Ate/Aze/Ato/Azo`, including coordinate-singularity axis recombination, free-boundary exterior blocks, non-stellarator-symmetric synthetic maps, and JAX autodiff through pack/unpack.
- No test yet compares JAX-native SPECTRE-geometry output against SPECTRE/SPEC HDF5 vector-potential coefficients.

### 3.4 Examples

Current examples:

- `examples/solve_spec_fixture.py`
  - Loads a packaged SPEC fixture, solves dense and GMRES, generates comparison figure.
- `examples/parameter_scan.py`
  - Uses internal geometry prototype, solves an outer helicity target, runs vectorized parameter scan, writes JSON/NPZ/figure.
- `examples/autodiff_mu.py`
  - Demonstrates autodiff through solved energy with respect to `mu`.
- `examples/benchmark_fixtures.py`
  - Demonstrates vacuum/GMRES workflow and benchmarking.
- `examples/validate_spectre_vector_potential.py`
  - Demonstrates SPECTRE TOML summary loading, SPECTRE HDF5 coefficient loading, coefficient comparison, JSON output, and figure generation.

Current gap:

- Examples now teach the current SPECTRE-facing IO/validation workflow.
- Add examples once JAX-native SPECTRE assembly exists:
  - `examples/spectre_toml_field_solve.py`
  - `examples/spectre_backend_dropin.py`

### 3.5 Documentation

Current docs are extensive, but several statements need tightening.

Problems to fix:

- The README says the repository covers "full supported Beltrami path", which is true only for the internal prototype model, not for SPECTRE parity.
- The SPECTRE integration section suggests `BeltramiProblem` and `assemble_fourier_beltrami_system` are likely integration entry points. That is now misleading because SPECTRE's real input is full interface geometry and constraint state, not `major_radius/minor_radius/elongation`.
- "Large-aspect-ratio" appears often and needs to be scoped explicitly to the prototype geometry.
- Fixtures are described prominently, which can make users think SPECTRE/SPEC fixtures are required input files.

Immediate doc fixes:

- Add a top-level "Input modes and parity status" section to README.
- Add a SPECTRE-specific limitations section.
- Move fixture generation to a developer/reference section.
- Add a collaborator-facing FAQ answering the three email questions.
- Add a SPECTRE integration page based on this file.

## 4. SPECTRE Assessment

### 4.1 Clone and Install

SPECTRE release source:

- Remote: `https://gitlab.com/spectre-eq/spectre.git`.
- Local checkout: `/Users/rogerio/local/spectre`.
- Commit assessed: `08e358a`.
- Branch: `main`.

Install commands used:

```bash
cd /Users/rogerio/local/spectre
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
HDF5_ROOT=/opt/homebrew/opt/hdf5-mpi \
FFTW_ROOT=/opt/homebrew/opt/fftw \
  .venv/bin/python -m pip install -e .
```

Build dependencies available locally:

- Python 3.13.7.
- CMake 4.3.1.
- GNU Fortran 15.2.0.
- `mpifort`.
- Homebrew `hdf5-mpi`.
- Homebrew `fftw`.
- Homebrew `open-mpi`.
- Homebrew `libomp`.

Runtime environment needed for local SPECTRE import:

```bash
export DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib
export OMP_NUM_THREADS=1
```

### 4.2 Local SPECTRE Bring-Up Patches

Two local patches were needed to run SPECTRE on this machine.

Patch 1: Pydantic serializer guard

- File: `/Users/rogerio/local/spectre/spectre/file_io/input_parameters.py`.
- Problem: `arrays_of_length_nvol_minus_1 = []`, but the class applies `@field_serializer(*arrays_of_length_nvol_minus_1)`.
- Newer Pydantic rejects a field serializer with no fields.
- Local fix: only define that serializer if the list is nonempty.

Patch 2: Fortran legacy `FORMAT`

- File: `/Users/rogerio/local/spectre/fortran_src/beltrami_field_mod.F90`.
- Problem: gfortran 15 rejects a legacy format string missing descriptor commas.
- Local fix: rewrite the `1040 format(...)` line with explicit comma-separated descriptors.

These are local SPECTRE bring-up patches, not `beltrami_jax` changes. They should be proposed upstream separately or handled by compiler flags if SPECTRE maintainers prefer.

### 4.3 SPECTRE Code Path for Beltrami

Python entry points:

- `scripts/run/calc_spectre_field.py`
  - `test = SPECTRE.from_input_file(inputfile)`
  - `xin = get_xinit_specwrap(test)`
  - `fr = force_real(xin, test)`
  - `test.run(save_output=True)`
  - `print_all_errors(test, beltrami=True, force=True)`
  - `save_errors_to_h5(...)`
- `spectre/force_targets.py`
  - `force_real(xin, test, ...)`
  - calls `test.field_mod.solve_field(xin, pert_axis)`.
- `spectre/utils.py`
  - `get_xinit_specwrap(test)` packs interfaces into the SPECTRE wrapper vector.
  - `get_vec_pot_flat(test)` calls `test.wrapper_funcs_mod.get_vec_pot(...)`.
  - `get_vec_pot(test)` splits flat vector-potential arrays per volume.
- `spectre/error_checking.py`
  - `get_beltrami_errors(test, ...)` calls `wrapper_funcs_mod.calc_beltrami_error`.

Fortran Beltrami path:

- `fortran_src/field_mod.F90`
  - Field solve, force calculations, derivative calculations.
  - Calls Beltrami construction and later uses `Ate/Aze/Ato/Azo` to build fields and forces.
- `fortran_src/beltrami_field_mod.F90`
  - `construct_beltrami_field(lvol, NN)`.
  - Determines local solve degrees of freedom from `Lconstraint`, plasma/vacuum branch, and coordinate-singularity branch.
  - Calls `solve_beltrami_system` directly for no-iteration cases.
  - Calls `hybrj2(solve_beltrami_system, ...)` for nonlinear local constraints.
- `fortran_src/beltrami_solver_mod.F90`
  - `solve_beltrami_system(Ndof, Xdof, Fdof, Ddof, Ldfjac, iflag)`.
  - Builds `matrix = dMA - lmu*dMD` for plasma.
  - Uses `matrix = dMA` for vacuum.
  - Builds RHS from `dMB`, `dMG`, and flux vector `dpsi`.
  - Solves with LAPACK `DGETRF`, `DGETRS`, and `DGERFS`.
  - Computes `lBBintegral` and `lABintegral`.
  - Calls `packab('U', ...)` to unpack the packed solution into vector-potential coefficients.
  - Evaluates constraint residuals and derivatives for `Lconstraint`.
- `fortran_src/matrices_mod.F90`
  - `matrix(lvol, mn, lrad)` assembles `dMA` and `dMD` using precomputed geometry integrals.
  - `matrixBG(lvol, mn, lrad)` assembles `dMB` and `dMG`.
  - Handles stellarator-symmetric and non-stellarator-symmetric branches.
  - Handles coordinate-singularity branch using Zernike-specific indexing.
- `fortran_src/vector_potential_writer_mod.F90`
  - Writes `Ate`, `Aze`, `Ato`, and `Azo` to HDF5 datasets under `vector_potential`.
- `fortran_src/wrapper_funcs_mod.F90`
  - `get_vec_pot(sumLrad, allAte, allAze, allAto, allAzo)` exposes flat vector-potential coefficients.
  - `calc_beltrami_error(...)` evaluates `curl B - mu B` error.

### 4.4 SPECTRE Validation Results From This Assessment

Manual field calculation:

```bash
OMP_NUM_THREADS=1 \
DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib \
  .venv/bin/python <manual script>
```

Case: `tests/wrapper/cyl_manyvol_test.toml`.

Observed:

- `xinit_shape = (17,)`.
- `force_shape = (2, 512)`.
- `force_l2 = 0.9487252443877742`.
- `vecpot_shapes = (17, 8)` for each of `Ate`, `Aze`, `Ato`, `Azo`.
- `vecpot_norms = 0.10518895571423069`, `0.011319261726391045`, `0.0`, `0.0`.

Manual comparison against SPECTRE reference HDF5:

- `tests/compare/G2V32L1Fi/reference.h5`
- `tests/compare/G3V3L3Fi/reference.h5`
- `tests/compare/G3V3L2Fi_stability/reference.h5`
- `tests/compare/G3V8L3Free/reference.h5`

Results:

| Case | force rel. error | Ate rel. error | Aze rel. error | Ato rel. error | Azo rel. error |
| --- | ---: | ---: | ---: | ---: | ---: |
| `G2V32L1Fi` | `1.23936974e-13` | `1.27044705e-15` | `4.69571461e-15` | `0` | `0` |
| `G3V3L3Fi` | `1.76842817e-13` | `5.67896913e-15` | `4.82616704e-14` | `0` | `0` |
| `G3V3L2Fi_stability` | `1.08021681e-13` | `5.68669115e-15` | `5.06561871e-14` | `0` | `0` |
| `G3V8L3Free` | `3.20058556e-12` | `2.77882766e-15` | `3.19764357e-15` | `0` | `0` |

This is the validation bar for `beltrami_jax`: reproduce SPECTRE/SPEC coefficient arrays at this level before claiming backend replacement.

### 4.5 Existing SPECTRE Reference Data

SPECTRE already ships reference data suitable for `beltrami_jax` validation:

- `tests/compare/G2V32L1Fi/input.toml`
- `tests/compare/G2V32L1Fi/reference.h5`
- `tests/compare/G3V3L3Fi/input.toml`
- `tests/compare/G3V3L3Fi/reference.h5`
- `tests/compare/G3V3L2Fi_stability/input.toml`
- `tests/compare/G3V3L2Fi_stability/reference.h5`
- `tests/compare/G3V8L3Free/input.toml`
- `tests/compare/G3V8L3Free/reference.h5`

Datasets needed:

- `output/force_final`
- `output/force_final_grad`
- `vector_potential/Ate`
- `vector_potential/Aze`
- `vector_potential/Ato`
- `vector_potential/Azo`
- optionally `stability/ohessian` for `Lconstraint=2` stability cases.

## 5. What It Takes To Remove SPEC/SPECTRE Fortran Beltrami Completely

Removing the current Fortran Beltrami backend means replacing all of the following capabilities, not just `jax.numpy.linalg.solve`.

### 5.1 Data and Input Layer

Implement a SPECTRE-native input layer:

- Parse SPECTRE TOML or consume a live SPECTRE object/state.
- Represent:
  - `Igeometry`, `Nfp`, `Mvol`, `Mpol`, `Ntor`, `Lrad`, `Nvol`, free-boundary state.
  - `im`, `in` mode arrays.
  - interface and boundary Fourier coefficients.
  - `tflux`, `pflux`, `mu`, `helicity`, `iota`, `oita`, `Ivolume`, `Isurf`, pressure.
  - `Lconstraint`, `Lcoordinatesingularity`, `Lplasmaregion`, `Lvacuumregion`, `enforce_stell_sym`.
  - quadrature and grid settings.

### 5.2 Packing and Vector-Potential Coefficients

Status: implemented in `src/beltrami_jax/spectre_pack.py` for the coefficient and multiplier id maps used by SPECTRE `packab`.

- Reproduce `packab('P')` and `packab('U')` behavior.
- Map packed vector `a` to per-volume coefficient arrays:
  - `Ate(lvol, ideriv, ii)%s(ll)`.
  - `Aze(lvol, ideriv, ii)%s(ll)`.
  - `Ato(lvol, ideriv, ii)%s(ll)`.
  - `Azo(lvol, ideriv, ii)%s(ll)`.
- Preserve radial indexing for Chebyshev and Zernike branches.
- Preserve stellarator-symmetric and non-stellarator-symmetric coefficient families.
- Expose `get_vec_pot_flat()` equivalent returning arrays matching SPECTRE's shape `(sum(Lrad + 1), mn)`.

Remaining connection:

- Feed these maps from JAX-native SPECTRE matrix assembly rather than from SPECTRE-produced coefficients.

### 5.3 Geometry and Integral Assembly

Port SPECTRE's exact assembly:

- Geometry coordinate construction from interface Fourier coefficients.
- Metric terms, Jacobian, quadrature grids, and derivative tensors used by SPECTRE.
- Precomputed integral arrays used by `matrices_mod.F90`, including the `TT*`, `TD*`, `DD*`, and helicity-integral terms.
- Coordinate-singularity Zernike branch.
- Non-singular Chebyshev branch.
- Stellarator-symmetric and non-stellarator-symmetric assembly.
- `dMA`, `dMD`, `dMB`, and `dMG` assembly.
- Vacuum and free-boundary forcing.

### 5.4 Linear and Nonlinear Solve Layer

Implement SPECTRE branch logic:

- Plasma:
  - `matrix = dMA - mu*dMD`.
  - `rhs = -dMB @ dpsi`.
- Vacuum:
  - `matrix = dMA`.
  - `rhs = -dMG - dMB @ dpsi`.
- Derivative solves:
  - `d solution / d mu`.
  - `d solution / d psi`.
  - derivatives needed by force Jacobians and SPECTRE constraint updates.
- Local nonlinear constraints:
  - `Lconstraint = -2, -1, 0, 1, 2, 3`.
  - rotational-transform constraints.
  - helicity constraints.
  - plasma-current/linking-current constraints.
  - global-current semi-global logic.
- Error and convergence behavior equivalent to SPECTRE.

### 5.5 Field, Force, and Diagnostics Coupling

For SPECTRE integration, Beltrami coefficients are not enough. The downstream force path needs:

- Reconstructed vector potential.
- Magnetic field from `B = curl A`.
- Current and Beltrami error from `curl B - mu B`.
- Interface force `p + B^2/(2 mu0)` jumps.
- Force Jacobian terms with respect to geometry.
- HDF5 output and comparison-compatible datasets.

The immediate `beltrami_jax` goal should be to replace the Beltrami coefficient backend first, then widen to derivative and force-coupled functionality.

## 6. Implementation Roadmap

### Phase 0: Correct User-Facing Documentation

Goal: stop overclaiming.

Tasks:

- Add a README section titled "Input modes and current parity status".
- State that fixtures are validation assets, not required user input.
- State that internal `FourierBeltramiGeometry` is a prototype shaped-torus model.
- State that arbitrary 3D SPECTRE geometry is planned, not currently implemented.
- Add a collaborator-facing FAQ in docs.
- Update `docs/integration.md` so SPECTRE integration points are not described as final.
- Update validation docs to distinguish implemented HDF5/pack-unpack parity from missing JAX-native SPECTRE assembly parity.

Acceptance:

- A reader can answer the three email questions without asking.
- README no longer implies SPECTRE drop-in parity.

### Phase 1: Add SPECTRE HDF5 Reference Support

Goal: compare against the validation target the collaborator asked about.

New code:

- `src/beltrami_jax/spectre_io.py`
  - `load_spectre_vector_potential_h5(path)`.
  - `load_spectre_reference_case(input_toml, reference_h5)`.
  - dataclass `SpectreVectorPotential`.
- `src/beltrami_jax/spectre_types.py`
  - SPECTRE-specific containers if separation from generic types is cleaner.

Tests:

- Package small SPECTRE reference extracts or use existing external path conditionally.
- Add unit tests for shape, transpose convention, norms, and field names.
- Add tests that read `Ate/Aze/Ato/Azo` from HDF5 and compare to a local packed vector once pack/unpack exists.

Docs/examples:

- `examples/spectre_h5_vecpot_validation.py`.
- `docs/spectre_integration.md`.

Acceptance:

- `beltrami_jax` can load SPECTRE reference HDF5 vector-potential datasets and print a summary.

### Phase 2: Implement Exact SPECTRE Pack/Unpack

Status: complete for coefficient maps and differentiable pack/unpack; still needs upstream solution vectors produced by JAX-native SPECTRE assembly.

Goal: map `beltrami_jax` packed solutions to SPECTRE's `Ate/Aze/Ato/Azo`.

New code:

- `src/beltrami_jax/spectre_pack.py`.
- `spectre_fourier_modes(...)`.
- `build_spectre_dof_layout(...)`.
- `pack_vector_potential(...)`.
- `unpack_solutions(...)`.
- `pack_vector_potential_jax(...)`.
- `unpack_solutions_jax(...)`.

Reference source:

- `fortran_src/packing_mod.F90`.
- `fortran_src/wrapper_funcs_mod.F90:get_vec_pot`.
- `fortran_src/vector_potential_writer_mod.F90`.

Tests:

- Packaged SPECTRE reference cases round-trip through the id maps exactly.
- Positive ids are unique and contiguous.
- Coordinate-singularity axis recombination and non-stellarator-symmetric odd components are covered.
- JAX autodiff through pack/unpack is covered.
- Future test: compare JAX-native solved vectors to SPECTRE's `get_vec_pot_flat()` once the JAX-native matrix assembly exists.

Acceptance:

- For SPECTRE-produced `Ate/Aze/Ato/Azo` arrays, `beltrami_jax` reconstructs the same per-volume solution-vector coefficient entries and unpacks them back exactly.

### Phase 3: SPECTRE Matrix/RHS Extraction

Status: complete for packaged released-case validation fixtures. A future
SPECTRE PR can still add a cleaner upstream helper, but `beltrami_jax` no
longer needs local SPECTRE source patches to validate matrix/RHS/solution
parity for the shipped cases.

Goal: expand validation from old SPEC text dumps to SPECTRE's current released code.

Implemented path:

- `tools/export_spectre_linear_system_npz.py` runs from a SPECTRE environment.
- The exporter finalizes SPECTRE state, allocates one volume in a fresh Python process to avoid f90wrap reallocation-cache hazards, calls SPECTRE's geometry and matrix assembly, calls `solve_beltrami_system`, and writes one `.npz` per volume.
- `src/beltrami_jax/spectre_linear.py` lists and loads packaged SPECTRE linear fixtures.
- `tests/test_spectre_linear.py` verifies exact operator/RHS reconstruction and JAX solution parity for all packaged SPECTRE volume systems.
- `tools/generate_spectre_linear_validation_assets.py` generates the reviewer-facing linear parity panel.

SPECTRE helper target:

- Add Python wrapper functions in a SPECTRE fork:
  - `get_beltrami_linear_system(lvol)`.
  - `get_beltrami_solution(lvol, ideriv=0)`.
  - `get_beltrami_matrix_rhs(lvol)`.

Acceptance:

- `beltrami_jax` solves SPECTRE-exported systems and matches matrix/RHS exactly and solved vectors to `1.59e-15` worst relative error across the current 19 packaged volume solves.
- The remaining coefficient-level acceptance belongs to Phase 4/5: JAX-native SPECTRE geometry assembly must produce solved vectors that unpack to SPECTRE `Ate/Aze/Ato/Azo`.

### Phase 3.5: Minimal SPECTRE Adapter Boundary

Status: implemented on the `beltrami_jax` side.

Goal: make the first SPECTRE fork change small, reversible, and easy to
benchmark.

Implemented `beltrami_jax` API:

- `solve_spectre_assembled`: JAX-array result for one assembled SPECTRE system.
- `solve_spectre_assembled_numpy`: NumPy-returning wrapper for a SPECTRE Python adapter.
- `solve_spectre_assembled_batch`: vectorized equal-size volume solve path.
- `benchmark_spectre_backend`: timing helper that measures compile+solve and steady-state calls without hard-coding a CI runtime budget.

Recommended SPECTRE-side change:

- Add one experimental option, for example `beltrami_backend = "fortran" | "jax"`, defaulting to `"fortran"`.
- Keep all SPECTRE geometry, quadrature, matrix assembly, and branch setup unchanged.
- In the experimental branch only, pass already assembled `dMA`, `dMD`, `dMB`, `dMG`, `mu`, `psi`, and branch flags to `solve_spectre_assembled_numpy`.
- Copy the returned solution vector into the same SPECTRE solution storage that `solve_beltrami_system` fills.
- Keep existing Fortran solve and tests as fallback until JAX-native assembly and coefficient parity are complete.

Why this boundary:

- It validates runtime behavior and packaging with minimal SPECTRE risk.
- It isolates Python/JAX import and array-copy overhead from the much larger geometry-assembly port.
- It gives a clean benchmark point: Fortran solve vs JAX solve for the exact same SPECTRE-assembled matrix.

### Phase 4: Port SPECTRE Assembly to JAX

Goal: remove dependency on SPECTRE Fortran matrix assembly.

Approach:

- Port the minimum exact branch set first:
  - stellarator-symmetric;
  - fixed-boundary;
  - plasma;
  - non-coordinate-singular volume.
- Then add:
  - coordinate-singularity Zernike branch;
  - vacuum branch;
  - free-boundary branch;
  - non-stellarator-symmetric branch;
  - derivative assembly.

New code candidates:

- `src/beltrami_jax/spectre_geometry.py`.
- `src/beltrami_jax/spectre_matrix.py`.
- `src/beltrami_jax/spectre_integrals.py`.
- `src/beltrami_jax/spectre_assembly.py`.
- `src/beltrami_jax/spectre_constraints.py`.

Acceptance:

- JAX-assembled `dMA`, `dMD`, `dMB`, `dMG` match SPECTRE for the same input cases.
- Status: `dMB/dMG` from SPECTRE `matrixBG` and `dMA/dMD` from SPECTRE `matrix`/`volume_integrate_chebyshev` are implemented for the packaged validated branches. Remaining Phase 4 work is broader fixture diversity and production sparse/matrix-free strategy, not the packaged dense matrix formulas.
- Start with `rtol=1e-12` for matrices after accounting for quadrature and ordering; tighten as branch parity stabilizes.

### Phase 5: Port Full Constraint Loop

Goal: reproduce `construct_beltrami_field` and `solve_beltrami_system` behavior.

Tasks:

- Implement SPECTRE's `Lconstraint` branch table.
- Use JAX-compatible root finding for local nonlinear constraints.
- Match `lBBintegral`, `lABintegral`, transforms, currents, and derivative outputs.
- Preserve informative verbose progress output.

Acceptance:

- `Ate/Aze/Ato/Azo` match SPECTRE reference HDF5 for the shipped comparison suite.
- Beltrami error `curl B - mu B` matches SPECTRE to numerical precision.

### Phase 6: SPECTRE Fork Integration

Goal: demonstrate a path to remove the Fortran Beltrami backend from SPECTRE.

Recommended first integration:

- Add an optional backend flag to SPECTRE:
  - `beltrami_backend = "fortran"` default.
  - `beltrami_backend = "jax"` experimental.
- Keep SPECTRE's existing input, force, and HDF5 workflows unchanged.
- Replace only the Beltrami solve coefficient path first.

Implementation pattern:

- In SPECTRE Python:
  - construct a `beltrami_jax` SPECTRE input/state object;
  - call JAX backend;
  - write returned coefficients into the same arrays SPECTRE currently uses: `Ate/Aze/Ato/Azo`.
- Add tests comparing the JAX backend to the Fortran backend for the existing SPECTRE cases.

Acceptance:

- A SPECTRE user can run the same TOML input with either backend.
- HDF5 vector-potential coefficients agree at machine precision.
- Force outputs and force Jacobians agree within established SPECTRE tolerances.

### Phase 7: Benchmarks and Reviewer-Ready Figures

Goal: make the result convincing for a PR and publication-style review.

Figures:

- Coefficient parity panel:
  - scatter `SPEC/SPECTRE H5 coefficient` vs `beltrami_jax coefficient`.
  - panels for `Ate`, `Aze`, `Ato`, `Azo`.
- Error spectrum panel:
  - coefficient-index error on log scale.
- Branch coverage panel:
  - cylindrical, 3D, `Lconstraint=2`, free-boundary.
- Performance panel:
  - Fortran backend wall time vs JAX backend compile+solve and steady-state.
- Differentiability panel:
  - autodiff derivative vs SPECTRE finite-difference/analytic force Jacobian.

Validation thresholds:

- Coefficient relative error target: `<= 1e-12` initially, machine precision where ordering and arithmetic are identical.
- Force-mode relative error target: match existing SPECTRE tolerances, currently around `1e-10` to `1e-6` depending case.
- Beltrami residual target: same or better than SPECTRE's current `calc_beltrami_error`.
- CI coverage: keep `beltrami_jax` coverage above 90%.

## 7. Design Decisions Needed

Please decide these before implementation of the SPECTRE fork work starts.

### Decision 1: Strict parity backend or standalone solver first?

Recommendation: strict SPECTRE parity backend first.

Consequences:

- Higher initial complexity.
- Much easier to convince SPECTRE maintainers.
- Avoids spending time on a solver that is mathematically plausible but incompatible with SPECTRE.

Alternative:

- Continue developing the standalone large-aspect-ratio geometry solver.
- Faster standalone progress, but it will not answer the collaborator's concerns.

### Decision 2: Initial integration point inside SPECTRE?

Recommendation: first replace only the linear solve and coefficient unpack path, while using SPECTRE/Fortran assembly as the oracle and fallback.

Consequences:

- Lower risk.
- Provides immediate vector-potential parity validation.
- Does not remove all Fortran initially, but creates a controlled stepping stone.

Alternative:

- Port full assembly first.
- Cleaner end state, but much larger risk and slower path to demonstrable parity.

### Decision 3: Input API for `beltrami_jax`?

Recommendation: three layers.

- `from_components(...)`: low-level matrices and vectors.
- `from_spectre_reference(...)`: TOML/HDF5/reference case for validation.
- `from_spectre_state(...)`: live SPECTRE state or exported SPECTRE state for integration.

Consequences:

- Clear user-facing API.
- Fixtures remain developer-only.
- SPECTRE integration does not depend on the prototype `FourierBeltramiGeometry`.

### Decision 4: Dependency policy?

Recommendation:

- Keep core runtime dependency as `jax`.
- Put `h5py`, `matplotlib`, and SPECTRE validation helpers in optional extras: `validation`, `examples`, or `dev`.

Consequences:

- `pip install beltrami_jax` remains small.
- HDF5 validation requires `pip install beltrami_jax[validation]`.

Alternative:

- Add `h5py` to core dependencies.
- Simpler validation examples, but heavier core install.

### Decision 5: SPECTRE fork location?

Recommendation:

- Fork SPECTRE where maintainers expect contributions. Since upstream is GitLab, use a GitLab fork if possible.
- Mirror or document the branch from GitHub only if needed for local workflow.

Consequences:

- Cleaner merge request path.
- May require GitLab authentication setup separate from GitHub credentials.

### Decision 6: Validation source of truth?

Recommendation:

- Use SPECTRE's shipped `tests/compare/*/reference.h5` as the first public target.
- Add new higher-shaped QA/QH cases after the SPECTRE parity path is proven.

Consequences:

- Immediate reproducibility.
- Avoids private/unreleased configuration dependence.

### Decision 7: Definition of "any 3D geometry"?

Recommendation:

- Claim support for "any 3D geometry representable by SPECTRE's Fourier interface model and implemented branch set".
- Do not claim support for arbitrary point clouds, meshes, or non-SPECTRE geometry representations.

Consequences:

- Scientifically precise.
- Aligns with SPECTRE's actual input model.

### Decision 8: Nonlinear constraints scope for first PR?

Recommendation:

- First PR: coefficient parity for existing SPECTRE reference cases.
- Second PR: JAX-native matrix assembly.
- Third PR: full nonlinear constraint parity and derivative path.

Consequences:

- Smaller review units.
- Easier to bisect correctness issues.

## 8. Immediate Next Tasks

1. Add SPECTRE wrapper functions or debug exports for Beltrami matrices, RHS, packed solution, and vector-potential arrays from each volume.
2. Start JAX port of SPECTRE geometry assembly branch by branch, beginning with one fixed-boundary stellarator-symmetric case.
3. Connect the JAX assembly output to the implemented `spectre_pack` maps and compare produced `Ate/Aze/Ato/Azo` against packaged SPECTRE HDF5 coefficients.
4. Add a SPECTRE fork/branch with a backend switch once the JAX path passes one coefficient-parity case.
5. Expand to free-boundary and coordinate-singularity cases.
6. Add force-mode and derivative parity after coefficient parity is stable.

## 9. Open Risks

- Exact SPECTRE packing was subtle and is now represented directly, but it remains a high-risk interface to preserve during later assembly work.
- The current internal geometry assembly may distract from the real SPECTRE target if docs are not clarified.
- SPECTRE's force and derivative path uses Beltrami derivatives, not only the base coefficient vector.
- Local SPECTRE build required patches on macOS with Python 3.13, Pydantic 2.13, and gfortran 15; this should be separated from Beltrami backend work.
- `beltrami_jax` currently avoids hard runtime dependencies beyond `jax`; SPECTRE validation uses optional `h5py`, `tomli`, `matplotlib`, and `numpy` extras.
- Machine precision validation requires matching not just equations, but coefficient ordering, symmetry conventions, radial basis conventions, and branch-specific zeroing rules.

## 10. Current Local State

`beltrami_jax`:

- Path: `/Users/rogerio/local/beltrami_jax`.
- Branch: `main`.
- Commit before this document: `fc120af`.
- Runtime dependency policy already corrected to unpinned `jax` in `pyproject.toml`.

SPECTRE:

- Path: `/Users/rogerio/local/spectre`.
- Branch: `main`.
- Commit: `08e358a`.
- Local uncommitted patches:
  - `spectre/file_io/input_parameters.py`.
  - `fortran_src/beltrami_field_mod.F90`.
- Build directory: `/Users/rogerio/local/spectre/build`.
- Virtual environment: `/Users/rogerio/local/spectre/.venv`.

Useful SPECTRE run command pattern:

```bash
cd /Users/rogerio/local/spectre
export OMP_NUM_THREADS=1
export DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib
.venv/bin/python -m pytest tests/compare/test_compare_to_spec.py -q
```

Note: on this local setup the full pytest comparison command terminated without a Python traceback after collection. The equivalent manual comparison logic ran successfully for all four reference cases and is the reliable result from this assessment.

## 11. Draft Response to the Collaborator

The honest technical response should be:

```text
Thanks, those are the right questions.

At the moment beltrami_jax has three paths. The fixture path is a developer validation path where SPEC/SPECTRE-style matrices are already assembled and loaded, and the JAX code solves the packed Beltrami system. The geometry-driven path currently in the repository is a prototype shaped-torus assembly, not yet the full SPECTRE geometry path. The SPECTRE-facing validation path now reads SPECTRE TOML metadata and HDF5 vector-potential coefficients. The target is to make the actual solve input SPECTRE-like: interface geometry, resolution, flux/current/helicity constraints, and branch flags, not prebuilt fixtures.

You are also right about vector-potential coefficients. I checked the released SPECTRE tests and they already provide exactly that reference path. beltrami_jax now loads those HDF5 vector_potential/Ate, Aze, Ato, Azo datasets and compares them against fresh SPECTRE exports. Across the four public SPECTRE compare cases the worst global relative coefficient error is 1.52e-14. The remaining work is to make the JAX-native SPECTRE geometry assembly produce those coefficients directly, rather than using SPECTRE as the exporter.

Finally, "large aspect-ratio" only describes the current internal prototype geometry. It should not be read as the final scope. For SPECTRE integration, the goal is any geometry representable by SPECTRE's Fourier interface model and supported branch set. I will clarify the docs so this distinction is explicit.
```

## 12. Definition of Done for the SPECTRE Replacement Claim

Do not claim SPECTRE Beltrami replacement until all of these are true:

- A user can start from a SPECTRE TOML file or a SPECTRE state, not from fixtures.
- `beltrami_jax` returns `Ate/Aze/Ato/Azo` with the same shapes and ordering as SPECTRE.
- Existing SPECTRE `tests/compare` reference cases pass coefficient parity.
- Plasma, vacuum, coordinate-singularity, non-coordinate-singularity, fixed-boundary, free-boundary, and at least one 3D case are covered.
- Force modes and Beltrami errors match SPECTRE within existing test tolerances.
- The SPECTRE fork can switch between Fortran and JAX backend with one flag.
- CI runs unit tests, docs, examples, and validation asset generation.
- README and docs clearly state exactly what input is supported and what is still experimental.

## 13. 2026-04-30 Progress: Branch Constraints and Interface Geometry

New completed ingredients:

- Added `src/beltrami_jax/spectre_constraints.py`.
  - Ports SPECTRE local `Nxdof` selection for `Lconstraint = -2, -1, 0, 1, 2, 3`.
  - Solves SPECTRE branch derivative right-hand sides for plasma, vacuum, and coordinate-singularity current branches.
  - Evaluates local `Lconstraint` residual/Jacobian formulas once rotational-transform/current/helicity diagnostics are supplied.
- Added `src/beltrami_jax/spectre_geometry.py`.
  - Builds SPECTRE interface Fourier geometry arrays in internal `gi00ab` mode order.
  - Parses `physics.allrzrz.interface_*` rows and appends free-boundary wall rows from `rwc/zws/rws/zwc`.
  - Implements SPECTRE coordinate-singularity interpolation for `Igeometry == 2` and `Igeometry == 3`.
  - Implements non-axis linear interpolation between interfaces.
  - Evaluates `R`, `Z`, first derivatives, Jacobian, inverse Jacobian, and covariant metric tensor in JAX.
- Added `examples/spectre_geometry_probe.py`.
  - Produces a SPECTRE interface/Jacobian/metric panel for the packaged `G3V8L3Free` free-boundary case.

What this closes:

- The branch/constraint orchestration is no longer just prose. It has a tested JAX representation.
- The SPECTRE-specific geometry input is no longer only TOML metadata. It can now be evaluated in JAX with differentiable radial interpolation.
- The next matrix-assembly work can target a concrete `SpectreCoordinateGrid` contract instead of re-parsing TOML.

What remains:

- Port SPECTRE/SPEC `matrix` and `intghs` volume-integral formulas onto the new `SpectreCoordinateGrid`.
- Add JAX-native field diagnostics for rotational transform and plasma current instead of injecting diagnostic arrays.
- Compare JAX-assembled `dMA` and `dMD` against the packaged released-SPECTRE linear fixtures, while continuing to validate `dMB/dMG` through the `matrixBG` tests.
- Solve those JAX-assembled systems, unpack with the existing `spectre_pack` maps, and compare generated `Ate/Aze/Ato/Azo` against SPECTRE `reference.h5`.

## 14. 2026-04-30 Progress: SPECTRE `matrixBG` Boundary Assembly

New completed ingredient:

- Added `src/beltrami_jax/spectre_matrix.py`.
  - Ports SPECTRE `matrices_mod.F90::matrixBG`.
  - Builds `SpectreBoundaryNormalField` arrays corresponding to SPECTRE `iVns`, `iBns`, `iVnc`, and `iBnc`.
  - Recombines SPECTRE TOML normal-field tables in internal `gi00ab` mode order.
  - Assembles `dMB` on `Lmg/Lmh` rows with the SPECTRE signs `(-1, +1)`.
  - Assembles `dMG` on `Lme/Lmf` rows from symmetric and non-stellarator-symmetric boundary-normal-field components.
  - Provides `assemble_spectre_matrix_bg_from_input(summary, lvol)` for TOML-driven initial-source assembly.
  - Provides `assemble_spectre_matrix_bg(volume_map, normal_field)` for exact parity when SPECTRE/free-boundary code supplies updated normal-field arrays.

Tests added:

- `tests/test_spectre_matrix.py`.
  - Fixed-boundary public fixtures: `dMB/dMG` match packaged released-SPECTRE fixtures exactly from TOML input.
  - Free-boundary public fixture: TOML-only assembly is tested as the initialized source, while exact post-Picard fixture parity is tested by supplying the updated normal-field source arrays.
  - All 19 packaged released-SPECTRE fixtures: `dMB/dMG` scatter parity is exact when supplied the same normal-field source represented by the fixture.
  - Synthetic non-stellarator-symmetric metadata exercises `Lmf` and the `iVnc/iBnc` branch.
  - Invalid volume and bad normal-field shape errors are covered.

Design decision:

- `matrixBG` was separated from the future `dMA/dMD` integral port because it depends only on degree-of-freedom maps and boundary-normal-field arrays, not on quadrature metrics. This gives a small, reviewable SPECTRE-removal ingredient and narrows the remaining assembly problem to the volume-integral matrices.

Free-boundary nuance:

- SPECTRE free-boundary runs can update `iBns/iBnc` during the Picard normal-field iteration. A SPECTRE TOML file only contains the initial tables. Exact parity with a final SPECTRE linear fixture therefore requires passing the updated arrays into `SpectreBoundaryNormalField`.

What remains:

- Port SPECTRE/SPEC `matrix` and `intghs` volume-integral contractions for `dMA/dMD`.
- Add JAX-native field diagnostics for rotational transform and plasma current.
- Combine JAX `dMA/dMD`, JAX `matrixBG` `dMB/dMG`, branch solves, and `spectre_pack` unpacking to generate `Ate/Aze/Ato/Azo` directly from SPECTRE TOML/interface geometry.

## 15. 2026-04-30 Progress: JAX-Native SPECTRE Volume Matrix Assembly and Per-Volume Coefficient Solves

New completed ingredients:

- Added `src/beltrami_jax/spectre_radial.py`.
  - Ports SPECTRE radial basis, axis basis, quadrature sizing, and angular-grid defaults.
- Added `src/beltrami_jax/spectre_integrals.py`.
  - Assembles the metric-integral tensors consumed by SPECTRE `matrices_mod.F90::matrix`.
- Added `src/beltrami_jax/spectre_volume_matrix.py`.
  - Ports SPECTRE `dMA/dMD` matrix contractions.
  - Combines `dMA/dMD` with the existing `matrixBG` `dMB/dMG` path.
- Added `src/beltrami_jax/spectre_solve.py`.
  - Provides SPECTRE-normalized `dtflux/dpflux` helpers.
  - Provides `solve_spectre_volume_from_input`, which assembles, solves, and unpacks one SPECTRE volume into `Ate/Aze/Ato/Azo`.
- Extended `src/beltrami_jax/spectre_geometry.py`.
  - Implements centroid `Lrzaxis=1` axis initialization for toroidal coordinate-axis branches.
  - Recomputes the free-boundary coordinate axis consistently with SPECTRE geometry unpacking.
  - Generates cylindrical and toroidal `Linitialize=1` interface rows from normalized SPECTRE fluxes.

What this closes:

- JAX-native `dMA/dMD` parity is no longer blocked on the packaged toroidal axis/generated-interface branches.
- The packaged public SPECTRE volume matrices now match at roundoff for:
  - `G2V32L1Fi`: cylindrical coordinate-singularity and bulk volumes.
  - `G3V3L3Fi`: toroidal generated-interface volumes.
  - `G3V3L2Fi_stability`: toroidal generated-interface stability case.
  - `G3V8L3Free`: explicit-interface free-boundary plasma volumes and vacuum exterior volume.
- `beltrami_jax` can now produce SPECTRE-shaped `Ate/Aze/Ato/Azo` arrays from TOML/interface geometry for validated branches when supplied the same post-constraint local `mu` and flux vector that SPECTRE used.

Verification:

- Full local test suite after this lane: `105 passed in 140.92s`, total coverage `91.71%`.

Important bug fixed:

- `G3V8L3Free/lvol1` previously showed a `dMA` relative mismatch around `2.2e-2` while `dMD` was exact. The cause was not the matrix formula. SPECTRE recomputes the toroidal coordinate axis during geometry unpacking when geometry degrees of freedom are active, even when the TOML file contains a positive `Rac(0)`. Implementing centroid `rzaxis` and using it for the free-boundary axis brings `lvol1` to roundoff matrix parity.

Current remaining blockers before a serious SPECTRE replacement PR:

- JAX-native rotational-transform and plasma-current diagnostics.
- The nonlinear local/global constraint loop that updates `mu`, `dtflux`, and `dpflux` from those diagnostics.
- More non-stellarator-symmetric and high-resolution public fixtures.
- Production sparse or matrix-free solve strategy if dense local solves become too expensive in SPECTRE-scale use.
- A small SPECTRE fork experiment after the above is in place, preferably with one backend flag and a narrow adapter surface.

## 16. 2026-04-30 Progress: Full TOML-to-`Ate/Aze/Ato/Azo` Linear Entry Point

New completed ingredients:

- Extended `src/beltrami_jax/spectre_solve.py`.
  - Adds `SpectreMultiVolumeSolve`.
  - Adds `solve_spectre_volumes_from_input(summary, ...)`.
  - Adds `solve_spectre_toml(path, ...)`.
- Added `examples/spectre_toml_full_solve.py`.
  - Demonstrates the full current SPECTRE linear replacement lane:
    SPECTRE TOML/interface geometry -> JAX `dMA/dMD/dMB/dMG` assembly -> JAX solve -> full `Ate/Aze/Ato/Azo` coefficient block.
  - Writes a coefficient archive, JSON summaries, and a static reviewer-facing parity figure.
- Added `docs/_static/spectre_toml_full_solve.png` and matching summary JSON.

What this closes:

- Users no longer need to call the per-volume helper manually to get full SPECTRE-shaped output.
- The high-level integration API can now return the same coefficient object that SPECTRE downstream code needs.
- The packaged `G3V3L3Fi` full-case linear validation now reconstructs all packed vector-potential coefficients with global relative error `1.685e-14` when supplied SPECTRE's final branch `mu`/flux state.
- The maximum relative residual across those three volume solves is `7.089e-13`.

What this still does not close:

- This is still a linear assembly/solve parity lane. It does not yet compute the post-constraint `mu`, `dtflux`, or `dpflux` state from TOML alone.
- Exact fixture parity still injects SPECTRE's final local branch state. That is now clearly documented as a validation bridge.
- Full SPEC/SPECTRE removal requires the next lane: JAX-native current/transform diagnostics plus the nonlinear `Lconstraint` update loop.

Best next implementation order:

1. Port `plasma_current_mod.F90::compute_plasma_current` first, because it is algebraic in the solved coefficients and interface metrics.
2. Validate current values and derivatives against a SPECTRE export instrumented to dump `dItGpdxtp`.
3. Port or approximate-scope `magnetic_field_mod.F90::compute_rotational_transform`, beginning with the released `Lsparse`/`Lsvdiota` branch used by public tests.
4. Add a local Newton loop that wraps `solve_spectre_beltrami_branch`, diagnostics, and `evaluate_spectre_constraints`.
5. Only then modify the SPECTRE fork to call the JAX backend without SPEC/Fortran Beltrami assembly.

## 17. 2026-04-30 Progress: Current Diagnostics, Local Constraint Hook, and SPECTRE Injection Seam

New completed ingredients in `beltrami_jax`:

- Extended `src/beltrami_jax/spectre_backend.py`.
  - The assembled SPECTRE backend now returns the two branch derivative solves, derivative residuals, magnetic-energy integral, and magnetic-helicity integral in addition to the primary solution.
  - The NumPy adapter returns the same data for a future SPECTRE-side call.
  - The equal-size batch path returns derivative solves and integrals for batched validation/performance tests.
- Added `src/beltrami_jax/spectre_diagnostics.py`.
  - Ports the algebraic SPECTRE plasma-current diagnostic path from solved `Ate/Aze/Ato/Azo` coefficients.
  - Returns toroidal/poloidal current values and derivative currents from the derivative vector-potential solves.
  - Supports the `Lconstraint=-2` outer-face branch with optional radial field inclusion.
- Extended `src/beltrami_jax/spectre_constraints.py`.
  - Preserves the existing branch-solve API and adds TOML/state-aware local constraint helpers.
  - Adds `SpectreLocalConstraintEvaluation`, `SpectreRotationalTransformDiagnostic`, `spectre_local_unknown_count`, `evaluate_spectre_local_constraints`, and `evaluate_spectre_helicity_constraint`.
- Extended `src/beltrami_jax/spectre_solve.py`.
  - Per-volume solves now expose derivative vector-potential blocks.
  - `solve_spectre_volume_from_input(..., solve_local_constraints=True)` can solve zero-unknown branches, plasma `Lconstraint=2` helicity, plasma `Lconstraint=-2` current, and vacuum `Lconstraint=0` current.
  - Full multi-volume TOML solves pass the local-constraint flags through to each selected volume.
- Extended `src/beltrami_jax/spectre_input.py`.
  - Includes `oita` in parsed constraint metadata so SPECTRE transform branches have access to both inner and outer targets.
- Updated docs and README.
  - Documents derivative outputs, plasma-current diagnostics, local helicity/current constraints, and the experimental SPECTRE injection hook.

New SPECTRE fork ingredients in `/Users/rogerio/local/spectre`:

- Added `wrapper_funcs_mod.set_vec_pot`.
  - This is the setter counterpart to SPECTRE's existing `get_vec_pot`.
  - It injects a full radial-first `Ate/Aze/Ato/Azo` block into SPECTRE Fortran memory.
- Added `spectre.utils.set_vec_pot_flat`.
  - Performs Python-side shape checks and calls the Fortran setter.
- Added `spectre/beltrami_jax_backend.py`.
  - Provides `solve_input_file_with_beltrami_jax(...)` and `apply_beltrami_jax_solution(...)`.
  - Calls `beltrami_jax.solve_spectre_toml(...)` and optionally injects the returned full coefficient block into SPECTRE.
- Added `SPECTRE.solve_beltrami_jax(...)`.
  - Experimental Python method for `SPECTRE.from_input_file(...)` objects.
  - Keeps the default Fortran Beltrami path untouched; this is a validation/integration seam, not a default backend switch.
- Added SPECTRE optional dependency extra `beltrami-jax`.
- Added SPECTRE docs under `docs/source/get_started/calculating_field.rst`.

Verification:

- `python -m py_compile` passed for the modified `beltrami_jax` modules.
- Targeted `beltrami_jax` tests passed: `41 passed in 109.06s`.
- SPECTRE Python files compile with `python -m py_compile`.
- SPECTRE Fortran has not been rebuilt in this lane, so `set_vec_pot` still needs a compile/f90wrap validation before an upstream PR.

Important design decision:

- The SPECTRE-side change is intentionally small and optional. It does not yet remove SPECTRE's Fortran Beltrami code path or change default runtime behavior. Instead it creates the minimal reviewed seam needed to compare a full `beltrami_jax` coefficient block inside SPECTRE.

Current remaining blockers:

- Port `magnetic_field_mod.F90::compute_rotational_transform` to JAX and validate `iota` plus derivative rows against SPECTRE dumps.
- Add the `Lconstraint=1` local Newton loop once rotational-transform diagnostics are available.
- Add the global/semi-global `Lconstraint=3` force-coupled updates.
- Rebuild SPECTRE with `set_vec_pot`, run `SPECTRE.solve_beltrami_jax(update_fortran=True)`, and compare injected coefficients, Beltrami errors, force modes, and downstream field diagnostics against the Fortran backend.
- Add a real backend flag in SPECTRE only after those validation plots are at roundoff-level agreement.

## 18. 2026-04-30 Progress: SPECTRE Runtime Backend Switch and Force-Seam Validation

New completed ingredients in the local SPECTRE fork:

- Rebuilt SPECTRE locally with the new `wrapper_funcs_mod.set_vec_pot` symbol.
- Made field tracing lazy/optional in `spectre.core.core` and `spectre.__init__`.
  - The local Python 3.13 environment can install `CyRK`, but `CyRK` cannot load `libomp.dylib`.
  - SPECTRE import, Beltrami solve, and force workflows no longer require field tracing unless the user explicitly calls tracing APIs.
- Extended `spectre.beltrami_jax_backend`.
  - Adds `solve_current_state_with_beltrami_jax(test, ...)`.
  - Serializes the current in-memory SPECTRE interface state to a temporary TOML file, so optimization/force workflows can solve the geometry currently packed from `xin`, not only the original input file.
  - Keeps the backend serial-only for now; MPI use raises a clear runtime error instead of silently doing the wrong thing.
- Extended `SPECTRE.solve_beltrami_jax(...)`.
  - Adds `use_current_state=True` for force workflows.
- Extended `spectre.force_targets.force_real(...)`.
  - Adds `beltrami_backend="fortran" | "jax"`.
  - The default remains the current Fortran backend.
  - The JAX path packs interfaces with `unpack_interfaces(xin, test)`, calls `test.solve_beltrami_jax(use_current_state=True, update_fortran=True)`, and then uses the existing SPECTRE `calc_b2mag_jump` force diagnostic.
- Extended `scripts/run/calc_spectre_field.py`.
  - Adds `--beltrami-backend {fortran,jax}`.
  - Adds `--beltrami-jax-local-constraints`.

Runtime validation completed:

- `python -m pip install -e .` succeeded in `/Users/rogerio/local/spectre`.
- `import spectre` now succeeds.
- `SPECTRE.from_input_file("tests/compare/G3V3L3Fi/input.toml")` succeeds.
- `hasattr(obj.wrapper_funcs_mod, "set_vec_pot")` is `True`.
- `obj.solve_beltrami_jax(update_fortran=True)` on `G3V3L3Fi` injects the returned coefficient block exactly:
  - JAX linear max relative residual: `8.240902103055836e-13`.
  - SPECTRE memory injection relative error: `0.0`.
  - injected coefficient shape: `(19, 23)`.
- `force_real(..., beltrami_backend="jax")` force-seam comparison against SPECTRE Fortran:
  - `G3V3L2Fi_stability`, `Lconstraint=2` local helicity branch: relative force error `1.2510234941523128e-12`.
  - `G3V3L3Fi`, `Lconstraint=3` semi-global flux branch: relative force error `1.6675304372497828e-3`.
  - `G2V32L1Fi`, `Lconstraint=1` rotational-transform branch: relative force error `2.4055140334128033e-2`.
- Added reviewer-facing figure:
  - `docs/_static/spectre_backend_seam_runtime.png`.
  - `docs/_static/spectre_backend_seam_runtime_summary.json`.

SPECTRE validation caveat:

- `python -m pytest tests/compare/test_compare_to_spec.py -vv -s` in `/Users/rogerio/local/spectre` still fails on the default Fortran cylinder comparison before any JAX backend call:
  - failure message: `error computing derivatives of mu wrt geometry at fixed transform`;
  - location: `fortran_src/forces_mod.F90`;
  - branch: `G2V32L1Fi`, `Lconstraint=1`.
- This failure is in SPECTRE's existing Fortran derivative path, not in the optional JAX injection path. It reinforces the same open blocker: the rotational-transform branch must be ported and validated before a full SPEC-removal PR can claim coverage of `Lconstraint=1`.

Interpretation:

- The SPECTRE injection mechanism is validated.
- The force backend switch reaches roundoff-level agreement for a branch whose nonlinear local constraint is now represented in `beltrami_jax`.
- The two non-roundoff force comparisons are not injection failures. They identify the remaining planned SPEC-removal work:
  - `Lconstraint=1` needs JAX-native rotational-transform diagnostics.
  - `Lconstraint=3` needs the SPECTRE global/semi-global flux update.

Current PR readiness status:

- Ready for an internal experimental branch and reviewer discussion of the adapter seam.
- Not ready for a claim that SPEC/SPECTRE Beltrami is fully removed.
- The next implementation lane must target `magnetic_field_mod.F90::compute_rotational_transform` and the `Lconstraint=3` global update before changing SPECTRE defaults.
