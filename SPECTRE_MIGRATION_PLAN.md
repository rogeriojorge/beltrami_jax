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

What is not yet done:

- It does not yet accept the same user-facing input as SPECTRE, namely a TOML configuration or interface Fourier geometry with SPECTRE's exact basis, packing, constraints, and branch logic.
- It does not yet reproduce SPECTRE/SPEC HDF5 vector-potential coefficient datasets `vector_potential/Ate`, `Aze`, `Ato`, and `Azo`.
- It does not yet implement SPECTRE's full geometry integral assembly, packing rules, coordinate-singularity handling, non-stellarator-symmetric branches, free-boundary details, or constraint logic in JAX.
- The current docs can be read as claiming more SPECTRE readiness than exists. They must be rewritten so fixtures are presented as developer validation assets and the current geometry mode is presented as a prototype, not as arbitrary 3D SPECTRE geometry.

Recommended immediate direction:

- Treat `beltrami_jax` as a strict SPECTRE Beltrami backend project, not as a loosely similar Beltrami solver.
- First match SPECTRE's existing Python/Fortran wrappers at the vector-potential coefficient level.
- Only after coefficient parity is established should we replace SPECTRE assembly and constraint logic with JAX-native code.

## 2. Direct Answers to the Collaborator's Email

### 2.1 What is the input to the code?

Current answer:

- `beltrami_jax` has two input modes today.
- Mode 1 is the regression/developer mode: load packaged fixtures or text dumps containing already assembled SPEC-style matrices and vectors: `d_ma`, `d_md`, `d_mb`, optional `d_mg`, `mu`, `psi`, RHS, matrix, and expected solution.
- Mode 2 is the internal geometry prototype: create a `FourierBeltramiGeometry`, build a `FourierModeBasis`, assemble a toy/SPEC-style linear system internally, and solve it.

Important correction:

- The current geometry-driven mode is not equivalent to SPECTRE or SPEC arbitrary 3D interface input.
- It uses a shaped large-aspect-ratio torus parameterization in `src/beltrami_jax/geometry.py`.
- The user-facing SPECTRE target must instead accept SPECTRE TOML or SPECTRE interface Fourier coefficients, then assemble the exact SPECTRE Beltrami system and return SPECTRE-compatible vector-potential coefficients.

Final target input contract:

- `BeltramiFromSpectreInput`: parse SPECTRE TOML or a SPECTRE object/state.
- `SpectreInterfaceGeometry`: hold `Igeometry`, `Nfp`, `Mvol`, `Mpol`, `Ntor`, `Lrad`, `im`, `in`, interface `R/Z` Fourier coefficients, fluxes, current/helicity/iota constraints, stellarator symmetry flags, free-boundary flags, and numerical resolution.
- `BeltramiLinearSystem`: remain as a low-level escape hatch for already assembled matrices.
- Fixtures should remain available for tests, but they should not be the primary user-facing workflow.

### 2.2 Have vector potential coefficients been compared to SPEC HDF5 output?

Current `beltrami_jax` answer:

- No, not directly yet.
- Current `beltrami_jax` validation compares to dense linear systems dumped from an instrumented local SPEC build: operator, RHS, and packed solution vector.
- This is useful but not enough for SPECTRE integration. The collaborator is right that `Ate`, `Aze`, `Ato`, and `Azo` in `.h5` should match up to numerical resolution and packing conventions.

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

Required `beltrami_jax` work:

- Add HDF5 readers for SPECTRE/SPEC reference files.
- Add vector-potential coefficient containers.
- Add exact pack/unpack mapping between packed JAX solution vectors and `Ate/Aze/Ato/Azo`.
- Add tests that compare `beltrami_jax` vector-potential coefficients to SPECTRE/SPEC `.h5` datasets.

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

- No test compares `beltrami_jax` output against SPECTRE/SPEC HDF5 vector-potential coefficients.
- No test exercises SPECTRE TOML input as the user-facing input.
- No test validates exact SPECTRE packing/unpacking of `Ate/Aze/Ato/Azo`.

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

Current gap:

- Examples teach the current package API, but not the final SPECTRE-facing workflow.
- Add examples that start from SPECTRE TOML or SPECTRE `.h5` files:
  - `examples/spectre_h5_vecpot_validation.py`
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

Implement exact SPECTRE packing:

- Reproduce `packab('P')` and `packab('U')` behavior.
- Map packed vector `a` to per-volume coefficient arrays:
  - `Ate(lvol, ideriv, ii)%s(ll)`.
  - `Aze(lvol, ideriv, ii)%s(ll)`.
  - `Ato(lvol, ideriv, ii)%s(ll)`.
  - `Azo(lvol, ideriv, ii)%s(ll)`.
- Preserve radial indexing for Chebyshev and Zernike branches.
- Preserve stellarator-symmetric and non-stellarator-symmetric coefficient families.
- Expose `get_vec_pot_flat()` equivalent returning arrays matching SPECTRE's shape `(sum(Lrad + 1), mn)`.

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
- Update validation docs to say vector-potential HDF5 comparison is not yet implemented in `beltrami_jax`.

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

Goal: map `beltrami_jax` packed solutions to SPECTRE's `Ate/Aze/Ato/Azo`.

New code:

- `src/beltrami_jax/spectre_packing.py`.
- `build_spectre_mode_table(...)`.
- `pack_vector_potential(...)`.
- `unpack_vector_potential(...)`.
- `flatten_vector_potential_like_spectre(...)`.

Reference source:

- `fortran_src/packing_mod.F90`.
- `fortran_src/wrapper_funcs_mod.F90:get_vec_pot`.
- `fortran_src/vector_potential_writer_mod.F90`.

Tests:

- Use SPECTRE reference cases and a known packed vector.
- Roundtrip pack/unpack.
- Compare flattened arrays to SPECTRE's `get_vec_pot_flat()`.

Acceptance:

- For a SPECTRE-produced packed solution, `beltrami_jax` reproduces `Ate/Aze/Ato/Azo` arrays bitwise or to roundoff.

### Phase 3: SPECTRE Matrix/RHS Extraction

Goal: expand validation from old SPEC text dumps to SPECTRE's current released code.

Options:

- Preferred temporary path: add a SPECTRE wrapper helper that exposes `dMA`, `dMD`, `dMB`, `dMG`, matrix, RHS, solution, and metadata after `solve_field`.
- Alternative: instrument SPECTRE locally as was done with SPEC.

New code:

- `tools/build_spectre_fixture.py`.
- `src/beltrami_jax/spectre_reference.py`.

SPECTRE helper target:

- Add Python wrapper functions in a SPECTRE fork:
  - `get_beltrami_linear_system(lvol)`.
  - `get_beltrami_solution(lvol, ideriv=0)`.
  - `get_beltrami_matrix_rhs(lvol)`.

Acceptance:

- `beltrami_jax` solves SPECTRE-exported systems and matches:
  - matrix;
  - RHS;
  - packed solution;
  - `Ate/Aze/Ato/Azo`.

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
- `src/beltrami_jax/spectre_integrals.py`.
- `src/beltrami_jax/spectre_assembly.py`.
- `src/beltrami_jax/spectre_constraints.py`.

Acceptance:

- JAX-assembled `dMA`, `dMD`, `dMB`, `dMG` match SPECTRE for the same input cases.
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

1. Rewrite README and docs to answer the collaborator's three questions explicitly.
2. Add `spectre_io.py` to load `Ate/Aze/Ato/Azo` from SPECTRE HDF5 files.
3. Add `SpectreVectorPotential` dataclass and comparison diagnostics.
4. Add tests using a small copied/extracted HDF5 reference fixture or generated miniature HDF5.
5. Implement SPECTRE flattening/unflattening conventions.
6. Add an example script that loads a SPECTRE reference HDF5 and produces coefficient validation plots.
7. Create a SPECTRE fork/branch after design decisions are confirmed.
8. Add SPECTRE wrapper functions to export Beltrami matrices, RHS, packed solution, and vector-potential arrays.
9. Add `beltrami_jax` tests that compare against SPECTRE-exported cases.
10. Start JAX port of SPECTRE assembly branch by branch.

## 9. Open Risks

- Exact SPECTRE packing is likely more important and subtle than the linear solve itself.
- The current internal geometry assembly may distract from the real SPECTRE target if docs are not clarified.
- SPECTRE's force and derivative path uses Beltrami derivatives, not only the base coefficient vector.
- Local SPECTRE build required patches on macOS with Python 3.13, Pydantic 2.13, and gfortran 15; this should be separated from Beltrami backend work.
- `beltrami_jax` currently avoids hard dependencies beyond `jax`; HDF5 validation needs optional dependencies.
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

At the moment beltrami_jax has two paths. The fixture path is a developer validation path where SPEC/SPECTRE-style matrices are already assembled and loaded, and the JAX code solves the packed Beltrami system. The geometry-driven path currently in the repository is a prototype shaped-torus assembly, not yet the full SPECTRE geometry path. The target is to make the user-facing input SPECTRE-like: interface geometry, resolution, flux/current/helicity constraints, and branch flags, not prebuilt fixtures.

You are also right about vector-potential coefficients. The current beltrami_jax validation checks dumped matrices, RHS, and packed solutions, but it does not yet directly compare HDF5 vector_potential/Ate, Aze, Ato, Azo. I checked the released SPECTRE tests and they already provide exactly that reference path. I am adding HDF5 vector-potential readers and pack/unpack parity tests so beltrami_jax will compare directly against those datasets.

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
