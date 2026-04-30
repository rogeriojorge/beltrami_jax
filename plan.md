# beltrami_jax Plan, Context, Restart Log

This file is the working memory for the `beltrami_jax` project. It is intended to be complete enough that the work can be restarted from scratch by reading only this document and the repository itself.

It serves two roles:

1. Project plan and technical design document.
2. Running log of commands, outcomes, failures, decisions, and next steps.

Update policy:

- Append new dated log entries whenever meaningful work is done.
- Keep "Current Status", "Open Gaps", and "Next Steps" current.
- Record exact commands when they matter for reproducibility.
- Record failures and dead ends, not only successful steps.

## 1. Project Goal

Create a public Python package at [github.com/rogeriojorge/beltrami_jax](https://github.com/rogeriojorge/beltrami_jax) that ports the SPEC/SPECTRE Beltrami linear solver interface from Fortran to JAX.

The immediate target is not all of SPEC. The target is the linear Beltrami module needed by SPECTRE:

- differentiable end to end
- fast and vectorized
- simple to install with `pip`
- minimal runtime dependencies, ideally only `jax`
- validated against SPEC outputs
- packaged with tests, CI, examples, documentation, and restartable developer workflow

The long-term objective is to make this code realistic for later integration into SPECTRE once SPECTRE becomes open source.

## 2. External Context

Origin of the project:

- EPFL/SPECTRE interest exists in replacing the current Fortran Beltrami interface with a JAX implementation.
- SPECTRE is not yet public, so SPEC is the temporary reference implementation.
- The relevant SPEC source files pointed out in the original discussion were approximately:
  - `mp00ac.f90`
  - `ma00aa.f90`
- The expectation is that SPECTRE's Beltrami stage is similar in structure but cleaner than SPEC's legacy Fortran.

Working interpretation of the request:

- understand SPEC enough to reproduce the linear Beltrami solve
- extract or reconstruct the linear algebra used by SPEC
- implement the equivalent operator solve in JAX
- make the code differentiable and suitable for future embedding in SPECTRE
- build the surrounding engineering quality now: tests, docs, CI, examples, packaging

## 3. Repositories and Local Paths

Main repository:

- Remote: [https://github.com/rogeriojorge/beltrami_jax](https://github.com/rogeriojorge/beltrami_jax)
- Local checkout: `/Users/rogerio/local/beltrami_jax`
- Current remote URL:
  - `https://github.com/rogeriojorge/beltrami_jax.git`

Reference code:

- SPEC remote: [https://github.com/PrincetonUniversity/SPEC](https://github.com/PrincetonUniversity/SPEC)
- Local checkout: `/Users/rogerio/local/SPEC`
- SPEC CMake build directory: `/Users/rogerio/local/SPEC/build`
- SPEC executable: `/Users/rogerio/local/SPEC/build/build/bin/xspec`

Reference run artifacts:

- Directory: `/Users/rogerio/local/SPEC_runs`
- Important files currently present:
  - `G3V01L0Fi.001.sp`
  - `G3V01L0Fi.001.log`
  - `G3V01L0Fi.001.sp.end`
  - `G3V01L0Fi.001.sp.h5`
  - `G3V01L0Fi.debug.sp`
  - `G3V01L0Fi.debug.log`
  - `G3V01L0Fi.debug.sp.end`
  - `G3V01L0Fi.debug.sp.h5`
  - `G3V01L0Fi.dump.sp`
  - `G3V01L0Fi.dump.log`
  - `G3V01L0Fi.dump.sp.end`
  - `G3V01L0Fi.dump.sp.h5`
  - `G3V01L0Fi.dump.lvol1.meta.txt`
  - `G3V01L0Fi.dump.lvol1.dma.txt`
  - `G3V01L0Fi.dump.lvol1.dmd.txt`
  - `G3V01L0Fi.dump.lvol1.dmb.txt`
  - `G3V01L0Fi.dump.lvol1.matrix.txt`
  - `G3V01L0Fi.dump.lvol1.rhs.txt`
  - `G3V01L0Fi.dump.lvol1.solution.txt`
  - `G1V03L0Fi.sp`
  - `G1V03L0Fi.dump.sp`
  - `G1V03L0Fi.dump.lvol2.meta.txt`
  - `G1V03L0Fi.dump.lvol2.dma.txt`
  - `G1V03L0Fi.dump.lvol2.dmd.txt`
  - `G1V03L0Fi.dump.lvol2.dmb.txt`
  - `G1V03L0Fi.dump.lvol2.dmg.txt`
  - `G1V03L0Fi.dump.lvol2.matrix.txt`
  - `G1V03L0Fi.dump.lvol2.rhs.txt`
  - `G1V03L0Fi.dump.lvol2.solution.txt`
  - `G3V02L0Fr_LU.sp`
  - `G3V02L0Fr_LU.dump.lvol2.meta.txt`
  - `G3V02L0Fr_LU.dump.lvol2.dma.txt`
  - `G3V02L0Fr_LU.dump.lvol2.dmd.txt`
  - `G3V02L0Fr_LU.dump.lvol2.dmb.txt`
  - `G3V02L0Fr_LU.dump.lvol2.dmg.txt`
  - `G3V02L0Fr_LU.dump.lvol2.matrix.txt`
  - `G3V02L0Fr_LU.dump.lvol2.rhs.txt`
  - `G3V02L0Fr_LU.dump.lvol2.solution.txt`
  - `G3V02L0Fr_LU.dump.lvol3.meta.txt`
  - `G3V02L0Fr_LU.dump.lvol3.dma.txt`
  - `G3V02L0Fr_LU.dump.lvol3.dmd.txt`
  - `G3V02L0Fr_LU.dump.lvol3.dmb.txt`
  - `G3V02L0Fr_LU.dump.lvol3.dmg.txt`
  - `G3V02L0Fr_LU.dump.lvol3.matrix.txt`
  - `G3V02L0Fr_LU.dump.lvol3.rhs.txt`
  - `G3V02L0Fr_LU.dump.lvol3.solution.txt`

Environment:

- Current working shell used during development: `zsh`
- Current date when this file was written: `2026-04-17`
- Local timezone at that time: `America/Chicago`

## 4. What SPEC Is Doing

High-level:

- SPEC computes stepped-pressure equilibria in the MRxMHD framework.
- Each relaxed plasma region solves a Beltrami problem:
  - `curl(B) = mu B`
- In discrete form, SPEC builds a linear system for the vector-potential coefficients in each region.
- The Beltrami stage is only one piece of a larger nonlinear equilibrium loop.

Relevant Fortran files identified in SPEC:

- `src/ma00aa.f90`
  - computes geometry-dependent spectral integrals used to assemble the operator
- `src/matrix.f90`
  - assembles the matrices used by the linear solve
- `src/mp00ac.f90`
  - constructs and solves the linear system
- `src/packab.f90`
  - packs and unpacks the vector-potential degrees of freedom
- `src/ma02aa.f90`
  - outer nonlinear constraint iteration around the linear solve

Current understanding of the linear stage:

- SPEC constructs a matrix
  - `M = A - mu D`
- For plasma regions, the right-hand side is
  - `rhs = -B psi`
- For vacuum regions, the right-hand side is
  - `rhs = -G - B psi`
- The solve returns vector-potential coefficients `a`
- SPEC also evaluates quadratic functionals from the solved coefficients:
  - magnetic energy
  - magnetic helicity
- These values feed later constraint handling and diagnostics

Important detail from `mp00ac.f90`:

- The Fortran comments explicitly document the linear system structure.
- The code supports both direct factorization and GMRES-based solves.
- For the first JAX port, reproducing the dense direct linear algebra is the priority.

## 5. Why This Is Physically Relevant

Beltrami fields are force-free fields:

- `curl(B) = mu B`

This means:

- `J x B = 0` in the relaxed region
- magnetic energy is minimized subject to helicity and flux constraints

In the MRxMHD/SPEC setting:

- the plasma is divided into regions
- each region relaxes to a Beltrami state
- ideal interfaces enforce jump conditions and preserve selected topological constraints
- this allows equilibria with islands, chaos, and non-nested field structure that are difficult for standard nested-flux-surface equilibrium solvers

For tokamaks and stellarators:

- tokamak axisymmetric equilibria are often treated with Grad-Shafranov solvers
- SPEC instead treats stepped-pressure relaxed regions, which is a different model
- stellarators and strongly shaped 3D devices benefit from MRxMHD because the field structure can be non-integrable

## 6. Primary Documentation and Literature

Official SPEC documentation:

- SPEC documentation homepage:
  - [https://princetonuniversity.github.io/SPEC/](https://princetonuniversity.github.io/SPEC/)
- SPEC manual PDF:
  - [https://princetonuniversity.github.io/SPEC/SPEC_manual.pdf](https://princetonuniversity.github.io/SPEC/SPEC_manual.pdf)
- SPEC flowchart PDF:
  - [https://princetonuniversity.github.io/SPEC/SPEC_flowchart.pdf](https://princetonuniversity.github.io/SPEC/SPEC_flowchart.pdf)

Core physics references:

- Taylor, 1974, "Relaxation of Toroidal Plasma and Generation of Reverse Magnetic Fields"
  - [https://doi.org/10.1103/PhysRevLett.33.1139](https://doi.org/10.1103/PhysRevLett.33.1139)
- Taylor, 1986, "Relaxation and magnetic reconnection in plasmas"
  - [https://doi.org/10.1103/RevModPhys.58.741](https://doi.org/10.1103/RevModPhys.58.741)

MRxMHD / SPEC references:

- Hudson et al., 2012, "Computation of multi-region relaxed magnetohydrodynamic equilibria"
  - arXiv: [https://arxiv.org/abs/1211.3072](https://arxiv.org/abs/1211.3072)
- Dennis et al., 2012, "The Infinite Interface Limit of Multiple-Region Relaxed MHD"
  - arXiv: [https://arxiv.org/abs/1212.4917](https://arxiv.org/abs/1212.4917)
- Qu et al., 2020, "Stepped pressure equilibrium with relaxed flow and applications in reversed-field pinch plasmas"
  - arXiv: [https://arxiv.org/abs/2001.06984](https://arxiv.org/abs/2001.06984)
- Loizu et al., 2021, "Computation of multi-region, relaxed magnetohydrodynamic equilibria with prescribed toroidal current profile"
  - Cambridge article:
    [https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/computation-of-multiregion-relaxed-magnetohydrodynamic-equilibria-with-prescribed-toroidal-current-profile/38C1F45C49272E111E28CDD763903BD8](https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/computation-of-multiregion-relaxed-magnetohydrodynamic-equilibria-with-prescribed-toroidal-current-profile/38C1F45C49272E111E28CDD763903BD8)

Beltrami/Taylor-state numerical solver references that may influence implementation choices:

- O'Neill and Cerfon, 2018, "An integral equation-based numerical solver for Taylor states in toroidal geometries"
  - [https://arxiv.org/abs/1611.01420](https://arxiv.org/abs/1611.01420)
- O'Neill et al., 2019, "Taylor States in Stellarators: A Fast High-order Boundary Integral Solver"
  - [https://arxiv.org/abs/1902.01205](https://arxiv.org/abs/1902.01205)
- Cerfon and O'Neil, 2014, "Exact axisymmetric Taylor states for shaped plasmas"
  - [https://arxiv.org/abs/1406.0481](https://arxiv.org/abs/1406.0481)

Literature study notes:

- The immediate port target is SPEC's discrete operator solve, not a new continuum solver from scratch.
- The boundary-integral Taylor-state literature is still useful for validation ideas, analytic test cases, and performance comparisons.
- The MRxMHD papers clarify the role of regionwise Beltrami states and why the solve is embedded in a constrained variational problem rather than used alone.

## 7. Work Completed So Far

### 7.1 GitHub and Repository Setup

Completed:

- Confirmed GitHub CLI authentication for account `rogeriojorge`
- Created the public repository:
  - `rogeriojorge/beltrami_jax`
- Cloned it locally into:
  - `/Users/rogerio/local/beltrami_jax`

Important current repo state:

- Initial commit created on `main`
- To check the current published commit at any time, run:
  - `git rev-parse HEAD`
- Branch `main` is pushed to `origin/main`

### 7.2 SPEC Clone, Build, and Run

Completed:

- Cloned SPEC into `/Users/rogerio/local/SPEC`
- Installed missing build dependency `cmake`
- Confirmed Homebrew MPI, HDF5, FFTW, OpenBLAS, GCC, and M4 were available
- Configured and built SPEC successfully
- Ran multiple SPEC examples successfully

Important issue encountered:

- A normal `git clone` attempt for SPEC failed with packfile/index-pack problems
- Workaround that succeeded:
  - `gh repo clone PrincetonUniversity/SPEC /Users/rogerio/local/SPEC -- --depth 1`

Important build issue encountered:

- SPEC built with modern `gfortran` but hit a runtime problem related to legacy FORMAT handling
- Workaround that succeeded:
  - rebuild SPEC with `-DCMAKE_Fortran_FLAGS='-std=legacy'`

This is important because a restart from scratch should not assume that a plain modern Fortran build will run correctly without the legacy flag.

### 7.3 SPEC Runs Performed

Baseline case:

- Input:
  - `/Users/rogerio/local/SPEC_runs/G3V01L0Fi.001.sp`
- Run command:
  - `OMP_NUM_THREADS=1 /Users/rogerio/local/SPEC/build/build/bin/xspec G3V01L0Fi.001.sp > G3V01L0Fi.001.log 2>&1`
- Result:
  - successful run
  - produced `.log`, `.sp.end`, and `.sp.h5`

Diagnostic case:

- Input:
  - `/Users/rogerio/local/SPEC_runs/G3V01L0Fi.debug.sp`
- Diagnostic changes included enabling verbose flags such as:
  - `Lcheck = 1`
  - `Wma00aa = T`
  - `Wmatrix = T`
  - `Wmp00ac = T`
  - `Wcurent = T`
  - `Wxspech = T`
  - `Wreadin = T`
- Result:
  - successful run
  - useful residual/error output in `jo00aa`

Recorded diagnostic error values:

- average errors:
  - `E^s = 1.176657295318059E-04`
  - `E^t = 7.095240669109291E-05`
  - `E^z = 1.352225005486860E-07`
- maximum errors:
  - `E^s = 7.109743391410444E-04`
  - `E^t = 3.619067459675571E-04`
  - `E^z = 1.704219808615110E-06`

Dump case for JAX fixture extraction:

- Input:
  - `/Users/rogerio/local/SPEC_runs/G3V01L0Fi.dump.sp`
- Purpose:
  - export the exact dense linear system solved by SPEC for one volume
- Run command:
  - `SPEC_DUMP_LINEAR_SYSTEM=/Users/rogerio/local/SPEC_runs/G3V01L0Fi.dump.lvol1 OMP_NUM_THREADS=1 /Users/rogerio/local/SPEC/build/build/bin/xspec G3V01L0Fi.dump.sp > G3V01L0Fi.dump.log 2>&1`
- Result:
  - successful run
  - exported `meta`, `dma`, `dmd`, `dmb`, `matrix`, `rhs`, and `solution` text files

Dump metadata recorded:

- `lvol = 1`
- `nn = 361`
- `mu = 0.0`
- `psi_t = 3.1830988618379069E-01`
- `psi_p = 0.0`

### 7.4 SPEC Instrumentation Patch

Important local-only modification:

- A temporary instrumentation hook was added to the local SPEC checkout in:
  - `/Users/rogerio/local/SPEC/src/mp00ac.f90`
- It adds support for the environment variable:
  - `SPEC_DUMP_LINEAR_SYSTEM`
- That hook writes the matrices and vectors needed for JAX regression tests.

This patch is only in the local SPEC clone at the moment. It is not part of `beltrami_jax`.

Implications:

- It is acceptable as a local developer tool for extracting reference fixtures.
- It should eventually be moved to either:
  - a clean patch file in this repository, or
  - a more stable extraction script/workflow, or
  - be discarded after enough fixtures are collected

### 7.5 Initial `beltrami_jax` Package Skeleton

The following files already exist locally:

- `pyproject.toml`
- `LICENSE`
- `.gitignore`
- `src/beltrami_jax/__init__.py`
- `src/beltrami_jax/types.py`
- `src/beltrami_jax/operators.py`
- `src/beltrami_jax/solver.py`
- `src/beltrami_jax/reference.py`
- `tools/build_spec_fixture.py`
- `examples/solve_spec_fixture.py`
- `examples/parameter_scan.py`
- `examples/autodiff_mu.py`
- `tests/test_reference.py`
- `tests/test_solver.py`
- `tests/test_examples.py`

Module intent:

- `types.py`
  - typed containers for the SPEC-style linear system and solve outputs
- `operators.py`
  - assembly of operator and right-hand side
  - energy, helicity, and residual helpers
- `solver.py`
  - direct dense JAX solve
  - verbose solve path
  - batched parameter scan via `jax.vmap`
- `reference.py`
  - readers for SPEC text dumps and packaged `.npz` fixtures
- `tools/build_spec_fixture.py`
  - converts text dumps into compressed packaged fixtures
- `examples/*`
  - smoke-test style usage demonstrations
- `tests/*`
  - regression, autodiff, vectorization, vacuum path, and examples

### 7.6 Initial Design Decisions Already Made

Design decisions already taken:

- start from the linear algebra boundary between `matrix.f90` and `mp00ac.f90`
- do not port all of SPEC yet
- make the first implementation dense and exact before attempting sparse or matrix-free variants
- enable JAX 64-bit mode for numerical agreement with SPEC
- support both plasma and vacuum right-hand-side structure
- keep runtime dependencies minimal
- make examples and solver verbose rather than silent
- validate against exact dumped SPEC systems first, then broaden to more cases

Deferred for later:

- direct use of `equinox`
- direct use of `lineax`
- sparse operators
- broader SPEC/SPECTRE branch parity beyond the current internal Fourier-geometry model

## 8. Current Status of the `beltrami_jax` Repository

What exists:

- package skeleton
- typed linear-system API
- internal Fourier geometry assembly API
- GMRES / matrix-free iterative solve API
- outer helicity-constrained nonlinear solve API
- JSON/NPZ input-output helpers for standalone workflows
- dense JAX solve
- helper operators
- diagnostics helpers for operator quality and agreement metrics
- benchmark helpers for dense solves and batched parameter scans
- SPEC text-dump loader
- packaged SPEC regression fixtures under `src/beltrami_jax/data/`
- package data module `src/beltrami_jax/data/__init__.py`
- expanded `README.md` with installation, validation workflow, and figures
- examples
- tests passing locally against multiple packaged SPEC fixtures
- editable install validated locally
- documentation tree under `docs/`
- `.readthedocs.yaml`
- local docs build configuration
- GitHub Actions CI workflow source under `.github/workflows/`
- generated figure assets under `docs/_static/`
- local Sphinx build validated with warnings treated as errors
- fixture support for both plasma and vacuum-region RHS reconstruction
- committed publication-style validation and benchmark panels under `docs/_static/`

Currently packaged fixtures:

- `src/beltrami_jax/data/g3v01l0fi_lvol1.npz`
  - toroidal fixed-boundary plasma region
  - size `361`
- `src/beltrami_jax/data/g1v03l0fi_lvol2.npz`
  - compact cylindrical plasma region with nonzero `mu`
  - size `51`
- `src/beltrami_jax/data/g3v02l1fi_lvol1.npz`
  - fixed-boundary 3D plasma region
  - size `361`
- `src/beltrami_jax/data/g3v02l0fr_lu_lvol3.npz`
  - toroidal free-boundary vacuum region
  - size `1548`

What does not exist yet:

- hosted Read the Docs project configuration outside the repository
- a live hosted docs deployment

Known resolved inconsistency:

- tests and examples expect a packaged fixture named something like:
  - `src/beltrami_jax/data/g3v01l0fi_lvol1.npz`
- that file now exists
- the fixture was generated from:
  - `/Users/rogerio/local/SPEC_runs/G3V01L0Fi.dump.lvol1`
- the dense regression workflow now runs successfully locally

Resolved packaging gap:

- `pyproject.toml` points `readme = "README.md"`
- `README.md` now exists
- `importlib.resources` access to packaged data required making `src/beltrami_jax/data/` an importable package

## 9. Current File Map and Responsibilities

Repository root:

- `.gitignore`
  - local env, caches, build products
- `LICENSE`
  - MIT license
- `README.md`
  - user-facing project overview, install instructions, examples, and figures
- `pyproject.toml`
  - package metadata, dependencies, pytest and coverage settings
- `plan.md`
  - this restartable project log and plan
- `.readthedocs.yaml`
  - Read the Docs build definition
- `.github/workflows/ci.yml`
  - GitHub Actions workflow for tests and docs builds

Source package:

- `src/beltrami_jax/__init__.py`
  - public API export list and version
- `src/beltrami_jax/types.py`
  - dataclasses for linear-system inputs and solve results
- `src/beltrami_jax/operators.py`
  - operator and RHS assembly plus functionals
- `src/beltrami_jax/solver.py`
  - JAX dense solve and vectorized parameter scan
- `src/beltrami_jax/diagnostics.py`
  - operator-quality and reference-agreement summaries
- `src/beltrami_jax/benchmark.py`
  - timing helpers for dense solves and batched scans
- `src/beltrami_jax/reference.py`
  - SPEC fixture loading
- `src/beltrami_jax/data/`
  - packaged `.npz` reference data and package marker for `importlib.resources`

Examples:

- `examples/solve_spec_fixture.py`
  - solve and compare against SPEC
- `examples/parameter_scan.py`
  - batched `mu` scan
- `examples/autodiff_mu.py`
  - differentiate solved energy with respect to `mu`
- `examples/benchmark_fixtures.py`
  - quick timing summary for one packaged fixture

Tests:

- `tests/test_reference.py`
  - operator and RHS reconstruction from fixture
- `tests/test_solver.py`
  - regression against SPEC solution, residuals, autodiff, batched solves, diagnostics, benchmarks, vacuum path
- `tests/test_examples.py`
  - smoke-test example scripts using the repository virtual environment

Tools:

- `tools/build_spec_fixture.py`
  - convert dumped SPEC text files into packaged compressed fixture data
- `tools/generate_validation_assets.py`
  - regenerate the committed validation and benchmark figure panels

Documentation:

- `docs/conf.py`
  - Sphinx configuration
- `docs/index.md`
  - landing page and table of contents
- `docs/overview.md`
  - package scope and design choices
- `docs/theory.md`
  - equations, physical interpretation, and numerical model
- `docs/validation.md`
  - SPEC fixture workflow and test strategy
- `docs/examples.md`
  - example scripts and generated figures
- `docs/api.md`
  - public API reference
- `docs/limitations.md`
  - current limitations and future work
- `docs/references.md`
  - literature and documentation references
- `docs/requirements.txt`
  - documentation build dependencies
- `docs/_static/`
  - committed figure assets used by the docs and README, including validation and benchmark panels

## 10. What Worked, What Did Not, and Why

What worked:

- GitHub authentication and repository creation
- SPEC clone through `gh repo clone -- --depth 1`
- SPEC build with CMake after installing `cmake`
- SPEC runtime after adding `-std=legacy`
- baseline SPEC run
- verbose SPEC run
- dumping a dense reference linear system from SPEC
- extending the local SPEC dump hook to select a target volume and export `dMG` / `is_vacuum`
- initial JAX package scaffolding
- packaging the dumped SPEC system into a repository fixture
- packaging multiple dumped SPEC systems into repository fixtures
- editable install with `pip install -e '.[dev]'`
- local test execution at 100 percent coverage
- docs-only editable install path through `pip install -e '.[docs]'`
- Sphinx documentation tree and Read the Docs configuration
- generated figures from the packaged-fixture examples
- generated publication-style validation and benchmark panels from committed fixture data
- local docs build with `sphinx -W`
- GitHub Actions workflow definition for tests and docs builds

What did not work:

- plain initial SPEC clone path failed with packfile/index-pack problems
- initial SPEC runtime without legacy Fortran mode hit format-related issues
- first fixture-generation attempt failed because `reference.py` used `Path.with_suffix`, which stripped the `.lvol1` portion from dump prefixes
- first additional nonzero-`mu` candidate `G1V03L2Fi.001.sp` did not run as shipped because `LBeltrami=2` in that input requires `Lconstraint=2`
- matrix-free GMRES cases do not emit dense matrix dumps through the current SPEC hook, so dump-only copies had to force `Lmatsolver = 1`
- first free-boundary dump target (`lvol = 2` in `G3V02L0Fr_LU`) was not the vacuum region; the actual vacuum system is `lvol = 3`
- first strict docs build failed because autodoc tried to render imported JAX aliases such as `Array`; fixed by excluding those imported symbols from the API page

Why this matters:

- the current code is a strong draft but not yet a ship-ready repository
- the next work should focus on broader fixture coverage and integration surface, not just more dense-kernel validation

## 11. Validation Strategy

Validation should proceed in layers.

Layer 1: exact linear regression

- Use dumped SPEC matrices and vectors
- Verify:
  - assembled JAX operator equals dumped SPEC matrix
  - assembled JAX RHS equals dumped SPEC RHS
  - solved JAX coefficients equal dumped SPEC solution
  - residual norms are near machine precision

Layer 2: differentiability and vectorization

- Differentiate energy or solution-dependent objectives with respect to:
  - `mu`
  - flux vector `psi`
  - possibly selected matrix entries in synthetic tests
- Validate `jax.grad`, `jax.jvp`, and `jax.vmap` behavior

Layer 3: multiple SPEC fixtures

- Export more volumes and more input cases from SPEC
- Include:
  - different `mu`
  - different grid sizes
  - vacuum regions
  - more shaped QA/QH cases when available

Layer 4: integration-oriented checks

- Build a clean Python API matching how SPECTRE may want to call the solver
- Benchmark latency and throughput
- Compare direct solve behavior against alternative JAX linear solvers if needed

Layer 5: physics-level checks

- compare energy and helicity values
- verify plasma and vacuum branches
- add shape and size scaling checks
- verify robust failure behavior on ill-conditioned systems

## 12. Testing Plan

Required test categories:

- fixture loading
- operator assembly
- RHS assembly
- exact solve regression
- residual checks
- energy/helicity checks
- autodiff checks
- vectorized parameter scan checks
- vacuum-region checks
- example execution smoke tests
- packaging/import tests

Coverage target:

- at least 90 percent line coverage in CI

Current configured state:

- `pyproject.toml` already enforces `--cov-fail-under=90`

Still needed:

- broader fixture coverage beyond the current four dumped systems

## 13. Documentation Plan

The documentation should be written as if the repository is intended for external scientific users and future SPECTRE contributors.

Must include:

- project overview and motivation
- installation instructions
- quickstart
- detailed API reference
- explanation of the SPEC linear system
- equations and notation
- relation to Beltrami fields, Taylor relaxation, and MRxMHD
- inputs and outputs
- validation methodology
- limitations and non-goals
- examples with figures
- references and citations

Files to add:

- none for the initial docs scaffold

Files now present:

- `.readthedocs.yaml`
- `docs/conf.py`
- `docs/index.md`
- `docs/overview.md`
- `docs/theory.md`
- `docs/validation.md`
- `docs/examples.md`
- `docs/api.md`
- `docs/limitations.md`
- `docs/references.md`
- `docs/requirements.txt`

Documentation quality bar:

- complete enough that a new contributor can understand:
  - what the solver does
  - what it does not do
  - how it maps to SPEC
  - how it is validated
  - how to regenerate fixtures

## 14. Implementation Plan

The staged implementation plan is:

1. Finish the current linear regression workflow.
2. Add README, docs, and Read the Docs configuration. (done)
3. Add CI workflow with coverage reporting. (done)
4. Add more SPEC fixtures and broader validations. (partially done)
5. Expand the solver API for integration use.
6. Add richer solver diagnostics and conditioning tools.
7. Benchmark and, if justified, introduce alternative JAX linear algebra backends.
8. Only after the linear stage is rock solid, consider porting more of the assembly logic upstream of the solve.

Concrete near-term solver tasks:

- add a higher-level public solve function for integration
- add synthetic analytic test cases
- add more multi-fixture parametrized tests as new SPEC dumps are added
- consider how to expose diagnostics in a future SPECTRE-facing integration API without bloating the base solve path

## 15. Immediate Next Steps

The next concrete tasks, in priority order, are:

1. Enable the hosted Read the Docs project and verify that the public docs URL stops returning `404`.
2. Add at least one more shaped 3D plasma fixture from SPEC, ideally one closer to the future SPECTRE target cases.
3. Add a higher-level public solve function oriented toward downstream integration instead of only raw system objects.
4. Broaden the benchmark set beyond the current dense fixture panels, especially if larger systems are added.
5. Add synthetic analytic tests that do not rely on dumped SPEC data.
6. Keep `plan.md` and the repository in sync as new work lands.

## 16. Open Gaps and Risks

Open gaps:

- hosted docs are not yet live; `https://beltrami-jax.readthedocs.io/` returned `404` when checked on `2026-04-17`
- fixture coverage is still narrow relative to the full SPEC branch space
- no QA/QH or SPECTRE-like shaping cases are yet packaged

Technical risks:

- dense direct solve may be too limited for larger production-scale systems
- JAX linear solve behavior and performance may vary by backend
- a later SPECTRE interface may not exactly match SPEC's packing conventions
- the current internal geometry model is still simpler than full SPEC/SPECTRE branch coverage

Project risk:

- it is easy to confuse "we have a working linear-system regression" with "we have ported the solver fully"
- that would be incorrect

Current honest status:

- the linear solve kernel, diagnostics helpers, benchmark helpers, and validation figures are implemented and regression-tested against multiple dumped SPEC systems
- the repository now also includes internal geometry assembly, GMRES, an outer helicity-constrained solve, standalone workflow examples, docs sources, figures, benchmark assets, and CI definitions
- exact parity with all SPEC/SPECTRE branches is still future work

## 17. Restart From Scratch Checklist

If all local context were lost, the restart procedure should be:

1. Clone `beltrami_jax` and SPEC into `/Users/rogerio/local`.
2. Confirm `gh auth status`.
3. Install build dependencies:
   - `cmake`
   - `gcc/gfortran`
   - `open-mpi`
   - `hdf5`
   - `fftw`
   - `openblas`
4. Build SPEC with legacy Fortran flag:
   - ensure `-std=legacy` is used
5. Run the shipped SPEC example `G3V01L0Fi.001.sp`.
6. Add or reapply the temporary `SPEC_DUMP_LINEAR_SYSTEM` instrumentation if fixture extraction is needed.
   - include `SPEC_DUMP_LINEAR_SYSTEM_LVOL`
   - include vacuum metadata and `dMG` dumping
7. For matrix-free SPEC inputs, create dump-only copies that set `Lmatsolver = 1`.
8. Run the dump cases to export dense matrices and vectors.
9. Create the Python virtual environment in `beltrami_jax/.venv`.
10. Install `jax`, `pytest`, `pytest-cov`, `sphinx`, `myst-parser`, `furo`, `matplotlib`, `numpy`, and `build`.
11. Package the dumped fixtures into `src/beltrami_jax/data/`.
12. Add `src/beltrami_jax/data/__init__.py` so packaged resources are importable.
13. Run `pytest`.
14. Run `sphinx -W`.
15. Regenerate committed validation assets with `tools/generate_validation_assets.py`.
16. Fill in README, docs, and CI.
17. Commit and push.

## 18. Chronological Log

### 2026-04-16: repository and SPEC bring-up

Completed:

- created GitHub repository `rogeriojorge/beltrami_jax`
- cloned SPEC to `/Users/rogerio/local/SPEC`
- found and documented the relevant Beltrami-related SPEC source files
- installed `cmake`
- built SPEC successfully
- discovered that `-std=legacy` is required for the local `gfortran` runtime path
- ran a baseline SPEC example successfully
- ran a verbose diagnostic SPEC example successfully
- added a temporary local SPEC instrumentation hook to dump the dense linear system
- produced a reference dump for one volume of `G3V01L0Fi`
- created the initial `beltrami_jax` Python package skeleton
- wrote initial tests and examples around the future packaged fixture

Worked:

- shallow clone path through `gh`
- SPEC build after dependency setup
- dense linear system extraction
- initial JAX API design

Did not finish:

- packaging the fixture into the repo
- running the tests end to end
- README/docs/CI
- first commit and push

### 2026-04-16: restartability planning

Completed:

- audited current repository state
- wrote this `plan.md` restart document and project log

Decision:

- treat `plan.md` as the single source of truth for project state until the repository has proper docs and CI

### 2026-04-16: fixture packaging and local validation

Completed:

- added `src/beltrami_jax/data/__init__.py` so packaged data can be accessed through `importlib.resources`
- added a minimal `README.md` so editable packaging succeeds
- generated packaged fixture:
  - `/Users/rogerio/local/beltrami_jax/src/beltrami_jax/data/g3v01l0fi_lvol1.npz`
- installed the repository in editable mode with dev dependencies
- ran the full test suite locally
- increased the test suite to cover the text-dump loader and verbose solver path

Bug found and fixed:

- the first fixture-generation attempt failed because `load_spec_text_dump()` used `Path.with_suffix()`
- dump prefixes such as `G3V01L0Fi.dump.lvol1` lost the trailing `.lvol1` when file names were constructed
- fix:
  - construct dump file paths by string concatenation instead of `with_suffix()`

Validation result:

- `./.venv/bin/python -m pytest`
- outcome:
  - `10 passed in 13.05s`
  - `Total coverage: 100.00%`

### 2026-04-16: initial publish

Completed:

- staged the full repository
- created the initial commit on `main`
- commit:
  - `062bda41b97129872ddb9d9524aa17e6edda6449`
- pushed `main` to:
  - `https://github.com/rogeriojorge/beltrami_jax.git`

State after publish:

- repository is now present on GitHub with the initial working scaffold
- remaining work is primarily docs, CI, and broader SPEC validation coverage

### 2026-04-16: plan synchronization publish

Completed:

- updated `plan.md` after the initial publish so repository status no longer claimed the repo was uncommitted or unpushed
- pushed the synchronized plan update

Note:

- do not record a fixed "current HEAD" hash inside this file unless it is part of a dated log entry
- otherwise the file becomes stale immediately after the next commit

### 2026-04-17: docs, figures, and CI scaffold

Completed:

- expanded `README.md` from a minimal placeholder into a fuller project overview
- added a Sphinx documentation tree under `docs/`
- added `.readthedocs.yaml`
- added a GitHub Actions workflow under `.github/workflows/ci.yml`
- generated committed figures:
  - `docs/_static/spec_fixture_spectrum.png`
  - `docs/_static/parameter_scan.png`
- built the docs locally with strict warnings enabled
- reran the full test suite locally

Bug found and fixed:

- the first strict docs build failed because the API page let autodoc inspect imported JAX aliases such as `Array`
- fix:
  - exclude those imported symbols from the API reference page

Validation results:

- docs build:
  - `./.venv/bin/python -m sphinx -W -b html docs docs/_build/html`
  - outcome:
    - build succeeded
- tests:
  - `./.venv/bin/python -m pytest`
  - outcome:
    - `10 passed in 3.93s`
    - `Total coverage: 100.00%`

### 2026-04-17: first CI run and portability fix

Completed:

- pushed the first docs and CI scaffold
- observed the first GitHub Actions run start on GitHub
- inspected the failing test-job logs
- fixed the example smoke test to use `sys.executable` instead of a hardcoded local `.venv/bin/python`
- reran local tests and strict docs build after the fix

Bug found and fixed:

- the first GitHub Actions test jobs failed because `tests/test_examples.py` assumed the interpreter lived at `.venv/bin/python`
- that assumption is true locally but false on GitHub-hosted runners
- fix:
  - use the interpreter running pytest via `sys.executable`

Validation results after the fix:

- docs build:
  - `./.venv/bin/python -m sphinx -W -b html docs docs/_build/html`
  - outcome:
    - build succeeded
- tests:
  - `./.venv/bin/python -m pytest`
  - outcome:
    - `10 passed in 4.44s`
    - `Total coverage: 100.00%`

Remote CI result:

- GitHub Actions run `24568541530`
- outcome:
  - docs job passed
  - test job on Python 3.11 passed
  - test job on Python 3.13 passed

### 2026-04-17: multi-fixture SPEC regression expansion

Completed:

- patched the local SPEC dump hook in `/Users/rogerio/local/SPEC/src/mp00ac.f90`
  - added support for `SPEC_DUMP_LINEAR_SYSTEM_LVOL`
  - added `is_vacuum` metadata to `.meta.txt`
  - added `.dmg.txt` output so vacuum-region RHS reconstruction can be validated exactly
- rebuilt SPEC from:
  - `/Users/rogerio/local/SPEC/build`
- exported a new compact nonzero-`mu` plasma-region dump from:
  - input file: `/Users/rogerio/local/SPEC/ci/G1V03L0Fi/G1V03L0Fi.sp`
  - dump-only copy: `/Users/rogerio/local/SPEC_runs/G1V03L0Fi.dump.sp`
  - region: `lvol = 2`
  - matrix size: `51`
  - `mu = 1.0000000000000009e-01`
  - `psi = (3.1830988618379082e-02, 0.0)`
- exported a new free-boundary vacuum-region dump from:
  - input file: `/Users/rogerio/local/SPEC/ci/toroidal_freeboundary_vacuum/G3V02L0Fr_LU.sp`
  - region: `lvol = 3`
  - matrix size: `1548`
  - `mu = 0.0`
  - `psi = (2.0806176444792901e-01, 1.3201381096868081e-02)`
  - `is_vacuum = 1`
- packaged new fixtures into the repository:
  - `/Users/rogerio/local/beltrami_jax/src/beltrami_jax/data/g1v03l0fi_lvol2.npz`
  - `/Users/rogerio/local/beltrami_jax/src/beltrami_jax/data/g3v02l0fr_lu_lvol3.npz`
- updated `beltrami_jax.reference` and `tools/build_spec_fixture.py` to preserve vacuum metadata and optional `d_mg`
- expanded regression tests to cover all packaged fixtures and both legacy and new text-dump formats
- updated `README.md`, `docs/validation.md`, and `docs/limitations.md` to reflect the broader fixture set

Important failed attempts and decisions:

- `G1V03L2Fi.001.sp` looked attractive as a small nonzero-`mu` candidate, but it failed as shipped because its `LBeltrami=2` setup requires `Lconstraint=2`
- `G1V03L0Fi.sp` ran successfully but did not emit a dump until a dump-only copy forced `Lmatsolver = 1`
- for the free-boundary toroidal vacuum case, `lvol = 2` was not the vacuum region despite `Nvol = 2`
- the correct vacuum-system dump for that case is `lvol = 3`, because SPEC carries `Mvol = 3` there

Exact commands used:

- rebuild SPEC:
  - `cmake --build /Users/rogerio/local/SPEC/build -j4`
- dump compact plasma fixture:
  - `env SPEC_DUMP_LINEAR_SYSTEM=/Users/rogerio/local/SPEC_runs/G1V03L0Fi.dump.lvol2 SPEC_DUMP_LINEAR_SYSTEM_LVOL=2 OMP_NUM_THREADS=1 /Users/rogerio/local/SPEC/build/build/bin/xspec /Users/rogerio/local/SPEC_runs/G1V03L0Fi.dump.sp`
- dump free-boundary vacuum fixture:
  - `env SPEC_DUMP_LINEAR_SYSTEM=/Users/rogerio/local/SPEC_runs/G3V02L0Fr_LU.dump.lvol3 SPEC_DUMP_LINEAR_SYSTEM_LVOL=3 OMP_NUM_THREADS=1 /Users/rogerio/local/SPEC/build/build/bin/xspec /Users/rogerio/local/SPEC_runs/G3V02L0Fr_LU.sp`
- package plasma fixture:
  - `PYTHONPATH=src ./.venv/bin/python tools/build_spec_fixture.py /Users/rogerio/local/SPEC_runs/G1V03L0Fi.dump.lvol2 /Users/rogerio/local/beltrami_jax/src/beltrami_jax/data/g1v03l0fi_lvol2.npz`
- package vacuum fixture:
  - `PYTHONPATH=src ./.venv/bin/python tools/build_spec_fixture.py /Users/rogerio/local/SPEC_runs/G3V02L0Fr_LU.dump.lvol3 /Users/rogerio/local/beltrami_jax/src/beltrami_jax/data/g3v02l0fr_lu_lvol3.npz`

Validation results:

- tests:
  - `./.venv/bin/python -m pytest`
  - outcome:
    - `15 passed in 5.19s`
    - `Total coverage: 100.00%`
- docs build:
  - `./.venv/bin/python -m sphinx -b html -W docs docs/_build/html`
  - outcome:
    - build succeeded

### 2026-04-17: diagnostics, benchmarks, validation panels, and 3D fixture expansion

Completed:

- exported and packaged an additional fixed-boundary 3D plasma fixture from SPEC:
  - input file: `/Users/rogerio/local/SPEC/InputFiles/TestCases/G3V02L1Fi.001.sp`
  - dump-only copy: `/Users/rogerio/local/SPEC_runs/G3V02L1Fi.dump.sp`
  - region: `lvol = 1`
  - matrix size: `361`
  - `mu = 1.8189908612531447e-04`
  - `psi = (9.6560529129151959e-02, 0.0)`
- added new public modules:
  - `src/beltrami_jax/diagnostics.py`
  - `src/beltrami_jax/benchmark.py`
- expanded `src/beltrami_jax/types.py` with diagnostic and benchmark dataclasses
- updated examples to print richer diagnostics and added:
  - `examples/benchmark_fixtures.py`
- added docs-only installation extra in `pyproject.toml`
- switched `docs/requirements.txt` to install the package through `-e .[docs]`
- generated and committed polished validation assets:
  - `/Users/rogerio/local/beltrami_jax/docs/_static/validation_panel.png`
  - `/Users/rogerio/local/beltrami_jax/docs/_static/benchmark_panel.png`
- updated README and docs pages to include the new validation and benchmark evidence

Important design decisions:

- diagnostics are exposed through dedicated helper functions instead of being baked into the solve result, so the core solve path stays lightweight
- expensive conditioning work is opt-in via `compute_solve_diagnostics(..., include_condition_number=True)`
- benchmark panels report steady-state fixture timings directly, while one-time compile costs are summarized by unique matrix size because JAX compilation is cached by shape
- the repository documentation URL in `pyproject.toml` was changed away from Read the Docs because the expected hosted URL was not live

Important failures and blockers:

- `https://beltrami-jax.readthedocs.io/` returned `404` when checked with:
  - `curl -I -L --max-redirs 3 https://beltrami-jax.readthedocs.io/`
- this means the repository-side Read the Docs configuration is present, but the hosted project still needs to be enabled externally

Exact commands used:

- dump 3D plasma fixture:
  - `env SPEC_DUMP_LINEAR_SYSTEM=/Users/rogerio/local/SPEC_runs/G3V02L1Fi.dump.lvol1 SPEC_DUMP_LINEAR_SYSTEM_LVOL=1 OMP_NUM_THREADS=1 /Users/rogerio/local/SPEC/build/build/bin/xspec /Users/rogerio/local/SPEC_runs/G3V02L1Fi.dump.sp`
- package 3D plasma fixture:
  - `PYTHONPATH=src ./.venv/bin/python tools/build_spec_fixture.py /Users/rogerio/local/SPEC_runs/G3V02L1Fi.dump.lvol1 /Users/rogerio/local/beltrami_jax/src/beltrami_jax/data/g3v02l1fi_lvol1.npz`
- generate validation assets:
  - `PYTHONPATH=src ./.venv/bin/python tools/generate_validation_assets.py --repeats 2`
- verify hosted docs URL:
  - `curl -I -L --max-redirs 3 https://beltrami-jax.readthedocs.io/`

Validation results:

- tests:
  - `./.venv/bin/python -m pytest`
  - outcome:
    - `21 passed in 8.97s`
    - `Total coverage: 100.00%`
- docs build:
  - `./.venv/bin/python -m sphinx -b html -W docs docs/_build/html`
  - outcome:
    - build succeeded

Final local release-gate rerun before push:

- tests:
  - `./.venv/bin/python -m pytest`
  - outcome:
    - `21 passed in 20.72s`
    - `Total coverage: 100.00%`
- docs build:
  - `./.venv/bin/python -m sphinx -b html -W docs docs/_build/html`
  - outcome:
    - build succeeded

Publication status:

- committed to `main` as:
  - `7d4f84e` (`Add diagnostics, benchmarks, and validation panels`)
- pushed successfully to:
  - `https://github.com/rogeriojorge/beltrami_jax.git`
- GitHub Actions verification:
  - run id: `24571939916`
  - outcome:
    - `docs` passed
    - `test (3.11)` passed
    - `test (3.13)` passed

### 2026-04-18: internal geometry assembly, GMRES, nonlinear solve, standalone workflows, and dependency cleanup

Completed:

- expanded `src/beltrami_jax/types.py` with:
  - `FourierBeltramiGeometry`
  - `FourierModeBasis`
  - `GeometryAssemblyResult`
  - `BeltramiProblem`
  - `IterativeSolveResult`
  - `NonlinearSolveResult`
- added new public modules:
  - `src/beltrami_jax/geometry.py`
  - `src/beltrami_jax/iterative.py`
  - `src/beltrami_jax/nonlinear.py`
  - `src/beltrami_jax/io.py`
- updated `src/beltrami_jax/solver.py` to support both dense and GMRES solve paths
- rewrote the example scripts as standalone parameter-at-the-top workflows:
  - `examples/solve_spec_fixture.py`
  - `examples/parameter_scan.py`
  - `examples/autodiff_mu.py`
  - `examples/benchmark_fixtures.py`
- examples now write generated inputs, outputs, and figures under `examples/_generated/`
- removed version pins from the runtime and optional dependencies in `pyproject.toml`
- updated README and docs to describe the broader Beltrami workflow instead of only the dumped-linear-system path

Important design decisions:

- the internal assembly uses an axis-regularized Fourier large-aspect-ratio torus so the package has a fully internal Beltrami workflow without pretending to be exact full-SPEC geometry parity
- the nonlinear update is currently helicity-constrained and secant-based because that gives a compact, testable analogue of the outer Beltrami loop
- the standalone examples avoid `argparse`, `main()`, and helper-file indirection; all user-facing parameters are declared at the top of each script
- the default SPEC example was switched to the compact `g1v03l0fi_lvol2` fixture so CI smoke tests stay fast while still covering dense, GMRES, and matrix-free agreement

Validation results:

- tests:
  - `./.venv/bin/python -m pytest`
  - outcome:
    - `28 passed in 39.99s`
    - `Total coverage: 96.10%`
- docs build:
  - `./.venv/bin/python -m sphinx -b html -W docs docs/_build/html`
  - outcome:
    - build succeeded
- example smoke checks:
  - `./.venv/bin/python examples/solve_spec_fixture.py`
  - `./.venv/bin/python examples/parameter_scan.py`
  - `./.venv/bin/python examples/autodiff_mu.py`
  - `./.venv/bin/python examples/benchmark_fixtures.py`
  - outcome:
    - all completed and wrote outputs under `examples/_generated/`

### 2026-04-19: release hardening for shipping

Completed:

- relaxed build-system requirements in `pyproject.toml` to unpinned `setuptools` and `wheel`
- lowered `requires-python` to `>=3.10`
- added Python `3.10` classifier
- expanded CI to test Python `3.10`, `3.11`, `3.12`, and `3.13`
- added a distribution build job to CI with `python -m build`
- added README badges for CI, Python support, MIT license, and coverage
- added and linked a full integration guide at `docs/integration.md`
- expanded README and docs with:
  - magnetic-field and fusion context
  - more equations
  - explicit integration patterns for dumped SPEC systems, direct components, internal geometry assembly, and future SPECTRE use
  - refreshed validation and example figures
  - release-gate test, coverage, and benchmark summaries
- refreshed committed documentation figures from the current standalone examples:
  - `docs/_static/spec_fixture_spectrum.png`
  - `docs/_static/parameter_scan.png`
  - `docs/_static/autodiff_gradient_check.png`
  - `docs/_static/vacuum_gmres_panel.png`

Important design decisions:

- kept runtime compatibility simple by avoiding any `tomllib` dependency in package code
- used README badges that do not require an external hosted docs service, since Read the Docs is still not live
- added a package-build job to CI so shipping quality is validated at the source/wheel level, not only through editable installs

Validation results:

- tests:
  - `./.venv/bin/python -m pytest`
  - outcome:
    - `28 passed in 37.15s`
    - `Total coverage: 96.10%`
- docs build:
  - `./.venv/bin/python -m sphinx -b html -W docs docs/_build/html`
  - outcome:
    - build succeeded

### 2026-04-30: SPECTRE release assessment and migration plan

Trigger:

- Collaborator asked what the actual `beltrami_jax` input is, whether vector-potential coefficients were compared to SPEC/SPECTRE `.h5` output, and whether the repeated "large aspect-ratio" language means the implementation is restricted to large-aspect-ratio tokamaks.
- SPECTRE is now open source at `https://gitlab.com/spectre-eq/spectre`.

Assessment outcome:

- The collaborator's concerns are valid.
- Current `beltrami_jax` fixture mode is a developer validation path, not the final user-facing input path.
- Current internal geometry mode is a shaped large-aspect-ratio torus prototype, not SPECTRE's exact arbitrary 3D Fourier-interface geometry path.
- Current `beltrami_jax` validation compares dumped matrices, RHS vectors, and packed solutions from SPEC text dumps. It does not yet directly compare HDF5 `vector_potential/Ate`, `Aze`, `Ato`, and `Azo`.
- The docs and README need to be tightened so they do not imply full SPECTRE backend parity.

SPECTRE local bring-up:

- Cloned SPECTRE into `/Users/rogerio/local/spectre`.
- SPECTRE remote:
  - `https://gitlab.com/spectre-eq/spectre.git`
- Assessed SPECTRE commit:
  - `08e358a`
- Installed SPECTRE with:
  - `python3 -m venv .venv`
  - `HDF5_ROOT=/opt/homebrew/opt/hdf5-mpi FFTW_ROOT=/opt/homebrew/opt/fftw .venv/bin/python -m pip install -e .`
- Runtime required:
  - `DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib`
  - `OMP_NUM_THREADS=1`

Local SPECTRE patches required to run on this machine:

- `/Users/rogerio/local/spectre/spectre/file_io/input_parameters.py`
  - guarded an empty `@field_serializer(*arrays_of_length_nvol_minus_1)` call because Pydantic 2.13 rejects a field serializer with no fields
- `/Users/rogerio/local/spectre/fortran_src/beltrami_field_mod.F90`
  - rewrote a legacy `1040 format(...)` descriptor that gfortran 15 rejects at runtime

SPECTRE code path identified:

- Python `spectre.force_targets.force_real(...)` calls `test.field_mod.solve_field(...)`.
- Fortran `beltrami_field_mod.construct_beltrami_field(...)` dispatches local constraint cases and calls `solve_beltrami_system(...)` directly or through `hybrj2`.
- Fortran `beltrami_solver_mod.solve_beltrami_system(...)` builds `dMA - mu*dMD`, builds the RHS from `dMB`, `dMG`, and fluxes, solves with LAPACK, computes energy/helicity integrals, and unpacks the solution into `Ate/Aze/Ato/Azo`.
- Fortran `matrices_mod.matrix(...)` and `matrixBG(...)` assemble the Beltrami matrices and RHS operators.
- Fortran `wrapper_funcs_mod.get_vec_pot(...)` exposes flat `Ate/Aze/Ato/Azo` arrays.
- SPECTRE's `tests/compare/test_compare_to_spec.py` already compares force modes and HDF5 vector-potential datasets against old reference output.

Manual SPECTRE validation performed:

- Ran SPECTRE field calculation for `tests/wrapper/cyl_manyvol_test.toml`.
- Obtained:
  - `xinit_shape = (17,)`
  - `force_shape = (2, 512)`
  - vector-potential coefficient shapes `(17, 8)` for `Ate`, `Aze`, `Ato`, `Azo`
- Manually reproduced the SPECTRE comparison-test logic for four shipped reference cases.
- Vector-potential relative errors against `reference.h5`:
  - `G2V32L1Fi`: `Ate 1.27e-15`, `Aze 4.70e-15`, `Ato 0`, `Azo 0`
  - `G3V3L3Fi`: `Ate 5.68e-15`, `Aze 4.83e-14`, `Ato 0`, `Azo 0`
  - `G3V3L2Fi_stability`: `Ate 5.69e-15`, `Aze 5.07e-14`, `Ato 0`, `Azo 0`
  - `G3V8L3Free`: `Ate 2.78e-15`, `Aze 3.20e-15`, `Ato 0`, `Azo 0`
- Force-mode relative errors in the same manual checks were `1.1e-13` to `3.2e-12`.
- The full pytest comparison command still terminated without a Python traceback in this local environment, so the reliable evidence from this session is the manual equivalent comparison.

New planning document:

- Added `SPECTRE_MIGRATION_PLAN.md`.
- It contains:
  - direct answers to the collaborator's email
  - `beltrami_jax` code assessment
  - SPECTRE source-code map
  - SPECTRE build/run notes
  - exact validation results from this assessment
  - what must be replaced to remove Fortran Beltrami from SPECTRE
  - phased implementation roadmap
  - design decisions for Rogerio to choose before SPECTRE fork work starts

Immediate next implementation lanes:

- Fix README/docs to clarify current input modes, fixture role, HDF5 coefficient gap, and large-aspect-ratio wording.
- Add SPECTRE HDF5 vector-potential readers to `beltrami_jax`.
- Implement exact SPECTRE pack/unpack mapping for `Ate/Aze/Ato/Azo`.
- Add tests and plots comparing `beltrami_jax` vector-potential coefficients to SPECTRE reference HDF5 files.
- After design choices are confirmed, create a SPECTRE fork/branch with an optional JAX Beltrami backend.

## 19. Notes For Future Updates

When future work is done, update the following sections:

- "Current Status of the `beltrami_jax` Repository"
- "Immediate Next Steps"
- "Open Gaps and Risks"
- "Chronological Log"

If a new SPEC fixture is added, record:

- input file used
- SPEC build hash or date
- whether the local SPEC instrumentation patch changed
- region index
- matrix size
- `mu`
- `psi`
- exact command used to produce the dump

If solver behavior changes, record:

- the API change
- the physical or numerical reason
- the validation evidence against SPEC
