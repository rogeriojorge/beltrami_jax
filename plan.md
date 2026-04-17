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
- SPEC build directory: `/Users/rogerio/local/SPEC/build/build`
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

Environment:

- Current working shell used during development: `zsh`
- Current date when this file was written: `2026-04-16`
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
- matrix-free iterative solvers
- full geometry/integral assembly corresponding to `ma00aa.f90`
- outer nonlinear constraint solve corresponding to `ma02aa.f90`

## 8. Current Status of the `beltrami_jax` Repository

What exists:

- package skeleton
- typed linear-system API
- dense JAX solve
- helper operators
- SPEC text-dump loader
- packaged SPEC regression fixture under `src/beltrami_jax/data/`
- package data module `src/beltrami_jax/data/__init__.py`
- minimal `README.md`
- examples
- tests passing locally against the packaged SPEC fixture
- editable install validated locally

What does not exist yet:

- documentation tree under `docs/`
- `.readthedocs.yaml`
- GitHub Actions CI workflow
- local docs build configuration

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
- `pyproject.toml`
  - package metadata, dependencies, pytest and coverage settings
- `plan.md`
  - this restartable project log and plan

Source package:

- `src/beltrami_jax/__init__.py`
  - public API export list and version
- `src/beltrami_jax/types.py`
  - dataclasses for linear-system inputs and solve results
- `src/beltrami_jax/operators.py`
  - operator and RHS assembly plus functionals
- `src/beltrami_jax/solver.py`
  - JAX dense solve and vectorized parameter scan
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

Tests:

- `tests/test_reference.py`
  - operator and RHS reconstruction from fixture
- `tests/test_solver.py`
  - regression against SPEC solution, residuals, autodiff, batched solves, vacuum path
- `tests/test_examples.py`
  - smoke-test example scripts using the repository virtual environment

Tools:

- `tools/build_spec_fixture.py`
  - convert dumped SPEC text files into packaged compressed fixture data

## 10. What Worked, What Did Not, and Why

What worked:

- GitHub authentication and repository creation
- SPEC clone through `gh repo clone -- --depth 1`
- SPEC build with CMake after installing `cmake`
- SPEC runtime after adding `-std=legacy`
- baseline SPEC run
- verbose SPEC run
- dumping a dense reference linear system from SPEC
- initial JAX package scaffolding
- packaging the dumped SPEC system into a repository fixture
- editable install with `pip install -e '.[dev]'`
- local test execution at 100 percent coverage

What did not work:

- plain initial SPEC clone path failed with packfile/index-pack problems
- initial SPEC runtime without legacy Fortran mode hit format-related issues
- first fixture-generation attempt failed because `reference.py` used `Path.with_suffix`, which stripped the `.lvol1` portion from dump prefixes
- repository metadata is still incomplete because docs/CI do not exist yet

Why this matters:

- the current code is a strong draft but not yet a ship-ready repository
- the next work should focus on finishing the engineering loop, not just adding more solver code

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

- real CI workflow
- broader fixture coverage beyond the first dumped system

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
2. Add README, docs, and Read the Docs configuration.
3. Add CI workflow with coverage reporting.
4. Add more SPEC fixtures and broader validations.
5. Expand the solver API for integration use.
6. Add richer solver diagnostics and conditioning tools.
7. Benchmark and, if justified, introduce alternative JAX linear algebra backends.
8. Only after the linear stage is rock solid, consider porting more of the assembly logic upstream of the solve.

Concrete near-term solver tasks:

- add richer residual/conditioning diagnostics
- add a higher-level public solve function for integration
- add synthetic analytic test cases
- add multiple-fixture parametrized tests

## 15. Immediate Next Steps

The next concrete tasks, in priority order, are:

1. Create docs and `.readthedocs.yaml`.
2. Create `.github/workflows/ci.yml`.
3. Generate example figures into `docs/_static/`.
4. Build docs locally with Sphinx.
5. Export at least one additional SPEC fixture:
   - different `mu`
   - preferably one vacuum case
6. Add parametrized multi-fixture tests.
7. Add richer solver diagnostics and conditioning checks.
8. Keep `plan.md` and the repository in sync as new work lands.

## 16. Open Gaps and Risks

Open gaps:

- no docs
- no CI
- no published validation summary in the repository
- only one dumped reference system is currently packaged

Technical risks:

- dense direct solve may be too limited for larger production-scale systems
- JAX linear solve behavior and performance may vary by backend
- a later SPECTRE interface may not exactly match SPEC's packing conventions
- current implementation does not yet cover coordinate-singularity-specific branches
- current implementation does not yet cover the outer nonlinear constraint solve

Project risk:

- it is easy to confuse "we have a working linear-system regression" with "we have ported the solver fully"
- that would be incorrect

Current honest status:

- the linear solve kernel is implemented and regression-tested against one dumped SPEC system
- the repository is not yet complete enough to call finished

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
7. Run the dump case to export the dense matrices and vectors.
8. Create the Python virtual environment in `beltrami_jax/.venv`.
9. Install `jax`, `pytest`, `pytest-cov`, `sphinx`, `myst-parser`, `furo`, `matplotlib`, `numpy`, and `build`.
10. Package the dumped fixture into `src/beltrami_jax/data/`.
11. Add `src/beltrami_jax/data/__init__.py` so packaged resources are importable.
12. Run `pytest`.
13. Fill in README, docs, and CI.
14. Commit and push.

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
