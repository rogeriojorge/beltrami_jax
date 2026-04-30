# beltrami_jax

[![CI](https://github.com/rogeriojorge/beltrami_jax/actions/workflows/ci.yml/badge.svg)](https://github.com/rogeriojorge/beltrami_jax/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/rogeriojorge/beltrami_jax/blob/main/LICENSE)
[![Coverage](https://img.shields.io/badge/coverage-95%25-brightgreen.svg)](https://github.com/rogeriojorge/beltrami_jax/actions/workflows/ci.yml)

`beltrami_jax` is a differentiable JAX implementation of the SPEC/SPECTRE-style Beltrami workflow used inside multi-region relaxed MHD calculations.

The repository currently covers two complementary paths:

- a SPEC-style assembled-system path, where already assembled `A`, `D`, `B`, optional `G`, fluxes, and reference solutions are loaded for validation and solving
- an internal geometry prototype, where a shaped large-aspect-ratio torus is assembled in JAX for examples, autodiff, and workflow development
- a SPECTRE-facing validation path, where SPECTRE TOML inputs and HDF5 vector-potential coefficients are loaded, normalized, compared, and plotted

The first path is the current scientifically relevant validation path. The second path is useful for development, but it is not yet SPECTRE's full arbitrary 3D Fourier-interface geometry assembly.

This is not yet a full SPECTRE Beltrami backend. SPECTRE TOML input summaries and HDF5 vector-potential coefficient validation are implemented, but exact JAX-native SPECTRE interface-geometry assembly, SPECTRE pack/unpack from a JAX solution vector, and full local-constraint branch parity remain the core replacement work documented in [SPECTRE_MIGRATION_PLAN.md](/Users/rogerio/local/beltrami_jax/SPECTRE_MIGRATION_PLAN.md).

The repository ships under the MIT License; see [LICENSE](/Users/rogerio/local/beltrami_jax/LICENSE).

## Motivation

The goal is to replace a legacy Fortran Beltrami-solver interface with a Python/JAX implementation that is:

- easy to install with `pip`
- easy to inspect and validate
- differentiable end to end
- compatible with JAX transformations such as `jit`, `grad`, and `vmap`
- realistic to integrate into future SPECTRE workflows

In the SPEC/SPECTRE formulation, each relaxed plasma region solves a discrete Beltrami problem of the form

```math
\nabla \times \mathbf{B} = \mu \mathbf{B},
```

which, after geometry-dependent assembly, becomes a linear system

```math
\mathbf{M}\mathbf{a} = \mathbf{r},
\qquad
\mathbf{M} = \mathbf{A} - \mu \mathbf{D}.
```

`beltrami_jax` currently supports both:

- direct regression against dumped SPEC linear systems
- internally assembled prototype geometry solves that go from a shaped-torus model to nonlinear `mu` update, output files, and postprocessing

In terms of magnetic-field physics, the code works with a discretized vector potential whose curl reconstructs the magnetic field. In Taylor-relaxed plasma regions the force-free relation

```math
\nabla \times \mathbf{B} = \mu \mathbf{B}
```

is the Euler-Lagrange condition that arises when magnetic energy is minimized at fixed helicity and fixed fluxes. That is why this solver matters for stellarators, tokamaks, and MRxMHD workflows: it is the kernel that turns geometry and constraints into a relaxed magnetic state.

## Implemented scope

Today the repository includes:

- a typed `BeltramiLinearSystem` container
- internal prototype Fourier geometry assembly via `assemble_fourier_beltrami_system`
- operator assembly for plasma and vacuum branches
- dense and GMRES solve paths with residual reporting
- matrix-free GMRES through `gmres_solve`
- a high-level `BeltramiProblem` input model with JSON save/load helpers
- an outer helicity-constrained nonlinear solve via `solve_helicity_constrained_equilibrium`
- coordinate-singularity-safe basis construction via `build_fourier_mode_basis`
- diagnostic helpers for conditioning, symmetry, and solution amplification
- benchmark helpers for steady-state solves and batched parameter scans
- vectorized parameter scans in `mu`
- autodifferentiation through the solved state
- packaged SPEC regression fixtures covering cylindrical, toroidal, 3D, and vacuum branches
- SPECTRE TOML input-summary loading for geometry, resolution, flux, constraint, and Fourier-boundary metadata
- SPECTRE HDF5 vector-potential readers for `Ate`, `Aze`, `Ato`, and `Azo`
- coefficient-level SPECTRE vector-potential comparison and plotting tools
- packed SPECTRE Beltrami layout helpers that split coefficients by volume and free-boundary exterior block
- packaged public SPECTRE compare cases for reproducible CI validation without a local SPECTRE checkout
- standalone example workflows that define geometries, write input files, run solves, save outputs, and generate figures
- tests that cover dumped SPEC systems and the internal geometry-driven workflow

## Input Modes and Current Parity Status

The current production-style solve input is an assembled Beltrami system: `d_ma`, `d_md`, `d_mb`, optional `d_mg`, `mu`, and a two-component flux vector `psi`. Packaged fixtures are examples of that input and are primarily developer validation assets, not something ordinary users should need to create by hand.

The current geometry-driven input is `FourierBeltramiGeometry`. It is intentionally limited to a shaped large-aspect-ratio torus prototype. It is not yet equivalent to SPECTRE's input model, where the user provides interface Fourier geometry, resolution, flux/current/helicity constraints, and branch flags.

The SPECTRE-facing input layer now reads SPECTRE TOML files into `SpectreInputSummary`, including `nvol`, `nfp`, `mpol`, `ntor`, `lrad`, flux arrays, constraint metadata, free-boundary settings, and normalized Fourier boundary tables. The HDF5 validation layer reads and compares `vector_potential/Ate`, `Aze`, `Ato`, and `Azo`.

Validation today has two levels:

- `beltrami_jax` dense solves reproduce dumped SPEC matrices, RHS vectors, and packed solutions at machine precision for the committed fixtures.
- Fresh SPECTRE exports reproduce SPECTRE `reference.h5` vector-potential coefficients with worst global relative coefficient error `1.52e-14` across four public SPECTRE compare cases.
- Those four SPECTRE compare cases are packaged under `beltrami_jax.data.spectre_compare` so CI and downstream users can reproduce the coefficient target without installing SPECTRE.

The remaining SPECTRE replacement milestone is stronger: make the JAX-native assembly and solve path produce those same `Ate`, `Aze`, `Ato`, and `Azo` coefficients directly from SPECTRE TOML/interface geometry.

## Installation

Clone the repository:

```bash
git clone https://github.com/rogeriojorge/beltrami_jax.git
cd beltrami_jax
```

Create a virtual environment and install in editable mode:

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -U pip
./.venv/bin/python -m pip install -e '.[dev]'
```

Runtime dependencies are intentionally minimal. The core package depends only on `jax`.

## Quick start

Run the test suite:

```bash
./.venv/bin/python -m pytest
```

Run the SPEC regression example:

```bash
./.venv/bin/python examples/solve_spec_fixture.py
```

Run the geometry-defined Beltrami workflow and postprocess a parameter scan:

```bash
./.venv/bin/python examples/parameter_scan.py
```

Run the geometry-defined autodiff example:

```bash
./.venv/bin/python examples/autodiff_mu.py
```

Run the vacuum/GMRES benchmark and export example:

```bash
./.venv/bin/python examples/benchmark_fixtures.py
```

Run the SPECTRE TOML/HDF5 vector-potential validation example:

```bash
./.venv/bin/python examples/validate_spectre_vector_potential.py
```

The example scripts are intentionally standalone. Each script keeps its input parameters at the top, writes files under `examples/_generated/`, prints progress to the terminal, and generates at least one figure or exported data product.

## Latest Release Checks

Latest local release gate:

- `44 passed in 26.11s`
- `95.36%` total line coverage
- strict Sphinx build passed with `-W`
- runtime code does not depend on `tomllib`, so Python `3.10+` support is not blocked by stdlib TOML parsing differences

Latest remote CI verification:

- Python `3.10`, `3.11`, `3.12`, and `3.13` test jobs pass
- docs build passes
- source and wheel builds pass

The current CI workflow is defined in [.github/workflows/ci.yml](/Users/rogerio/local/beltrami_jax/.github/workflows/ci.yml).

## Validation figures

Reviewer-facing validation summary generated from the committed SPEC fixtures:

![Validation panel](docs/_static/validation_panel.png)

Benchmark summary generated from the same packaged fixture set:

![Benchmark panel](docs/_static/benchmark_panel.png)

SPECTRE HDF5 vector-potential coefficient parity target generated from public SPECTRE compare cases:

![SPECTRE vector-potential parity](docs/_static/spectre_vecpot_parity.png)

Standalone workflow outputs generated from the current example scripts:

![SPEC fixture workflow](docs/_static/spec_fixture_spectrum.png)

![Geometry parameter scan](docs/_static/parameter_scan.png)

![Autodiff gradient check](docs/_static/autodiff_gradient_check.png)

![Vacuum GMRES workflow](docs/_static/vacuum_gmres_panel.png)

Current quantitative highlights from the committed validation and benchmark runs:

- SPEC regression error stays at or below roughly `1e-15` across the packaged fixtures
- packaged fixture condition numbers span roughly `2.6e4` to `5.4e7`
- the compact SPEC GMRES example converges in `13` iterations with solution-relative error around `3.5e-16`
- the internal vacuum GMRES example converges with relative residual around `4.6e-11`
- the current validation asset generator measures per-system batched scan costs down to about `5.9e-4` seconds on the local release machine for the compact fixture set
- SPECTRE HDF5 vector-potential parity reaches worst global relative coefficient error `1.52e-14` across `G2V32L1Fi`, `G3V3L3Fi`, `G3V3L2Fi_stability`, and `G3V8L3Free`

## Repository layout

- `src/beltrami_jax/`
  - core package
- `src/beltrami_jax/data/`
  - packaged SPEC regression fixtures
- `tests/`
  - regression, autodiff, vacuum-path, and smoke tests
- `examples/`
  - runnable examples that print progress to the terminal
- `tools/`
  - developer tools such as SPEC dump packaging
- `docs/`
  - Sphinx documentation for Read the Docs
- `.github/workflows/ci.yml`
  - CI/CD workflow for tests, docs, and package builds
- `LICENSE`
  - MIT license text
- `plan.md`
  - full project context, setup, restart log, and running implementation plan

## Validation against SPEC

The current regression workflow uses dense matrices and vectors dumped from a local SPEC build. The packaged references currently include:

- `g3v01l0fi_lvol1`: toroidal fixed-boundary plasma region, size `361`
- `g1v03l0fi_lvol2`: compact cylindrical fixed-boundary plasma region with nonzero `mu`, size `51`
- `g3v02l1fi_lvol1`: 3D fixed-boundary plasma region, size `361`
- `g3v02l0fr_lu_lvol3`: toroidal free-boundary vacuum region with nonzero `d_mg`, size `1548`

Together these fixtures verify:

- operator reconstruction
- right-hand-side reconstruction
- exact dense solution agreement
- residual norms
- conditioning and symmetry diagnostics on real operators
- differentiability with respect to `mu`
- vectorized batch solves
- end-to-end timing behavior across fixture sizes

The current committed validation set reaches solution-relative agreement at or below roughly `1e-15` across the packaged SPEC fixtures, while the measured operator 2-norm condition numbers span approximately `2.6e4` to `5.4e7`.

## Internal Beltrami workflow

In addition to SPEC-regression fixtures, the package now exposes an internal geometry-driven workflow:

- define a `FourierBeltramiGeometry`
- build a packed basis with `build_fourier_mode_basis`
- assemble the SPEC-style matrices with `assemble_fourier_beltrami_system`
- package the setup into a `BeltramiProblem`
- save or reload the problem with `save_problem_json` / `load_problem_json`
- solve the outer helicity-constrained problem with `solve_helicity_constrained_equilibrium`
- export outputs with `save_nonlinear_solution`
- postprocess with diagnostics, parameter scans, autodiff, and custom figures

That is the path used by the standalone examples. It should be read as a prototype workflow and teaching interface, not as the final SPECTRE backend API. The final SPECTRE path must start from SPECTRE's interface Fourier geometry and reproduce SPECTRE's `Ate`, `Aze`, `Ato`, and `Azo` coefficients.

## Using `beltrami_jax` From Other Codes

The package is designed so different upstream codes can meet it at different levels.

### From SPEC dumps

If an upstream code can export SPEC-style matrices, load and solve them directly:

```python
from beltrami_jax import load_spec_text_dump, solve_from_components

reference = load_spec_text_dump("/path/to/run.dump.lvol1")
result = solve_from_components(reference.system, method="dense", verbose=True)
```

### From a Python, C++, or Fortran code that already knows the geometry

If the upstream code can pass geometry parameters and fluxes instead of a dumped matrix, use the internal assembly path:

```python
from beltrami_jax import (
    BeltramiProblem,
    FourierBeltramiGeometry,
    build_fourier_mode_basis,
    solve_helicity_constrained_equilibrium,
)

geometry = FourierBeltramiGeometry(major_radius=3.0, minor_radius=1.0, elongation=1.2)
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

### From SPECTRE or another nonlinear equilibrium code

The current SPECTRE-facing utilities can already inspect SPECTRE TOML inputs and validate HDF5 vector-potential coefficients:

```python
from beltrami_jax import (
    build_spectre_beltrami_layout_for_vector_potential,
    compare_vector_potentials,
    load_packaged_spectre_case,
    load_spectre_input_toml,
    load_spectre_reference_h5,
    load_spectre_vector_potential_npz,
)

summary = load_spectre_input_toml("/path/to/spectre/input.toml")
reference = load_spectre_reference_h5("/path/to/spectre/reference.h5").vector_potential
candidate = load_spectre_vector_potential_npz("/path/to/fresh_spectre_export.npz")
comparison = compare_vector_potentials(candidate, reference, label="case")
packaged_case = load_packaged_spectre_case("G3V3L3Fi")
layout = build_spectre_beltrami_layout_for_vector_potential(summary, reference)

print(summary.lrad)
print(comparison.global_relative_error)
print(packaged_case.comparison.global_relative_error)
print(layout.blocks[0].radial_slice)
```

The intended JAX-native replacement contract is:

1. provide SPECTRE interface Fourier geometry, basis metadata, fluxes, and target constraints
2. assemble SPECTRE-equivalent `A`, `D`, `B`, and optional `G` operators in JAX
3. solve with the same branch logic as SPECTRE's Beltrami path
4. return packed coefficients and SPECTRE-compatible `Ate`, `Aze`, `Ato`, and `Azo`
5. consume the returned energies, helicities, residuals, and histories

See the full integration notes in [docs/integration.md](/Users/rogerio/local/beltrami_jax/docs/integration.md).
See the current SPECTRE replacement roadmap in [SPECTRE_MIGRATION_PLAN.md](/Users/rogerio/local/beltrami_jax/SPECTRE_MIGRATION_PLAN.md).

## Generating new fixtures from SPEC

If you have a SPEC dump prefix such as:

```text
/path/to/G3V01L0Fi.dump.lvol1
```

you can package it into a compressed fixture with:

```bash
PYTHONPATH=src ./.venv/bin/python tools/build_spec_fixture.py \
  /path/to/G3V01L0Fi.dump.lvol1 \
  src/beltrami_jax/data/new_fixture_name.npz
```

The corresponding SPEC dump is expected to provide:

- `.meta.txt`
- `.dma.txt`
- `.dmd.txt`
- `.dmb.txt`
- `.dmg.txt` for vacuum-region exports
- `.matrix.txt`
- `.rhs.txt`
- `.solution.txt`

To regenerate the committed validation panels from the packaged fixtures:

```bash
PYTHONPATH=src ./.venv/bin/python tools/generate_validation_assets.py --repeats 2
```

To regenerate the SPECTRE vector-potential parity panel, first export fresh SPECTRE coefficients from the SPECTRE environment:

```bash
OMP_NUM_THREADS=1 DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib \
  /Users/rogerio/local/spectre/.venv/bin/python tools/export_spectre_vecpot_npz.py \
  /Users/rogerio/local/spectre/tests/compare/G2V32L1Fi/input.toml \
  examples/_generated/spectre_vecpot_exports/G2V32L1Fi.npz
```

Then generate the figure from the `beltrami_jax` environment:

```bash
PYTHONPATH=src ./.venv/bin/python tools/generate_spectre_validation_assets.py --use-packaged
```

Omit `--use-packaged` to compare against a local SPECTRE checkout and fresh local exports instead.

## Documentation

Documentation sources live under `docs/` and are intended to build both locally and on Read the Docs.

Local HTML build:

```bash
./.venv/bin/python -m sphinx -b html docs docs/_build/html
```

Docs-only installation:

```bash
./.venv/bin/python -m pip install -e '.[docs]'
```

Planned documentation coverage includes:

- theory and equations
- mapping from SPEC matrices to the JAX API
- integration with external codes
- validation workflow
- examples
- API reference
- limitations and future work

Read the Docs note:

- the repository contains a committed `.readthedocs.yaml`
- the expected hosted project URL `https://beltrami-jax.readthedocs.io/` returned `404` when checked on April 17, 2026
- until that project is enabled, the repository docs tree is the reliable documentation entry point

## References

- SPEC docs: [https://princetonuniversity.github.io/SPEC/](https://princetonuniversity.github.io/SPEC/)
- SPEC manual: [https://princetonuniversity.github.io/SPEC/SPEC_manual.pdf](https://princetonuniversity.github.io/SPEC/SPEC_manual.pdf)
- Taylor 1974: [https://doi.org/10.1103/PhysRevLett.33.1139](https://doi.org/10.1103/PhysRevLett.33.1139)
- Taylor 1986: [https://doi.org/10.1103/RevModPhys.58.741](https://doi.org/10.1103/RevModPhys.58.741)
- Hudson et al. 2012 MRxMHD/SPEC: [https://arxiv.org/abs/1211.3072](https://arxiv.org/abs/1211.3072)
- Dennis et al. 2012 infinite-interface MRxMHD: [https://arxiv.org/abs/1212.4917](https://arxiv.org/abs/1212.4917)
- O'Neill and Cerfon 2018 Taylor-state solver: [https://arxiv.org/abs/1611.01420](https://arxiv.org/abs/1611.01420)

## Project status

This is an active build-out repository. The dense regression-tested kernel, diagnostics, benchmark tooling, and validation figures are in place, while the next steps are broader fixture coverage, richer integration APIs, and future alignment with public SPECTRE source.
