# beltrami_jax

`beltrami_jax` is a differentiable JAX implementation of the SPEC/SPECTRE-style linear Beltrami solve used inside multi-region relaxed MHD workflows.

The current repository scope is intentionally narrow and explicit:

- reproduce the dense linear algebra at the core of SPEC's Beltrami stage
- expose it through a compact JAX API
- validate it against real dumped SPEC systems
- make it easy to differentiate, vectorize, document, and test

This is not yet a full port of SPEC, and it is not yet a full port of SPECTRE. It is the linear Beltrami solve kernel plus the tooling around it.

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

`beltrami_jax` focuses on this linear stage.

## Implemented scope

Today the repository includes:

- a typed `BeltramiLinearSystem` container
- operator assembly for plasma and vacuum branches
- a dense JAX solve path with residual reporting
- vectorized parameter scans in `mu`
- autodifferentiation through the solved state
- packaged SPEC regression fixtures covering plasma and vacuum branches
- tests that compare the JAX solve to the dumped SPEC system

The current implementation intentionally does not yet include:

- geometry/integral assembly corresponding to `ma00aa.f90`
- sparse or matrix-free Krylov solvers
- the outer nonlinear constraint loop corresponding to `ma02aa.f90`
- coordinate-singularity-specific branches beyond the current linear operator interface

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

Solve the packaged SPEC fixture:

```bash
./.venv/bin/python examples/solve_spec_fixture.py
```

Scan over `mu` values:

```bash
./.venv/bin/python examples/parameter_scan.py
```

Differentiate magnetic energy with respect to `mu`:

```bash
./.venv/bin/python examples/autodiff_mu.py
```

## Example figures

Coefficient spectrum for the packaged SPEC fixture:

![SPEC fixture coefficient spectrum](docs/_static/spec_fixture_spectrum.png)

Vectorized energy scan around the reference `mu`:

![Vectorized mu scan](docs/_static/parameter_scan.png)

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
- `plan.md`
  - full project context, setup, restart log, and running implementation plan

## Validation against SPEC

The current regression workflow uses dense matrices and vectors dumped from a local SPEC build. The packaged references currently include:

- `g3v01l0fi_lvol1`: toroidal fixed-boundary plasma region, size `361`
- `g1v03l0fi_lvol2`: compact cylindrical fixed-boundary plasma region with nonzero `mu`, size `51`
- `g3v02l0fr_lu_lvol3`: toroidal free-boundary vacuum region with nonzero `d_mg`, size `1548`

Together these fixtures verify:

- operator reconstruction
- right-hand-side reconstruction
- exact dense solution agreement
- residual norms
- differentiability with respect to `mu`
- vectorized batch solves

At the time of the current scaffold, the local test suite passes with 100% line coverage on the implemented code.

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

## Documentation

Documentation sources live under `docs/` and are intended to build both locally and on Read the Docs.

Local HTML build:

```bash
./.venv/bin/python -m sphinx -b html docs docs/_build/html
```

Planned documentation coverage includes:

- theory and equations
- mapping from SPEC matrices to the JAX API
- validation workflow
- examples
- API reference
- limitations and future work

## References

- SPEC docs: [https://princetonuniversity.github.io/SPEC/](https://princetonuniversity.github.io/SPEC/)
- SPEC manual: [https://princetonuniversity.github.io/SPEC/SPEC_manual.pdf](https://princetonuniversity.github.io/SPEC/SPEC_manual.pdf)
- Taylor 1974: [https://doi.org/10.1103/PhysRevLett.33.1139](https://doi.org/10.1103/PhysRevLett.33.1139)
- Taylor 1986: [https://doi.org/10.1103/RevModPhys.58.741](https://doi.org/10.1103/RevModPhys.58.741)
- Hudson et al. 2012 MRxMHD/SPEC: [https://arxiv.org/abs/1211.3072](https://arxiv.org/abs/1211.3072)
- Dennis et al. 2012 infinite-interface MRxMHD: [https://arxiv.org/abs/1212.4917](https://arxiv.org/abs/1212.4917)
- O'Neill and Cerfon 2018 Taylor-state solver: [https://arxiv.org/abs/1611.01420](https://arxiv.org/abs/1611.01420)

## Project status

This is an active build-out repository. The dense regression-tested kernel is in place and published, while the next steps are broader fixture coverage, richer docs, CI, and a cleaner integration-oriented API for future SPECTRE work.
