# beltrami_jax

`beltrami_jax` is a differentiable JAX implementation of the SPEC/SPECTRE-style linear Beltrami solve used inside multi-region relaxed MHD workflows.

Current status:

- initial dense JAX linear solver implemented
- SPEC text-dump loader implemented
- regression tests and examples drafted
- detailed restartable project log maintained in `plan.md`
- docs and CI still in progress

## Why this repository exists

The goal is to replace the legacy Fortran Beltrami solve interface used by SPEC/SPECTRE-style workflows with a Python/JAX implementation that is:

- easy to install
- easy to differentiate
- easy to benchmark and validate
- realistic to integrate into future SPECTRE workflows

This repository is currently focused on the linear solve stage, not the entire SPEC code base.

## Current repository layout

- `src/beltrami_jax/`
  - package source
- `tests/`
  - regression and smoke tests
- `examples/`
  - example scripts
- `tools/`
  - developer utilities for SPEC fixture generation
- `plan.md`
  - complete project context, status, setup, and restart log

## Quick start

Create a virtual environment and install in editable mode:

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -U pip
./.venv/bin/python -m pip install -e .[dev]
```

Run the tests:

```bash
./.venv/bin/python -m pytest
```

Run an example:

```bash
./.venv/bin/python examples/solve_spec_fixture.py
```

## SPEC validation workflow

The repository is being validated against dense linear systems dumped from a local SPEC build. The current reference dump comes from a `G3V01L0Fi` case and is used to verify:

- operator assembly
- right-hand-side assembly
- exact dense solution agreement
- residual behavior
- differentiability with respect to `mu`

## References

- SPEC docs: [https://princetonuniversity.github.io/SPEC/](https://princetonuniversity.github.io/SPEC/)
- SPEC manual: [https://princetonuniversity.github.io/SPEC/SPEC_manual.pdf](https://princetonuniversity.github.io/SPEC/SPEC_manual.pdf)
- Taylor 1974: [https://doi.org/10.1103/PhysRevLett.33.1139](https://doi.org/10.1103/PhysRevLett.33.1139)
- Taylor 1986: [https://doi.org/10.1103/RevModPhys.58.741](https://doi.org/10.1103/RevModPhys.58.741)
- Hudson et al. 2012 MRxMHD/SPEC: [https://arxiv.org/abs/1211.3072](https://arxiv.org/abs/1211.3072)

## Project status note

This is still an active build-out repository. The dense regression-tested kernel is in place, but the full documentation site, CI configuration, and broader SPEC fixture coverage are still being added.
