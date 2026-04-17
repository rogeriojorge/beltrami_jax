# Validation

## Validation philosophy

The package is validated against real dense systems exported from SPEC rather than only against synthetic toy matrices. This keeps the implementation anchored to the actual Fortran interface that motivated the JAX port.

## Current reference fixture

The first committed fixture comes from a local SPEC run of a `G3V01L0Fi` case. The exported dense system corresponds to:

- volume index `lvol = 1`
- matrix dimension `361`
- `mu = 0.0`
- `psi_t = 3.1830988618379069e-01`
- `psi_p = 0.0`

The committed compressed fixture lives at:

- `src/beltrami_jax/data/g3v01l0fi_lvol1.npz`

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

### Vectorization

- a batched `mu` scan reproduces the scalar solution at the reference `mu`

### Vacuum branch

- the vacuum right-hand-side path including `d_mg` behaves as expected on a synthetic system

### Example smoke tests

- all example scripts execute successfully and print progress messages

## Coverage target

The repository enforces a coverage threshold in `pyproject.toml`:

- required line coverage: at least 90%

At the current stage of the scaffold, the local test suite reaches 100% line coverage on the implemented package code.

## Known validation gaps

The current validation is strong for the implemented dense linear stage, but still incomplete in project terms.

Remaining validation work includes:

- more than one dumped SPEC fixture
- at least one vacuum-region fixture from SPEC
- different matrix sizes and nonzero `mu`
- performance benchmarks
- comparisons against later SPECTRE integration points

## Why exact dense regression matters

The most important current risk is accidentally drifting away from the exact discrete system that SPEC solves. Exact dense regression is therefore the right first milestone. Once that is stable, broader performance-oriented changes can be judged against a known-correct baseline.
