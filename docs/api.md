# API Reference

This page documents the currently exported public API.

## Public exports

```{eval-rst}
.. automodule:: beltrami_jax
   :members:
   :show-inheritance:
```

## Core types

```{eval-rst}
.. automodule:: beltrami_jax.types
   :members:
   :show-inheritance:
   :exclude-members: Array, ArrayLike
```

## Operator helpers

```{eval-rst}
.. automodule:: beltrami_jax.operators
   :members:
   :exclude-members: Array
```

## Solver functions

```{eval-rst}
.. automodule:: beltrami_jax.solver
   :members:
   :exclude-members: Array, ArrayLike
```

## Geometry assembly

```{eval-rst}
.. automodule:: beltrami_jax.geometry
   :members:
   :exclude-members: Array
```

## Krylov solve

```{eval-rst}
.. automodule:: beltrami_jax.iterative
   :members:
   :exclude-members: Array
```

## Nonlinear workflow

```{eval-rst}
.. automodule:: beltrami_jax.nonlinear
   :members:
```

## Input and output helpers

```{eval-rst}
.. automodule:: beltrami_jax.io
   :members:
```

## Diagnostics helpers

```{eval-rst}
.. automodule:: beltrami_jax.diagnostics
   :members:
```

## Benchmark helpers

```{eval-rst}
.. automodule:: beltrami_jax.benchmark
   :members:
```

## Fixture loading

```{eval-rst}
.. automodule:: beltrami_jax.reference
   :members:
```

## Developer utility

The command-line fixture builder converts a raw SPEC text dump prefix into a compressed packaged fixture:

```bash
PYTHONPATH=src ./.venv/bin/python tools/build_spec_fixture.py PREFIX OUTPUT
```

Inputs:

- `PREFIX`
  - path prefix for the SPEC dump, for example `.../G3V01L0Fi.dump.lvol1`
- `OUTPUT`
  - target `.npz` file to create

Outputs:

- a compressed NumPy archive containing:
  - `d_ma`
  - `d_md`
  - `d_mb`
  - `d_mg` when present in the source dump
  - `mu`
  - `psi`
  - `is_vacuum`
  - `matrix`
  - `rhs`
  - `solution`
  - `volume_index`
  - `label`
  - `source`
