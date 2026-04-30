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

## Source-code map

The main implementation files are:

- `src/beltrami_jax/types.py`
- `src/beltrami_jax/geometry.py`
- `src/beltrami_jax/operators.py`
- `src/beltrami_jax/iterative.py`
- `src/beltrami_jax/solver.py`
- `src/beltrami_jax/nonlinear.py`
- `src/beltrami_jax/io.py`
- `src/beltrami_jax/reference.py`
- `src/beltrami_jax/spectre_input.py`
- `src/beltrami_jax/spectre_io.py`
- `src/beltrami_jax/spectre_layout.py`
- `src/beltrami_jax/spectre_linear.py`
- `src/beltrami_jax/spectre_pack.py`
- `src/beltrami_jax/spectre_validation.py`
- `src/beltrami_jax/diagnostics.py`
- `src/beltrami_jax/benchmark.py`

These are the files to read if you want to understand how the package maps geometry, constraints, and diagnostics onto the Beltrami solve.

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

## SPECTRE input summaries

```{eval-rst}
.. automodule:: beltrami_jax.spectre_input
   :members:
```

## SPECTRE HDF5 vector-potential IO

```{eval-rst}
.. automodule:: beltrami_jax.spectre_io
   :members:
```

## SPECTRE packed coefficient layout

```{eval-rst}
.. automodule:: beltrami_jax.spectre_layout
   :members:
```

## SPECTRE solution-vector packing

```{eval-rst}
.. automodule:: beltrami_jax.spectre_pack
   :members:
```

## Packaged SPECTRE linear systems

```{eval-rst}
.. automodule:: beltrami_jax.spectre_linear
   :members:
```

## Packaged SPECTRE validation cases

```{eval-rst}
.. automodule:: beltrami_jax.spectre_validation
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
