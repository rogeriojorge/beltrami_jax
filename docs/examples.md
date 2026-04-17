# Examples

The repository includes small executable examples under `examples/`. They are designed to be:

- simple to run
- verbose in the terminal
- directly tied to the packaged SPEC fixture

## Solve the packaged SPEC fixture

Run:

```bash
./.venv/bin/python examples/solve_spec_fixture.py --name g3v02l1fi_lvol1
```

This example:

- loads the packaged SPEC regression fixture
- assembles and solves the dense JAX system
- prints residual metrics, conditioning diagnostics, magnetic energy, and helicity
- optionally writes a coefficient-spectrum figure with `--plot`

Generated figure:

![Coefficient spectrum](_static/spec_fixture_spectrum.png)

## Parameter scan in `mu`

Run:

```bash
./.venv/bin/python examples/parameter_scan.py --name g3v02l1fi_lvol1
```

This example:

- builds a small range of `mu` values around the reference case
- uses `solve_parameter_scan`
- prints the corresponding energy for each batched solve
- optionally writes a line plot with `--plot`

Generated figure:

![Parameter scan](_static/parameter_scan.png)

## Differentiate magnetic energy with respect to `mu`

Run:

```bash
./.venv/bin/python examples/autodiff_mu.py --name g3v02l1fi_lvol1
```

This example:

- reconstructs a `BeltramiLinearSystem` for varying `mu`
- solves the dense linear system in JAX
- evaluates magnetic energy
- applies `jax.grad` to obtain `dE/dmu`
- prints a centered finite-difference check alongside the autodiff derivative

## Benchmark a packaged fixture

Run:

```bash
./.venv/bin/python examples/benchmark_fixtures.py --name g1v03l0fi_lvol2
```

This example:

- measures one first-call timing for `solve_from_components`
- measures steady-state repeated solve timing
- benchmarks `solve_parameter_scan` for batch sizes `1`, `4`, and `8`
- prints all timings in a compact, script-friendly format

## Regenerate the committed validation panels

The repository-level validation and benchmark figures are built from the packaged fixtures with:

```bash
PYTHONPATH=src ./.venv/bin/python tools/generate_validation_assets.py --repeats 2
```

## Regenerate the figures

The committed figures in this documentation can be rebuilt with:

```bash
./.venv/bin/python examples/solve_spec_fixture.py --plot docs/_static/spec_fixture_spectrum.png
./.venv/bin/python examples/parameter_scan.py --plot docs/_static/parameter_scan.png
PYTHONPATH=src ./.venv/bin/python tools/generate_validation_assets.py --repeats 2
```

## Example usage from Python

```python
from beltrami_jax import load_packaged_reference, solve_from_components

reference = load_packaged_reference()
result = solve_from_components(reference.system, verbose=True)

print(result.relative_residual_norm)
print(result.magnetic_energy)
```
