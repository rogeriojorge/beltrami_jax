# Examples

The repository includes executable examples under `examples/`. They are designed to be:

- standalone scripts with parameters at the top
- verbose in the terminal
- capable of writing inputs, outputs, and figures under `examples/_generated/`
- representative of both dumped-SPEC and internal-assembly workflows

## Solve the packaged SPEC fixture

Run:

```bash
./.venv/bin/python examples/solve_spec_fixture.py
```

This example:

- loads the packaged SPEC regression fixture
- runs both dense and GMRES solves
- demonstrates matrix-free GMRES on the same operator
- writes a JSON summary and a validation panel under `examples/_generated/solve_spec_fixture/`

Committed figure:

![SPEC fixture workflow](_static/spec_fixture_spectrum.png)

## Geometry-driven parameter scan and nonlinear solve

Run:

```bash
./.venv/bin/python examples/parameter_scan.py
```

This example:

- defines a shaped geometry with `FourierBeltramiGeometry`
- builds a packed basis with `build_fourier_mode_basis`
- saves a JSON input file with `save_problem_json`
- runs the outer helicity-constrained nonlinear solve
- performs a vectorized `mu` scan with `solve_parameter_scan`
- writes scan data and a postprocessed figure under `examples/_generated/parameter_scan/`

Committed figure:

![Geometry parameter scan](_static/parameter_scan.png)

## Differentiate magnetic energy with respect to `mu`

Run:

```bash
./.venv/bin/python examples/autodiff_mu.py
```

This example:

- defines a geometry-driven problem
- solves for a target helicity
- rebuilds the linear system for varying `mu`
- applies `jax.grad` to a solved-energy objective
- writes a gradient report and a verification figure under `examples/_generated/autodiff_mu/`

Committed figure:

![Autodiff gradient check](_static/autodiff_gradient_check.png)

## Vacuum/GMRES export and benchmark workflow

Run:

```bash
./.venv/bin/python examples/benchmark_fixtures.py
```

This example:

- builds a vacuum problem with internal geometry assembly
- runs the outer loop with the GMRES solve path
- exports the solution bundle with `save_nonlinear_solution`
- benchmarks a dense solve and a batched parameter scan
- writes a compact performance figure under `examples/_generated/benchmark_fixtures/`

Committed figure:

![Vacuum GMRES benchmark](_static/vacuum_gmres_panel.png)

## Validate SPECTRE vector-potential coefficients

Run:

```bash
./.venv/bin/python examples/validate_spectre_vector_potential.py
```

This example:

- reads SPECTRE TOML metadata with `load_spectre_input_toml`
- loads SPECTRE HDF5 vector-potential datasets with `load_spectre_reference_h5`
- loads fresh exported SPECTRE coefficients from `.npz` files when they are present
- compares `Ate`, `Aze`, `Ato`, and `Azo` with `compare_vector_potentials`
- falls back to a synthetic SPECTRE-layout HDF5 file when SPECTRE is not installed locally
- writes a JSON summary and a compact validation figure under `examples/_generated/validate_spectre_vector_potential/`

Committed SPECTRE parity figure:

![SPECTRE vector-potential parity](_static/spectre_vecpot_parity.png)

## Regenerate the committed validation panels

The repository-level validation and benchmark figures are built from the packaged fixtures with:

```bash
PYTHONPATH=src ./.venv/bin/python tools/generate_validation_assets.py --repeats 2
```

The SPECTRE HDF5 coefficient parity figure is built with:

```bash
PYTHONPATH=src ./.venv/bin/python tools/generate_spectre_validation_assets.py
```

## Example usage from Python

```python
from beltrami_jax import (
    BeltramiProblem,
    FourierBeltramiGeometry,
    assemble_fourier_beltrami_system,
    build_fourier_mode_basis,
    solve_helicity_constrained_equilibrium,
)

geometry = FourierBeltramiGeometry(major_radius=3.0, minor_radius=1.0, elongation=1.2)
basis = build_fourier_mode_basis(max_radial_order=1, max_poloidal_mode=2, max_toroidal_mode=1)
assembly = assemble_fourier_beltrami_system(geometry, basis, mu=0.05, psi=(0.1, 0.0))

problem = BeltramiProblem.from_arraylike(
    geometry=geometry,
    basis=basis,
    psi=(0.1, 0.0),
    target_helicity=0.05,
    initial_mu=0.05,
)
result = solve_helicity_constrained_equilibrium(problem)

print(result.solve.relative_residual_norm)
print(result.solve.magnetic_energy)
```
