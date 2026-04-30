# Examples

The repository includes executable examples under `examples/`. They are designed to be:

- standalone scripts with parameters at the top
- verbose in the terminal
- capable of writing inputs, outputs, and figures under `examples/_generated/`
- representative of dumped-SPEC, SPECTRE-adapter, SPECTRE-geometry, and internal-assembly workflows

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
- uses packaged SPECTRE compare cases when local SPECTRE exports are absent
- compares `Ate`, `Aze`, `Ato`, and `Azo` with `compare_vector_potentials`
- falls back to a synthetic SPECTRE-layout HDF5 file only if packaged validation fixtures are unavailable
- writes a JSON summary and a compact validation figure under `examples/_generated/validate_spectre_vector_potential/`

Committed SPECTRE parity figure:

![SPECTRE vector-potential parity](_static/spectre_vecpot_parity.png)

## SPECTRE assembled-matrix backend adapter

Run:

```bash
./.venv/bin/python examples/spectre_backend_dropin.py
```

This example:

- loads packaged released-SPECTRE per-volume linear systems
- calls `solve_spectre_assembled_numpy`, the minimal adapter intended for a small SPECTRE-side Python branch
- solves a batch of equal-size SPECTRE plasma volumes with `solve_spectre_assembled_batch`
- records compile+solve and steady-state adapter timings
- writes a JSON summary and panel under `examples/_generated/spectre_backend_dropin/`

Committed SPECTRE linear parity figure:

![SPECTRE linear-system parity](_static/spectre_linear_parity.png)

## SPECTRE TOML volume and full-case solves

The user-facing SPECTRE replacement helpers are `solve_spectre_volume_from_input`
for one packed volume and `solve_spectre_volumes_from_input` for a full
SPECTRE-compatible `Ate/Aze/Ato/Azo` coefficient block. They start from a
SPECTRE TOML summary, assemble the local matrices in JAX, solve the Beltrami
system, and unpack directly to `Ate/Aze/Ato/Azo`:

```python
from beltrami_jax import load_packaged_spectre_case, solve_spectre_volume_from_input

case = load_packaged_spectre_case("G3V3L3Fi")
result = solve_spectre_volume_from_input(case.input_summary, lvol=2, verbose=True)

print(result.vector_potential.ate.shape)
print(float(result.relative_residual_norm))
```

For exact comparison against a completed SPECTRE fixture, pass the final
SPECTRE branch values:

```python
from beltrami_jax import load_packaged_spectre_linear_system

fixture = load_packaged_spectre_linear_system("G3V3L3Fi/lvol2")
result = solve_spectre_volume_from_input(
    case.input_summary,
    lvol=2,
    mu=fixture.system.mu,
    psi=fixture.system.psi,
)
```

Without explicit `mu`/`psi`, the helper uses the normalized TOML initial state.

Run the full multi-volume example:

```bash
./.venv/bin/python examples/spectre_toml_full_solve.py
```

This example:

- loads one packaged released-SPECTRE TOML compare case
- injects the packaged post-constraint `mu`/`psi` state for strict validation
- assembles `dMA/dMD/dMB/dMG` for all packed volumes in JAX
- solves all packed volumes and concatenates a full vector-potential block
- writes `jax_vector_potential.npz`, a JSON summary, and a parity figure

Committed SPECTRE TOML full-solve figure:

![SPECTRE TOML full solve](_static/spectre_toml_full_solve.png)

## SPECTRE interface-geometry probe

Run:

```bash
./.venv/bin/python examples/spectre_geometry_probe.py
```

This example:

- loads a packaged SPECTRE TOML compare case
- builds JAX-native SPECTRE interface Fourier coefficient arrays with `build_spectre_interface_geometry`
- interpolates a volume with the same coordinate-singularity and linear-interface rules used by SPECTRE `compute_coordinates`
- evaluates real-space `R`, `Z`, first derivatives, Jacobian, and metric tensor with `evaluate_spectre_volume_coordinates`
- writes a JSON summary and a geometry/metric panel under `examples/_generated/spectre_geometry_probe/`

Committed SPECTRE geometry probe:

![SPECTRE geometry probe](_static/spectre_geometry_probe.png)

## Regenerate the committed validation panels

The repository-level validation and benchmark figures are built from the packaged fixtures with:

```bash
PYTHONPATH=src ./.venv/bin/python tools/generate_validation_assets.py --repeats 2
```

The SPECTRE HDF5 coefficient parity figure is built with:

```bash
PYTHONPATH=src ./.venv/bin/python tools/generate_spectre_validation_assets.py --use-packaged
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
