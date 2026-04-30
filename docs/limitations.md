# Limitations and Future Work

## Current limitations

The current package is deliberately narrower than the long-term project goal.

### Not a full SPEC port

`beltrami_jax` now includes an internal geometry/integral assembly path, a GMRES path, and a helicity-constrained outer loop, but it still does not match every branch and auxiliary structure in SPEC's Fortran implementation.

### Not a full SPECTRE port

The package is intended as a candidate Beltrami kernel for SPECTRE integration, but it is not yet wired into SPECTRE and does not yet match SPECTRE's full backend contract. SPECTRE TOML input summaries, HDF5 vector-potential coefficient comparison, packed radial layouts, solution-vector pack/unpack maps, released SPECTRE matrix/RHS/solution parity fixtures, local `Lconstraint` branch formulas, and a JAX-native interface-geometry evaluator are now implemented; the remaining milestone is making the JAX-native matrix/integral assembly and solve path produce those SPECTRE coefficients directly from interface geometry.

### Limited linear-algebra coverage

The package now supports dense solves and a compact GMRES implementation, including matrix-free usage. It still does not provide a production sparse backend or the full solver menu present in mature equilibrium codes.

### Limited branch coverage

The current implementation supports:

- plasma, vacuum, coordinate-singularity source, and local `Lconstraint` residual/Jacobian branches
- internal axis-regularized Fourier assembly
- SPECTRE interface-Fourier coordinate interpolation, Jacobian, and metric evaluation
- dense and GMRES solve paths
- an outer helicity-constrained nonlinear update

It does not yet cover every branch and auxiliary matrix path present in SPEC/SPECTRE Fortran, including exact SPECTRE matrix integral assembly and field diagnostics such as rotational transform/current evaluation from the JAX-native geometry.

### Limited fixture diversity

The repository now includes multiple dumped SPEC systems, including a nonzero-`mu` plasma fixture and a free-boundary vacuum fixture. That is enough for a stronger regression baseline, but it is still not broad confidence across all geometries, resolutions, and solver branches present in SPEC.

## Planned next steps

The near-term roadmap is:

1. enable and verify the hosted Read the Docs project
2. add more public SPECTRE HDF5 vector-potential comparison cases
3. use the new SPECTRE assembled-matrix backend adapter in a small SPECTRE fork experiment
4. implement JAX-native SPECTRE matrix/integral assembly on top of the new interface-geometry evaluator
5. broaden benchmarks beyond the current dense-regression and compact internal-geometry cases
6. add broader SPEC/SPECTRE parity tests for branch-specific geometry terms

## Medium-term technical directions

Once the current regression and internal-assembly baseline is stable across more fixtures, the next technical candidates are:

- sparse linear algebra
- matrix-free operators
- GMRES-style iterative paths
- optional use of other JAX ecosystem libraries where they clearly improve ergonomics or performance

## Scientific caution

It is easy to overstate progress on a project like this. The correct current statement is:

- the package now supports a full internal Beltrami workflow for its current Fourier-geometry model
- the linear kernel has also been regression-tested against multiple SPEC dumps, including plasma and vacuum cases
- SPECTRE HDF5 coefficient loading/comparison reaches machine-precision parity for fresh SPECTRE exports
- SPECTRE coefficient pack/unpack maps now match the public SPECTRE layout and round-trip packaged coefficients exactly
- JAX dense solves reproduce released SPECTRE per-volume Beltrami linear systems once SPECTRE has assembled the matrices
- SPECTRE local branch-solve and `Lconstraint` formulas are now represented in JAX with injected transform/current diagnostics
- SPECTRE interface geometry can now be evaluated natively in JAX, including free-boundary wall rows, coordinate-singularity interpolation, Jacobian, and metric tensor
- exact JAX-native parity with all SPEC/SPECTRE branches and HDF5 vector-potential coefficients still remains future work

That distinction matters for both scientific correctness and future integration planning.
