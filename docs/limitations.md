# Limitations and Future Work

## Current limitations

The current package is deliberately narrower than the long-term project goal.

### Not a full SPEC port

`beltrami_jax` now includes an internal geometry/integral assembly path, a GMRES path, and a helicity-constrained outer loop, but it still does not match every branch and auxiliary structure in SPEC's Fortran implementation.

### Not a full SPECTRE port

The package is intended as a candidate Beltrami kernel for SPECTRE integration, but it does not yet match SPECTRE's full backend contract. SPECTRE TOML input summaries, HDF5 vector-potential coefficient comparison, packed radial layouts, solution-vector pack/unpack maps, released SPECTRE matrix/RHS/solution parity fixtures, local `Lconstraint` branch formulas, a JAX-native interface-geometry evaluator, SPECTRE `matrixBG` `dMB/dMG` boundary assembly, SPECTRE `dMA/dMD` matrix assembly, centroid-style toroidal axis initialization, TOML-driven per-volume and multi-volume `Ate/Aze/Ato/Azo` solves, branch derivative vector-potential blocks, SPECTRE current diagnostics, a SPECTRE Fourier rotational-transform diagnostic for validated stellarator-symmetric `Lsparse=0/3` cases, selected local constraint updates, and an experimental SPECTRE injection seam are now implemented. The remaining milestone is making the global/semi-global update path and the non-stellarator-symmetric transform/current branches work directly instead of receiving final SPECTRE-updated values as validation inputs.

### Limited linear-algebra coverage

The package now supports dense solves and a compact GMRES implementation, including matrix-free usage. It still does not provide a production sparse backend or the full solver menu present in mature equilibrium codes.

### Limited branch coverage

The current implementation supports:

- plasma, vacuum, coordinate-singularity source, and local `Lconstraint` residual/Jacobian branches
- internal axis-regularized Fourier assembly
- SPECTRE interface-Fourier coordinate interpolation, Jacobian, and metric evaluation
- SPECTRE `matrixBG` boundary-source assembly for `dMB/dMG`
- SPECTRE radial basis, quadrature, metric-integral, and `dMA/dMD` matrix contraction for the packaged cylindrical, toroidal, free-boundary, and vacuum branches
- TOML-driven per-volume and multi-volume solves that unpack to SPECTRE-compatible `Ate/Aze/Ato/Azo`
- derivative vector-potential blocks, magnetic energy/helicity integrals, SPECTRE current diagnostics, SPECTRE rotational-transform diagnostics for validated stellarator-symmetric Fourier cases, and selected local current/helicity/transform constraint updates
- dense and GMRES solve paths
- an outer helicity-constrained nonlinear update

It does not yet cover every branch and auxiliary path present in SPEC/SPECTRE Fortran. The main open branch work is global/semi-global nonlinear updates, non-stellarator-symmetric transform/current diagnostics, broader high-resolution fixtures, and production sparse/matrix-free scaling.

### Limited fixture diversity

The repository now includes multiple dumped SPEC systems, including a nonzero-`mu` plasma fixture and a free-boundary vacuum fixture. That is enough for a stronger regression baseline, but it is still not broad confidence across all geometries, resolutions, and solver branches present in SPEC.

## Planned next steps

The near-term roadmap is:

1. enable and verify the hosted Read the Docs project
2. add more public SPECTRE HDF5 vector-potential comparison cases
3. rebuild and validate the new SPECTRE `set_vec_pot`/`solve_beltrami_jax` injection seam
4. extend rotational-transform/current diagnostics beyond the validated stellarator-symmetric local branches and wire them into the global/semi-global update path
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
- SPECTRE local branch-solve and `Lconstraint` formulas are now represented in JAX with current diagnostics and a validated Fourier rotational-transform diagnostic
- SPECTRE interface geometry can now be evaluated natively in JAX, including free-boundary wall rows, coordinate-singularity interpolation, Jacobian, and metric tensor
- SPECTRE `matrixBG` boundary assembly now produces `dMB/dMG` from SPECTRE packed maps and normal-field arrays
- SPECTRE `dMA/dMD` matrix assembly now matches released SPECTRE fixtures for cylindrical axis/bulk volumes, toroidal generated-interface volumes, explicit-interface free-boundary volumes, and the free-boundary vacuum volume
- TOML-driven SPECTRE volume solves now produce per-volume and full-case `Ate/Aze/Ato/Azo` directly for validated branches when supplied the same post-constraint `mu`/flux values used by SPECTRE
- SPECTRE current diagnostics, local `Lconstraint=2` helicity updates, and local `Lconstraint=1` rotational-transform updates are now represented and tested from JAX-solved coefficients
- the local SPECTRE fork has an optional `SPECTRE.solve_beltrami_jax(...)` seam that can inject a full `beltrami_jax` coefficient block after rebuild
- exact JAX-native parity without injected post-constraint SPECTRE metadata is now demonstrated for the packaged local `Lconstraint=1` branch and still remains future work for global/semi-global branches and broader geometry coverage

That distinction matters for both scientific correctness and future integration planning.
