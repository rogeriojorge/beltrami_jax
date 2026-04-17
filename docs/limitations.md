# Limitations and Future Work

## Current limitations

The current package is deliberately narrower than the long-term project goal.

### Not a full SPEC port

`beltrami_jax` does not yet reconstruct the geometry-dependent matrix assembly performed upstream in SPEC. It assumes that the dense matrices and vectors are already available.

### Not a full SPECTRE port

The package is intended as a candidate Beltrami kernel for future SPECTRE integration, but it is not yet wired into SPECTRE and does not yet match any final SPECTRE API contract.

### Dense-only linear algebra

The current solve path uses dense `jax.numpy.linalg.solve`. This is useful for exact regression and autodiff, but may not be sufficient for larger systems where sparse or iterative strategies become important.

### Limited branch coverage

The current implementation supports:

- the standard dense plasma branch
- a vacuum right-hand-side branch via `d_mg`

It does not yet cover every branch and auxiliary matrix path present in SPEC's legacy Fortran, such as coordinate-singularity-specific terms and matrix-free solver paths.

### Limited fixture diversity

The repository now includes multiple dumped SPEC systems, including a nonzero-`mu` plasma fixture and a free-boundary vacuum fixture. That is enough for a stronger regression baseline, but it is still not broad confidence across all geometries, resolutions, and solver branches present in SPEC.

## Planned next steps

The near-term roadmap is:

1. enable and verify the hosted Read the Docs project
2. add more dumped SPEC fixtures
3. add a higher-level integration-oriented solve API
4. broaden benchmarks beyond the current dense-regression fixture set
5. evaluate sparse and matrix-free directions once the dense baseline is frozen

## Medium-term technical directions

Once the dense regression baseline is stable across more fixtures, the next technical candidates are:

- sparse linear algebra
- matrix-free operators
- GMRES-style iterative paths
- optional use of other JAX ecosystem libraries where they clearly improve ergonomics or performance

## Scientific caution

It is easy to overstate progress on a project like this. The correct current statement is:

- the dense linear Beltrami kernel has been reproduced and regression-tested against multiple SPEC dumps, including plasma and vacuum cases
- the larger equilibrium workflow has not yet been ported

That distinction matters for both scientific correctness and future integration planning.
