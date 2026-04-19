# beltrami_jax

`beltrami_jax` is a differentiable JAX implementation of the Beltrami workflow used in SPEC/SPECTRE-style multi-region relaxed MHD problems.

The package reproduces the core operator solve

$$
\mathbf{M}\mathbf{a} = \mathbf{r},
\qquad
\mathbf{M} = \mathbf{A} - \mu \mathbf{D},
$$

and now supports both dumped-SPEC regression and an internal geometry-driven path that assembles the system, updates `mu`, and postprocesses the solution.

```{toctree}
:maxdepth: 2
:caption: Contents

overview
theory
validation
examples
api
limitations
references
```

## Scope

The repository currently covers the supported Beltrami workflow:

- typed system representation
- internal Fourier geometry assembly
- operator and right-hand-side assembly
- dense and GMRES solves
- a helicity-constrained outer update for `mu`
- diagnostics and benchmark helpers
- vectorized parameter scans
- autodifferentiation through solved states
- regression testing against SPEC fixtures

It does not yet include all of SPEC or all SPECTRE-specific integration details.

## Quick links

- GitHub repository: [rogeriojorge/beltrami_jax](https://github.com/rogeriojorge/beltrami_jax)
- SPEC documentation: [princetonuniversity.github.io/SPEC](https://princetonuniversity.github.io/SPEC/)
- Project log and restart document: [`plan.md`](https://github.com/rogeriojorge/beltrami_jax/blob/main/plan.md)

## Why this exists

The medium-term objective is to make the Beltrami stage easier to:

- install without Fortran bindings
- inspect and benchmark
- differentiate in optimization and inverse-design workflows
- integrate into future SPECTRE development once the SPECTRE source is public
