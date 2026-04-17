# beltrami_jax

`beltrami_jax` is a differentiable JAX implementation of the dense linear Beltrami solve used in SPEC/SPECTRE-style multi-region relaxed MHD workflows.

The package currently reproduces the linear operator solve

$$
\mathbf{M}\mathbf{a} = \mathbf{r},
\qquad
\mathbf{M} = \mathbf{A} - \mu \mathbf{D},
$$

and validates it against dense systems dumped from SPEC.

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

The repository is intentionally focused on the discrete linear stage of the Beltrami solve:

- typed system representation
- operator and right-hand-side assembly
- dense JAX solve
- vectorized parameter scans
- autodifferentiation through solved states
- regression testing against SPEC fixtures

It does not yet include all of SPEC or the full outer equilibrium iteration.

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
