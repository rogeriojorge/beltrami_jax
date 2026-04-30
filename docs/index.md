# beltrami_jax

`beltrami_jax` is a differentiable JAX implementation of the Beltrami workflow used in SPEC/SPECTRE-style multi-region relaxed MHD problems.

The package reproduces the core operator solve

$$
\mathbf{M}\mathbf{a} = \mathbf{r},
\qquad
\mathbf{M} = \mathbf{A} - \mu \mathbf{D},
$$

and now supports dumped-SPEC regression, an internal prototype geometry-driven path, and SPECTRE TOML/HDF5/linear-system validation utilities.

```{toctree}
:maxdepth: 2
:caption: Contents

overview
theory
validation
examples
integration
api
limitations
references
```

## Scope

The repository currently covers the supported Beltrami workflow components:

- typed system representation
- internal prototype Fourier geometry assembly
- operator and right-hand-side assembly
- dense and GMRES solves
- a helicity-constrained outer update for `mu`
- diagnostics and benchmark helpers
- vectorized parameter scans
- autodifferentiation through solved states
- regression testing against SPEC fixtures
- SPECTRE TOML input summaries
- SPECTRE HDF5 vector-potential coefficient loading and comparison
- SPECTRE solution-vector pack/unpack maps and packaged linear-system parity fixtures
- SPECTRE TOML/interface-geometry matrix assembly for the packaged validated branches
- SPECTRE-shaped `Ate/Aze/Ato/Azo` solves, derivative vector-potential blocks, current diagnostics, and selected local constraint updates

It does not yet include all of SPEC or a full JAX-native SPECTRE Beltrami backend because rotational-transform diagnostics and global/semi-global constraint updates remain open.

## Quick links

- GitHub repository: [rogeriojorge/beltrami_jax](https://github.com/rogeriojorge/beltrami_jax)
- SPEC documentation: [princetonuniversity.github.io/SPEC](https://princetonuniversity.github.io/SPEC/)
- Project log and restart document: [`plan.md`](https://github.com/rogeriojorge/beltrami_jax/blob/main/plan.md)

## Why this exists

The medium-term objective is to make the Beltrami stage easier to:

- install without Fortran bindings
- inspect and benchmark
- differentiate in optimization and inverse-design workflows
- integrate into released SPECTRE development once the JAX-native assembly path reaches coefficient parity
