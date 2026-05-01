# Theory and Numerics

## Beltrami fields

A Beltrami field satisfies

$$
\nabla \times \mathbf{B} = \mu \mathbf{B},
$$

with scalar multiplier $\mu$. In a relaxed plasma region this implies

$$
\mathbf{J} \times \mathbf{B} = \mathbf{0},
$$

so the field is force free inside that region. In Taylor relaxation theory, the relaxed state minimizes magnetic energy while preserving appropriate helicity and flux constraints.

If the field is written through a vector potential,

$$
\mathbf{B} = \nabla \times \mathbf{A},
$$

then a constrained variation of magnetic energy and helicity leads to the Beltrami relation. In schematic form,

$$
\delta \left(
  W - \mu K - \lambda_t \psi_t - \lambda_p \psi_p
\right) = 0,
$$

with

$$
W = \int_{\Omega} \frac{|\mathbf{B}|^2}{2\mu_0}\, dV,
\qquad
K = \int_{\Omega} \mathbf{A}\cdot\mathbf{B}\, dV.
$$

## Role inside MRxMHD

In multi-region relaxed magnetohydrodynamics (MRxMHD), the plasma is partitioned into regions separated by ideal interfaces. Each region relaxes internally to a Beltrami state, while the interfaces preserve additional constraints. This allows equilibria that are not restricted to globally nested flux surfaces and is one of the reasons SPEC is useful for strongly 3D stellarator and shaped tokamak configurations.

This matters in fusion because:

- stellarators require intrinsically three-dimensional magnetic geometry
- tokamaks often involve strongly shaped boundaries and current-profile constraints
- relaxed subregions are useful reduced models for understanding confinement and force-free magnetic structure

## Discrete linear system in SPEC

The part of SPEC currently reproduced here is the dense linear solve described in `mp00ac.f90`. The operator is

$$
\mathbf{M} = \mathbf{A} - \mu \mathbf{D},
$$

where:

- $\mathbf{A}$ is the quadratic magnetic-energy matrix
- $\mathbf{D}$ is the quadratic helicity matrix
- $\mu$ is the Beltrami multiplier for the region

The right-hand side depends on the branch.

### Plasma region

For the current non-singular plasma branch implemented in `beltrami_jax`,

$$
\mathbf{r} = -\mathbf{B}\boldsymbol{\psi},
$$

with $\boldsymbol{\psi} = (\Delta\psi_t, \Delta\psi_p)^T$.

### Vacuum region

For the vacuum branch,

$$
\mathbf{r} = -\mathbf{G} - \mathbf{B}\boldsymbol{\psi}.
$$

The operator itself is reduced to

$$
\mathbf{M} = \mathbf{A},
$$

which is represented in the package by setting `is_vacuum=True`.

### Coordinate-singularity source term

SPECTRE also uses the $\mathbf{G}$ source in selected plasma-region branches,
for example the coordinate-singularity path with `Lconstraint == -2`. In
`beltrami_jax`, this is represented by keeping `is_vacuum=False` while setting
`include_d_mg_in_rhs=True` on `BeltramiLinearSystem`. This keeps the operator
as $\mathbf{A} - \mu\mathbf{D}$ while using

$$
\mathbf{r} = -\mathbf{G} - \mathbf{B}\boldsymbol{\psi}.
$$

## SPECTRE interface geometry

SPECTRE represents interfaces with Fourier coefficients in cylindrical or
toroidal coordinates. For a mode pair $(m,n)$ and field-period-scaled toroidal
mode $n_\mathrm{int}=n\,N_\mathrm{fp}$, `beltrami_jax` evaluates

$$
R(\theta,\zeta) =
\sum_{m,n}
R^c_{mn}\cos(m\theta - n_\mathrm{int}\zeta)
+
R^s_{mn}\sin(m\theta - n_\mathrm{int}\zeta),
$$

$$
Z(\theta,\zeta) =
\sum_{m,n}
Z^c_{mn}\cos(m\theta - n_\mathrm{int}\zeta)
+
Z^s_{mn}\sin(m\theta - n_\mathrm{int}\zeta).
$$

For non-axis volumes, SPECTRE uses a local radial coordinate $s\in[-1,1]$ and
linear interpolation between neighboring interfaces:

$$
C_{mn}(s) = \frac{1-s}{2}C_{mn}^{\mathrm{left}}
          + \frac{1+s}{2}C_{mn}^{\mathrm{right}}.
$$

For the coordinate-singularity volume, SPECTRE uses regularizing radial powers.
With $\bar{s}=(s+1)/2$, the current implementation mirrors the public SPECTRE
rules:

$$
f_m(\bar{s}) =
\begin{cases}
\bar{s}, & I_\mathrm{geometry}=2,\ m=0,\\
\bar{s}^{m+1}, & I_\mathrm{geometry}=2,\ m>0,\\
\bar{s}^2, & I_\mathrm{geometry}=3,\ m=0,\\
\bar{s}^{m}, & I_\mathrm{geometry}=3,\ m>0,
\end{cases}
$$

and

$$
C_{mn}(s) = C_{mn}^{\mathrm{axis}}
          + \left(C_{mn}^{\mathrm{interface}} - C_{mn}^{\mathrm{axis}}\right)
            f_m(\bar{s}).
$$

The resulting coordinate derivatives build the Jacobian and metric tensor used
by the SPECTRE matrix integrals. For toroidal geometry,

$$
J = R\left(Z_s R_\theta - R_s Z_\theta\right),
$$

and the nonzero covariant metric entries are assembled from dot products of
the coordinate derivatives, with

$$
g_{\zeta\zeta}
= R_\zeta^2 + Z_\zeta^2 + R^2.
$$

The current JAX implementation covers this interface-geometry layer and the
first SPECTRE `dMA/dMD` matrix-integral path for the validated branches.

## SPECTRE volume-matrix integrals

SPECTRE's `chebyshev_mod.F90::volume_integrate_chebyshev` builds tensors such
as

$$
DToocc_{\ell p i j}
=
\int_{-1}^{1} ds\, T'_{\ell i}(s)\,T_{p j}(s)
\int d\theta\,d\zeta\,\cos\alpha_i\,\cos\alpha_j,
$$

and

$$
TTssss_{\ell p i j}
=
\int_{-1}^{1} ds\, T_{\ell i}(s)\,T_{p j}(s)
\int d\theta\,d\zeta\,\sin\alpha_i\,\sin\alpha_j\,
\frac{g_{ss}}{\sqrt{g}},
$$

with analogous `TDst*`, `TDsz*`, `DDtt*`, `DDtz*`, and `DDzz*` tensors for
the other metric components and radial derivative combinations. The JAX path
evaluates the same integrals directly on SPECTRE's angular grid and
Gauss-Legendre radial quadrature, then contracts them with the packed
`Ate/Aze/Ato/Azo` maps in `matrices_mod.F90::matrix` form.

## SPECTRE `matrixBG` boundary assembly

SPECTRE separates the low-dimensional boundary-source assembly from the
geometry-dependent volume integrals. In `matrices_mod.F90::matrixBG`, the
flux-coupling matrix `dMB` is nonzero only on the axis/mean-mode Lagrange rows:

$$
(dMB)_{L_{mg}(0,0),\,\Delta\psi_t} = -1,
\qquad
(dMB)_{L_{mh}(0,0),\,\Delta\psi_p} = +1,
$$

when those rows exist in the packed volume map. The boundary-normal-field
source is assembled on the outer-boundary Lagrange rows. For stellarator
symmetry,

$$
(dMG)_{L_{me}(m,n)} = -\left(iV^{ns}_{mn} + iB^{ns}_{mn}\right),
\qquad (m,n)\ne(0,0),
$$

and without stellarator symmetry the odd-parity rows also receive

$$
(dMG)_{L_{mf}(m,n)} = -\left(iV^{nc}_{mn} + iB^{nc}_{mn}\right).
$$

`build_spectre_boundary_normal_field` reconstructs the initialized SPECTRE
normal-field arrays from TOML tables using the same mode recombination as
`preset_mod.F90`. Free-boundary runs may update `iBns/iBnc` during Picard
iterations, so exact post-update parity requires passing those live updated
arrays through `SpectreBoundaryNormalField`.

## SPECTRE branch derivatives and constraints

SPECTRE solves additional right-hand sides with the same matrix factorization
to compute derivatives needed by local constraints. In plasma regions,

$$
\frac{\partial \mathbf{a}}{\partial \mu}
=
\mathbf{M}^{-1}\mathbf{D}\mathbf{a},
\qquad
\frac{\partial \mathbf{a}}{\partial \Delta\psi_p}
=
-\mathbf{M}^{-1}\mathbf{B}\begin{bmatrix}0\\1\end{bmatrix}.
$$

In vacuum regions, the derivative unknowns are the toroidal and poloidal flux
increments:

$$
\frac{\partial \mathbf{a}}{\partial \Delta\psi_t}
=
-\mathbf{A}^{-1}\mathbf{B}\begin{bmatrix}1\\0\end{bmatrix},
\qquad
\frac{\partial \mathbf{a}}{\partial \Delta\psi_p}
=
-\mathbf{A}^{-1}\mathbf{B}\begin{bmatrix}0\\1\end{bmatrix}.
$$

For the coordinate-singularity current branch `Lconstraint == -2`, the local
unknown is $\Delta\psi_t$ and the source term includes $\mathbf{G}$. The
constraint residual/Jacobian table in `evaluate_spectre_constraints` mirrors
SPECTRE's branch formulas once rotational-transform and plasma-current
diagnostics are supplied by a field evaluator.

For local rotational-transform constraints, SPECTRE constructs a straight-field-line
angle correction in Fourier space. On an interface, the radial derivatives of
the vector-potential components are first evaluated as

$$
A'_\theta(m,n) = \partial_s A_\theta(m,n),
\qquad
A'_\zeta(m,n) = -\partial_s A_\zeta(m,n),
$$

with the same Chebyshev or Zernike endpoint basis used by SPECTRE. The unknown
Fourier vector $\lambda$ is then obtained from a dense linear system

$$
\mathbf{T}\lambda = \mathbf{b},
$$

where the first component $\lambda_0$ is the measured rotational transform
$\iota$ on that interface. The SPECTRE Fourier branch fills

$$
b_k = A'_\zeta(k),
\qquad
T_{k0} = A'_\theta(k),
$$

and for nonzero straight-field-line modes $j=(m_j,n_j)$ adds convolution terms
of the form

$$
\frac{1}{2}\left(-m_j A'_\zeta(k)+n_j A'_\theta(k)\right)
$$

to the $k+j$ and $k-j$ rows. `compute_spectre_rotational_transform` ports this
`Lsparse=0/3` Fourier branch for validated stellarator-symmetric inputs. For
derivatives, it solves

$$
\mathbf{T}\,\partial_x\lambda
=
\partial_x\mathbf{b} - \partial_x\mathbf{T}\lambda,
$$

matching SPECTRE's `DGEMV` correction before the derivative solve. The local
`Lconstraint=1` residuals then use SPECTRE's target indexing:

$$
F =
\begin{bmatrix}
\iota_{\mathrm{inner}} - \mathrm{oita}(lvol-1)\\
\iota_{\mathrm{outer}} - \mathrm{iota}(lvol)
\end{bmatrix},
$$

with the coordinate-axis branch using only the outer residual.

## Internal assembly model in `beltrami_jax`

The internal workflow in this repository constructs a packed Fourier basis over a shaped large-aspect-ratio torus. At the continuous level, the assembled operators approximate

$$
A_{ij} \approx \int_{\Omega}
\left(
  \nabla \phi_i \cdot \nabla \phi_j
  + \alpha \phi_i \phi_j
\right) J \, dr\, d\theta\, d\zeta,
$$

$$
D_{ij} \approx \int_{\Omega} \phi_i \phi_j J \, dr\, d\theta\, d\zeta,
$$

$$
B_{if} \approx \int_{\Omega} \phi_i \chi_f J \, dr\, d\theta\, d\zeta,
$$

where $\phi_i$ are packed basis functions, $\chi_f$ are flux-driving modes, $J$ is the geometry Jacobian, and $\alpha$ is a small stabilization shift. This is not a symbolic line-by-line copy of SPEC's Fortran assembly, but it produces the same discrete Beltrami structure that downstream code needs.

## Quadratic diagnostics

After solving for the coefficient vector $\mathbf{a}$, the code evaluates the same quadratic diagnostics that SPEC uses in its linear stage:

magnetic energy
: $$
  W(\mathbf{a}) =
  \frac{1}{2}\mathbf{a}^T \mathbf{A}\mathbf{a}
  + \mathbf{a}^T \mathbf{B}\boldsymbol{\psi}
  $$

magnetic helicity
: $$
  K(\mathbf{a}) =
  \frac{1}{2}\mathbf{a}^T \mathbf{D}\mathbf{a}
  $$

linear residual
: $$
  \mathbf{e} = \mathbf{M}\mathbf{a} - \mathbf{r}
  $$

relative residual norm
: $$
  \frac{\lVert \mathbf{e} \rVert_2}
       {\max(\lVert \mathbf{r} \rVert_2, 10^{-30})}
  $$

## Numerical implementation in JAX

The current dense solve path uses

$$
\mathbf{a} = \operatorname{solve}(\mathbf{M}, \mathbf{r}),
$$

implemented through `jax.numpy.linalg.solve`.

This choice has three advantages for the current stage of the project:

1. It matches the dense linear-regression use case against dumped SPEC systems.
2. It remains differentiable with respect to matrix entries and parameters such as `mu`.
3. It is simple to batch with `jax.vmap`.

The package also includes a compact GMRES implementation, which can be driven either by an explicit matrix or by a matrix-vector callback.

## JAX transformations used

The implemented package already relies on three important JAX features.

### `jit`

`solve_operator` and `solve_parameter_scan` are JIT compiled. This reduces repeated solve overhead once shapes are fixed and is the first step toward a performance-oriented interface.

### `vmap`

`solve_parameter_scan` builds a batch of operators

$$
\mathbf{M}_b = \mathbf{A} - \mu_b \mathbf{D}
$$

and solves them in parallel across a batch index.

### `grad`

Because the solve is written in JAX primitives, scalar objectives depending on the solved state can be differentiated. The repository's example and tests differentiate magnetic energy with respect to `mu`.

## Current theory-to-code boundary

The repository now covers the supported Beltrami workflow end to end:

- internal geometry/integral assembly for a Fourier large-aspect-ratio torus
- SPECTRE interface-Fourier coordinate interpolation, Jacobian, and metric evaluation
- SPECTRE radial basis/quadrature, metric-integral tensors, and `dMA/dMD` matrix contraction for packaged cylindrical, toroidal, free-boundary, and vacuum branches
- SPECTRE TOML-to-volume and TOML-to-full-coefficient solves that unpack directly to `Ate/Aze/Ato/Azo` for validated branches
- dense and GMRES linear solves
- SPECTRE local branch derivative solves and `Lconstraint` residual/Jacobian formulas with JAX current and rotational-transform diagnostics for validated branches
- a helicity-constrained outer update for `mu`
- axis-regularized basis handling near the coordinate singularity

What still remains outside the current implementation is the global/semi-global SPECTRE nonlinear update layer, non-stellarator-symmetric transform/current branches, and broader fixture coverage needed before the SPECTRE fork can remove its Fortran Beltrami path by default.
