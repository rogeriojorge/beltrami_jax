Packaged SPECTRE Beltrami linear systems
========================================

These fixtures were exported from the open-source SPECTRE Python/Fortran
wrapper with `tools/export_spectre_linear_system_npz.py`. Each `.npz` file is
one SPECTRE volume after SPECTRE has assembled the Beltrami matrices and called
`solve_beltrami_system`.

Stored arrays:

- `d_ma`, `d_md`, `d_mb`, `d_mg`: SPECTRE matrix components for the volume.
- `matrix`, `rhs`: dense linear system used by the SPECTRE Beltrami solve.
- `solution`: SPECTRE's solved vector-potential degree-of-freedom vector.
- scalar metadata: case label, volume index, radial order, Fourier resolution,
  coordinate-singularity/vacuum flags, residual norms, and source input path.

The files are regression fixtures for validating `beltrami_jax` against
SPECTRE without requiring SPECTRE or its Fortran extension at test time.
