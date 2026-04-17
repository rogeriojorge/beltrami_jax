from __future__ import annotations

import jax
import jax.numpy as jnp

from beltrami_jax.operators import magnetic_energy
from beltrami_jax.reference import load_packaged_reference
from beltrami_jax.solver import solve_from_components
from beltrami_jax.types import BeltramiLinearSystem


def main() -> None:
    reference = load_packaged_reference()

    def solved_energy(mu_value: float) -> jax.Array:
        system = BeltramiLinearSystem.from_arraylike(
            d_ma=reference.system.d_ma,
            d_md=reference.system.d_md,
            d_mb=reference.system.d_mb,
            mu=mu_value,
            psi=reference.system.psi,
            label="autodiff scan",
        )
        result = solve_from_components(system)
        return magnetic_energy(result.solution, system)

    energy = solved_energy(float(reference.system.mu))
    gradient = jax.grad(solved_energy)(float(reference.system.mu))
    print(f"[beltrami_jax] energy(mu0)={float(energy):.8e}")
    print(f"[beltrami_jax] denergy/dmu={float(gradient):.8e}")


if __name__ == "__main__":
    main()
