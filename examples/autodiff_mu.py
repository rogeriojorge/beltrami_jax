from __future__ import annotations

import argparse

import jax

from beltrami_jax.operators import magnetic_energy
from beltrami_jax.reference import load_packaged_reference
from beltrami_jax.solver import solve_from_components
from beltrami_jax.types import BeltramiLinearSystem


def main() -> None:
    parser = argparse.ArgumentParser(description="Differentiate solved magnetic energy with respect to mu.")
    parser.add_argument("--name", default="g3v02l1fi_lvol1", help="Non-vacuum packaged fixture name.")
    args = parser.parse_args()

    reference = load_packaged_reference(args.name)
    if reference.system.is_vacuum:
        raise ValueError("autodiff_mu.py requires a non-vacuum packaged fixture")

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
    delta = 1e-4
    finite_difference = (solved_energy(float(reference.system.mu) + delta) - solved_energy(float(reference.system.mu) - delta)) / (2.0 * delta)
    print(f"[beltrami_jax] fixture={args.name}")
    print(f"[beltrami_jax] energy(mu0)={float(energy):.8e}")
    print(f"[beltrami_jax] denergy/dmu={float(gradient):.8e}")
    print(f"[beltrami_jax] finite_difference={float(finite_difference):.8e}")


if __name__ == "__main__":
    main()
