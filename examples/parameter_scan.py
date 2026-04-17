from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from beltrami_jax.operators import magnetic_energy
from beltrami_jax.reference import load_packaged_reference
from beltrami_jax.solver import solve_parameter_scan


def main() -> None:
    parser = argparse.ArgumentParser(description="Vectorized mu scan around the packaged SPEC fixture.")
    parser.add_argument("--name", default="g3v02l1fi_lvol1", help="Non-vacuum packaged fixture name.")
    parser.add_argument("--plot", type=Path, help="Optional output plot path.")
    args = parser.parse_args()

    reference = load_packaged_reference(args.name)
    if reference.system.is_vacuum:
        raise ValueError("parameter_scan.py requires a non-vacuum packaged fixture")
    mu0 = float(reference.system.mu)
    mu_values = np.linspace(mu0 - 0.05, mu0 + 0.05, 9)
    psi_values = np.repeat(np.asarray(reference.system.psi)[None, :], len(mu_values), axis=0)
    solutions = solve_parameter_scan(reference.system.d_ma, reference.system.d_md, reference.system.d_mb, mu_values, psi_values)
    energies = np.asarray(
        [magnetic_energy(solution, reference.system) for solution in np.asarray(solutions)]
    )

    for mu_value, energy in zip(mu_values, energies, strict=True):
        print(f"[beltrami_jax] mu={mu_value:.8e} energy={float(energy):.8e}")

    if args.plot:
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(8, 4.5))
        plt.plot(mu_values, energies, marker="o")
        plt.xlabel("mu")
        plt.ylabel("magnetic energy")
        plt.title(f"Vectorized parameter scan: {args.name}")
        plt.tight_layout()
        plt.savefig(args.plot, dpi=180)
        print(f"[beltrami_jax] wrote {args.plot}")


if __name__ == "__main__":
    main()
