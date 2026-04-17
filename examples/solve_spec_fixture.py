from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from beltrami_jax.reference import load_packaged_reference
from beltrami_jax.solver import solve_from_components


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve the packaged SPEC fixture with beltrami_jax.")
    parser.add_argument("--plot", type=Path, help="Optional path for a coefficient-spectrum plot.")
    args = parser.parse_args()

    reference = load_packaged_reference()
    result = solve_from_components(reference.system, verbose=True)
    difference = np.asarray(result.solution - reference.expected_solution)

    print(f"[beltrami_jax] max_abs_solution_error={np.max(np.abs(difference)):.8e}")
    print(f"[beltrami_jax] magnetic_energy={float(result.magnetic_energy):.8e}")
    print(f"[beltrami_jax] magnetic_helicity={float(result.magnetic_helicity):.8e}")

    if args.plot:
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(8, 4.5))
        plt.semilogy(np.abs(np.asarray(result.solution)) + 1e-18, label="beltrami_jax")
        plt.semilogy(np.abs(np.asarray(reference.expected_solution)) + 1e-18, "--", label="SPEC")
        plt.xlabel("Packed coefficient index")
        plt.ylabel("|a_i|")
        plt.title("SPEC fixture coefficient spectrum")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.plot, dpi=180)
        print(f"[beltrami_jax] wrote {args.plot}")


if __name__ == "__main__":
    main()
