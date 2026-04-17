from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from beltrami_jax.diagnostics import compare_against_reference, compute_solve_diagnostics
from beltrami_jax.reference import load_packaged_reference
from beltrami_jax.solver import solve_from_components


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve the packaged SPEC fixture with beltrami_jax.")
    parser.add_argument("--name", default="g3v01l0fi_lvol1", help="Packaged fixture name.")
    parser.add_argument("--plot", type=Path, help="Optional path for a coefficient-spectrum plot.")
    args = parser.parse_args()

    reference = load_packaged_reference(args.name)
    result = solve_from_components(reference.system, verbose=True)
    diagnostics = compute_solve_diagnostics(result, include_condition_number=True)
    comparison = compare_against_reference(reference, result)
    difference = np.asarray(result.solution - reference.expected_solution)

    print(f"[beltrami_jax] fixture={args.name} lvol={reference.volume_index} size={reference.system.size}")
    print(f"[beltrami_jax] max_abs_solution_error={np.max(np.abs(difference)):.8e}")
    print(f"[beltrami_jax] solution_relative_error={comparison.solution_relative_error:.8e}")
    print(f"[beltrami_jax] operator_relative_error={comparison.operator_relative_error:.8e}")
    print(f"[beltrami_jax] rhs_relative_error={comparison.rhs_relative_error:.8e}")
    print(f"[beltrami_jax] symmetry_defect={diagnostics.symmetry_defect:.8e}")
    print(f"[beltrami_jax] amplification_factor={diagnostics.amplification_factor:.8e}")
    if diagnostics.condition_number_2 is not None:
        print(f"[beltrami_jax] condition_number_2={diagnostics.condition_number_2:.8e}")
    print(f"[beltrami_jax] magnetic_energy={float(result.magnetic_energy):.8e}")
    print(f"[beltrami_jax] magnetic_helicity={float(result.magnetic_helicity):.8e}")

    if args.plot:
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(8, 4.5))
        plt.semilogy(np.abs(np.asarray(result.solution)) + 1e-18, label="beltrami_jax")
        plt.semilogy(np.abs(np.asarray(reference.expected_solution)) + 1e-18, "--", label="SPEC")
        plt.xlabel("Packed coefficient index")
        plt.ylabel("|a_i|")
        plt.title(f"Coefficient spectrum: {args.name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.plot, dpi=180)
        print(f"[beltrami_jax] wrote {args.plot}")


if __name__ == "__main__":
    main()
