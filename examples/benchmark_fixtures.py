from __future__ import annotations

import argparse

from beltrami_jax.benchmark import benchmark_parameter_scan, benchmark_solve
from beltrami_jax.reference import load_packaged_reference


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark dense solves and batched parameter scans.")
    parser.add_argument("--name", default="g1v03l0fi_lvol2", help="Non-vacuum packaged fixture name.")
    parser.add_argument("--repeats", type=int, default=1, help="Number of steady-state timing repetitions.")
    args = parser.parse_args()

    reference = load_packaged_reference(args.name)
    if reference.system.is_vacuum:
        raise ValueError("benchmark_fixtures.py requires a non-vacuum packaged fixture")

    solve_benchmark = benchmark_solve(reference, repeats=args.repeats)
    print(
        "[beltrami_jax] "
        f"fixture={args.name} compile_and_solve_seconds={solve_benchmark.compile_and_solve_seconds:.6f} "
        f"steady_state_seconds={solve_benchmark.steady_state_seconds:.6f}"
    )

    for item in benchmark_parameter_scan(reference, batch_sizes=(1, 4, 8), repeats=args.repeats, relative_span=0.01):
        print(
            "[beltrami_jax] "
            f"batch_size={item.batch_size} compile_and_solve_seconds={item.compile_and_solve_seconds:.6f} "
            f"steady_state_seconds={item.steady_state_seconds:.6f} per_system_seconds={item.per_system_seconds:.6f}"
        )


if __name__ == "__main__":
    main()
