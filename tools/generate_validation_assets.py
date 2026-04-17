from __future__ import annotations

import argparse
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np

from beltrami_jax.benchmark import benchmark_parameter_scan, benchmark_solve
from beltrami_jax.diagnostics import compare_against_reference, compute_solve_diagnostics
from beltrami_jax.operators import magnetic_energy
from beltrami_jax.reference import load_packaged_reference
from beltrami_jax.solver import solve_from_components, solve_parameter_scan
from beltrami_jax.types import BeltramiLinearSystem


FIXTURE_ORDER = [
    "g1v03l0fi_lvol2",
    "g3v01l0fi_lvol1",
    "g3v02l1fi_lvol1",
    "g3v02l0fr_lu_lvol3",
]

FIXTURE_LABELS = {
    "g1v03l0fi_lvol2": "G1V03L0Fi\ncyl plasma",
    "g3v01l0fi_lvol1": "G3V01L0Fi\ntoroidal plasma",
    "g3v02l1fi_lvol1": "G3V02L1Fi\n3D plasma",
    "g3v02l0fr_lu_lvol3": "G3V02L0Fr\nvacuum",
}

FIXTURE_COLORS = {
    "g1v03l0fi_lvol2": "#0b3954",
    "g3v01l0fi_lvol1": "#087e8b",
    "g3v02l1fi_lvol1": "#bf1363",
    "g3v02l0fr_lu_lvol3": "#f39237",
}

ERROR_COLORS = {
    "operator_relative_error": "#0b3954",
    "rhs_relative_error": "#087e8b",
    "solution_relative_error": "#bf1363",
    "relative_residual_norm": "#f39237",
}


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.titlesize": 13,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "legend.frameon": False,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.15,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="left",
    )


def solved_energy_factory(reference):
    def solved_energy(mu_value: float) -> jax.Array:
        system = BeltramiLinearSystem.from_arraylike(
            d_ma=reference.system.d_ma,
            d_md=reference.system.d_md,
            d_mb=reference.system.d_mb,
            mu=mu_value,
            psi=reference.system.psi,
            label=f"{reference.system.label} energy scan",
        )
        result = solve_from_components(system)
        return magnetic_energy(result.solution, system)

    return solved_energy


def write_validation_panel(
    output_path: Path,
    *,
    references,
    results,
    diagnostics,
    comparisons,
) -> None:
    fig = plt.figure(figsize=(15.5, 11.0), constrained_layout=True)
    axes = fig.subplots(2, 2)

    ax = axes[0, 0]
    all_expected = []
    all_solved = []
    for name in FIXTURE_ORDER:
        expected = np.asarray(references[name].expected_solution)
        solved = np.asarray(results[name].solution)
        all_expected.append(expected)
        all_solved.append(solved)
        ax.scatter(
            expected,
            solved,
            s=8,
            alpha=0.32,
            color=FIXTURE_COLORS[name],
            label=FIXTURE_LABELS[name],
            rasterized=True,
        )
    combined_expected = np.concatenate(all_expected)
    combined_solved = np.concatenate(all_solved)
    limit = 1.1 * max(np.max(np.abs(combined_expected)), np.max(np.abs(combined_solved)))
    line = np.linspace(-limit, limit, 1000)
    ax.plot(line, line, color="#2f2f2f", linewidth=1.2, linestyle="--")
    ax.set_xscale("symlog", linthresh=1e-10)
    ax.set_yscale("symlog", linthresh=1e-10)
    ax.set_xlabel("SPEC dumped coefficient")
    ax.set_ylabel("beltrami_jax coefficient")
    ax.set_title("Coefficient-level agreement across all packaged SPEC fixtures")
    ax.legend(loc="lower right", ncols=2, fontsize=9)
    max_rel = max(item.solution_relative_error for item in comparisons.values())
    max_abs = max(item.max_abs_solution_error for item in comparisons.values())
    ax.text(
        0.02,
        0.98,
        f"max relative solution error = {max_rel:.2e}\nmax absolute solution error = {max_abs:.2e}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#d0d0d0", "boxstyle": "round,pad=0.35"},
    )
    panel_label(ax, "A")

    ax = axes[0, 1]
    x = np.arange(len(FIXTURE_ORDER))
    width = 0.18
    metric_names = [
        "operator_relative_error",
        "rhs_relative_error",
        "solution_relative_error",
        "relative_residual_norm",
    ]
    for idx, metric_name in enumerate(metric_names):
        if metric_name == "relative_residual_norm":
            values = [max(diagnostics[name].relative_residual_norm, 1e-18) for name in FIXTURE_ORDER]
        else:
            values = [max(getattr(comparisons[name], metric_name), 1e-18) for name in FIXTURE_ORDER]
        ax.bar(
            x + (idx - 1.5) * width,
            values,
            width=width,
            color=ERROR_COLORS[metric_name],
            label=metric_name.replace("_", " "),
        )
    ax.set_yscale("log")
    ax.set_xticks(x, [FIXTURE_LABELS[name] for name in FIXTURE_ORDER])
    ax.set_ylabel("relative error")
    ax.set_title("Reconstruction, solve, and residual agreement metrics")
    ax.legend(loc="upper left", fontsize=8)
    panel_label(ax, "B")

    ax = axes[1, 0]
    diag_metrics = [
        ("condition_number_2", "#0b3954", "2-norm condition number"),
        ("amplification_factor", "#087e8b", "solution amplification"),
        ("symmetry_defect", "#bf1363", "symmetry defect"),
    ]
    for idx, (metric_name, color, label) in enumerate(diag_metrics):
        values = []
        for name in FIXTURE_ORDER:
            value = getattr(diagnostics[name], metric_name)
            values.append(max(float(value) if value is not None else np.nan, 1e-18))
        ax.bar(x + (idx - 1) * 0.24, values, width=0.22, color=color, label=label)
    ax.set_yscale("log")
    ax.set_xticks(x, [FIXTURE_LABELS[name] for name in FIXTURE_ORDER])
    ax.set_ylabel("diagnostic magnitude")
    ax.set_title("Operator quality indicators from the solved dense systems")
    ax.legend(loc="upper left", fontsize=8)
    panel_label(ax, "C")

    ax = axes[1, 1]
    reference = references["g3v02l1fi_lvol1"]
    solved_energy = solved_energy_factory(reference)
    mu0 = float(reference.system.mu)
    mu_values = np.linspace(mu0 - 5.0e-4, mu0 + 5.0e-4, 21)
    psi_values = np.repeat(np.asarray(reference.system.psi)[None, :], len(mu_values), axis=0)
    solutions = solve_parameter_scan(reference.system.d_ma, reference.system.d_md, reference.system.d_mb, mu_values, psi_values)
    energies = np.asarray([magnetic_energy(solution, reference.system) for solution in np.asarray(solutions)], dtype=np.float64)
    energy0 = float(solved_energy(mu0))
    gradient = float(jax.grad(solved_energy)(mu0))
    tangent = energy0 + gradient * (mu_values - mu0)
    ax.plot(mu_values, energies, color=FIXTURE_COLORS["g3v02l1fi_lvol1"], linewidth=2.2, label="vmapped solve + energy")
    ax.plot(mu_values, tangent, color="#2f2f2f", linestyle="--", linewidth=1.4, label="autodiff tangent at $\\mu_0$")
    ax.scatter([mu0], [energy0], color="#111111", s=28, zorder=3)
    ax.set_xlabel("$\\mu$")
    ax.set_ylabel("magnetic energy")
    ax.set_title("Autodiff tangent check on the 3D plasma reference fixture")
    ax.legend(loc="best", fontsize=8)
    ax.text(
        0.02,
        0.98,
        f"$\\mu_0$ = {mu0:.4e}\n$dE/d\\mu$ = {gradient:.4e}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#d0d0d0", "boxstyle": "round,pad=0.35"},
    )
    panel_label(ax, "D")

    fig.suptitle(
        "beltrami_jax validation against dumped SPEC linear systems",
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.01,
        "All agreement metrics are computed directly against packaged dense SPEC operator, RHS, and solution dumps. "
        "Conditioning and symmetry indicators are measured on the solved operator without modifying the physical system.",
        ha="center",
        va="top",
        fontsize=10,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_benchmark_panel(
    output_path: Path,
    *,
    solve_benchmarks,
    scan_benchmarks,
) -> None:
    fig = plt.figure(figsize=(15.5, 5.8), constrained_layout=True)
    axes = fig.subplots(1, 2)

    ax = axes[0]
    x = np.arange(len(FIXTURE_ORDER))
    steady_times = [solve_benchmarks[name].steady_state_seconds for name in FIXTURE_ORDER]
    ax.bar(x, steady_times, width=0.5, color=[FIXTURE_COLORS[name] for name in FIXTURE_ORDER], alpha=0.92)
    ax.set_yscale("log")
    ax.set_xticks(x, [FIXTURE_LABELS[name] for name in FIXTURE_ORDER])
    ax.set_ylabel("wall time [s]")
    ax.set_title("Steady-state dense solve timings by fixture")
    unique_compile = {}
    for name in FIXTURE_ORDER:
        size = solve_benchmarks[name].size
        unique_compile.setdefault(size, solve_benchmarks[name].compile_and_solve_seconds)
    compile_text = "\n".join(
        f"first solve for n={size}: {elapsed:.3f}s" for size, elapsed in sorted(unique_compile.items())
    )
    ax.text(
        0.02,
        0.98,
        compile_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#d0d0d0", "boxstyle": "round,pad=0.35"},
    )
    for idx, name in enumerate(FIXTURE_ORDER):
        ax.text(
            idx,
            steady_times[idx] * 1.12,
            f"n={solve_benchmarks[name].size}",
            va="bottom",
            ha="center",
            fontsize=8,
            color="#404040",
        )
    panel_label(ax, "E")

    ax = axes[1]
    batch_sizes = [item.batch_size for item in scan_benchmarks]
    per_system = [item.per_system_seconds for item in scan_benchmarks]
    compile_times = [item.compile_and_solve_seconds for item in scan_benchmarks]
    scalar_baseline = per_system[0]
    ax.plot(batch_sizes, per_system, marker="o", color="#0b3954", linewidth=2.2, label="vmapped scan (per system)")
    ax.plot(batch_sizes, compile_times, marker="s", color="#f39237", linewidth=1.8, linestyle="--", label="first call total time")
    ax.axhline(scalar_baseline, color="#2f2f2f", linestyle=":", linewidth=1.4, label="batch size 1 baseline")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("batch size")
    ax.set_ylabel("wall time [s]")
    ax.set_title("Batched parameter-scan scaling on the 3D plasma fixture")
    ax.legend(loc="best", fontsize=8)
    panel_label(ax, "F")

    fig.suptitle(
        "beltrami_jax benchmark summary",
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.02,
        "Timings were measured on the packaged SPEC regression fixtures with JAX x64 enabled. "
        "The batch-scan panel uses the 3D fixed-boundary plasma fixture `g3v02l1fi_lvol1`.",
        ha="center",
        va="top",
        fontsize=10,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate polished validation and benchmark figures from packaged fixtures.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/_static"),
        help="Directory where the figure assets will be written.",
    )
    parser.add_argument("--repeats", type=int, default=2, help="Number of steady-state repetitions per benchmark.")
    args = parser.parse_args()

    configure_matplotlib()

    references = {}
    results = {}
    diagnostics = {}
    comparisons = {}
    solve_benchmarks = {}

    for name in FIXTURE_ORDER:
        reference = load_packaged_reference(name)
        solve_benchmarks[name] = benchmark_solve(reference, repeats=args.repeats)
        result = solve_from_components(reference.system)
        references[name] = reference
        results[name] = result
        diagnostics[name] = compute_solve_diagnostics(result, include_condition_number=True)
        comparisons[name] = compare_against_reference(reference, result)
        print(
            "[beltrami_jax] "
            f"fixture={name} size={reference.system.size} "
            f"solution_relative_error={comparisons[name].solution_relative_error:.3e} "
            f"condition_number_2={diagnostics[name].condition_number_2:.3e}"
        )

    scan_benchmarks = benchmark_parameter_scan(
        references["g3v02l1fi_lvol1"],
        batch_sizes=(1, 2, 4, 8, 16),
        repeats=args.repeats,
        relative_span=1.0e-4,
    )
    for item in scan_benchmarks:
        print(
            "[beltrami_jax] "
            f"batch_size={item.batch_size} per_system_seconds={item.per_system_seconds:.3e}"
        )

    validation_path = args.output_dir / "validation_panel.png"
    benchmark_path = args.output_dir / "benchmark_panel.png"
    write_validation_panel(validation_path, references=references, results=results, diagnostics=diagnostics, comparisons=comparisons)
    write_benchmark_panel(benchmark_path, solve_benchmarks=solve_benchmarks, scan_benchmarks=scan_benchmarks)
    print(f"[beltrami_jax] wrote {validation_path}")
    print(f"[beltrami_jax] wrote {benchmark_path}")


if __name__ == "__main__":
    main()
