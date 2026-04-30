#!/usr/bin/env python3
"""Export SPECTRE Beltrami linear systems to ``.npz`` fixtures.

Run this script from a SPECTRE environment, not from the core ``beltrami_jax``
environment. It intentionally imports only SPECTRE and NumPy.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np
import spectre4py_wrap as spectre_wrap
from spectre import SPECTRE, force_modes_speclike, get_xinit_specwrap, update_bnorm


def _case_label(input_toml: Path, explicit_label: str | None) -> str:
    if explicit_label:
        return explicit_label
    return input_toml.parent.name or input_toml.stem


def _finalize_spectre_state(test: SPECTRE) -> None:
    xin = get_xinit_specwrap(test)
    force_modes_speclike(xin, test)
    if test.input_list_mod.lfreebound:
        for iteration in range(test.input_list_mod.mfreeits):
            update_bnorm(test, first_free_bound=(iteration == 0), print_info=False)
            force_modes_speclike(xin, test)


def _set_region_flags(test: SPECTRE, volume_index: int) -> None:
    test.allglobal_mod.lcoordinatesingularity = (
        test.input_list_mod.igeometry != 1 and volume_index == 1
    )
    test.allglobal_mod.lplasmaregion = volume_index <= test.input_list_mod.nvol
    test.allglobal_mod.lvacuumregion = not test.allglobal_mod.lplasmaregion
    test.allglobal_mod.ivol = volume_index


def _volume_parameter(test: SPECTRE, volume_index: int) -> float:
    if test.allglobal_mod.lvacuumregion:
        return float(test.allglobal_mod.dtflux[volume_index - 1])
    if test.input_list_mod.lconstraint == -2:
        return float(test.allglobal_mod.dtflux[volume_index - 1])
    return float(test.input_list_mod.mu[volume_index - 1])


def _assemble_matrix_rhs(test: SPECTRE, volume_index: int) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    n_dof = int(test.allglobal_mod.nadof[volume_index - 1])
    d_ma = np.asarray(test.allglobal_mod.dma)[1 : n_dof + 1, 1 : n_dof + 1].astype(np.float64, copy=False)
    d_md = np.asarray(test.allglobal_mod.dmd)[1 : n_dof + 1, 1 : n_dof + 1].astype(np.float64, copy=False)
    d_mb = np.asarray(test.allglobal_mod.dmb)[1 : n_dof + 1, 1:3].astype(np.float64, copy=False)
    d_mg = np.asarray(test.allglobal_mod.dmg)[1 : n_dof + 1].astype(np.float64, copy=False)
    dpsi = np.asarray(
        [
            float(test.allglobal_mod.dtflux[volume_index - 1]),
            float(test.allglobal_mod.dpflux[volume_index - 1]),
        ],
        dtype=np.float64,
    )

    include_d_mg_in_rhs = bool(
        test.allglobal_mod.lvacuumregion
        or (
            test.allglobal_mod.lcoordinatesingularity
            and test.input_list_mod.lconstraint == -2
        )
    )

    if test.allglobal_mod.lvacuumregion:
        mu = 0.0
        matrix = d_ma
        rhs = -d_mg - d_mb @ dpsi
    else:
        mu = float(test.input_list_mod.mu[volume_index - 1])
        matrix = d_ma - mu * d_md
        rhs = -d_mb @ dpsi
        if include_d_mg_in_rhs:
            rhs = -d_mg + rhs
    return matrix, rhs, mu, dpsi


def _export_one_volume(test: SPECTRE, input_toml: Path, output_dir: Path, case_label: str, volume_index: int) -> Path:
    n_dof = int(test.allglobal_mod.nadof[volume_index - 1])
    _set_region_flags(test, volume_index)

    try:
        spectre_wrap.memory_mod.allocate_geometry_matrices(volume_index, True, test.allglobal_mod.ntz)
        spectre_wrap.memory_mod.allocate_beltrami_matrices(volume_index, True)
        spectre_wrap.intghs_mod.intghs_workspace_init(volume_index, test.allglobal_mod.ntz)

        spectre_wrap.packing_mod.packab(
            "P",
            volume_index,
            n_dof,
            test.allglobal_mod.solution[:, 1],
            0,
        )
        spectre_wrap.chebyshev_mod.volume_integrate_chebyshev(
            test.allglobal_mod.iquad[volume_index - 1],
            test.allglobal_mod.mn,
            volume_index,
            test.input_list_mod.lrad[volume_index - 1],
            test.allglobal_mod.nt,
            test.allglobal_mod.nz,
        )
        spectre_wrap.matrices_mod.matrix(
            volume_index,
            test.allglobal_mod.mn,
            test.input_list_mod.lrad[volume_index - 1],
        )

        matrix, rhs, mu, dpsi = _assemble_matrix_rhs(test, volume_index)
        x_dof = np.asarray([_volume_parameter(test, volume_index) + test.allglobal_mod.xoffset], dtype=np.float64)
        f_dof = np.zeros((1,), dtype=np.float64, order="F")
        d_dof = np.zeros((1, 1), dtype=np.float64, order="F")
        iflag = spectre_wrap.beltrami_solver_mod.solve_beltrami_system(1, x_dof, f_dof, d_dof, 1)
        solution = np.asarray(test.allglobal_mod.solution)[:n_dof, 1].astype(np.float64, copy=True)
        d_ma = np.asarray(test.allglobal_mod.dma)[1 : n_dof + 1, 1 : n_dof + 1].astype(np.float64, copy=True)
        d_md = np.asarray(test.allglobal_mod.dmd)[1 : n_dof + 1, 1 : n_dof + 1].astype(np.float64, copy=True)
        d_mb = np.asarray(test.allglobal_mod.dmb)[1 : n_dof + 1, 1:3].astype(np.float64, copy=True)
        d_mg = np.asarray(test.allglobal_mod.dmg)[1 : n_dof + 1].astype(np.float64, copy=True)
    finally:
        try:
            spectre_wrap.intghs_mod.intghs_workspace_destroy()
        finally:
            try:
                spectre_wrap.memory_mod.deallocate_beltrami_matrices(True)
            finally:
                spectre_wrap.memory_mod.deallocate_geometry_matrices(True)

    residual_norm = float(np.linalg.norm(matrix @ solution - rhs))
    rhs_norm = float(np.linalg.norm(rhs))
    relative_residual_norm = residual_norm if rhs_norm == 0.0 else residual_norm / rhs_norm

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{case_label}_lvol{volume_index}.npz"
    np.savez_compressed(
        output_path,
        d_ma=d_ma,
        d_md=d_md,
        d_mb=d_mb,
        d_mg=d_mg,
        matrix=matrix,
        rhs=rhs,
        solution=solution,
        mu=np.asarray(mu, dtype=np.float64),
        psi=dpsi,
        is_vacuum=np.asarray(int(test.allglobal_mod.lvacuumregion), dtype=np.int32),
        include_d_mg_in_rhs=np.asarray(
            int(
                test.allglobal_mod.lvacuumregion
                or (
                    test.allglobal_mod.lcoordinatesingularity
                    and test.input_list_mod.lconstraint == -2
                )
            ),
            dtype=np.int32,
        ),
        coordinate_singularity=np.asarray(int(test.allglobal_mod.lcoordinatesingularity), dtype=np.int32),
        plasma_region=np.asarray(int(test.allglobal_mod.lplasmaregion), dtype=np.int32),
        volume_index=np.asarray(volume_index, dtype=np.int32),
        n_dof=np.asarray(n_dof, dtype=np.int32),
        lrad=np.asarray(int(test.input_list_mod.lrad[volume_index - 1]), dtype=np.int32),
        mn=np.asarray(int(test.allglobal_mod.mn), dtype=np.int32),
        nvol=np.asarray(int(test.input_list_mod.nvol), dtype=np.int32),
        mvol=np.asarray(int(test.allglobal_mod.mvol), dtype=np.int32),
        mpol=np.asarray(int(test.input_list_mod.mpol), dtype=np.int32),
        ntor=np.asarray(int(test.input_list_mod.ntor), dtype=np.int32),
        nfp=np.asarray(int(test.input_list_mod.nfp), dtype=np.int32),
        lconstraint=np.asarray(int(test.input_list_mod.lconstraint), dtype=np.int32),
        iflag=np.asarray(int(iflag), dtype=np.int32),
        residual_norm=np.asarray(residual_norm, dtype=np.float64),
        relative_residual_norm=np.asarray(relative_residual_norm, dtype=np.float64),
        label=np.asarray(f"SPECTRE {case_label} lvol={volume_index}"),
        case_label=np.asarray(case_label),
        source=np.asarray(str(input_toml)),
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_toml", type=Path, help="SPECTRE input.toml path")
    parser.add_argument("output_dir", type=Path, help="Directory for per-volume .npz exports")
    parser.add_argument("--case-label", default=None, help="Label used in output filenames")
    parser.add_argument("--volume-index", type=int, default=None, help="Export only this one-based SPECTRE volume")
    args = parser.parse_args()

    input_toml = args.input_toml.resolve()
    case_label = _case_label(input_toml, args.case_label)

    if args.volume_index is None:
        test = SPECTRE.from_input_file(str(input_toml), verbose=False)
        volume_indices = tuple(range(1, int(test.allglobal_mod.mvol) + 1))
        for volume_index in volume_indices:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                str(input_toml),
                str(args.output_dir),
                "--case-label",
                case_label,
                "--volume-index",
                str(volume_index),
            ]
            subprocess.run(command, check=True)
        print(f"[beltrami_jax] exported {len(volume_indices)} SPECTRE Beltrami systems for {case_label}")
        return

    test = SPECTRE.from_input_file(str(input_toml), verbose=False)
    _finalize_spectre_state(test)

    output_path = _export_one_volume(test, input_toml, args.output_dir, case_label, args.volume_index)
    print(f"[beltrami_jax] wrote {output_path}")


if __name__ == "__main__":
    main()
