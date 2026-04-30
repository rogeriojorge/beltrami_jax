#!/usr/bin/env python
"""Export SPECTRE vector-potential coefficients to a NumPy archive.

Run this script with the SPECTRE Python environment, not necessarily the
``beltrami_jax`` environment. It intentionally imports only SPECTRE and NumPy so
the exported arrays can be compared later by ``beltrami_jax``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def reported_force_norm(force_modes: np.ndarray, mask: np.ndarray) -> float:
    if np.shape(mask) == np.shape(force_modes):
        return float(np.linalg.norm(force_modes[mask]))
    return float(np.linalg.norm(force_modes))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_toml", type=Path, help="SPECTRE input TOML file")
    parser.add_argument("output_npz", type=Path, help="Output vector-potential archive")
    parser.add_argument(
        "--free-boundary-iterations",
        type=int,
        default=None,
        help=(
            "Number of fixed-point updates for free-boundary SPECTRE cases. "
            "Defaults to input_list_mod.mfreeits, which matches SPECTRE reference.h5 generation."
        ),
    )
    parser.add_argument("--verbose", action="store_true", help="Enable SPECTRE verbose output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)

    from spectre import (
        SPECTRE,
        force_modes_speclike,
        get_fmn_force_mask,
        get_vec_pot_flat,
        get_xinit_specwrap,
        update_bnorm,
    )

    print(f"[export] loading SPECTRE input: {args.input_toml}")
    test = SPECTRE.from_input_file(str(args.input_toml), verbose=args.verbose)
    xin = get_xinit_specwrap(test)
    mask = get_fmn_force_mask(test)

    free_boundary = bool(getattr(test.input_list_mod, "lfreebound", False))
    if free_boundary:
        free_boundary_iterations = (
            int(test.input_list_mod.mfreeits)
            if args.free_boundary_iterations is None
            else args.free_boundary_iterations
        )
        print(f"[export] running {free_boundary_iterations} free-boundary bnorm updates")
        for iteration in range(free_boundary_iterations):
            force_modes = force_modes_speclike(xin, test)
            print(
                f"[export]   iteration {iteration + 1}: "
                f"force_norm={reported_force_norm(force_modes, mask):.6e}"
            )
            update_bnorm(test, first_free_bound=iteration == 0, print_info=args.verbose)

    force_modes = force_modes_speclike(xin, test)
    ate, aze, ato, azo = get_vec_pot_flat(test)
    force_norm = reported_force_norm(force_modes, mask)
    np.savez_compressed(
        args.output_npz,
        ate=np.asarray(ate, dtype=float),
        aze=np.asarray(aze, dtype=float),
        ato=np.asarray(ato, dtype=float),
        azo=np.asarray(azo, dtype=float),
        force_modes=np.asarray(force_modes, dtype=float),
        force_mask=np.asarray(mask, dtype=bool),
        input_toml=np.asarray(str(args.input_toml)),
        free_boundary=np.asarray(free_boundary),
    )
    print(f"[export] wrote {args.output_npz}")
    print(f"[export] coefficient shape: {ate.shape}")
    print(f"[export] force norm: {force_norm:.6e}")


if __name__ == "__main__":
    main()
