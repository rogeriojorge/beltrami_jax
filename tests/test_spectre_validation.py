from __future__ import annotations

import json
import subprocess
import sys

import pytest

from beltrami_jax import (
    list_packaged_spectre_cases,
    load_all_packaged_spectre_cases,
    load_packaged_spectre_case,
    packaged_spectre_case_paths,
)

pytest.importorskip("h5py")


def test_packaged_spectre_cases_have_machine_precision_coefficient_parity():
    cases = load_all_packaged_spectre_cases()
    assert [case.label for case in cases] == list(list_packaged_spectre_cases())

    worst_global_relative_error = max(case.comparison.global_relative_error for case in cases)
    assert worst_global_relative_error < 2.0e-14

    for case in cases:
        assert case.vector_potential_shape[0] == case.input_summary.radial_size
        assert case.layout.shape == case.vector_potential_shape
        assert case.comparison.global_max_abs_error < 2.0e-15
        assert case.comparison.component_relative_errors["ato"] == 0.0
        assert case.comparison.component_relative_errors["azo"] == 0.0


def test_packaged_spectre_case_paths_and_unknown_label():
    paths = packaged_spectre_case_paths("G2V32L1Fi")
    assert paths["input_toml"].name == "input.toml"
    assert paths["reference_h5"].name == "reference.h5"
    assert paths["candidate_npz"].name == "fresh_spectre_export.npz"

    with pytest.raises(KeyError, match="unknown packaged SPECTRE case"):
        load_packaged_spectre_case("missing")


def test_packaged_spectre_plot_generator(tmp_path):
    output = tmp_path / "spectre_vecpot_parity.png"
    summary = tmp_path / "summary.json"
    completed = subprocess.run(
        [
            sys.executable,
            "tools/generate_spectre_validation_assets.py",
            "--use-packaged",
            "--output",
            str(output),
            "--summary",
            str(summary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "worst global" in completed.stdout
    assert output.exists()
    rows = json.loads(summary.read_text(encoding="utf-8"))
    assert len(rows) == 4
    assert max(row["global_relative_error"] for row in rows) < 2.0e-14
