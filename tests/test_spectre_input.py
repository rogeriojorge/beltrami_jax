from __future__ import annotations

import json

import pytest

from beltrami_jax import load_spectre_input_toml


def test_load_spectre_input_summary(tmp_path):
    path = tmp_path / "input.toml"
    path.write_text(
        """
[physics]
igeometry = 3
nfp = 5
nvol = 2
mpol = 4
ntor = 2
lconstraint = 3
lfreebound = true
enforce_stell_sym = true
lrad = [8, 4]
tflux = [0.1, 0.2]
pflux = [0.0, 0.03]
mu = [0.01, 0.02]
helicity = [1.0e-3, 2.0e-3]
iota = [0.25, 0.26, 0.27]

[physics.rbc]
"(0, 0)" = 10.0
"(1, 0)" = 1.0
"(1, 1)" = 0.25

[physics.zbs]
"(1, 0)" = -1.0

[physics.rbs]

[physics.zbc]

[numeric]
lsparse = 0

[global]
mfreeits = 2

[local]

[diagnostics]
""",
        encoding="utf-8",
    )

    summary = load_spectre_input_toml(path)
    assert summary.nvol == 2
    assert summary.nfp == 5
    assert summary.igeometry == 3
    assert summary.lrad == (8, 4)
    assert summary.radial_size == 14
    assert summary.packed_volume_count == 2
    assert summary.is_free_boundary is True
    assert summary.free_boundary_iterations == 2
    assert summary.rbc[(1, 1)] == 0.25
    assert summary.zbs[(1, 0)] == -1.0

    as_dict = summary.as_dict()
    assert as_dict["boundary_mode_counts"] == {"rbc": 3, "zbs": 1, "rbs": 0, "zbc": 0}
    json.dumps(as_dict)


def test_spectre_input_validation_rejects_inconsistent_lengths(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text(
        """
[physics]
nvol = 2
lrad = [4]
tflux = [0.1, 0.2]
pflux = [0.0, 0.03]
mu = [0.01, 0.02]
helicity = [1.0e-3, 2.0e-3]

[numeric]
[global]
[local]
[diagnostics]
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="lrad length"):
        load_spectre_input_toml(path)


def test_spectre_input_accepts_free_boundary_extra_packed_block(tmp_path):
    path = tmp_path / "free.toml"
    path.write_text(
        """
[physics]
nvol = 2
lfreebound = 1
lrad = [4, 5, 6]
tflux = [0.1, 0.2, 0.3]
pflux = [0.0, 0.03, 0.04]
mu = [0.01, 0.02]
helicity = [1.0e-3, 2.0e-3, 3.0e-3]

[numeric]
[global]
mfreeits = 2
[local]
[diagnostics]
""",
        encoding="utf-8",
    )

    summary = load_spectre_input_toml(path)
    assert summary.is_free_boundary is True
    assert summary.packed_volume_count == 3
    assert summary.radial_size == 18


def test_spectre_input_rejects_invalid_mode_key(tmp_path):
    path = tmp_path / "bad_mode.toml"
    path.write_text(
        """
[physics]
nvol = 1
lrad = [4]
tflux = [0.1]
pflux = [0.0]
mu = [0.01]
helicity = [1.0e-3]

[physics.rbc]
"bad" = 1.0

[numeric]
[global]
[local]
[diagnostics]
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid SPECTRE Fourier mode key"):
        load_spectre_input_toml(path)
