from __future__ import annotations

import numpy as np
import pytest

from beltrami_jax import (
    SpectreVectorPotential,
    compare_vector_potentials,
    load_spectre_reference_h5,
    load_spectre_vector_potential_h5,
    load_spectre_vector_potential_npz,
    save_spectre_vector_potential_npz,
)

h5py = pytest.importorskip("h5py")


def _sample_vector_potential() -> SpectreVectorPotential:
    base = np.arange(15, dtype=float).reshape(5, 3)
    return SpectreVectorPotential(
        ate=base + 1.0,
        aze=base + 2.0,
        ato=np.zeros_like(base),
        azo=base - 3.0,
        source="synthetic",
    )


def test_h5_loader_transposes_spectre_layout(tmp_path):
    vector_potential = _sample_vector_potential()
    path = tmp_path / "reference.h5"

    with h5py.File(path, "w") as handle:
        group = handle.create_group("vector_potential")
        for h5_name, array in {
            "Ate": vector_potential.ate,
            "Aze": vector_potential.aze,
            "Ato": vector_potential.ato,
            "Azo": vector_potential.azo,
        }.items():
            group.create_dataset(h5_name, data=array.T)
        output = handle.create_group("output")
        output.create_dataset("force_final", data=np.array([1.0, 2.0]))
        output.create_dataset("force_final_grad", data=np.array([[3.0, 4.0]]))

    loaded = load_spectre_vector_potential_h5(path)
    assert loaded.shape == (5, 3)
    np.testing.assert_allclose(loaded.ate, vector_potential.ate)
    np.testing.assert_allclose(loaded.azo, vector_potential.azo)

    reference = load_spectre_reference_h5(path)
    assert reference.source == str(path)
    np.testing.assert_allclose(reference.force_final, np.array([1.0, 2.0]))
    np.testing.assert_allclose(reference.force_final_grad, np.array([[3.0, 4.0]]))


def test_npz_roundtrip_and_component_comparison(tmp_path):
    reference = _sample_vector_potential()
    archive = tmp_path / "vecpot.npz"
    save_spectre_vector_potential_npz(archive, reference, case=np.asarray("unit"))

    loaded = load_spectre_vector_potential_npz(archive)
    exact = compare_vector_potentials(loaded, reference, label="exact")
    assert exact.shape == (5, 3)
    assert exact.global_relative_error == 0.0
    assert exact.global_max_abs_error == 0.0
    assert exact.as_dict()["label"] == "exact"

    perturbed = SpectreVectorPotential(
        ate=reference.ate + 1.0e-6,
        aze=reference.aze,
        ato=reference.ato,
        azo=reference.azo,
    )
    comparison = compare_vector_potentials(perturbed, reference, label="perturbed")
    assert comparison.component_relative_errors["ate"] > 0.0
    assert comparison.component_relative_errors["ato"] == 0.0
    assert comparison.global_max_abs_error == pytest.approx(1.0e-6)


def test_split_by_lrad_checks_packed_volume_widths():
    vector_potential = _sample_vector_potential()
    volumes = vector_potential.split_by_lrad([1, 2])
    assert len(volumes) == 2
    assert volumes[0].shape == (2, 3)
    assert volumes[1].shape == (3, 3)
    np.testing.assert_allclose(volumes[1].aze, vector_potential.aze[2:5])

    with pytest.raises(ValueError, match="does not match"):
        vector_potential.split_by_lrad([0, 0])


def test_loader_rejects_missing_dataset(tmp_path):
    path = tmp_path / "broken.h5"
    with h5py.File(path, "w") as handle:
        group = handle.create_group("vector_potential")
        group.create_dataset("Ate", data=np.zeros((3, 5)))
        group.create_dataset("Aze", data=np.zeros((3, 5)))
        group.create_dataset("Ato", data=np.zeros((3, 5)))

    with pytest.raises(KeyError, match="vector_potential/Azo"):
        load_spectre_vector_potential_h5(path)


def test_vector_potential_shape_validation():
    with pytest.raises(ValueError, match="same shape"):
        SpectreVectorPotential(
            ate=np.zeros((2, 3)),
            aze=np.zeros((2, 3)),
            ato=np.zeros((1, 3)),
            azo=np.zeros((2, 3)),
        )

    with pytest.raises(ValueError, match="2D"):
        SpectreVectorPotential(
            ate=np.zeros(3),
            aze=np.zeros((2, 3)),
            ato=np.zeros((2, 3)),
            azo=np.zeros((2, 3)),
        )
