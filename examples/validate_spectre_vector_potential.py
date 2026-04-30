"""Validate SPECTRE vector-potential coefficient exports.

This script demonstrates the SPECTRE-facing IO layer in ``beltrami_jax``. It has
two modes:

1. If local SPECTRE reference files and exported candidate ``.npz`` files exist,
   it compares fresh SPECTRE vector-potential coefficients against the committed
   SPECTRE ``reference.h5`` files.
2. Otherwise, it creates a compact synthetic SPECTRE-layout HDF5 file and shows
   the same loading, comparison, output, and plotting workflow without requiring
   SPECTRE to be installed.

The script is intentionally standalone: edit the parameters below, run it, and
inspect ``examples/_generated/validate_spectre_vector_potential``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from beltrami_jax import (
    SpectreVectorPotential,
    compare_vector_potentials,
    list_packaged_spectre_cases,
    load_packaged_spectre_case,
    load_spectre_reference_h5,
    load_spectre_input_toml,
    load_spectre_vector_potential_h5,
    load_spectre_vector_potential_npz,
    save_spectre_vector_potential_npz,
)

SPECTRE_ROOT = Path("/Users/rogerio/local/spectre")
CASE_LABELS = ("G2V32L1Fi", "G3V3L3Fi", "G3V3L2Fi_stability", "G3V8L3Free")
EXPORTED_CANDIDATE_DIR = Path("examples/_generated/spectre_vecpot_exports")
OUTPUT_DIR = Path("examples/_generated/validate_spectre_vector_potential")

print("[beltrami_jax] validating SPECTRE vector-potential coefficient IO")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

h5py_error = None
try:
    import h5py
except ImportError as exc:
    h5py = None
    h5py_error = exc

if h5py is None:
    raise ImportError("This example needs h5py. Install with `python -m pip install -e '.[dev]'`.") from h5py_error

real_cases = []
for label in CASE_LABELS:
    reference_h5 = SPECTRE_ROOT / "tests" / "compare" / label / "reference.h5"
    candidate_npz = EXPORTED_CANDIDATE_DIR / f"{label}.npz"
    if reference_h5.exists() and candidate_npz.exists():
        real_cases.append((label, reference_h5, candidate_npz))

comparisons = []
if real_cases:
    print(f"[beltrami_jax] found {len(real_cases)} real SPECTRE comparison cases")
    for label, reference_h5, candidate_npz in real_cases:
        input_summary = load_spectre_input_toml(reference_h5.with_name("input.toml"))
        reference = load_spectre_reference_h5(reference_h5).vector_potential
        candidate = load_spectre_vector_potential_npz(candidate_npz)
        comparison = compare_vector_potentials(candidate, reference, label=label)
        comparisons.append(comparison.as_dict())
        print(
            f"[beltrami_jax] {label}: shape={comparison.shape}, "
            f"nvol={input_summary.nvol}, lrad={input_summary.lrad}, "
            f"global_relative_error={comparison.global_relative_error:.3e}"
        )
elif list_packaged_spectre_cases():
    print("[beltrami_jax] local SPECTRE exports not found; using packaged SPECTRE fixtures")
    for label in list_packaged_spectre_cases():
        packaged = load_packaged_spectre_case(label)
        comparison = packaged.comparison
        comparisons.append(comparison.as_dict())
        print(
            f"[beltrami_jax] {label}: shape={comparison.shape}, "
            f"nvol={packaged.input_summary.nvol}, lrad={packaged.input_summary.lrad}, "
            f"global_relative_error={comparison.global_relative_error:.3e}"
        )
else:
    print("[beltrami_jax] real SPECTRE exports not found; creating a synthetic SPECTRE-layout file")
    synthetic_h5 = OUTPUT_DIR / "synthetic_reference.h5"
    radial_size = 5
    mode_count = 4
    base = np.linspace(-0.5, 0.5, radial_size * mode_count).reshape(radial_size, mode_count)
    reference = SpectreVectorPotential(
        ate=base,
        aze=base**2,
        ato=np.sin(base),
        azo=np.cos(base) - 1.0,
        source="synthetic",
    )
    with h5py.File(synthetic_h5, "w") as handle:
        group = handle.create_group("vector_potential")
        group.create_dataset("Ate", data=reference.ate.T)
        group.create_dataset("Aze", data=reference.aze.T)
        group.create_dataset("Ato", data=reference.ato.T)
        group.create_dataset("Azo", data=reference.azo.T)
        output = handle.create_group("output")
        output.create_dataset("force_final", data=np.array([0.0]))

    loaded = load_spectre_vector_potential_h5(synthetic_h5)
    candidate_npz = OUTPUT_DIR / "synthetic_candidate.npz"
    save_spectre_vector_potential_npz(candidate_npz, loaded)
    candidate = load_spectre_vector_potential_npz(candidate_npz)
    comparison = compare_vector_potentials(candidate, reference, label="synthetic")
    comparisons.append(comparison.as_dict())
    print(
        f"[beltrami_jax] synthetic: shape={comparison.shape}, "
        f"global_relative_error={comparison.global_relative_error:.3e}"
    )

summary_path = OUTPUT_DIR / "summary.json"
summary_path.write_text(json.dumps(comparisons, indent=2), encoding="utf-8")
print(f"[beltrami_jax] wrote {summary_path}")

labels = [item["label"] for item in comparisons]
global_errors = [max(float(item["global_relative_error"]), 1.0e-18) for item in comparisons]
max_errors = [max(float(item["global_max_abs_error"]), 1.0e-18) for item in comparisons]

fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
axes[0].bar(labels, global_errors, color="#264653")
axes[0].set_yscale("log")
axes[0].set_ylabel("global relative error")
axes[0].set_title("Vector-potential parity")
axes[0].tick_params(axis="x", rotation=20)

axes[1].bar(labels, max_errors, color="#E76F51")
axes[1].set_yscale("log")
axes[1].set_ylabel("max absolute coefficient error")
axes[1].set_title("Worst coefficient mismatch")
axes[1].tick_params(axis="x", rotation=20)

figure_path = OUTPUT_DIR / "vector_potential_validation.png"
fig.savefig(figure_path, dpi=180)
plt.close(fig)
print(f"[beltrami_jax] wrote {figure_path}")
