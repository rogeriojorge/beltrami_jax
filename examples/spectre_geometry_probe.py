from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from beltrami_jax import (
    build_spectre_interface_geometry,
    evaluate_spectre_volume_coordinates,
    interpolate_spectre_volume_geometry,
    load_packaged_spectre_case,
)


CASE = "G3V8L3Free"
PROBE_VOLUME = 2
PROBE_S = 0.0
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "examples" / "_generated" / "spectre_geometry_probe"
SUMMARY_PATH = OUTPUT_DIR / "spectre_geometry_summary.json"
FIGURE_PATH = OUTPUT_DIR / "spectre_geometry_probe.png"


print("[beltrami_jax] running SPECTRE interface-geometry probe")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

case = load_packaged_spectre_case(CASE)
geometry = build_spectre_interface_geometry(case.input_summary)
theta_curve = np.linspace(0.0, 2.0 * np.pi, 240, endpoint=True)
zeta_curve = np.asarray([0.0])

interface_curves = []
for lvol in range(1, geometry.interface_count + 1):
    volume = interpolate_spectre_volume_geometry(geometry, lvol=lvol, s=1.0)
    coordinates = evaluate_spectre_volume_coordinates(volume, theta=theta_curve, zeta=zeta_curve)
    interface_curves.append((np.asarray(coordinates.r[:, 0]), np.asarray(coordinates.z[:, 0])))

theta_grid = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
zeta_grid = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
probe_volume = interpolate_spectre_volume_geometry(geometry, lvol=PROBE_VOLUME, s=PROBE_S)
probe_coordinates = evaluate_spectre_volume_coordinates(probe_volume, theta=theta_grid, zeta=zeta_grid)
jacobian = np.asarray(probe_coordinates.jacobian)
metric_trace = np.trace(np.asarray(probe_coordinates.metric), axis1=-2, axis2=-1)

summary = {
    "case": CASE,
    "igeometry": case.input_summary.igeometry,
    "nfp": case.input_summary.nfp,
    "interfaces": geometry.interface_count,
    "modes": geometry.mode_count,
    "probe_volume": PROBE_VOLUME,
    "probe_s": PROBE_S,
    "jacobian_min": float(np.min(jacobian)),
    "jacobian_max": float(np.max(jacobian)),
    "metric_trace_min": float(np.min(metric_trace)),
    "metric_trace_max": float(np.max(metric_trace)),
}
SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), constrained_layout=True)
colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(interface_curves)))
for index, ((r_values, z_values), color) in enumerate(zip(interface_curves, colors), start=1):
    linewidth = 2.2 if index in (1, len(interface_curves)) else 1.0
    axes[0].plot(r_values, z_values, color=color, linewidth=linewidth)
axes[0].set_aspect("equal", adjustable="box")
axes[0].set_title(f"{CASE} interfaces")
axes[0].set_xlabel("R")
axes[0].set_ylabel("Z")

image0 = axes[1].imshow(
    jacobian,
    origin="lower",
    aspect="auto",
    extent=(0.0, 2.0 * np.pi, 0.0, 2.0 * np.pi),
    cmap="cividis",
)
axes[1].set_title(f"Jacobian, lvol={PROBE_VOLUME}, s={PROBE_S:g}")
axes[1].set_xlabel("zeta")
axes[1].set_ylabel("theta")
figure.colorbar(image0, ax=axes[1], shrink=0.85)

image1 = axes[2].imshow(
    metric_trace,
    origin="lower",
    aspect="auto",
    extent=(0.0, 2.0 * np.pi, 0.0, 2.0 * np.pi),
    cmap="magma",
)
axes[2].set_title("metric trace")
axes[2].set_xlabel("zeta")
axes[2].set_ylabel("theta")
figure.colorbar(image1, ax=axes[2], shrink=0.85)

figure.savefig(FIGURE_PATH, dpi=180)
plt.close(figure)

print(f"[beltrami_jax] case={CASE} interfaces={geometry.interface_count} modes={geometry.mode_count}")
print(f"[beltrami_jax] jacobian_range=({summary['jacobian_min']:.6e}, {summary['jacobian_max']:.6e})")
print(f"[beltrami_jax] metric_trace_range=({summary['metric_trace_min']:.6e}, {summary['metric_trace_max']:.6e})")
print(f"[beltrami_jax] wrote {SUMMARY_PATH}")
print(f"[beltrami_jax] wrote {FIGURE_PATH}")
