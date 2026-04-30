from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .spectre_input import ModeTable, SpectreInputSummary
from .spectre_pack import spectre_fourier_modes


@dataclass(frozen=True)
class SpectreInterfaceGeometry:
    """SPECTRE Fourier interface coefficients in internal mode order.

    Arrays have shape ``(n_interfaces + 1, mn)``. Row zero is the magnetic-axis
    row used by the coordinate-singularity volume, and rows ``1..n_interfaces``
    are material interfaces from ``physics.allrzrz`` or the outer boundary.
    """

    poloidal_modes: Array
    toroidal_modes: Array
    rbc: Array
    zbs: Array
    rbs: Array
    zbc: Array
    nfp: int
    igeometry: int
    enforce_stellarator_symmetry: bool

    @property
    def interface_count(self) -> int:
        return int(self.rbc.shape[0] - 1)

    @property
    def mode_count(self) -> int:
        return int(self.rbc.shape[1])


@dataclass(frozen=True)
class SpectreVolumeGeometry:
    """Interpolated SPECTRE volume Fourier coefficients and radial derivative."""

    coefficients: SpectreInterfaceGeometry
    lvol: int
    s: Array
    rbc: Array
    zbs: Array
    rbs: Array
    zbc: Array
    drbc_ds: Array
    dzbs_ds: Array
    drbs_ds: Array
    dzbc_ds: Array
    coordinate_singularity: bool


@dataclass(frozen=True)
class SpectreCoordinateGrid:
    """Real-space SPECTRE coordinates, first derivatives, Jacobian, and metric."""

    r: Array
    z: Array
    dr_ds: Array
    dz_ds: Array
    dr_dtheta: Array
    dz_dtheta: Array
    dr_dzeta: Array
    dz_dzeta: Array
    jacobian: Array
    inverse_jacobian: Array
    metric: Array


def _parse_mode_key(key: str) -> tuple[int, int]:
    stripped = key.strip()
    values = [part.strip() for part in stripped[1:-1].split(",")]
    return int(values[0]), int(values[1])


def _parse_mode_table(table: Mapping[str, Any] | None) -> ModeTable:
    if not table:
        return {}
    return {_parse_mode_key(key): float(value) for key, value in table.items()}


def _lookup(table: ModeTable, m: int, internal_n: int, nfp: int) -> float:
    if nfp == 0:
        raise ValueError("nfp must be nonzero")
    if internal_n % nfp != 0:
        return 0.0
    return float(table.get((m, internal_n // nfp), 0.0))


def _tables_to_mode_array(
    table: ModeTable,
    modes: tuple[tuple[int, int], ...],
    *,
    nfp: int,
) -> np.ndarray:
    return np.asarray([_lookup(table, m, n, nfp) for m, n in modes], dtype=np.float64)


def _axis_table(summary: SpectreInputSummary, key: str) -> ModeTable:
    raw = summary.physics.get(key, ())
    if not raw:
        return {}
    modes = list(spectre_fourier_modes(summary))
    values = [float(value) for value in raw]
    table: ModeTable = {}
    for index, value in enumerate(values[: len(modes)]):
        internal_m, internal_n = modes[index]
        logical_n = internal_n // summary.nfp if summary.nfp else internal_n
        table[(internal_m, logical_n)] = value
    return table


def _interface_tables(summary: SpectreInputSummary) -> tuple[dict[str, ModeTable], ...]:
    allrzrz = summary.physics.get("allrzrz", {})
    interfaces: list[dict[str, ModeTable]] = []
    if isinstance(allrzrz, Mapping) and allrzrz:
        def interface_index(name: str) -> int:
            return int(name.removeprefix("interface_"))

        for name in sorted(allrzrz, key=interface_index):
            raw = allrzrz[name]
            interfaces.append(
                {
                    "rbc": _parse_mode_table(raw.get("rbc", {})),
                    "zbs": _parse_mode_table(raw.get("zbs", {})),
                    "rbs": _parse_mode_table(raw.get("rbs", {})),
                    "zbc": _parse_mode_table(raw.get("zbc", {})),
                }
            )
        if summary.is_free_boundary:
            interfaces.append(
                {
                    "rbc": _parse_mode_table(summary.physics.get("rwc", {})),
                    "zbs": _parse_mode_table(summary.physics.get("zws", {})),
                    "rbs": _parse_mode_table(summary.physics.get("rws", {})),
                    "zbc": _parse_mode_table(summary.physics.get("zwc", {})),
                }
            )
    elif summary.boundary_tables()["rbc"]:
        interfaces.append(summary.boundary_tables())
    return tuple(interfaces)


def build_spectre_interface_geometry(summary: SpectreInputSummary) -> SpectreInterfaceGeometry:
    """Build SPECTRE interface coefficient arrays in internal Fourier order."""

    modes = spectre_fourier_modes(summary)
    poloidal_modes = np.asarray([mode[0] for mode in modes], dtype=np.int32)
    toroidal_modes = np.asarray([mode[1] for mode in modes], dtype=np.int32)
    axis_tables = {
        "rbc": _axis_table(summary, "rac"),
        "zbs": _axis_table(summary, "zas"),
        "rbs": _axis_table(summary, "ras"),
        "zbc": _axis_table(summary, "zac"),
    }
    interfaces = _interface_tables(summary)
    if not interfaces:
        raise ValueError("SPECTRE input does not provide boundary or allrzrz interface geometry")

    rows: dict[str, list[np.ndarray]] = {name: [] for name in ("rbc", "zbs", "rbs", "zbc")}
    for name in rows:
        rows[name].append(_tables_to_mode_array(axis_tables[name], modes, nfp=summary.nfp))
    for interface in interfaces:
        for name in rows:
            rows[name].append(_tables_to_mode_array(interface.get(name, {}), modes, nfp=summary.nfp))

    return SpectreInterfaceGeometry(
        poloidal_modes=jnp.asarray(poloidal_modes, dtype=jnp.int32),
        toroidal_modes=jnp.asarray(toroidal_modes, dtype=jnp.int32),
        rbc=jnp.asarray(np.stack(rows["rbc"]), dtype=jnp.float64),
        zbs=jnp.asarray(np.stack(rows["zbs"]), dtype=jnp.float64),
        rbs=jnp.asarray(np.stack(rows["rbs"]), dtype=jnp.float64),
        zbc=jnp.asarray(np.stack(rows["zbc"]), dtype=jnp.float64),
        nfp=summary.nfp,
        igeometry=summary.igeometry,
        enforce_stellarator_symmetry=summary.enforce_stellarator_symmetry,
    )


def _coordinate_singularity_factor(
    *,
    igeometry: int,
    poloidal_modes: Array,
    s: Array,
) -> tuple[Array, Array]:
    sbar = 0.5 * (s + 1.0)
    m = poloidal_modes.astype(jnp.float64)
    if igeometry == 2:
        factor = jnp.where(poloidal_modes == 0, sbar, sbar ** (m + 1.0))
        derivative = jnp.where(
            poloidal_modes == 0,
            0.5,
            0.5 * (m + 1.0) * jnp.where(sbar > 0.0, factor / sbar, 0.0),
        )
    elif igeometry == 3:
        factor = jnp.where(poloidal_modes == 0, sbar**2, sbar**m)
        derivative = jnp.where(
            poloidal_modes == 0,
            sbar,
            0.5 * m * jnp.where(sbar > 0.0, factor / sbar, 0.0),
        )
    else:
        raise ValueError(f"coordinate-singularity interpolation is unsupported for Igeometry={igeometry}")
    return factor, derivative


def interpolate_spectre_volume_geometry(
    geometry: SpectreInterfaceGeometry,
    *,
    lvol: int,
    s: ArrayLike,
) -> SpectreVolumeGeometry:
    """Interpolate SPECTRE interface coefficients inside one volume.

    ``lvol`` is one-based and ``s`` is SPECTRE's local radial coordinate in
    ``[-1, 1]``. The first volume is treated as the coordinate-singularity
    branch when ``Igeometry != 1``.
    """

    if lvol < 1 or lvol > geometry.interface_count:
        raise ValueError(f"lvol={lvol} outside 1..{geometry.interface_count}")
    s_j = jnp.asarray(s, dtype=jnp.float64)
    coordinate_singularity = bool(geometry.igeometry != 1 and lvol == 1)

    def interpolate_component(component: Array) -> tuple[Array, Array]:
        left = component[lvol - 1]
        right = component[lvol]
        if coordinate_singularity:
            factor, derivative = _coordinate_singularity_factor(
                igeometry=geometry.igeometry,
                poloidal_modes=geometry.poloidal_modes,
                s=s_j,
            )
            return left + (right - left) * factor, (right - left) * derivative
        return 0.5 * (1.0 - s_j) * left + 0.5 * (1.0 + s_j) * right, 0.5 * (right - left)

    rbc, drbc_ds = interpolate_component(geometry.rbc)
    zbs, dzbs_ds = interpolate_component(geometry.zbs)
    rbs, drbs_ds = interpolate_component(geometry.rbs)
    zbc, dzbc_ds = interpolate_component(geometry.zbc)
    return SpectreVolumeGeometry(
        coefficients=geometry,
        lvol=lvol,
        s=s_j,
        rbc=rbc,
        zbs=zbs,
        rbs=rbs,
        zbc=zbc,
        drbc_ds=drbc_ds,
        dzbs_ds=dzbs_ds,
        drbs_ds=drbs_ds,
        dzbc_ds=dzbc_ds,
        coordinate_singularity=coordinate_singularity,
    )


@jax.jit
def _evaluate_series(
    poloidal_modes: Array,
    toroidal_modes: Array,
    even_cos: Array,
    odd_sin: Array,
    theta: Array,
    zeta: Array,
) -> tuple[Array, Array, Array]:
    phase = poloidal_modes[:, None, None] * theta[None, :, None] - toroidal_modes[:, None, None] * zeta[None, None, :]
    cos_phase = jnp.cos(phase)
    sin_phase = jnp.sin(phase)
    m = poloidal_modes[:, None, None]
    n = toroidal_modes[:, None, None]
    value = jnp.sum(even_cos[:, None, None] * cos_phase + odd_sin[:, None, None] * sin_phase, axis=0)
    dtheta = jnp.sum(-m * even_cos[:, None, None] * sin_phase + m * odd_sin[:, None, None] * cos_phase, axis=0)
    dzeta = jnp.sum(n * even_cos[:, None, None] * sin_phase - n * odd_sin[:, None, None] * cos_phase, axis=0)
    return value, dtheta, dzeta


def evaluate_spectre_volume_coordinates(
    volume: SpectreVolumeGeometry,
    *,
    theta: ArrayLike,
    zeta: ArrayLike,
) -> SpectreCoordinateGrid:
    """Evaluate SPECTRE coordinates, first derivatives, Jacobian, and metric."""

    theta_j = jnp.asarray(theta, dtype=jnp.float64)
    zeta_j = jnp.asarray(zeta, dtype=jnp.float64)
    modes_m = volume.coefficients.poloidal_modes
    modes_n = volume.coefficients.toroidal_modes

    r, dr_dtheta, dr_dzeta = _evaluate_series(modes_m, modes_n, volume.rbc, volume.rbs, theta_j, zeta_j)
    z, dz_dtheta, dz_dzeta = _evaluate_series(modes_m, modes_n, volume.zbc, volume.zbs, theta_j, zeta_j)
    dr_ds, _, _ = _evaluate_series(modes_m, modes_n, volume.drbc_ds, volume.drbs_ds, theta_j, zeta_j)
    dz_ds, _, _ = _evaluate_series(modes_m, modes_n, volume.dzbc_ds, volume.dzbs_ds, theta_j, zeta_j)

    if volume.coefficients.igeometry == 1:
        rpol = jnp.asarray(1.0, dtype=jnp.float64)
        rtor = jnp.asarray(1.0, dtype=jnp.float64)
        jacobian = dr_ds * rpol * rtor
        gss = dr_ds * dr_ds
        gst = dr_ds * dr_dtheta
        gsz = dr_ds * dr_dzeta
        gtt = dr_dtheta * dr_dtheta + rpol * rpol
        gtz = dr_dtheta * dr_dzeta
        gzz = dr_dzeta * dr_dzeta + rtor * rtor
    elif volume.coefficients.igeometry == 2:
        jacobian = dr_ds * r
        gss = dr_ds * dr_ds
        gst = dr_ds * dr_dtheta
        gsz = dr_ds * dr_dzeta
        gtt = dr_dtheta * dr_dtheta + r * r
        gtz = dr_dtheta * dr_dzeta
        gzz = dr_dzeta * dr_dzeta + 1.0
    elif volume.coefficients.igeometry == 3:
        jacobian = r * (dz_ds * dr_dtheta - dr_ds * dz_dtheta)
        gss = dr_ds * dr_ds + dz_ds * dz_ds
        gst = dr_ds * dr_dtheta + dz_ds * dz_dtheta
        gsz = dr_ds * dr_dzeta + dz_ds * dz_dzeta
        gtt = dr_dtheta * dr_dtheta + dz_dtheta * dz_dtheta
        gtz = dr_dtheta * dr_dzeta + dz_dtheta * dz_dzeta
        gzz = dr_dzeta * dr_dzeta + dz_dzeta * dz_dzeta + r * r
    else:
        raise ValueError(f"unsupported SPECTRE Igeometry={volume.coefficients.igeometry}")

    metric = jnp.stack(
        (
            jnp.stack((gss, gst, gsz), axis=-1),
            jnp.stack((gst, gtt, gtz), axis=-1),
            jnp.stack((gsz, gtz, gzz), axis=-1),
        ),
        axis=-2,
    )
    inverse_jacobian = 1.0 / jacobian
    return SpectreCoordinateGrid(
        r=r,
        z=z,
        dr_ds=dr_ds,
        dz_ds=dz_ds,
        dr_dtheta=dr_dtheta,
        dz_dtheta=dz_dtheta,
        dr_dzeta=dr_dzeta,
        dz_dzeta=dz_dzeta,
        jacobian=jacobian,
        inverse_jacobian=inverse_jacobian,
        metric=metric,
    )
