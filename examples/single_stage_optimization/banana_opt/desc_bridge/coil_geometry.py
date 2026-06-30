"""Shared periodic XYZ coil geometry diagnostics for DESC bridge reports."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from simsopt.geo import CurveXYZFourier


@dataclass(frozen=True, slots=True)
class PeriodicXyzFit:
    reconstructed_xyz: np.ndarray
    coefficients_xyz: np.ndarray
    max_residual_m: float
    rms_residual_m: float


def fit_periodic_xyz(coords: np.ndarray, *, order: int) -> PeriodicXyzFit:
    sample_count = coords.shape[0]
    s = np.linspace(0.0, 2.0 * np.pi, sample_count, endpoint=False)
    basis = periodic_fit_matrix(s, order=order)
    coefficients, _residuals, _rank, _singular_values = np.linalg.lstsq(
        basis,
        coords,
        rcond=None,
    )
    reconstructed = np.ascontiguousarray(basis @ coefficients)
    residual_norms = np.linalg.norm(reconstructed - coords, axis=1)
    return PeriodicXyzFit(
        reconstructed_xyz=reconstructed,
        coefficients_xyz=np.ascontiguousarray(coefficients),
        max_residual_m=float(np.max(residual_norms)),
        rms_residual_m=float(np.sqrt(np.mean(residual_norms**2))),
    )


def periodic_fit_matrix(s: np.ndarray, *, order: int) -> np.ndarray:
    columns = [np.ones_like(s)]
    for mode in range(1, order + 1):
        columns.append(np.cos(float(mode) * s))
        columns.append(np.sin(float(mode) * s))
    return np.column_stack(columns)


def simsopt_dofs_from_periodic_xyz_coefficients(
    coefficients_xyz: np.ndarray,
    *,
    order: int,
) -> np.ndarray:
    expected_shape = (1 + 2 * order, 3)
    if coefficients_xyz.shape != expected_shape:
        raise ValueError(
            "periodic XYZ coefficients must have shape "
            f"{expected_shape}; got {coefficients_xyz.shape}."
        )
    coordinate_dofs = []
    for coordinate_index in range(3):
        coefficients = coefficients_xyz[:, coordinate_index]
        values = [coefficients[0]]
        for mode in range(1, order + 1):
            cos_index = 2 * mode - 1
            sin_index = 2 * mode
            values.append(coefficients[sin_index])
            values.append(coefficients[cos_index])
        coordinate_dofs.extend(values)
    return np.asarray(coordinate_dofs, dtype=float)


def curve_xyz_fourier_from_periodic_xyz_coefficients(
    coefficients_xyz: np.ndarray,
    *,
    sample_count: int,
    order: int,
) -> CurveXYZFourier:
    curve = CurveXYZFourier(sample_count, order)
    dofs = curve.get_dofs().copy()
    expected_dofs = 3 * (1 + 2 * order)
    if dofs.size != expected_dofs:
        raise ValueError(
            "Unexpected CurveXYZFourier DOF count: "
            f"{dofs.size} vs {expected_dofs}."
        )
    dofs[:] = simsopt_dofs_from_periodic_xyz_coefficients(
        coefficients_xyz,
        order=order,
    )
    curve.set_dofs(dofs)
    return curve


def periodic_length(coords: np.ndarray) -> float:
    deltas = np.roll(coords, -1, axis=0) - coords
    return float(np.sum(np.linalg.norm(deltas, axis=1)))


def max_periodic_curvature(coords: np.ndarray) -> float:
    sample_count = coords.shape[0]
    ds = 2.0 * np.pi / float(sample_count)
    first = (np.roll(coords, -1, axis=0) - np.roll(coords, 1, axis=0)) / (2.0 * ds)
    second = (
        np.roll(coords, -1, axis=0) - 2.0 * coords + np.roll(coords, 1, axis=0)
    ) / (ds**2)
    numerator = np.linalg.norm(np.cross(first, second), axis=1)
    denominator = np.linalg.norm(first, axis=1) ** 3
    curvature = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0.0,
    )
    return float(np.max(curvature))


def min_pairwise_periodic_distance(curves: Sequence[np.ndarray]) -> float | None:
    if len(curves) < 2:
        return None
    min_distance = np.inf
    for first_index, first_curve in enumerate(curves[:-1]):
        first = _coerce_curve_coords(first_curve)
        for second_curve in curves[first_index + 1 :]:
            second = _coerce_curve_coords(second_curve)
            deltas = first[:, np.newaxis, :] - second[np.newaxis, :, :]
            distances = np.linalg.norm(deltas, axis=2)
            min_distance = min(min_distance, float(np.min(distances)))
    return float(min_distance)


def resample_periodic_xyz(coords: np.ndarray, *, sample_count: int) -> np.ndarray:
    native_coords = np.asarray(coords, dtype=float)
    _validate_curve_coords(native_coords)
    if native_coords.shape[0] < 4 or sample_count < 4:
        raise ValueError("periodic XYZ resampling requires at least 4 samples.")
    if native_coords.shape[0] == sample_count:
        return np.ascontiguousarray(native_coords)
    native_s = np.linspace(0.0, 1.0, native_coords.shape[0], endpoint=False)
    target_s = np.linspace(0.0, 1.0, sample_count, endpoint=False)
    closed_s = np.concatenate((native_s, np.asarray([1.0])))
    closed_coords = np.vstack((native_coords, native_coords[0]))
    return np.ascontiguousarray(
        np.column_stack(
            [
                np.interp(target_s, closed_s, closed_coords[:, coordinate_index])
                for coordinate_index in range(3)
            ]
        )
    )


def _coerce_curve_coords(coords: np.ndarray) -> np.ndarray:
    coerced = np.asarray(coords, dtype=float)
    _validate_curve_coords(coerced)
    return coerced


def _validate_curve_coords(coords: np.ndarray) -> None:
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(
            "periodic XYZ coordinates must have shape (N, 3); got "
            f"{coords.shape}."
        )
    if not np.isfinite(coords).all():
        raise ValueError("periodic XYZ coordinates must be finite.")


__all__ = [
    "PeriodicXyzFit",
    "curve_xyz_fourier_from_periodic_xyz_coefficients",
    "fit_periodic_xyz",
    "max_periodic_curvature",
    "min_pairwise_periodic_distance",
    "periodic_fit_matrix",
    "periodic_length",
    "resample_periodic_xyz",
    "simsopt_dofs_from_periodic_xyz_coefficients",
]
