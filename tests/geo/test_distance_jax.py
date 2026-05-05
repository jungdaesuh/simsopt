"""JAX-native distance-candidate culler tests."""

from __future__ import annotations

import numpy as np
import pytest

import simsopt.geo.curveobjectives as curveobjectives_module
from simsopt.geo._distance_jax import (
    get_close_candidates_between_collections,
    get_close_candidates_within_collection,
)
from simsopt.geo.curveobjectives import CurveCurveDistance, CurveSurfaceDistance
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
import simsoptpp as sopp


def _random_point_clouds(seed: int, count: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [
        rng.uniform(low=-1.0, high=1.0, size=(5 + index, 3)).astype(np.float64)
        for index in range(count)
    ]


def test_jax_within_collection_candidates_match_cpp_lower_triangle():
    point_clouds = _random_point_clouds(seed=1729, count=5)
    threshold = 0.75
    num_base_curves = 3

    cpp_candidates = sopp.get_pointclouds_closer_than_threshold_within_collection(
        point_clouds,
        threshold,
        num_base_curves,
    )
    jax_candidates = get_close_candidates_within_collection(
        point_clouds,
        threshold,
        num_base_curves,
    )

    assert set(jax_candidates) == set(cpp_candidates)
    assert all(j < i and j < num_base_curves for i, j in jax_candidates)


def test_jax_between_collection_candidates_match_cpp_rectangular_pairs():
    left_point_clouds = _random_point_clouds(seed=1730, count=4)
    right_point_clouds = _random_point_clouds(seed=1731, count=3)
    threshold = 0.65

    cpp_candidates = sopp.get_pointclouds_closer_than_threshold_between_two_collections(
        left_point_clouds,
        right_point_clouds,
        threshold,
    )
    jax_candidates = get_close_candidates_between_collections(
        left_point_clouds,
        right_point_clouds,
        threshold,
    )

    assert set(jax_candidates) == set(cpp_candidates)


def test_curve_curve_distance_uses_jax_candidate_culler(monkeypatch):
    monkeypatch.setattr(curveobjectives_module, "is_jax_backend", lambda: True)

    def reject_cpp_culler(*_args, **_kwargs):
        raise AssertionError("C++ curve-curve candidate culler should not run")

    monkeypatch.setattr(
        curveobjectives_module.sopp,
        "get_pointclouds_closer_than_threshold_within_collection",
        reject_cpp_culler,
    )

    curves = [CurveXYZFourier(8, 1) for _ in range(3)]
    objective = CurveCurveDistance(curves, minimum_distance=10.0)
    objective.compute_candidates()

    assert set(objective.candidates) == {(1, 0), (2, 0), (2, 1)}


def test_curve_surface_distance_uses_jax_candidate_culler(monkeypatch):
    monkeypatch.setattr(curveobjectives_module, "is_jax_backend", lambda: True)

    def reject_cpp_culler(*_args, **_kwargs):
        raise AssertionError("C++ curve-surface candidate culler should not run")

    monkeypatch.setattr(
        curveobjectives_module.sopp,
        "get_pointclouds_closer_than_threshold_between_two_collections",
        reject_cpp_culler,
    )

    curves = [CurveXYZFourier(8, 1) for _ in range(2)]
    surface = SurfaceRZFourier.from_nphi_ntheta(
        nfp=1,
        nphi=8,
        ntheta=8,
        mpol=1,
        ntor=0,
    )
    objective = CurveSurfaceDistance(curves, surface, minimum_distance=10.0)
    objective.compute_candidates()

    assert set(objective.candidates) == {(0, 0), (1, 0)}
