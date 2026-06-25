"""JAX-native distance-candidate culler tests."""

from __future__ import annotations

import numpy as np

import simsopt.geo.curveobjectives as curveobjectives_module
from simsopt_jax.geo.distance import (
    _between_collections_candidate_mask,
    _stack_point_clouds,
    _within_collection_candidate_mask,
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


def test_jax_within_collection_candidate_mask_has_static_pair_shape():
    point_clouds = _random_point_clouds(seed=1729, count=5)
    points, valid = _stack_point_clouds(point_clouds)

    mask = _within_collection_candidate_mask(points, valid, 0.75, num_base_curves=3)

    assert mask.shape == (5, 5)
    assert mask.dtype == np.dtype(bool)


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


def test_jax_between_collection_candidate_mask_has_static_pair_shape():
    left_point_clouds = _random_point_clouds(seed=1730, count=4)
    right_point_clouds = _random_point_clouds(seed=1731, count=3)
    left_points, left_valid = _stack_point_clouds(left_point_clouds)
    right_points, right_valid = _stack_point_clouds(right_point_clouds)

    mask = _between_collections_candidate_mask(
        left_points,
        left_valid,
        right_points,
        right_valid,
        0.65,
    )

    assert mask.shape == (4, 3)
    assert mask.dtype == np.dtype(bool)


def test_curve_curve_distance_uses_cpp_candidate_culler(monkeypatch):
    calls = []

    def record_cpp_culler(point_clouds, threshold, num_base_curves):
        calls.append((len(point_clouds), threshold, num_base_curves))
        return [(1, 0)]

    curves = [CurveXYZFourier(8, 1) for _ in range(3)]
    monkeypatch.setattr(
        curveobjectives_module.sopp,
        "get_pointclouds_closer_than_threshold_within_collection",
        record_cpp_culler,
    )
    objective = CurveCurveDistance(curves, minimum_distance=10.0, num_basecurves=2)
    objective.compute_candidates()

    assert calls == [(3, 10.0, 2)]
    assert objective.candidates == [(1, 0)]


def test_curve_surface_distance_uses_cpp_candidate_culler(monkeypatch):
    calls = []

    def record_cpp_culler(left_point_clouds, right_point_clouds, threshold):
        calls.append((len(left_point_clouds), len(right_point_clouds), threshold))
        return [(0, 0)]

    curves = [CurveXYZFourier(8, 1) for _ in range(2)]
    surface = SurfaceRZFourier.from_nphi_ntheta(
        nfp=1,
        nphi=8,
        ntheta=8,
        mpol=1,
        ntor=0,
    )
    monkeypatch.setattr(
        curveobjectives_module.sopp,
        "get_pointclouds_closer_than_threshold_between_two_collections",
        record_cpp_culler,
    )
    objective = CurveSurfaceDistance(curves, surface, minimum_distance=10.0)
    objective.compute_candidates()

    assert calls == [(2, 1, 10.0)]
    assert objective.candidates == [(0, 0)]
