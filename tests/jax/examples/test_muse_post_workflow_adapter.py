"""Focused contracts for the MUSE post-optimization workflow adapter.

The official post-GPMO half of ``2_Intermediate/permanent_magnet_MUSE.py``
(upstream ``9e027eac38028d57aa23777be52a781aa860e347``) evaluates the dipole
squared flux and the total magnet volume on the plotting surface and traces the
TF-coil field alone. These tests pin the settings that reach the interpolant
and the integrator -- not only the policy dataclass -- and that the native lane
runs ``simsopt.field.compute_fieldlines`` while the JAX lane runs the port.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from simsopt_jax_adapters.examples import muse


class _Surface:
    nfp = 2
    stellsym = True

    def gamma(self) -> np.ndarray:
        return np.asarray([[[0.34, 0.0, 0.0]]], dtype=np.float64)

    def unitnormal(self) -> np.ndarray:
        return np.asarray([[[1.0, 0.0, 0.0]]], dtype=np.float64)


class _CoilField:
    def __init__(self) -> None:
        self.points = np.empty((0, 3), dtype=np.float64)

    def set_points(self, points: np.ndarray) -> None:
        self.points = points

    def B(self) -> np.ndarray:
        return np.asarray([[0.0, 1.0, 0.0]], dtype=np.float64)


class _DipoleField:
    """Stand-in with ``DipoleField``'s real signature."""

    def __init__(
        self,
        dipole_grid_xyz: np.ndarray,
        moments: np.ndarray,
        *,
        nfp: int,
        coordinate_flag: str,
        m_maxima: np.ndarray,
    ) -> None:
        self.dipole_grid_xyz = dipole_grid_xyz
        self.moments = moments
        self.nfp = nfp
        self.coordinate_flag = coordinate_flag
        self.m_maxima = m_maxima

    def set_points(self, points: np.ndarray) -> None:
        self.points = points


class _SquaredFlux:
    """Stand-in with ``SquaredFlux``'s positional signature."""

    def __init__(self, surface, field, target) -> None:
        self.surface = surface
        self.field = field
        self.target = target

    def J(self) -> float:
        return 2.5


class _Classifier:
    def __init__(self, surface, *, h: float, p: int) -> None:
        self.surface = surface
        self.h = h
        self.p = p

    def dist(self, points: np.ndarray) -> np.ndarray:
        return np.ones(points.shape[0], dtype=np.float64)

    def evaluate_rphiz(self, points: np.ndarray) -> np.ndarray:
        return np.ones(points.shape[0], dtype=np.float64)


def _interpolant_recorder(captured: dict[str, object]):
    def fake_interpolated(
        field,
        degree,
        radial_range,
        toroidal_range,
        vertical_range,
        extrapolate,
        *,
        nfp,
        stellsym,
        skip,
    ):
        captured["trace_source"] = field
        captured["degree"] = degree
        captured["radial_range"] = radial_range
        captured["toroidal_range"] = toroidal_range
        captured["vertical_range"] = vertical_range
        captured["extrapolate"] = extrapolate
        captured["nfp"] = nfp
        captured["stellsym"] = stellsym
        captured["skip"] = skip
        return SimpleNamespace(name="interpolant")

    return fake_interpolated


def _install_shared_fakes(
    monkeypatch: pytest.MonkeyPatch,
    captured: dict[str, object],
) -> None:
    monkeypatch.setattr(muse, "DipoleField", _DipoleField)
    monkeypatch.setattr(muse, "SquaredFlux", _SquaredFlux)
    monkeypatch.setattr(muse, "SurfaceClassifier", _Classifier)
    monkeypatch.setattr(
        muse,
        "LevelsetStoppingCriterion",
        lambda criterion: SimpleNamespace(criterion=criterion),
    )
    captured.clear()


def _assert_official_interpolant(captured: dict[str, object]) -> None:
    """Bounded policy settings as they reach the interpolant constructor."""

    assert captured["degree"] == 2
    assert captured["radial_range"] == (0.34, 0.34, 5)
    assert captured["toroidal_range"] == (0.0, np.pi, 10)
    assert captured["vertical_range"] == (0.0, 0.0, 2)
    assert captured["extrapolate"] is True
    assert captured["nfp"] == 2
    assert captured["stellsym"] is True
    skip = captured["skip"]
    assert callable(skip)
    # ``distance < -0.05`` is the official skip rule; this classifier is at +1.
    assert skip(np.zeros(1), np.zeros(1), np.zeros(1)).tolist() == [False]


def _assert_official_trace_settings(trace_kwargs: object) -> None:
    assert isinstance(trace_kwargs, dict)
    assert trace_kwargs["tmax"] == 50
    assert trace_kwargs["tol"] == 1.0e-16
    assert trace_kwargs["phis"] == (0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0)
    assert len(trace_kwargs["stopping_criteria"]) == 1


def test_muse_post_policy_preserves_official_native_settings() -> None:
    native = muse.muse_post_workflow_policy("native_default")
    bounded = muse.muse_post_workflow_policy("bounded")

    assert native.fieldline_count == 30
    assert native.fieldline_tmax == 20_000
    assert native.interpolation_grid_size == 20
    assert native.interpolation_degree == 2
    assert native.fieldline_tolerance == 1.0e-16
    assert native.classifier_h == 0.03
    assert native.classifier_p == 2
    assert native.skip_distance == -0.05
    assert bounded.fieldline_count == 3
    assert bounded.fieldline_tmax == 50
    assert bounded.interpolation_grid_size == 5
    assert bounded.fieldline_tolerance == native.fieldline_tolerance


def test_jax_post_workflow_traces_coils_only_with_the_official_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coil_field = _CoilField()
    geometry = muse.MusePostGeometry(_Surface(), _Surface(), coil_field)
    captured: dict[str, object] = {}
    _install_shared_fakes(monkeypatch, captured)

    def fake_trace(field, radial_initial, vertical_initial, **kwargs):
        captured["traced_field"] = field
        captured["radial_initial"] = radial_initial
        captured["vertical_initial"] = vertical_initial
        captured["trace_kwargs"] = kwargs
        trajectories = [
            np.asarray([[0.0, 0.34, 0.0, 0.0], [2.0, 0.34, 0.0, 0.0]]) for _ in range(3)
        ]
        hits = [np.empty((count, 5), dtype=np.float64) for count in (0, 1, 2)]
        return trajectories, hits, np.asarray([1, -1, 0], dtype=np.int64)

    monkeypatch.setattr(muse, "InterpolatedFieldJAX", _interpolant_recorder(captured))
    monkeypatch.setattr(muse, "compute_fieldlines_with_status", fake_trace)

    diagnostics = muse.run_jax_muse_post_workflow(
        geometry,
        muse.muse_post_workflow_policy("bounded"),
        np.asarray([[3.0, 4.0, 0.0]], dtype=np.float64),
        np.asarray([[0.4, 0.0, 0.0]], dtype=np.float64),
        np.asarray([5.0], dtype=np.float64),
        coordinate_flag="cartesian",
    )

    assert captured["trace_source"] is coil_field
    assert captured["traced_field"] is not coil_field
    _assert_official_interpolant(captured)
    np.testing.assert_allclose(captured["radial_initial"], [0.32, 0.34, 0.36])
    np.testing.assert_allclose(captured["vertical_initial"], [0.0, 0.0, 0.0])
    _assert_official_trace_settings(captured["trace_kwargs"])
    assert diagnostics.dipole_squared_flux == 2.5
    assert diagnostics.total_magnet_volume == pytest.approx(
        5.0 * 2.0 * 2.0 * muse.MUSE_VACUUM_PERMEABILITY / muse.MUSE_MAGNETIC_FIELD_LIMIT
    )
    assert diagnostics.trace_statuses == (1, -1, 0)
    assert diagnostics.trace_final_times == (2.0, 2.0, 2.0)
    assert diagnostics.trace_hit_counts == (0, 1, 2)
    assert diagnostics.trace_success is False


def test_native_post_workflow_uses_simsopt_compute_fieldlines_and_the_dist_levelset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coil_field = _CoilField()
    geometry = muse.MusePostGeometry(_Surface(), _Surface(), coil_field)
    captured: dict[str, object] = {}
    _install_shared_fakes(monkeypatch, captured)

    def fake_native_trace(field, radial_initial, vertical_initial, **kwargs):
        captured["traced_field"] = field
        captured["radial_initial"] = radial_initial
        captured["trace_kwargs"] = kwargs
        # Line 0 runs the full horizon; line 1 stops on a negative-index hit;
        # line 2 stops with no negative hit recorded.
        trajectories = [
            np.asarray([[0.0, 0.34, 0.0, 0.0], [50.0, 0.34, 0.0, 0.0]]),
            np.asarray([[0.0, 0.34, 0.0, 0.0], [2.0, 0.34, 0.0, 0.0]]),
            np.asarray([[0.0, 0.34, 0.0, 0.0], [3.0, 0.34, 0.0, 0.0]]),
        ]
        hits = [
            np.empty((0, 5), dtype=np.float64),
            np.asarray([[2.0, -1.0, 0.34, 0.0, 0.0]], dtype=np.float64),
            np.asarray([[3.0, 1.0, 0.34, 0.0, 0.0]], dtype=np.float64),
        ]
        return trajectories, hits

    monkeypatch.setattr(muse, "InterpolatedField", _interpolant_recorder(captured))
    monkeypatch.setattr(muse, "compute_fieldlines", fake_native_trace)

    diagnostics = muse.run_native_muse_post_workflow(
        geometry,
        muse.muse_post_workflow_policy("bounded"),
        np.asarray([[3.0, 4.0, 0.0]], dtype=np.float64),
        np.asarray([[0.4, 0.0, 0.0]], dtype=np.float64),
        np.asarray([5.0], dtype=np.float64),
        coordinate_flag="cartesian",
    )

    assert captured["trace_source"] is coil_field
    _assert_official_interpolant(captured)
    trace_kwargs = captured["trace_kwargs"]
    _assert_official_trace_settings(trace_kwargs)
    # Official line 302 passes ``sc_fieldline.dist``, not the classifier.
    criterion = trace_kwargs["stopping_criteria"][0].criterion
    assert getattr(criterion, "__name__", "") == "dist"
    assert diagnostics.dipole_squared_flux == 2.5
    assert diagnostics.trace_statuses == (0, -1, 1)
    assert diagnostics.trace_final_times == (50.0, 2.0, 3.0)
    assert diagnostics.trace_hit_counts == (0, 1, 1)
    # Last accepted step per line; the single-row line reports its own time.
    assert diagnostics.trace_final_steps == (50.0, 2.0, 3.0)
    assert diagnostics.trace_success is False


def test_public_muse_script_routes_post_diagnostics_through_shared_helper() -> None:
    path = (
        Path(__file__).resolve().parents[3]
        / "examples/jax/2_Intermediate/permanent_magnet_MUSE.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    solve = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "solve"
    )

    assert any(
        isinstance(node, ast.ImportFrom)
        and node.module == "simsopt_jax_adapters.examples.muse"
        and any(alias.name == "run_jax_muse_post_workflow" for alias in node.names)
        for node in tree.body
    )
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_jax_muse_post_workflow"
        for node in ast.walk(solve)
    )
    source = ast.unparse(solve)
    # The trace outcome and the objective comparison are published diagnostics,
    # not completion gates (official MUSE asserts neither).
    assert "'post_coil_only_trace_terminal': post_diagnostics.trace_success" in source
    assert "status='ok' if solver_success else 'failed'" in source
    # The success gate is the shared GPMO rule, evaluated on the FULL moment
    # array: a NaN row is dropped by ``norm(...) > 0.0`` before any filtered
    # finiteness check can see it, which is how the example published
    # ``status='ok'`` with a NaN moment.
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "gpmo_backtracking_outputs_usable"
        for node in ast.walk(solve)
    )
    # The post metrics follow upstream's argmin-R2 snapshot, not the endpoint.
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "select_minimum_objective_snapshot"
        for node in ast.walk(solve)
    )
    assert "run_jax_muse_post_workflow(post_geometry" in source
    assert "snapshot_moments, dipole_grid_xyz" in source
