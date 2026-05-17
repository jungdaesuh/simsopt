"""Public-wrapper parity tests for item 30 ``WireframeFieldJAX``."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks.validation_ladder_contract import parity_ladder_tolerances
from simsopt.backend import invalidate_backend_cache
from simsopt.field import (
    MagneticFieldSum,
    WireframeField,
    WireframeFieldJAX as ExportedWireframeFieldJAX,
)
from simsopt.field.toroidal_field_jax import ToroidalFieldJAX
from simsopt.field.wireframefield_jax import WireframeFieldJAX
from simsopt.geo import SurfaceRZFourier, ToroidalWireframe

_DIRECT_KERNEL = parity_ladder_tolerances("direct_kernel")
_RTOL = _DIRECT_KERNEL["rtol"]
_ATOL = _DIRECT_KERNEL["atol"]


def _surf_torus(nfp: int, rmaj: float, rmin: float) -> SurfaceRZFourier:
    surface = SurfaceRZFourier(nfp=nfp, mpol=1, ntor=0)
    surface.set_rc(0, 0, rmaj)
    surface.set_rc(1, 0, rmin)
    surface.set_zs(1, 0, rmin)
    return surface


def _wireframe_case() -> tuple[ToroidalWireframe, np.ndarray]:
    wireframe = ToroidalWireframe(_surf_torus(nfp=2, rmaj=2.0, rmin=0.7), 4, 6)
    wireframe.currents[:] = np.linspace(
        -2.0e5,
        3.5e5,
        wireframe.n_segments,
        dtype=np.float64,
    )
    points = np.array(
        [
            [1.25, 0.10, -0.18],
            [1.70, 0.35, 0.22],
            [2.15, -0.20, 0.05],
            [2.55, 0.42, -0.14],
            [1.55, -0.48, 0.31],
            [2.30, 0.18, -0.35],
            [1.88, 0.62, 0.16],
        ],
        dtype=np.float64,
    )
    return wireframe, np.ascontiguousarray(points)


def test_package_export():
    assert ExportedWireframeFieldJAX is WireframeFieldJAX


def test_wireframe_wrappers_reject_oversized_segment_indices():
    wireframe, _ = _wireframe_case()
    oversized_index = int(np.iinfo(np.int32).max) + 1
    wireframe.segments = np.asarray([[0, oversized_index]], dtype=np.int64)
    wireframe.n_segments = 1
    wireframe.currents = np.asarray([1.0], dtype=np.float64)

    with pytest.raises(ValueError, match="wframe.segments indices"):
        WireframeField(wireframe)
    with pytest.raises(ValueError, match="wframe.segments indices"):
        WireframeFieldJAX(wireframe)


def test_public_B_dB_and_segment_contributions_match_cpu():
    """Public field and segment-current derivatives match ``WireframeField``."""

    wireframe, points = _wireframe_case()
    cpu = WireframeField(wireframe)
    jax_field = WireframeFieldJAX(wireframe)
    cpu.set_points(points)
    jax_field.set_points(points)

    np.testing.assert_allclose(
        np.asarray(jax_field.B()), np.asarray(cpu.B()), rtol=_RTOL, atol=_ATOL
    )
    np.testing.assert_allclose(
        np.asarray(jax_field.dB_by_dX()),
        np.asarray(cpu.dB_by_dX()),
        rtol=_RTOL,
        atol=_ATOL,
    )
    np.testing.assert_allclose(
        np.stack(jax_field.dB_by_dsegmentcurrents(0), axis=0),
        np.stack(cpu.dB_by_dsegmentcurrents(0), axis=0),
        rtol=_RTOL,
        atol=_ATOL,
    )
    np.testing.assert_allclose(
        np.stack(jax_field.dB_by_dsegmentcurrents(1), axis=0),
        np.stack(cpu.dB_by_dsegmentcurrents(1), axis=0),
        rtol=_RTOL,
        atol=_ATOL,
    )


@pytest.mark.parametrize("area_weighted", (False, True))
def test_normal_field_matrix_matches_cpu(area_weighted):
    """Surface normal derivative matrix matches the C++ wrapper."""

    wireframe, _ = _wireframe_case()
    surface = _surf_torus(nfp=2, rmaj=1.85, rmin=0.45)
    points = np.ascontiguousarray(surface.gamma().reshape((-1, 3)))
    cpu = WireframeField(wireframe)
    jax_field = WireframeFieldJAX(wireframe)
    cpu.set_points(points)
    jax_field.set_points(points)

    np.testing.assert_allclose(
        jax_field.dBnormal_by_dsegmentcurrents_matrix(
            surface,
            area_weighted=area_weighted,
        ),
        cpu.dBnormal_by_dsegmentcurrents_matrix(
            surface,
            area_weighted=area_weighted,
        ),
        rtol=_RTOL,
        atol=_ATOL,
    )


def test_current_snapshot_matches_cpu_wrapper_semantics():
    """Construction snapshots currents; later wireframe mutation is not observed."""

    wireframe, points = _wireframe_case()
    cpu = WireframeField(wireframe)
    jax_field = WireframeFieldJAX(wireframe)
    wireframe.currents[:] = np.linspace(
        9.0e5,
        -4.0e5,
        wireframe.n_segments,
        dtype=np.float64,
    )
    cpu.set_points(points)
    jax_field.set_points(points)

    np.testing.assert_allclose(
        np.asarray(jax_field.B()), np.asarray(cpu.B()), rtol=_RTOL, atol=_ATOL
    )
    np.testing.assert_allclose(
        np.asarray(jax_field.dB_by_dX()),
        np.asarray(cpu.dB_by_dX()),
        rtol=_RTOL,
        atol=_ATOL,
    )


def test_segment_current_cache_invalidates_on_point_change():
    wireframe, points = _wireframe_case()
    jax_field = WireframeFieldJAX(wireframe)
    jax_field.set_points(points)

    first = np.stack(jax_field.dB_by_dsegmentcurrents(0), axis=0)
    assert jax_field._dB_by_dcoilcurrents is not None

    shifted = np.ascontiguousarray(points + np.array([0.11, -0.04, 0.07]))
    jax_field.set_points(shifted)
    assert jax_field._dB_by_dcoilcurrents is None

    cpu = WireframeField(wireframe)
    cpu.set_points(shifted)
    second = np.stack(jax_field.dB_by_dsegmentcurrents(0), axis=0)
    np.testing.assert_allclose(
        second,
        np.stack(cpu.dB_by_dsegmentcurrents(0), axis=0),
        rtol=_RTOL,
        atol=_ATOL,
    )
    assert not np.allclose(first, second, rtol=0.0, atol=0.0)


def test_segment_current_derivative_flag_returns_spatial_derivative_contributions():
    wireframe, points = _wireframe_case()
    cpu = WireframeField(wireframe)
    jax_field = WireframeFieldJAX(wireframe)
    cpu.set_points(points)
    jax_field.set_points(points)

    for field in (cpu, jax_field):
        dB_by_current = np.stack(field.dB_by_dsegmentcurrents(1), axis=0)
        reconstructed = np.tensordot(
            np.asarray(wireframe.currents, dtype=np.float64),
            dB_by_current,
            axes=(0, 0),
        )

        assert dB_by_current.shape == (wireframe.n_segments, points.shape[0], 3, 3)
        np.testing.assert_allclose(
            reconstructed,
            np.asarray(field.dB_by_dX()),
            rtol=_RTOL,
            atol=_ATOL,
        )


def test_segment_count_is_snapshotted_at_construction():
    wireframe, points = _wireframe_case()
    jax_field = WireframeFieldJAX(wireframe)
    segment_count = wireframe.n_segments
    wireframe.n_segments = 0
    jax_field.set_points(points)

    assert len(jax_field.dB_by_dsegmentcurrents(0)) == segment_count


def test_rejects_second_spatial_derivative_request():
    wireframe, points = _wireframe_case()
    cpu = WireframeField(wireframe)
    jax_field = WireframeFieldJAX(wireframe)
    cpu.set_points(points)
    jax_field.set_points(points)

    with pytest.raises(NotImplementedError):
        cpu.dB_by_dsegmentcurrents(2)
    with pytest.raises(NotImplementedError):
        jax_field.dB_by_dsegmentcurrents(2)


def test_wireframefield_jax_does_not_advertise_native_vjp_contract(monkeypatch):
    monkeypatch.setenv("SIMSOPT_BACKEND", "jax")
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_cpu_parity")
    monkeypatch.setenv("SIMSOPT_BACKEND_STRICT", "1")
    monkeypatch.delenv("STAGE2_BACKEND", raising=False)
    invalidate_backend_cache()
    wireframe, _ = _wireframe_case()

    try:
        assert WireframeFieldJAX._simsopt_jax_native_field is False
        with pytest.raises(RuntimeError, match=r"MagneticFieldSum.*CPU-only"):
            MagneticFieldSum(
                [
                    WireframeFieldJAX(wireframe),
                    ToroidalFieldJAX(R0=1.3, B0=0.8),
                ]
            )
    finally:
        invalidate_backend_cache()


def test_wireframefield_jax_runs_under_strict_transfer_guard():
    """Compiled hot paths run after explicit host-to-device staging."""

    wireframe, points = _wireframe_case()
    jax_field = WireframeFieldJAX(wireframe)
    jax_field.set_points(points)
    for arr in (
        jax_field._points_device,
        jax_field._nodes_device,
        jax_field._segments_device,
        jax_field._seg_signs_device,
        jax_field._currents_device,
    ):
        arr.block_until_ready()

    jnp.asarray(jax_field.B()).block_until_ready()
    jnp.asarray(jax_field.dB_by_dX()).block_until_ready()
    jnp.asarray(jax_field.dB_by_dsegmentcurrents(0)[0]).block_until_ready()

    with jax.transfer_guard("disallow"):
        jax_field.clear_cached_properties()
        jnp.asarray(jax_field.B()).block_until_ready()
        jnp.asarray(jax_field.dB_by_dX()).block_until_ready()
        jnp.asarray(jax_field.dB_by_dsegmentcurrents(0)[0]).block_until_ready()
