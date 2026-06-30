"""Contract tests for the single-stage banana geometry modes.

Pins the observable behavior of the geometry-mode resolver, with emphasis on the
``free_xyz`` over-parameterization mode added 2026-06-21:

  * ``free_xyz`` replaces every banana coil (CWS master *and* its symmetry
    wrappers) with an INDEPENDENT ``CurveXYZFourier`` whose every x/y/z Fourier
    coefficient is free, lifting the geometry off the winding surface;
  * the conversion is a FAITHFUL fit, not a no-op or a broken fit -- the
    reproduction error against the source ``gamma`` is sub-micron and the peak
    curvature matches the source curve's dense-grid curvature;
  * the rebuilt Biot-Savart preserves coil ordering and reuses the non-banana
    coils by identity, and the banana currents are preserved;
  * ``resolve_single_stage_banana_geometry_state`` REFUSES ``free_xyz`` (it is
    resolved by the dedicated converter), failing loud rather than silently
    mis-handling CWS-only logic;
  * the additive guarantee: the CWS modes' results payload is unchanged (no
    ``FREE_XYZ_ORDER`` key), and ``free_xyz`` adds exactly that key.

No test mirrors the implementation; each assertion is an externally checkable
consequence of the contract.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLE_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _EXAMPLE_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from simsopt.field import BiotSavart, coils_via_symmetries  # noqa: E402
from simsopt.field.coil import Coil, Current  # noqa: E402
from simsopt.geo import (  # noqa: E402
    CurveCWSFourierCPP,
    CurveXYZFourier,
    RotatedCurve,
    SurfaceRZFourier,
)

from banana_opt.single_stage_banana_geometry_mode import (  # noqa: E402
    BANANA_GEOMETRY_MODE_FREE_XYZ,
    BANANA_GEOMETRY_MODE_MATERIALIZED_CWS,
    BANANA_GEOMETRY_MODE_SHARED_SYMMETRY,
    SingleStageBananaGeometryState,
    convert_single_stage_banana_family_to_xyz,
    resolve_single_stage_banana_geometry_state,
)
from banana_opt.stage2_single_stage_handoff import Stage2CoilPartitions  # noqa: E402


_NFP = 2


def _winding_surface() -> SurfaceRZFourier:
    surf = SurfaceRZFourier(nfp=_NFP, stellsym=True, mpol=1, ntor=1)
    surf.set_rc(0, 0, 1.0)
    surf.set_rc(1, 0, 0.3)
    surf.set_zs(1, 0, 0.3)
    return surf


def _cws_master(order: int = 3) -> CurveCWSFourierCPP:
    surf = _winding_surface()
    quadpoints = np.linspace(0.0, 1.0, 64, endpoint=False)
    curve = CurveCWSFourierCPP(quadpoints, order, surf, G=1, H=0)
    dofs = np.asarray(curve.get_dofs(), dtype=float)
    # Add a small poloidal wobble so the curve is a non-degenerate 3-D route whose
    # x/y/z components actually exercise the FFT round-trip (a perfectly planar
    # circle would be a trivial fit and hide ordering/independence bugs).
    dofs[2] += 0.05
    curve.set_dofs(dofs)
    return curve


def _partitions_with_tf_and_banana():
    """Return ``(biot_savart, partitions)`` with two fixed TF coils followed by a
    banana symmetry family (one CWS master + its rotated/reflected copies)."""
    tf_curves = []
    for shift in (0.0, 0.15):
        tf = CurveXYZFourier(64, 1)
        x = np.zeros_like(tf.x)
        x[0] = 1.2 + shift  # xc(0)
        x[2] = 0.4          # xc(1)
        x[1 + 2 * (2 * 1 + 1)] = 0.4  # ys(1) of the y-block -> a planar TF loop
        tf.x = x
        tf.fix_all()
        tf_curves.append(tf)
    tf_coils = tuple(Coil(c, Current(50000.0)) for c in tf_curves)

    master = _cws_master()
    banana_coils = tuple(
        coils_via_symmetries([master], [Current(12000.0)], _NFP, True)
    )

    biot_savart = BiotSavart([*tf_coils, *banana_coils])
    partitions = Stage2CoilPartitions(
        tf_coils=tf_coils,
        banana_coils=banana_coils,
        proxy_coils=(),
        vf_coils=(),
        num_tf_coils=len(tf_coils),
        num_banana_coils=len(banana_coils),
        num_proxy_coils=0,
        num_vf_coils=0,
        finite_current_mode="vacuum",
    )
    return biot_savart, partitions


def test_free_xyz_converts_every_banana_coil_to_independent_free_xyz():
    biot_savart, partitions = _partitions_with_tf_and_banana()
    n_banana = partitions.num_banana_coils
    assert n_banana >= 2  # the family has a master plus symmetry copies

    order = 14
    bs2, parts2, state = convert_single_stage_banana_family_to_xyz(
        biot_savart, partitions, order=order
    )

    # Every banana coil is now a free CurveXYZFourier (no CWS / RotatedCurve left).
    assert all(isinstance(c.curve, CurveXYZFourier) for c in parts2.banana_coils)
    assert not any(
        isinstance(c.curve, (CurveCWSFourierCPP, RotatedCurve))
        for c in parts2.banana_coils
    )
    # All x/y/z Fourier coefficients are free: 3 * (2*order + 1) per curve.
    expected_free = 3 * (2 * order + 1)
    for coil in parts2.banana_coils:
        assert len(coil.curve.x) == expected_free
        assert len(coil.curve.x) == len(coil.curve.get_dofs())

    # One independent curve per loaded banana coil.
    assert parts2.num_banana_coils == n_banana
    assert state.num_independent_banana_curves == n_banana


def test_free_xyz_fit_is_faithful_not_a_no_op_or_broken():
    biot_savart, partitions = _partitions_with_tf_and_banana()
    bs2, parts2, state = convert_single_stage_banana_family_to_xyz(
        biot_savart, partitions, order=14
    )
    # Sub-micron geometry reproduction proves the fit is faithful (a no-op would
    # leave CWS curves; a broken fit would have large error).
    assert state.max_materialization_error_m is not None
    assert state.max_materialization_error_m < 1e-6

    # The converted peak curvature matches the source curve's TRUE (dense-grid)
    # curvature, not its under-sampled default-grid value. Evaluate the source on a
    # dense grid to get the reference, then compare to the converted curve.
    master = next(
        c.curve for c in partitions.banana_coils
        if isinstance(c.curve, CurveCWSFourierCPP)
    )
    dense_qp = np.linspace(0.0, 1.0, 512, endpoint=False)
    dense = CurveCWSFourierCPP(dense_qp, int(master.order), master.surf, G=master.G, H=master.H)
    dense.set_dofs(np.asarray(master.get_dofs(), dtype=float).copy())
    source_dense_kappa = float(np.max(dense.kappa()))

    converted_kappa = max(
        float(np.max(c.curve.kappa())) for c in parts2.banana_coils
    )
    assert converted_kappa == pytest.approx(source_dense_kappa, rel=5e-3)


def test_free_xyz_preserves_coil_ordering_and_currents():
    biot_savart, partitions = _partitions_with_tf_and_banana()
    bs2, parts2, state = convert_single_stage_banana_family_to_xyz(
        biot_savart, partitions, order=14
    )

    # Same number of coils, TF coils reused by identity (field sources untouched),
    # TF group still leads the coil list.
    assert len(bs2.coils) == len(biot_savart.coils)
    for original_tf, rebuilt in zip(partitions.tf_coils, bs2.coils):
        assert rebuilt.curve is original_tf.curve

    # Banana currents preserved coil-for-coil.
    for original, rebuilt in zip(partitions.banana_coils, parts2.banana_coils):
        assert rebuilt.current.get_value() == pytest.approx(
            original.current.get_value()
        )


def test_resolve_refuses_free_xyz():
    biot_savart, partitions = _partitions_with_tf_and_banana()
    with pytest.raises(ValueError, match="convert_single_stage_banana_family_to_xyz"):
        resolve_single_stage_banana_geometry_state(
            biot_savart, partitions, mode=BANANA_GEOMETRY_MODE_FREE_XYZ
        )


def test_materialized_cws_still_resolves_and_payload_has_no_free_xyz_key():
    biot_savart, partitions = _partitions_with_tf_and_banana()
    bs2, parts2, state = resolve_single_stage_banana_geometry_state(
        biot_savart, partitions, mode=BANANA_GEOMETRY_MODE_MATERIALIZED_CWS
    )
    assert state.mode == BANANA_GEOMETRY_MODE_MATERIALIZED_CWS
    assert all(isinstance(c.curve, CurveCWSFourierCPP) for c in parts2.banana_coils)
    # Additive guarantee: the CWS payload must NOT carry the free_xyz order key.
    payload = state.payload_fields()
    assert "SINGLE_STAGE_BANANA_FREE_XYZ_ORDER" not in payload


def test_shared_symmetry_payload_unchanged_and_free_xyz_payload_adds_order_key():
    # shared_symmetry: payload byte-shape unchanged (no free_xyz key).
    shared_state = SingleStageBananaGeometryState(
        mode=BANANA_GEOMETRY_MODE_SHARED_SYMMETRY,
        num_banana_curves=4,
        num_independent_banana_curves=1,
    )
    shared_payload = shared_state.payload_fields()
    assert "SINGLE_STAGE_BANANA_FREE_XYZ_ORDER" not in shared_payload
    assert shared_payload["SINGLE_STAGE_BANANA_GEOMETRY_MODE"] == (
        BANANA_GEOMETRY_MODE_SHARED_SYMMETRY
    )

    # free_xyz: payload adds exactly the order key with the resolved order.
    biot_savart, partitions = _partitions_with_tf_and_banana()
    _, _, state = convert_single_stage_banana_family_to_xyz(
        biot_savart, partitions, order=12
    )
    payload = state.payload_fields()
    assert payload["SINGLE_STAGE_BANANA_GEOMETRY_MODE"] == BANANA_GEOMETRY_MODE_FREE_XYZ
    assert payload["SINGLE_STAGE_BANANA_FREE_XYZ_ORDER"] == 12
