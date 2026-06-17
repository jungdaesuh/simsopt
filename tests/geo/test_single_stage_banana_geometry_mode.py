import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = REPO_ROOT / "examples" / "single_stage_optimization"
EXAMPLES_ROOT_STR = str(EXAMPLES_ROOT)
if EXAMPLES_ROOT_STR not in sys.path:
    sys.path.insert(0, EXAMPLES_ROOT_STR)

from banana_opt.single_stage_banana_geometry_mode import (  # noqa: E402
    BANANA_GEOMETRY_MODE_MATERIALIZED_CWS,
    BANANA_GEOMETRY_MODE_SHARED_SYMMETRY,
    materialize_cws_symmetry_curve,
    resolve_single_stage_banana_geometry_state,
)
from banana_opt.single_stage_geometry import (  # noqa: E402
    evaluate_single_stage_hardware_snapshot,
    evaluate_single_stage_search_hardware_snapshot,
)
from banana_opt.stage2_single_stage_handoff import Stage2CoilPartitions  # noqa: E402
from simsopt.field import BiotSavart  # noqa: E402
from simsopt.field.coil import Current, coils_via_symmetries  # noqa: E402
from simsopt.geo import CurveCWSFourierCPP, RotatedCurve, SurfaceRZFourier  # noqa: E402


class _FakeDistanceObjective:
    def __init__(self, distance):
        self._distance = float(distance)

    def shortest_distance(self):
        return self._distance


class _FakeKappaCurve:
    def __init__(self, kappa):
        self._kappa = np.asarray(kappa, dtype=float)

    def kappa(self):
        return self._kappa


def _base_cws_curve():
    surface = SurfaceRZFourier(nfp=5, stellsym=True, mpol=1, ntor=0)
    curve = CurveCWSFourierCPP(
        np.linspace(0.0, 1.0, 64, endpoint=False),
        order=2,
        surf=surface,
    )
    dofs = curve.get_dofs().copy()
    dofs[:] = np.array(
        [
            0.03,
            0.01,
            -0.004,
            0.007,
            0.002,
            0.11,
            -0.02,
            0.005,
            0.003,
            -0.006,
        ]
    )
    curve.set_dofs(dofs)
    return curve


def _stage2_partitions(banana_coils):
    return Stage2CoilPartitions(
        tf_coils=(),
        banana_coils=tuple(banana_coils),
        proxy_coils=(),
        vf_coils=(),
        num_tf_coils=0,
        num_banana_coils=len(banana_coils),
        num_proxy_coils=0,
        num_vf_coils=0,
        finite_current_mode="vacuum",
    )


def test_materialize_cws_symmetry_curve_matches_symmetry_wrappers():
    base_curve = _base_cws_curve()
    banana_coils = coils_via_symmetries(
        [base_curve],
        [Current(1200.0)],
        base_curve.surf.nfp,
        base_curve.surf.stellsym,
    )

    for coil in banana_coils:
        materialized_curve = materialize_cws_symmetry_curve(coil.curve)
        assert isinstance(materialized_curve, CurveCWSFourierCPP)
        np.testing.assert_allclose(
            materialized_curve.gamma(),
            coil.curve.gamma(),
            atol=3.0e-14,
            rtol=0.0,
        )
        assert materialized_curve is not coil.curve


def test_materialized_mode_rebuilds_independent_banana_curves_in_biot_savart_order():
    base_curve = _base_cws_curve()
    banana_coils = coils_via_symmetries(
        [base_curve],
        [Current(1200.0)],
        base_curve.surf.nfp,
        base_curve.surf.stellsym,
    )
    biot_savart = BiotSavart(banana_coils)
    partitions = _stage2_partitions(banana_coils)

    rebuilt_bs, rebuilt_partitions, state = resolve_single_stage_banana_geometry_state(
        biot_savart,
        partitions,
        mode=BANANA_GEOMETRY_MODE_MATERIALIZED_CWS,
    )

    assert state.mode == BANANA_GEOMETRY_MODE_MATERIALIZED_CWS
    assert state.num_banana_curves == 10
    assert state.num_independent_banana_curves == 10
    assert state.max_materialization_error_m is not None
    assert state.max_materialization_error_m < 3.0e-14
    assert len(rebuilt_bs.coils) == len(banana_coils)
    assert tuple(rebuilt_bs.coils) == tuple(rebuilt_partitions.banana_coils)
    assert all(
        isinstance(coil.curve, CurveCWSFourierCPP)
        for coil in rebuilt_partitions.banana_coils
    )
    assert not any(
        isinstance(coil.curve, RotatedCurve) for coil in rebuilt_partitions.banana_coils
    )


def test_shared_symmetry_mode_preserves_loaded_objects():
    base_curve = _base_cws_curve()
    banana_coils = coils_via_symmetries(
        [base_curve],
        [Current(1200.0)],
        base_curve.surf.nfp,
        base_curve.surf.stellsym,
    )
    biot_savart = BiotSavart(banana_coils)
    partitions = _stage2_partitions(banana_coils)

    rebuilt_bs, rebuilt_partitions, state = resolve_single_stage_banana_geometry_state(
        biot_savart,
        partitions,
        mode=BANANA_GEOMETRY_MODE_SHARED_SYMMETRY,
    )

    assert rebuilt_bs is biot_savart
    assert rebuilt_partitions is partitions
    assert state.mode == BANANA_GEOMETRY_MODE_SHARED_SYMMETRY
    assert state.num_banana_curves == 10
    assert state.num_independent_banana_curves == 1


def test_hardware_snapshot_uses_worst_curvature_across_materialized_curves():
    low_curvature_curve = _FakeKappaCurve([12.0, 18.0])
    high_curvature_curve = _FakeKappaCurve([45.0, 72.0])

    snapshot = evaluate_single_stage_hardware_snapshot(
        _FakeDistanceObjective(0.08),
        0.0462,
        _FakeDistanceObjective(0.03),
        0.01,
        {"outer_vessel_gap": 0.05},
        0.04,
        low_curvature_curve,
        100.0,
        banana_curves=(low_curvature_curve, high_curvature_curve),
    )

    assert snapshot["max_curvature"] == 72.0


def test_search_hardware_snapshot_accepts_materialized_banana_curves():
    low_curvature_curve = _FakeKappaCurve([12.0, 18.0])
    high_curvature_curve = _FakeKappaCurve([45.0, 72.0])

    snapshot = evaluate_single_stage_search_hardware_snapshot(
        {
            "constraint_names": ["coil_length_upper_bound"],
            "dual_update_values": np.array([0.0]),
            "search_hardware_constraint_payload_kind": "penalty_objective",
        },
        cc_dist=0.0462,
        cs_dist=0.01,
        ss_dist=0.04,
        curvature_threshold=100.0,
        banana_curves=(low_curvature_curve, high_curvature_curve),
    )

    assert snapshot["max_curvature"] == 72.0
