import importlib
import importlib.util
import json
import sys
import tempfile
import time
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from simsopt.field import BiotSavart, Coil, Current
from simsopt.geo import CurveXYZFourier


EXAMPLE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "single_stage_optimization"
)
WRAPPER_PATH = EXAMPLE_ROOT / "run_stage2_to_single_stage.py"
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))


def load_module(path: Path, stem: str):
    spec = importlib.util.spec_from_file_location(f"{stem}_{uuid.uuid4().hex}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_wrapper_module():
    return load_module(WRAPPER_PATH, "run_stage2_to_single_stage")


def load_hardware_schema_module():
    return importlib.import_module("banana_opt.hardware_constraint_schema")


def load_artifact_contracts_module():
    return importlib.import_module("banana_opt.artifact_contracts")


def load_handoff_module():
    return importlib.import_module("banana_opt.stage2_single_stage_handoff")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_circle_curve(*, center, radius, normal):
    curve = CurveXYZFourier(96, 1)
    center_x, center_y, center_z = center
    if normal == "z":
        curve.set_dofs(
            [
                center_x,
                radius,
                0.0,
                center_y,
                radius,
                0.0,
                center_z,
                0.0,
                0.0,
            ]
        )
    elif normal == "x":
        curve.set_dofs(
            [
                center_x,
                0.0,
                0.0,
                center_y,
                radius,
                0.0,
                center_z,
                0.0,
                radius,
            ]
        )
    else:
        raise ValueError(f"Unsupported normal {normal!r}.")
    curve.fix_all()
    return curve


def _build_round_trip_seed(
    seed_dir: Path,
    *,
    include_proxy_vf: bool,
) -> tuple[Path, dict, np.ndarray, np.ndarray]:
    tf_coils = [
        Coil(
            _make_circle_curve(
                center=(0.9 + 0.01 * index, 0.0, 0.02 * ((index % 4) - 1.5)),
                radius=0.18,
                normal="z",
            ),
            Current(8.0e4),
        )
        for index in range(20)
    ]
    banana_coils = [
        Coil(_make_circle_curve(center=(1.02, 0.0, -0.08), radius=0.07, normal="z"), Current(1.1e4)),
        Coil(_make_circle_curve(center=(1.02, 0.0, 0.08), radius=0.07, normal="z"), Current(-1.1e4)),
    ]
    proxy_coils = (
        [
            Coil(
                _make_circle_curve(center=(0.82, 0.0, 0.0), radius=0.05, normal="z"),
                Current(9.0e3),
            )
        ]
        if include_proxy_vf
        else []
    )
    vf_coils = (
        [
            Coil(
                _make_circle_curve(center=(1.15, 0.0, 0.0), radius=0.22, normal="x"),
                Current(-5.0e2),
            )
        ]
        if include_proxy_vf
        else []
    )
    coils = [*tf_coils, *banana_coils, *proxy_coils, *vf_coils]
    for coil in coils:
        coil.current.fix_all()
    bs = BiotSavart(coils)
    points = np.array(
        [
            [0.25, 0.10, -0.15],
            [0.35, -0.05, 0.20],
            [0.55, 0.15, 0.05],
            [0.70, -0.10, -0.25],
        ],
        dtype=float,
    )
    bs.set_points(points)
    expected_field = bs.B().copy()
    stage2_bs_path = seed_dir / "biot_savart_opt.json"
    bs.save(str(stage2_bs_path))
    stage2_results = {
        "PLASMA_SURF_FILENAME": "demo.nc",
        "NUM_TF_COILS": len(tf_coils),
        "NUM_BANANA_COILS": len(banana_coils),
        "NUM_PROXY_COILS": len(proxy_coils),
        "NUM_VF_COILS": len(vf_coils),
        "FINITE_CURRENT_MODE": (
            "wataru_proxy_field" if include_proxy_vf else "boozer_surrogate"
        ),
    }
    _write_json(stage2_bs_path.with_name("results.json"), stage2_results)
    return stage2_bs_path, stage2_results, points, expected_field


def _bootability_status(
    handoff_module,
    *,
    stage: str,
    reason: str,
    bootable: bool,
    iota_feasible: bool,
    solved_iota: float | None,
    self_intersecting: bool | None = None,
) -> dict[str, object]:
    abs_iota_error = None
    if solved_iota is not None:
        abs_iota_error = abs(float(solved_iota) - 0.2)
    return {
        "BOOZER_BOOTABLE": bootable,
        "IOTA_FEASIBLE": iota_feasible,
        "BOOTABILITY_REASON": reason,
        "BOOTABILITY_STAGE": stage,
        "BOOTABILITY_TARGET_IOTA": 0.2,
        "BOOTABILITY_SOLVED_IOTA": solved_iota,
        "BOOTABILITY_SELF_INTERSECTING": self_intersecting,
        "BOOTABILITY_SOLVE_SUCCESS": bootable,
        "BOOTABILITY_ABS_IOTA_ERROR": abs_iota_error,
        "BOOTABILITY_ERROR_TYPE": None,
        "BOOTABILITY_ERROR_MESSAGE": None,
    }


class HandoffSchemaTests(unittest.TestCase):
    def test_build_bootability_recovery_payload_fields_shapes_all_expected_keys(self):
        module = load_hardware_schema_module()

        payload = module.build_bootability_recovery_payload_fields(
            {
                "BOOZER_BOOTABLE": True,
                "IOTA_FEASIBLE": False,
                "BOOTABILITY_REASON": "iota_mismatch",
                "BOOTABILITY_STAGE": "probe",
                "BOOTABILITY_TARGET_IOTA": 0.2,
                "BOOTABILITY_SOLVED_IOTA": 0.18,
                "BOOTABILITY_SELF_INTERSECTING": False,
                "BOOTABILITY_SOLVE_SUCCESS": True,
                "BOOTABILITY_ABS_IOTA_ERROR": 0.02,
                "BOOTABILITY_ERROR_TYPE": None,
                "BOOTABILITY_ERROR_MESSAGE": None,
            },
            stage2_bs_path="/tmp/stage2/biot_savart_opt.json",
            stage2_results_path="/tmp/stage2/results.json",
            recovery_attempted=True,
            recovery_succeeded=False,
            recovery_iters=7,
            recovery_termination_reason="not_bootable_after_budget",
        )

        self.assertEqual(
            module.bootability_recovery_payload_field_names(),
            (
                "BOOZER_BOOTABLE",
                "IOTA_FEASIBLE",
                "BOOTABILITY_REASON",
                "BOOTABILITY_STAGE",
                "BOOTABILITY_TARGET_IOTA",
                "BOOTABILITY_SOLVED_IOTA",
                "BOOTABILITY_SELF_INTERSECTING",
                "BOOTABILITY_SOLVE_SUCCESS",
                "BOOTABILITY_ABS_IOTA_ERROR",
                "BOOTABILITY_ERROR_TYPE",
                "BOOTABILITY_ERROR_MESSAGE",
                "STAGE2_BS_PATH",
                "STAGE2_RESULTS_PATH",
                "RECOVERY_ATTEMPTED",
                "RECOVERY_SUCCEEDED",
                "RECOVERY_ITERS",
                "RECOVERY_TERMINATION_REASON",
            ),
        )
        self.assertTrue(payload["BOOZER_BOOTABLE"])
        self.assertFalse(payload["IOTA_FEASIBLE"])
        self.assertEqual(payload["BOOTABILITY_REASON"], "iota_mismatch")
        self.assertAlmostEqual(payload["BOOTABILITY_ABS_IOTA_ERROR"], 0.02)
        self.assertEqual(payload["RECOVERY_ITERS"], 7)
        self.assertEqual(
            payload["RECOVERY_TERMINATION_REASON"],
            "not_bootable_after_budget",
        )

    def test_upgrade_legacy_stage2_artifact_results_backfills_handoff_defaults(self):
        module = load_artifact_contracts_module()

        upgraded = module.upgrade_legacy_stage2_artifact_results({})

        self.assertIsNone(upgraded["BOOZER_BOOTABLE"])
        self.assertIsNone(upgraded["BOOTABILITY_REASON"])
        self.assertFalse(upgraded["RECOVERY_ATTEMPTED"])
        self.assertFalse(upgraded["RECOVERY_SUCCEEDED"])
        self.assertIsNone(upgraded["RECOVERY_ITERS"])
        self.assertIsNone(upgraded["RECOVERY_TERMINATION_REASON"])
        self.assertFalse(upgraded["STAGE2_SECONDARY_ARTIFACT_PRESERVED"])
        self.assertIsNone(upgraded["STAGE2_SECONDARY_ARTIFACT_REASON"])
        self.assertIsNone(upgraded["STAGE2_SECONDARY_ARTIFACT_SOURCE"])
        self.assertIsNone(upgraded["STAGE2_SECONDARY_BS_PATH"])
        self.assertIsNone(upgraded["STAGE2_SECONDARY_RESULTS_PATH"])
        self.assertEqual(upgraded["FINITE_CURRENT_MODE"], "wataru_proxy_field")
        self.assertEqual(upgraded["FINITE_CURRENT_MODE_SOURCE"], "legacy_assumed_default")
        self.assertEqual(upgraded["BOOZER_CURRENT_CONVENTION"], "mu0")
        self.assertEqual(upgraded["NUM_PROXY_COILS"], 0)
        self.assertEqual(upgraded["NUM_VF_COILS"], 0)
        self.assertEqual(upgraded["PROXY_PLASMA_CURRENT_A"], 0.0)
        self.assertEqual(upgraded["VF_CURRENT_A"], 0.0)
        self.assertIsNone(upgraded["VF_TEMPLATE_PATH"])


class HandoffModuleTests(unittest.TestCase):
    @staticmethod
    def _fixed_current(current_A: float):
        return SimpleNamespace(get_value=lambda: float(current_A))

    def _fixed_current_coil(self, current_A: float):
        return SimpleNamespace(current=self._fixed_current(current_A))

    def _bootability_smoke_inputs(self, *, include_proxy_vf: bool):
        tf_coils = [self._fixed_current_coil(8.0e4) for _ in range(20)]
        banana_coils = [
            self._fixed_current_coil(1.1e4),
            self._fixed_current_coil(-1.1e4),
        ]
        proxy_coils = [self._fixed_current_coil(9.0e3)] if include_proxy_vf else []
        vf_coils = [self._fixed_current_coil(-5.0e2)] if include_proxy_vf else []
        fake_bs = SimpleNamespace(coils=[*tf_coils, *banana_coils, *proxy_coils, *vf_coils])
        stage2_artifact_results = {
            "PLASMA_SURF_FILENAME": "demo.nc",
            "TF_CURRENT_A": 8.0e4,
            "MAJOR_RADIUS": 0.976,
            "TOROIDAL_FLUX": 0.24,
            "banana_surf_radius": 0.21,
            "CURVATURE_THRESHOLD": 100.0,
        }
        if include_proxy_vf:
            stage2_artifact_results.update(
                {
                    "NUM_TF_COILS": 20,
                    "NUM_BANANA_COILS": 2,
                    "NUM_PROXY_COILS": 1,
                    "NUM_VF_COILS": 1,
                    "FINITE_CURRENT_MODE": "wataru_proxy_field",
                    "PROXY_PLASMA_CURRENT_A": 9.0e3,
                    "VF_CURRENT_A": 5.0e2,
                }
            )
        return tf_coils, fake_bs, stage2_artifact_results

    def _assert_restored_fake_boozer_surface(self, boozer_surface):
        np.testing.assert_allclose(boozer_surface.surface.x, [1.0, -2.0])
        self.assertAlmostEqual(boozer_surface.res["iota"], 0.21)
        self.assertAlmostEqual(boozer_surface.res["G"], 0.35)
        self.assertTrue(boozer_surface.res["success"])
        self.assertFalse(boozer_surface.need_to_run_code)

    def _assert_failed_boozer_attempt(self, attempt):
        self.assertFalse(attempt.solve_success)
        self.assertAlmostEqual(attempt.solved_iota, 0.41)
        self.assertAlmostEqual(attempt.solved_G, 0.72)
        self.assertIsNone(attempt.error_type)

    def test_classify_bootability_result_rejects_iota_mismatch(self):
        module = load_handoff_module()

        status = module.classify_bootability_result(
            module.BoozerInitializationResult(
                boozer_surface=None,
                solve_success=True,
                self_intersecting=False,
                success=True,
                solved_iota=0.12,
                solved_G=1.0,
                volume=0.1,
            ),
            stage=module.BOOTABILITY_STAGE_PROBE,
            target_iota=0.2,
            iota_tolerance=1.0e-3,
        )

        self.assertFalse(module.bootability_passes(status))
        self.assertEqual(status["BOOTABILITY_REASON"], module.BOOTABILITY_REASON_IOTA_MISMATCH)
        self.assertAlmostEqual(status["BOOTABILITY_ABS_IOTA_ERROR"], 0.08)

    def test_attempt_initialize_boozer_surface_keeps_probe_failures_visible(self):
        module = load_handoff_module()
        surf_prev = SimpleNamespace(
            quadpoints_theta=np.array([0.0, 0.5]),
            quadpoints_phi=np.array([0.0, 0.2]),
            gamma=lambda: np.zeros((2, 2, 3), dtype=float),
        )

        class _FakeSurface:
            def __init__(
                self,
                *,
                mpol,
                ntor,
                nfp,
                stellsym,
                quadpoints_theta,
                quadpoints_phi,
                dofs=None,
            ):
                del mpol, ntor, nfp, stellsym
                self.quadpoints_theta = quadpoints_theta
                self.quadpoints_phi = quadpoints_phi
                self.dofs = (
                    np.zeros(2, dtype=float)
                    if dofs is None
                    else np.asarray(dofs, dtype=float)
                )
                self.x = np.zeros(2, dtype=float)
                self._gamma = np.zeros((2, 2, 3), dtype=float)

            def least_squares_fit(self, gamma):
                self._gamma = np.asarray(gamma, dtype=float)

            def gamma(self):
                return self._gamma.copy()

            def is_self_intersecting(self):
                return False

            def volume(self):
                return 0.1

        class _FakeVolume:
            def __init__(self, surface):
                self.surface = surface

        class _FakeBoozerSurface:
            def __init__(
                self,
                bs,
                surf,
                vol,
                vol_target,
                constraint_weight,
                options,
                I=0.0,
            ):
                del bs, vol, vol_target, constraint_weight, options, I
                self.surface = surf
                self.res = {"iota": 0.21, "G": 0.35, "success": True}

            def run_code(self, iota, G):
                del iota, G
                self.surface.x = np.array([9.0, -4.0], dtype=float)
                self.res["iota"] = 0.41
                self.res["G"] = 0.72
                self.res["success"] = False
                return {"success": False}

        result = module.attempt_initialize_boozer_surface(
            surf_prev,
            mpol=8,
            ntor=6,
            bs=object(),
            vol_target=0.1,
            constraint_weight=1.0,
            iota=0.2,
            G0=0.35,
            boozer_I=0.0,
            nfp=5,
            surface_cls=_FakeSurface,
            volume_cls=_FakeVolume,
            boozer_surface_cls=_FakeBoozerSurface,
        )

        self.assertFalse(result.solve_success)
        self.assertFalse(result.success)
        self.assertAlmostEqual(result.solved_iota, 0.41)
        self.assertAlmostEqual(result.solved_G, 0.72)
        self.assertIsNone(result.error_type)
        np.testing.assert_allclose(result.boozer_surface.surface.x, [9.0, -4.0])

    def test_attempt_initialize_boozer_surface_assigns_seed_dofs_after_construction(self):
        module = load_handoff_module()
        surf_prev = SimpleNamespace(
            quadpoints_theta=np.array([0.0, 0.5]),
            quadpoints_phi=np.array([0.0, 0.2]),
            gamma=lambda: np.zeros((2, 2, 3), dtype=float),
        )
        initial_surface_guess = SimpleNamespace(
            get_dofs=lambda: np.array([2.5, -1.5], dtype=float)
        )

        class _CtorRejectsRawArraySurface:
            assigned_dofs = []

            def __init__(
                self,
                *,
                mpol,
                ntor,
                nfp,
                stellsym,
                quadpoints_theta,
                quadpoints_phi,
                dofs=None,
            ):
                del mpol, ntor, nfp, stellsym
                if dofs is not None:
                    raise AssertionError(
                        "Warm-start regression: attempt_initialize_boozer_surface "
                        "should not pass raw arrays through the constructor."
                    )
                self.quadpoints_theta = quadpoints_theta
                self.quadpoints_phi = quadpoints_phi
                self._local_full_x = np.zeros(2, dtype=float)
                self._gamma = np.zeros((2, 2, 3), dtype=float)

            @property
            def local_full_x(self):
                return self._local_full_x

            @local_full_x.setter
            def local_full_x(self, value):
                resolved = np.asarray(value, dtype=float)
                self._local_full_x = resolved
                self.dofs = resolved
                type(self).assigned_dofs.append(resolved.copy())

            def least_squares_fit(self, gamma):
                self._gamma = np.asarray(gamma, dtype=float)

            def gamma(self):
                return self._gamma.copy()

            def is_self_intersecting(self):
                return False

            def volume(self):
                return 0.1

        class _FakeVolume:
            def __init__(self, surface):
                self.surface = surface

        class _FakeBoozerSurface:
            def __init__(
                self,
                bs,
                surf,
                vol,
                vol_target,
                constraint_weight,
                options,
                I=0.0,
            ):
                del bs, vol, vol_target, constraint_weight, options, I
                self.surface = surf
                self.res = {"iota": 0.2, "G": 0.35, "success": True}

            def run_code(self, iota, G):
                del iota, G
                return {"success": True}

        result = module.attempt_initialize_boozer_surface(
            surf_prev,
            mpol=8,
            ntor=6,
            bs=object(),
            vol_target=0.1,
            constraint_weight=1.0,
            iota=0.2,
            G0=0.35,
            boozer_I=0.0,
            initial_surface_guess=initial_surface_guess,
            nfp=5,
            surface_cls=_CtorRejectsRawArraySurface,
            volume_cls=_FakeVolume,
            boozer_surface_cls=_FakeBoozerSurface,
        )

        self.assertTrue(result.solve_success)
        self.assertTrue(result.success)
        self.assertGreaterEqual(len(_CtorRejectsRawArraySurface.assigned_dofs), 1)
        np.testing.assert_allclose(
            _CtorRejectsRawArraySurface.assigned_dofs[0],
            np.array([2.5, -1.5], dtype=float),
        )

    def test_run_boozer_with_failure_policy_accepts_cached_result_state(self):
        module = load_handoff_module()

        class _FakeBoozerSurface:
            def __init__(self):
                self.surface = SimpleNamespace(x=np.array([1.0, -2.0], dtype=float))
                self.res = {"iota": 0.21, "G": 0.35, "success": True}
                self.calls = []

            def run_code(self, iota, G):
                self.calls.append((float(iota), float(G)))
                return None

        boozer_surface = _FakeBoozerSurface()

        attempt = module.run_boozer_with_failure_policy(
            boozer_surface,
            0.21,
            0.35,
            failure_policy=module.BOOZER_FAILURE_POLICY_REPORT_FAILURE,
        )

        self.assertTrue(attempt.solve_success)
        self.assertAlmostEqual(attempt.solved_iota, 0.21)
        self.assertAlmostEqual(attempt.solved_G, 0.35)
        self.assertIsNone(attempt.error_type)
        self.assertEqual(boozer_surface.calls, [(0.21, 0.35)])

    def test_run_boozer_with_failure_policy_handles_fresh_surface_without_cached_res(self):
        module = load_handoff_module()

        class _FakeBoozerSurface:
            def __init__(self):
                self.surface = SimpleNamespace(x=np.array([1.0, -2.0], dtype=float))
                self.calls = []

            def run_code(self, iota, G):
                self.calls.append((float(iota), float(G)))
                self.res = {"iota": 0.21, "G": 0.35, "success": True}
                return None

        boozer_surface = _FakeBoozerSurface()

        attempt = module.run_boozer_with_failure_policy(
            boozer_surface,
            0.21,
            0.35,
            failure_policy=module.BOOZER_FAILURE_POLICY_REPORT_FAILURE,
        )

        self.assertTrue(attempt.solve_success)
        self.assertAlmostEqual(attempt.solved_iota, 0.21)
        self.assertAlmostEqual(attempt.solved_G, 0.35)
        self.assertIsNone(attempt.error_type)
        self.assertEqual(boozer_surface.calls, [(0.21, 0.35)])

    def test_run_boozer_with_failure_policy_restores_last_successful_state_on_failed_result(self):
        module = load_handoff_module()

        class _FakeBoozerSurface:
            def __init__(self):
                self.surface = SimpleNamespace(x=np.array([1.0, -2.0], dtype=float))
                self.res = {"iota": 0.21, "G": 0.35, "success": True}
                self.need_to_run_code = True
                self.calls = []

            def run_code(self, iota, G):
                self.calls.append((float(iota), float(G)))
                self.surface.x = np.array([9.0, -4.0], dtype=float)
                self.res["iota"] = 0.41
                self.res["G"] = 0.72
                self.res["success"] = False
                self.need_to_run_code = False
                return {"success": False}

        boozer_surface = _FakeBoozerSurface()
        last_successful_state = module.snapshot_boozer_solve_state(boozer_surface)

        attempt = module.run_boozer_with_failure_policy(
            boozer_surface,
            0.21,
            0.35,
            failure_policy=module.BOOZER_FAILURE_POLICY_RESTORE_LAST_SUCCESS,
            last_successful_state=last_successful_state,
        )

        self._assert_failed_boozer_attempt(attempt)
        self._assert_restored_fake_boozer_surface(boozer_surface)
        self.assertEqual(boozer_surface.calls, [(0.21, 0.35)])

    def test_run_boozer_with_failure_policy_restores_cached_failed_state_on_reported_failure(self):
        module = load_handoff_module()

        class _FakeBoozerSurface:
            def __init__(self):
                self.surface = SimpleNamespace(x=np.array([1.0, -2.0], dtype=float))
                self.res = {"iota": 0.21, "G": 0.35, "success": True}
                self.need_to_run_code = False
                self.calls = []

            def run_code(self, iota, G):
                self.calls.append((float(iota), float(G)))
                return None

        boozer_surface = _FakeBoozerSurface()
        last_successful_state = module.snapshot_boozer_solve_state(boozer_surface)
        boozer_surface.surface.x = np.array([9.0, -4.0], dtype=float)
        boozer_surface.res["iota"] = 0.41
        boozer_surface.res["G"] = 0.72
        boozer_surface.res["success"] = False
        boozer_surface.need_to_run_code = False

        attempt = module.run_boozer_with_failure_policy(
            boozer_surface,
            0.21,
            0.35,
            failure_policy=module.BOOZER_FAILURE_POLICY_RESTORE_LAST_SUCCESS,
            last_successful_state=last_successful_state,
        )

        self._assert_failed_boozer_attempt(attempt)
        self._assert_restored_fake_boozer_surface(boozer_surface)
        self.assertEqual(boozer_surface.calls, [(0.21, 0.35)])

    def test_probe_stage2_seed_bootability_reports_missing_metadata_without_loading_bs(self):
        module = load_handoff_module()

        status = module.probe_stage2_seed_bootability(
            stage2_bs_path="/tmp/demo/biot_savart_opt.json",
            stage2_artifact_results={"PLASMA_SURF_FILENAME": "demo.nc"},
            plasma_surf_filename="demo.nc",
            equilibria_dir="/tmp/equilibria",
            num_tf_coils=20,
            nphi=91,
            ntheta=32,
            mpol=8,
            ntor=6,
            vol_target=0.1,
            iota_target=0.2,
            iota_tolerance=5.0e-3,
            constraint_weight=1.0,
        )

        self.assertEqual(
            status["BOOTABILITY_REASON"],
            module.BOOTABILITY_REASON_MISSING_ARTIFACT_METADATA,
        )
        self.assertFalse(module.bootability_passes(status))

    def test_partition_loaded_stage2_coils_uses_recorded_proxy_and_vf_counts(self):
        module = load_handoff_module()
        coils = [object() for _ in range(24)]

        partitions = module.partition_loaded_stage2_coils(
            coils,
            stage2_results={
                "NUM_TF_COILS": 20,
                "NUM_BANANA_COILS": 2,
                "NUM_PROXY_COILS": 1,
                "NUM_VF_COILS": 1,
                "FINITE_CURRENT_MODE": "wataru_proxy_field",
            },
            requested_num_tf_coils=20,
        )

        self.assertEqual(len(partitions.tf_coils), 20)
        self.assertEqual(len(partitions.banana_coils), 2)
        self.assertEqual(len(partitions.proxy_coils), 1)
        self.assertEqual(len(partitions.vf_coils), 1)
        self.assertEqual(partitions.finite_current_mode, "wataru_proxy_field")

    def test_partition_loaded_stage2_coils_prefers_explicit_manifest(self):
        module = load_handoff_module()
        coils = [object() for _ in range(24)]
        manifest_payload = [
            {"role": "tf", "start": 0, "count": 20},
            {"role": "banana", "start": 20, "count": 2},
            {"role": "proxy", "start": 22, "count": 1},
            {"role": "vf", "start": 23, "count": 1},
        ]

        partitions = module.partition_loaded_stage2_coils(
            coils,
            stage2_results={
                "COIL_GROUPS": manifest_payload,
                # Legacy counts are deliberately wrong; manifest should win.
                "NUM_TF_COILS": 20,
                "NUM_BANANA_COILS": 99,
                "NUM_PROXY_COILS": 99,
                "NUM_VF_COILS": 99,
            },
            requested_num_tf_coils=20,
        )

        self.assertEqual(partitions.num_tf_coils, 20)
        self.assertEqual(partitions.num_banana_coils, 2)
        self.assertEqual(partitions.num_proxy_coils, 1)
        self.assertEqual(partitions.num_vf_coils, 1)
        self.assertFalse(partitions.coil_groups_manifest_is_legacy_inferred)

    def test_partition_loaded_stage2_coils_flags_legacy_inference(self):
        module = load_handoff_module()
        coils = [object() for _ in range(22)]

        partitions = module.partition_loaded_stage2_coils(
            coils,
            stage2_results={
                "NUM_TF_COILS": 20,
                "NUM_BANANA_COILS": 2,
                "NUM_PROXY_COILS": 0,
                "NUM_VF_COILS": 0,
            },
            requested_num_tf_coils=20,
        )

        self.assertTrue(partitions.coil_groups_manifest_is_legacy_inferred)

    def test_partition_loaded_stage2_coils_rejects_inconsistent_partition_total(self):
        module = load_handoff_module()
        coils = [object() for _ in range(22)]

        with self.assertRaisesRegex(
            ValueError,
            r"manifest expects 24 coils but the loaded BiotSavart artifact contains 22",
        ):
            module.partition_loaded_stage2_coils(
                coils,
                stage2_results={
                    "NUM_TF_COILS": 20,
                    "NUM_BANANA_COILS": 2,
                    "NUM_PROXY_COILS": 1,
                    "NUM_VF_COILS": 1,
                },
                requested_num_tf_coils=20,
            )

    def test_wataru_round_trip_field_parity_survives_stage2_write_and_reload(self):
        module = load_handoff_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            stage2_bs_path, stage2_results, points, expected_field = _build_round_trip_seed(
                Path(tmpdir),
                include_proxy_vf=True,
            )

            loaded_bs = module.load(str(stage2_bs_path))
            loaded_bs.set_points(points)
            actual_field = loaded_bs.B()
            partitions = module.partition_loaded_stage2_coils(
                loaded_bs.coils,
                stage2_results=stage2_results,
                requested_num_tf_coils=20,
            )

        np.testing.assert_allclose(actual_field, expected_field, rtol=1.0e-12, atol=1.0e-12)
        self.assertEqual(len(partitions.tf_coils), 20)
        self.assertEqual(len(partitions.banana_coils), 2)
        self.assertEqual(len(partitions.proxy_coils), 1)
        self.assertEqual(len(partitions.vf_coils), 1)
        self.assertEqual(partitions.finite_current_mode, "wataru_proxy_field")

    def test_round_trip_field_timing_smoke_covers_legacy_and_wataru_partition_shapes(self):
        module = load_handoff_module()

        timings: dict[str, float] = {}
        for label, include_proxy_vf in (
            ("legacy", False),
            ("wataru_proxy_field", True),
        ):
            with self.subTest(mode=label), tempfile.TemporaryDirectory() as tmpdir:
                stage2_bs_path, stage2_results, points, _ = _build_round_trip_seed(
                    Path(tmpdir),
                    include_proxy_vf=include_proxy_vf,
                )
                loaded_bs = module.load(str(stage2_bs_path))
                loaded_bs.set_points(points)
                start = time.perf_counter()
                field = loaded_bs.B()
                elapsed_s = time.perf_counter() - start
                partitions = module.partition_loaded_stage2_coils(
                    loaded_bs.coils,
                    stage2_results=stage2_results,
                    requested_num_tf_coils=20,
                )
                self.assertEqual(field.shape, (4, 3))
                self.assertGreaterEqual(elapsed_s, 0.0)
                timings[label] = elapsed_s
                if include_proxy_vf:
                    self.assertEqual(len(partitions.proxy_coils), 1)
                    self.assertEqual(len(partitions.vf_coils), 1)
                else:
                    self.assertEqual(len(partitions.proxy_coils), 0)
                    self.assertEqual(len(partitions.vf_coils), 0)

        self.assertEqual(set(timings), {"legacy", "wataru_proxy_field"})

    def test_probe_stage2_seed_bootability_smoke_legacy_donor_uses_remainder_partition(self):
        module = load_handoff_module()
        tf_coils, fake_bs, stage2_artifact_results = self._bootability_smoke_inputs(
            include_proxy_vf=False
        )
        fake_surface = SimpleNamespace(nfp=5)
        recorded = {}

        def fake_attempt_initialize_boozer_surface(
            surf_prev,
            mpol,
            ntor,
            bs,
            vol_target,
            constraint_weight,
            iota,
            G0,
            boozer_I=0.0,
            *,
            initial_surface_guess=None,
            nfp,
        ):
            recorded.update(
                bs=bs,
                vol_target=vol_target,
                constraint_weight=constraint_weight,
                iota=iota,
                G0=G0,
                boozer_I=boozer_I,
                initial_surface_guess=initial_surface_guess,
                nfp=nfp,
            )
            return module.BoozerInitializationResult(
                boozer_surface=SimpleNamespace(surface=SimpleNamespace(volume=lambda: 0.1)),
                solve_success=True,
                self_intersecting=False,
                success=True,
                solved_iota=0.2,
                solved_G=G0,
                volume=0.1,
            )

        with patch.object(
            module,
            "build_surface_configs",
            return_value=[{"initial_surface": fake_surface, "target_volume": 0.1}],
        ), patch.object(
            module,
            "attempt_initialize_boozer_surface",
            side_effect=fake_attempt_initialize_boozer_surface,
        ):
            status = module.probe_stage2_seed_bootability(
                stage2_bs_path="/tmp/legacy/biot_savart_opt.json",
                stage2_artifact_results=stage2_artifact_results,
                plasma_surf_filename="demo.nc",
                equilibria_dir="/tmp/equilibria",
                num_tf_coils=20,
                nphi=31,
                ntheta=16,
                mpol=8,
                ntor=6,
                vol_target=0.1,
                iota_target=0.2,
                iota_tolerance=5.0e-3,
                constraint_weight=1.0,
                boozer_I=0.0,
                bs_loader=lambda _path: fake_bs,
            )

        self.assertTrue(module.bootability_passes(status))
        self.assertEqual(recorded["bs"], fake_bs)
        self.assertEqual(recorded["nfp"], 5)
        self.assertAlmostEqual(recorded["G0"], module.compute_tf_G0(tf_coils))
        self.assertEqual(recorded["boozer_I"], 0.0)

    def test_probe_stage2_seed_bootability_smoke_wataru_donor_preserves_extra_coil_metadata(self):
        module = load_handoff_module()
        current_contracts = importlib.import_module("banana_opt.current_contracts")
        tf_coils, fake_bs, stage2_artifact_results = self._bootability_smoke_inputs(
            include_proxy_vf=True
        )
        fake_surface = SimpleNamespace(nfp=5)
        recorded = {}

        plasma_settings = current_contracts.resolve_plasma_current_settings(
            raw_boozer_I=None,
            plasma_current_A=None,
            finite_current_mode="wataru_proxy_field",
            default_plasma_current_A=9.0e3,
        )

        def fake_attempt_initialize_boozer_surface(
            surf_prev,
            mpol,
            ntor,
            bs,
            vol_target,
            constraint_weight,
            iota,
            G0,
            boozer_I=0.0,
            *,
            initial_surface_guess=None,
            nfp,
        ):
            recorded.update(
                bs=bs,
                G0=G0,
                boozer_I=boozer_I,
                initial_surface_guess=initial_surface_guess,
                nfp=nfp,
                total_loaded_coils=len(bs.coils),
            )
            return module.BoozerInitializationResult(
                boozer_surface=SimpleNamespace(surface=SimpleNamespace(volume=lambda: 0.1)),
                solve_success=True,
                self_intersecting=False,
                success=True,
                solved_iota=0.2,
                solved_G=G0,
                volume=0.1,
            )

        with patch.object(
            module,
            "build_surface_configs",
            return_value=[{"initial_surface": fake_surface, "target_volume": 0.1}],
        ), patch.object(
            module,
            "attempt_initialize_boozer_surface",
            side_effect=fake_attempt_initialize_boozer_surface,
        ):
            status = module.probe_stage2_seed_bootability(
                stage2_bs_path="/tmp/wataru/biot_savart_opt.json",
                stage2_artifact_results=stage2_artifact_results,
                plasma_surf_filename="demo.nc",
                equilibria_dir="/tmp/equilibria",
                num_tf_coils=20,
                nphi=31,
                ntheta=16,
                mpol=8,
                ntor=6,
                vol_target=0.1,
                iota_target=0.2,
                iota_tolerance=5.0e-3,
                constraint_weight=1.0,
                boozer_I=plasma_settings.boozer_I,
                bs_loader=lambda _path: fake_bs,
            )

        self.assertTrue(module.bootability_passes(status))
        self.assertEqual(recorded["total_loaded_coils"], 24)
        self.assertEqual(recorded["nfp"], 5)
        self.assertAlmostEqual(recorded["G0"], module.compute_tf_G0(tf_coils))
        self.assertAlmostEqual(recorded["boozer_I"], plasma_settings.boozer_I)

    def test_probe_stage2_seed_bootability_uses_loaded_surface_as_seed_surface(self):
        module = load_handoff_module()
        tf_coils, fake_bs, stage2_artifact_results = self._bootability_smoke_inputs(
            include_proxy_vf=False
        )
        warm_start_surface = SimpleNamespace(
            nfp=5,
            dofs=np.array([3.0, -2.0], dtype=float),
        )
        recorded = {}

        def fake_attempt_initialize_boozer_surface(
            surf_prev,
            mpol,
            ntor,
            bs,
            vol_target,
            constraint_weight,
            iota,
            G0,
            boozer_I=0.0,
            *,
            initial_surface_guess,
            nfp,
        ):
            recorded.update(
                surf_prev=surf_prev,
                initial_surface_guess=initial_surface_guess,
                iota=iota,
                G0=G0,
                boozer_I=boozer_I,
                nfp=nfp,
            )
            return module.BoozerInitializationResult(
                boozer_surface=SimpleNamespace(surface=SimpleNamespace(volume=lambda: 0.1)),
                solve_success=True,
                self_intersecting=False,
                success=True,
                solved_iota=0.2,
                solved_G=G0,
                volume=0.1,
            )

        with patch.object(
            module,
            "build_equilibrium_path",
            side_effect=AssertionError("warm-start probe should not read the equilibrium"),
        ), patch.object(
            module,
            "build_surface_configs",
            side_effect=AssertionError("warm-start probe should not rebuild a cold-start surface"),
        ), patch.object(
            module,
            "load_warm_start_boozer_seed",
            return_value=module.WarmStartBoozerSeed(
                surface=warm_start_surface,
                iota=0.2,
                G=module.compute_tf_G0(tf_coils),
                source_path=Path("/tmp/legacy/surf_opt_boozer_surface.json"),
            ),
        ), patch.object(
            module,
            "attempt_initialize_boozer_surface",
            side_effect=fake_attempt_initialize_boozer_surface,
        ):
            status = module.probe_stage2_seed_bootability(
                stage2_bs_path="/tmp/legacy/biot_savart_opt.json",
                stage2_artifact_results=stage2_artifact_results,
                plasma_surf_filename="demo.nc",
                equilibria_dir="/tmp/equilibria",
                num_tf_coils=20,
                nphi=31,
                ntheta=16,
                mpol=8,
                ntor=6,
                vol_target=0.1,
                iota_target=0.2,
                iota_tolerance=5.0e-3,
                constraint_weight=1.0,
                boozer_I=0.0,
                stage2_seed_surf_path="/tmp/legacy/surf_opt_boozer_surface.json",
                bs_loader=lambda _path: fake_bs,
            )

        self.assertTrue(module.bootability_passes(status))
        self.assertIs(recorded["surf_prev"], warm_start_surface)
        self.assertIs(recorded["initial_surface_guess"], warm_start_surface)
        self.assertAlmostEqual(recorded["iota"], 0.2)
        self.assertEqual(recorded["nfp"], 5)

    def test_probe_stage2_seed_bootability_uses_warm_start_boozer_surface_artifact(self):
        module = load_handoff_module()
        _, fake_bs, stage2_artifact_results = self._bootability_smoke_inputs(
            include_proxy_vf=False
        )
        warm_start_surface = SimpleNamespace(nfp=5)
        warm_start_path = Path(
            "/tmp/recovery/surf_best_feasible_outer_boozer_surface.json"
        )
        recorded = {}

        def fake_loader(path):
            if path == "/tmp/recovery/biot_savart_best_feasible.json":
                return fake_bs
            if path == str(warm_start_path):
                return SimpleNamespace(
                    surface=warm_start_surface,
                    res={"iota": 0.2003, "G": 0.377},
                )
            raise AssertionError(f"unexpected load path: {path}")

        def fake_attempt_initialize_boozer_surface(
            surf_prev,
            mpol,
            ntor,
            bs,
            vol_target,
            constraint_weight,
            iota,
            G0,
            boozer_I=0.0,
            *,
            initial_surface_guess=None,
            nfp,
        ):
            recorded.update(
                surf_prev=surf_prev,
                bs=bs,
                iota=iota,
                G0=G0,
                initial_surface_guess=initial_surface_guess,
                nfp=nfp,
            )
            return module.BoozerInitializationResult(
                boozer_surface=SimpleNamespace(surface=SimpleNamespace(volume=lambda: 0.1)),
                solve_success=True,
                self_intersecting=False,
                success=True,
                solved_iota=0.2003,
                solved_G=0.377,
                volume=0.1,
            )

        with patch.object(
            module,
            "build_equilibrium_path",
            side_effect=AssertionError("warm-start probe should not read the equilibrium"),
        ), patch.object(
            module,
            "build_surface_configs",
            side_effect=AssertionError("warm-start probe should not rebuild a cold-start surface"),
        ), patch.object(
            module,
            "attempt_initialize_boozer_surface",
            side_effect=fake_attempt_initialize_boozer_surface,
        ):
            status = module.probe_stage2_seed_bootability(
                stage2_bs_path="/tmp/recovery/biot_savart_best_feasible.json",
                stage2_artifact_results=stage2_artifact_results,
                plasma_surf_filename="demo.nc",
                equilibria_dir="/tmp/equilibria",
                num_tf_coils=20,
                nphi=31,
                ntheta=16,
                mpol=8,
                ntor=6,
                vol_target=0.1,
                iota_target=0.2,
                iota_tolerance=5.0e-3,
                constraint_weight=1.0,
                boozer_I=0.0,
                stage2_seed_surf_path=warm_start_path,
                bs_loader=fake_loader,
            )

        self.assertTrue(module.bootability_passes(status))
        self.assertIs(recorded["surf_prev"], warm_start_surface)
        self.assertIs(recorded["bs"], fake_bs)
        self.assertAlmostEqual(recorded["iota"], 0.2003)
        self.assertAlmostEqual(recorded["G0"], 0.377)
        self.assertIs(recorded["initial_surface_guess"], warm_start_surface)
        self.assertEqual(recorded["nfp"], 5)


class UnifiedRunnerTests(unittest.TestCase):
    def _stage2_seed_paths(self, root: Path) -> tuple[Path, Path]:
        stage2_dir = root / "stage2_seed"
        stage2_bs_path = stage2_dir / "biot_savart_opt.json"
        stage2_results_path = stage2_dir / "results.json"
        stage2_bs_path.parent.mkdir(parents=True, exist_ok=True)
        stage2_bs_path.write_text("{}", encoding="utf-8")
        _write_json(
            stage2_results_path,
            {
                "PLASMA_SURF_FILENAME": "demo.nc",
                "init_only": False,
            },
        )
        return stage2_bs_path, stage2_results_path

    def test_parse_args_accepts_seed_order_upgrade(self):
        wrapper = load_wrapper_module()

        args = wrapper.parse_args(
            [
                "--plasma-surf-filename",
                "demo.nc",
                "--stage2-bs-path",
                "/tmp/stage2/biot_savart_opt.json",
                "--seed-order-upgrade",
                "4",
            ]
        )

        self.assertEqual(args.seed_order_upgrade, 4)

    def test_parse_args_accepts_stage2_seed_surf_path(self):
        wrapper = load_wrapper_module()

        args = wrapper.parse_args(
            [
                "--plasma-surf-filename",
                "demo.nc",
                "--stage2-bs-path",
                "/tmp/stage2/biot_savart_opt.json",
                "--stage2-seed-surf-path",
                "/tmp/stage2/surf_opt_boozer_surface.json",
            ]
        )

        self.assertEqual(
            args.stage2_seed_surf_path,
            "/tmp/stage2/surf_opt_boozer_surface.json",
        )

    def test_probe_only_writes_summary_with_bootability_status(self):
        wrapper = load_wrapper_module()
        handoff = load_handoff_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            stage2_bs_path, _ = self._stage2_seed_paths(root)
            summary_root = root / "summary"

            with patch.object(
                wrapper,
                "build_probe_status",
                return_value=_bootability_status(
                    handoff,
                    stage=handoff.BOOTABILITY_STAGE_PROBE,
                    reason=handoff.BOOTABILITY_REASON_IOTA_MISMATCH,
                    bootable=False,
                    iota_feasible=False,
                    solved_iota=0.05,
                    self_intersecting=False,
                ),
            ):
                result = wrapper.main(
                    [
                        "--probe-only",
                        "--plasma-surf-filename",
                        "demo.nc",
                        "--stage2-bs-path",
                        str(stage2_bs_path),
                        "--output-root",
                        str(summary_root),
                    ]
                )

            self.assertEqual(result, 0)
            summary = json.loads(
                (summary_root / wrapper.DEFAULT_SUMMARY_JSON).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(summary["mode"], "probe_only")
            self.assertEqual(
                summary["bootability_probe"]["BOOTABILITY_REASON"],
                handoff.BOOTABILITY_REASON_IOTA_MISMATCH,
            )
            self.assertIsNone(summary["recovery"])
            self.assertIsNone(summary["full_single_stage"])

    def test_load_stage2_seed_metadata_for_handoff_backfills_legacy_tf_current_from_cli(self):
        wrapper = load_wrapper_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            stage2_bs_path, stage2_results_path = self._stage2_seed_paths(root)
            _write_json(
                stage2_results_path,
                {
                    "PLASMA_SURF_FILENAME": "demo.nc",
                    "init_only": False,
                },
            )
            args = wrapper.parse_args(
                [
                    "--plasma-surf-filename",
                    "demo.nc",
                    "--stage2-bs-path",
                    str(stage2_bs_path),
                    "--stage2-seed-tf-current-A",
                    "12345.0",
                    "--num-tf-coils",
                    "18",
                ]
            )

            _, stage2_results = wrapper.load_stage2_seed_metadata_for_handoff(
                args,
                stage2_bs_path=stage2_bs_path,
            )

            self.assertEqual(stage2_results["TF_CURRENT_A"], 12345.0)
            self.assertEqual(stage2_results["NUM_TF_COILS"], 18)
            self.assertEqual(stage2_results["TF_CURRENT_SUM_ABS_A"], 222210.0)

    def test_build_probe_status_uses_exact_boozer_semantics_for_negative_constraint_weight(self):
        wrapper = load_wrapper_module()

        args = wrapper.parse_args(
            [
                "--plasma-surf-filename",
                "demo.nc",
                "--stage2-bs-path",
                "/tmp/stage2/biot_savart_opt.json",
                "--constraint-weight",
                "-1.0",
            ]
        )

        with patch.object(wrapper, "probe_stage2_seed_bootability", return_value={}) as probe:
            wrapper.build_probe_status(
                args,
                stage2_bs_path=Path("/tmp/stage2/biot_savart_opt.json"),
                stage2_results={"PLASMA_SURF_FILENAME": "demo.nc"},
                stage="probe",
            )

        self.assertIsNone(probe.call_args.kwargs["constraint_weight"])

    def test_build_probe_status_derives_boozer_current_from_physical_current(self):
        wrapper = load_wrapper_module()

        args = wrapper.parse_args(
            [
                "--plasma-surf-filename",
                "demo.nc",
                "--stage2-bs-path",
                "/tmp/stage2/biot_savart_opt.json",
                "--plasma-current-A",
                "9000.0",
            ]
        )

        with patch.object(wrapper, "probe_stage2_seed_bootability", return_value={}) as probe:
            wrapper.build_probe_status(
                args,
                stage2_bs_path=Path("/tmp/stage2/biot_savart_opt.json"),
                stage2_results={
                    "PLASMA_SURF_FILENAME": "demo.nc",
                    "FINITE_CURRENT_MODE": "boozer_surrogate",
                    "PROXY_PLASMA_CURRENT_A": 0.0,
                },
                stage="probe",
            )

        self.assertAlmostEqual(
            probe.call_args.kwargs["boozer_I"],
            4.0e-7 * 3.141592653589793 * 9000.0,
        )

    def test_build_probe_status_forwards_stage2_seed_surface_path(self):
        wrapper = load_wrapper_module()

        args = wrapper.parse_args(
            [
                "--plasma-surf-filename",
                "demo.nc",
                "--stage2-bs-path",
                "/tmp/stage2/biot_savart_opt.json",
                "--stage2-seed-surf-path",
                "seed/surf_opt_boozer_surface.json",
            ]
        )

        with patch.object(wrapper, "probe_stage2_seed_bootability", return_value={}) as probe:
            wrapper.build_probe_status(
                args,
                stage2_bs_path=Path("/tmp/stage2/biot_savart_opt.json"),
                stage2_results={"PLASMA_SURF_FILENAME": "demo.nc"},
                stage="probe",
            )

        self.assertEqual(
            probe.call_args.kwargs["stage2_seed_surf_path"],
            Path("seed/surf_opt_boozer_surface.json").resolve(),
        )

    def test_build_probe_status_uses_stage2_proxy_current_default_in_wataru_mode(self):
        wrapper = load_wrapper_module()

        args = wrapper.parse_args(
            [
                "--plasma-surf-filename",
                "demo.nc",
                "--stage2-bs-path",
                "/tmp/stage2/biot_savart_opt.json",
            ]
        )

        with patch.object(wrapper, "probe_stage2_seed_bootability", return_value={}) as probe:
            wrapper.build_probe_status(
                args,
                stage2_bs_path=Path("/tmp/stage2/biot_savart_opt.json"),
                stage2_results={
                    "PLASMA_SURF_FILENAME": "demo.nc",
                    "FINITE_CURRENT_MODE": "wataru_proxy_field",
                    "PROXY_PLASMA_CURRENT_A": 9000.0,
                },
                stage="probe",
            )

        self.assertAlmostEqual(
            probe.call_args.kwargs["boozer_I"],
            4.0e-7 * 3.141592653589793 * 9000.0,
        )

    def test_build_probe_status_single_surface_rejects_conflicting_requested_mode(self):
        wrapper = load_wrapper_module()

        args = wrapper.parse_args(
            [
                "--plasma-surf-filename",
                "demo.nc",
                "--stage2-bs-path",
                "/tmp/stage2/biot_savart_opt.json",
                "--plasma-current-A",
                "9000.0",
            ]
        )
        args.finite_current_mode = "boozer_surrogate"

        with self.assertRaisesRegex(ValueError, "Single-surface mode is locked to"):
            wrapper.build_probe_status(
                args,
                stage2_bs_path=Path("/tmp/stage2/biot_savart_opt.json"),
                stage2_results={
                    "PLASMA_SURF_FILENAME": "demo.nc",
                    "FINITE_CURRENT_MODE": "boozer_surrogate",
                    "PROXY_PLASMA_CURRENT_A": 0.0,
                },
                stage="probe",
            )

    def test_build_recovery_command_forwards_stage2_seed_surface_path(self):
        wrapper = load_wrapper_module()

        args = wrapper.parse_args(
            [
                "--plasma-surf-filename",
                "demo.nc",
                "--stage2-bs-path",
                "/tmp/stage2/biot_savart_opt.json",
                "--stage2-seed-surf-path",
                "seed/surf_opt_boozer_surface.json",
                "--single-stage-banana-current-mode",
                "independent",
            ]
        )

        command = wrapper.build_recovery_command(
            args,
            stage2_bs_path=Path("/tmp/stage2/biot_savart_opt.json"),
            recovery_output_root=Path("/tmp/recovery"),
        )

        self.assertEqual(
            command[command.index("--stage2-seed-surf-path") + 1],
            str(Path("seed/surf_opt_boozer_surface.json").resolve()),
        )
        self.assertEqual(
            command[command.index("--single-stage-banana-current-mode") + 1],
            "independent",
        )

    def test_recovery_only_updates_recovery_results_with_handoff_metadata(self):
        wrapper = load_wrapper_module()
        handoff = load_handoff_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            stage2_bs_path, stage2_results_path = self._stage2_seed_paths(root)
            output_root = root / "outputs"
            recovery_case_dir = output_root / "recovery" / "mpol=8-ntor=6-test"

            def fake_recovery_run(command, *, output_root, timeout_seconds):
                self.assertEqual(Path(output_root), output_root)
                recovery_case_dir.mkdir(parents=True, exist_ok=True)
                _write_json(
                    recovery_case_dir / "results.json",
                    {
                        "PLASMA_SURF_FILENAME": "demo.nc",
                        "init_only": False,
                        "iterations": 7,
                    },
                )
                (recovery_case_dir / "biot_savart_opt.json").write_text(
                    "{}",
                    encoding="utf-8",
                )
                return (
                    "final",
                    recovery_case_dir / "results.json",
                    json.loads(
                        (recovery_case_dir / "results.json").read_text(
                            encoding="utf-8"
                        )
                    ),
                )

            initial_probe = _bootability_status(
                handoff,
                stage=handoff.BOOTABILITY_STAGE_PROBE,
                reason=handoff.BOOTABILITY_REASON_SELF_INTERSECTION,
                bootable=False,
                iota_feasible=False,
                solved_iota=0.0003,
                self_intersecting=True,
            )
            recovered_probe = _bootability_status(
                handoff,
                stage=handoff.BOOTABILITY_STAGE_RECOVERY,
                reason=handoff.BOOTABILITY_REASON_OK,
                bootable=True,
                iota_feasible=True,
                solved_iota=0.2004,
                self_intersecting=False,
            )

            with patch.object(
                wrapper,
                "build_probe_status",
                side_effect=[initial_probe, recovered_probe],
            ), patch.object(
                wrapper,
                "run_single_stage_command_with_salvage",
                side_effect=fake_recovery_run,
            ):
                result = wrapper.main(
                    [
                        "--recovery-only",
                        "--plasma-surf-filename",
                        "demo.nc",
                        "--stage2-bs-path",
                        str(stage2_bs_path),
                        "--output-root",
                        str(output_root),
                    ]
                )

            self.assertEqual(result, 0)
            recovered_results = json.loads(
                (recovery_case_dir / "results.json").read_text(encoding="utf-8")
            )
            expected_stage2_bs_path = str(stage2_bs_path.resolve())
            expected_stage2_results_path = str(stage2_results_path.resolve())
            self.assertTrue(recovered_results["BOOZER_BOOTABLE"])
            self.assertTrue(recovered_results["IOTA_FEASIBLE"])
            self.assertTrue(recovered_results["RECOVERY_ATTEMPTED"])
            self.assertTrue(recovered_results["RECOVERY_SUCCEEDED"])
            self.assertEqual(recovered_results["RECOVERY_ITERS"], 7)
            self.assertEqual(
                recovered_results["RECOVERY_TERMINATION_REASON"],
                "bootable",
            )
            self.assertEqual(
                recovered_results["STAGE2_BS_PATH"],
                expected_stage2_bs_path,
            )
            self.assertEqual(
                recovered_results["STAGE2_RESULTS_PATH"],
                expected_stage2_results_path,
            )
            self.assertEqual(
                recovered_results["UNIFIED_SEED_SOURCE"],
                wrapper.SEED_SOURCE_RECOVERED_STAGE2_DONOR,
            )

    def test_full_mode_augments_final_results_with_recovered_handoff_metadata(self):
        wrapper = load_wrapper_module()
        handoff = load_handoff_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            stage2_bs_path, stage2_results_path = self._stage2_seed_paths(root)
            output_root = root / "outputs"
            recovery_case_dir = output_root / "recovery" / "mpol=8-ntor=6-test"
            full_case_dir = output_root / "full" / "target" / "mpol=8-ntor=6-test"

            def fake_recovery_run(command, *, output_root, timeout_seconds):
                recovery_case_dir.mkdir(parents=True, exist_ok=True)
                _write_json(
                    recovery_case_dir / "results.json",
                    {
                        "PLASMA_SURF_FILENAME": "demo.nc",
                        "init_only": False,
                        "iterations": 11,
                    },
                )
                (recovery_case_dir / "biot_savart_opt.json").write_text(
                    "{}",
                    encoding="utf-8",
                )
                return (
                    "final",
                    recovery_case_dir / "results.json",
                    json.loads(
                        (recovery_case_dir / "results.json").read_text(
                            encoding="utf-8"
                        )
                    ),
                )

            def fake_full_run(
                args,
                *,
                stage2_bs_path,
                full_output_root,
                warm_start_surface_stem=None,
            ):
                full_case_dir.mkdir(parents=True, exist_ok=True)
                self.assertEqual(
                    warm_start_surface_stem.resolve(),
                    (recovery_case_dir / "surf_opt").resolve(),
                )
                _write_json(
                    full_case_dir / "results.json",
                    {
                        "PLASMA_SURF_FILENAME": "demo.nc",
                        "OPTIMIZER_SUCCESS": True,
                    },
                )
                return {
                    "status": "completed",
                    "command": ["python", "single_stage_banana_example.py"],
                    "output_root": str(full_output_root),
                    "results_path": str(full_case_dir / "results.json"),
                    "result_source": "final",
                    "results": json.loads(
                        (full_case_dir / "results.json").read_text(
                            encoding="utf-8"
                        )
                    ),
                }

            initial_probe = _bootability_status(
                handoff,
                stage=handoff.BOOTABILITY_STAGE_PROBE,
                reason=handoff.BOOTABILITY_REASON_SELF_INTERSECTION,
                bootable=False,
                iota_feasible=False,
                solved_iota=0.0002,
                self_intersecting=True,
            )
            recovered_probe = _bootability_status(
                handoff,
                stage=handoff.BOOTABILITY_STAGE_RECOVERY,
                reason=handoff.BOOTABILITY_REASON_OK,
                bootable=True,
                iota_feasible=True,
                solved_iota=0.1999,
                self_intersecting=False,
            )

            with patch.object(
                wrapper,
                "build_probe_status",
                side_effect=[initial_probe, recovered_probe],
            ), patch.object(
                wrapper,
                "run_single_stage_command_with_salvage",
                side_effect=fake_recovery_run,
            ), patch.object(
                wrapper,
                "run_full_single_stage",
                side_effect=fake_full_run,
            ):
                result = wrapper.main(
                    [
                        "--plasma-surf-filename",
                        "demo.nc",
                        "--stage2-bs-path",
                        str(stage2_bs_path),
                        "--output-root",
                        str(output_root),
                    ]
                )

            self.assertEqual(result, 0)
            final_results = json.loads(
                (full_case_dir / "results.json").read_text(encoding="utf-8")
            )
            expected_stage2_bs_path = str(stage2_bs_path.resolve())
            expected_stage2_results_path = str(stage2_results_path.resolve())
            self.assertTrue(final_results["BOOZER_BOOTABLE"])
            self.assertTrue(final_results["IOTA_FEASIBLE"])
            self.assertTrue(final_results["RECOVERY_ATTEMPTED"])
            self.assertTrue(final_results["RECOVERY_SUCCEEDED"])
            self.assertEqual(final_results["RECOVERY_ITERS"], 11)
            self.assertEqual(
                final_results["RECOVERY_TERMINATION_REASON"],
                "bootable",
            )
            self.assertEqual(
                final_results["STAGE2_BS_PATH"],
                expected_stage2_bs_path,
            )
            self.assertEqual(
                final_results["STAGE2_RESULTS_PATH"],
                expected_stage2_results_path,
            )
            self.assertEqual(
                final_results["UNIFIED_SEED_SOURCE"],
                wrapper.SEED_SOURCE_RECOVERED_STAGE2_DONOR,
            )

    def test_recovery_only_conflict_with_skip_recovery_is_rejected(self):
        wrapper = load_wrapper_module()

        with self.assertRaisesRegex(
            ValueError,
            "--recovery-only cannot be combined with --skip-recovery",
        ):
            wrapper.main(
                [
                    "--recovery-only",
                    "--skip-recovery",
                    "--plasma-surf-filename",
                    "demo.nc",
                    "--stage2-bs-path",
                    "/tmp/stage2/biot_savart_opt.json",
                ]
            )

    def test_run_recovery_stage_probes_recovered_bs_with_original_stage2_metadata(self):
        """Guard against a regression where the recovery probe was fed the recovery
        single-stage results.json (which uses the STAGE2_* prefix convention and omits
        TF_CURRENT_A / NUM_TF_COILS / FINITE_CURRENT_MODE) instead of the original
        Stage 2 artifact metadata. That regression silently returned
        BOOTABILITY_REASON_MISSING_ARTIFACT_METADATA even on successful recoveries.
        """
        wrapper = load_wrapper_module()
        handoff = load_handoff_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            stage2_bs_path, stage2_results_path = self._stage2_seed_paths(root)
            recovery_output_root = root / "recovery"
            recovery_case_dir = recovery_output_root / "mpol=8-ntor=6-test"

            original_stage2_results = {
                "PLASMA_SURF_FILENAME": "demo.nc",
                "TF_CURRENT_A": 8.0e4,
                "NUM_TF_COILS": 20,
                "MAJOR_RADIUS": 0.976,
                "TOROIDAL_FLUX": 0.24,
                "banana_surf_radius": 0.21,
                "FINITE_CURRENT_MODE": "boozer_surrogate",
                "CURVATURE_THRESHOLD": 100.0,
            }

            def fake_recovery_run(command, *, output_root, timeout_seconds):
                recovery_case_dir.mkdir(parents=True, exist_ok=True)
                recovery_single_stage_results = {
                    "PLASMA_SURF_FILENAME": "demo.nc",
                    "init_only": False,
                    "iterations": 7,
                    # Single-stage schema: uses STAGE2_* prefix, does not surface
                    # TF_CURRENT_A / NUM_TF_COILS / FINITE_CURRENT_MODE directly.
                    "STAGE2_TF_CURRENT_A": 8.0e4,
                    "STAGE2_FINITE_CURRENT_MODE": "boozer_surrogate",
                    "MAJOR_RADIUS": 0.976,
                    "TOROIDAL_FLUX": 0.24,
                    "banana_surf_radius": 0.21,
                }
                _write_json(
                    recovery_case_dir / "results.json",
                    recovery_single_stage_results,
                )
                (recovery_case_dir / "biot_savart_opt.json").write_text(
                    "{}",
                    encoding="utf-8",
                )
                return (
                    "final",
                    recovery_case_dir / "results.json",
                    recovery_single_stage_results,
                )

            captured_probe_calls: list[dict[str, object]] = []

            def fake_build_probe_status(
                args,
                *,
                stage2_bs_path,
                stage2_results,
                stage,
                warm_start_boozer_surface_path=None,
            ):
                captured_probe_calls.append(
                    {
                        "stage2_bs_path": stage2_bs_path,
                        "stage2_results": stage2_results,
                        "stage": stage,
                        "warm_start_boozer_surface_path": warm_start_boozer_surface_path,
                    }
                )
                return _bootability_status(
                    handoff,
                    stage=stage,
                    reason=handoff.BOOTABILITY_REASON_OK,
                    bootable=True,
                    iota_feasible=True,
                    solved_iota=0.2004,
                    self_intersecting=False,
                )

            args = wrapper.parse_args(
                [
                    "--recovery-only",
                    "--plasma-surf-filename",
                    "demo.nc",
                    "--stage2-bs-path",
                    str(stage2_bs_path),
                    "--output-root",
                    str(root / "outputs"),
                ]
            )

            with patch.object(
                wrapper,
                "build_probe_status",
                side_effect=fake_build_probe_status,
            ), patch.object(
                wrapper,
                "run_single_stage_command_with_salvage",
                side_effect=fake_recovery_run,
            ):
                wrapper.run_recovery_stage(
                    args,
                    original_stage2_bs_path=stage2_bs_path,
                    original_stage2_results_path=stage2_results_path,
                    original_stage2_results=original_stage2_results,
                    recovery_output_root=recovery_output_root,
                )

            self.assertEqual(len(captured_probe_calls), 1)
            probe_call = captured_probe_calls[0]
            self.assertEqual(probe_call["stage"], handoff.BOOTABILITY_STAGE_RECOVERY)
            # The recovered coils live at the recovery output, not the original seed.
            self.assertEqual(
                probe_call["stage2_bs_path"],
                recovery_case_dir / "biot_savart_opt.json",
            )
            self.assertEqual(
                probe_call["warm_start_boozer_surface_path"],
                recovery_case_dir / "surf_opt_boozer_surface.json",
            )
            # But the probe must receive the *original* Stage 2 metadata so that
            # TF_CURRENT_A / NUM_TF_COILS / FINITE_CURRENT_MODE / banana_surf_radius
            # can be validated. The recovery single-stage results.json does not
            # surface these keys directly, so passing it would silently fail.
            self.assertIs(probe_call["stage2_results"], original_stage2_results)
            self.assertEqual(probe_call["stage2_results"]["TF_CURRENT_A"], 8.0e4)
            self.assertEqual(probe_call["stage2_results"]["NUM_TF_COILS"], 20)
            self.assertEqual(
                probe_call["stage2_results"]["FINITE_CURRENT_MODE"],
                "boozer_surrogate",
            )

    def test_run_recovery_stage_uses_preserved_artifact_bundle_for_salvaged_results(self):
        wrapper = load_wrapper_module()
        handoff = load_handoff_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            stage2_bs_path, stage2_results_path = self._stage2_seed_paths(root)
            recovery_output_root = root / "recovery"
            recovery_case_dir = recovery_output_root / "mpol=8-ntor=6-test"

            original_stage2_results = {
                "PLASMA_SURF_FILENAME": "demo.nc",
                "TF_CURRENT_A": 8.0e4,
                "NUM_TF_COILS": 20,
                "MAJOR_RADIUS": 0.976,
                "TOROIDAL_FLUX": 0.24,
                "banana_surf_radius": 0.21,
                "FINITE_CURRENT_MODE": "boozer_surrogate",
                "CURVATURE_THRESHOLD": 100.0,
            }

            def fake_recovery_run(command, *, output_root, timeout_seconds):
                recovery_case_dir.mkdir(parents=True, exist_ok=True)
                partial_results = {
                    "PLASMA_SURF_FILENAME": "demo.nc",
                    "init_only": False,
                    "iterations": 9,
                }
                partial_results_path = (
                    recovery_case_dir / "results_best_feasible.partial.json"
                )
                _write_json(partial_results_path, partial_results)
                (recovery_case_dir / "biot_savart_best_feasible.json").write_text(
                    "{}",
                    encoding="utf-8",
                )
                (recovery_case_dir / "surf_best_feasible_outer_boozer_surface.json").write_text(
                    "{}",
                    encoding="utf-8",
                )
                return (
                    "best_feasible_partial",
                    partial_results_path,
                    partial_results,
                )

            captured_probe_calls: list[dict[str, object]] = []

            def fake_build_probe_status(
                args,
                *,
                stage2_bs_path,
                stage2_results,
                stage,
                warm_start_boozer_surface_path=None,
            ):
                captured_probe_calls.append(
                    {
                        "stage2_bs_path": stage2_bs_path,
                        "stage2_results": stage2_results,
                        "stage": stage,
                        "warm_start_boozer_surface_path": warm_start_boozer_surface_path,
                    }
                )
                return _bootability_status(
                    handoff,
                    stage=stage,
                    reason=handoff.BOOTABILITY_REASON_OK,
                    bootable=True,
                    iota_feasible=True,
                    solved_iota=0.2002,
                    self_intersecting=False,
                )

            args = wrapper.parse_args(
                [
                    "--recovery-only",
                    "--plasma-surf-filename",
                    "demo.nc",
                    "--stage2-bs-path",
                    str(stage2_bs_path),
                    "--output-root",
                    str(root / "outputs"),
                ]
            )

            with patch.object(
                wrapper,
                "build_probe_status",
                side_effect=fake_build_probe_status,
            ), patch.object(
                wrapper,
                "run_single_stage_command_with_salvage",
                side_effect=fake_recovery_run,
            ):
                payload = wrapper.run_recovery_stage(
                    args,
                    original_stage2_bs_path=stage2_bs_path,
                    original_stage2_results_path=stage2_results_path,
                    original_stage2_results=original_stage2_results,
                    recovery_output_root=recovery_output_root,
                )

            self.assertEqual(payload["status"], "completed")
            self.assertEqual(
                payload["recovered_bs_path"],
                str(recovery_case_dir / "biot_savart_best_feasible.json"),
            )
            self.assertEqual(
                payload["warm_start_surface_stem"],
                str(recovery_case_dir / "surf_best_feasible"),
            )
            self.assertEqual(len(captured_probe_calls), 1)
            probe_call = captured_probe_calls[0]
            self.assertEqual(
                probe_call["stage2_bs_path"],
                recovery_case_dir / "biot_savart_best_feasible.json",
            )
            self.assertEqual(
                probe_call["warm_start_boozer_surface_path"],
                recovery_case_dir / "surf_best_feasible_outer_boozer_surface.json",
            )
            self.assertIs(probe_call["stage2_results"], original_stage2_results)
