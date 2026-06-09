import importlib
import importlib.util
import ast
import hashlib
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
from examples.single_stage_optimization.banana_opt.finite_current_profiles import (
    get_finite_current_profile,
)
from simsopt.field import BiotSavart, Coil, Current
from simsopt.geo import CurveXYZFourier


EXAMPLE_ROOT = (
    Path(__file__).resolve().parents[2] / "examples" / "single_stage_optimization"
)
WRAPPER_PATH = EXAMPLE_ROOT / "run_stage2_to_single_stage.py"
BANANA_SCAN_PATH = EXAMPLE_ROOT / "run_banana_current_scan.py"
STAGE2_SOLVER_PATH = EXAMPLE_ROOT / "STAGE_2" / "banana_coil_solver.py"
STAGE2_ALM_PATH = EXAMPLE_ROOT / "run_stage2_alm.py"
SIGNED_CW_WOUT_PATH = (
    Path(__file__).resolve().parents[1] / "test_files" / "wout_10x10.nc"
)
POSITIVE_CCW_WOUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "test_files"
    / "wout_LandremanPaul2021_QA_lowres.nc"
)
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


def load_banana_scan_module():
    return load_module(BANANA_SCAN_PATH, "run_banana_current_scan")


def load_stage2_solver_module():
    return load_module(STAGE2_SOLVER_PATH, "banana_coil_solver")


def load_stage2_alm_module():
    return load_module(STAGE2_ALM_PATH, "run_stage2_alm")


def load_hardware_schema_module():
    return importlib.import_module("banana_opt.hardware_constraint_schema")


def load_artifact_contracts_module():
    return importlib.import_module("banana_opt.artifact_contracts")


def load_handoff_module():
    return importlib.import_module("banana_opt.stage2_single_stage_handoff")


def load_current_mode_module():
    return importlib.import_module("banana_opt.single_stage_banana_current_mode")


def load_workflow_runner_common_module():
    return importlib.import_module("workflow_runner_common")


def _banana_source_paths():
    for relative_root in ("banana_opt", "SINGLE_STAGE", "STAGE_2"):
        yield from (EXAMPLE_ROOT / relative_root).rglob("*.py")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    stage2_bs_path = path.with_name("biot_savart_opt.json")
    if path.name == "results.json" and stage2_bs_path.is_file():
        payload.setdefault(
            "STAGE2_BS_SHA256",
            hashlib.sha256(stage2_bs_path.read_bytes()).hexdigest(),
        )
    path.write_text(json.dumps(payload), encoding="utf-8")


def _valid_stage2_contract_fields() -> dict[str, object]:
    hardware_contracts = importlib.import_module("banana_opt.hardware_contracts")
    return {
        "MAJOR_RADIUS": 0.976,
        "TOROIDAL_FLUX": 0.24,
        "banana_surf_radius": hardware_contracts.BANANA_WINDING_MINOR_RADIUS_M,
        "COIL_LENGTH": 2.0,
        "CURVE_CURVE_MIN_DIST": 0.05,
        "CURVE_SURFACE_MIN_DIST": 0.015,
        "SURFACE_VESSEL_MIN_DIST": 0.04,
        "MAX_CURVATURE": 100.0,
        "CURVATURE_THRESHOLD": 100.0,
        "POLOIDAL_EXTENT_RAD": 45.0 * np.pi / 180.0,
        "POLOIDAL_EXTENT_THRESHOLD_RAD": 45.0 * np.pi / 180.0,
        "COIL_WIDTH": 0.10,
        "WIDTH_MIN_THRESHOLD": 0.1,
        "WIDTH_MAX_THRESHOLD": hardware_contracts.BANANA_WIDTH_MAX_M,
        "SELF_INTERSECT_PENALTY": 0.0,
        "SELF_INTERSECT_THRESHOLD": 0.0,
        "SHORTEST_SELF_DISTANCE": 0.01,
        "SELF_INTERSECT_MIN_DISTANCE": 0.01,
        "BANANA_CURRENT_A": 1.1e4,
        "BANANA_INIT_CURRENT_A": -1.0e4,
        "BANANA_CURRENT_MAX_A": 1.6e4,
        "TF_CURRENT_A": -8.0e4,
        "FINAL_LCFS_MAJOR_RADIUS_M": hardware_contracts.TARGET_LCFS_MAX_MAJOR_RADIUS_M,
        "FINAL_LCFS_MINOR_RADIUS_M": hardware_contracts.TARGET_LCFS_MAX_MINOR_RADIUS_M,
        # LCFS edge-envelope metrics (Type KK contract): outboard = major + minor,
        # inboard = major - minor, evaluated at the max envelope so they sit exactly
        # on LCFS_{OUTBOARD,INBOARD}_RADIUS_{MAX,MIN}_M (valid by construction).
        "FINAL_LCFS_OUTBOARD_EDGE_M": (
            hardware_contracts.TARGET_LCFS_MAX_MAJOR_RADIUS_M
            + hardware_contracts.TARGET_LCFS_MAX_MINOR_RADIUS_M
        ),
        "FINAL_LCFS_INBOARD_EDGE_M": (
            hardware_contracts.TARGET_LCFS_MAX_MAJOR_RADIUS_M
            - hardware_contracts.TARGET_LCFS_MAX_MINOR_RADIUS_M
        ),
        "CC_THRESHOLD": 0.05,
        "CC_WEIGHT": 100.0,
        "CURVATURE_WEIGHT": 1.0e-4,
        "LENGTH_TARGET": 1.9,
        "LENGTH_MIN_TARGET": 0.95,
        "LENGTH_WEIGHT": 5.0e-4,
        "order": 2,
        "PROXY_PLASMA_CURRENT_A": 0.0,
        "VF_CURRENT_A": 0.0,
        "VF_TEMPLATE_PATH": None,
        "BOOZER_BOOTABLE": True,
        "IOTA_NEAR_TARGET": True,
        "IOTA_FEASIBLE": True,
        "BOOTABILITY_REASON": "ok",
        "BOOTABILITY_STAGE": "probe",
        "BOOTABILITY_TARGET_IOTA": 0.2,
        "BOOTABILITY_SOLVED_IOTA": 0.2,
        "BOOTABILITY_SELF_INTERSECTING": False,
        "BOOTABILITY_SOLVE_SUCCESS": True,
        "BOOTABILITY_ABS_IOTA_ERROR": 0.0,
        "BOOTABILITY_ERROR_TYPE": None,
        "BOOTABILITY_ERROR_MESSAGE": None,
        "BOOZER_SOLVE_SUCCESS": True,
        "BOOZER_SELF_INTERSECTING": False,
        "BOOZER_CONSTRAINED_RESIDUAL_NORM": 1.0e-14,
        "BOOZER_TRUSTED": True,
        "IOTA_OBJECTIVE_ACTIVE": True,
        "BOOZER_TRUST_REASON": "trusted",
        "BOOZER_TRUST_TOL": 1.0e-11,
        "PLASMA_SURF_PATH": str(SIGNED_CW_WOUT_PATH),
        "WOUT_CONVENTION": "signed_cw",
        "WOUT_OFF_SPEC": False,
        "SEED_ROLE": "coil_seed_handoff",
        "DIAGNOSTIC_ONLY": False,
        "PRODUCTION_HANDOFF_READY": True,
        "HANDOFF_BLOCKING_GATE": None,
        "PROMOTION_READY": True,
    }


def _valid_jhalpern_stage2_contract_fields() -> dict[str, object]:
    profile = get_finite_current_profile("jhalpern30_proxy_field")
    return {
        **_valid_stage2_contract_fields(),
        "FINITE_CURRENT_MODE": "jhalpern30_proxy_field",
        "BOOZER_CURRENT_CONVENTION": "mu0",
        "BOOZER_I": -0.008168140899333462,
        "G0_POLICY": "signed_explicit_tf_current",
        "PROXY_PLACEMENT_MODE": "surface_major_radius_z0",
        "PROXY_VF_CURRENT_SCALAR_POLICY": "signed_physical_scalar",
        "PROXY_PLASMA_CURRENT_A": -6.5e3,
        "VF_CURRENT_A": -1.0e3,
        "VF_TEMPLATE_PATH": str(profile.default_vf_template_path),
        "VF_TEMPLATE_SHA256": profile.vf_template_sha256,
        "VF_CURRENT_SIGN_POLICY": "template_sign_abs_proxy_current",
        "VF_CURRENT_MUTABILITY": "shared_unfixed_scaled_current",
        "NUM_TF_COILS": 20,
        "NUM_BANANA_COILS": 10,
        "NUM_PROXY_COILS": 1,
        "NUM_VF_COILS": 20,
        "TOTAL_COILS": 51,
        "COIL_GROUPS": profile.build_default_coil_groups_manifest().to_json_payload(),
        "FLIP_BANANA": True,
        "BANANA_CURRENT_SIGN": -1,
        "BANANA_CURRENT_PINNED": False,
        "BANANA_I_FIXED_S2_KA": None,
        "IOTA_TARGET_SIGN": -1,
        "JHALPERN30_STAGE_NAME": "stage00",
        "JHALPERN30_STAGE_STATE": {
            "iota": 0.29789,
            "G": -2.0106,
            "volume": 0.03993,
        },
    }


def _stage2_coil_partitions(
    module,
    *,
    finite_current_mode: str,
    num_tf_coils: int = 20,
    num_banana_coils: int = 10,
    num_proxy_coils: int = 0,
    num_vf_coils: int = 0,
):
    return module.Stage2CoilPartitions(
        tf_coils=tuple(object() for _ in range(num_tf_coils)),
        banana_coils=tuple(object() for _ in range(num_banana_coils)),
        proxy_coils=tuple(object() for _ in range(num_proxy_coils)),
        vf_coils=tuple(object() for _ in range(num_vf_coils)),
        num_tf_coils=num_tf_coils,
        num_banana_coils=num_banana_coils,
        num_proxy_coils=num_proxy_coils,
        num_vf_coils=num_vf_coils,
        finite_current_mode=finite_current_mode,
    )


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
            Current(-8.0e4),
        )
        for index in range(20)
    ]
    banana_coils = [
        Coil(
            _make_circle_curve(center=(1.02, 0.0, -0.08), radius=0.07, normal="z"),
            Current(1.1e4),
        ),
        Coil(
            _make_circle_curve(center=(1.02, 0.0, 0.08), radius=0.07, normal="z"),
            Current(-1.1e4),
        ),
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
        **_valid_stage2_contract_fields(),
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
        "IOTA_NEAR_TARGET": iota_feasible,
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
        "BOOZER_SOLVE_SUCCESS": bootable,
        "BOOZER_SELF_INTERSECTING": self_intersecting,
        "BOOZER_CONSTRAINED_RESIDUAL_NORM": 1.0e-14 if bootable else None,
        "BOOZER_TRUSTED": bootable and self_intersecting is not True,
        "IOTA_OBJECTIVE_ACTIVE": bootable,
        "BOOZER_TRUST_REASON": (
            handoff_module.BOOZER_TRUST_REASON_OK
            if bootable and self_intersecting is not True
            else handoff_module.BOOZER_TRUST_REASON_SOLVE_FAILED
        ),
        "BOOZER_TRUST_TOL": 1.0e-11 if bootable else None,
    }


class HandoffSchemaTests(unittest.TestCase):
    def test_hbt_proxy_vf_current_convention_requires_nonnegative_ratio(self):
        current_contracts = importlib.import_module("banana_opt.current_contracts")

        self.assertEqual(
            current_contracts.validate_hbt_proxy_vf_current_convention(
                proxy_plasma_current_A=9.0e3,
                vf_current_A=9.0e3 / 6.5,
            ),
            (9.0e3, 9.0e3 / 6.5),
        )
        with self.assertRaisesRegex(ValueError, "proxy plasma current"):
            current_contracts.validate_hbt_proxy_vf_current_convention(
                proxy_plasma_current_A=-1.0,
                vf_current_A=0.0,
            )
        with self.assertRaisesRegex(ValueError, "proxy/VF convention"):
            current_contracts.validate_hbt_proxy_vf_current_convention(
                proxy_plasma_current_A=9.0e3,
                vf_current_A=5.0e2,
            )

    def test_validate_stage2_seed_bootability_contract_accepts_fact_based_handoff(self):
        module = load_handoff_module()
        payload = {
            **_valid_stage2_contract_fields(),
            "BOOZER_TRUSTED": True,
        }

        module.validate_stage2_seed_bootability_contract(payload)

        off_target_payload = dict(payload)
        off_target_payload["IOTA_NEAR_TARGET"] = False
        off_target_payload["IOTA_FEASIBLE"] = False
        with self.assertRaisesRegex(ValueError, "IOTA_NEAR_TARGET=False"):
            module.validate_stage2_seed_bootability_contract(off_target_payload)

        offspec_wout_payload = dict(payload)
        offspec_wout_payload["WOUT_OFF_SPEC"] = True
        with self.assertRaisesRegex(ValueError, "WOUT_OFF_SPEC=True"):
            module.validate_stage2_seed_bootability_contract(offspec_wout_payload)

        repair_payload = dict(payload)
        repair_payload.update(
            {
                "SEED_ROLE": "hardware_repair",
                "DIAGNOSTIC_ONLY": True,
                "PRODUCTION_HANDOFF_READY": False,
                "HANDOFF_BLOCKING_GATE": "ALM_CONVERGED",
                "PROMOTION_READY": False,
            }
        )
        with self.assertRaisesRegex(ValueError, "SEED_ROLE='hardware_repair'"):
            module.validate_stage2_seed_bootability_contract(repair_payload)

        missing_blocking_gate_payload = dict(payload)
        missing_blocking_gate_payload.pop("HANDOFF_BLOCKING_GATE")
        with self.assertRaisesRegex(ValueError, "HANDOFF_BLOCKING_GATE=None"):
            module.validate_stage2_seed_bootability_contract(
                missing_blocking_gate_payload
            )

    def test_build_bootability_recovery_payload_fields_shapes_all_expected_keys(self):
        module = load_hardware_schema_module()

        payload = module.build_bootability_recovery_payload_fields(
            {
                "BOOZER_BOOTABLE": True,
                "IOTA_NEAR_TARGET": False,
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
                "BOOZER_SOLVE_SUCCESS": True,
                "BOOZER_SELF_INTERSECTING": False,
                "BOOZER_CONSTRAINED_RESIDUAL_NORM": 1.0e-14,
                "BOOZER_TRUSTED": True,
                "IOTA_OBJECTIVE_ACTIVE": True,
                "BOOZER_TRUST_REASON": "trusted",
                "BOOZER_TRUST_TOL": 1.0e-11,
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
                "IOTA_NEAR_TARGET",
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
                "BOOZER_SOLVE_SUCCESS",
                "BOOZER_SELF_INTERSECTING",
                "BOOZER_CONSTRAINED_RESIDUAL_NORM",
                "BOOZER_TRUSTED",
                "IOTA_OBJECTIVE_ACTIVE",
                "BOOZER_TRUST_REASON",
                "BOOZER_TRUST_TOL",
                "STAGE2_BS_PATH",
                "STAGE2_RESULTS_PATH",
                "RECOVERY_ATTEMPTED",
                "RECOVERY_SUCCEEDED",
                "RECOVERY_ITERS",
                "RECOVERY_TERMINATION_REASON",
            ),
        )
        self.assertTrue(payload["BOOZER_BOOTABLE"])
        self.assertFalse(payload["IOTA_NEAR_TARGET"])
        self.assertFalse(payload["IOTA_FEASIBLE"])
        self.assertEqual(payload["BOOTABILITY_REASON"], "iota_mismatch")
        self.assertAlmostEqual(payload["BOOTABILITY_ABS_IOTA_ERROR"], 0.02)
        self.assertTrue(payload["BOOZER_TRUSTED"])
        self.assertEqual(payload["BOOZER_TRUST_REASON"], "trusted")
        self.assertEqual(payload["RECOVERY_ITERS"], 7)
        self.assertEqual(
            payload["RECOVERY_TERMINATION_REASON"],
            "not_bootable_after_budget",
        )

    def test_upgrade_legacy_stage2_artifact_results_backfills_handoff_defaults(self):
        module = load_artifact_contracts_module()

        upgraded = module.upgrade_legacy_stage2_artifact_results({})

        self.assertIsNone(upgraded["BOOZER_BOOTABLE"])
        self.assertIsNone(upgraded["IOTA_NEAR_TARGET"])
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
        self.assertEqual(
            upgraded["FINITE_CURRENT_MODE_SOURCE"], "legacy_assumed_default"
        )
        self.assertEqual(upgraded["BOOZER_CURRENT_CONVENTION"], "mu0")
        self.assertEqual(upgraded["NUM_PROXY_COILS"], 0)
        self.assertEqual(upgraded["NUM_VF_COILS"], 0)
        self.assertEqual(upgraded["PROXY_PLASMA_CURRENT_A"], 0.0)
        self.assertEqual(upgraded["VF_CURRENT_A"], 0.0)
        self.assertIsNone(upgraded["VF_TEMPLATE_PATH"])

    def test_upgrade_legacy_stage2_artifact_results_uses_stage2_prefixed_coil_counts(
        self,
    ):
        artifact_contracts = load_artifact_contracts_module()
        coil_groups = importlib.import_module("banana_opt.coil_groups")

        upgraded = artifact_contracts.upgrade_legacy_stage2_artifact_results(
            {
                "NFP": 5,
                "STAGE2_NUM_TF_COILS": 20,
                "STAGE2_NUM_BANANA_COILS": 10,
                "STAGE2_NUM_PROXY_COILS": 1,
                "STAGE2_NUM_VF_COILS": 2,
            }
        )

        self.assertEqual(upgraded["NUM_TF_COILS"], 20)
        self.assertEqual(upgraded["NUM_BANANA_COILS"], 10)
        self.assertEqual(upgraded["NUM_PROXY_COILS"], 1)
        self.assertEqual(upgraded["NUM_VF_COILS"], 2)

        manifest = coil_groups.infer_manifest_from_legacy_counts(
            upgraded,
            total_loaded_coils=33,
        )
        self.assertEqual(manifest.total(), 33)
        self.assertEqual(manifest.count_for_role("tf"), 20)
        self.assertEqual(manifest.count_for_role("banana"), 10)
        self.assertEqual(manifest.count_for_role("proxy"), 1)
        self.assertEqual(manifest.count_for_role("vf"), 2)

    def test_upgrade_legacy_stage2_artifact_results_backfills_wout_convention_for_legacy_single_stage_sidecar(
        self,
    ):
        """Salvaged single-stage sidecars omit WOUT_CONVENTION/WOUT_OFF_SPEC.

        The Stage 2 seed-contract validator requires both keys; backfilling
        from PLASMA_SURF_PATH + TF_CURRENT_A in the upgrader keeps every
        validator caller passing without forcing every single-stage write
        site to stamp the fields independently.
        """
        artifact_contracts = load_artifact_contracts_module()
        handoff = load_handoff_module()
        legacy_sidecar_payload = {
            **_valid_stage2_contract_fields(),
            "PLASMA_SURF_PATH": str(SIGNED_CW_WOUT_PATH),
            "TF_CURRENT_A": -8.0e4,
        }
        legacy_sidecar_payload.pop("WOUT_CONVENTION")
        legacy_sidecar_payload.pop("WOUT_OFF_SPEC")

        upgraded = artifact_contracts.upgrade_legacy_stage2_artifact_results(
            legacy_sidecar_payload
        )

        self.assertEqual(upgraded["WOUT_CONVENTION"], "signed_cw")
        self.assertFalse(upgraded["WOUT_OFF_SPEC"])
        handoff.validate_stage2_seed_contract(upgraded)

    def test_upgrade_legacy_stage2_artifact_results_preserves_existing_wout_convention_stamps(
        self,
    ):
        """An already-stamped artifact must round-trip unchanged.

        The upgrader is not a producer-side validator. Overwriting a drifted
        stamp would mask the violation that ``validate_stage2_seed_contract``
        is designed to catch.
        """
        artifact_contracts = load_artifact_contracts_module()
        drifted_payload = {
            **_valid_stage2_contract_fields(),
            "TF_CURRENT_A": -8.0e4,
            "WOUT_CONVENTION": "positive_ccw",
            "WOUT_OFF_SPEC": True,
        }

        upgraded = artifact_contracts.upgrade_legacy_stage2_artifact_results(
            drifted_payload
        )

        self.assertEqual(upgraded["WOUT_CONVENTION"], "positive_ccw")
        self.assertTrue(upgraded["WOUT_OFF_SPEC"])

    def test_upgrade_legacy_stage2_artifact_results_skips_wout_backfill_without_inputs(
        self,
    ):
        """Without PLASMA_SURF_PATH or TF_CURRENT_A, no stamping happens.

        The validator surfaces the precise missing key rather than the
        upgrader fabricating a value from incomplete provenance.
        """
        artifact_contracts = load_artifact_contracts_module()

        upgraded_missing_path = (
            artifact_contracts.upgrade_legacy_stage2_artifact_results(
                {"TF_CURRENT_A": -8.0e4},
            )
        )
        self.assertNotIn("WOUT_CONVENTION", upgraded_missing_path)
        self.assertNotIn("WOUT_OFF_SPEC", upgraded_missing_path)

        upgraded_missing_tf = artifact_contracts.upgrade_legacy_stage2_artifact_results(
            {"PLASMA_SURF_PATH": str(SIGNED_CW_WOUT_PATH)},
        )
        self.assertNotIn("WOUT_CONVENTION", upgraded_missing_tf)
        self.assertNotIn("WOUT_OFF_SPEC", upgraded_missing_tf)

    def test_upgrade_legacy_stage2_artifact_results_skips_wout_backfill_when_file_absent(
        self,
    ):
        """A nonexistent PLASMA_SURF_PATH triggers no I/O and no stamping.

        Lets the validator's own ``PLASMA_SURF_PATH`` precondition surface
        the broken provenance, rather than the upgrader raising a netCDF
        read error one layer deeper than the validator.
        """
        artifact_contracts = load_artifact_contracts_module()

        upgraded = artifact_contracts.upgrade_legacy_stage2_artifact_results(
            {
                "PLASMA_SURF_PATH": "/nonexistent/wout_missing.nc",
                "TF_CURRENT_A": -8.0e4,
            }
        )

        self.assertNotIn("WOUT_CONVENTION", upgraded)
        self.assertNotIn("WOUT_OFF_SPEC", upgraded)


class HandoffModuleTests(unittest.TestCase):
    def test_warm_start_boozer_seed_default_loader_handles_legacy_finite_i(self):
        handoff = load_handoff_module()
        compat = importlib.import_module("banana_opt.json_compat")

        self.assertIs(
            handoff.load_warm_start_boozer_seed.__kwdefaults__["artifact_loader"],
            compat.load_boozer_finite_i,
        )
        self.assertIs(
            handoff.probe_stage2_seed_bootability.__kwdefaults__["bs_loader"],
            compat.load_boozer_finite_i,
        )
        self.assertIs(
            handoff.probe_stage2_seed_bootability.__kwdefaults__["warm_start_loader"],
            compat.load_boozer_finite_i,
        )

    @staticmethod
    def _fixed_current(current_A: float):
        return SimpleNamespace(get_value=lambda: float(current_A))

    def _fixed_current_coil(self, current_A: float):
        return SimpleNamespace(current=self._fixed_current(current_A))

    def _bootability_smoke_inputs(self, *, include_proxy_vf: bool):
        tf_coils = [self._fixed_current_coil(-8.0e4) for _ in range(20)]
        banana_coils = [
            self._fixed_current_coil(1.1e4),
            self._fixed_current_coil(-1.1e4),
        ]
        proxy_coils = [self._fixed_current_coil(9.0e3)] if include_proxy_vf else []
        vf_coils = (
            [self._fixed_current_coil(-(9.0e3 / 6.5))] if include_proxy_vf else []
        )
        fake_bs = SimpleNamespace(
            coils=[*tf_coils, *banana_coils, *proxy_coils, *vf_coils]
        )
        stage2_artifact_results = {
            **_valid_stage2_contract_fields(),
            "PLASMA_SURF_FILENAME": "demo.nc",
            "TF_CURRENT_A": -8.0e4,
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
                    "VF_CURRENT_A": 9.0e3 / 6.5,
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

    class _ConstantVolumeLabel:
        def __init__(self, surface):
            self.surface = surface

        def J(self):
            return 0.1

    def test_construct_boozer_surface_for_current_uses_plain_boozer_for_vacuum(self):
        module = load_handoff_module()

        class _PlainBoozerSurface:
            def __init__(
                self,
                bs,
                surf,
                vol,
                vol_target,
                constraint_weight,
                options=None,
            ):
                del bs, vol, vol_target, constraint_weight
                self.surface = surf
                self.options = options

        class _FiniteIBoozerSurface:
            def __init__(
                self,
                bs,
                surf,
                vol,
                vol_target,
                constraint_weight,
                options=None,
                I=0.0,
            ):
                del bs, vol, vol_target, constraint_weight
                self.surface = surf
                self.options = options
                self.I = I

        with (
            patch.object(module, "BoozerSurface", _PlainBoozerSurface),
            patch.object(module, "BoozerSurfaceFiniteI", _FiniteIBoozerSurface),
        ):
            vacuum_surface = module._construct_boozer_surface_for_current(
                object(),
                object(),
                object(),
                0.1,
                1.0,
                boozer_I=0.0,
            )
            finite_surface = module._construct_boozer_surface_for_current(
                object(),
                object(),
                object(),
                0.1,
                1.0,
                boozer_I=1.0e-3,
            )

        self.assertIsInstance(vacuum_surface, _PlainBoozerSurface)
        self.assertFalse(hasattr(vacuum_surface, "I"))
        self.assertIsInstance(finite_surface, _FiniteIBoozerSurface)
        self.assertAlmostEqual(finite_surface.I, 1.0e-3)

    def test_vacuum_boozer_surface_validator_rejects_finite_i_lineage(self):
        module = load_artifact_contracts_module()
        finite_i_payload = {
            "simsopt_objs": {
                "surface": {
                    "@module": "banana_opt.boozer_finite_current",
                    "@class": "BoozerSurfaceFiniteI",
                    "I": 0.0,
                }
            }
        }

        validation = module.validate_vacuum_boozer_surface_payload(finite_i_payload)

        self.assertFalse(validation["passed"])
        self.assertFalse(validation["no_finite_i_boozer_surface"])
        self.assertFalse(validation["no_i_field"])

    def test_vacuum_boozer_surface_validator_accepts_plain_upstream_lineage(self):
        module = load_artifact_contracts_module()
        plain_payload = {
            "simsopt_objs": {
                "surface": {
                    "@module": "simsopt.geo.boozersurface",
                    "@class": "BoozerSurface",
                }
            }
        }

        validation = module.validate_vacuum_boozer_surface_payload(plain_payload)

        self.assertTrue(validation["passed"])

    def test_validate_boozer_surface_json_current_lineage_round_trips_real_serialization(
        self,
    ):
        # The hand-built-dict tests above pin the validator logic; this one pins
        # the lineage markers (@class / @module / "I") to what simsopt's real
        # serializer actually emits, so a future serialization-format change that
        # silently defeats the save-time gate is caught here.
        from simsopt.geo import SurfaceXYZTensorFourier
        from simsopt.geo.boozersurface import BoozerSurface
        from simsopt.geo.surfaceobjectives import Volume

        artifact_contracts = load_artifact_contracts_module()
        BoozerSurfaceFiniteI = importlib.import_module(
            "banana_opt.boozer_finite_current"
        ).BoozerSurfaceFiniteI

        def make_boozer(cls, **extra):
            curve = CurveXYZFourier(16, 1)
            curve.set("xc(1)", 1.0)
            curve.set("zs(1)", 1.0)
            biotsavart = BiotSavart([Coil(curve, Current(1.0e5))])
            surface = SurfaceXYZTensorFourier(
                mpol=1,
                ntor=1,
                stellsym=True,
                nfp=1,
                quadpoints_phi=np.linspace(0, 1, 3, endpoint=False),
                quadpoints_theta=np.linspace(0, 1, 3, endpoint=False),
            )
            return cls(
                biotsavart,
                surface,
                Volume(surface),
                1.0,
                constraint_weight=100.0,
                options={"verbose": False},
                **extra,
            )

        def validate(boozer_surface):
            with tempfile.TemporaryDirectory() as tmpdir:
                surface_path = Path(tmpdir) / "boozer_surface.json"
                boozer_surface.save(str(surface_path))
                return artifact_contracts.validate_boozer_surface_json_current_lineage(
                    surface_path
                )

        plain = validate(make_boozer(BoozerSurface))
        vacuum_finite = validate(make_boozer(BoozerSurfaceFiniteI, I=0.0))
        nonzero_finite = validate(make_boozer(BoozerSurfaceFiniteI, I=1.0e-3))

        self.assertTrue(plain["passed"])
        self.assertTrue(plain["has_plain_boozer_surface"])
        self.assertFalse(vacuum_finite["passed"])
        self.assertFalse(vacuum_finite["no_finite_i_boozer_surface"])
        self.assertFalse(vacuum_finite["no_i_field"])
        self.assertTrue(nonzero_finite["passed"])
        self.assertGreaterEqual(nonzero_finite["finite_i_field_count"], 1)

    def test_construct_boozer_surface_for_current_rejects_finite_i_for_vacuum(self):
        module = load_handoff_module()
        BoozerSurfaceFiniteI = importlib.import_module(
            "banana_opt.boozer_finite_current"
        ).BoozerSurfaceFiniteI

        with self.assertRaises(ValueError):
            module._construct_boozer_surface_for_current(
                object(),
                object(),
                object(),
                1.0,
                1.0,
                boozer_I=0.0,
                boozer_surface_cls=BoozerSurfaceFiniteI,
            )

    def test_vacuum_boozer_surface_validator_ignores_unrelated_i_field(self):
        module = load_artifact_contracts_module()
        payload = {
            "simsopt_objs": {
                "surface": {
                    "@module": "simsopt.geo.boozersurface",
                    "@class": "BoozerSurface",
                },
                "unrelated": {"@class": "SomethingElse", "I": 3.0},
            }
        }

        validation = module.validate_vacuum_boozer_surface_payload(payload)

        self.assertTrue(validation["passed"])
        self.assertTrue(validation["no_i_field"])

    def test_attempt_initialize_vacuum_uses_plain_boozer_and_threads_signed_G(self):
        module = load_handoff_module()
        surf_prev = SimpleNamespace(
            quadpoints_theta=np.array([0.0, 0.5]),
            quadpoints_phi=np.array([0.0, 0.2]),
            gamma=lambda: np.zeros((2, 2, 3), dtype=float),
        )
        captured = {}

        class _FakeSurface:
            def __init__(self, **kwargs):
                self.quadpoints_theta = kwargs["quadpoints_theta"]
                self.quadpoints_phi = kwargs["quadpoints_phi"]
                self.dofs = np.zeros(2, dtype=float)
                self.x = np.zeros(2, dtype=float)
                self._gamma = np.zeros((2, 2, 3), dtype=float)

            def least_squares_fit(self, gamma):
                self._gamma = np.asarray(gamma, dtype=float)

            def gamma(self):
                return self._gamma.copy()

            def is_self_intersecting(self):
                return False

        class _CapturingPlainBoozerSurface:
            def __init__(
                self, bs, surf, vol, vol_target, constraint_weight, options=None
            ):
                del bs, vol, vol_target, constraint_weight, options
                self.surface = surf
                self.res = {"iota": 0.2, "G": -0.37, "success": True}
                self.need_to_run_code = True

            def run_code(self, iota, G):
                captured["iota"] = iota
                captured["G"] = G
                self.need_to_run_code = False
                return {"success": True}

        signed_G0 = -0.37
        with patch.object(module, "BoozerSurface", _CapturingPlainBoozerSurface):
            result = module.attempt_initialize_boozer_surface(
                surf_prev,
                mpol=8,
                ntor=6,
                bs=object(),
                vol_target=0.035,
                constraint_weight=1.0,
                iota=0.2,
                G0=signed_G0,
                boozer_I=0.0,
                nfp=5,
                surface_cls=_FakeSurface,
                volume_cls=self._ConstantVolumeLabel,
            )

        self.assertIsInstance(result.boozer_surface, _CapturingPlainBoozerSurface)
        self.assertFalse(hasattr(result.boozer_surface, "I"))
        self.assertEqual(captured["G"], signed_G0)

    def test_attempt_initialize_boozer_surface_threads_requested_volume_target(self):
        module = load_handoff_module()
        surf_prev = SimpleNamespace(
            quadpoints_theta=np.array([0.0, 0.5]),
            quadpoints_phi=np.array([0.0, 0.2]),
            gamma=lambda: np.zeros((2, 2, 3), dtype=float),
        )
        seen_volume_targets = []

        class _FakeSurface:
            def __init__(self, **kwargs):
                self.quadpoints_theta = kwargs["quadpoints_theta"]
                self.quadpoints_phi = kwargs["quadpoints_phi"]
                self.dofs = np.zeros(2, dtype=float)
                self.x = np.zeros(2, dtype=float)
                self._gamma = np.zeros((2, 2, 3), dtype=float)

            def least_squares_fit(self, gamma):
                self._gamma = np.asarray(gamma, dtype=float)

            def gamma(self):
                return self._gamma.copy()

            def is_self_intersecting(self):
                return False

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
                del bs, vol, constraint_weight, options, I
                seen_volume_targets.append(vol_target)
                self.surface = surf
                self.res = {"iota": 0.2, "G": 0.35, "success": True}
                self.need_to_run_code = True

            def run_code(self, iota, G):
                del iota, G
                self.need_to_run_code = False
                return {"success": True}

        result = module.attempt_initialize_boozer_surface(
            surf_prev,
            mpol=8,
            ntor=6,
            bs=object(),
            vol_target=0.035,
            constraint_weight=1.0,
            iota=0.2,
            G0=0.35,
            boozer_I=0.0,
            nfp=5,
            surface_cls=_FakeSurface,
            volume_cls=self._ConstantVolumeLabel,
            boozer_surface_cls=_FakeBoozerSurface,
        )

        self.assertTrue(result.success)
        self.assertEqual(seen_volume_targets, [0.035])

    def test_compute_tf_G0_preserves_tf_current_sign(self):
        module = load_handoff_module()
        tf_coils = [self._fixed_current_coil(-8.0e4) for _ in range(20)]

        self.assertAlmostEqual(module.compute_tf_G0(tf_coils), -4.0e-7 * np.pi * 1.6e6)

    def test_banana_boozer_run_code_calls_pass_explicit_G(self):
        violations = []
        for path in _banana_source_paths():
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
            for node in ast.walk(tree):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "run_code"
                ):
                    continue
                if len(node.args) >= 2 or any(
                    keyword.arg == "G" for keyword in node.keywords
                ):
                    continue
                violations.append(f"{path.relative_to(EXAMPLE_ROOT)}:{node.lineno}")

        self.assertEqual(violations, [])

    def test_validate_stage2_seed_contract_accepts_nonbootable_recovery_input(self):
        module = load_handoff_module()
        stage2_results = {
            **_valid_stage2_contract_fields(),
            "TF_CURRENT_A": -8.0e4,
            "BOOZER_BOOTABLE": False,
            "IOTA_FEASIBLE": False,
            "BOOTABILITY_REASON": module.BOOTABILITY_REASON_SELF_INTERSECTION,
            "BOOTABILITY_SOLVED_IOTA": -1.0e-5,
            "BOOTABILITY_TARGET_IOTA": -0.16,
        }

        module.validate_stage2_seed_contract(stage2_results)

    def test_validate_stage2_seed_contract_rejects_stale_wout_convention_telemetry(
        self,
    ):
        module = load_handoff_module()
        stage2_results = {
            **_valid_stage2_contract_fields(),
            "TF_CURRENT_A": -8.0e4,
            "WOUT_CONVENTION": "positive_ccw",
        }

        with self.assertRaisesRegex(ValueError, "WOUT_CONVENTION"):
            module.validate_stage2_seed_contract(stage2_results)

    def test_validate_stage2_seed_contract_rejects_offspec_wout_convention(self):
        module = load_handoff_module()
        stage2_results = {
            **_valid_stage2_contract_fields(),
            "PLASMA_SURF_PATH": str(POSITIVE_CCW_WOUT_PATH),
            "TF_CURRENT_A": -8.0e4,
            "WOUT_CONVENTION": "positive_ccw",
            "WOUT_OFF_SPEC": True,
        }

        with self.assertRaisesRegex(ValueError, "WOUT_OFF_SPEC=True"):
            module.validate_stage2_seed_contract(stage2_results)

    def test_validate_stage2_seed_contract_accepts_positive_tf_current_magnitude(self):
        module = load_handoff_module()
        stage2_results = {
            **_valid_stage2_contract_fields(),
            "PLASMA_SURF_PATH": str(POSITIVE_CCW_WOUT_PATH),
            "TF_CURRENT_A": 8.0e4,
            "WOUT_CONVENTION": "positive_ccw",
            "WOUT_OFF_SPEC": False,
        }

        module.validate_stage2_seed_contract(stage2_results)

    def test_validate_stage2_seed_contract_rejects_over_limit_tf_current(self):
        module = load_handoff_module()
        stage2_results = {
            **_valid_stage2_contract_fields(),
            "PLASMA_SURF_PATH": str(POSITIVE_CCW_WOUT_PATH),
            "TF_CURRENT_A": 8.00001e4,
            "WOUT_CONVENTION": "positive_ccw",
            "WOUT_OFF_SPEC": False,
        }

        with self.assertRaisesRegex(ValueError, "TF current magnitude limit"):
            module.validate_stage2_seed_contract(stage2_results)

    def test_validate_stage2_seed_contract_rejects_zero_tf_current(self):
        module = load_handoff_module()
        stage2_results = {
            **_valid_stage2_contract_fields(),
            "TF_CURRENT_A": 0.0,
        }

        with self.assertRaisesRegex(ValueError, "finite, nonzero"):
            module.validate_stage2_seed_contract(stage2_results)

    def test_validate_stage2_seed_contract_accepts_in_vessel_banana_radius_drift(self):
        module = load_handoff_module()
        stage2_results = {
            **_valid_stage2_contract_fields(),
            "banana_surf_radius": 0.21,
        }

        module.validate_stage2_seed_contract(stage2_results)

    def test_validate_stage2_seed_contract_rejects_out_of_vessel_banana_radius(self):
        module = load_handoff_module()
        stage2_results = {
            **_valid_stage2_contract_fields(),
            "banana_surf_radius": 0.223,
        }

        with self.assertRaisesRegex(ValueError, "vacuum vessel minor radius"):
            module.validate_stage2_seed_contract(stage2_results)

    def test_validate_stage2_seed_contract_accepts_curvature_threshold_below_ceiling(
        self,
    ):
        module = load_handoff_module()
        stage2_results = {
            **_valid_stage2_contract_fields(),
            "CURVATURE_THRESHOLD": 50.0,
        }

        module.validate_stage2_seed_contract(stage2_results)

    def test_validate_stage2_seed_contract_rejects_curvature_threshold_above_ceiling(
        self,
    ):
        module = load_handoff_module()
        stage2_results = {
            **_valid_stage2_contract_fields(),
            "CURVATURE_THRESHOLD": 100.1,
        }

        with self.assertRaisesRegex(ValueError, "curvature threshold exceeds"):
            module.validate_stage2_seed_contract(stage2_results)

    def test_validate_stage2_seed_contract_rejects_nan_curvature_threshold(self):
        module = load_handoff_module()
        stage2_results = {
            **_valid_stage2_contract_fields(),
            "CURVATURE_THRESHOLD": float("nan"),
        }

        with self.assertRaisesRegex(ValueError, "CURVATURE_THRESHOLD must be finite"):
            module.validate_stage2_seed_contract(stage2_results)

    def test_validate_stage2_seed_contract_accepts_proxy_vf_ratio_drift_telemetry(self):
        module = load_handoff_module()
        stage2_results = {
            **_valid_stage2_contract_fields(),
            "TF_CURRENT_A": -8.0e4,
            "PROXY_PLASMA_CURRENT_A": 9.0e3,
            "VF_CURRENT_A": 5.0e2,
        }

        module.validate_stage2_seed_contract(stage2_results)

    def test_validate_stage2_seed_contract_accepts_jhalpern_signed_proxy_vf(self):
        module = load_handoff_module()
        stage2_results = _valid_jhalpern_stage2_contract_fields()

        module.validate_stage2_seed_contract(stage2_results)

    def test_validate_stage2_seed_contract_rejects_jhalpern_missing_replay_metadata(
        self,
    ):
        module = load_handoff_module()
        stage2_results = _valid_jhalpern_stage2_contract_fields()
        stage2_results.pop("IOTA_TARGET_SIGN")

        with self.assertRaisesRegex(ValueError, "IOTA_TARGET_SIGN"):
            module.validate_stage2_seed_contract(stage2_results)

    def test_validate_stage2_seed_contract_rejects_jhalpern_null_replay_metadata(self):
        module = load_handoff_module()
        stage2_results = _valid_jhalpern_stage2_contract_fields()
        stage2_results["IOTA_TARGET_SIGN"] = None

        with self.assertRaisesRegex(ValueError, "IOTA_TARGET_SIGN"):
            module.validate_stage2_seed_contract(stage2_results)

    def test_validate_stage2_seed_contract_rejects_jhalpern_policy_drift(self):
        module = load_handoff_module()
        stage2_results = _valid_jhalpern_stage2_contract_fields()
        stage2_results["PROXY_VF_CURRENT_SCALAR_POLICY"] = "nonnegative_magnitude"

        with self.assertRaisesRegex(ValueError, "PROXY_VF_CURRENT_SCALAR_POLICY"):
            module.validate_stage2_seed_contract(stage2_results)

    def test_validate_stage2_seed_contract_rejects_jhalpern_coil_count_drift(self):
        module = load_handoff_module()
        stage2_results = _valid_jhalpern_stage2_contract_fields()
        stage2_results["NUM_VF_COILS"] = 0

        with self.assertRaisesRegex(ValueError, "NUM_VF_COILS"):
            module.validate_stage2_seed_contract(stage2_results)

    def test_upgrade_does_not_fabricate_jhalpern_required_policy_metadata(self):
        artifact_contracts = load_artifact_contracts_module()
        handoff = load_handoff_module()
        stage2_results = _valid_jhalpern_stage2_contract_fields()
        stage2_results.pop("PROXY_VF_CURRENT_SCALAR_POLICY")

        upgraded = artifact_contracts.upgrade_legacy_stage2_artifact_results(
            stage2_results,
        )

        self.assertNotIn("PROXY_VF_CURRENT_SCALAR_POLICY", upgraded)
        with self.assertRaisesRegex(ValueError, "PROXY_VF_CURRENT_SCALAR_POLICY"):
            handoff.validate_stage2_seed_contract(upgraded)

    def test_upgrade_does_not_fabricate_jhalpern_required_current_metadata(self):
        artifact_contracts = load_artifact_contracts_module()
        handoff = load_handoff_module()
        stage2_results = _valid_jhalpern_stage2_contract_fields()
        stage2_results.pop("VF_CURRENT_A")

        upgraded = artifact_contracts.upgrade_legacy_stage2_artifact_results(
            stage2_results,
        )

        self.assertNotIn("VF_CURRENT_A", upgraded)
        with self.assertRaisesRegex(ValueError, "VF_CURRENT_A"):
            handoff.validate_stage2_seed_contract(upgraded)

    def test_upgrade_does_not_fabricate_jhalpern_boozer_convention_metadata(self):
        artifact_contracts = load_artifact_contracts_module()
        handoff = load_handoff_module()
        stage2_results = _valid_jhalpern_stage2_contract_fields()
        stage2_results.pop("BOOZER_CURRENT_CONVENTION")

        upgraded = artifact_contracts.upgrade_legacy_stage2_artifact_results(
            stage2_results,
        )

        self.assertNotIn("BOOZER_CURRENT_CONVENTION", upgraded)
        with self.assertRaisesRegex(ValueError, "BOOZER_CURRENT_CONVENTION"):
            handoff.validate_stage2_seed_contract(upgraded)

    def test_validate_stage2_seed_contract_rejects_jhalpern_zero_proxy_current(self):
        module = load_handoff_module()
        stage2_results = _valid_jhalpern_stage2_contract_fields()
        stage2_results["PROXY_PLASMA_CURRENT_A"] = 0.0

        with self.assertRaisesRegex(ValueError, "non-zero PROXY_PLASMA_CURRENT_A"):
            module.validate_stage2_seed_contract(stage2_results)

    def test_validate_stage2_seed_recovery_contract_accepts_traversal_geometry(self):
        module = load_handoff_module()
        stage2_results = {
            **_valid_stage2_contract_fields(),
            "TF_CURRENT_A": -8.0e4,
            "COIL_LENGTH": 0.91,
            "COIL_WIDTH": 0.252,
            "SELF_INTERSECT_PENALTY": 2.0e-4,
        }

        module.validate_stage2_seed_recovery_contract(stage2_results)
        module.validate_stage2_seed_contract(stage2_results)

    def test_stage2_seed_hardware_contract_uses_artifact_length_min_target(self):
        module = load_handoff_module()
        stage2_results = {
            **_valid_stage2_contract_fields(),
            "LENGTH_TARGET": 1.80,
            "LENGTH_MIN_TARGET": 0.90,
            "COIL_LENGTH": 0.91,
        }

        hardware_status = module.evaluate_stage2_seed_hardware_contract(stage2_results)

        self.assertTrue(hardware_status["success"])
        self.assertEqual(
            hardware_status["constraints"]["coil_length_min"]["threshold"],
            0.90,
        )

    def test_validate_stage2_seed_recovery_contract_rejects_current_violation(self):
        module = load_handoff_module()
        stage2_results = {
            **_valid_stage2_contract_fields(),
            "TF_CURRENT_A": -8.0e4,
            "BANANA_CURRENT_A": 1.6001e4,
        }

        with self.assertRaisesRegex(
            ValueError,
            "non-traversable HBT-EP hardware contract: .*banana_current",
        ):
            module.validate_stage2_seed_recovery_contract(stage2_results)

    def test_validate_stage2_seed_recovery_contract_rejects_missing_metrics(self):
        module = load_handoff_module()
        stage2_results = {
            **_valid_stage2_contract_fields(),
            "TF_CURRENT_A": -8.0e4,
        }
        stage2_results.pop("COIL_LENGTH")

        with self.assertRaisesRegex(
            ValueError,
            "missing required hardware constraint metric coil_length",
        ):
            module.validate_stage2_seed_recovery_contract(stage2_results)

    def test_validate_stage2_seed_handoff_contract_accepts_nonbootable_telemetry(self):
        module = load_handoff_module()
        stage2_results = {
            **_valid_stage2_contract_fields(),
            "TF_CURRENT_A": -8.0e4,
            "BOOZER_BOOTABLE": False,
            "IOTA_FEASIBLE": False,
            "BOOTABILITY_REASON": module.BOOTABILITY_REASON_SELF_INTERSECTION,
            "BOOTABILITY_SOLVED_IOTA": -1.0e-5,
            "BOOTABILITY_TARGET_IOTA": -0.16,
        }

        module.validate_stage2_seed_handoff_contract(stage2_results)

    def test_classify_bootability_result_records_iota_miss_as_telemetry(
        self,
    ):
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

        self.assertTrue(module.bootability_passes(status))
        self.assertTrue(status["BOOZER_BOOTABLE"])
        self.assertFalse(status["IOTA_NEAR_TARGET"])
        self.assertFalse(status["IOTA_FEASIBLE"])
        self.assertEqual(status["BOOTABILITY_REASON"], module.BOOTABILITY_REASON_OK)
        self.assertAlmostEqual(status["BOOTABILITY_ABS_IOTA_ERROR"], 0.08)

    def test_bootability_passes_rejects_truthy_non_boolean_flags(self):
        module = load_handoff_module()

        self.assertFalse(
            module.bootability_passes(
                {
                    "BOOZER_BOOTABLE": 1,
                    "IOTA_FEASIBLE": "true",
                }
            )
        )

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
                self.need_to_run_code = True

            def run_code(self, iota, G):
                del iota, G
                self.surface.x = np.array([9.0, -4.0], dtype=float)
                self.res["iota"] = 0.41
                self.res["G"] = 0.72
                self.res["success"] = False
                self.need_to_run_code = False
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
            volume_cls=self._ConstantVolumeLabel,
            boozer_surface_cls=_FakeBoozerSurface,
        )

        self.assertFalse(result.solve_success)
        self.assertFalse(result.success)
        self.assertAlmostEqual(result.solved_iota, 0.41)
        self.assertAlmostEqual(result.solved_G, 0.72)
        self.assertIsNone(result.error_type)
        np.testing.assert_allclose(result.boozer_surface.surface.x, [9.0, -4.0])

    def test_attempt_initialize_boozer_surface_classifies_self_intersection_check_failure(
        self,
    ):
        module = load_handoff_module()
        surf_prev = SimpleNamespace(
            quadpoints_theta=np.array([0.0, 0.5]),
            quadpoints_phi=np.array([0.0, 0.2]),
            gamma=lambda: np.zeros((2, 2, 3), dtype=float),
        )
        created_boozer_surfaces = []

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
            ):
                del mpol, ntor, nfp, stellsym
                self.quadpoints_theta = quadpoints_theta
                self.quadpoints_phi = quadpoints_phi
                self.dofs = np.zeros(2, dtype=float)
                self.x = np.zeros(2, dtype=float)
                self._gamma = np.zeros((2, 2, 3), dtype=float)
                self.self_intersection_calls = 0

            def least_squares_fit(self, gamma):
                self._gamma = np.asarray(gamma, dtype=float)

            def gamma(self):
                return self._gamma.copy()

            def set_dofs(self, dofs):
                resolved_dofs = np.asarray(dofs, dtype=float)
                self.dofs = resolved_dofs
                self.x = resolved_dofs.copy()

            def is_self_intersecting(self):
                self.self_intersection_calls += 1
                raise Exception("surface 'goes back' on itself")

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
                self.res = None
                self.need_to_run_code = True
                created_boozer_surfaces.append(self)

            def run_code(self, iota, G):
                self.surface.x = np.array([9.0, -4.0], dtype=float)
                self.res = {"iota": float(iota), "G": float(G), "success": True}
                self.need_to_run_code = False
                return {"success": True, "iota": float(iota), "G": float(G)}

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
            volume_cls=self._ConstantVolumeLabel,
            boozer_surface_cls=_FakeBoozerSurface,
        )

        self.assertTrue(result.solve_success)
        self.assertTrue(result.self_intersecting)
        self.assertFalse(result.success)
        boozer_surface = created_boozer_surfaces[0]
        np.testing.assert_allclose(boozer_surface.surface.x, [0.0, 0.0])
        self.assertIsNone(boozer_surface.res)
        self.assertTrue(boozer_surface.need_to_run_code)
        self.assertEqual(boozer_surface.surface.self_intersection_calls, 1)

    def test_surface_self_intersection_status_propagates_unrelated_error(self):
        geometry = importlib.import_module("banana_opt.single_stage_geometry")

        class _FakeSurface:
            def is_self_intersecting(self):
                raise RuntimeError("ground missing")

        with self.assertRaisesRegex(RuntimeError, "ground missing"):
            geometry.surface_self_intersection_status(_FakeSurface())

    def test_attempt_initialize_boozer_surface_propagates_volume_exception(self):
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
            ):
                del mpol, ntor, nfp, stellsym
                self.quadpoints_theta = quadpoints_theta
                self.quadpoints_phi = quadpoints_phi
                self.dofs = np.zeros(2, dtype=float)
                self.x = np.zeros(2, dtype=float)
                self._gamma = np.zeros((2, 2, 3), dtype=float)

            def least_squares_fit(self, gamma):
                self._gamma = np.asarray(gamma, dtype=float)

            def gamma(self):
                return self._gamma.copy()

            def is_self_intersecting(self):
                return False

        class _FakeVolume:
            def __init__(self, surface):
                self.surface = surface

            def J(self):
                raise RuntimeError("volume broken")

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
                self.res = None
                self.need_to_run_code = True

            def run_code(self, iota, G):
                self.res = {"iota": float(iota), "G": float(G), "success": True}
                self.need_to_run_code = False
                return {"success": True, "iota": float(iota), "G": float(G)}

        with self.assertRaisesRegex(RuntimeError, "volume broken"):
            module.attempt_initialize_boozer_surface(
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

    def test_attempt_initialize_boozer_surface_assigns_seed_dofs_after_construction(
        self,
    ):
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
                self.need_to_run_code = True

            def run_code(self, iota, G):
                del iota, G
                self.need_to_run_code = False
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
            volume_cls=self._ConstantVolumeLabel,
            boozer_surface_cls=_FakeBoozerSurface,
        )

        self.assertTrue(result.solve_success)
        self.assertTrue(result.success)
        self.assertGreaterEqual(len(_CtorRejectsRawArraySurface.assigned_dofs), 1)
        np.testing.assert_allclose(
            _CtorRejectsRawArraySurface.assigned_dofs[0],
            np.array([2.5, -1.5], dtype=float),
        )

    def test_attempt_initialize_boozer_surface_fits_mismatched_seed_order(self):
        module = load_handoff_module()
        seed_gamma = np.ones((2, 2, 3), dtype=float)
        surf_prev = SimpleNamespace(
            quadpoints_theta=np.array([0.0, 0.5]),
            quadpoints_phi=np.array([0.0, 0.2]),
            gamma=lambda: np.zeros((2, 2, 3), dtype=float),
        )
        initial_surface_guess = SimpleNamespace(
            get_dofs=lambda: np.array([2.5, -1.5], dtype=float),
            gamma=lambda: seed_gamma,
        )

        class _HigherOrderSurface:
            fit_inputs = []

            def __init__(self, **kwargs):
                del kwargs
                self.dofs = np.zeros(4, dtype=float)
                self._gamma = np.zeros((2, 2, 3), dtype=float)

            def least_squares_fit(self, gamma):
                self._gamma = np.asarray(gamma, dtype=float)
                type(self).fit_inputs.append(self._gamma.copy())

            def gamma(self):
                return self._gamma.copy()

            def is_self_intersecting(self):
                return False

        class _FakeBoozerSurface:
            def __init__(
                self, bs, surf, vol, vol_target, constraint_weight, options, I=0.0
            ):
                del bs, vol, vol_target, constraint_weight, options, I
                self.surface = surf
                self.res = {"iota": 0.2, "G": 0.35, "success": True}
                self.need_to_run_code = True

            def run_code(self, iota, G):
                del iota, G
                self.need_to_run_code = False
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
            surface_cls=_HigherOrderSurface,
            volume_cls=self._ConstantVolumeLabel,
            boozer_surface_cls=_FakeBoozerSurface,
        )

        self.assertTrue(result.success)
        np.testing.assert_allclose(_HigherOrderSurface.fit_inputs[0], seed_gamma)

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

    def test_run_boozer_with_failure_policy_handles_fresh_surface_without_cached_res(
        self,
    ):
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

    def test_run_boozer_with_failure_policy_restores_last_successful_state_on_failed_result(
        self,
    ):
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

    def test_run_boozer_with_failure_policy_restores_cached_failed_state_on_reported_failure(
        self,
    ):
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

    def test_probe_stage2_seed_bootability_reports_missing_metadata_without_loading_bs(
        self,
    ):
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

    def test_probe_stage2_seed_bootability_does_not_gate_on_full_hardware_contract(
        self,
    ):
        module = load_handoff_module()
        initial_surface = SimpleNamespace(nfp=5)
        initialization = module.BoozerInitializationResult(
            boozer_surface=object(),
            solve_success=True,
            self_intersecting=False,
            success=True,
            solved_iota=0.2,
            solved_G=0.3,
            volume=0.1,
        )

        with (
            patch.object(
                module,
                "validate_stage2_seed_contract",
                side_effect=ValueError("hardware contract failure"),
            ) as validate_contract,
            patch.object(
                module,
                "partition_loaded_stage2_coils",
                return_value=SimpleNamespace(tf_coils=[object()]),
            ),
            patch.object(
                module,
                "resolve_stage2_tf_current_A",
                return_value=-80000.0,
            ),
            patch.object(
                module,
                "_probe_initialization_inputs",
                return_value=(initial_surface, 0.1, None, 0.2, 0.3),
            ),
            patch.object(
                module,
                "attempt_initialize_boozer_surface",
                return_value=initialization,
            ),
        ):
            status = module.probe_stage2_seed_bootability(
                stage2_bs_path="/tmp/demo/biot_savart_opt.json",
                stage2_artifact_results={
                    "MAJOR_RADIUS": 0.976,
                    "TOROIDAL_FLUX": 1.0,
                    "banana_surf_radius": 0.21,
                },
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
                bs_loader=lambda path: SimpleNamespace(coils=[object()]),
            )

        validate_contract.assert_not_called()
        self.assertEqual(status["BOOTABILITY_REASON"], module.BOOTABILITY_REASON_OK)
        self.assertTrue(module.bootability_passes(status))

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

    def test_loaded_seed_current_source_contract_rejects_unbacked_current_override(
        self,
    ):
        module = load_handoff_module()
        partitions = _stage2_coil_partitions(
            module,
            finite_current_mode="jhalpern30_proxy_field",
        )
        stage2_results = {
            **_valid_stage2_contract_fields(),
            "FINITE_CURRENT_MODE": "jhalpern30_proxy_field",
            "PROXY_PLASMA_CURRENT_A": 0.0,
            "VF_CURRENT_A": 0.0,
        }

        with self.assertRaisesRegex(
            ValueError, "cannot retarget physical plasma current"
        ):
            module.validate_loaded_seed_current_source_contract(
                finite_current_mode="jhalpern30_proxy_field",
                effective_current_mode="jhalpern30_proxy_field",
                plasma_current_A=-400.0,
                plasma_current_input_source="physical_A",
                stage2_results=stage2_results,
                coil_partitions=partitions,
            )

    def test_loaded_seed_current_source_contract_rejects_explicit_zero_retarget(
        self,
    ):
        module = load_handoff_module()
        partitions = _stage2_coil_partitions(
            module,
            num_proxy_coils=1,
            num_vf_coils=1,
            finite_current_mode="wataru_proxy_field",
        )

        with self.assertRaisesRegex(
            ValueError, "cannot retarget physical plasma current"
        ):
            module.validate_loaded_seed_current_source_contract(
                finite_current_mode="wataru_proxy_field",
                effective_current_mode="vacuum",
                plasma_current_A=0.0,
                plasma_current_input_source="physical_A",
                stage2_results={
                    **_valid_stage2_contract_fields(),
                    "FINITE_CURRENT_MODE": "wataru_proxy_field",
                    "PROXY_PLASMA_CURRENT_A": 9000.0,
                    "VF_CURRENT_A": 9000.0 / 6.5,
                    "NUM_PROXY_COILS": 1,
                    "NUM_VF_COILS": 1,
                },
                coil_partitions=partitions,
            )

    def test_loaded_seed_current_source_contract_rejects_missing_proxy_vf_sources(
        self,
    ):
        module = load_handoff_module()
        partitions = _stage2_coil_partitions(
            module,
            finite_current_mode="jhalpern30_proxy_field",
        )
        stage2_results = {
            **_valid_jhalpern_stage2_contract_fields(),
            "NUM_PROXY_COILS": 0,
            "NUM_VF_COILS": 0,
        }

        with self.assertRaisesRegex(
            ValueError, "requires loaded proxy/VF field sources"
        ):
            module.validate_loaded_seed_current_source_contract(
                finite_current_mode="jhalpern30_proxy_field",
                effective_current_mode="jhalpern30_proxy_field",
                plasma_current_A=-6500.0,
                plasma_current_input_source="artifact_default_A",
                stage2_results=stage2_results,
                coil_partitions=partitions,
            )

    def test_loaded_seed_current_source_contract_rejects_nonvacuum_boozer_surrogate(
        self,
    ):
        module = load_handoff_module()
        partitions = _stage2_coil_partitions(
            module,
            finite_current_mode="boozer_surrogate",
        )

        with self.assertRaisesRegex(
            ValueError, "cannot replay non-vacuum 'boozer_surrogate' current"
        ):
            module.validate_loaded_seed_current_source_contract(
                finite_current_mode="boozer_surrogate",
                effective_current_mode="boozer_surrogate",
                plasma_current_A=-400.0,
                plasma_current_input_source="artifact_default_A",
                stage2_results={
                    **_valid_stage2_contract_fields(),
                    "FINITE_CURRENT_MODE": "boozer_surrogate",
                    "BOOZER_I": -0.0005026548245743669,
                },
                coil_partitions=partitions,
            )

    def test_loaded_seed_current_source_contract_accepts_backed_wataru_one_vf(
        self,
    ):
        module = load_handoff_module()
        profile = get_finite_current_profile("wataru_proxy_field")
        partitions = _stage2_coil_partitions(
            module,
            num_proxy_coils=1,
            num_vf_coils=1,
            finite_current_mode="wataru_proxy_field",
        )

        module.validate_loaded_seed_current_source_contract(
            finite_current_mode="wataru_proxy_field",
            effective_current_mode="wataru_proxy_field",
            plasma_current_A=9000.0,
            plasma_current_input_source="artifact_default_A",
            stage2_results={
                **_valid_stage2_contract_fields(),
                "FINITE_CURRENT_MODE": "wataru_proxy_field",
                "BOOZER_CURRENT_CONVENTION": "mu0",
                "G0_POLICY": "signed_explicit_tf_current",
                "PROXY_PLACEMENT_MODE": "vmec_axis_zeroth_coefficients",
                "PROXY_VF_CURRENT_SCALAR_POLICY": "nonnegative_magnitude",
                "PROXY_PLASMA_CURRENT_A": 9000.0,
                "VF_CURRENT_A": 9000.0 / 6.5,
                "VF_TEMPLATE_PATH": str(profile.default_vf_template_path),
                "VF_TEMPLATE_SHA256": profile.vf_template_sha256,
                "VF_CURRENT_SIGN_POLICY": "template_sign_vf_current_scalar",
                "VF_CURRENT_MUTABILITY": "independent_fixed_current",
                "NUM_PROXY_COILS": 1,
                "NUM_VF_COILS": 1,
                "TOTAL_COILS": 31,
            },
            coil_partitions=partitions,
        )

    def test_loaded_seed_current_source_contract_accepts_backed_proxy_field_current(
        self,
    ):
        module = load_handoff_module()
        profile = get_finite_current_profile("jhalpern30_proxy_field")
        partitions = _stage2_coil_partitions(
            module,
            num_tf_coils=profile.default_num_tf_coils,
            num_banana_coils=profile.default_num_banana_coils,
            num_proxy_coils=profile.default_num_proxy_coils,
            num_vf_coils=profile.default_num_vf_coils,
            finite_current_mode=profile.mode,
        )

        module.validate_loaded_seed_current_source_contract(
            finite_current_mode=profile.mode,
            effective_current_mode=profile.mode,
            plasma_current_A=-6500.0,
            plasma_current_input_source="artifact_default_A",
            stage2_results=_valid_jhalpern_stage2_contract_fields(),
            coil_partitions=partitions,
        )

    def test_materialize_stage2_seed_variant_from_currents_preserves_order_and_metadata(
        self,
    ):
        handoff_module = load_handoff_module()
        banana_scan_module = load_banana_scan_module()
        workflow_runner_common = load_workflow_runner_common_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            stage2_bs_path, stage2_results, points, _ = _build_round_trip_seed(
                Path(tmpdir),
                include_proxy_vf=True,
            )
            stage2_results = {
                **stage2_results,
                "COIL_GROUPS": [
                    {"role": "tf", "start": 0, "count": 20},
                    {"role": "banana", "start": 20, "count": 2},
                    {"role": "proxy", "start": 22, "count": 1},
                    {"role": "vf", "start": 23, "count": 1},
                ],
                "BANANA_CURRENT_A": 1.1e4,
                "MAJOR_RADIUS": 0.976,
                "TOROIDAL_FLUX": 0.24,
                "banana_surf_radius": 0.21,
            }
            _write_json(stage2_bs_path.with_name("results.json"), stage2_results)
            donor_loaded = handoff_module.load(str(stage2_bs_path))
            donor_currents = [
                float(coil.current.get_value()) for coil in donor_loaded.coils
            ]

            variant_bs_path, variant_results_path = (
                banana_scan_module.materialize_stage2_seed_variant_from_currents(
                    stage2_bs_path=stage2_bs_path,
                    stage2_results=stage2_results,
                    variant_root=Path(tmpdir) / "variant",
                    banana_currents_a=[1.2e4, -9.5e3],
                    requested_num_tf_coils=20,
                    extra_results_updates={"CUSTOM_PROVENANCE": "handoff-test"},
                )
            )

            loaded_results_path, loaded_results = (
                workflow_runner_common.load_stage2_artifact_results(variant_bs_path)
            )
            variant_loaded = handoff_module.load(str(variant_bs_path))
            variant_loaded.set_points(points)
            variant_currents = [
                float(coil.current.get_value()) for coil in variant_loaded.coils
            ]
            partitions = handoff_module.partition_loaded_stage2_coils(
                variant_loaded.coils,
                stage2_results=loaded_results,
                requested_num_tf_coils=20,
            )
            expected_digest = banana_scan_module.compute_stage2_bs_sha256(
                variant_bs_path
            )
            raw_variant_results = json.loads(
                variant_results_path.read_text(encoding="utf-8")
            )

        self.assertEqual(loaded_results_path, variant_results_path)
        self.assertEqual(variant_currents[:20], donor_currents[:20])
        self.assertEqual(variant_currents[20:22], [1.2e4, -9.5e3])
        self.assertEqual(variant_currents[22:], donor_currents[22:])
        self.assertEqual(
            [coil.current.get_value() for coil in partitions.banana_coils],
            [1.2e4, -9.5e3],
        )
        self.assertEqual(loaded_results["COIL_GROUPS"], stage2_results["COIL_GROUPS"])
        self.assertEqual(loaded_results["NUM_PROXY_COILS"], 1)
        self.assertEqual(loaded_results["NUM_VF_COILS"], 1)
        self.assertEqual(loaded_results["BANANA_CURRENT_MODE"], "independent")
        self.assertEqual(loaded_results["BANANA_CURRENTS_A"], [1.2e4, -9.5e3])
        self.assertEqual(loaded_results["BANANA_CURRENT_A"], 1.2e4)
        self.assertEqual(
            loaded_results["DONOR_BANANA_CURRENTS_A"],
            [1.1e4, -1.1e4],
        )
        self.assertEqual(loaded_results["DONOR_BANANA_CURRENT_A"], 1.1e4)
        self.assertEqual(loaded_results["CUSTOM_PROVENANCE"], "handoff-test")
        self.assertEqual(loaded_results["STAGE2_BS_SHA256"], expected_digest)
        self.assertEqual(raw_variant_results["STAGE2_BS_SHA256"], expected_digest)

    def test_materialized_vector_seed_survives_into_independent_single_stage_state(
        self,
    ):
        handoff_module = load_handoff_module()
        banana_scan_module = load_banana_scan_module()
        current_mode_module = load_current_mode_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            stage2_bs_path, stage2_results, points, _ = _build_round_trip_seed(
                Path(tmpdir),
                include_proxy_vf=False,
            )
            variant_bs_path, _ = (
                banana_scan_module.materialize_stage2_seed_variant_from_currents(
                    stage2_bs_path=stage2_bs_path,
                    stage2_results=stage2_results,
                    variant_root=Path(tmpdir) / "variant",
                    banana_currents_a=[1.2e4, -9.5e3],
                    requested_num_tf_coils=20,
                )
            )
            variant_loaded = handoff_module.load(str(variant_bs_path))
            variant_loaded.set_points(points)
            partitions = handoff_module.partition_loaded_stage2_coils(
                variant_loaded.coils,
                stage2_results=stage2_results,
                requested_num_tf_coils=20,
            )

            _, resolved_partitions, current_state = (
                current_mode_module.resolve_single_stage_banana_current_state(
                    variant_loaded,
                    partitions,
                    mode="independent",
                )
            )

        self.assertEqual(current_state.seed_currents_A, (1.2e4, -9.5e3))
        self.assertEqual(current_state.current_values_A(), (1.2e4, -9.5e3))
        self.assertEqual(current_state.num_control_currents(), 2)
        self.assertEqual(current_state.compatibility_current_A(), 1.2e4)
        self.assertIsNot(
            resolved_partitions.banana_coils[0].current,
            resolved_partitions.banana_coils[1].current,
        )

    def test_wataru_round_trip_field_parity_survives_stage2_write_and_reload(self):
        module = load_handoff_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            stage2_bs_path, stage2_results, points, expected_field = (
                _build_round_trip_seed(
                    Path(tmpdir),
                    include_proxy_vf=True,
                )
            )

            loaded_bs = module.load(str(stage2_bs_path))
            loaded_bs.set_points(points)
            actual_field = loaded_bs.B()
            partitions = module.partition_loaded_stage2_coils(
                loaded_bs.coils,
                stage2_results=stage2_results,
                requested_num_tf_coils=20,
            )

        np.testing.assert_allclose(
            actual_field, expected_field, rtol=1.0e-12, atol=1.0e-12
        )
        self.assertEqual(len(partitions.tf_coils), 20)
        self.assertEqual(len(partitions.banana_coils), 2)
        self.assertEqual(len(partitions.proxy_coils), 1)
        self.assertEqual(len(partitions.vf_coils), 1)
        self.assertEqual(partitions.finite_current_mode, "wataru_proxy_field")

    def test_round_trip_field_timing_smoke_covers_legacy_and_wataru_partition_shapes(
        self,
    ):
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

    def test_probe_stage2_seed_bootability_smoke_legacy_donor_uses_remainder_partition(
        self,
    ):
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
                boozer_surface=SimpleNamespace(
                    surface=SimpleNamespace(volume=lambda: 0.1)
                ),
                solve_success=True,
                self_intersecting=False,
                success=True,
                solved_iota=0.2,
                solved_G=G0,
                volume=0.1,
            )

        with (
            patch.object(
                module,
                "build_surface_configs",
                return_value=[{"initial_surface": fake_surface, "target_volume": 0.1}],
            ),
            patch.object(
                module,
                "attempt_initialize_boozer_surface",
                side_effect=fake_attempt_initialize_boozer_surface,
            ),
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

    def test_probe_stage2_seed_bootability_smoke_wataru_donor_preserves_extra_coil_metadata(
        self,
    ):
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
                boozer_surface=SimpleNamespace(
                    surface=SimpleNamespace(volume=lambda: 0.1)
                ),
                solve_success=True,
                self_intersecting=False,
                success=True,
                solved_iota=0.2,
                solved_G=G0,
                volume=0.1,
            )

        with (
            patch.object(
                module,
                "build_surface_configs",
                return_value=[{"initial_surface": fake_surface, "target_volume": 0.1}],
            ),
            patch.object(
                module,
                "attempt_initialize_boozer_surface",
                side_effect=fake_attempt_initialize_boozer_surface,
            ),
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
                boozer_surface=SimpleNamespace(
                    surface=SimpleNamespace(volume=lambda: 0.1)
                ),
                solve_success=True,
                self_intersecting=False,
                success=True,
                solved_iota=0.2,
                solved_G=G0,
                volume=0.1,
            )

        with (
            patch.object(
                module,
                "build_equilibrium_path",
                side_effect=AssertionError(
                    "warm-start probe should not read the equilibrium"
                ),
            ),
            patch.object(
                module,
                "build_surface_configs",
                side_effect=AssertionError(
                    "warm-start probe should not rebuild a cold-start surface"
                ),
            ),
            patch.object(
                module,
                "load_warm_start_boozer_seed",
                return_value=module.WarmStartBoozerSeed(
                    surface=warm_start_surface,
                    iota=0.2,
                    G=module.compute_tf_G0(tf_coils),
                    source_path=Path("/tmp/legacy/surf_opt_boozer_surface.json"),
                ),
            ) as warm_start_loader,
            patch.object(
                module,
                "attempt_initialize_boozer_surface",
                side_effect=fake_attempt_initialize_boozer_surface,
            ),
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
        compat = importlib.import_module("banana_opt.json_compat")
        self.assertIs(
            warm_start_loader.call_args.kwargs["artifact_loader"],
            compat.load_boozer_finite_i,
        )
        self.assertIs(recorded["surf_prev"], warm_start_surface)
        self.assertIs(recorded["initial_surface_guess"], warm_start_surface)
        self.assertAlmostEqual(recorded["iota"], 0.2)
        self.assertEqual(recorded["nfp"], 5)

    def test_probe_stage2_seed_bootability_rebuilds_when_loaded_surface_has_no_solved_state(
        self,
    ):
        module = load_handoff_module()
        tf_coils, fake_bs, stage2_artifact_results = self._bootability_smoke_inputs(
            include_proxy_vf=False
        )
        cold_surface = SimpleNamespace(nfp=5)
        surface_only = SimpleNamespace(nfp=5)
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
                boozer_surface=SimpleNamespace(
                    surface=SimpleNamespace(volume=lambda: 0.1)
                ),
                solve_success=True,
                self_intersecting=False,
                success=True,
                solved_iota=0.2,
                solved_G=G0,
                volume=0.1,
            )

        with (
            patch.object(
                module,
                "build_equilibrium_path",
                return_value="/tmp/equilibria/demo.nc",
            ) as build_equilibrium_path,
            patch.object(
                module,
                "build_surface_configs",
                return_value=[{"initial_surface": cold_surface, "target_volume": 0.1}],
            ) as build_surface_configs,
            patch.object(
                module,
                "load_warm_start_boozer_seed",
                return_value=module.WarmStartBoozerSeed(
                    surface=surface_only,
                    iota=0.2,
                    G=None,
                    source_path=Path("/tmp/legacy/surf_opt_boozer_surface.json"),
                ),
            ),
            patch.object(
                module,
                "attempt_initialize_boozer_surface",
                side_effect=fake_attempt_initialize_boozer_surface,
            ),
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
        build_equilibrium_path.assert_called_once()
        build_surface_configs.assert_called_once()
        self.assertIs(recorded["surf_prev"], cold_surface)
        self.assertIsNone(recorded["initial_surface_guess"])
        self.assertAlmostEqual(recorded["iota"], 0.2)
        self.assertAlmostEqual(recorded["G0"], module.compute_tf_G0(tf_coils))
        self.assertEqual(recorded["nfp"], 5)

    def test_probe_stage2_seed_bootability_uses_warm_start_boozer_surface_artifact(
        self,
    ):
        module = load_handoff_module()
        tf_coils, fake_bs, stage2_artifact_results = self._bootability_smoke_inputs(
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
                boozer_surface=SimpleNamespace(
                    surface=SimpleNamespace(volume=lambda: 0.1)
                ),
                solve_success=True,
                self_intersecting=False,
                success=True,
                solved_iota=0.2003,
                solved_G=0.377,
                volume=0.1,
            )

        with (
            patch.object(
                module,
                "build_equilibrium_path",
                side_effect=AssertionError(
                    "warm-start probe should not read the equilibrium"
                ),
            ),
            patch.object(
                module,
                "build_surface_configs",
                side_effect=AssertionError(
                    "warm-start probe should not rebuild a cold-start surface"
                ),
            ),
            patch.object(
                module,
                "attempt_initialize_boozer_surface",
                side_effect=fake_attempt_initialize_boozer_surface,
            ),
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
                bs_loader=lambda _path: fake_bs,
                warm_start_loader=fake_loader,
            )

        self.assertTrue(module.bootability_passes(status))
        self.assertIs(recorded["surf_prev"], warm_start_surface)
        self.assertIs(recorded["bs"], fake_bs)
        self.assertAlmostEqual(recorded["iota"], 0.2003)
        self.assertAlmostEqual(recorded["G0"], 0.377)
        self.assertIs(recorded["initial_surface_guess"], warm_start_surface)
        self.assertEqual(recorded["nfp"], 5)

    def test_load_warm_start_boozer_seed_reads_sidecar_solved_state(self):
        module = load_handoff_module()
        warm_start_surface = SimpleNamespace(nfp=5)

        with tempfile.TemporaryDirectory() as tmpdir:
            warm_start_path = Path(tmpdir) / "surf_opt_boozer_surface.json"
            warm_start_path.write_text("{}", encoding="utf-8")
            state_path = module.warm_start_boozer_state_path(warm_start_path)
            state_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "surface_path": warm_start_path.name,
                        "iota": 0.175,
                        "G": -0.41,
                    }
                ),
                encoding="utf-8",
            )

            seed = module.load_warm_start_boozer_seed(
                warm_start_path,
                artifact_loader=lambda _path: SimpleNamespace(
                    surface=warm_start_surface,
                    res={},
                ),
            )

        self.assertIs(seed.surface, warm_start_surface)
        self.assertAlmostEqual(seed.iota, 0.175)
        self.assertAlmostEqual(seed.G, -0.41)
        self.assertTrue(seed.has_solved_state)

    def test_load_warm_start_boozer_seed_raises_without_solved_state(self):
        # A Boozer-surface warm start with neither in-object .res state nor a
        # _boozer_state.json sidecar must raise, not silently return iota=None
        # (which downstream defaults into a single-stage solve that collapses).
        module = load_handoff_module()
        warm_start_surface = SimpleNamespace(nfp=5)

        with tempfile.TemporaryDirectory() as tmpdir:
            warm_start_path = Path(tmpdir) / "surf_opt_boozer_surface.json"
            warm_start_path.write_text("{}", encoding="utf-8")
            # No state sidecar written.

            with self.assertRaisesRegex(ValueError, "no solved"):
                module.load_warm_start_boozer_seed(
                    warm_start_path,
                    artifact_loader=lambda _path: SimpleNamespace(
                        surface=warm_start_surface,
                        res={},
                    ),
                )


class Stage2SolverIotaReportTests(unittest.TestCase):
    def test_iota_report_probe_uses_absolute_surface_as_equilibrium_path(self):
        solver = load_stage2_solver_module()
        handoff = load_handoff_module()
        plasma_surf_filename = "/tmp/hbt_stage1/wout_hbt_v2_abs.nc"
        stage2_bs_path = "/tmp/stage2/biot_savart_opt.json"
        stage2_results_path = "/tmp/stage2/results.json"
        stage2_seed_surf_path = "/tmp/stage2/surf_opt_boozer_surface.json"
        bootability_status = _bootability_status(
            handoff,
            stage=handoff.BOOTABILITY_STAGE_PROBE,
            reason=handoff.BOOTABILITY_REASON_OK,
            bootable=True,
            iota_feasible=True,
            solved_iota=0.2,
            self_intersecting=False,
        )
        args = SimpleNamespace(
            stage2_iota_target=0.2,
            stage2_iota_tolerance=5.0e-3,
            stage2_iota_constraint_weight=1.0,
            stage2_iota_num_tf_coils=20,
            stage2_iota_nphi=91,
            stage2_iota_ntheta=32,
            stage2_iota_mpol=8,
            stage2_iota_ntor=6,
            stage2_iota_vol_target=0.1,
            plasma_surf_filename=plasma_surf_filename,
            equilibria_dir="/tmp/database/equilibria",
            equilibrium_path=None,
            accept_offspec_tf_current_sign=False,
            accept_offspec_tf_current_magnitude=False,
            accept_offspec_banana_current_max=False,
        )

        with patch.object(
            solver,
            "probe_stage2_seed_bootability",
            return_value=bootability_status,
        ) as probe:
            payload = solver.build_stage2_iota_report_payload(
                args=args,
                stage2_bs_artifact_path=stage2_bs_path,
                stage2_results_payload={
                    "STAGE2_BS_PATH": stage2_bs_path,
                    "STAGE2_RESULTS_PATH": stage2_results_path,
                },
                stage2_seed_surf_path=stage2_seed_surf_path,
            )

        self.assertEqual(
            probe.call_args.kwargs["plasma_surf_filename"],
            Path(plasma_surf_filename).name,
        )
        self.assertEqual(
            probe.call_args.kwargs["equilibrium_path"],
            plasma_surf_filename,
        )
        self.assertEqual(
            probe.call_args.kwargs["stage2_seed_surf_path"],
            stage2_seed_surf_path,
        )
        self.assertEqual(payload["BOOTABILITY_STAGE2_SURF_PATH"], stage2_seed_surf_path)
        self.assertTrue(payload["STAGE2_ROOT_FIX_ENABLED"])
        self.assertEqual(payload["BOOTABILITY_REASON"], handoff.BOOTABILITY_REASON_OK)


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
                **_valid_stage2_contract_fields(),
                "PLASMA_SURF_FILENAME": "demo.nc",
                "init_only": False,
                "TF_CURRENT_A": -8.0e4,
                "NUM_TF_COILS": 20,
                "TF_CURRENT_SUM_ABS_A": 1.6e6,
            },
        )
        return stage2_bs_path, stage2_results_path

    def test_stage2_alm_artifact_reuse_requires_wout_metadata(self):
        module = load_stage2_alm_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            stage2_bs_path = root / "biot_savart_opt.json"
            stage2_results_path = root / "results.json"
            stage2_results = {
                "CONTRACT_SCHEMA_VERSION": 0,
                "CONTRACT_HASH": None,
                "PLASMA_SURF_PATH": str(root / "wout.nc"),
                "TF_CURRENT_A": -8.0e4,
            }

            with (
                patch.object(
                    module,
                    "resolve_stage2_artifact_path",
                    return_value=stage2_bs_path,
                ),
                patch.object(
                    module,
                    "load_stage2_artifact_results",
                    return_value=(stage2_results_path, stage2_results),
                ),
                patch.object(
                    module,
                    "upgrade_legacy_stage2_artifact_results",
                    side_effect=lambda payload: payload,
                ),
                patch.object(
                    module,
                    "_backfill_missing_stage2_alm_solver_metadata",
                    side_effect=lambda payload, _config: payload,
                ),
                patch.object(
                    module,
                    "_expected_stage2_artifact_metadata",
                    return_value={},
                ),
                patch.object(module, "validate_stage2_artifact_metadata"),
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "missing WOUT convention metadata",
                ):
                    module.load_validated_stage2_artifact(
                        SimpleNamespace(),
                        constraint_metadata={},
                    )

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

    def test_generated_stage2_stays_geometry_only_for_coil_seed_handoff(self):
        wrapper = load_wrapper_module()

        args = wrapper.parse_args(
            [
                "--plasma-surf-filename",
                "demo.nc",
                "--iota-target",
                "0.2",
                "--vol-target",
                "0.13",
            ]
        )
        stage2_args = wrapper.build_stage2_generation_args(
            args,
            output_root=Path("/tmp/stage2"),
        )

        self.assertEqual(stage2_args.constraint_method, "alm")
        self.assertIsNone(stage2_args.stage2_iota_target)

    def test_unified_handoff_forwards_alm_signal_mismatch_guard_to_generated_stage2(
        self,
    ):
        wrapper = load_wrapper_module()

        args = wrapper.parse_args(
            [
                "--plasma-surf-filename",
                "demo.nc",
                "--iota-target",
                "0.2",
                "--alm-fix-signal-mismatch-guard",
            ]
        )
        stage2_args = wrapper.build_stage2_generation_args(
            args,
            output_root=Path("/tmp/stage2"),
        )

        self.assertTrue(stage2_args.alm_fix_signal_mismatch_guard)

    def test_production_handoff_payload_requires_explicit_wout_status(self):
        wrapper = load_wrapper_module()
        handoff = load_handoff_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bootability_status = _bootability_status(
                handoff,
                stage=handoff.BOOTABILITY_STAGE_PROBE,
                reason=handoff.BOOTABILITY_REASON_OK,
                bootable=True,
                iota_feasible=True,
                solved_iota=0.2,
                self_intersecting=False,
            )

            with self.assertRaisesRegex(ValueError, "missing WOUT_OFF_SPEC"):
                wrapper.stage2_production_handoff_payload(
                    bootability_status,
                    source_stage2_results={},
                    original_stage2_bs_path=root / "biot_savart_opt.json",
                    original_stage2_results_path=root / "results.json",
                    recovery_attempted=False,
                    recovery_succeeded=False,
                    recovery_iters=None,
                    recovery_termination_reason=None,
                    seed_source=wrapper.SEED_SOURCE_DIRECT_STAGE2_DONOR,
                )

    def test_production_handoff_stamp_validates_before_writing_results(self):
        wrapper = load_wrapper_module()
        handoff = load_handoff_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            results_path = root / "results.json"
            original_results = {"ORIGINAL_RESULTS": True}
            _write_json(results_path, original_results)
            bootability_status = _bootability_status(
                handoff,
                stage=handoff.BOOTABILITY_STAGE_PROBE,
                reason=handoff.BOOTABILITY_REASON_OK,
                bootable=True,
                iota_feasible=True,
                solved_iota=0.2,
                self_intersecting=False,
            )

            with self.assertRaisesRegex(ValueError, "WOUT_OFF_SPEC=True"):
                wrapper.stamp_and_validate_stage2_production_handoff(
                    results_path,
                    bootability_status,
                    source_stage2_results={
                        "WOUT_OFF_SPEC": True,
                    },
                    original_stage2_bs_path=root / "biot_savart_opt.json",
                    original_stage2_results_path=results_path,
                    recovery_attempted=False,
                    recovery_succeeded=False,
                    recovery_iters=None,
                    recovery_termination_reason=None,
                    seed_source=wrapper.SEED_SOURCE_DIRECT_STAGE2_DONOR,
                )

            self.assertEqual(
                json.loads(results_path.read_text(encoding="utf-8")),
                original_results,
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
                    bootable=True,
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

    def test_build_probe_status_derives_equilibrium_path_from_absolute_surface(self):
        wrapper = load_wrapper_module()

        args = wrapper.parse_args(
            [
                "--plasma-surf-filename",
                "/tmp/stage1/wout_new_stage1.nc",
                "--stage2-bs-path",
                "/tmp/stage2/biot_savart_opt.json",
            ]
        )

        with patch.object(
            wrapper,
            "probe_stage2_seed_bootability",
            return_value={},
        ) as probe_mock:
            wrapper.build_probe_status(
                args,
                stage2_bs_path=Path("/tmp/stage2/biot_savart_opt.json"),
                stage2_results={
                    **_valid_stage2_contract_fields(),
                    "TF_CURRENT_A": -8.0e4,
                },
                stage="probe",
            )

        probe_kwargs = probe_mock.call_args.kwargs
        self.assertEqual(
            probe_kwargs["plasma_surf_filename"],
            "wout_new_stage1.nc",
        )
        self.assertEqual(
            probe_kwargs["equilibrium_path"],
            Path("/tmp/stage1/wout_new_stage1.nc").resolve(),
        )

    def test_load_stage2_seed_metadata_for_handoff_backfills_legacy_tf_current_from_cli(
        self,
    ):
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
                    "-12345.0",
                    "--num-tf-coils",
                    "18",
                ]
            )

            _, stage2_results = wrapper.load_stage2_seed_metadata_for_handoff(
                args,
                stage2_bs_path=stage2_bs_path,
            )

            self.assertEqual(stage2_results["TF_CURRENT_A"], -12345.0)
            self.assertEqual(stage2_results["NUM_TF_COILS"], 18)
            self.assertEqual(stage2_results["TF_CURRENT_SUM_ABS_A"], 222210.0)

    def test_build_probe_status_uses_exact_boozer_semantics_for_negative_constraint_weight(
        self,
    ):
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

        with patch.object(
            wrapper, "probe_stage2_seed_bootability", return_value={}
        ) as probe:
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

        with patch.object(
            wrapper, "probe_stage2_seed_bootability", return_value={}
        ) as probe:
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

        with patch.object(
            wrapper, "probe_stage2_seed_bootability", return_value={}
        ) as probe:
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
                "--iota-target",
                "0.2",
                "--vol-target",
                "0.13",
            ]
        )

        with patch.object(
            wrapper, "probe_stage2_seed_bootability", return_value={}
        ) as probe:
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

    def test_build_recovery_command_defaults_to_pre_boozer_stage2_repair(self):
        wrapper = load_wrapper_module()

        args = wrapper.parse_args(
            [
                "--plasma-surf-filename",
                "demo.nc",
                "--stage2-bs-path",
                "/tmp/stage2/biot_savart_opt.json",
                "--iota-target",
                "0.2",
                "--vol-target",
                "0.13",
            ]
        )

        command = wrapper.build_recovery_command(
            args,
            stage2_bs_path=Path("/tmp/stage2/biot_savart_opt.json"),
            recovery_output_root=Path("/tmp/recovery"),
            original_stage2_results={
                **_valid_stage2_contract_fields(),
                "TF_CURRENT_A": -8.0e4,
                "FINITE_CURRENT_MODE": "wataru_proxy_field",
            },
        )

        self.assertIn("banana_coil_solver.py", command[1])
        self.assertEqual(
            command[command.index("--stage2-bs-path") + 1],
            "/tmp/stage2/biot_savart_opt.json",
        )
        self.assertEqual(
            command[command.index("--constraint-method") + 1],
            "alm",
        )
        self.assertNotIn("--stage2-iota-mode", command)
        self.assertNotIn("--stage2-iota-target", command)
        self.assertNotIn("--stage2-iota-vol-target", command)
        self.assertEqual(
            command[command.index("--maxiter") + 1],
            str(args.recovery_maxiter),
        )

    def test_unified_handoff_forwards_alm_signal_mismatch_guard_to_pre_boozer_repair(
        self,
    ):
        wrapper = load_wrapper_module()

        args = wrapper.parse_args(
            [
                "--plasma-surf-filename",
                "demo.nc",
                "--stage2-bs-path",
                "/tmp/stage2/biot_savart_opt.json",
                "--alm-fix-signal-mismatch-guard",
            ]
        )

        command = wrapper.build_recovery_command(
            args,
            stage2_bs_path=Path("/tmp/stage2/biot_savart_opt.json"),
            recovery_output_root=Path("/tmp/recovery"),
            original_stage2_results={
                **_valid_stage2_contract_fields(),
                "TF_CURRENT_A": -8.0e4,
                "FINITE_CURRENT_MODE": "wataru_proxy_field",
            },
        )

        self.assertIn("--alm-fix-signal-mismatch-guard", command)

    def test_pre_boozer_stage2_repair_rejects_unsupported_finite_current_mode(self):
        wrapper = load_wrapper_module()

        args = wrapper.parse_args(
            [
                "--plasma-surf-filename",
                "demo.nc",
                "--stage2-bs-path",
                "/tmp/stage2/biot_savart_opt.json",
            ]
        )

        with self.assertRaisesRegex(
            ValueError,
            "FINITE_CURRENT_MODE='wataru_proxy_field'",
        ):
            wrapper.build_recovery_command(
                args,
                stage2_bs_path=Path("/tmp/stage2/biot_savart_opt.json"),
                recovery_output_root=Path("/tmp/recovery"),
                original_stage2_results={
                    **_valid_stage2_contract_fields(),
                    "TF_CURRENT_A": -8.0e4,
                    "FINITE_CURRENT_MODE": "boozer_surrogate",
                },
            )

    def test_thresholded_recovery_command_forwards_stage2_seed_surface_path(self):
        wrapper = load_wrapper_module()

        args = wrapper.parse_args(
            [
                "--plasma-surf-filename",
                "/tmp/stage1/wout_new_stage1.nc",
                "--stage2-bs-path",
                "/tmp/stage2/biot_savart_opt.json",
                "--recovery-stage",
                wrapper.RECOVERY_STAGE_THRESHOLDED_PHYSICS_ALM,
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
            command[command.index("--plasma-surf-filename") + 1],
            "wout_new_stage1.nc",
        )
        self.assertEqual(
            command[command.index("--equilibrium-path") + 1],
            str(Path("/tmp/stage1/wout_new_stage1.nc").resolve()),
        )
        self.assertEqual(
            command[command.index("--stage2-seed-role") + 1],
            "recovery",
        )
        self.assertEqual(
            command[command.index("--single-stage-banana-current-mode") + 1],
            "independent",
        )

    def test_unified_handoff_forwards_alm_signal_mismatch_guard_to_thresholded_recovery(
        self,
    ):
        wrapper = load_wrapper_module()

        args = wrapper.parse_args(
            [
                "--plasma-surf-filename",
                "/tmp/stage1/wout_new_stage1.nc",
                "--stage2-bs-path",
                "/tmp/stage2/biot_savart_opt.json",
                "--recovery-stage",
                wrapper.RECOVERY_STAGE_THRESHOLDED_PHYSICS_ALM,
                "--alm-fix-signal-mismatch-guard",
            ]
        )

        command = wrapper.build_recovery_command(
            args,
            stage2_bs_path=Path("/tmp/stage2/biot_savart_opt.json"),
            recovery_output_root=Path("/tmp/recovery"),
        )

        self.assertIn("--alm-fix-signal-mismatch-guard", command)

    def test_unified_handoff_forwards_alm_signal_mismatch_guard_to_full_single_stage(
        self,
    ):
        wrapper = load_wrapper_module()

        args = wrapper.parse_args(
            [
                "--plasma-surf-filename",
                "demo.nc",
                "--stage2-bs-path",
                "/tmp/stage2/biot_savart_opt.json",
                "--constraint-method",
                "alm",
                "--alm-fix-signal-mismatch-guard",
            ]
        )

        command = wrapper.build_full_single_stage_command(
            args,
            stage2_bs_path=Path("/tmp/stage2/biot_savart_opt.json"),
            full_output_root=Path("/tmp/full"),
        )

        self.assertIn("--alm-fix-signal-mismatch-guard", command)

    def test_recovery_only_nonbootable_reports_pre_boozer_repair_required(self):
        wrapper = load_wrapper_module()
        handoff = load_handoff_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            stage2_bs_path, _ = self._stage2_seed_paths(root)
            output_root = root / "outputs"
            summary_path = output_root / wrapper.DEFAULT_SUMMARY_JSON

            initial_probe = _bootability_status(
                handoff,
                stage=handoff.BOOTABILITY_STAGE_PROBE,
                reason=handoff.BOOTABILITY_REASON_SELF_INTERSECTION,
                bootable=False,
                iota_feasible=False,
                solved_iota=0.0003,
                self_intersecting=True,
            )
            recovery_payload = {
                "status": "completed",
                "recovery_succeeded": False,
                "recovery_iters": 3,
                "recovery_termination_reason": "not_bootable_after_budget",
            }

            with (
                patch.object(
                    wrapper,
                    "build_probe_status",
                    return_value=initial_probe,
                ),
                patch.object(
                    wrapper,
                    "run_recovery_stage",
                    return_value=recovery_payload,
                ) as recovery_mock,
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
            recovery_mock.assert_called_once()
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(
                summary["blocking_reason"],
                wrapper.BLOCKING_REASON_PRE_BOOZER_REPAIR_REQUIRED,
            )
            self.assertEqual(
                summary["next_required_lane"], wrapper.LANE_PRE_BOOZER_REPAIR
            )
            self.assertEqual(summary["recovery"], recovery_payload)
            self.assertIsNone(summary["full_single_stage"])

    def test_pre_boozer_recovery_runs_stage2_repair_and_probes_repaired_stage2_artifact(
        self,
    ):
        wrapper = load_wrapper_module()
        handoff = load_handoff_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            stage2_bs_path, stage2_results_path = self._stage2_seed_paths(root)
            recovery_output_root = root / "recovery"
            original_stage2_results = {
                **_valid_stage2_contract_fields(),
                "PLASMA_SURF_FILENAME": "demo.nc",
                "TF_CURRENT_A": -8.0e4,
                "NUM_TF_COILS": 20,
                "FINITE_CURRENT_MODE": "wataru_proxy_field",
            }

            def fake_run_command(command, *, timeout_seconds, dry_run):
                self.assertIn("banana_coil_solver.py", command[1])
                self.assertEqual(
                    command[command.index("--stage2-bs-path") + 1],
                    str(stage2_bs_path),
                )
                self.assertEqual(
                    command[command.index("--finite-current-mode") + 1],
                    "wataru_proxy_field",
                )
                self.assertNotIn("--stage2-iota-mode", command)
                self.assertNotIn("--stage2-iota-target", command)
                self.assertNotIn("--stage2-iota-vol-target", command)
                repaired_bs_path.parent.mkdir(parents=True, exist_ok=True)
                repaired_bs_path.write_text("{}", encoding="utf-8")
                _write_json(
                    repaired_results_path,
                    {
                        **original_stage2_results,
                        "iterations": 5,
                    },
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
                    "--iota-target",
                    "0.2",
                    "--vol-target",
                    "0.13",
                ]
            )
            repaired_bs_path = wrapper.resolve_stage2_artifact_path(
                wrapper.build_pre_boozer_stage2_repair_config(
                    args,
                    original_stage2_results=original_stage2_results,
                    recovery_output_root=recovery_output_root,
                )
            )
            repaired_results_path = repaired_bs_path.with_name("results.json")

            with (
                patch.object(
                    wrapper,
                    "run_command",
                    side_effect=fake_run_command,
                ),
                patch.object(
                    wrapper,
                    "build_probe_status",
                    side_effect=fake_build_probe_status,
                ),
            ):
                payload = wrapper.run_recovery_stage(
                    args,
                    original_stage2_bs_path=stage2_bs_path,
                    original_stage2_results_path=stage2_results_path,
                    original_stage2_results=original_stage2_results,
                    recovery_output_root=recovery_output_root,
                )

            self.assertEqual(payload["status"], "completed")
            self.assertTrue(payload["recovery_succeeded"])
            self.assertEqual(
                payload["result_source"], wrapper.RECOVERY_STAGE_PRE_BOOZER_STAGE2_ALM
            )
            self.assertEqual(payload["recovered_bs_path"], str(repaired_bs_path))
            self.assertIsNone(payload["warm_start_surface_stem"])
            self.assertEqual(payload["results"]["SEED_ROLE"], "coil_seed_handoff")
            self.assertFalse(payload["results"]["WOUT_OFF_SPEC"])
            self.assertFalse(payload["results"]["DIAGNOSTIC_ONLY"])
            self.assertTrue(payload["results"]["PRODUCTION_HANDOFF_READY"])
            self.assertIsNone(payload["results"]["HANDOFF_BLOCKING_GATE"])
            self.assertTrue(payload["results"]["PROMOTION_READY"])
            self.assertEqual(
                payload["results"]["UNIFIED_SEED_SOURCE"],
                wrapper.SEED_SOURCE_RECOVERED_STAGE2_DONOR,
            )
            self.assertTrue(payload["results"]["RECOVERY_SUCCEEDED"])
            self.assertEqual(len(captured_probe_calls), 1)
            probe_call = captured_probe_calls[0]
            self.assertEqual(probe_call["stage"], handoff.BOOTABILITY_STAGE_RECOVERY)
            self.assertEqual(probe_call["stage2_bs_path"], repaired_bs_path)
            self.assertEqual(probe_call["stage2_results"]["iterations"], 5)
            self.assertIsNone(probe_call["warm_start_boozer_surface_path"])

    def test_full_mode_runs_coil_seed_without_bootability_probe(self):
        wrapper = load_wrapper_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            stage2_bs_path, _ = self._stage2_seed_paths(root)
            output_root = root / "outputs"
            full_case_dir = output_root / "full" / "target" / "mpol=8-ntor=6-test"
            expected_handoff_bs_path = stage2_bs_path.resolve()

            def fake_full_run(
                args,
                *,
                stage2_bs_path,
                full_output_root,
                warm_start_surface_stem=None,
            ):
                full_case_dir.mkdir(parents=True, exist_ok=True)
                self.assertEqual(stage2_bs_path.resolve(), expected_handoff_bs_path)
                self.assertIsNone(warm_start_surface_stem)
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
                        (full_case_dir / "results.json").read_text(encoding="utf-8")
                    ),
                }

            with (
                patch.object(
                    wrapper,
                    "build_probe_status",
                ) as probe_mock,
                patch.object(
                    wrapper,
                    "run_recovery_stage",
                ) as recovery_mock,
                patch.object(
                    wrapper,
                    "run_full_single_stage",
                    side_effect=fake_full_run,
                ) as full_mock,
            ):
                self.assertEqual(
                    wrapper.main(
                        [
                            "--plasma-surf-filename",
                            "demo.nc",
                            "--stage2-bs-path",
                            str(stage2_bs_path),
                            "--output-root",
                            str(output_root),
                        ]
                    ),
                    0,
                )

            probe_mock.assert_not_called()
            recovery_mock.assert_not_called()
            full_mock.assert_called_once()
            summary = json.loads(
                (output_root / wrapper.DEFAULT_SUMMARY_JSON).read_text(encoding="utf-8")
            )
            self.assertIsNone(summary["bootability_probe"])
            self.assertEqual(summary["next_required_lane"], wrapper.LANE_PROMOTION)

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
                **_valid_stage2_contract_fields(),
                "PLASMA_SURF_FILENAME": "demo.nc",
                "TF_CURRENT_A": -8.0e4,
                "NUM_TF_COILS": 20,
                "FINITE_CURRENT_MODE": "boozer_surrogate",
            }

            def fake_recovery_run(command, *, output_root, timeout_seconds):
                recovery_case_dir.mkdir(parents=True, exist_ok=True)
                recovery_single_stage_results = {
                    "PLASMA_SURF_FILENAME": "demo.nc",
                    "init_only": False,
                    "iterations": 7,
                    # Single-stage schema: uses STAGE2_* prefix, does not surface
                    # TF_CURRENT_A / NUM_TF_COILS / FINITE_CURRENT_MODE directly.
                    "STAGE2_TF_CURRENT_A": -8.0e4,
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
                    "--recovery-stage",
                    wrapper.RECOVERY_STAGE_THRESHOLDED_PHYSICS_ALM,
                    "--output-root",
                    str(root / "outputs"),
                ]
            )

            with (
                patch.object(
                    wrapper,
                    "build_probe_status",
                    side_effect=fake_build_probe_status,
                ),
                patch.object(
                    wrapper,
                    "run_single_stage_command_with_salvage",
                    side_effect=fake_recovery_run,
                ),
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
            self.assertEqual(probe_call["stage2_results"]["TF_CURRENT_A"], -8.0e4)
            self.assertEqual(probe_call["stage2_results"]["NUM_TF_COILS"], 20)
            self.assertEqual(
                probe_call["stage2_results"]["FINITE_CURRENT_MODE"],
                "boozer_surrogate",
            )

    def test_run_recovery_stage_uses_preserved_artifact_bundle_for_salvaged_results(
        self,
    ):
        wrapper = load_wrapper_module()
        handoff = load_handoff_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            stage2_bs_path, stage2_results_path = self._stage2_seed_paths(root)
            recovery_output_root = root / "recovery"
            recovery_case_dir = recovery_output_root / "mpol=8-ntor=6-test"

            original_stage2_results = {
                **_valid_stage2_contract_fields(),
                "PLASMA_SURF_FILENAME": "demo.nc",
                "TF_CURRENT_A": -8.0e4,
                "NUM_TF_COILS": 20,
                "FINITE_CURRENT_MODE": "boozer_surrogate",
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
                (
                    recovery_case_dir / "surf_best_feasible_outer_boozer_surface.json"
                ).write_text(
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
                    "--recovery-stage",
                    wrapper.RECOVERY_STAGE_THRESHOLDED_PHYSICS_ALM,
                    "--output-root",
                    str(root / "outputs"),
                ]
            )

            with (
                patch.object(
                    wrapper,
                    "build_probe_status",
                    side_effect=fake_build_probe_status,
                ),
                patch.object(
                    wrapper,
                    "run_single_stage_command_with_salvage",
                    side_effect=fake_recovery_run,
                ),
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
