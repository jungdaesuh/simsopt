"""Checkpoint seed sidecar contract tests.

Covers the 2026-06-10 root fix for "the checkpoint is the answer, but
checkpoints are not loadable seeds": contract-only sidecar emission, the
materializer promotion flow, and acceptance by the REAL seed loaders
(``load_stage2_artifact_results`` checksum binding and the single-stage
resume loader/contract). A sidecar must never carry realized run metrics.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
import uuid
from pathlib import Path

import numpy as np

EXAMPLE_ROOT = (
    Path(__file__).resolve().parents[2] / "examples" / "single_stage_optimization"
)
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

from banana_opt.checkpoint_seed_sidecar import (  # noqa: E402
    CHECKPOINT_SEED_ARTIFACT_KIND,
    CHECKPOINT_SEED_SIDECAR_FILENAME,
    bind_results_to_bs,
    build_checkpoint_seed_sidecar,
    materialize_checkpoint_seed,
    write_checkpoint_seed_sidecar,
)
from banana_opt.artifact_contracts import (  # noqa: E402
    STAGE2_BS_SHA256_KEY,
    compute_stage2_bs_sha256,
)


def load_module(path: Path, stem: str):
    spec = importlib.util.spec_from_file_location(f"{stem}_{uuid.uuid4().hex}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_NUM_TF = 2
_BANANA_CURRENTS = (-15000.0, 15000.0)


def write_test_biot_savart(path: Path) -> None:
    """Serialize a minimal real BiotSavart graph: 2 TF coils at -80 kA plus 2
    banana-role coils with negative-base alternating currents (legacy
    partition inference: first NUM_TF_COILS are TF, the rest banana)."""
    from simsopt.field import BiotSavart, Coil, Current
    from simsopt.geo import CurveXYZFourier

    coils = []
    for index in range(_NUM_TF):
        curve = CurveXYZFourier(quadpoints=32, order=1)
        curve.set("xc(1)", 1.0)
        curve.set("ys(1)", 1.0)
        curve.set("zc(0)", float(index))
        coils.append(Coil(curve, Current(-80000.0)))
    for index, banana_current_A in enumerate(_BANANA_CURRENTS):
        curve = CurveXYZFourier(quadpoints=32, order=1)
        curve.set("xc(0)", 2.0 + index)
        curve.set("xc(1)", 0.3)
        curve.set("zs(1)", 0.3)
        coils.append(Coil(curve, Current(banana_current_A)))
    BiotSavart(coils).save(str(path))


def parent_results_payload() -> dict:
    """Seed-contract fields plus realized metrics that must NEVER leak into a
    checkpoint sidecar."""
    return {
        "MAJOR_RADIUS": 0.976,
        "TOROIDAL_FLUX": 0.24,
        "banana_surf_radius": 0.21,
        "order": 4,
        "mpol": 10,
        "ntor": 10,
        "TF_CURRENT_A": -80000.0,
        "NUM_TF_COILS": _NUM_TF,
        "FINITE_CURRENT_MODE": "wataru_proxy_field",
        "FLIP_BANANA": False,
        "CURVATURE_THRESHOLD": 32.6012,
        "PLASMA_SURF_PATH": "/tmp/wout_signed_cw.nc",
        "OFFSPEC_REPLAY_DEBUG_ONLY": True,
        # Realized metrics of the PARENT endpoint (true only for the endpoint,
        # not for any checkpoint) — the builder must not copy these.
        "MAX_CURVATURE": 43.477,
        "FINAL_IOTA": 0.0987,
        "FINAL_VOLUME": 0.110,
        "J": 0.18,
        "COIL_LENGTH": 1.6357,
    }


class CheckpointSeedSidecarBuilderTest(unittest.TestCase):
    def setUp(self):
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.parent_out = self.tmp / "mpol=10-ntor=10-deadbeef"
        self.ckpt_dir = self.parent_out / "checkpoint_iter0003"
        self.ckpt_dir.mkdir(parents=True)
        write_test_biot_savart(self.ckpt_dir / "biot_savart.json")
        (self.parent_out / "results.json").write_text(
            json.dumps(parent_results_payload()), encoding="utf-8"
        )

    def tearDown(self):
        self._tmp.cleanup()

    def test_builder_copies_contract_and_never_metrics(self):
        sidecar = build_checkpoint_seed_sidecar(
            parent_results_payload(),
            checkpoint_bs_path=self.ckpt_dir / "biot_savart.json",
            checkpoint_iteration=3,
            parent_out_dir=self.parent_out,
        )
        for contract_key in (
            "MAJOR_RADIUS",
            "TOROIDAL_FLUX",
            "banana_surf_radius",
            "order",
            "TF_CURRENT_A",
            "NUM_TF_COILS",
            "FINITE_CURRENT_MODE",
            "CURVATURE_THRESHOLD",
            "PLASMA_SURF_PATH",
            "OFFSPEC_REPLAY_DEBUG_ONLY",
        ):
            self.assertIn(
                contract_key,
                sidecar,
                f"seed-contract key {contract_key} must be copied from the parent",
            )
        for metric_key in ("MAX_CURVATURE", "FINAL_IOTA", "FINAL_VOLUME", "J", "COIL_LENGTH"):
            self.assertNotIn(
                metric_key,
                sidecar,
                f"realized metric {metric_key} leaked into a contract-only sidecar "
                "(fabrication: it was never measured for this checkpoint)",
            )
        self.assertEqual(sidecar["ARTIFACT_KIND"], CHECKPOINT_SEED_ARTIFACT_KIND)
        self.assertEqual(sidecar["CHECKPOINT_ITERATION"], 3)
        self.assertEqual(
            sidecar[STAGE2_BS_SHA256_KEY],
            compute_stage2_bs_sha256(self.ckpt_dir / "biot_savart.json"),
            "sidecar must bind by content to the checkpoint's own artifact",
        )

    def test_builder_requires_hard_keys(self):
        payload = parent_results_payload()
        del payload["banana_surf_radius"]
        with self.assertRaisesRegex(
            ValueError, "banana_surf_radius.*resolve_single_stage_banana_surf_radius"
        ):
            build_checkpoint_seed_sidecar(
                payload,
                checkpoint_bs_path=self.ckpt_dir / "biot_savart.json",
                checkpoint_iteration=3,
                parent_out_dir=self.parent_out,
            )

    def test_builder_rejects_runtime_sha_override(self):
        with self.assertRaisesRegex(ValueError, "realized-only"):
            build_checkpoint_seed_sidecar(
                parent_results_payload(),
                checkpoint_bs_path=self.ckpt_dir / "biot_savart.json",
                checkpoint_iteration=3,
                parent_out_dir=self.parent_out,
                runtime_fields={STAGE2_BS_SHA256_KEY: "forged"},
            )

    def test_builder_rejects_orphan_iota_target_sign(self):
        payload = parent_results_payload()
        del payload["FLIP_BANANA"]
        payload["IOTA_TARGET_SIGN"] = 1
        with self.assertRaisesRegex(ValueError, "IOTA_TARGET_SIGN without FLIP_BANANA"):
            build_checkpoint_seed_sidecar(
                payload,
                checkpoint_bs_path=self.ckpt_dir / "biot_savart.json",
                checkpoint_iteration=3,
                parent_out_dir=self.parent_out,
            )

    def test_bind_results_to_bs_stamps_content_digest(self):
        results: dict = {"unrelated": 1}
        bs_path = self.ckpt_dir / "biot_savart.json"
        bind_results_to_bs(results, bs_path)
        self.assertEqual(
            results[STAGE2_BS_SHA256_KEY], compute_stage2_bs_sha256(bs_path)
        )


class MaterializeCheckpointSeedTest(unittest.TestCase):
    def setUp(self):
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.parent_out = self.tmp / "mpol=10-ntor=10-deadbeef"
        self.ckpt_dir = self.parent_out / "checkpoint_iter0003"
        self.ckpt_dir.mkdir(parents=True)
        write_test_biot_savart(self.ckpt_dir / "biot_savart.json")
        (self.parent_out / "results.json").write_text(
            json.dumps(parent_results_payload()), encoding="utf-8"
        )
        self.seed_dir = self.tmp / "seed_out"

    def tearDown(self):
        self._tmp.cleanup()

    def test_emitted_sidecar_promotion_satisfies_real_loaders(self):
        write_checkpoint_seed_sidecar(
            self.ckpt_dir,
            parent_results_payload(),
            checkpoint_iteration=3,
            parent_out_dir=self.parent_out,
        )
        results_path = materialize_checkpoint_seed(self.ckpt_dir, self.seed_dir)
        seed_bs_path = self.seed_dir / "biot_savart_opt.json"
        self.assertTrue(seed_bs_path.is_file())

        # REAL stage2-seed loader: existence + checksum binding.
        workflow_common = load_module(
            EXAMPLE_ROOT / "workflow_runner_common.py", "workflow_runner_common"
        )
        loaded_path, loaded = workflow_common.load_stage2_artifact_results(seed_bs_path)
        self.assertEqual(loaded_path, results_path)

        # REAL single-stage resume loader + contract.
        single_stage = load_module(
            EXAMPLE_ROOT / "SINGLE_STAGE" / "single_stage_banana_example.py",
            "single_stage_banana_example",
        )
        _, resume_results = single_stage.load_single_stage_resume_seed_results(
            seed_bs_path
        )
        single_stage.validate_single_stage_resume_seed_contract(
            resume_results, accept_offspec_r0_seed=False
        )

    def test_historical_checkpoint_derives_from_parent_results(self):
        # No emitted sidecar: the materializer must derive the contract from
        # the parent run's results.json plus realized currents from the graph.
        results_path = materialize_checkpoint_seed(self.ckpt_dir, self.seed_dir)
        sidecar = json.loads(results_path.read_text(encoding="utf-8"))
        np.testing.assert_allclose(
            sidecar["BANANA_CURRENTS_A"], list(_BANANA_CURRENTS), rtol=0, atol=1e-9
        )
        self.assertEqual(sidecar["NUM_TF_COILS"], _NUM_TF)
        self.assertEqual(sidecar["CHECKPOINT_ITERATION"], 3)
        self.assertEqual(
            sidecar["CHECKPOINT_SEED_MATERIALIZED_FROM_PARENT_RESULTS"],
            str(self.parent_out / "results.json"),
        )
        for metric_key in ("MAX_CURVATURE", "FINAL_IOTA", "J"):
            self.assertNotIn(metric_key, sidecar)
        self.assertEqual(
            sidecar[STAGE2_BS_SHA256_KEY],
            compute_stage2_bs_sha256(self.seed_dir / "biot_savart_opt.json"),
        )

    def test_materialize_refuses_overwrite(self):
        materialize_checkpoint_seed(self.ckpt_dir, self.seed_dir)
        with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
            materialize_checkpoint_seed(self.ckpt_dir, self.seed_dir)

    def test_materialize_detects_tampered_checkpoint(self):
        sidecar_path = write_checkpoint_seed_sidecar(
            self.ckpt_dir,
            parent_results_payload(),
            checkpoint_iteration=3,
            parent_out_dir=self.parent_out,
        )
        self.assertEqual(sidecar_path.name, CHECKPOINT_SEED_SIDECAR_FILENAME)
        bs_path = self.ckpt_dir / "biot_savart.json"
        bs_path.write_text(bs_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            materialize_checkpoint_seed(self.ckpt_dir, self.seed_dir)


if __name__ == "__main__":
    unittest.main()
