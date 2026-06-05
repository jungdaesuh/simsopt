import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from simsopt.field import BiotSavart, Coil, Current
from simsopt.geo import CurveXYZFourier


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = REPO_ROOT / "examples" / "single_stage_optimization"
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

from banana_opt.design_only_fields import (  # noqa: E402
    DESIGN_ONLY_RESULTS_KEY,
    DesignOnlyTopologyFieldError,
    assert_topology_field_allowed,
    build_design_only_results_fields,
    field_is_design_only,
    mark_design_only_field,
)
from banana_opt.finite_current_profiles import (  # noqa: E402
    JHALPERN30_FINITE_CURRENT_MODE,
    VACUUM_FINITE_CURRENT_MODE,
    WATARU_FINITE_CURRENT_MODE,
)
from banana_opt.jhalpern30_compat import (  # noqa: E402
    build_jhalpern30_proxy_plasma_current_coils,
)
from banana_opt.json_compat import load_boozer_finite_i  # noqa: E402
from banana_opt.stage2_geometry import build_proxy_plasma_current_coils  # noqa: E402
from POINCARE_PLOTTING.poincare_surfaces import (  # noqa: E402
    build_poincare_aggregate_artifact,
    build_poincare_mode_artifact,
    design_only_override_enabled,
    load_design_only_results_metadata,
)


SIGNED_CW_WOUT_PATH = (
    Path(__file__).resolve().parents[1] / "test_files" / "wout_10x10.nc"
)


class _TraceDomain:
    def as_metadata(self) -> dict[str, object]:
        return {"domain": "test"}


def _biot_savart() -> BiotSavart:
    curve = CurveXYZFourier(16, 1)
    curve.set("xc(1)", 0.1)
    curve.set("ys(1)", 0.1)
    curve.fix_all()
    current = Current(1.0)
    current.fix_all()
    return BiotSavart([Coil(curve, current)])


class DesignOnlyFieldTests(unittest.TestCase):
    def test_wataru_design_only_marker_survives_through_results_sidecar(self):
        bs = _biot_savart()
        reason = f"finite_current_proxy_line_current: {WATARU_FINITE_CURRENT_MODE}"
        mark_design_only_field(bs, reason=reason)

        self.assertTrue(field_is_design_only(bs, None))

        with tempfile.TemporaryDirectory() as temp_dir:
            bs_path = Path(temp_dir) / "biot_savart_opt.json"
            results_path = Path(temp_dir) / "results.json"
            bs.save(str(bs_path))
            results_path.write_text(
                json.dumps(build_design_only_results_fields(reason=reason)),
                encoding="utf-8",
            )

            reloaded = load_boozer_finite_i(str(bs_path))
            results = json.loads(results_path.read_text(encoding="utf-8"))

        self.assertFalse(field_is_design_only(reloaded, None))
        with self.assertRaises(DesignOnlyTopologyFieldError):
            assert_topology_field_allowed(
                reloaded,
                results,
                allow_design_only_field=False,
                consumer="test_wataru",
            )

    def test_jhalpern_design_only_marker_uses_surface_major_radius_proxy(self):
        proxy_coils = build_jhalpern30_proxy_plasma_current_coils(
            SimpleNamespace(major_radius=lambda: 0.976),
            -6.5e3,
        )
        bs = BiotSavart(proxy_coils)
        reason = f"finite_current_proxy_line_current: {JHALPERN30_FINITE_CURRENT_MODE}"
        mark_design_only_field(bs, reason=reason)
        results = build_design_only_results_fields(reason=reason)

        self.assertEqual(len(proxy_coils), 1)
        self.assertEqual(proxy_coils[0].curve.get("xc(1)"), 0.976)
        self.assertEqual(proxy_coils[0].curve.get("ys(1)"), 0.976)
        with self.assertRaises(DesignOnlyTopologyFieldError):
            assert_topology_field_allowed(
                bs,
                results,
                allow_design_only_field=False,
                consumer="test_jhalpern",
            )

    def test_vacuum_mode_has_no_design_only_marker(self):
        bs = _biot_savart()
        results = {
            "FINITE_CURRENT_MODE": VACUUM_FINITE_CURRENT_MODE,
            "NUM_PROXY_COILS": 0,
        }

        self.assertFalse(field_is_design_only(bs, results))
        assert_topology_field_allowed(
            bs,
            results,
            allow_design_only_field=False,
            consumer="test_vacuum",
        )

    def test_poincare_override_allows_design_only_field_and_builds_artifact_flag(self):
        bs = _biot_savart()
        reason = f"finite_current_proxy_line_current: {WATARU_FINITE_CURRENT_MODE}"
        results = build_design_only_results_fields(reason=reason)

        self.assertTrue(
            design_only_override_enabled({"POINCARE_ALLOW_DESIGN_ONLY_FIELD": "true"})
        )
        assert_topology_field_allowed(
            bs,
            results,
            allow_design_only_field=True,
            consumer="poincare_surfaces",
        )
        render_modes = [
            {
                "mode": "validation",
                "seed_contract": "validation",
                "trace_domain": _TraceDomain(),
                "stop_labels": ("none",),
                "trace_semantics": "validation",
            },
            {
                "mode": "diagnostic",
                "seed_contract": "diagnostic",
                "trace_domain": _TraceDomain(),
                "stop_labels": ("none",),
                "trace_semantics": "diagnostic",
            },
            {
                "mode": "default",
                "seed_contract": "default",
                "trace_domain": _TraceDomain(),
                "stop_labels": ("none",),
                "trace_semantics": "default",
            },
        ]
        metrics = {
            "plot_filename": "PoincarePlot_opt.png",
            "validation_status": "diagnostic_only",
        }
        override = design_only_override_enabled(
            {"POINCARE_ALLOW_DESIGN_ONLY_FIELD": "true"}
        )
        mode_artifact = build_poincare_mode_artifact(
            field_label="opt",
            render_mode=render_modes[0],
            nfieldlines=4,
            tmax=1000,
            tol=1.0e-9,
            phis=(0.0,),
            field_model="BiotSavart",
            metrics=metrics,
            design_only_override=override,
        )
        aggregate_artifact = build_poincare_aggregate_artifact(
            field_label="opt",
            nfieldlines=4,
            tmax=1000,
            tol=1.0e-9,
            phis=(0.0,),
            render_modes=render_modes,
            field_trace_domain=_TraceDomain(),
            field_models_by_mode={
                "validation": "BiotSavart",
                "diagnostic": "BiotSavart",
                "default": "BiotSavart",
            },
            metrics_by_mode={
                "validation": metrics,
                "diagnostic": metrics,
                "default": metrics,
            },
            design_only_override=override,
        )
        self.assertIs(mode_artifact["design_only_override"], True)
        self.assertIs(aggregate_artifact["design_only_override"], True)

    def test_results_metadata_loader_reads_design_only_sidecar(self):
        reason = f"finite_current_proxy_line_current: {WATARU_FINITE_CURRENT_MODE}"
        with tempfile.TemporaryDirectory() as temp_dir:
            Path(temp_dir, "results.json").write_text(
                json.dumps(build_design_only_results_fields(reason=reason)),
                encoding="utf-8",
            )

            metadata = load_design_only_results_metadata(temp_dir)

        self.assertIsNotNone(metadata)
        self.assertIs(metadata[DESIGN_ONLY_RESULTS_KEY], True)

    def test_wataru_proxy_wrapper_preserves_axis_zeroth_placement(self):
        proxy_coils = build_proxy_plasma_current_coils(
            equilibrium_file=str(SIGNED_CW_WOUT_PATH),
            surface_scale_factor=1.0,
            nphi=16,
            ntheta=8,
            toroidal_flux=0.5,
            plasma_current_A=800.0,
        )

        self.assertEqual(len(proxy_coils), 1)
        self.assertEqual(proxy_coils[0].current.get_value(), 800.0)
        self.assertNotEqual(proxy_coils[0].curve.get("xc(1)"), 0.0)


if __name__ == "__main__":
    unittest.main()
