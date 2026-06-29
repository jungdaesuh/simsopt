"""Warm-start donor contract for high-resolution single-stage parity runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from benchmarks.single_stage_init_parity import (
    _case_timing_breakdown,
    _load_single_stage_case_from_output_root,
    _needs_shared_init_seed,
    _run_single_stage_case_pair,
    _resolve_warm_start_seed_contract_G,
    _single_stage_full_run_lane_contract,
    _single_stage_pair_timing_breakdown,
    _single_stage_elapsed_s_and_source_from_results,
    _single_stage_elapsed_s_from_results,
    _validate_warm_start_seed_contract,
)
from examples.single_stage_optimization.SINGLE_STAGE import (
    single_stage_banana_example as single_stage_example,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = REPO_ROOT / "benchmarks" / "fixtures" / "single_stage_seed_iota15"


class CurveXYZFourier:
    pass


class CurveCWSFourierCPP:
    pass


class RotatedCurve:
    def __init__(self, curve):
        self.curve = curve


class _Coil:
    def __init__(self, curve):
        self.curve = curve


def _coils(curve_cls, count):
    return [_Coil(curve_cls()) for _ in range(count)]


def _array_payload(values):
    return {
        "@module": "numpy",
        "@class": "array",
        "dtype": "float64",
        "data": values,
    }


def test_warm_start_g_uses_signed_boozer_state_sidecar(tmp_path):
    surface_path = tmp_path / "surf_opt_boozer_surface.json"
    surface_path.write_text("{}", encoding="utf-8")
    (tmp_path / "surf_opt_boozer_state.json").write_text(
        json.dumps({"G": -2.01061560706768}),
        encoding="utf-8",
    )

    resolved_g = single_stage_example.resolve_single_stage_warm_start_G(
        {"FINAL_IOTA": 0.11015671329581699},
        surface_path,
    )

    assert resolved_g == pytest.approx(-2.01061560706768)


def test_warm_start_g_stays_missing_without_results_or_state(tmp_path):
    surface_path = tmp_path / "surf_opt_boozer_surface.json"
    surface_path.write_text("{}", encoding="utf-8")

    resolved_g = single_stage_example.resolve_single_stage_warm_start_G(
        {"FINAL_IOTA": 0.11015671329581699},
        surface_path,
    )

    assert resolved_g is None


def test_warm_start_contract_accepts_signed_g_from_boozer_state_sidecar(tmp_path):
    """Donors without ``FINAL_G`` remain valid when the Boozer state sidecar has G."""
    seed_dir = tmp_path / "seed"
    seed_dir.mkdir()
    results = {
        "init_only": False,
        "FINAL_IOTA": 0.22650585872006085,
        "HARDWARE_CONSTRAINTS_OK": True,
        "SELF_INTERSECTING": False,
    }
    (seed_dir / "results.json").write_text(json.dumps(results), encoding="utf-8")
    (seed_dir / "surf_opt_boozer_surface.json").write_text("{}", encoding="utf-8")
    (seed_dir / "surf_opt_boozer_state.json").write_text(
        json.dumps({"G": -2.01061560706768}),
        encoding="utf-8",
    )
    args = argparse.Namespace(iota_target=0.22650585872006085, num_tf_coils=20)

    resolved_g = _resolve_warm_start_seed_contract_G(args, seed_dir, results)

    assert resolved_g == pytest.approx(-2.01061560706768)
    _validate_warm_start_seed_contract(args, seed_dir)


def test_runtime_seed_contract_rejects_flattened_materialized_cws_banana_field():
    donor_results = {"SINGLE_STAGE_BANANA_GEOMETRY_MODE": "materialized_cws"}
    banana_coils = _coils(CurveXYZFourier, 10)

    with pytest.raises(ValueError, match="0 referenced CurveCWSFourierCPP"):
        single_stage_example.validate_materialized_cws_runtime_seed_coil_contract(
            donor_results,
            banana_coils,
            biot_savart_path="/seed/biot_savart_opt.json",
        )


def test_runtime_seed_contract_rejects_flattened_cws_banana_curve_class():
    donor_results = {"banana_curve_class": "CurveCWSFourierCPP"}
    banana_coils = _coils(CurveXYZFourier, 10)

    with pytest.raises(ValueError, match="0 referenced CurveCWSFourierCPP"):
        single_stage_example.validate_materialized_cws_runtime_seed_coil_contract(
            donor_results,
            banana_coils,
            biot_savart_path="/seed/biot_savart_opt.json",
        )


def test_runtime_seed_contract_accepts_materialized_cws_banana_field():
    donor_results = {"SINGLE_STAGE_BANANA_GEOMETRY_MODE": "materialized_cws"}
    banana_coils = [_Coil(RotatedCurve(CurveCWSFourierCPP())) for _ in range(10)]

    inventory = single_stage_example.validate_materialized_cws_runtime_seed_coil_contract(
        donor_results,
        banana_coils,
        biot_savart_path="/seed/slid_cws_field_chomp.json",
    )

    assert inventory["RotatedCurve"] == 10
    assert inventory["CurveCWSFourierCPP"] == 10


def test_runtime_seed_contract_rejects_partially_flattened_materialized_cws_field():
    donor_results = {"SINGLE_STAGE_BANANA_GEOMETRY_MODE": "materialized_cws"}
    banana_coils = [
        *[_Coil(RotatedCurve(CurveCWSFourierCPP())) for _ in range(9)],
        _Coil(CurveXYZFourier()),
    ]

    with pytest.raises(ValueError, match="9/10 active banana coils"):
        single_stage_example.validate_materialized_cws_runtime_seed_coil_contract(
            donor_results,
            banana_coils,
            biot_savart_path="/seed/biot_savart_opt.json",
        )


def test_runtime_seed_contract_allows_flattened_free_xyz_banana_field():
    donor_results = {"SINGLE_STAGE_BANANA_GEOMETRY_MODE": "free_xyz"}
    banana_coils = _coils(CurveXYZFourier, 10)

    inventory = single_stage_example.validate_materialized_cws_runtime_seed_coil_contract(
        donor_results,
        banana_coils,
        biot_savart_path="/seed/biot_savart_opt.json",
    )

    assert inventory == {"CurveXYZFourier": 10}


def test_runtime_seed_contract_free_xyz_overrides_legacy_cws_curve_class():
    donor_results = {
        "SINGLE_STAGE_BANANA_GEOMETRY_MODE": "free_xyz",
        "banana_curve_class": "CurveCWSFourierCPP",
    }
    banana_coils = _coils(CurveXYZFourier, 10)

    inventory = single_stage_example.validate_materialized_cws_runtime_seed_coil_contract(
        donor_results,
        banana_coils,
        biot_savart_path="/seed/biot_savart_opt.json",
    )

    assert inventory == {"CurveXYZFourier": 10}


def test_boozer_surface_sidecar_warm_start_loads_underlying_surface(tmp_path):
    seed_dir = tmp_path / "seed"
    seed_dir.mkdir()
    (seed_dir / "results.json").write_text(
        json.dumps({"FINAL_IOTA": 0.11}),
        encoding="utf-8",
    )
    (seed_dir / "surf_opt_boozer_state.json").write_text(
        json.dumps({"G": -2.01061560706768}),
        encoding="utf-8",
    )
    (seed_dir / "surf_opt_boozer_surface.json").write_text(
        json.dumps(
            {
                "@class": "SIMSON",
                "graph": [{"$type": "ref", "value": "BoozerSurface1"}],
                "simsopt_objs": {
                    "BoozerSurface1": {
                        "@class": "BoozerSurface",
                        "surface": {
                            "$type": "ref",
                            "value": "SurfaceXYZTensorFourier1",
                        },
                    },
                    "SurfaceXYZTensorFourier1": {
                        "@class": "SurfaceXYZTensorFourier",
                        "mpol": 1,
                        "ntor": 1,
                        "nfp": 5,
                        "stellsym": True,
                        "quadpoints_phi": _array_payload([0.0, 0.1]),
                        "quadpoints_theta": _array_payload([0.0, 0.2, 0.4]),
                        "dofs": {"$type": "ref", "value": "SurfaceDofs1"},
                    },
                    "SurfaceDofs1": {
                        "x": _array_payload([0.0, 1.0, 2.0, 3.0]),
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    warm_start_state = single_stage_example.load_single_stage_warm_start_state(
        seed_dir
    )

    assert warm_start_state["surface"].surface_class == "SurfaceXYZTensorFourier"
    assert warm_start_state["surface"].mpol == 1
    assert warm_start_state["G"] == pytest.approx(-2.01061560706768)
    assert warm_start_state["surface_path"].endswith(
        "surf_opt_boozer_surface.json"
    )


def test_warm_start_prefers_surface_artifact_over_boozer_surface_sidecar(tmp_path):
    seed_dir = tmp_path / "seed"
    seed_dir.mkdir()
    (seed_dir / "results.json").write_text(
        json.dumps({"FINAL_IOTA": 0.11, "FINAL_G": 1.25}),
        encoding="utf-8",
    )
    (seed_dir / "surf_opt.json").write_text(
        json.dumps(
            {
                "@class": "SIMSON",
                "graph": [{"$type": "ref", "value": "SurfaceXYZTensorFourier1"}],
                "simsopt_objs": {
                    "SurfaceXYZTensorFourier1": {
                        "@class": "SurfaceXYZTensorFourier",
                        "mpol": 2,
                        "ntor": 2,
                        "nfp": 5,
                        "stellsym": True,
                        "quadpoints_phi": _array_payload([0.0, 0.1]),
                        "quadpoints_theta": _array_payload([0.0, 0.2, 0.4]),
                        "dofs": {"$type": "ref", "value": "SurfaceDofs1"},
                    },
                    "SurfaceDofs1": {
                        "x": _array_payload([0.0, 1.0, 2.0, 3.0]),
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (seed_dir / "surf_opt_boozer_surface.json").write_text(
        json.dumps(
            {
                "@class": "SIMSON",
                "graph": [{"$type": "ref", "value": "BoozerSurface1"}],
                "simsopt_objs": {
                    "BoozerSurface1": {
                        "@class": "BoozerSurface",
                        "surface": {
                            "$type": "ref",
                            "value": "SurfaceXYZTensorFourier1",
                        },
                    },
                    "SurfaceXYZTensorFourier1": {
                        "@class": "SurfaceXYZTensorFourier",
                        "mpol": 1,
                        "ntor": 1,
                        "nfp": 5,
                        "stellsym": True,
                        "quadpoints_phi": _array_payload([0.0, 0.1]),
                        "quadpoints_theta": _array_payload([0.0, 0.2, 0.4]),
                        "dofs": {"$type": "ref", "value": "SurfaceDofs1"},
                    },
                    "SurfaceDofs1": {
                        "x": _array_payload([0.0, 1.0, 2.0, 3.0]),
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    warm_start_state = single_stage_example.load_single_stage_warm_start_state(
        seed_dir
    )

    assert warm_start_state["surface"].mpol == 2
    assert warm_start_state["surface_path"].endswith("surf_opt.json")


def test_cpu_reference_outer_run_uses_explicit_warm_start_without_seed_child():
    """Production donor runs should not re-solve an init-only shared seed."""
    args = argparse.Namespace(
        maxiter=1500,
        warm_start_run_dir="/workspace/seeds/donor",
        jax_runtime_seed_spec=None,
    )

    assert not _needs_shared_init_seed(args, reference_backend="cpu")


def test_cpu_reference_outer_run_keeps_seed_child_without_explicit_seed():
    """Cold outer runs still materialize a shared seed before comparing lanes."""
    args = argparse.Namespace(
        maxiter=1500,
        warm_start_run_dir=None,
        jax_runtime_seed_spec=None,
    )

    assert _needs_shared_init_seed(args, reference_backend="cpu")


def test_loaded_reference_elapsed_uses_script_total_timing():
    results = {"TIMINGS": {"script_total_s": 12.5, "other": "ignored"}}

    assert _single_stage_elapsed_s_from_results(results) == 12.5
    assert _single_stage_elapsed_s_and_source_from_results(results) == (
        12.5,
        "results_TIMINGS_script_total_s",
    )


def test_loaded_reference_elapsed_is_unknown_without_timing_fields():
    results = {"TIMINGS": {"other": "ignored"}}

    assert _single_stage_elapsed_s_from_results(results) is None
    assert _single_stage_elapsed_s_and_source_from_results(results) == (None, None)


def test_loaded_reference_elapsed_source_prefers_elapsed_seconds():
    results = {"ELAPSED_SECONDS": 13.5, "TIMINGS": {"script_total_s": 12.5}}

    assert _single_stage_elapsed_s_and_source_from_results(results) == (
        13.5,
        "results_ELAPSED_SECONDS",
    )


def test_loaded_reference_elapsed_falls_back_to_progress_events(monkeypatch, tmp_path):
    run_dir = tmp_path / "loaded_reference_run"
    run_dir.mkdir()
    progress_json = run_dir / "outer_optimizer_progress.json"
    progress_json.write_text(
        json.dumps(
            {
                "events": [
                    {"label": "phase2_started", "event_elapsed_s": 1.5},
                    {"label": "phase2_returned", "event_elapsed_s": 7.25},
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "benchmarks.single_stage_init_parity._load_single_stage_final_payload",
        lambda output_root: (
            {"backend": "cpu", "provenance": {}},
            run_dir / "results.json",
        ),
    )

    case = _load_single_stage_case_from_output_root(
        tmp_path / "cpu_outputs",
        argparse.Namespace(),
        backend="cpu",
        load_surface_gamma=False,
    )

    assert case["elapsed_s"] == pytest.approx(7.25)
    assert case["elapsed_source"] == "outer_optimizer_progress_event_elapsed_s"
    assert case["artifact_source"] == "external_reference_artifact"
    assert case["outer_optimizer_progress_json"] == str(progress_json)


def test_loaded_reference_requires_progress_trace(monkeypatch, tmp_path):
    run_dir = tmp_path / "loaded_reference_run"
    run_dir.mkdir()
    monkeypatch.setattr(
        "benchmarks.single_stage_init_parity._load_single_stage_final_payload",
        lambda output_root: (
            {"backend": "cpu", "provenance": {}},
            run_dir / "results.json",
        ),
    )

    with pytest.raises(FileNotFoundError, match="outer_optimizer_progress.json"):
        _load_single_stage_case_from_output_root(
            tmp_path / "cpu_outputs",
            argparse.Namespace(),
            backend="cpu",
            load_surface_gamma=False,
        )


def test_loaded_reference_rejects_backend_mismatch(monkeypatch, tmp_path):
    run_dir = tmp_path / "loaded_reference_run"
    run_dir.mkdir()
    (run_dir / "outer_optimizer_progress.json").write_text(
        '{"events": []}',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "benchmarks.single_stage_init_parity._load_single_stage_final_payload",
        lambda output_root: (
            {"backend": "jax", "provenance": {}},
            run_dir / "results.json",
        ),
    )

    with pytest.raises(ValueError, match="backend mismatch"):
        _load_single_stage_case_from_output_root(
            tmp_path / "cpu_outputs",
            argparse.Namespace(),
            backend="cpu",
            load_surface_gamma=False,
        )


def test_external_reference_contract_does_not_stamp_current_seed_hash(tmp_path):
    run_dir = tmp_path / "loaded_reference_run"
    run_dir.mkdir()
    progress_json = run_dir / "outer_optimizer_progress.json"
    progress_json.write_text('{"events": []}', encoding="utf-8")
    case = {
        "results": {"provenance": {}, "init_only": False},
        "run_dir": str(run_dir),
        "final_artifact_json": str(run_dir / "results.json"),
        "final_artifact_accepted": True,
        "outer_optimizer_progress_json": str(progress_json),
        "artifact_source": "external_reference_artifact",
        "artifact_source_root": str(tmp_path / "cpu_outputs"),
    }

    contract = _single_stage_full_run_lane_contract(
        case,
        runtime_seed_spec_hash="current-seed-hash",
        run_family_id="current-run-family",
    )

    assert contract["loaded_external_reference"] is True
    assert contract["runtime_seed_spec_hash"] is None
    assert contract["runtime_seed_spec_hash_verified"] is False
    assert contract["current_run_runtime_seed_spec_hash"] == "current-seed-hash"
    assert contract["run_family_id"] is None
    assert contract["run_family_id_verified"] is False
    assert contract["current_run_family_id"] == "current-run-family"


def test_timing_breakdown_uses_progress_events_for_legacy_artifacts(tmp_path):
    progress_json = tmp_path / "outer_optimizer_progress.json"
    progress_json.write_text(
        json.dumps(
            {
                "events": [
                    {
                        "label": "initial_hardware_status_returned",
                        "elapsed_s": 50.24,
                    },
                    {"label": "phase2_returned", "elapsed_s": 86.98},
                    {
                        "label": "final_penalty_metrics_returned",
                        "elapsed_s": 54.90,
                    },
                    {"label": "final_hardware_metrics_returned", "elapsed_s": 0.25},
                    {"label": "final_reporting_returned", "elapsed_s": 55.03},
                ],
            }
        ),
        encoding="utf-8",
    )
    case = {
        "elapsed_s": 208.87,
        "elapsed_source": "results_TIMINGS_script_total_s",
        "phase_timings": {},
        "outer_optimizer_progress_json": str(progress_json),
    }

    breakdown = _case_timing_breakdown(case)

    assert breakdown["core_optimizer_s"] == pytest.approx(86.98)
    assert breakdown["elapsed_source"] == "results_TIMINGS_script_total_s"
    assert breakdown["initial_hardware_status_s"] == pytest.approx(50.24)
    assert breakdown["final_penalty_metrics_s"] == pytest.approx(54.90)
    assert breakdown["final_reporting_s"] == pytest.approx(55.03)
    assert breakdown["status_reporting_s"] == pytest.approx(50.24 + 55.03)
    assert breakdown["optimizer_wall_excluding_status_reporting_s"] == pytest.approx(
        208.87 - 50.24 - 55.03
    )
    assert breakdown["status_reporting_fraction_of_elapsed"] == pytest.approx(
        (50.24 + 55.03) / 208.87
    )
    assert breakdown["non_core_s"] == pytest.approx(208.87 - 86.98)


def test_timing_breakdown_prefers_explicit_phase_timings(tmp_path):
    progress_json = tmp_path / "outer_optimizer_progress.json"
    progress_json.write_text(
        json.dumps(
            {
                "events": [
                    {
                        "label": "initial_hardware_status_returned",
                        "elapsed_s": 50.24,
                    },
                    {"label": "phase2_returned", "elapsed_s": 86.98},
                ],
            }
        ),
        encoding="utf-8",
    )
    case = {
        "elapsed_s": 10.0,
        "elapsed_source": "measured_subprocess_wall_s",
        "phase_timings": {
            "initial_hardware_status_s": 1.5,
            "final_penalty_metrics_s": 2.0,
            "outer_optimizer_main_s": 4.0,
        },
        "outer_optimizer_progress_json": str(progress_json),
    }

    breakdown = _case_timing_breakdown(case)

    assert breakdown["elapsed_source"] == "measured_subprocess_wall_s"
    assert breakdown["initial_hardware_status_s"] == pytest.approx(1.5)
    assert breakdown["core_optimizer_s"] == pytest.approx(4.0)
    assert breakdown["final_reporting_s"] == pytest.approx(2.0)
    assert breakdown["status_reporting_s"] == pytest.approx(3.5)
    assert breakdown["optimizer_wall_excluding_status_reporting_s"] == pytest.approx(
        6.5
    )
    assert breakdown[
        "optimizer_wall_excluding_status_reporting_fraction_of_elapsed"
    ] == pytest.approx(0.65)


def test_pair_timing_breakdown_reports_core_and_overhead_ratios(tmp_path):
    cpu_progress_json = tmp_path / "cpu_outer_optimizer_progress.json"
    gpu_progress_json = tmp_path / "gpu_outer_optimizer_progress.json"
    cpu_progress_json.write_text(
        json.dumps(
            {
                "events": [
                    {"label": "phase2_returned", "elapsed_s": 124.98},
                    {
                        "label": "initial_hardware_status_returned",
                        "elapsed_s": 0.22,
                    },
                    {
                        "label": "final_penalty_metrics_returned",
                        "elapsed_s": 0.78,
                    },
                    {"label": "final_reporting_returned", "elapsed_s": 3.16},
                ],
            }
        ),
        encoding="utf-8",
    )
    gpu_progress_json.write_text(
        json.dumps(
            {
                "events": [
                    {"label": "phase2_returned", "elapsed_s": 86.98},
                    {
                        "label": "initial_hardware_status_returned",
                        "elapsed_s": 50.24,
                    },
                    {
                        "label": "final_penalty_metrics_returned",
                        "elapsed_s": 54.90,
                    },
                    {"label": "final_reporting_returned", "elapsed_s": 55.03},
                ],
            }
        ),
        encoding="utf-8",
    )
    cpu_case = {
        "elapsed_s": 131.44,
        "elapsed_source": "results_TIMINGS_script_total_s",
        "phase_timings": {},
        "outer_optimizer_progress_json": str(cpu_progress_json),
    }
    gpu_case = {
        "elapsed_s": 208.87,
        "elapsed_source": "measured_subprocess_wall_s",
        "phase_timings": {},
        "outer_optimizer_progress_json": str(gpu_progress_json),
    }

    breakdown = _single_stage_pair_timing_breakdown(cpu_case, gpu_case)

    assert breakdown["jax_core_vs_cpu_core_ratio"] == pytest.approx(
        86.98 / 124.98
    )
    assert breakdown["jax_wall_vs_cpu_wall_ratio"] == pytest.approx(
        208.87 / 131.44
    )
    assert breakdown[
        "jax_optimizer_wall_excluding_status_reporting_vs_cpu_ratio"
    ] == pytest.approx((208.87 - 50.24 - 55.03) / (131.44 - 0.22 - 3.16))
    assert breakdown["cpu"]["elapsed_source"] == "results_TIMINGS_script_total_s"
    assert breakdown["jax"]["elapsed_source"] == "measured_subprocess_wall_s"
    assert breakdown["jax_status_reporting_minus_cpu_s"] == pytest.approx(
        (50.24 + 55.03) - (0.22 + 3.16)
    )


def test_explicit_warm_start_spec_comes_from_donor_after_rejected_cpu_reference(
    monkeypatch, tmp_path
):
    """Rejected CPU final artifacts are diagnostics, not JAX runtime seeds."""
    seed_dir = tmp_path / "donor"
    seed_dir.mkdir()
    (seed_dir / "results.json").write_text(
        json.dumps(
            {
                "init_only": False,
                "FINAL_IOTA": 0.22650585872006085,
                "FINAL_G": 1.0,
                "HARDWARE_CONSTRAINTS_OK": True,
                "SELF_INTERSECTING": False,
            }
        ),
        encoding="utf-8",
    )

    args = argparse.Namespace(
        maxiter=1500,
        warm_start_run_dir=str(seed_dir),
        jax_runtime_seed_spec=None,
        mpol=8,
        ntor=8,
        nphi=16,
        ntheta=8,
        iota_target=0.22650585872006085,
        num_tf_coils=20,
        optimizer_backend="host-jax",
        record_objective_evaluation_trace=False,
        platform="cuda",
    )
    compiled_from: list[Path] = []
    jax_seed_specs: list[Path] = []

    def fake_compile(run_dir, output_path, args):
        compiled_from.append(Path(run_dir))
        output_path.write_text("{}", encoding="utf-8")
        return output_path

    def fake_run_case(
        args,
        backend,
        *,
        platform,
        benchmark_mode=False,
        load_surface_gamma=True,
        output_root=None,
        jax_runtime_seed_spec=None,
        **kwargs,
    ):
        run_dir = Path(output_root) / f"{backend}_{platform}_run"
        if backend == "jax":
            jax_seed_specs.append(Path(jax_runtime_seed_spec))
        return {
            "results": {"provenance": {}},
            "surface_gamma": None,
            "elapsed_s": 0.0,
            "phase_timings": {},
            "run_dir": str(run_dir),
            "final_artifact_json": str(run_dir / "REJECTED.json"),
            "final_artifact_accepted": False,
            "outer_optimizer_progress_json": str(
                run_dir / "outer_optimizer_progress.json"
            ),
        }

    monkeypatch.setattr(
        "benchmarks.single_stage_init_parity._compile_jax_runtime_seed_spec_from_run_dir",
        fake_compile,
    )
    monkeypatch.setattr(
        "benchmarks.single_stage_init_parity._validate_jax_runtime_seed_spec_contract",
        lambda args, seed_spec_path: None,
    )
    monkeypatch.setattr(
        "benchmarks.single_stage_init_parity._run_single_stage_case",
        fake_run_case,
    )

    case_root = tmp_path / "case"
    case_root.mkdir()

    _, _, jax_seed_spec, _, _ = _run_single_stage_case_pair(
        args,
        benchmark_mode=False,
        reference_backend="cpu",
        reference_benchmark_mode=False,
        case_root=case_root,
    )

    assert compiled_from == [seed_dir]
    assert jax_seed_specs == [jax_seed_spec]


def test_precomputed_cpu_reference_artifacts_skip_cpu_child(monkeypatch, tmp_path):
    """A100 target-only runs should consume CPU-pod reference artifacts."""
    seed_dir = tmp_path / "donor"
    seed_dir.mkdir()
    (seed_dir / "results.json").write_text(
        json.dumps(
            {
                "init_only": False,
                "FINAL_IOTA": 0.22650585872006085,
                "FINAL_G": 1.0,
                "HARDWARE_CONSTRAINTS_OK": True,
                "SELF_INTERSECTING": False,
            }
        ),
        encoding="utf-8",
    )
    reference_artifacts_dir = tmp_path / "cpu_outputs"

    args = argparse.Namespace(
        maxiter=1500,
        warm_start_run_dir=str(seed_dir),
        jax_runtime_seed_spec=None,
        reference_case_artifacts_dir=str(reference_artifacts_dir),
        mpol=8,
        ntor=8,
        nphi=16,
        ntheta=8,
        iota_target=0.22650585872006085,
        num_tf_coils=20,
        optimizer_backend="host-jax",
        record_objective_evaluation_trace=False,
        platform="cuda",
    )
    compiled_from: list[Path] = []
    run_calls: list[tuple[str, str]] = []

    def fake_compile(run_dir, output_path, args):
        compiled_from.append(Path(run_dir))
        output_path.write_text("{}", encoding="utf-8")
        return output_path

    def fake_load_reference(output_root, args, *, backend, load_surface_gamma):
        assert Path(output_root) == reference_artifacts_dir
        assert backend == "cpu"
        assert load_surface_gamma is False
        run_dir = reference_artifacts_dir / "mpol=8-ntor=8-reference"
        return {
            "results": {"provenance": {}, "FINAL_IOTA": 0.1},
            "surface_gamma": None,
            "elapsed_s": 12.5,
            "phase_timings": {},
            "run_dir": str(run_dir),
            "final_artifact_json": str(run_dir / "REJECTED.json"),
            "final_artifact_accepted": False,
            "outer_optimizer_progress_json": str(
                run_dir / "outer_optimizer_progress.json"
            ),
        }

    def fake_run_case(
        args,
        backend,
        *,
        platform,
        benchmark_mode=False,
        load_surface_gamma=True,
        output_root=None,
        jax_runtime_seed_spec=None,
        **kwargs,
    ):
        assert backend == "jax"
        run_calls.append((backend, platform))
        run_dir = Path(output_root) / f"{backend}_{platform}_run"
        return {
            "results": {"provenance": {}, "FINAL_IOTA": 0.1},
            "surface_gamma": None,
            "elapsed_s": 1.0,
            "phase_timings": {},
            "run_dir": str(run_dir),
            "final_artifact_json": str(run_dir / "results.json"),
            "final_artifact_accepted": True,
            "outer_optimizer_progress_json": str(
                run_dir / "outer_optimizer_progress.json"
            ),
        }

    monkeypatch.setattr(
        "benchmarks.single_stage_init_parity._compile_jax_runtime_seed_spec_from_run_dir",
        fake_compile,
    )
    monkeypatch.setattr(
        "benchmarks.single_stage_init_parity._validate_jax_runtime_seed_spec_contract",
        lambda args, seed_spec_path: None,
    )
    monkeypatch.setattr(
        "benchmarks.single_stage_init_parity._load_single_stage_case_from_output_root",
        fake_load_reference,
    )
    monkeypatch.setattr(
        "benchmarks.single_stage_init_parity._run_single_stage_case",
        fake_run_case,
    )

    case_root = tmp_path / "case"
    case_root.mkdir()
    cpu_case, jax_case, _, seed_case, _ = _run_single_stage_case_pair(
        args,
        benchmark_mode=False,
        reference_backend="cpu",
        reference_benchmark_mode=False,
        case_root=case_root,
    )

    assert compiled_from == [seed_dir]
    assert run_calls == [("jax", "cuda")]
    assert seed_case is None
    assert cpu_case["run_dir"].endswith("mpol=8-ntor=8-reference")
    assert jax_case["final_artifact_accepted"] is True


def test_precomputed_reference_artifacts_reject_jax_reference_backend(tmp_path):
    seed_dir = tmp_path / "donor"
    seed_dir.mkdir()
    (seed_dir / "results.json").write_text(
        json.dumps(
            {
                "init_only": False,
                "FINAL_IOTA": 0.22650585872006085,
                "FINAL_G": 1.0,
                "HARDWARE_CONSTRAINTS_OK": True,
                "SELF_INTERSECTING": False,
            }
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        maxiter=1500,
        warm_start_run_dir=str(seed_dir),
        jax_runtime_seed_spec=None,
        reference_case_artifacts_dir=str(tmp_path / "cpu_outputs"),
        mpol=8,
        ntor=8,
        nphi=16,
        ntheta=8,
        iota_target=0.22650585872006085,
        num_tf_coils=20,
        optimizer_backend="host-jax",
        record_objective_evaluation_trace=False,
        platform="cuda",
    )

    with pytest.raises(ValueError, match="reference-case-artifacts-dir"):
        _run_single_stage_case_pair(
            args,
            benchmark_mode=False,
            reference_backend="jax",
            reference_benchmark_mode=False,
            case_root=tmp_path / "case",
        )
