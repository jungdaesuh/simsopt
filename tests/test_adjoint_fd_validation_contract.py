from __future__ import annotations

from argparse import ArgumentTypeError, Namespace
import sys

import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks.benchmark_timing_labels import GPU_TARGET_ONLY, MIXED_PARITY_REFERENCE
from benchmarks import adjoint_fd_validation as fd_validation
from benchmarks import tier5_performance_characterization as tier5
from benchmarks.single_stage_smoke_fixture import (
    DEFAULT_SINGLE_STAGE_JAX_RUNTIME_SEED_MPOL,
    DEFAULT_SINGLE_STAGE_JAX_RUNTIME_SEED_NPHI,
    DEFAULT_SINGLE_STAGE_JAX_RUNTIME_SEED_NTHETA,
    DEFAULT_SINGLE_STAGE_JAX_RUNTIME_SEED_NTOR,
    SMOKE_TEST_SINGLE_STAGE_JAX_RUNTIME_SEED_SPEC,
)
from benchmarks.validation_ladder_contract import TIER4_ADJOINT_FD_EPS_LADDER


def _synthetic_objective(coil_dofs):
    return 0.5 * jnp.sum(coil_dofs**2) + 0.25 * jnp.sum(coil_dofs)


def _synthetic_value_and_grad(coil_dofs):
    return _synthetic_objective(coil_dofs), coil_dofs + 0.25


def _synthetic_forward_result(coil_dofs):
    return {
        "value": _synthetic_objective(coil_dofs),
        "success": jnp.asarray(True),
        "primal_success": jnp.asarray(True),
        "adjoint_linear_solve_available": jnp.asarray(True),
        "iota": jnp.asarray(0.15, dtype=jnp.float64),
        "G": jnp.asarray(-2.0, dtype=jnp.float64),
    }


def _valid_traceable_fd_record():
    return fd_validation.compute_traceable_single_stage_fd_ladder(
        _synthetic_forward_result,
        _synthetic_value_and_grad,
        np.asarray([0.5, -1.0, 2.0, 0.25], dtype=np.float64),
        samples=3,
        eps_ladder=TIER4_ADJOINT_FD_EPS_LADDER,
    )


def _valid_metrics():
    return {
        "adjoint_residual_rel": 1e-12,
        "implicit_gradient_finite": True,
        "implicit_gradient_norm": 1.0,
        "total_gradient_finite": True,
        "total_gradient_norm": 1.0,
        "recomposed_total_rel": 1e-14,
        "fd_samples": [
            {
                "sample_index": 0,
                "accepted": True,
                "rel_err": 0.0,
                "abs_err": 0.0,
            }
        ],
        "stable_resolve_fd_samples": 1,
        "min_stable_resolve_fd_samples": 1,
        "full_resolve_fd_samples": [
            {
                "sample_index": 0,
                "stable": True,
                "accepted": True,
                "rel_err": 0.0,
                "abs_err": 0.0,
            }
        ],
        "forward_operator_contract": {
            "forward_success": True,
            "linear_solve_factors_none": True,
            "dense_linear_solve_factors_available": False,
            "linear_solve_backend": "operator",
        },
        "operator_dot_product_contract": {
            "validated_quantity": "runtime_linearization_transpose_dot_product",
            "linear_solve_backend": "operator",
            "decision_size": 4,
            "rel_tol": 1e-10,
            "abs_tol": 1e-10,
            "samples": [
                {
                    "sample_index": 0,
                    "accepted": True,
                    "lhs_forward_dot_covector": 1.0,
                    "rhs_vector_dot_transpose": 1.0,
                    "abs_err": 0.0,
                    "rel_err": 0.0,
                }
            ],
            "passed": True,
            "status": "evaluated",
            "blocking": True,
        },
        "jvp_vjp_feasibility": {
            "status": "unsupported",
            "blocking": False,
            "reason": "public_objective_custom_vjp_forward_mode_unavailable",
        },
        "traceable_single_stage_fd": _valid_traceable_fd_record(),
    }


def test_default_fd_eps_ladder_uses_production_noise_window():
    args = Namespace(eps=None, eps_ladder=TIER4_ADJOINT_FD_EPS_LADDER)

    assert fd_validation._resolve_fd_eps_ladder(args) == TIER4_ADJOINT_FD_EPS_LADDER


def test_single_fd_eps_cannot_certify_production_ladder():
    args = Namespace(eps=3e-3, eps_ladder=TIER4_ADJOINT_FD_EPS_LADDER)

    with pytest.raises(ValueError, match="single --eps cannot certify"):
        fd_validation._resolve_fd_eps_ladder(args)


def test_traceable_fd_ladder_rejects_single_step_certificate():
    with pytest.raises(ValueError, match="at least two"):
        fd_validation.compute_traceable_single_stage_fd_ladder(
            _synthetic_forward_result,
            _synthetic_value_and_grad,
            np.asarray([0.5, -1.0, 2.0, 0.25], dtype=np.float64),
            samples=1,
            eps_ladder=(3e-3,),
        )


def test_adjoint_fd_parse_requires_explicit_seed_source(monkeypatch):
    """No default seed: a certificate must state which seed it certifies."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["adjoint_fd_validation.py", "--output-json", "/tmp/adjoint-default.json"],
    )

    with pytest.raises(SystemExit):
        fd_validation.parse_args()


def test_adjoint_fd_parse_accepts_explicit_runtime_seed_spec(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "adjoint_fd_validation.py",
            "--output-json",
            "/tmp/adjoint-default.json",
            "--jax-runtime-seed-spec",
            "/seed/single_stage_jax_runtime_spec.json",
        ],
    )

    args = fd_validation.parse_args()

    assert args.jax_runtime_seed_spec == "/seed/single_stage_jax_runtime_spec.json"
    assert args.nphi == DEFAULT_SINGLE_STAGE_JAX_RUNTIME_SEED_NPHI
    assert args.ntheta == DEFAULT_SINGLE_STAGE_JAX_RUNTIME_SEED_NTHETA
    assert args.mpol == DEFAULT_SINGLE_STAGE_JAX_RUNTIME_SEED_MPOL
    assert args.ntor == DEFAULT_SINGLE_STAGE_JAX_RUNTIME_SEED_NTOR


def test_adjoint_fd_parse_raw_stage2_seed_requires_stage2_path(monkeypatch):
    """--raw-stage2-seed must name the seed; it cannot fall back to a fixture."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "adjoint_fd_validation.py",
            "--output-json",
            "/tmp/adjoint-raw.json",
            "--raw-stage2-seed",
        ],
    )

    with pytest.raises(SystemExit):
        fd_validation.parse_args()


def test_adjoint_fd_parse_raw_stage2_seed_clears_runtime_seed(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "adjoint_fd_validation.py",
            "--output-json",
            "/tmp/adjoint-raw.json",
            "--raw-stage2-seed",
            "--stage2-bs-path",
            "/seed/biot_savart_opt.json",
        ],
    )

    args = fd_validation.parse_args()

    assert args.jax_runtime_seed_spec is None
    assert args.stage2_bs_path == "/seed/biot_savart_opt.json"


def test_tier5_forwards_default_runtime_seed_and_fd_ladder():
    args = Namespace(
        equilibrium_path=None,
        plasma_surf_filename="wout.nc",
        equilibria_dir="equilibria",
        stage2_bs_path="stage2.json",
        single_stage_nphi=DEFAULT_SINGLE_STAGE_JAX_RUNTIME_SEED_NPHI,
        single_stage_ntheta=DEFAULT_SINGLE_STAGE_JAX_RUNTIME_SEED_NTHETA,
        mpol=DEFAULT_SINGLE_STAGE_JAX_RUNTIME_SEED_MPOL,
        ntor=DEFAULT_SINGLE_STAGE_JAX_RUNTIME_SEED_NTOR,
        vol_target=0.10,
        iota_target=0.15,
        optimizer_backend="scipy-jax",
        jax_runtime_seed_spec=str(SMOKE_TEST_SINGLE_STAGE_JAX_RUNTIME_SEED_SPEC),
        samples=3,
        eps=None,
        eps_ladder=TIER4_ADJOINT_FD_EPS_LADDER,
    )

    command = tier5._tier4_probe_args(args)

    assert "--eps" not in command
    assert command[command.index("--eps-ladder") + 1 :] == [
        str(value) for value in TIER4_ADJOINT_FD_EPS_LADDER
    ]
    assert command[command.index("--jax-runtime-seed-spec") + 1] == str(
        SMOKE_TEST_SINGLE_STAGE_JAX_RUNTIME_SEED_SPEC
    )
    assert command[command.index("--mpol") + 1] == str(
        DEFAULT_SINGLE_STAGE_JAX_RUNTIME_SEED_MPOL
    )


def test_tier5_forwards_runtime_seed_spec_to_tier4_probe():
    args = Namespace(
        equilibrium_path=None,
        plasma_surf_filename="wout.nc",
        equilibria_dir="equilibria",
        stage2_bs_path="stage2.json",
        single_stage_nphi=255,
        single_stage_ntheta=64,
        mpol=10,
        ntor=10,
        vol_target=0.10,
        iota_target=0.15,
        optimizer_backend="scipy-jax",
        jax_runtime_seed_spec="seed/single_stage_jax_runtime_spec.json",
        samples=1,
        eps=None,
        eps_ladder=TIER4_ADJOINT_FD_EPS_LADDER,
    )

    command = tier5._tier4_probe_args(args)

    assert command[command.index("--jax-runtime-seed-spec") + 1] == (
        "seed/single_stage_jax_runtime_spec.json"
    )


def test_tier5_rejects_single_fd_eps_forwarding():
    args = Namespace(
        equilibrium_path=None,
        plasma_surf_filename="wout.nc",
        equilibria_dir="equilibria",
        stage2_bs_path="stage2.json",
        single_stage_nphi=DEFAULT_SINGLE_STAGE_JAX_RUNTIME_SEED_NPHI,
        single_stage_ntheta=DEFAULT_SINGLE_STAGE_JAX_RUNTIME_SEED_NTHETA,
        mpol=DEFAULT_SINGLE_STAGE_JAX_RUNTIME_SEED_MPOL,
        ntor=DEFAULT_SINGLE_STAGE_JAX_RUNTIME_SEED_NTOR,
        vol_target=0.10,
        iota_target=0.15,
        optimizer_backend="scipy-jax",
        jax_runtime_seed_spec=str(SMOKE_TEST_SINGLE_STAGE_JAX_RUNTIME_SEED_SPEC),
        samples=1,
        eps=3e-3,
        eps_ladder=TIER4_ADJOINT_FD_EPS_LADDER,
    )

    with pytest.raises(ValueError, match="single --eps cannot certify"):
        tier5._tier4_probe_args(args)


def test_tier5_stage2_summary_labels_mixed_artifact_and_target_headline():
    summary = tier5.summarize_stage2_e2e_performance_probe(
        payload={
            "passed": True,
            "comparison": {
                "cpu_elapsed_s": 10.0,
                "jax_elapsed_s": 6.0,
            },
            "timings": {
                "jax_outer_elapsed_s": 4.0,
                "jax_optimizer_warm_run_s": 2.0,
                "jax_optimizer_compile_overhead_s": 1.0,
            },
        },
        outer_elapsed_s=12.0,
        lane_label="jax-cuda",
    )

    assert summary["timing_classification"] == MIXED_PARITY_REFERENCE
    assert summary["supports_performance_headline"] is True
    assert summary["headline_timing_classification"] == GPU_TARGET_ONLY
    assert summary["outer_speedup_vs_cpu"] == pytest.approx(2.5)


def test_smaller_single_fd_eps_is_inside_adaptive_window():
    assert fd_validation._production_fd_eps("1e-4") == 1e-4


def test_oversized_single_fd_eps_is_rejected():
    with pytest.raises(
        ArgumentTypeError,
        match="production FD eps must be in",
    ):
        fd_validation._production_fd_eps("1e-2")


def test_reduced_mpol_is_rejected_for_phase1_fd_certification():
    with pytest.raises(ArgumentTypeError, match="mpol >= 8"):
        fd_validation.validate_production_fd_scale(Namespace(mpol=2))


def test_runtime_seed_fixture_requires_traceable_linearization_install():
    class _BoozerSurface:
        def __init__(self):
            self.called = False

        def install_traceable_hessian_linearization_for_value_only_state(self):
            self.called = True

    boozer_surface = _BoozerSurface()

    fd_validation._install_traceable_runtime_seed_linearization(boozer_surface)

    assert boozer_surface.called is True


def test_runtime_seed_fixture_rejects_missing_linearization_installer():
    with pytest.raises(TypeError, match="install_traceable_hessian_linearization"):
        fd_validation._install_traceable_runtime_seed_linearization(object())


def test_runtime_seed_coil_layout_comes_from_runtime_spec():
    class _Seed:
        num_tf_coils = 12
        banana_curve_index = 14

    class _RuntimeSpec:
        seed = _Seed()

    assert fd_validation._runtime_seed_coil_layout(_RuntimeSpec()) == (12, 14)


def test_runtime_bundle_forward_result_proves_operator_metadata_contract():
    def forward_result(coil_dofs):
        return {
            "success": jnp.asarray(True),
            "linear_solve_factors": None,
            "dense_linear_solve_factors_available": False,
            "linear_solve_backend": "operator",
            "x": jnp.zeros_like(coil_dofs),
        }

    contract = fd_validation.traceable_single_stage_forward_operator_contract(
        {"forward_result": forward_result},
        np.asarray([1.0, 2.0], dtype=np.float64),
    )

    assert contract == {
        "forward_success": True,
        "linear_solve_factors_none": True,
        "dense_linear_solve_factors_available": False,
        "linear_solve_backend": "operator",
    }


def test_traceable_single_stage_fd_ladder_accepts_quadratic_oracle():
    record = _valid_traceable_fd_record()

    assert record["validated_quantity"] == (
        "fused_traceable_single_stage_gradient_after_inner_resolve"
    )
    assert record["eps_ladder"] == list(TIER4_ADJOINT_FD_EPS_LADDER)
    assert record["gradient_finite"] is True
    assert record["gradient_norm"] > 0.0
    assert len(record["directions"]) == 3
    for direction in record["directions"]:
        assert direction["accepted"] is True
        assert direction["status"] == "accepted"
        assert direction["taylor_rate_decrease"] is True
        assert direction["richardson"] is not None
        assert len(direction["eps_records"]) == len(TIER4_ADJOINT_FD_EPS_LADDER)
        for eps_record in direction["eps_records"]:
            assert eps_record["plus"]["success"] is True
            assert eps_record["minus"]["success"] is True
            assert eps_record["plus"]["primal_success"] is True
            assert eps_record["minus"]["primal_success"] is True


def test_traceable_single_stage_fd_ladder_marks_invalid_window():
    def rejected_forward_result(coil_dofs):
        result = _synthetic_forward_result(coil_dofs)
        result["success"] = jnp.asarray(False)
        return result

    record = fd_validation.compute_traceable_single_stage_fd_ladder(
        rejected_forward_result,
        _synthetic_value_and_grad,
        np.asarray([0.5, -1.0, 2.0, 0.25], dtype=np.float64),
        samples=1,
        eps_ladder=TIER4_ADJOINT_FD_EPS_LADDER,
    )

    direction = record["directions"][0]

    assert direction["accepted"] is False
    assert direction["status"] == "invalid_fd_window"
    assert direction["eps_records"][0]["plus"]["success"] is False


def test_traceable_single_stage_fd_ladder_marks_inner_solve_failure():
    def failed_forward_result(coil_dofs):
        result = _synthetic_forward_result(coil_dofs)
        result["success"] = jnp.asarray(False)
        result["primal_success"] = jnp.asarray(False)
        return result

    record = fd_validation.compute_traceable_single_stage_fd_ladder(
        failed_forward_result,
        _synthetic_value_and_grad,
        np.asarray([0.5, -1.0, 2.0, 0.25], dtype=np.float64),
        samples=1,
        eps_ladder=TIER4_ADJOINT_FD_EPS_LADDER,
    )

    direction = record["directions"][0]

    assert direction["accepted"] is False
    assert direction["status"] == "inner_solve_failed"


def test_traceable_single_stage_validation_reports_invalid_fd_window():
    metrics = _valid_metrics()
    fd_record = dict(metrics["traceable_single_stage_fd"])
    directions = [dict(direction) for direction in fd_record["directions"]]
    directions[0]["accepted"] = False
    directions[0]["status"] = "invalid_fd_window"
    directions[0]["eps_records"] = [
        {
            **dict(record),
            "status": "invalid_fd_window",
            "accepted": False,
            "plus": {**dict(record["plus"]), "success": False},
        }
        for record in directions[0]["eps_records"]
    ]
    fd_record["directions"] = directions
    metrics["traceable_single_stage_fd"] = fd_record

    failures = fd_validation.evaluate_traceable_single_stage_validation(
        {
            "forward_operator_contract": metrics["forward_operator_contract"],
            "traceable_single_stage_fd": metrics["traceable_single_stage_fd"],
        }
    )

    assert any("status=invalid_fd_window" in failure for failure in failures)


def test_operator_dot_product_contract_accepts_symmetric_operator():
    matrix = np.asarray(
        [
            [2.0, 0.25, -0.5],
            [0.25, 3.0, 0.75],
            [-0.5, 0.75, 4.0],
        ],
        dtype=np.float64,
    )

    class _AdjointRuntimeState:
        decision_size = 3
        linear_solve_backend = "operator"

        def apply_forward(self, vector):
            return jnp.asarray(matrix) @ vector

        def apply_transpose(self, covector):
            return jnp.asarray(matrix.T) @ covector

    contract = fd_validation.compute_operator_dot_product_contract(
        _AdjointRuntimeState(),
        samples=3,
    )

    assert contract["passed"] is True
    assert all(sample["accepted"] for sample in contract["samples"])


def test_operator_dot_product_contract_rejects_bad_transpose():
    matrix = np.asarray(
        [
            [2.0, 0.25, -0.5],
            [0.25, 3.0, 0.75],
            [-0.5, 0.75, 4.0],
        ],
        dtype=np.float64,
    )

    class _AdjointRuntimeState:
        decision_size = 3
        linear_solve_backend = "operator"

        def apply_forward(self, vector):
            return jnp.asarray(matrix) @ vector

        def apply_transpose(self, covector):
            return jnp.asarray(matrix) @ covector + 0.1

    contract = fd_validation.compute_operator_dot_product_contract(
        _AdjointRuntimeState(),
        samples=3,
    )

    assert contract["passed"] is False
    assert any(not sample["accepted"] for sample in contract["samples"])


def test_optional_operator_dot_product_contract_records_unavailable_state():
    class _BoozerSurface:
        def get_adjoint_runtime_state(self):
            raise RuntimeError(
                "BoozerSurfaceJAX has no valid adjoint state. "
                "Call boozer_surface.run_code(...) before requesting adjoints."
            )

    contract = fd_validation.compute_optional_operator_dot_product_contract(
        _BoozerSurface(),
        samples=3,
    )

    assert contract["status"] == "unsupported"
    assert contract["blocking"] is False
    assert contract["passed"] is True
    assert contract["samples"] == []


def test_jvp_vjp_feasibility_records_nonblocking_public_custom_vjp_boundary():
    feasibility = fd_validation.traceable_single_stage_jvp_vjp_feasibility()

    assert feasibility == {
        "status": "unsupported",
        "blocking": False,
        "reason": "public_objective_custom_vjp_forward_mode_unavailable",
    }


def test_adjoint_validation_requires_forward_operator_metadata():
    metrics = _valid_metrics()
    metrics["forward_operator_contract"] = {
        "forward_success": True,
        "linear_solve_factors_none": False,
        "dense_linear_solve_factors_available": True,
        "linear_solve_backend": "dense_plu",
    }

    failures = fd_validation.evaluate_adjoint_validation(metrics)

    assert any("forward operator path" in failure for failure in failures)


def test_adjoint_validation_accepts_complete_traceable_fd_contract():
    assert fd_validation.evaluate_adjoint_validation(_valid_metrics()) == []


def test_traceable_runtime_seed_validation_accepts_traceable_only_contract():
    metrics = _valid_metrics()

    failures = fd_validation.evaluate_traceable_single_stage_validation(
        {
            "forward_operator_contract": metrics["forward_operator_contract"],
            "traceable_single_stage_fd": metrics["traceable_single_stage_fd"],
        }
    )

    assert failures == []
