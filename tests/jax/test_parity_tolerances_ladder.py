"""The parity tolerance ladder keeps its centrally owned, source-owned thresholds."""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import pytest
from simsopt_jax.parity_tolerances import parity_ladder_tolerances


def test_native_workflow_tolerance_is_centrally_owned_and_adversarial() -> None:
    tolerance = parity_ladder_tolerances("native_workflow")

    assert tolerance["requires_native_workflow_oracle"] is True
    assert tolerance["requires_direct_cpp_oracle"] is False
    assert float(tolerance["same_state_value_rtol"]) < 1.0e-8
    assert float(tolerance["terminal_relative_reduction"]) == 1.0e-12
    assert float(tolerance["terminal_constraint_norm_atol"]) == 1.0e-10
    assert float(tolerance["terminal_orthonormality_atol"]) == 1.0e-12


@pytest.mark.parametrize(
    ("bucket", "rtol", "atol"),
    (
        ("mirror_boozer_value", 1.0e-3, 1.0e-8),
        ("mirror_boozer_parameters", 0.0, 2.0e-3),
        ("mirror_optimization_5e2", 5.0e-2, 1.0e-9),
        ("mirror_optimization_3e2", 3.0e-2, 1.0e-9),
        ("mirror_optimization_2e2", 2.0e-2, 1.0e-9),
        ("mirror_optimization_5e3", 5.0e-3, 1.0e-10),
        ("mirror_optimization_1e1", 1.0e-1, 1.0e-9),
        ("mirror_pmqa_final", 5.0e-4, 0.0),
        ("mirror_qfm_value", 5.0e-5, 1.0e-7),
        ("mirror_qfm_parameters", 2.0e-3, 2.0e-4),
        ("mirror_qfm_persistence", 2.0e-2, 1.0e-5),
        ("mirror_surface_invariant", 1.0e-8, 1.0e-10),
        ("mirror_trace_ncsx_time", 0.0, 2.0e-2),
        ("mirror_trace_ncsx_state", 0.0, 3.0e-2),
        ("mirror_trace_qa_time", 0.0, 6.0e-3),
        ("mirror_trace_qa_state", 0.0, 2.0e-3),
        ("mirror_trace_qa_poincare", 0.0, 7.0e-3),
        ("mirror_trace_particle_time", 0.0, 2.0e-6),
        ("mirror_trace_particle_state", 0.0, 2.0e-3),
    ),
)
def test_mirror_parity_tolerances_preserve_source_owned_thresholds(
    bucket: str,
    rtol: float,
    atol: float,
) -> None:
    tolerance = parity_ladder_tolerances(bucket)

    assert tolerance["rtol"] == rtol
    assert tolerance["atol"] == atol
