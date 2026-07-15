"""Pure-Python tests for run-code diagnostic timing comparability."""

from benchmarks import cpp_baseline_benchmark, run_code_benchmark_common
from benchmarks.benchmark_config import BenchmarkConfig
from benchmarks.run_code_benchmark_common import (
    BenchmarkBackendResult,
    BenchmarkRepeatResult,
    BenchmarkTimingResult,
    assess_benchmark_diagnostic_ratio,
    assess_benchmark_timing,
)


def _timing_result(
    *,
    median_seconds: float,
    success: bool = True,
    iterations: int = 8,
    final_fun: float = 2.0e-8,
    final_iota: float = 3.0e-4,
) -> BenchmarkTimingResult:
    return BenchmarkTimingResult(
        repeats=(
            BenchmarkRepeatResult(
                elapsed_seconds=median_seconds,
                success=success,
                iterations=iterations,
                final_fun=final_fun,
                final_iota=final_iota,
            ),
        )
    )


def _backend_result(
    timed_repeats: BenchmarkTimingResult,
    *,
    first_call_seconds: float,
    least_squares_seconds: float,
    newton_seconds: float,
) -> BenchmarkBackendResult:
    first_call = BenchmarkRepeatResult(
        elapsed_seconds=first_call_seconds,
        success=True,
        iterations=8,
        final_fun=2.0e-8,
        final_iota=3.0e-4,
    )
    stage_split = BenchmarkRepeatResult(
        elapsed_seconds=least_squares_seconds + newton_seconds,
        success=True,
        iterations=8,
        final_fun=2.0e-8,
        final_iota=3.0e-4,
    )
    return BenchmarkBackendResult(
        first_call=first_call,
        least_squares_seconds=least_squares_seconds,
        newton_seconds=newton_seconds,
        stage_split=stage_split,
        timed_repeats=timed_repeats,
    )


def test_diagnostic_ratio_is_available_for_matching_converged_outcomes():
    reference = _timing_result(median_seconds=12.0)
    candidate = _timing_result(
        median_seconds=3.0,
        iterations=6,
        final_fun=2.1e-8,
        final_iota=3.0001e-4,
    )

    assessment = assess_benchmark_diagnostic_ratio(reference, candidate)

    assert assessment.observed_ratio == 4.0
    assert assessment.time_reduction_percent == 75.0
    assert assessment.reasons == ()


def test_single_timing_assessment_rejects_empty_repeat_set():
    assessment = assess_benchmark_timing(
        BenchmarkTimingResult(repeats=()),
        role="C++",
    )

    assert assessment.diagnostic_comparable is False
    assert assessment.reasons == ("C++ has no timed repeats",)


def test_diagnostic_ratio_is_unavailable_when_both_repeat_sets_are_empty():
    empty_timing = BenchmarkTimingResult(repeats=())

    assessment = assess_benchmark_diagnostic_ratio(empty_timing, empty_timing)

    assert assessment.observed_ratio is None
    assert assessment.time_reduction_percent is None
    assert assessment.reasons == (
        "reference has no timed repeats",
        "candidate has no timed repeats",
    )


def test_diagnostic_ratio_is_unavailable_when_either_solver_did_not_converge():
    reference = _timing_result(median_seconds=12.0, success=False)
    candidate = _timing_result(median_seconds=3.0, success=False)

    assessment = assess_benchmark_diagnostic_ratio(reference, candidate)

    assert assessment.observed_ratio is None
    assert assessment.time_reduction_percent is None
    assert assessment.reasons == (
        "reference repeat 1 solver did not converge",
        "candidate repeat 1 solver did not converge",
    )


def test_diagnostic_ratio_is_unavailable_for_mismatched_final_objective():
    reference = _timing_result(median_seconds=12.0, final_fun=1.0e-8)
    candidate = _timing_result(median_seconds=3.0, final_fun=2.0e-6)

    assessment = assess_benchmark_diagnostic_ratio(reference, candidate)

    assert assessment.observed_ratio is None
    assert assessment.time_reduction_percent is None
    assert assessment.reasons == (
        "repeat final objectives are not mutually comparable under the "
        "whole-solve parity tolerance",
    )


def test_diagnostic_ratio_is_unavailable_for_mismatched_final_iota():
    reference = _timing_result(median_seconds=12.0, final_iota=1.0e-4)
    candidate = _timing_result(median_seconds=3.0, final_iota=3.0e-4)

    assessment = assess_benchmark_diagnostic_ratio(reference, candidate)

    assert assessment.observed_ratio is None
    assert assessment.time_reduction_percent is None
    assert assessment.reasons == (
        "repeat final iotas are not mutually comparable under the "
        "whole-solve parity tolerance",
    )


def test_earlier_failed_repeat_remains_visible_and_suppresses_diagnostic_ratio(
    monkeypatch,
    capsys,
):
    config = BenchmarkConfig("contract fixture", 4, 8, 8, 2, 2)
    reference_timing = BenchmarkTimingResult(
        repeats=(
            BenchmarkRepeatResult(
                elapsed_seconds=4.0,
                success=False,
                iterations=10,
                final_fun=2.0e-6,
                final_iota=3.0e-4,
            ),
            BenchmarkRepeatResult(
                elapsed_seconds=2.0,
                success=True,
                iterations=8,
                final_fun=2.0e-8,
                final_iota=3.0e-4,
            ),
        )
    )
    candidate_timing = BenchmarkTimingResult(
        repeats=(
            BenchmarkRepeatResult(
                elapsed_seconds=2.0,
                success=True,
                iterations=8,
                final_fun=2.0e-8,
                final_iota=3.0e-4,
            ),
            BenchmarkRepeatResult(
                elapsed_seconds=1.0,
                success=True,
                iterations=8,
                final_fun=2.0e-8,
                final_iota=3.0e-4,
            ),
        )
    )
    timing_by_backend = {
        "scipy": reference_timing,
        "ondevice": candidate_timing,
    }

    def fake_benchmark_backend(
        _config,
        optimizer_backend,
        *,
        repeats,
        option_overrides,
    ):
        assert repeats == 2
        assert option_overrides is None
        return _backend_result(
            timing_by_backend[optimizer_backend],
            first_call_seconds=7.0,
            least_squares_seconds=2.0,
            newton_seconds=5.0,
        )

    monkeypatch.setattr(
        run_code_benchmark_common,
        "benchmark_backend",
        fake_benchmark_backend,
    )

    run_code_benchmark_common.run_benchmarks(
        title="contract fixture",
        configs=(config,),
        backends=("scipy", "ondevice"),
        repeats=2,
    )

    output = capsys.readouterr().out
    assert "repeat fresh solve: 3000.0ms median" in output
    assert (
        "repeat 1: 4000.0ms  success=False  iter=10  fun=2.000000e-06  iota=0.000300"
    ) in output
    assert (
        "repeat 2: 2000.0ms  success=True  iter=8  fun=2.000000e-08  iota=0.000300"
    ) in output
    assert "reference repeat 1 solver did not converge" in output
    assert "repeat final objectives are not mutually comparable" in output
    assert "diagnostic timing ratio unavailable" in output
    assert "No performance or break-even verdict is emitted" in output
    assert "diagnostic timing ratio only" not in output


def test_diagnostic_ratio_output_states_formula_and_time_reduction(
    monkeypatch,
    capsys,
):
    config = BenchmarkConfig("contract fixture", 4, 8, 8, 2, 2)
    timing_by_backend = {
        "scipy": _timing_result(median_seconds=12.0),
        "ondevice": _timing_result(median_seconds=3.0),
    }

    def fake_benchmark_backend(
        _config,
        optimizer_backend,
        *,
        repeats,
        option_overrides,
    ):
        assert repeats == 1
        assert option_overrides is None
        return _backend_result(
            timing_by_backend[optimizer_backend],
            first_call_seconds=0.0,
            least_squares_seconds=0.0,
            newton_seconds=0.0,
        )

    monkeypatch.setattr(
        run_code_benchmark_common,
        "benchmark_backend",
        fake_benchmark_backend,
    )

    run_code_benchmark_common.run_benchmarks(
        title="contract fixture",
        configs=(config,),
        backends=("scipy", "ondevice"),
        repeats=1,
    )

    output = capsys.readouterr().out
    assert "scipy median / ondevice median = 4.00x" in output
    assert "ondevice time reduction vs scipy = 75.00%" in output
    assert "not a performance claim" in output
    assert "No performance or break-even verdict is emitted" in output


def test_cpp_earlier_failed_repeat_remains_visible_and_ineligible(
    monkeypatch,
    capsys,
):
    config = BenchmarkConfig("C++ contract fixture", 4, 8, 8, 2, 2)
    converged_result = {
        "success": True,
        "iter": 8,
        "fun": 2.0e-8,
        "iota": 3.0e-4,
    }
    timed_results = iter(
        (
            (
                4.0,
                {
                    "success": False,
                    "iter": 10,
                    "fun": 2.0e-6,
                    "iota": 3.0e-4,
                },
            ),
            (2.0, converged_result),
        )
    )

    monkeypatch.setattr(cpp_baseline_benchmark, "get_git_sha", lambda: "fixture")
    monkeypatch.setattr(
        cpp_baseline_benchmark,
        "time_run_code_stage_split",
        lambda _config: (2.0, 5.0, converged_result),
    )

    first_call_complete = False

    def fake_time_run_code(_config):
        nonlocal first_call_complete
        if not first_call_complete:
            first_call_complete = True
            return 7.0, converged_result
        return next(timed_results)

    monkeypatch.setattr(
        cpp_baseline_benchmark,
        "time_run_code",
        fake_time_run_code,
    )

    cpp_baseline_benchmark.run_benchmarks(configs=(config,), repeats=2)

    output = capsys.readouterr().out
    assert "repeat fresh solve: 3000.0ms median" in output
    assert (
        "repeat 1: 4000.0ms  success=False  iter=10  fun=2.000000e-06  iota=0.000300"
    ) in output
    assert (
        "repeat 2: 2000.0ms  success=True  iter=8  fun=2.000000e-08  iota=0.000300"
    ) in output
    assert "C++ repeat 1 solver did not converge" in output
    assert "repeat final objectives are not mutually comparable" in output
    assert "diagnostic outcome comparability: no" in output
    assert "diagnostic outcome comparability: yes" not in output
