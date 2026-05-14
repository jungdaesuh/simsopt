"""Parser invariants for JAX compile-diagnostic accounting.

This file verifies the bookkeeping/parsing invariants of the
``JAX_COMPILE_DIAGNOSTICS`` recorder threaded through the single-stage
example: that parsed compile-target and cache-miss-site counts sum to
the event counts minus the recorded parse misses. It does **not** test
physics parity — it tests that the diagnostic recorder/parser itself
stays internally consistent across runtime changes.

Moved here from ``tests/integration/test_single_stage_physics_parity.py``
per audit finding #23 in
``.artifacts/jax-test-audit-2026-05-13/TEST_QUALITY_TODOS.md``: a
diagnostic/instrumentation test should not live in a file named
``*_physics_parity.py`` (downstream agents read that file as physics
evidence).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Reach the heavy subprocess driver that already lives in the integration
# test file. Duplicating ~200 lines of subprocess plumbing here would
# create a divergence risk against the integration suite's source of
# truth; instead, add ``tests/`` to ``sys.path`` so the package-style
# import ``integration.test_single_stage_physics_parity`` resolves to
# the same module pytest collects directly.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_TESTS_ROOT = _REPO_ROOT / "tests"
if str(_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TESTS_ROOT))

from integration.test_single_stage_physics_parity import (  # noqa: E402
    DEFAULT_STAGE2_BS_PATH,
    _run_single_stage_script_results,
)


class TestJaxCompileDiagnosticParser:
    """Audit #23: verifies parser invariants on JAX_COMPILE_DIAGNOSTICS.

    Each test runs the single-stage example with
    ``--record-jax-compile-diagnostics`` and inspects the recorded
    accounting structure. The invariant under test is that the parser
    accounts for every recorded compile / cache-miss event exactly once:
    the per-target / per-site bucket totals equal the global event count
    minus the recorded parse-miss count.
    """

    @pytest.mark.slow
    def test_cpu_target_lane_case_records_compile_diagnostic_accounting(self):
        results = _run_single_stage_script_results(
            backend="jax",
            optimizer_backend="ondevice",
            platform="cpu",
            stage2_bs_path=DEFAULT_STAGE2_BS_PATH,
            maxiter=2,
            benchmark_mode=True,
            record_jax_compile_diagnostics=True,
            target_lane_accepted_step_sync="final-only",
        )

        diagnostics = results.get("JAX_COMPILE_DIAGNOSTICS")
        assert isinstance(diagnostics, dict)
        compile_targets = diagnostics.get("compile_targets")
        cache_miss_sites = diagnostics.get("cache_miss_sites")
        assert isinstance(compile_targets, dict)
        assert isinstance(cache_miss_sites, dict)
        compile_event_count = int(diagnostics.get("compile_event_count", -1))
        cache_miss_count = int(diagnostics.get("cache_miss_count", -1))
        compile_target_parse_miss_count = int(
            diagnostics.get("compile_target_parse_miss_count", -1)
        )
        cache_miss_site_parse_miss_count = int(
            diagnostics.get("cache_miss_site_parse_miss_count", -1)
        )
        assert compile_event_count >= 0
        assert cache_miss_count >= 0
        assert compile_target_parse_miss_count >= 0
        assert cache_miss_site_parse_miss_count >= 0
        assert sum(int(value) for value in compile_targets.values()) == (
            compile_event_count - compile_target_parse_miss_count
        )
        assert sum(int(value) for value in cache_miss_sites.values()) == (
            cache_miss_count - cache_miss_site_parse_miss_count
        )
