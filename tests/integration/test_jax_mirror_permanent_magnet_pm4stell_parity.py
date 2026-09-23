"""Exact parity for the ``2_Intermediate/permanent_magnet_PM4Stell.py`` mirror.

The bounded configuration is the official ``in_github_actions`` configuration
(K=100, max_nMagnets=20, nBacktracking=200, nAdjacent=10, nHistory=10,
downsample=100, N=2), so the native lane has an official counterpart: the
``ci`` variant of ``examples/jax/parity/official_reference``, which is tracked
and therefore readable on a clean checkout.

The native lane and its construction run in a child process at
``OMP_NUM_THREADS=1``, the thread count every official number was measured at.

The PM4Stell backtracking predicate of this branch (``aa04f698c``) is an open
question with the user (C1): it is NOT changed here, so the official PM4Stell
end point at the shipped scale is not expected to be reproduced. What is
checked below is the official CI run, which this branch does reproduce.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from examples.jax.parity.cases import get_case
from examples.jax.parity.cases.native_permanent_magnet_pm4stell import (
    _scale_configuration,
)
from examples.jax.parity.input_bundle import read_input_bundle
from examples.jax.parity.official_reference import load_official_reference

# venv site-packages/tests shadows the repo tests package, so the helpers are
# imported as top-level modules from the tests/ directory.
_TESTS_ROOT = str(Path(__file__).resolve().parents[1])
if _TESTS_ROOT not in sys.path:
    sys.path.append(_TESTS_ROOT)
from mirror_example_constants import (  # noqa: E402
    mirror_example_call_bindings,
    mirror_example_constants,
)
from parity_gpmo_native_child import (  # noqa: E402
    NativeChildResult,
    run_permanent_magnet_native_lane,
)

CASE_ID = "native-permanent-magnet-pm4stell"
MIRROR_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "jax"
    / "2_Intermediate"
    / "permanent_magnet_PM4Stell.py"
)
OFFICIAL_CI = load_official_reference(CASE_ID, variant="ci")


@pytest.fixture(scope="module")
def bounded_native(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[
    NativeChildResult,
    Path,
]:
    """Build the bounded input and run the native lane once, at one thread."""
    root = tmp_path_factory.mktemp("permanent-magnet-pm4stell-bounded")
    input_root = root / "inputs"
    result = run_permanent_magnet_native_lane(
        CASE_ID,
        "bounded",
        input_root,
        root / "native-observation.pkl",
    )
    assert result.omp_num_threads == "1", (
        "the native reference lane must run with OpenMP pinned before libgomp "
        f"starts; the child saw OMP_NUM_THREADS={result.omp_num_threads!r}"
    )
    return result, input_root


def test_exact_permanent_magnet_pm4stell_matches_native_and_jax_cpu(
    bounded_native: tuple[NativeChildResult, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child, input_root = bounded_native
    native = child.observation
    bundle, arrays = read_input_bundle(input_root)

    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_cpu_parity")
    monkeypatch.setenv("SIMSOPT_PRECISION", "fp64")
    monkeypatch.setenv("JAX_ENABLE_X64", "1")
    jax = get_case(CASE_ID).execute("jax-cpu", bundle, arrays)

    assert native.success is True
    assert jax.success is True
    assert native.scale == jax.scale == "bounded"
    assert native.input_fingerprint == jax.input_fingerprint
    assert native.configuration_fingerprint == jax.configuration_fingerprint
    assert native.effective_construction_fingerprint == (
        jax.effective_construction_fingerprint
    )
    assert native.completed_workflow_stages == jax.completed_workflow_stages
    # GPMO reports no convergence status and the configured K is not a measured
    # count: the lanes publish the derived stop reason and null counters.
    assert native.normalized_status == jax.normalized_status == "not_applicable"
    assert native.raw_status == jax.raw_status == "magnet_cap_reached"
    assert native.nit is jax.nit is None
    assert native.nfev is jax.nfev is None
    assert native.njev is jax.njev is None
    assert set(native.values) == set(jax.values)

    for observable in (
        "construction:response_matrix",
        "construction:target",
        "construction:moment_maxima",
        "construction:dipole_grid_xyz",
        "construction:polarization_vectors",
        "initial:moments",
        "initial:residual",
        "initial:objective_sum_squares",
        "final:moments",
        "final:residual",
        "final:objective_sum_squares",
        "final:nonzero_mask",
    ):
        np.testing.assert_allclose(
            jax.values[observable],
            native.values[observable],
            rtol=1.0e-13,
            atol=1.0e-14,
        )

    assert int(np.count_nonzero(native.values["final:nonzero_mask"])) == (
        OFFICIAL_CI.scalar("final:nonzero_count")
    )
    assert float(native.values["final:objective_sum_squares"]) < float(
        native.values["initial:objective_sum_squares"]
    )


def test_bounded_pm4stell_native_lane_reproduces_the_official_ci_capture(
    bounded_native: tuple[NativeChildResult, Path],
) -> None:
    """The official CI run, including its RISING objective history.

    ``history:R2`` is 0.1413 -> 0.00102 -> 0.00340 -> 0.00364: the recorded
    objective is not monotone, and upstream PM4Stell nevertheless keeps the
    GPMO endpoint (``pm_ncsx.m``; only the MUSE script selects the minimum).
    The record's ``final:objective_half_sum_squares`` is the LAST history row,
    which is what pins that rule here.
    """

    child, _input_root = bounded_native
    official_configuration = OFFICIAL_CI.structure("configuration")
    assert isinstance(official_configuration, dict)

    assert child.configuration["ndipoles"] == official_configuration["ndipoles"]
    for name, official_name in (
        ("iterations", "K"),
        ("max_magnets", "max_nMagnets"),
        ("backtracking", "backtracking"),
        ("adjacent_count", "Nadjacent"),
        ("history_count", "nhistory"),
        ("downsample", "downsample"),
        ("nphi", "nphi"),
        ("ntheta", "ntheta"),
    ):
        assert child.configuration[name] == official_configuration[official_name]

    history = child.objective_history
    assert history is not None
    official_history = OFFICIAL_CI.array("history:R2")
    assert history.size == OFFICIAL_CI.scalar("history:length")
    np.testing.assert_allclose(history, official_history, rtol=1.0e-10, atol=0.0)
    assert official_history[2] > official_history[1]
    assert history[2] > history[1]

    values = child.observation.values
    np.testing.assert_allclose(
        0.5 * float(values["final:objective_sum_squares"]),
        OFFICIAL_CI.scalar("final:objective_half_sum_squares"),
        rtol=1.0e-10,
        atol=0.0,
    )
    np.testing.assert_allclose(
        0.5 * float(values["final:objective_sum_squares"]),
        history[-1],
        rtol=1.0e-12,
        atol=0.0,
    )
    assert int(np.count_nonzero(values["final:nonzero_mask"])) == (
        OFFICIAL_CI.scalar("final:nonzero_count")
    )


def test_the_mirror_and_the_case_carry_upstreams_own_configuration() -> None:
    """One authority for the GPMO configuration: the tracked official record.

    The mirror keeps literal constants, as upstream's own script does under
    ``in_github_actions``; what must not drift is their VALUES and where they
    are USED. The values are asserted against upstream's own record -- the
    ``ci`` variant at bounded, the canonical run at shipped scale, the same
    record the parity case now reads -- and then the BINDINGS: which constant
    each keyword of the mirror's GPMO call actually carries.

    The binding half replaces a substring membership test over the unparsed
    source, which passed for a constant that was merely mentioned, was
    satisfied by any superstring, and could not see a value handed to the wrong
    keyword.
    """
    constants = mirror_example_constants(MIRROR_SOURCE)
    for scale, prefix in (("native_default", "NATIVE"), ("bounded", "BOUNDED")):
        configuration = _scale_configuration(scale)
        official = load_official_reference(
            CASE_ID,
            variant="canonical" if scale == "native_default" else "ci",
        ).structure("configuration")
        assert isinstance(official, dict)
        for constant, case_key, official_key in (
            (f"{prefix}_ITERATIONS", "iterations", "K"),
            (f"{prefix}_MAGNET_CAP", "max_magnets", "max_nMagnets"),
            (f"{prefix}_NPHI", "nphi", "nphi"),
            (f"{prefix}_NPHI", "ntheta", "ntheta"),
            (f"{prefix}_DOWNSAMPLE", "downsample", "downsample"),
            # Upstream's ``nBacktracking``, ``nHistory`` and ``nAdjacent`` do
            # not depend on the scale, so one constant serves both.
            ("BACKTRACKING", "backtracking", "backtracking"),
            ("HISTORY_COUNT", "history_count", "nhistory"),
            ("ADJACENT_COUNT", "adjacent_count", "Nadjacent"),
        ):
            assert constants[constant] == official[official_key], constant
            assert configuration[case_key] == official[official_key], case_key

    solve_bindings = mirror_example_call_bindings(
        MIRROR_SOURCE, "solve", "GPMO_ArbVec_backtracking_jax"
    )
    assert solve_bindings["K"] == {"max_steps"}
    assert solve_bindings["Nadjacent"] == {"ADJACENT_COUNT"}
    assert solve_bindings["backtracking"] == {"BACKTRACKING"}
    assert solve_bindings["max_nMagnets"] == {
        "scale",
        "NATIVE_MAGNET_CAP",
        "BOUNDED_MAGNET_CAP",
    }
    # The record period is upstream's ``int(K / nHistory)``, clamped to a budget
    # below ``nHistory``; ``HISTORY_COUNT`` reaches it and nothing else does.
    assert solve_bindings["record_every"] == {
        "gpmo_history_period",
        "min",
        "HISTORY_COUNT",
        "max_steps",
    }
    grid_bindings = mirror_example_call_bindings(
        MIRROR_SOURCE, "_build_grid", "geo_setup_from_famus"
    )
    assert grid_bindings["downsample"] == {
        "scale",
        "NATIVE_DOWNSAMPLE",
        "BOUNDED_DOWNSAMPLE",
    }
    surface_bindings = mirror_example_call_bindings(
        MIRROR_SOURCE, "_build_grid", "from_focus"
    )
    assert surface_bindings["nphi"] == {"scale", "NATIVE_NPHI", "BOUNDED_NPHI"}
    assert surface_bindings["ntheta"] == surface_bindings["nphi"]
    main_bindings = mirror_example_call_bindings(MIRROR_SOURCE, "main", "run_example")
    assert main_bindings["bounded_steps"] == {"BOUNDED_ITERATIONS"}
    assert main_bindings["native_default_steps"] == {"NATIVE_ITERATIONS"}


def test_the_history_and_the_endpoint_come_from_one_solve(
    bounded_native: tuple[NativeChildResult, Path],
) -> None:
    """The compared history and the compared endpoint are the SAME run.

    The child used to run the GPMO solve twice -- once through
    ``execute_arbvec_case`` for the recorded history, once through
    ``case.execute`` for the observation -- and compare both with one official
    record, which is only sound if the solve is deterministic, an assumption no
    test stated or checked. It now solves once and derives both.
    """
    child, _input_root = bounded_native

    assert child.native_solve_count == 1


def test_the_mirror_accepts_every_documented_max_steps() -> None:
    """``--max-steps 1`` runs the example, it does not abort it.

    ``run_example`` accepts any ``--max-steps >= 1``
    (``simsopt_contracts.examples_runtime.run_example``), while upstream's
    GPMO refuses ``nhistory > K``. Deriving the record period from the official
    ``nHistory`` without clamping it to the budget therefore turned every
    budget below HISTORY_COUNT (10) into ``ValueError: nhistory must be less than or
    equal to K``. The shipped script is executed here, at its smallest
    documented budget, because that is the claim being made.
    """

    completed = subprocess.run(
        (sys.executable, "-S", str(MIRROR_SOURCE), "--smoke", "--max-steps", "1"),
        cwd=MIRROR_SOURCE.parents[3],
        env=os.environ,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr[-2000:]
    assert "status=ok" in completed.stdout
