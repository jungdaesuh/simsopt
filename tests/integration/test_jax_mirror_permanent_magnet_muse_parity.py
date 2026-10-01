"""Exact parity for the ``2_Intermediate/permanent_magnet_MUSE.py`` mirror.

The bounded configuration is the official ``in_github_actions`` configuration
of the script (K=100, nBacktracking=50, max_nMagnets=20, downsample=100,
nphi=ntheta=2, nHistory=20), so the native lane has an official counterpart:
the ``ci`` variant of ``examples/jax/parity/official_reference``, which is
upstream's own CI run of the same script at ``9e027eac3``. That fixture is
tracked, so this file's upstream evidence runs on a clean checkout.

The native lane -- and the construction that feeds it -- runs in a child
process at ``OMP_NUM_THREADS=1``: the official numbers were measured at one
thread and ``OMP_NUM_THREADS`` is read when libgomp starts, so an in-process
pin cannot undo the pytest process team (``tests/conftest.py`` pins nothing).
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from examples.jax.parity.cases import get_case
from examples.jax.parity.cases.native_permanent_magnet_muse import (
    SURFACE_INPUT,
    TEST_DATA,
    _scale_configuration,
)
from examples.jax.parity.input_bundle import read_input_bundle
from examples.jax.parity.official_reference import load_official_reference
from simsopt_jax_adapters.examples.muse import (
    build_muse_post_geometry,
    muse_post_workflow_policy,
    run_jax_muse_post_workflow,
    run_native_muse_post_workflow,
)
from simsopt_jax_adapters.isolated_kernel import repo_child_pythonpath

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

CASE_ID = "native-permanent-magnet-muse"
MIRROR_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "jax"
    / "2_Intermediate"
    / "permanent_magnet_MUSE.py"
)
#: Upstream's own CI run of the official script: the bounded counterpart.
OFFICIAL_CI = load_official_reference(CASE_ID, variant="ci")


@pytest.fixture(scope="module")
def bounded_native(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[
    NativeChildResult,
    Path,
]:
    """Build the bounded input and run the native lane once, at one thread."""
    root = tmp_path_factory.mktemp("permanent-magnet-muse-bounded")
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


def test_exact_permanent_magnet_muse_matches_native_and_jax_cpu(
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
    assert native.normalized_status == jax.normalized_status == "not_applicable"
    assert native.raw_status == jax.raw_status == "magnet_cap_reached"
    assert native.nit is jax.nit is None
    assert native.nfev is jax.nfev is None
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
        "final:post_dipole_squared_flux",
        "final:post_total_magnet_volume",
        "final:post_coil_only_trace_status",
        "final:post_coil_only_trace_hit_counts",
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
    # Reported diagnostic, not a success gate (official PM4Stell rises).
    assert float(native.values["final:objective_sum_squares"]) < float(
        native.values["initial:objective_sum_squares"]
    )
    # Both lanes run the official post-GPMO half; the stage contract carries it.
    assert native.completed_workflow_stages[-2:] == (
        "evaluate_post_optimization_dipole_squared_flux_and_total_volume",
        "trace_interpolated_tf_coil_fieldlines",
    )
    assert np.array_equal(
        native.values["final:post_coil_only_trace_status"],
        np.full(
            muse_post_workflow_policy("bounded").fieldline_count,
            -1,
            dtype=np.int64,
        ),
    )
    # Both lanes select the same snapshot and it is the run endpoint, because
    # the recorded objective decreases monotonically here (official CI
    # ``history:R2``). The post metrics above are therefore the official ones.
    for observation in (native, jax):
        assert bool(observation.values["final:selected_history_equals_endpoint"]) is (
            True
        )
    np.testing.assert_allclose(
        float(native.values["final:selected_history_objective_sum_squares"]),
        float(native.values["final:objective_sum_squares"]),
        rtol=0.0,
        atol=0.0,
    )


def test_muse_trace_times_agree_to_within_one_accepted_step(
    bounded_native: tuple[NativeChildResult, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bound on the cross-lane trace-time gap, not a required divergence.

    ``final:post_coil_only_trace_final_times`` is declared applicable=false in
    the parity manifest because the two integrators localize the level-set
    event differently. That is a reason for a *bound*, not for asserting that
    the lanes disagree: upstream fires a stopping criterion on the accepted
    post-step state and keeps the pre-crossing row
    (``simsoptpp/tracing.cpp:387-443``), so the stop time is only defined to
    within one accepted step.

    The bound is the NATIVE integrator's own last accepted step -- upstream's
    tracer is the reference that defines both the event time and the resolution
    it is defined to. It is never the larger of the two lanes: that would let a
    regression in the JAX tracer widen the bound it is being judged against.
    Measured here: gap 1.78e-3 on line 3 against a native step of 6.64e-3, and
    exact equality on the two lines that stop at t=0.
    """

    child, input_root = bounded_native
    _bundle, arrays = read_input_bundle(input_root)
    policy = muse_post_workflow_policy("bounded")
    moments = np.asarray(
        child.observation.values["final:moments"],
        dtype=np.float64,
    )

    def post(workflow):
        return workflow(
            build_muse_post_geometry(
                SURFACE_INPUT,
                TEST_DATA,
                nphi=2,
                ntheta=2,
            ),
            policy,
            moments,
            arrays["dipole_grid_xyz"],
            arrays["moment_maxima"],
            coordinate_flag="cartesian",
        )

    native_post = post(run_native_muse_post_workflow)
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_cpu_parity")
    monkeypatch.setenv("SIMSOPT_PRECISION", "fp64")
    monkeypatch.setenv("JAX_ENABLE_X64", "1")
    jax_post = post(run_jax_muse_post_workflow)

    assert native_post.trace_statuses == jax_post.trace_statuses
    assert native_post.trace_hit_counts == jax_post.trace_hit_counts
    native_times = np.asarray(native_post.trace_final_times, dtype=np.float64)
    jax_times = np.asarray(jax_post.trace_final_times, dtype=np.float64)
    bound = np.asarray(native_post.trace_final_steps, dtype=np.float64)
    # A step is a step: one that reached the trace horizon would make the
    # assertion below unfalsifiable, so the bound is checked before it is used.
    assert np.all(np.isfinite(bound))
    assert np.all(bound < float(policy.fieldline_tmax))
    assert np.all(np.abs(native_times - jax_times) <= bound)


def test_bounded_muse_native_lane_reproduces_the_official_ci_capture(
    bounded_native: tuple[NativeChildResult, Path],
) -> None:
    """The bounded configuration IS the official CI configuration.

    Agreement between the two lanes is not evidence of correctness; agreement
    with the official run is. The ``ci`` record was produced by the official
    script on an independent official build at one OpenMP thread, so every
    number below is upstream's.
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

    # The recorded history is upstream's: one initial row, the verbose rows at
    # k in {0, 5, 10, 15} and the magnet-limit row.
    history = child.objective_history
    assert history is not None
    assert history.size == OFFICIAL_CI.scalar("history:length")
    np.testing.assert_allclose(
        history,
        OFFICIAL_CI.array("history:R2"),
        rtol=1.0e-12,
        atol=0.0,
    )

    values = child.observation.values
    np.testing.assert_allclose(
        0.5 * float(values["final:objective_sum_squares"]),
        OFFICIAL_CI.scalar("final:objective_half_sum_squares"),
        rtol=1.0e-12,
        atol=0.0,
    )
    assert int(np.count_nonzero(values["final:nonzero_mask"])) == (
        OFFICIAL_CI.scalar("final:nonzero_count")
    )
    # The post metrics upstream computes from the SELECTED snapshot.
    np.testing.assert_allclose(
        float(values["final:post_dipole_squared_flux"]),
        OFFICIAL_CI.scalar("final:squared_flux_dipoles_on_plot_surface"),
        rtol=1.0e-12,
        atol=0.0,
    )
    np.testing.assert_allclose(
        float(values["final:post_total_magnet_volume"]),
        OFFICIAL_CI.scalar("final:total_volume"),
        rtol=1.0e-12,
        atol=0.0,
    )


def test_the_mirror_and_the_case_carry_upstreams_own_configuration() -> None:
    """One authority for the GPMO configuration: the tracked official record.

    The mirror keeps literal constants, as upstream's own script does under
    ``in_github_actions``; what must not drift is their VALUES and where they
    are USED. So this asserts the values against upstream's record (the ``ci``
    variant at bounded, the canonical run at shipped scale) -- the same record
    the parity case now reads -- and then asserts the BINDINGS: which constant
    each keyword of the mirror's GPMO call actually carries.

    The binding half replaces a substring membership test over the unparsed
    source, which passed for a constant that was merely mentioned, was
    satisfied by any superstring (``"HISTORY_COUNT" in source`` also matches
    ``BOUNDED_HISTORY_COUNT``), and could not see a value handed to the wrong
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
            (f"{prefix}_BACKTRACKING", "backtracking", "backtracking"),
            (f"{prefix}_MAGNET_CAP", "max_magnets", "max_nMagnets"),
            (f"{prefix}_NPHI", "nphi", "nphi"),
            (f"{prefix}_NPHI", "ntheta", "ntheta"),
            (f"{prefix}_DOWNSAMPLE", "downsample", "downsample"),
            ("HISTORY_COUNT", "history_count", "nhistory"),
            ("ADJACENT_COUNT", "adjacent_count", "Nadjacent"),
            ("RADIAL_EXTENT", "radial_extent", "dr"),
        ):
            assert constants[constant] == official[official_key], constant
            assert configuration[case_key] == official[official_key], case_key

    solve_bindings = mirror_example_call_bindings(
        MIRROR_SOURCE, "solve", "GPMO_ArbVec_backtracking_jax"
    )
    assert solve_bindings["K"] == {"max_steps"}
    assert solve_bindings["Nadjacent"] == {"ADJACENT_COUNT"}
    assert solve_bindings["backtracking"] == {
        "scale",
        "NATIVE_BACKTRACKING",
        "BOUNDED_BACKTRACKING",
    }
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
    assert grid_bindings["dr"] == {"RADIAL_EXTENT"}
    geometry_bindings = mirror_example_call_bindings(
        MIRROR_SOURCE, "_build_grid", "build_muse_post_geometry"
    )
    assert geometry_bindings["nphi"] == {"scale", "NATIVE_NPHI", "BOUNDED_NPHI"}
    assert geometry_bindings["ntheta"] == geometry_bindings["nphi"]
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
    budget below HISTORY_COUNT (20) into ``ValueError: nhistory must be less than or
    equal to K``. The shipped script is executed here, at its smallest
    documented budget, because that is the claim being made.
    """

    completed = subprocess.run(
        (sys.executable, "-S", str(MIRROR_SOURCE), "--smoke", "--max-steps", "1"),
        cwd=MIRROR_SOURCE.parents[3],
        # ``-S`` drops site-packages: the sources, the loaded kernel and the
        # dependency root are passed explicitly.
        env={
            **os.environ,
            "PYTHONPATH": repo_child_pythonpath(
                MIRROR_SOURCE.parents[3], os.environ.get("PYTHONPATH")
            ),
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr[-2000:]
    assert "status=ok" in completed.stdout
