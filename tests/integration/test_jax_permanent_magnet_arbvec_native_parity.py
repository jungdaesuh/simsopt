"""Native-vs-JAX parity for the arbitrary-vector GPMO mirrors: MUSE and PM4Stell.

Both mirrors run ``GPMO_ArbVec_backtracking`` at upstream's own
``in_github_actions`` configuration of ``2_Intermediate/permanent_magnet_MUSE.py``
(K=100, nBacktracking=50, max_nMagnets=20, downsample=100, nphi=ntheta=2,
nHistory=20) and ``2_Intermediate/permanent_magnet_PM4Stell.py`` (K=100,
max_nMagnets=20, nBacktracking=200, nAdjacent=10, nHistory=10, downsample=100,
N=2). The native lane is ``simsopt.solve.GPMO`` on a live grid; the JAX lane is
``GPMO_ArbVec_backtracking_jax`` on that grid staged with
``PermanentMagnetGridJAX.from_cpu`` (the native solve rebinds only ``m`` and
``m_proxy``, which the JAX kernel does not start from), so both lanes consume the
same construction. Nothing is read from stored reference data.

The two providers record on different iteration grids. Native: one row for the
initial state, then ``k in {0, P, 2P, ...}`` and ``k == K - 1`` under ``verbose``
with ``P = int(K / nhistory)``, then one row at the magnet-limit break. JAX
(``record_every=P``): rows ``k in {P - 1, 2P - 1, ...} + {K - 1}``. Both grids
contain the run endpoint.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import ast
import io
import os
import subprocess
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path

import jax
import numpy as np
import pytest
from simsopt.field import BiotSavart, Coil
from simsopt.geo import PermanentMagnetGrid, SurfaceRZFourier
from simsopt.solve import GPMO
from simsopt.util import (
    FocusData,
    FocusPlasmaBnormal,
    discretize_polarizations,
    orientation_phi,
    polarization_axes,
    read_focus_coils,
)
from simsopt_jax.geo.permanent_magnet_grid import PermanentMagnetGridJAX
from simsopt_jax.solve.permanent_magnet import GPMO_ArbVec_backtracking_jax
from simsopt_jax_adapters.examples.gpmo_rules import (
    gpmo_history_period,
    select_minimum_objective_snapshot,
)
from simsopt_jax_adapters.examples.muse import (
    MusePostDiagnostics,
    build_muse_post_geometry,
    muse_post_workflow_policy,
    run_jax_muse_post_workflow,
    run_native_muse_post_workflow,
)
from simsopt_jax_adapters.isolated_kernel import repo_child_pythonpath

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TEST_DATA = REPOSITORY_ROOT / "tests" / "test_files"
MUSE_SURFACE_INPUT = TEST_DATA / "input.muse"
MUSE_MIRROR_SOURCE = (
    REPOSITORY_ROOT / "examples" / "jax" / "2_Intermediate" / "permanent_magnet_MUSE.py"
)
PM4STELL_MIRROR_SOURCE = (
    REPOSITORY_ROOT
    / "examples"
    / "jax"
    / "2_Intermediate"
    / "permanent_magnet_PM4Stell.py"
)
VACUUM_PERMEABILITY = 4.0 * np.pi * 1.0e-7
#: Upstream's ``in_github_actions`` resolution of both scripts.
CI_RESOLUTION = 2
CI_DOWNSAMPLE = 100
#: ``permanent_magnet_MUSE.py`` under ``in_github_actions``.
MUSE_CI_OPTIONS = {
    "K": 100,
    "nhistory": 20,
    "backtracking": 50,
    "Nadjacent": 1,
    "thresh_angle": np.pi,
    "max_nMagnets": 20,
}
MUSE_RADIAL_EXTENT = 0.01
#: ``permanent_magnet_PM4Stell.py`` under ``in_github_actions``.
PM4STELL_CI_OPTIONS = {
    "K": 100,
    "nhistory": 10,
    "backtracking": 200,
    "Nadjacent": 10,
    "thresh_angle": np.pi,
    "max_nMagnets": 20,
}
#: The JAX lane's runtime: the parity backend in FP64.
JAX_LANE_ENVIRONMENT = {
    "SIMSOPT_BACKEND_MODE": "jax_cpu_parity",
    "SIMSOPT_PRECISION": "fp64",
    "JAX_ENABLE_X64": "1",
}
#: Observables both lanes publish, compared at the exact-mirror bucket.
GPMO_OBSERVABLES = (
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
)


@dataclass(frozen=True)
class _LaneRun:
    """One lane's published values plus the history its provider recorded."""

    values: dict[str, np.ndarray]
    #: Upstream's ``R2 = 0.5 * |A m - b|^2`` per recorded row.
    objective_history: np.ndarray
    #: ``(ndipoles, 3, rows)`` physical moments, the C++ ``m_history`` layout.
    moment_history: np.ndarray


@dataclass(frozen=True)
class _NativeCase:
    """A built CPU grid and the native lane solved on it."""

    grid: PermanentMagnetGrid
    native: _LaneRun
    dipole_grid_xyz: np.ndarray
    moment_maxima: np.ndarray


def _face_polarized_muse_grid() -> PermanentMagnetGrid:
    geometry = build_muse_post_geometry(
        MUSE_SURFACE_INPUT,
        TEST_DATA,
        nphi=CI_RESOLUTION,
        ntheta=CI_RESOLUTION,
    )
    surface = geometry.optimization_surface
    field = geometry.coil_field
    field.set_points(surface.gamma().reshape((-1, 3)))
    normal_field = np.sum(
        field.B().reshape((CI_RESOLUTION, CI_RESOLUTION, 3)) * surface.unitnormal(),
        axis=2,
    )
    famus = TEST_DATA / "zot80.focus"
    magnet_data = FocusData(famus, downsample=CI_DOWNSAMPLE)
    axes, polarization_types = polarization_axes(["face"])
    positive_count = len(polarization_types) // 2
    discretize_polarizations(
        magnet_data,
        np.arctan2(magnet_data.oy, magnet_data.ox),
        axes[:positive_count],
        polarization_types[:positive_count],
    )
    with redirect_stdout(io.StringIO()):
        return PermanentMagnetGrid.geo_setup_from_famus(
            surface,
            normal_field,
            famus,
            pol_vectors=np.stack(
                (magnet_data.pol_x, magnet_data.pol_y, magnet_data.pol_z), axis=-1
            ),
            downsample=CI_DOWNSAMPLE,
            dr=MUSE_RADIAL_EXTENT,
        )


def _pm4stell_grid() -> PermanentMagnetGrid:
    plasma = TEST_DATA / "c09r00_B_axis_half_tesla_PM4Stell.plasma"
    famus = TEST_DATA / "magpie_trial104b_PM4Stell.focus"
    surface = SurfaceRZFourier.from_focus(
        plasma, range="half period", nphi=CI_RESOLUTION, ntheta=CI_RESOLUTION
    )
    curves, currents, count = read_focus_coils(
        TEST_DATA / "tf_only_half_tesla_symmetry_baxis_PM4Stell.focus"
    )
    field = BiotSavart([Coil(curves[index], currents[index]) for index in range(count)])
    field.set_points(surface.gamma().reshape((-1, 3)))
    coil_normal = np.sum(
        field.B().reshape((CI_RESOLUTION, CI_RESOLUTION, 3)) * surface.unitnormal(),
        axis=2,
    )
    plasma_normal = FocusPlasmaBnormal(plasma).bnormal_grid(
        CI_RESOLUTION, CI_RESOLUTION, "half period"
    )
    magnet_data = FocusData(famus, downsample=CI_DOWNSAMPLE)
    # The positive half of each official polarization family.
    families = [polarization_axes([name]) for name in ("face", "fe_ftri", "fc_ftri")]
    axes = np.concatenate([axes[: len(types) // 2] for axes, types in families])
    types = np.concatenate(
        [types[: len(types) // 2] + index for index, (_, types) in enumerate(families)]
    )
    orientation = orientation_phi(TEST_DATA / "magpie_trial104b_corners_PM4Stell.csv")
    discretize_polarizations(
        magnet_data, orientation[: magnet_data.nMagnets], axes, types
    )
    with redirect_stdout(io.StringIO()):
        return PermanentMagnetGrid.geo_setup_from_famus(
            surface,
            plasma_normal + coil_normal,
            famus,
            pol_vectors=np.stack(
                (magnet_data.pol_x, magnet_data.pol_y, magnet_data.pol_z), axis=-1
            ),
            m_maxima=5.0 / VACUUM_PERMEABILITY,
            downsample=CI_DOWNSAMPLE,
        )


def _values(
    construction: dict[str, np.ndarray],
    final_moments: np.ndarray,
    final_residual: np.ndarray,
) -> dict[str, np.ndarray]:
    """One lane's construction, start state and endpoint, by observable name."""
    response = construction["response_matrix"]
    initial_moments = construction["initial_moments"]
    initial_residual = response @ initial_moments.reshape(-1) - construction["target"]
    return {
        "construction:response_matrix": response,
        "construction:target": construction["target"],
        "construction:moment_maxima": construction["moment_maxima"],
        "construction:dipole_grid_xyz": construction["dipole_grid_xyz"],
        "construction:polarization_vectors": construction["polarization_vectors"],
        "initial:moments": initial_moments,
        "initial:residual": initial_residual,
        "initial:objective_sum_squares": np.asarray(
            np.vdot(initial_residual, initial_residual), dtype=np.float64
        ),
        "final:moments": final_moments,
        "final:residual": final_residual,
        "final:objective_sum_squares": np.asarray(
            np.vdot(final_residual, final_residual), dtype=np.float64
        ),
        "final:nonzero_mask": np.linalg.norm(final_moments, axis=1) != 0.0,
    }


def _host_construction(grid: PermanentMagnetGrid) -> dict[str, np.ndarray]:
    ndipoles = int(grid.ndipoles)
    return {
        "response_matrix": np.array(grid.A_obj, dtype=np.float64, copy=True),
        "target": np.array(grid.b_obj, dtype=np.float64, copy=True),
        "moment_maxima": np.array(grid.m_maxima, dtype=np.float64, copy=True).reshape(
            (ndipoles,)
        ),
        "dipole_grid_xyz": np.array(grid.dipole_grid_xyz, dtype=np.float64, copy=True),
        "polarization_vectors": np.array(grid.pol_vectors, dtype=np.float64, copy=True),
        "initial_moments": np.array(grid.m0, dtype=np.float64, copy=True).reshape(
            (ndipoles, 3)
        ),
    }


def _solve_native_case(
    grid: PermanentMagnetGrid, options: dict[str, float]
) -> _NativeCase:
    """Solve the grid natively with the official kwargs."""
    construction = _host_construction(grid)
    # Official kwargs: ``initialize_default_kwargs('GPMO')`` sets verbose=True.
    with redirect_stdout(io.StringIO()):
        errors, _, moment_history = GPMO(
            grid,
            "ArbVec_backtracking",
            dipole_grid_xyz=grid.dipole_grid_xyz,
            verbose=True,
            **options,
        )
    objective_history = np.asarray(errors, dtype=np.float64)
    final_moments = np.asarray(grid.m, dtype=np.float64).reshape((-1, 3))
    final_residual = (
        construction["response_matrix"] @ final_moments.reshape(-1)
        - construction["target"]
    )
    return _NativeCase(
        grid=grid,
        native=_LaneRun(
            values=_values(construction, final_moments, final_residual),
            objective_history=objective_history,
            moment_history=np.ascontiguousarray(
                np.asarray(moment_history, dtype=np.float64)[
                    :, :, : objective_history.size
                ]
            ),
        ),
        dipole_grid_xyz=construction["dipole_grid_xyz"],
        moment_maxima=construction["moment_maxima"],
    )


def _solve_jax_lane(grid: PermanentMagnetGrid, options: dict[str, float]) -> _LaneRun:
    staged = PermanentMagnetGridJAX.from_cpu(grid)
    device_result = GPMO_ArbVec_backtracking_jax(
        staged,
        K=int(options["K"]),
        reg_l2=0.0,
        Nadjacent=int(options["Nadjacent"]),
        backtracking=int(options["backtracking"]),
        thresh_angle=float(options["thresh_angle"]),
        max_nMagnets=int(options["max_nMagnets"]),
        # Mirror of upstream's print period int(K / nhistory).
        record_every=gpmo_history_period(
            iterations=int(options["K"]), history_count=int(options["nhistory"])
        ),
    )
    host_result, construction = jax.device_get(
        (
            device_result,
            {
                "response_matrix": staged.A_obj,
                "target": staged.b_obj,
                "moment_maxima": staged.m_maxima,
                "dipole_grid_xyz": staged.dipole_grid_xyz,
                "polarization_vectors": staged.pol_vectors,
                "initial_moments": staged.m0,
            },
        )
    )
    # ``residual_history`` is ``sum(r * r)``; upstream records half of it.
    # ``m_history`` arrives as ``(rows, ndipoles, 3)``.
    return _LaneRun(
        values=_values(
            {
                name: np.asarray(value, dtype=np.float64)
                for name, value in construction.items()
            },
            np.asarray(host_result.m, dtype=np.float64),
            np.asarray(host_result.residual, dtype=np.float64),
        ),
        objective_history=0.5
        * np.asarray(host_result.residual_history, dtype=np.float64).reshape(-1),
        moment_history=np.ascontiguousarray(
            np.transpose(np.asarray(host_result.m_history, dtype=np.float64), (1, 2, 0))
        ),
    )


def _post_diagnostics(
    workflow,
    moments: np.ndarray,
    dipole_grid_xyz: np.ndarray,
    moment_maxima: np.ndarray,
) -> MusePostDiagnostics:
    """The official post-GPMO half of MUSE, in the given lane's implementation."""
    return workflow(
        build_muse_post_geometry(
            MUSE_SURFACE_INPUT,
            TEST_DATA,
            nphi=CI_RESOLUTION,
            ntheta=CI_RESOLUTION,
        ),
        muse_post_workflow_policy("bounded"),
        moments,
        dipole_grid_xyz,
        moment_maxima,
        coordinate_flag="cartesian",
    )


def _post_values(diagnostics: MusePostDiagnostics) -> dict[str, np.ndarray]:
    return {
        "final:post_dipole_squared_flux": np.asarray(
            diagnostics.dipole_squared_flux, dtype=np.float64
        ),
        "final:post_total_magnet_volume": np.asarray(
            diagnostics.total_magnet_volume, dtype=np.float64
        ),
        "final:post_coil_only_trace_status": np.asarray(
            diagnostics.trace_statuses, dtype=np.int64
        ),
        "final:post_coil_only_trace_hit_counts": np.asarray(
            diagnostics.trace_hit_counts, dtype=np.int64
        ),
    }


def _assert_lanes_match(
    native: dict[str, np.ndarray],
    jax_values: dict[str, np.ndarray],
    observables: tuple[str, ...],
) -> None:
    for observable in observables:
        np.testing.assert_allclose(
            jax_values[observable],
            native[observable],
            rtol=1.0e-13,
            atol=1.0e-14,
            err_msg=observable,
        )


def _call_bindings(path: Path, function: str, callee: str) -> dict[str, frozenset[str]]:
    """Which names each keyword of the one ``callee`` call in ``function`` carries.

    Every keyword is reduced to the set of names its expression depends on,
    expanded through the function's own local assignments (``magnet_cap =
    NATIVE_MAGNET_CAP if native_scale else BOUNDED_MAGNET_CAP`` resolves to the
    two constants). A name that is not a local assignment target is returned as
    itself. A binding read from an ambiguous match is not evidence, so exactly
    one call must match.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function
    )
    calls = [
        node
        for node in ast.walk(function_node)
        if isinstance(node, ast.Call) and _callee_name(node) == callee
    ]
    assert len(calls) == 1, f"{path.name}:{function} calls {callee} {len(calls)}x"
    assignments = {
        node.targets[0].id: _names(node.value)
        for node in ast.walk(function_node)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }
    return {
        keyword.arg: _expand(_names(keyword.value), assignments)
        for keyword in calls[0].keywords
        if keyword.arg is not None
    }


def _callee_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


def _names(node: ast.expr) -> frozenset[str]:
    return frozenset(
        child.id for child in ast.walk(node) if isinstance(child, ast.Name)
    )


def _expand(
    names: frozenset[str], assignments: dict[str, frozenset[str]]
) -> frozenset[str]:
    """Replace every local assignment target by the names it was assigned."""
    resolved: set[str] = set()
    pending = list(names)
    seen: set[str] = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        if name in assignments:
            pending.extend(assignments[name])
        else:
            resolved.add(name)
    return frozenset(resolved)


def _run_mirror_at_one_step(mirror: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        (sys.executable, "-S", str(mirror), "--smoke", "--max-steps", "1"),
        cwd=mirror.parents[3],
        # ``-S`` drops site-packages: the sources, the loaded kernel and the
        # dependency root are passed explicitly.
        env={
            **os.environ,
            "PYTHONPATH": repo_child_pythonpath(
                mirror.parents[3], os.environ.get("PYTHONPATH")
            ),
        },
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.fixture(scope="module")
def muse_native() -> tuple[_NativeCase, dict[str, np.ndarray]]:
    """The bounded MUSE grid, its native solve and the native post half, once."""
    case = _solve_native_case(_face_polarized_muse_grid(), MUSE_CI_OPTIONS)
    # Official ``permanent_magnet_MUSE.py:184-185``: the post metrics are
    # computed from the recorded snapshot with the smallest ``R2``.
    selection = select_minimum_objective_snapshot(
        case.native.objective_history, case.native.moment_history
    )
    post = _post_diagnostics(
        run_native_muse_post_workflow,
        selection.moments,
        case.dipole_grid_xyz,
        case.moment_maxima,
    )
    return case, _post_values(post)


@pytest.fixture(scope="module")
def pm4stell_native() -> _NativeCase:
    return _solve_native_case(_pm4stell_grid(), PM4STELL_CI_OPTIONS)


def test_exact_permanent_magnet_muse_matches_native_and_jax_cpu(
    muse_native: tuple[_NativeCase, dict[str, np.ndarray]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case, native_post = muse_native
    native = case.native
    for name, value in JAX_LANE_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    jax_lane = _solve_jax_lane(case.grid, MUSE_CI_OPTIONS)
    jax_selection = select_minimum_objective_snapshot(
        jax_lane.objective_history, jax_lane.moment_history
    )
    jax_post = _post_values(
        _post_diagnostics(
            run_jax_muse_post_workflow,
            jax_selection.moments,
            case.dipole_grid_xyz,
            case.moment_maxima,
        )
    )

    assert set(native.values) == set(jax_lane.values)
    _assert_lanes_match(native.values, jax_lane.values, GPMO_OBSERVABLES)
    assert set(native_post) == set(jax_post)
    _assert_lanes_match(native_post, jax_post, tuple(native_post))

    # Upstream's CI run of this configuration stops at its magnet cap.
    assert (
        int(np.count_nonzero(native.values["final:nonzero_mask"]))
        == (MUSE_CI_OPTIONS["max_nMagnets"])
    )
    # Reported diagnostic, not a success gate (official PM4Stell rises).
    assert float(native.values["final:objective_sum_squares"]) < float(
        native.values["initial:objective_sum_squares"]
    )
    assert np.array_equal(
        native_post["final:post_coil_only_trace_status"],
        np.full(
            muse_post_workflow_policy("bounded").fieldline_count,
            -1,
            dtype=np.int64,
        ),
    )
    # Both lanes select the same snapshot and it is the run endpoint, because
    # the recorded objective decreases monotonically here. The post metrics
    # above are therefore the endpoint's.
    for lane in (native, jax_lane):
        selection = select_minimum_objective_snapshot(
            lane.objective_history, lane.moment_history
        )
        assert np.array_equal(
            selection.moments,
            np.asarray(lane.values["final:moments"], dtype=np.float64),
        )
    native_selection = select_minimum_objective_snapshot(
        native.objective_history, native.moment_history
    )
    selected_residual = (
        native.values["construction:response_matrix"]
        @ native_selection.moments.reshape(-1)
        - native.values["construction:target"]
    )
    np.testing.assert_allclose(
        float(np.vdot(selected_residual, selected_residual)),
        float(native.values["final:objective_sum_squares"]),
        rtol=0.0,
        atol=0.0,
    )


def test_muse_trace_times_agree_to_within_one_accepted_step(
    muse_native: tuple[_NativeCase, dict[str, np.ndarray]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bound on the cross-lane trace-time gap, not a required divergence.

    The two integrators localize the level-set event differently. That is a
    reason for a *bound*, not for asserting that the lanes disagree: upstream
    fires a stopping criterion on the accepted post-step state and keeps the
    pre-crossing row (``simsoptpp/tracing.cpp:387-443``), so the stop time is
    only defined to within one accepted step.

    The bound is the NATIVE integrator's own last accepted step -- upstream's
    tracer is the reference that defines both the event time and the resolution
    it is defined to. It is never the larger of the two lanes: that would let a
    regression in the JAX tracer widen the bound it is being judged against.
    """

    case, _native_post = muse_native
    policy = muse_post_workflow_policy("bounded")
    moments = np.asarray(case.native.values["final:moments"], dtype=np.float64)

    native_post = _post_diagnostics(
        run_native_muse_post_workflow,
        moments,
        case.dipole_grid_xyz,
        case.moment_maxima,
    )
    for name, value in JAX_LANE_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    jax_post = _post_diagnostics(
        run_jax_muse_post_workflow,
        moments,
        case.dipole_grid_xyz,
        case.moment_maxima,
    )

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


def test_exact_permanent_magnet_pm4stell_matches_native_and_jax_cpu(
    pm4stell_native: _NativeCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native = pm4stell_native.native
    for name, value in JAX_LANE_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    jax_lane = _solve_jax_lane(pm4stell_native.grid, PM4STELL_CI_OPTIONS)

    assert set(native.values) == set(jax_lane.values)
    _assert_lanes_match(native.values, jax_lane.values, GPMO_OBSERVABLES)

    # Upstream's CI run of this configuration stops at its magnet cap.
    assert (
        int(np.count_nonzero(native.values["final:nonzero_mask"]))
        == (PM4STELL_CI_OPTIONS["max_nMagnets"])
    )
    assert float(native.values["final:objective_sum_squares"]) < float(
        native.values["initial:objective_sum_squares"]
    )


def test_bounded_pm4stell_native_history_rises_and_ends_at_the_endpoint(
    pm4stell_native: _NativeCase,
) -> None:
    """The CI run's RISING objective history, and the endpoint is its last row.

    The recorded objective is not monotone, and upstream PM4Stell nevertheless
    keeps the GPMO endpoint (``pm_ncsx.m``; only the MUSE script selects the
    minimum). The endpoint's half objective is the LAST history row, which is
    what pins that rule here.
    """

    native = pm4stell_native.native
    history = native.objective_history
    assert history[2] > history[1]
    np.testing.assert_allclose(
        0.5 * float(native.values["final:objective_sum_squares"]),
        history[-1],
        rtol=1.0e-12,
        atol=0.0,
    )


def test_the_muse_mirror_binds_upstreams_configuration_to_its_gpmo_call() -> None:
    """Which constant each keyword of the mirror's GPMO call actually carries.

    This replaces a substring membership test over the unparsed source, which
    passed for a constant that was merely mentioned, was satisfied by any
    superstring (``"HISTORY_COUNT" in source`` also matches
    ``BOUNDED_HISTORY_COUNT``), and could not see a value handed to the wrong
    keyword.
    """
    solve_bindings = _call_bindings(
        MUSE_MIRROR_SOURCE, "solve", "GPMO_ArbVec_backtracking_jax"
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
    grid_bindings = _call_bindings(
        MUSE_MIRROR_SOURCE, "_build_grid", "geo_setup_from_famus"
    )
    assert grid_bindings["downsample"] == {
        "scale",
        "NATIVE_DOWNSAMPLE",
        "BOUNDED_DOWNSAMPLE",
    }
    assert grid_bindings["dr"] == {"RADIAL_EXTENT"}
    geometry_bindings = _call_bindings(
        MUSE_MIRROR_SOURCE, "_build_grid", "build_muse_post_geometry"
    )
    assert geometry_bindings["nphi"] == {"scale", "NATIVE_NPHI", "BOUNDED_NPHI"}
    assert geometry_bindings["ntheta"] == geometry_bindings["nphi"]
    main_bindings = _call_bindings(MUSE_MIRROR_SOURCE, "main", "run_example")
    assert main_bindings["bounded_steps"] == {"BOUNDED_ITERATIONS"}
    assert main_bindings["native_default_steps"] == {"NATIVE_ITERATIONS"}


def test_the_pm4stell_mirror_binds_upstreams_configuration_to_its_gpmo_call() -> None:
    """Which constant each keyword of the mirror's GPMO call actually carries."""
    solve_bindings = _call_bindings(
        PM4STELL_MIRROR_SOURCE, "solve", "GPMO_ArbVec_backtracking_jax"
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
    grid_bindings = _call_bindings(
        PM4STELL_MIRROR_SOURCE, "_build_grid", "geo_setup_from_famus"
    )
    assert grid_bindings["downsample"] == {
        "scale",
        "NATIVE_DOWNSAMPLE",
        "BOUNDED_DOWNSAMPLE",
    }
    surface_bindings = _call_bindings(
        PM4STELL_MIRROR_SOURCE, "_build_grid", "from_focus"
    )
    assert surface_bindings["nphi"] == {"scale", "NATIVE_NPHI", "BOUNDED_NPHI"}
    assert surface_bindings["ntheta"] == surface_bindings["nphi"]
    main_bindings = _call_bindings(PM4STELL_MIRROR_SOURCE, "main", "run_example")
    assert main_bindings["bounded_steps"] == {"BOUNDED_ITERATIONS"}
    assert main_bindings["native_default_steps"] == {"NATIVE_ITERATIONS"}


@pytest.mark.parametrize(
    "mirror",
    (MUSE_MIRROR_SOURCE, PM4STELL_MIRROR_SOURCE),
    ids=("muse", "pm4stell"),
)
def test_the_mirror_accepts_every_documented_max_steps(mirror: Path) -> None:
    """``--max-steps 1`` runs the example, it does not abort it.

    ``run_example`` accepts any ``--max-steps >= 1``
    (``simsopt_contracts.examples_runtime.run_example``), while upstream's
    GPMO refuses ``nhistory > K``. Deriving the record period from the official
    ``nHistory`` without clamping it to the budget therefore turned every
    budget below HISTORY_COUNT into ``ValueError: nhistory must be less than or
    equal to K``. The shipped script is executed here, at its smallest
    documented budget, because that is the claim being made.
    """

    completed = _run_mirror_at_one_step(mirror)

    assert completed.returncode == 0, completed.stderr[-2000:]
    assert "status=ok" in completed.stdout
