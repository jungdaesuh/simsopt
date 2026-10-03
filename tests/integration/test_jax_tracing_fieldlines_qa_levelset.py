"""The ``1_Simple/tracing_fieldlines_QA.py`` workflow: JAX against live native SIMSOPT.

Both lanes trace the same three field lines through the interpolant of the
optimized QA coils, with the QA boundary as a level-set stop, at a small scale:
natively with ``simsopt.field.tracing.compute_fieldlines`` over
``InterpolatedField``, and with ``compute_fieldlines_with_status`` over
``InterpolatedFieldJAX(BiotSavartJAX)``. The native run is the reference.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import ast
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import pytest
import simsopt
from simsopt.field import (
    InterpolatedField,
    LevelsetStoppingCriterion,
    SurfaceClassifier,
)
from simsopt.field.tracing import compute_fieldlines
from simsopt.geo import SurfaceRZFourier
from simsopt_jax_adapters.field import tracing as tracing_adapter
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.field.interpolated import InterpolatedFieldJAX
from simsopt_jax_adapters.field.toroidal_field import ToroidalFieldJAX

REPO_ROOT = Path(__file__).resolve().parents[2]
SURFACE_INPUT = REPO_ROOT / "tests/test_files/input.LandremanPaul2021_QA"
FIELD_INPUT = REPO_ROOT / "examples/1_Simple/inputs/biot_savart_opt.json"

SURFACE_NPHI = 24
SURFACE_NTHETA = 10
GRID_SIZE = 5
INTERPOLATION_DEGREE = 2
TMAX = 50.0
#: The official integrator tolerance, below double epsilon.
TOLERANCE = 1.0e-16
CLASSIFIER_H = 0.08
CLASSIFIER_ORDER = 2
SKIP_DISTANCE = -0.05
INITIAL_RADII = np.linspace(1.2125346, 1.295, 3)
INITIAL_STATES = np.column_stack((INITIAL_RADII, np.zeros(3), np.zeros(3)))

Values = dict[str, np.ndarray]


@dataclass(frozen=True)
class _Lane:
    """One lane's published observables and the raw tracer output behind them."""

    values: Values
    trajectories: list[np.ndarray]
    event_rows: list[np.ndarray]


def _jax_parity_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_cpu_parity")
    monkeypatch.setenv("SIMSOPT_PRECISION", "fp64")
    monkeypatch.setenv("JAX_ENABLE_X64", "1")


def _qa_objects():
    surface = SurfaceRZFourier.from_vmec_input(
        str(SURFACE_INPUT),
        nphi=SURFACE_NPHI,
        ntheta=SURFACE_NTHETA,
        range="full torus",
    )
    return surface, simsopt.load(FIELD_INPUT)


def _geometry():
    surface, native_field = _qa_objects()
    classifier = SurfaceClassifier(surface, h=CLASSIFIER_H, p=CLASSIFIER_ORDER)
    surface_points = surface.gamma()
    radii = np.linalg.norm(surface_points[:, :, :2], axis=2)
    heights = surface_points[:, :, 2]

    def skip(
        radial_values: np.ndarray, phi_values: np.ndarray, height_values: np.ndarray
    ) -> np.ndarray:
        points = np.column_stack((radial_values, phi_values, height_values))
        return (classifier.evaluate_rphiz(points) < SKIP_DISTANCE).reshape(-1)

    interpolation_arguments = (
        INTERPOLATION_DEGREE,
        (float(np.min(radii)), float(np.max(radii)), GRID_SIZE),
        (0.0, 2.0 * np.pi / surface.nfp, 2 * GRID_SIZE),
        (0.0, float(np.max(heights)), GRID_SIZE // 2),
        True,
    )
    return surface, native_field, classifier, skip, interpolation_arguments


def _phi_planes(nfp: int) -> tuple[float, ...]:
    return tuple(index * 0.5 * np.pi / nfp for index in range(4))


def _values(
    *,
    surface,
    native_field,
    surface_field: np.ndarray,
    direct_surface_field: np.ndarray,
    trajectories: list[np.ndarray],
    phi_hits: list[np.ndarray],
    statuses: np.ndarray,
    classifier: SurfaceClassifier,
) -> Values:
    final_states = np.stack([trajectory[-1, 1:4] for trajectory in trajectories])
    return {
        "construction:surface_dofs": np.asarray(surface.local_full_x, dtype=np.float64),
        "construction:field_dofs": np.asarray(native_field.x, dtype=np.float64),
        "initial:states": INITIAL_STATES,
        "interpolation:surface_field": surface_field,
        "interpolation:relative_error": np.asarray(
            np.linalg.norm(surface_field - direct_surface_field)
            / np.linalg.norm(direct_surface_field),
            dtype=np.float64,
        ),
        "final:states": final_states,
        "final:times": np.asarray(
            [trajectory[-1, 0] for trajectory in trajectories], dtype=np.float64
        ),
        "final:status": np.asarray(statuses, dtype=np.int64),
        "final:levelset_distance": np.asarray(
            classifier.evaluate_xyz(final_states), dtype=np.float64
        ).reshape(-1),
        "poincare:counts": np.asarray(
            [hits.shape[0] for hits in phi_hits], dtype=np.int64
        ),
        "poincare:positions": np.concatenate(
            [np.asarray(hits[:, 2:5], dtype=np.float64) for hits in phi_hits], axis=0
        ),
    }


def _native_statuses(
    trajectories: list[np.ndarray], phi_hits: list[np.ndarray]
) -> np.ndarray:
    """Native ``compute_fieldlines`` reports no status; derive the core's from its rows.

    ``0`` reached ``tmax``, ``-1 - i`` criterion ``i`` fired (its event row),
    ``1`` neither.
    """
    return np.asarray(
        [
            0
            if np.isclose(trajectory[-1, 0], TMAX, rtol=0.0, atol=1.0e-10)
            else int(hits[hits[:, 1] < 0][-1, 1])
            if hits.ndim == 2 and np.any(hits[:, 1] < 0)
            else 1
            for trajectory, hits in zip(trajectories, phi_hits, strict=True)
        ],
        dtype=np.int64,
    )


def _native_lane() -> _Lane:
    surface, native_field, classifier, skip, interpolation_arguments = _geometry()
    interpolated = InterpolatedField(
        native_field,
        *interpolation_arguments,
        nfp=surface.nfp,
        stellsym=True,
        skip=skip,
    )
    surface_points = np.asarray(surface.gamma(), dtype=np.float64).reshape((-1, 3))
    native_field.set_points(surface_points)
    interpolated.set_points(surface_points)
    direct_surface_field = np.asarray(native_field.B(), dtype=np.float64)
    surface_field = np.asarray(interpolated.B(), dtype=np.float64)
    trajectories, phi_hits = compute_fieldlines(
        interpolated,
        INITIAL_STATES[:, 0],
        INITIAL_STATES[:, 2],
        tmax=TMAX,
        tol=TOLERANCE,
        phis=_phi_planes(surface.nfp),
        stopping_criteria=[LevelsetStoppingCriterion(classifier.dist)],
    )
    values = _values(
        surface=surface,
        native_field=native_field,
        surface_field=surface_field,
        direct_surface_field=direct_surface_field,
        trajectories=trajectories,
        phi_hits=phi_hits,
        statuses=_native_statuses(trajectories, phi_hits),
        classifier=classifier,
    )
    return _Lane(values, list(trajectories), list(phi_hits))


def _jax_lane() -> _Lane:
    surface, native_field, classifier, skip, interpolation_arguments = _geometry()
    source_field = BiotSavartJAX(native_field.coils)
    interpolated = InterpolatedFieldJAX(
        source_field,
        *interpolation_arguments,
        nfp=surface.nfp,
        stellsym=True,
        skip=skip,
    )
    surface_points = np.asarray(surface.gamma(), dtype=np.float64).reshape((-1, 3))
    source_field.set_points(surface_points)
    interpolated.set_points(surface_points)
    direct_surface_field, surface_field = jax.device_get(
        (source_field.B(), interpolated.B())
    )
    trajectories, phi_hits, statuses = tracing_adapter.compute_fieldlines_with_status(
        interpolated,
        INITIAL_STATES[:, 0],
        INITIAL_STATES[:, 2],
        tmax=TMAX,
        tol=TOLERANCE,
        phis=_phi_planes(surface.nfp),
        stopping_criteria=[LevelsetStoppingCriterion(classifier)],
    )
    values = _values(
        surface=surface,
        native_field=native_field,
        surface_field=np.asarray(surface_field, dtype=np.float64),
        direct_surface_field=np.asarray(direct_surface_field, dtype=np.float64),
        trajectories=trajectories,
        phi_hits=phi_hits,
        statuses=statuses,
        classifier=classifier,
    )
    return _Lane(values, list(trajectories), list(phi_hits))


def _lane_is_healthy(values: Values) -> bool:
    """Every published array finite, a usable interpolant, no failed or capped line."""
    return bool(
        all(
            np.all(np.isfinite(np.asarray(value, dtype=np.float64)))
            for value in values.values()
        )
        and float(values["interpolation:relative_error"]) < 0.5
        and np.all(values["final:status"] <= 0)
    )


def test_qa_stopped_fieldline_remains_in_levelset_localization_band(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native = _native_lane()
    _jax_parity_environment(monkeypatch)
    jax_lane = _jax_lane()

    for observation in (native.values, jax_lane.values):
        stopped = observation["final:status"] < 0
        distances = observation["final:levelset_distance"]
        assert np.count_nonzero(stopped) == 1
        assert np.max(np.abs(distances[stopped])) <= 3.0e-3


def test_exact_tracing_fieldlines_qa_matches_native_and_jax_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native_lane = _native_lane()
    _jax_parity_environment(monkeypatch)
    jax_lane = _jax_lane()
    native, jax_values = native_lane.values, jax_lane.values

    assert _lane_is_healthy(native) is True
    assert _lane_is_healthy(jax_values) is True
    assert set(native) == set(jax_values)

    for observable in (
        "construction:surface_dofs",
        "construction:field_dofs",
        "initial:states",
        "interpolation:surface_field",
        "interpolation:relative_error",
        "final:status",
        "poincare:counts",
    ):
        np.testing.assert_allclose(
            jax_values[observable], native[observable], rtol=1.0e-12, atol=1.0e-14
        )

    np.testing.assert_allclose(
        jax_values["final:times"], native["final:times"], rtol=0.0, atol=6.0e-3
    )
    np.testing.assert_allclose(
        jax_values["final:states"], native["final:states"], rtol=0.0, atol=2.0e-3
    )
    np.testing.assert_allclose(
        jax_values["poincare:positions"],
        native["poincare:positions"],
        rtol=0.0,
        atol=7.0e-3,
    )

    assert native["final:status"].tolist() == [0, 0, -1]
    assert native["poincare:counts"].tolist() == [9, 9, 5]

    # Lines 0 and 1 run to ``tmax``: their end point is a property of the ODE,
    # not of the step grid, so it is held tight.
    for line in (0, 1):
        assert native["final:times"][line] == TMAX
        np.testing.assert_allclose(
            jax_values["final:states"][line],
            native["final:states"][line],
            rtol=0.0,
            atol=1.0e-11,
        )

    # Line 2 stops on the level set. Upstream does NOT root-find a
    # stopping-criterion crossing: ``simsoptpp/tracing.cpp`` evaluates every
    # criterion on the accepted post-step state and, on the first firing, records
    # ``{t, -1 - i, y}`` and breaks without pushing that state into the
    # trajectory. The published end point is therefore the last accepted step
    # still INSIDE the surface, quantised to the accepted-step grid, and the two
    # lanes' grids cannot coincide at ``tol = 1e-16`` (below double epsilon) where
    # the local error estimate sits on the floating-point noise floor of the two
    # interpolated-field implementations. What the rule does pin is asserted here.
    _levelset_stop_obeys_the_upstream_rule(native_lane, jax_lane, line=2)


def _levelset_stop_obeys_the_upstream_rule(
    native_lane: _Lane, jax_lane: _Lane, *, line: int
) -> None:
    native_values, jax_values = native_lane.values, jax_lane.values
    assert int(native_values["final:status"][line]) == -1
    assert int(jax_values["final:status"][line]) == -1
    # The kept row is the last accepted step still inside (signed distance > 0);
    # a lane that published the post-crossing state would be negative here.
    assert float(native_values["final:levelset_distance"][line]) > 0.0
    assert float(jax_values["final:levelset_distance"][line]) > 0.0

    stop_time_gap = abs(
        float(jax_values["final:times"][line])
        - float(native_values["final:times"][line])
    )
    state_gap = float(
        np.max(
            np.abs(
                jax_values["final:states"][line] - native_values["final:states"][line]
            )
        )
    )
    # Both factors of the bound are COMPUTED from this run, not written down.
    # The step that matters is the CROSSING step, the one the level set was
    # crossed inside -- not the last recorded one, which is the step BEFORE it
    # and which the controller may grow by up to ``_SAFETY * 5 = 4.5`` before
    # taking the crossing step. The crossing step is in the run: upstream pushes
    # the pre-step row at the top of the loop and records the criterion event at
    # the POST-step time (``simsoptpp/tracing.cpp``
    # ``res_phi_hits.push_back(join<2, RHS::Size>({t, -1-double(i)}, y))``), so
    # for each lane it is ``t_criterion_hit - t_final``.
    crossing_step = max(
        _criterion_crossing_step(
            native_lane.trajectories[line], native_lane.event_rows[line]
        ),
        _criterion_crossing_step(
            jax_lane.trajectories[line], jax_lane.event_rows[line]
        ),
    )
    # The path speed of a field line is |B|, evaluated with the coils at the two
    # published end points. The tracer integrates the interpolant of this field,
    # which the same test pins to a relative error of order 1e-7 through
    # ``interpolation:relative_error`` -- far below the scale of this bound.
    _surface, coil_field = _qa_objects()
    end_points = np.ascontiguousarray(
        np.stack(
            (
                np.asarray(native_values["final:states"][line], dtype=np.float64),
                np.asarray(jax_values["final:states"][line], dtype=np.float64),
            )
        )
    )
    coil_field.set_points(end_points)
    max_field_strength = float(
        np.max(np.linalg.norm(np.asarray(coil_field.B(), dtype=np.float64), axis=1))
    )
    bound = max_field_strength * crossing_step

    # Upstream does not root-find a criterion stop: it fires on the accepted
    # post-step state and keeps the pre-crossing row, so a stop state is only
    # defined to within ONE accepted step. That is the whole content of the rule,
    # and it is what is asserted -- with the guarantee that the bound is not
    # vacuous, i.e. strictly tighter than the blanket ``final:states`` bound the
    # same test already applies to all three lines.
    assert bound < 2.0e-3, f"level-set bound {bound:.3e} is not tighter than 2.0e-3"
    assert stop_time_gap <= crossing_step
    assert state_gap <= bound


def _criterion_crossing_step(trajectory: np.ndarray, event_rows: np.ndarray) -> float:
    """The accepted step the stopping criterion fired inside, for one lane.

    ``trajectory[-1, 0]`` is the last PRE-step row upstream keeps; the criterion
    event row carries the POST-step time of the step that fired
    (``simsoptpp/tracing.cpp``, the ``stopping_criteria`` loop). Their difference
    is that step, measured.
    """
    criterion_rows = event_rows[event_rows[:, 1] < 0]
    assert criterion_rows.shape[0] == 1
    crossing_step = float(criterion_rows[-1, 0] - trajectory[-1, 0])
    # Upstream's order: the pre-step row is pushed, the step is taken, the
    # criterion is evaluated on the POST-step state. So the event time is
    # strictly after the published end time, by exactly one accepted step.
    assert crossing_step > 0.0
    return crossing_step


@pytest.mark.parametrize("positive_status", [1, 2])
def test_qa_tracer_publishes_a_positive_core_status_as_reported(
    monkeypatch: pytest.MonkeyPatch,
    positive_status: int,
) -> None:
    """A budget stop (1) or a step-control failure (2) reaches the caller as-is.

    The chunked core is replaced by one that reports the status directly, so the
    status cannot be re-derived from the end time (line 0 stops at ``t = 1``
    with status ``positive_status``, not ``-1`` or ``0``).
    """
    trajectory = np.stack(
        [
            np.stack((np.r_[0.0, point], np.r_[time, point]))
            for point, time in zip(INITIAL_STATES, (1.0, 1.0, TMAX), strict=True)
        ]
    )
    hits = np.zeros((3, 1, 5), dtype=np.float64)
    hits[1, 0, 1] = -1.0
    core_result = SimpleNamespace(
        trajectory=trajectory,
        mask=np.ones((3, 2), dtype=bool),
        phi_hits=hits,
        phi_hits_count=np.asarray([0, 1, 0]),
        status=np.asarray([positive_status, -1, 0]),
        t_final=np.asarray([1.0, 1.0, TMAX]),
        steps_taken=np.ones(3, dtype=np.int32),
    )
    monkeypatch.setattr(
        tracing_adapter,
        "_trace_cartesian_chunks",
        lambda *_args, **_kwargs: (
            list(trajectory),
            [hits[0, :0], hits[1, :1], hits[2, :0]],
            core_result,
        ),
    )
    _jax_parity_environment(monkeypatch)

    trajectories, _phi_hits, statuses = tracing_adapter.compute_fieldlines_with_status(
        ToroidalFieldJAX(1.3, 0.8),
        INITIAL_STATES[:, 0],
        INITIAL_STATES[:, 2],
        tmax=TMAX,
        tol=TOLERANCE,
        phis=(),
        stopping_criteria=(),
    )

    assert statuses.tolist() == [positive_status, -1, 0]
    assert [float(path[-1, 0]) for path in trajectories] == [1.0, 1.0, TMAX]

    paths, crossings = tracing_adapter.compute_fieldlines(
        ToroidalFieldJAX(1.3, 0.8),
        INITIAL_STATES[:, 0],
        INITIAL_STATES[:, 2],
        tmax=TMAX,
        tol=1.0e-9,
        phis=(),
        stopping_criteria=(),
    )
    assert len(paths) == len(crossings) == 3


def test_qa_public_mirror_uses_official_integrator_tolerance() -> None:
    """The shipped mirror retains the official ``1e-16`` integrator tolerance."""
    source = (REPO_ROOT / "examples/jax/1_Simple/tracing_fieldlines_QA.py").read_text()
    module = ast.parse(source)
    solve = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "solve"
    )
    trace_call = next(
        node
        for node in ast.walk(solve)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "compute_fieldlines_with_status"
    )
    tolerance = next(
        keyword.value for keyword in trace_call.keywords if keyword.arg == "tol"
    )
    assert isinstance(tolerance, ast.Constant)
    assert tolerance.value == TOLERANCE
