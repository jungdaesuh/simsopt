"""Adapter continuation against uninterrupted Cartesian core traces."""

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import inspect
import os
import subprocess
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from simsopt.field.tracing import IterationStoppingCriterion, MaxRStoppingCriterion
from simsopt_jax.core.sharding import TrajectoryBatchShardingConfig
from simsopt_jax.core.tracing import (
    TRACING_STATUS_STEP_CONTROL_FAILED,
    FieldlineTracingSpec,
    GuidingCenterTracingSpec,
    IterStoppingCriterion,
    _MAX_CONSECUTIVE_NO_PROGRESS_TRIALS,
    _cartesian_particle_dtmaxs,
    _fieldline_dtmaxs,
    _magnetic_moments,
    _trace_fieldline_chunk,
    trace_fieldline,
    trace_guiding_center,
)
from simsopt_jax_adapters.field import tracing as adapter
from simsopt_jax_adapters.field.toroidal_field import ToroidalFieldJAX
from simsopt_jax_adapters.isolated_kernel import repo_child_pythonpath


class _NonFiniteField:
    """Field whose ``B`` is ``nan`` everywhere.

    Every trial then rejects (``_dopri5_adaptive_step`` maps a non-finite error
    to ``+inf``), ``h`` underflows and ``t`` never advances -- the stall the
    chunk loop used to spin on for ever.
    """

    @staticmethod
    def jax_B_at(point):
        return jnp.full((3,), jnp.nan, dtype=point.dtype)


def _discontinuous_field(point):
    """``B_x`` flips sign across ``x = 0``; the embedded error never shrinks.

    The RK stages straddle the jump for every step size, so ``|y_err|`` stays
    proportional to ``h`` while the error scale keeps its ``atol`` floor: the
    trial error is above one at every step size a double can represent between
    the initial step and the point where ``h`` would underflow.
    """
    sign = jnp.where(point[0] <= 0.0, 1.0, -1.0)
    return jnp.asarray([sign, 0.0, 0.0], dtype=jnp.float64)


def _stepped_field(point):
    """``|B|`` jumps at ``y = 0.5``: a burst of rejections, then recovery."""
    strength = jnp.where(point[1] < 0.5, 1.0, 2.0)
    return jnp.asarray([0.0, strength, 0.0], dtype=jnp.float64)


class _StatefulRotatingField:
    def __init__(self, strength: float):
        self.strength = strength

    def jax_tracing_state(self):
        return jnp.asarray(self.strength, dtype=jnp.float64)

    @staticmethod
    def jax_B_at_state(strength, point):
        return jnp.stack((-strength * point[1], strength * point[0], 0.0))


def _live(result):
    return np.asarray(result.trajectory)[np.asarray(result.mask)]


def _hits(result):
    return np.asarray(result.phi_hits)[: int(result.phi_hits_count)]


def test_fieldline_adapter_matches_uninterrupted_horizon_and_plane_hits(monkeypatch):
    monkeypatch.setattr(adapter, "_TRACING_CHUNK_TRIALS", 2)
    field = ToroidalFieldJAX(1.3, 0.8)
    y0 = np.array((1.4, 0.0, 0.0))
    tmax = 4.0
    tol = 1e-9
    initial_B = np.asarray(field.jax_B_at(jnp.asarray(y0)))
    dtmax = float(_fieldline_dtmaxs(y0[None, :], [np.linalg.norm(initial_B)])[0])
    reference = trace_fieldline(
        FieldlineTracingSpec(
            tmax=tmax, rtol=tol, atol=tol, max_steps=200,
            dtmax=dtmax, max_phi_hits=4096,
        ),
        y0,
        field.jax_B_at,
        phis=jnp.array((0.3,)),
    )
    paths, hits, statuses = adapter.compute_fieldlines_with_status(
        field, [y0[0]], [y0[2]], tmax, tol, [0.3], []
    )
    assert statuses.tolist() == [0]
    assert paths[0].shape[0] > 3
    assert np.all(np.diff(paths[0][:, 0]) > 0)
    assert hits[0].shape[0] > 0
    np.testing.assert_allclose(paths[0], _live(reference), rtol=0, atol=1e-12)
    np.testing.assert_allclose(hits[0], _hits(reference), rtol=0, atol=1e-12)


def test_fieldline_operational_total_cap_remains_incomplete(monkeypatch):
    monkeypatch.setattr(adapter, "_TRACING_CHUNK_TRIALS", 2)
    field = ToroidalFieldJAX(1.3, 0.8)
    paths, hits, statuses = adapter.compute_fieldlines_with_status(
        field, [1.4], [0.0], 4.0, 1e-9, (), [], max_steps=3
    )
    assert statuses.tolist() == [1]
    assert paths[0][-1, 0] < 4.0
    assert hits[0].shape == (0, 5)
    assert np.all(np.diff(paths[0][:, 0]) > 0)


def test_fieldline_chunking_keeps_finished_lanes_and_dynamic_field_state(monkeypatch):
    monkeypatch.setattr(adapter, "_TRACING_CHUNK_TRIALS", 2)
    field = _StatefulRotatingField(1.0)
    paths, hits, statuses = adapter.compute_fieldlines_with_status(
        field, [1.2, 1.4], [0.0, 0.0], 1.0, 1e-8, (),
        [MaxRStoppingCriterion(1.3)],
    )
    assert statuses.tolist() == [0, -1]
    assert paths[0][-1, 0] == 1.0
    assert paths[1].shape == (1, 4)
    assert hits[0].shape == (0, 5)
    assert hits[1].shape == (1, 5)
    assert hits[1][0, 1] == -1

    stronger_paths, _, stronger_statuses = adapter.compute_fieldlines_with_status(
        _StatefulRotatingField(2.0), [1.2], [0.0], 1.0, 1e-8, (), []
    )
    assert stronger_statuses.tolist() == [0]
    assert not np.allclose(paths[0][-1, 1:3], stronger_paths[0][-1, 1:3])


def test_guiding_center_adapter_preserves_event_and_forget_path(monkeypatch):
    monkeypatch.setattr(adapter, "_TRACING_CHUNK_TRIALS", 2)
    field = ToroidalFieldJAX(1.3, 0.8)
    xyz = np.array([[1.4, 0.0, 0.0]])
    speed_total = 1.0
    speed_par = 0.8
    initial_B = np.asarray(field.jax_B_at(jnp.asarray(xyz[0])))
    abs_B = np.linalg.norm(initial_B)
    mu = float(_magnetic_moments(speed_total, [speed_par], [abs_B])[0])
    dtmax = float(_cartesian_particle_dtmaxs(xyz, speed_total)[0])
    core = trace_guiding_center(
        GuidingCenterTracingSpec(
            tmax=4.0, rtol=1e-9, atol=1e-9, max_steps=100,
            dtmax=dtmax, max_phi_hits=4096,
        ),
        jnp.array((1.4, 0.0, 0.0, speed_par)),
        field.jax_B_dB_at,
        m=1.0, q=1.0, mu=mu,
        stopping_criteria=(IterStoppingCriterion(max_iter=9),),
    )
    kwargs = dict(
        tmax=4.0, mass=1.0, charge=1.0, Ekin=0.5,
        tol=1e-9, comm=None, phis=(),
        stopping_criteria=[IterationStoppingCriterion(9)],
        mode="gc_vac", max_steps=100,
    )
    full_paths, full_hits, full_statuses = adapter.trace_particles_with_status(
        field, xyz, [speed_par], forget_exact_path=False, **kwargs
    )
    short_paths, short_hits, short_statuses = adapter.trace_particles_with_status(
        field, xyz, [speed_par], forget_exact_path=True, **kwargs
    )
    assert full_statuses.tolist() == short_statuses.tolist() == [-1]
    assert full_paths[0].shape[0] > 2
    assert short_paths[0].shape == (2, 5)
    np.testing.assert_allclose(full_paths[0], _live(core), rtol=0, atol=1e-12)
    np.testing.assert_allclose(full_hits[0], _hits(core), rtol=0, atol=1e-12)
    np.testing.assert_allclose(short_paths[0], full_paths[0][[0, -1]], rtol=0, atol=0)
    np.testing.assert_allclose(short_hits[0], full_hits[0], rtol=0, atol=0)

    capped_paths, capped_hits, capped_statuses = (
        adapter.trace_particles_with_status(
            field, xyz, [speed_par], forget_exact_path=False,
            **{**kwargs, "max_steps": 3},
        )
    )
    assert capped_statuses.tolist() == [1]
    assert capped_paths[0][-1, 0] < 4.0
    assert capped_hits[0].shape == (0, 6)


def test_adapter_device_boundaries_obey_strict_transfer_guard(monkeypatch):
    monkeypatch.setattr(adapter, "_TRACING_CHUNK_TRIALS", 2)
    field = ToroidalFieldJAX(1.3, 0.8)
    with jax.transfer_guard("disallow"):
        paths, hits, statuses = adapter.compute_fieldlines_with_status(
            field, [1.4], [0.0], 4.0, 1e-9, (0.3,),
            [MaxRStoppingCriterion(2.0)], max_steps=4,
        )
        particle_paths, particle_hits, particle_statuses = (
            adapter.trace_particles_with_status(
                field, np.array([[1.4, 0.0, 0.0]]), [0.8],
                tmax=4.0, mass=1.0, charge=1.0, Ekin=0.5, tol=1e-9,
                comm=None, phis=(0.3,),
                stopping_criteria=(IterationStoppingCriterion(9),), mode="gc_vac",
                forget_exact_path=False, max_steps=4,
            )
        )
    assert statuses.tolist() == [1]
    assert paths[0].shape[0] >= 2
    assert hits[0].shape[1] == 5
    assert particle_statuses.tolist() == [1]
    assert particle_paths[0].shape[0] >= 2
    assert particle_hits[0].shape[1] == 6


# Set in the child of the two-device test: one re-execution, never a chain.
_FORCED_TWO_DEVICES = "SIMSOPT_TRACING_TEST_FORCED_TWO_DEVICES"


def test_two_device_criterion_chunk_sharding_matches_single_device(monkeypatch):
    """The sharded chunk path, on a box that has one CPU device by default.

    Repository convention for a multi-device assertion
    (``tests/jax/core/test_points_coils_sharding.py:15-16``,
    ``tests/jax/core/test_surface_seed_sharding.py:16``): re-execute with
    ``JAX_PLATFORMS=cpu`` and
    ``XLA_FLAGS=--xla_force_host_platform_device_count``. Those two drive a
    shared subprocess script; this test re-runs ITSELF, so the assertions stay in
    one place. The child sees two devices and runs the body, so the
    ``shard_map`` branch -- ``run_shard``, ``_chunk_output_specs``, the ``None``
    in-spec when ``phis`` is absent, the continuation state in-spec and
    ``check_vma=True`` -- is exercised on every run, with no skip.

    The CPU pin matters: ``--xla_force_host_platform_device_count`` acts on the
    host platform only, so on a box whose default backend is not CPU the child
    would see one device, take this same branch and spawn a grandchild. The
    sentinel below bounds the re-execution at depth 1 and makes that case fail
    HERE, in the child, where the assertion can actually fail.
    """
    devices = jax.devices()
    if len(devices) < 2:
        assert os.environ.get(_FORCED_TWO_DEVICES) is None, (
            f"child ran with {_FORCED_TWO_DEVICES} set and JAX_PLATFORMS=cpu but "
            f"still sees {len(devices)} device(s); "
            "--xla_force_host_platform_device_count did not take effect"
        )
        environment = dict(os.environ)
        environment["JAX_PLATFORMS"] = "cpu"
        environment[_FORCED_TWO_DEVICES] = "1"
        environment["XLA_FLAGS"] = (
            environment.get("XLA_FLAGS", "")
            + " --xla_force_host_platform_device_count=2"
        ).strip()
        # ``-S`` drops site-packages, so the child gets the sources, the loaded
        # kernel and the dependency root explicitly.
        environment["PYTHONPATH"] = repo_child_pythonpath(
            Path(__file__).resolve().parents[2], environment.get("PYTHONPATH")
        )
        completed = subprocess.run(
            (
                sys.executable, "-S", "-m", "pytest", "-p", "no:cacheprovider", "-q",
                f"{Path(__file__).resolve()}::"
                "test_two_device_criterion_chunk_sharding_matches_single_device",
            ),
            check=False, capture_output=True, text=True, timeout=900,
            env=environment,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        return
    monkeypatch.setattr(adapter, "_TRACING_CHUNK_TRIALS", 2)
    field = ToroidalFieldJAX(1.3, 0.8)
    inputs = ([1.2, 1.4, 1.2, 1.4], [0.0] * 4)
    kwargs = dict(
        tmax=1.0, tol=1e-8, phis=(0.3,),
        stopping_criteria=[MaxRStoppingCriterion(1.3)],
    )
    monkeypatch.setattr(adapter, "trajectory_batch_sharding_config", lambda _y0s: None)
    reference = adapter.compute_fieldlines_with_status(field, *inputs, **kwargs)

    mesh = Mesh(np.asarray(devices[:2], dtype=object), ("traj",))
    sharding = TrajectoryBatchShardingConfig(
        mesh=mesh, axis_name="traj", device_count=2, strategy="points",
    )
    monkeypatch.setattr(
        adapter, "trajectory_batch_sharding_config", lambda _y0s: sharding
    )
    sharded = adapter.compute_fieldlines_with_status(field, *inputs, **kwargs)
    np.testing.assert_array_equal(sharded[2], reference[2])
    assert sharded[2].tolist() == [0, -1, 0, -1]
    for sharded_rows, reference_rows in zip(sharded[0], reference[0], strict=True):
        np.testing.assert_allclose(sharded_rows, reference_rows, rtol=0, atol=1e-12)
    for sharded_hits, reference_hits in zip(sharded[1], reference[1], strict=True):
        np.testing.assert_allclose(sharded_hits, reference_hits, rtol=0, atol=1e-12)

    xyz = np.array([[1.2, 0.0, 0.0], [1.4, 0.0, 0.0]] * 2)
    particle_kwargs = dict(
        tmax=1.0, mass=1.0, charge=1.0, Ekin=0.5, tol=1e-8,
        comm=None, phis=(0.3,),
        stopping_criteria=[MaxRStoppingCriterion(1.3)], mode="gc_vac",
        forget_exact_path=False, max_steps=100,
    )
    monkeypatch.setattr(adapter, "trajectory_batch_sharding_config", lambda _y0s: None)
    particle_reference = adapter.trace_particles_with_status(
        field, xyz, [0.8] * 4, **particle_kwargs
    )
    monkeypatch.setattr(
        adapter, "trajectory_batch_sharding_config", lambda _y0s: sharding
    )
    particle_sharded = adapter.trace_particles_with_status(
        field, xyz, [0.8] * 4, **particle_kwargs
    )
    np.testing.assert_array_equal(particle_sharded[2], particle_reference[2])
    assert particle_sharded[2].tolist() == [0, -1, 0, -1]
    for sharded_rows, reference_rows in zip(
        particle_sharded[0], particle_reference[0], strict=True
    ):
        np.testing.assert_allclose(sharded_rows, reference_rows, rtol=0, atol=1e-10)
    for sharded_hits, reference_hits in zip(
        particle_sharded[1], particle_reference[1], strict=True
    ):
        np.testing.assert_allclose(sharded_hits, reference_hits, rtol=0, atol=1e-10)

    # The device-side endpoint gather must also hold across a sharded mesh.
    forget_kwargs = {**particle_kwargs, "forget_exact_path": True}
    forget_sharded = adapter.trace_particles_with_status(
        field, xyz, [0.8] * 4, **forget_kwargs
    )
    monkeypatch.setattr(adapter, "trajectory_batch_sharding_config", lambda _y0s: None)
    forget_reference = adapter.trace_particles_with_status(
        field, xyz, [0.8] * 4, **forget_kwargs
    )
    np.testing.assert_array_equal(forget_sharded[2], forget_reference[2])
    for sharded_rows, reference_rows in zip(
        forget_sharded[0], forget_reference[0], strict=True
    ):
        assert sharded_rows.shape[0] <= 2
        np.testing.assert_allclose(sharded_rows, reference_rows, rtol=0, atol=1e-10)


def test_fieldline_horizon_is_unbounded_unless_the_caller_bounds_it(monkeypatch):
    """Upstream has no trial limit, so neither may the default JAX horizon.

    ``simsoptpp/tracing.cpp`` loops ``do { ... } while(t < tmax && !stop);`` and
    ``simsopt.field.tracing.compute_fieldlines`` has no step parameter at all, so
    the only way a JAX line may end early is a bound the caller asked for.
    """
    monkeypatch.setattr(adapter, "_TRACING_CHUNK_TRIALS", 2)
    field = ToroidalFieldJAX(1.3, 0.8)
    assert (
        inspect.signature(adapter.compute_fieldlines_with_status)
        .parameters["max_steps"]
        .default
        is None
    )
    assert (
        inspect.signature(adapter.compute_fieldlines).parameters["max_steps"].default
        is None
    )

    _paths, _hits, statuses = adapter.compute_fieldlines_with_status(
        field, [1.4], [0.0], 4.0, 1e-9, (), []
    )
    assert statuses.tolist() == [0]

    bounded_paths, _bounded_hits, bounded_statuses = (
        adapter.compute_fieldlines_with_status(
            field, [1.4], [0.0], 4.0, 1e-9, (), [], max_steps=3
        )
    )
    assert bounded_statuses.tolist() == [1]
    assert bounded_paths[0][-1, 0] < 4.0

    with pytest.raises(ValueError, match="max_steps must be positive"):
        adapter.compute_fieldlines_with_status(
            field, [1.4], [0.0], 4.0, 1e-9, (), [], max_steps=0
        )


def test_public_compute_fieldlines_wraps_the_status_route(monkeypatch):
    monkeypatch.setattr(adapter, "_TRACING_CHUNK_TRIALS", 2)
    field = ToroidalFieldJAX(1.3, 0.8)
    paths, hits, statuses = adapter.compute_fieldlines_with_status(
        field, [1.4], [0.0], 4.0, 1e-9, (0.3,), []
    )
    wrapped = adapter.compute_fieldlines(field, [1.4], [0.0], 4.0, 1e-9, (0.3,), [])
    assert len(wrapped) == 2
    assert statuses.tolist() == [0]
    np.testing.assert_allclose(wrapped[0][0], paths[0], rtol=0, atol=0)
    np.testing.assert_allclose(wrapped[1][0], hits[0], rtol=0, atol=0)


@pytest.mark.parametrize("chunk_trials", [1, 2, 3, 7, 13, 64, 200])
def test_chunked_fieldline_result_is_independent_of_the_chunk_size(
    monkeypatch, chunk_trials
):
    """The seam may not be observable: only the chunk size varies here."""
    field = ToroidalFieldJAX(1.3, 0.8)
    y0 = np.array((1.4, 0.0, 0.0))
    tol = 1e-9
    initial_B = np.asarray(field.jax_B_at(jnp.asarray(y0)))
    dtmax = float(_fieldline_dtmaxs(y0[None, :], [np.linalg.norm(initial_B)])[0])
    reference = trace_fieldline(
        FieldlineTracingSpec(
            tmax=4.0, rtol=tol, atol=tol, max_steps=400,
            dtmax=dtmax, max_phi_hits=4096,
        ),
        y0,
        field.jax_B_at,
        phis=jnp.array((0.3,)),
    )
    monkeypatch.setattr(adapter, "_TRACING_CHUNK_TRIALS", chunk_trials)
    paths, hits, statuses = adapter.compute_fieldlines_with_status(
        field, [y0[0]], [y0[2]], 4.0, tol, [0.3], []
    )
    assert statuses.tolist() == [0]
    np.testing.assert_allclose(paths[0], _live(reference), rtol=0, atol=0)
    np.testing.assert_allclose(hits[0], _hits(reference), rtol=0, atol=0)


def test_forget_exact_path_never_drains_a_chunk_trajectory(monkeypatch):
    """With ``forget_exact_path`` the chunk loop keeps two rows, not the path."""
    monkeypatch.setattr(adapter, "_TRACING_CHUNK_TRIALS", 2)
    field = ToroidalFieldJAX(1.3, 0.8)
    xyz = np.array([[1.4, 0.0, 0.0]])
    kwargs = dict(
        tmax=4.0, mass=1.0, charge=1.0, Ekin=0.5, tol=1e-9, comm=None,
        phis=(0.3,), stopping_criteria=[], mode="gc_vac", max_steps=100,
    )
    full_paths, full_hits, _ = adapter.trace_particles_with_status(
        field, xyz, [0.8], forget_exact_path=False, **kwargs
    )

    def _forbidden(_result):
        raise AssertionError("chunk trajectory drained under forget_exact_path")

    monkeypatch.setattr(adapter, "_batched_jax_live_rows", _forbidden)
    short_paths, short_hits, short_statuses = adapter.trace_particles_with_status(
        field, xyz, [0.8], forget_exact_path=True, **kwargs
    )
    assert short_statuses.tolist() == [0]
    assert short_paths[0].shape == (2, 5)
    np.testing.assert_allclose(short_paths[0], full_paths[0][[0, -1]], rtol=0, atol=0)
    np.testing.assert_allclose(short_hits[0], full_hits[0], rtol=0, atol=0)


def test_uncontinued_routes_carry_their_own_named_step_limit():
    """Boozer guiding centres and full orbits keep a hard cap; upstream has none.

    Both upstream routes call the same unbounded ``solve()``
    (``simsoptpp/tracing.cpp`` ``particle_guiding_center_boozer_tracing`` and
    ``particle_fullorbit_tracing``). The JAX versions are not continued, so their
    limit is also the trajectory buffer and stays finite -- but it is named,
    caller-settable, and can only produce status 1, never 0.
    """
    assert adapter._BOOZER_GUIDING_CENTRE_STEP_LIMIT == 4_000
    assert adapter._FULLORBIT_STEP_LIMIT == 20_000
    for function in (adapter.trace_particles_boozer, adapter.trace_particles):
        assert inspect.signature(function).parameters["max_steps"].default is None
    boozer_source = inspect.getsource(adapter._trace_particles_boozer_jax)
    assert "_BOOZER_GUIDING_CENTRE_STEP_LIMIT" in boozer_source
    assert "_TRACING_CHUNK_TRIALS" not in boozer_source
    fullorbit_source = inspect.getsource(adapter._trace_particles_jax_fullorbit_vacuum)
    assert "_FULLORBIT_STEP_LIMIT" in fullorbit_source


def test_full_orbit_route_reports_status_one_when_its_limit_ends_the_run():
    field = ToroidalFieldJAX(1.3, 0.8)
    xyz = np.array([[1.4, 0.0, 0.0]])
    kwargs = dict(
        tmax=1.0, mass=1.0, charge=1.0, Ekin=0.5, tol=1e-9, comm=None,
        phis=(), stopping_criteria=[], mode="full", forget_exact_path=False,
    )
    paths, _hits, statuses = adapter.trace_particles_with_status(
        field, xyz, [0.8], max_steps=3, **kwargs
    )
    assert statuses.tolist() == [1]
    assert paths[0][-1, 0] < 1.0


class _CountingComm:
    """One-rank comm that only records how many collectives a route issues."""

    size = 1
    rank = 0

    def __init__(self):
        self.allgather_calls = 0

    def allgather(self, value):
        self.allgather_calls += 1
        return [value]

    def allreduce(self, value):
        return value


def test_two_value_routes_keep_upstream_collective_traffic(monkeypatch):
    """``compute_fieldlines`` / ``trace_particles`` mirror upstream's two values.

    Upstream gathers the trajectories and the event rows and nothing else, so the
    two-value routes must too; only the status routes pay for a third gather.
    """
    monkeypatch.setattr(adapter, "_TRACING_CHUNK_TRIALS", 2)
    field = ToroidalFieldJAX(1.3, 0.8)

    comm = _CountingComm()
    adapter.compute_fieldlines(field, [1.4], [0.0], 4.0, 1e-9, (), [], comm=comm)
    assert comm.allgather_calls == 2

    comm = _CountingComm()
    adapter.compute_fieldlines_with_status(
        field, [1.4], [0.0], 4.0, 1e-9, (), [], comm=comm
    )
    assert comm.allgather_calls == 3

    particle_kwargs = dict(
        tmax=4.0, mass=1.0, charge=1.0, Ekin=0.5, tol=1e-9, phis=(),
        stopping_criteria=[], mode="gc_vac", max_steps=100,
    )
    comm = _CountingComm()
    adapter.trace_particles(
        field, np.array([[1.4, 0.0, 0.0]]), [0.8], comm=comm, **particle_kwargs
    )
    assert comm.allgather_calls == 2

    comm = _CountingComm()
    adapter.trace_particles_with_status(
        field, np.array([[1.4, 0.0, 0.0]]), [0.8], comm=comm, **particle_kwargs
    )
    assert comm.allgather_calls == 3


def test_non_finite_field_ends_the_lane_with_the_step_control_failure_status():
    """A NaN right-hand side must end the lane, not spin the chunk loop.

    Before the guard this call never returned: every trial rejected, ``h``
    underflowed to ``0.0`` near trial 460, ``t`` stayed at ``0``, the core kept
    reporting status ``1`` and ``_trace_cartesian_chunks`` re-issued chunks for
    ever while its host buffers grew. Upstream cannot spin here either -- boost's
    ``failed_step_checker`` throws after 500 consecutive failed trials inside one
    ``dense_output_runge_kutta::do_step``.
    """
    started = time.perf_counter()
    paths, hits, statuses = adapter.compute_fieldlines_with_status(
        _NonFiniteField(), [1.4], [0.0], tmax=4.0, tol=1.0e-9,
        phis=(), stopping_criteria=(),
    )
    elapsed = time.perf_counter() - started
    assert elapsed < 60.0, f"stall guard did not end the lane quickly ({elapsed:.1f}s)"
    assert statuses.tolist() == [TRACING_STATUS_STEP_CONTROL_FAILED]
    # No accepted step: the published path is the initial row only.
    assert paths[0].shape == (1, 4)
    assert float(paths[0][0, 0]) == 0.0
    assert hits[0].shape[0] == 0


def test_persistent_rejection_stops_the_lane_at_the_upstream_trial_limit():
    """500 consecutive non-advancing trials end the lane, as boost's checker does.

    ``_discontinuous_field`` keeps the trial error above one at every step size
    between the initial step and ``h`` underflow, so nothing is ever accepted and
    the counter -- not the underflow -- is what ends the lane.
    """
    spec = FieldlineTracingSpec(
        tmax=1.0e80, rtol=1.0e-300, atol=1.0e-300,
        max_steps=_MAX_CONSECUTIVE_NO_PROGRESS_TRIALS + 100,
        dtmax=1.0e80, max_phi_hits=8,
    )
    result, state = _trace_fieldline_chunk(
        spec, jnp.zeros(3, dtype=jnp.float64), _discontinuous_field
    )
    assert int(result.status) == TRACING_STATUS_STEP_CONTROL_FAILED
    assert int(result.steps_taken) == 0
    assert float(result.t_final) == 0.0
    assert int(state.trial_count) == _MAX_CONSECUTIVE_NO_PROGRESS_TRIALS
    assert int(state.no_progress) == _MAX_CONSECUTIVE_NO_PROGRESS_TRIALS
    # The step size is still finite: the trial counter ended the lane, not an
    # underflow, so the rule really is upstream's.
    assert np.isfinite(float(state.h)) and float(state.h) > 0.0


def test_rejection_burst_that_recovers_does_not_trip_the_guard():
    """The counter resets on an accepted step that advances ``t``.

    A guard keyed on rejections alone, or one that never reset, would kill this
    lane: it rejects 40 trials at the field discontinuity and then runs to
    ``tmax`` with status 0.
    """
    spec = FieldlineTracingSpec(
        tmax=1.0, rtol=1.0e-10, atol=1.0e-10, max_steps=4_000,
        dtmax=0.1, max_phi_hits=8,
    )
    result, state = _trace_fieldline_chunk(
        spec, jnp.asarray([1.0, 0.0, 0.0], dtype=jnp.float64), _stepped_field
    )
    assert int(result.status) == 0
    assert float(result.t_final) == 1.0
    assert int(state.trial_count) > int(result.steps_taken)
    assert int(state.no_progress) == 0


def test_chunked_routes_use_the_early_exit_loop_and_match_the_scan_form(monkeypatch):
    """The chunked Cartesian routes take ``while``; both forms agree bitwise.

    ``scan`` exists to keep the core drivers reverse-mode differentiable, which a
    chunked route -- drained to NumPy and assembled in a host loop -- cannot be.
    """
    assert adapter._CHUNKED_ADAPTIVE_LOOP == "while"
    field = ToroidalFieldJAX(1.3, 0.8)
    xyz = np.array([[1.2, 0.0, 0.0], [1.4, 0.0, 0.0]])
    fieldline_kwargs = dict(
        tmax=2.0, tol=1e-9, phis=(0.3,),
        stopping_criteria=[MaxRStoppingCriterion(1.3)],
    )
    particle_kwargs = dict(
        tmax=2.0, mass=1.0, charge=1.0, Ekin=0.5, tol=1e-9, comm=None,
        phis=(0.3,), stopping_criteria=[MaxRStoppingCriterion(1.3)],
        mode="gc_vac", forget_exact_path=False, max_steps=4_000,
    )
    monkeypatch.setattr(adapter, "_TRACING_CHUNK_TRIALS", 64)
    outputs = {}
    for form in ("while", "scan"):
        monkeypatch.setattr(adapter, "_CHUNKED_ADAPTIVE_LOOP", form)
        outputs[form] = (
            adapter.compute_fieldlines_with_status(
                field, [1.2, 1.4], [0.0, 0.0], **fieldline_kwargs
            ),
            adapter.trace_particles_with_status(
                field, xyz, [0.8, 0.8], **particle_kwargs
            ),
        )
    for while_route, scan_route in zip(outputs["while"], outputs["scan"], strict=True):
        np.testing.assert_array_equal(while_route[2], scan_route[2])
        assert set(np.asarray(while_route[2]).tolist()) == {0, -1}
        for while_rows, scan_rows in zip(while_route[0], scan_route[0], strict=True):
            np.testing.assert_array_equal(while_rows, scan_rows)
        for while_hits, scan_hits in zip(while_route[1], scan_route[1], strict=True):
            np.testing.assert_array_equal(while_hits, scan_hits)


@pytest.mark.parametrize("chunk_trials", [1, 3, 8, 64, 400])
def test_chunked_guiding_centre_stop_is_independent_of_the_chunk_size(
    monkeypatch, chunk_trials
):
    """Same seam invariance on the criterion-stopped guiding-centre route.

    The dense-output localizer reads the stage derivatives of the step being
    taken and the no-progress counter is carried in the continuation state, so
    both are per-chunk quantities that a chunk boundary could have made
    observable. Nothing here may change with the chunk size, bit for bit.
    """
    field = ToroidalFieldJAX(1.3, 0.8)
    xyz = np.array([[1.2, 0.0, 0.0], [1.4, 0.0, 0.0]])
    kwargs = dict(
        tmax=2.0, mass=1.0, charge=1.0, Ekin=0.5, tol=1e-9, comm=None,
        phis=(0.3, 1.1), stopping_criteria=[MaxRStoppingCriterion(1.3)],
        mode="gc_vac", forget_exact_path=False, max_steps=4_000,
    )
    monkeypatch.setattr(adapter, "_TRACING_CHUNK_TRIALS", 4_000)
    reference = adapter.trace_particles_with_status(field, xyz, [0.8, 0.8], **kwargs)
    assert reference[2].tolist() == [0, -1]

    monkeypatch.setattr(adapter, "_TRACING_CHUNK_TRIALS", chunk_trials)
    chunked = adapter.trace_particles_with_status(field, xyz, [0.8, 0.8], **kwargs)
    np.testing.assert_array_equal(chunked[2], reference[2])
    for chunked_rows, reference_rows in zip(chunked[0], reference[0], strict=True):
        np.testing.assert_array_equal(chunked_rows, reference_rows)
    for chunked_hits, reference_hits in zip(chunked[1], reference[1], strict=True):
        np.testing.assert_array_equal(chunked_hits, reference_hits)
