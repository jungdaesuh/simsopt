"""Cache and host-boundary regressions for the BiotSavart correctness follow-up."""

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from simsopt.field import BiotSavart, Coil, Current, coils_via_symmetries
from simsopt.geo import SurfaceXYZTensorFourier, Volume, create_equally_spaced_curves
from simsopt.geo.curveperturbed import CurvePerturbed, GaussianSampler, PerturbationSample
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.geo.surfaceobjectives import boozer_surface_residual
from simsopt_jax_adapters.field import biotsavart_backend as backend
from simsopt_jax_adapters.geo.boozer_surface import BoozerSurfaceJAX
from simsopt_jax_adapters.geo.surface_objectives import BoozerResidualJAX


_POINTS = np.array([[0.8, 0.1, 0.2], [1.1, -0.2, -0.1]])


def _coils(*, perturbed=False):
    curves = create_equally_spaced_curves(
        2, 1, stellsym=False, R0=1.0, R1=0.25, order=2, numquadpoints=32
    )
    if perturbed:
        sampler = GaussianSampler(curves[0].quadpoints, 1e-3, 0.2, n_derivs=1)
        curves = [CurvePerturbed(
            curves[0], PerturbationSample(sampler, randomgen=np.random.default_rng(713))
        )]
    currents = [Current(1e5) for _curve in curves]
    return curves, coils_via_symmetries(curves, currents, 2, True)


def test_warmed_full_owner_projection_host_boundaries(monkeypatch):
    _, coils = _coils()
    field = backend.BiotSavartJAX(coils)
    field.set_points(_POINTS)
    cotangent = np.arange(6, dtype=float).reshape(2, 3) / 7
    native = BiotSavart(coils)
    native.set_points(_POINTS)
    native.B()
    expected = native.B_vjp(cotangent)
    pullback = field.B_pullback_native(cotangent)
    arrays, indices = pullback.d_coil_arrays, pullback.coil_indices
    field.coil_cotangents_to_derivative(arrays, indices)
    placements = []
    materializations = []
    metadata_placements = []
    put = jax.device_put
    materialize = backend.host_array
    place_metadata = backend._place_array_tree_on_device

    def counted_put(value, *args, **kwargs):
        placements.extend(leaf.nbytes for leaf in jax.tree.leaves(value)
                          if isinstance(leaf, np.ndarray) and leaf.ndim > 0)
        return put(value, *args, **kwargs)

    def counted_materialize(value, **kwargs):
        materializations.append(1)
        return materialize(value, **kwargs)

    def counted_metadata(tree, device):
        metadata_placements.append(1)
        return place_metadata(tree, device)

    monkeypatch.setattr(jax, "device_put", counted_put)
    monkeypatch.setattr(backend, "host_array", counted_materialize)
    monkeypatch.setattr(backend, "_place_array_tree_on_device", counted_metadata)
    actual = field.coil_cotangents_to_derivative(arrays, indices)
    print(f"BF2: placements={len(placements)} bytes={sum(placements)} "
          f"materializations={len(materializations)}")
    np.testing.assert_allclose(actual(field), expected(native), rtol=1e-11, atol=1e-13)
    # 82d2e8805: one free-vector H2D boundary (32 doubles), one packed D2H.
    assert (len(placements), sum(placements), len(materializations)) == (1, 256, 1)
    assert not metadata_placements


def test_packed_projection_preserves_distinct_owners_with_shared_dofs():
    curves, _ = _coils()
    curve = curves[0]
    shared = CurveXYZFourier(curve.quadpoints, curve.order, dofs=curve.dofs)
    curve.fix(0)
    first, second = Current(1e5), Current(2e4)
    second.fix_all()
    coils = [Coil(curve, first + second), Coil(shared, -2.5 * (first - second))]
    field, native = backend.BiotSavartJAX(coils), BiotSavart(coils)
    for evaluator in (field, native):
        evaluator.set_points(_POINTS)
        evaluator.B()
    cotangent = np.arange(6, dtype=float).reshape(2, 3) / 7
    expected = native.B_vjp(cotangent)
    actual = field.B_vjp(cotangent)
    assert set(actual.data) == set(expected.data)
    for owner in expected.data:
        np.testing.assert_allclose(actual.data[owner], expected.data[owner],
                                   rtol=1e-11, atol=1e-13)
    assert not np.array_equal(actual.data[curve], actual.data[shared])
    assert actual.data[curve].flags.writeable
    assert not np.shares_memory(actual.data[curve], actual.data[shared])


@pytest.mark.parametrize("fixed", ["partial", "full"])
def test_packed_projection_preserves_fixed_nested_curve_partials(fixed):
    curves, coils = _coils(perturbed=True)
    if fixed == "partial":
        curves[0].curve.fix(2)
    else:
        curves[0].curve.fix_all()
    field, native = backend.BiotSavartJAX(coils), BiotSavart(coils)
    for evaluator in (field, native):
        evaluator.set_points(_POINTS)
    np.testing.assert_allclose(np.asarray(field.B()), native.B(), rtol=1e-12, atol=1e-14)
    cotangent = np.arange(6, dtype=float).reshape(2, 3) / 7
    expected, actual = native.B_vjp(cotangent), field.B_vjp(cotangent)
    assert set(actual.data) == set(expected.data)
    for owner in expected.data:
        np.testing.assert_allclose(actual.data[owner], expected.data[owner],
                                   rtol=1e-11, atol=1e-13)


class _CountedSample(np.ndarray):
    copies = []

    def tobytes(self, order="C"):
        self.copies.append(self.nbytes)
        return super().tobytes(order)

    def copy(self, order="C"):
        self.copies.append(self.nbytes)
        return super().copy(order)


def test_unchanged_perturbation_evaluations_do_not_copy_samples(monkeypatch):
    curves, coils = _coils(perturbed=True)
    curve = curves[0]
    curve.sample._sample = [sample.view(_CountedSample) for sample in curve.sample._sample]
    field = backend.BiotSavartJAX(coils)
    field.set_points(_POINTS)
    expected = np.asarray(field.B()).copy()
    native = BiotSavart(coils)
    native.set_points(_POINTS)
    np.testing.assert_allclose(expected, native.B(), rtol=1e-12, atol=1e-14)
    spec = field.coil_dof_extraction_spec()
    fingerprint_reads = []
    fingerprint = field._current_captured_coil_state_fingerprint

    def counted_fingerprint():
        fingerprint_reads.append(1)
        return fingerprint()

    monkeypatch.setattr(field, "_current_captured_coil_state_fingerprint", counted_fingerprint)
    _CountedSample.copies.clear()
    for _ in range(10):
        np.testing.assert_allclose(np.asarray(field.B()), expected, rtol=0, atol=0)
    print(f"BF3: sample_copies={len(_CountedSample.copies)} "
          f"bytes={sum(_CountedSample.copies)}")
    assert not _CountedSample.copies
    assert not fingerprint_reads
    assert field.coil_dof_extraction_spec() is spec


def _boozer_problem():
    curves, coils = _coils(perturbed=True)
    field = backend.BiotSavartJAX(coils)
    surface = SurfaceXYZTensorFourier(
        mpol=1, ntor=1, nfp=2, stellsym=True,
        quadpoints_phi=np.linspace(0, 0.5, 3, endpoint=False),
        quadpoints_theta=np.linspace(0, 1, 3, endpoint=False),
    )
    solver = BoozerSurfaceJAX(
        field, surface, Volume(surface), targetlabel=0.1, constraint_weight=1.0,
        options={"bfgs_maxiter": 0, "newton_maxiter": 0, "verbose": False},
    )
    solver.install_value_only_solved_runtime_state(
        sdofs=jnp.asarray(surface.get_dofs()), iota=0.23, G=1.7,
    )
    return curves[0], field, solver, BoozerResidualJAX(solver, field)


@pytest.mark.parametrize("resample_source", ["curve", "sample"])
def test_resampling_invalidates_downstream_objective(resample_source, monkeypatch):
    curve, field, solver, objective = _boozer_problem()
    before = objective.J()
    state = solver.get_solved_runtime_state()
    solves = []

    def install_trusted_state(iota, G=None):
        # Keep the reviewer's public installed surface state while exercising
        # objective-triggered recomputation, independently of convergence.
        solves.append(1)
        return solver.install_value_only_solved_runtime_state(
            sdofs=state.sdofs, iota=iota, G=G,
        )

    monkeypatch.setattr(solver, "run_code", install_trusted_state)
    if resample_source == "sample":
        curve.sample.resample()
    else:
        curve.resample()
    if resample_source == "sample":
        field.coil_dof_extraction_spec()  # Detection must notify dependents too.
    actual = objective.J()
    # A fresh objective/field is an independent cache oracle at the same state.
    fresh_field = backend.BiotSavartJAX(field.coils)
    expected = BoozerResidualJAX(solver, fresh_field).J()
    native = BiotSavart(field.coils)
    residual, = boozer_surface_residual(
        solver.surface, float(state.iota), float(state.G), native,
        derivatives=0, weight_inv_modB=bool(state.weight_inv_modB),
    )
    native_value = (np.sum(residual**2) / (2 * residual.size)
                    + 0.5 * (Volume(solver.surface).J() - 0.1)**2)
    print(f"BF1 objective {resample_source}: before={before:.17g} "
          f"cached={actual:.17g} fresh={expected:.17g} solves={len(solves)}")
    assert abs(expected - before) > 1e-9
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(actual, native_value, rtol=1e-12, atol=1e-14)
    assert len(solves) == 1


@pytest.mark.parametrize("resample_source", ["curve", "sample"])
def test_resampling_invalidates_downstream_solver(resample_source):
    curve, field, solver, _ = _boozer_problem()
    before = np.asarray(solver.coil_set_spec.groups[0].gammas).copy()
    if resample_source == "curve":
        curve.resample()
    else:
        curve.sample.resample()
    field.coil_dof_extraction_spec()
    dirty = solver.need_to_run_code
    result = solver.run_code(0.23, G=1.7)
    after = np.asarray(solver.coil_set_spec.groups[0].gammas)
    delta = np.max(np.abs(after - before))
    print(f"BF1 solver {resample_source}: dirty={dirty} "
          f"returned_result={result is not None} geometry_delta={delta:.17g}")
    assert dirty
    assert result is not None
    assert delta > 1e-6
    np.testing.assert_allclose(after, np.stack([coil.curve.gamma() for coil in field.coils]),
                               rtol=1e-12, atol=1e-14)
