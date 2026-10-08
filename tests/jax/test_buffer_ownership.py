"""Caller mutation cannot change PR 3's captured state or pending kernels."""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from simsopt.field import Coil, Current
from simsopt.geo import create_equally_spaced_curves
from simsopt.geo.curvehelical import CurveHelical
from simsopt.geo.curvexyzfouriersymmetries import CurveXYZFourierSymmetries
from simsopt_jax.backend import set_backend
from simsopt_jax.backend.dtypes import (
    as_compute_array,
    as_jax_array,
    explicit_device_array,
    runtime_device_put,
    runtime_device_put_tree,
)
from simsopt_jax.core import biotsavart as kernels
from simsopt_jax.core import specs
from simsopt_jax.core.field import group_biot_savart_B_vjp
from simsopt_jax.runtime.host_boundary import snapshot_host_tree
from simsopt_jax_adapters.field import BiotSavartJAX
from simsopt_jax_adapters.field import biotsavart_backend as adapter
from simsopt_jax_adapters.geo.curve_specs import curve_spec_from_adapter_curve


def _host_array(shape, *, misaligned=False):
    """64-byte alignment forces CPU aliasing; offset one tests delayed copies."""
    size = int(np.prod(shape))
    storage = np.empty(size + 9, dtype=np.float64)
    offset = (-storage.ctypes.data % 64) // storage.itemsize + int(misaligned)
    array = storage[offset:offset + size].reshape(shape)
    array[:] = np.arange(size).reshape(shape) / max(size, 1) + 1.0
    return array


def _assert_tree_equal(actual, expected):
    jax.block_until_ready(actual)
    for observed, saved in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_array_equal(observed, saved)


def _assert_pending(result):
    assert any(not leaf.is_ready() for leaf in jax.tree.leaves(result)), "consumer finished before caller mutation"


@pytest.mark.parametrize("misaligned", [False, True])
@pytest.mark.parametrize("place", [
    runtime_device_put,
    runtime_device_put_tree,
    lambda x: runtime_device_put_tree(x, preserve_placement=True),
    lambda x: explicit_device_array(x, dtype=np.float64),
    lambda x: as_jax_array(x, dtype=np.float64),
    as_compute_array,
])
def test_placement_snapshots_numpy_before_dispatch(place, misaligned):
    set_backend("jax", device="cpu", intent="parity")
    source = _host_array((5_000_000,), misaligned=misaligned)
    expected = source.copy()
    result = place(source)
    source[:] = -7.0
    _assert_tree_equal(result, expected)


def test_snapshot_preserves_device_arrays_and_tracers():
    device = jnp.arange(16, dtype=jnp.float64)
    assert snapshot_host_tree(device) is device
    tree = snapshot_host_tree({"device": device, "host": np.arange(16)})
    assert tree["device"] is device
    np.testing.assert_array_equal(jax.jit(snapshot_host_tree)(device), device)
    assert runtime_device_put_tree(device, preserve_placement=True) is device


@pytest.mark.parametrize("convert", [
    lambda x: as_jax_array(x, dtype=np.float64),
    as_compute_array,
    runtime_device_put,
])
def test_mixed_device_host_sequence_snapshots_numpy(convert, execution_gate):
    source = _host_array((65536,))
    expected = source.copy()
    zeros = jnp.zeros(source.shape)
    gate = execution_gate
    gate(zeros).block_until_ready()
    convert([zeros, source]).block_until_ready()
    result = convert([gate(zeros), source])
    _assert_pending(result)
    source[:] = -7.0
    np.testing.assert_array_equal(result.block_until_ready()[1], expected)


class _SpecInputs:
    def __init__(self):
        self.dofs = _host_array((10,))
        self.quadpoints = _host_array((65536,))
        self.samples = _host_array((65536, 3))
        self.rotmat = _host_array((3, 3))

    def mutate(self):
        for source in (self.dofs, self.quadpoints, self.samples, self.rotmat):
            source[:] = -7.0


def _curve_spec(factory, source, dof_count=9):
    return factory(dofs=source.dofs[:dof_count], quadpoints=source.quadpoints, order=1)


def _map_spec(source):
    return specs.make_optimizable_dof_map_spec(
        template_full_dofs=source.dofs[:9], owner_segments=(),
        input_mode="full", input_start=0, input_end=9,
    )


_SPEC_FACTORIES = [
    ("xyz", lambda x: _curve_spec(specs.make_curve_xyzfourier_spec, x)),
    ("oriented_xyz", lambda x: _curve_spec(specs.make_oriented_curve_xyzfourier_spec, x)),
    ("planar", lambda x: _curve_spec(specs.make_curve_planarfourier_spec, x, 10)),
    ("rz", lambda x: specs.make_curve_rzfourier_spec(dofs=x.dofs[:6], quadpoints=x.quadpoints, order=1, nfp=1, stellsym=False)),
    ("helical", lambda x: specs.make_curve_helical_spec(dofs=x.dofs[:3], quadpoints=x.quadpoints, order=1, m=1, ell=1, R0=1.0, r=0.1)),
    ("symmetries", lambda x: specs.make_curve_xyzfouriersymmetries_spec(dofs=x.dofs[:9], quadpoints=x.quadpoints, order=1, nfp=1, stellsym=False, ntor=1)),
    ("dof_map", _map_spec),
    ("frame_rotation", lambda x: specs.make_frame_rotation_spec(dofs=x.dofs[:3], quadpoints=x.quadpoints, order=1, scale=1.0)),
    ("zero_rotation", lambda x: specs.make_zero_rotation_spec(quadpoints=x.quadpoints)),
    ("coil_symmetry", lambda x: specs.make_coil_symmetry_spec(rotmat=x.rotmat)),
    ("field", lambda x: specs.make_field_eval_spec(x.samples)),
    ("coil_group", lambda x: specs.make_coil_group_spec(x.samples[None], x.samples[None], x.dofs[:1], (0,))),
    ("perturbed", lambda x: specs.make_curve_perturbed_spec(
        dofs=x.dofs[:9], quadpoints=x.quadpoints,
        base_curve=_curve_spec(specs.make_curve_xyzfourier_spec, x), base_curve_map=_map_spec(x),
        sample_gamma=x.samples, sample_gammadash=x.samples,
        sample_gammadashdash=x.samples, sample_gammadashdashdash=x.samples,
    )),
    ("filament", lambda x: specs.make_curve_filament_spec(
        dofs=x.dofs[:9], quadpoints=x.quadpoints,
        base_curve=_curve_spec(specs.make_curve_xyzfourier_spec, x), base_curve_map=_map_spec(x),
        rotation=specs.make_zero_rotation_spec(quadpoints=x.quadpoints), rotation_map=_map_spec(x),
        frame_kind="frenet", dn=0.1, db=0.1,
    )),
]


@pytest.mark.parametrize("factory", [factory for _, factory in _SPEC_FACTORIES], ids=[name for name, _ in _SPEC_FACTORIES])
def test_spec_factories_own_numpy_leaves(factory):
    source = _SpecInputs()
    result = factory(source)
    expected = jax.tree.map(lambda leaf: np.array(leaf, copy=True), result)
    source.mutate()
    _assert_tree_equal(result, expected)


@pytest.mark.parametrize("native_curve", [
    lambda points: CurveHelical(points, order=1),
    lambda points: CurveXYZFourierSymmetries(points, order=1, nfp=1, stellsym=False),
], ids=["helical", "symmetries"])
def test_adapter_curve_capture_owns_native_coefficients(native_curve):
    quadpoints = _host_array((65536,))
    quadpoints[:] -= 1.0
    curve = native_curve(quadpoints)
    coefficients = _host_array((curve.num_dofs(),))
    curve.coefficients = coefficients
    result = curve_spec_from_adapter_curve(curve)
    expected = jax.tree.map(lambda x: np.array(x, copy=True), result)
    coefficients[:] = -7.0
    quadpoints[:] = -7.0
    _assert_tree_equal(result, expected)


@pytest.fixture
def execution_gate():
    """Queue enough CPU work for the warmed consumer to remain pending."""
    matrix = jnp.ones((4096, 4096), dtype=jnp.float32)

    @jax.jit
    def gate(value, matrix):
        product = jax.lax.fori_loop(0, 4, lambda _, x: (x @ matrix) / 4096, matrix)
        return jnp.where(product[0, 0] > 0, value, -value)

    return lambda value: gate(value, matrix)


@pytest.mark.parametrize("misaligned", [False, True])
def test_cylindrical_points_own_numpy_before_conversion(execution_gate, monkeypatch, misaligned):
    gate = execution_gate
    source = _host_array((1_666_667, 3), misaligned=misaligned)
    saved = source.copy()
    canonical = adapter._canonical_set_points_cyl
    gate(jnp.asarray(saved)).block_until_ready()
    expected = np.array(jax.block_until_ready(adapter._cyl_points_to_cart(canonical(jnp.asarray(saved)))), copy=True)
    monkeypatch.setattr(adapter, "_canonical_set_points_cyl", lambda points: canonical(gate(points)))
    field = BiotSavartJAX([]).set_points_cyl(source)
    _assert_pending(field._points_jax)
    source[:] = -7.0
    np.testing.assert_array_equal(jax.block_until_ready(field._points_jax), expected)


def _kernel_inputs():
    points = _host_array((32, 3))
    points *= 0.2
    gamma = _host_array((1, 64, 3))
    dash = _host_array((1, 64, 3))
    current = _host_array((1,))
    return points, gamma, dash, current


_FORWARD_KERNELS = [
    kernels.biot_savart_B, kernels.biot_savart_A,
    kernels.biot_savart_dB_by_dX, kernels.biot_savart_dA_by_dX,
    kernels.biot_savart_d2B_by_dXdX, kernels.biot_savart_d2A_by_dXdX,
    kernels.biot_savart_B_and_dB,
]


@pytest.mark.parametrize("kernel", _FORWARD_KERNELS, ids=[kernel.__name__ for kernel in _FORWARD_KERNELS])
@pytest.mark.parametrize("pending_input", ["points", "gamma"])
def test_raw_forward_kernel_owns_numpy_inputs(kernel, pending_input, execution_gate):
    gate = execution_gate
    inputs = list(_kernel_inputs())
    expected = jax.tree.map(lambda x: np.array(x, copy=True), jax.block_until_ready(kernel(*inputs)))
    index = 0 if pending_input == "points" else 1
    gate(jnp.asarray(inputs[index])).block_until_ready()
    inputs[index] = gate(jnp.asarray(inputs[index]))
    result = kernel(*inputs)
    _assert_pending(result)
    for source in inputs:
        if isinstance(source, np.ndarray):
            source[:] = -7.0
    _assert_tree_equal(result, expected)


@pytest.mark.parametrize("kernel", [kernels.biot_savart_B_vjp, group_biot_savart_B_vjp], ids=["raw", "grouped"])
def test_raw_pullback_owns_numpy_inputs(kernel, execution_gate):
    gate = execution_gate
    points, gamma, dash, current = _kernel_inputs()
    cotangent = _host_array(points.shape)
    expected = jax.tree.map(lambda x: np.array(x, copy=True), jax.block_until_ready(kernel(points, cotangent, gamma, dash, current)))
    gate(jnp.asarray(points)).block_until_ready()
    pending_points = gate(jnp.asarray(points))
    result = kernel(pending_points, cotangent, gamma, dash, current)
    _assert_pending(result)
    for source in (cotangent, gamma, dash, current):
        source[:] = -7.0
    _assert_tree_equal(result, expected)


def _field():
    curve = create_equally_spaced_curves(1, 1, False, order=1, numquadpoints=64)[0]
    return BiotSavartJAX([Coil(curve, Current(1e5))]).set_points(_host_array((32, 3)) * 0.2)


@pytest.mark.parametrize("entry", ["_normalize_explicit_coil_dofs", "coil_specs_from_dofs", "grouped_coil_arrays_from_dofs", "coil_set_spec_from_dofs"])
def test_explicit_dofs_are_owned(entry, execution_gate, monkeypatch):
    field = _field()
    source = _host_array(field.x.shape)
    source[:] = field.x
    call = getattr(field, entry)
    expected = jax.tree.map(lambda x: np.array(x, copy=True), jax.block_until_ready(call(source)))
    gate = execution_gate
    normalize = field._normalize_explicit_coil_dofs
    gate(normalize(source)).block_until_ready()
    monkeypatch.setattr(field, "_normalize_explicit_coil_dofs", lambda dofs: gate(normalize(dofs)))
    result = call(source)
    # Grouped builders materialize captured curve constants while lowering;
    # the normalization boundary and per-coil payload remain deferred.
    if entry == "coil_specs_from_dofs":
        _assert_pending(result)
    source[:] = -7.0
    _assert_tree_equal(result, expected)


@pytest.mark.parametrize("entry,shape", [
    ("B_pullback_native", (32, 3)), ("A_pullback_native", (32, 3)),
    ("dB_by_dX_pullback_native", (32, 3, 3)), ("dA_by_dX_pullback_native", (32, 3, 3)),
])
def test_adapter_pullback_owns_numpy_cotangent(entry, shape, execution_gate):
    field = _field()
    call = getattr(field, entry)
    source = _host_array(shape)
    expected = jax.tree.map(lambda x: np.array(x, copy=True), jax.block_until_ready(call(source)))
    gate = execution_gate
    gate(field._points_jax).block_until_ready()
    field._points_jax = gate(field._points_jax)
    result = call(source)
    _assert_pending(result)
    source[:] = -7.0
    _assert_tree_equal(result, expected)


@pytest.mark.parametrize("stacked", [False, True])
@pytest.mark.parametrize("use_compute_dtype", [False, True])
def test_group_coil_data_owns_numpy_inputs(stacked, use_compute_dtype, execution_gate, monkeypatch):
    _, gamma, dash, current = _kernel_inputs()
    inputs = (gamma, dash, current) if stacked else ([gamma[0]], [dash[0]], [current[0:1].reshape(())])
    expected = jax.tree.map(lambda x: np.array(x, copy=True), jax.block_until_ready(kernels.group_coil_data(*inputs, use_compute_dtype=use_compute_dtype)))
    gate = execution_gate
    name = "_as_compute_array" if use_compute_dtype else "_as_jax_float64"
    convert = getattr(kernels, name)
    if stacked:
        axis0 = kernels._axis0_entries
        for entry in (gamma, current):
            gate(convert(entry)).block_until_ready()
        monkeypatch.setattr(kernels, "_axis0_entries", lambda x: axis0(gate(convert(x))))
    else:
        for entry in (gamma[0], current[0:1].reshape(())):
            gate(convert(entry)).block_until_ready()
        monkeypatch.setattr(kernels, name, lambda x: gate(convert(x)))
    result = kernels.group_coil_data(*inputs, use_compute_dtype=use_compute_dtype)
    _assert_pending(tuple(group[:3] for group in result))
    for source in (gamma, dash, current):
        source[:] = -7.0
    _assert_tree_equal(result, expected)
