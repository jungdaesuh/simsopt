"""Tests for the phi-parametrized JAX Poincare tracer.

The tracer integrates directly in toroidal angle, so the core contract is not
the arc-length step grid used by ``compute_fieldlines``. These tests use
closed-form cylindrical fields where the expected ``R(phi), Z(phi)`` is known,
plus one native-tracer comparison where the two contracts overlap.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from repo_bootstrap import bootstrap_local_simsopt

bootstrap_local_simsopt(_REPO_ROOT / "src")

from simsopt_jax.core.tracing import FieldlineTracingSpec, trace_fieldline
from simsopt_jax.core.tracing_poincare_phi import (
    PoincareTracingSpec,
    _cyl_B_from_cartesian_field,
    trace_poincare_phi,
    trace_poincare_phi_batched,
)
from simsopt_jax_adapters.field.toroidal_field import ToroidalFieldJAX
from simsopt_jax_adapters.field.tracing import (
    compute_poincare_phi,
    poincare_phi_to_phi_hits,
)


def _constant_cylindrical_field(B_R, B_phi, B_Z):
    """Cartesian JAX field with constant cylindrical components."""

    def field(point):
        x, y, _z = point[0], point[1], point[2]
        R = jnp.sqrt(x * x + y * y)
        cos_phi = x / R
        sin_phi = y / R
        return jnp.array(
            [
                B_R * cos_phi - B_phi * sin_phi,
                B_R * sin_phi + B_phi * cos_phi,
                B_Z,
            ],
            dtype=jnp.float64,
        )

    return field


def _pure_toroidal_field(point):
    x, y, _z = point[0], point[1], point[2]
    return jnp.array([-y, x, 0.0], dtype=jnp.float64)


def _spec(**kw):
    values = dict(rtol=1e-9, atol=1e-9, min_step_size=1e-4, max_steps=4000)
    values.update(kw)
    return PoincareTracingSpec(**values)


def test_cylindrical_wrapper_rotates_cartesian_field():
    """Constant Cartesian B rotates to the analytic cylindrical components."""
    field = lambda _point: jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)
    rpz_B = _cyl_B_from_cartesian_field(field)
    phi = jnp.array(np.pi / 3.0)
    got = np.asarray(rpz_B(jnp.array(1.4), phi, jnp.array(-0.2)))
    expected = np.array(
        [
            np.cos(np.pi / 3.0) + 2.0 * np.sin(np.pi / 3.0),
            -np.sin(np.pi / 3.0) + 2.0 * np.cos(np.pi / 3.0),
            3.0,
        ]
    )
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)


def test_closed_surface_invariant_pure_toroidal_field():
    """Pure toroidal field has dR/dphi=dZ/dphi=0 at every saved plane."""
    phis = jnp.linspace(0.0, 4.0 * np.pi, 9)
    result = trace_poincare_phi(
        _spec(),
        jnp.array(1.23),
        jnp.array(-0.17),
        phis,
        _pure_toroidal_field,
    )
    assert int(result.status) == 0
    assert not bool(jnp.any(result.escaped))
    expected = np.column_stack(
        [
            np.full(phis.shape[0], 1.23),
            np.full(phis.shape[0], -0.17),
        ]
    )
    np.testing.assert_allclose(np.asarray(result.punctures), expected, rtol=0, atol=1e-9)


def test_cross_tracer_semantic_alignment_on_pure_toroidal_hits():
    """Native hit alignment is checked after both tracers match the analytic R/Z."""
    r0 = 1.4
    z0 = 0.2
    phis = jnp.array([0.0, 0.5, 1.0, 1.5], dtype=jnp.float64)
    phi_result = trace_poincare_phi(_spec(), r0, z0, phis, _pure_toroidal_field)

    native_spec = FieldlineTracingSpec(
        tmax=1.6,
        rtol=1e-10,
        atol=1e-10,
        max_steps=1000,
        max_phi_hits=16,
    )
    native = trace_fieldline(
        native_spec,
        jnp.array([r0, 0.0, z0], dtype=jnp.float64),
        _pure_toroidal_field,
        phis=phis[1:],
    )
    hits = np.asarray(native.phi_hits[: int(native.phi_hits_count)])
    assert int(native.status) == 0
    assert hits.shape[0] == phis.shape[0] - 1
    for idx in range(hits.shape[0]):
        hit = hits[hits[:, 1] == idx][0]
        native_RZ = np.array([np.hypot(hit[2], hit[3]), hit[4]])
        expected_RZ = np.array([r0, z0])
        np.testing.assert_allclose(native_RZ, expected_RZ, rtol=0, atol=1e-8)
        np.testing.assert_allclose(
            np.asarray(phi_result.punctures[idx + 1]),
            expected_RZ,
            rtol=0,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            np.asarray(phi_result.punctures[idx + 1]),
            native_RZ,
            rtol=0,
            atol=1e-8,
        )


def test_decreasing_phi_grid_preserves_static_outputs():
    """The signed step path supports strictly decreasing unwrapped phi grids."""
    phis = jnp.array([1.0, 0.5, 0.0], dtype=jnp.float64)
    result = trace_poincare_phi(
        _spec(),
        jnp.array(1.25),
        jnp.array(0.1),
        phis,
        _pure_toroidal_field,
    )
    assert int(result.status) == 0
    assert not bool(jnp.any(result.escaped))
    np.testing.assert_allclose(
        np.asarray(result.punctures),
        np.array([[1.25, 0.1], [1.25, 0.1], [1.25, 0.1]]),
        rtol=0,
        atol=1e-9,
    )


def test_phi_rhs_quotient_has_no_pseudotime_sign_multiplier():
    """Negative B_phi reverses dZ/dphi through the quotient itself."""
    phis = jnp.array([0.0, 1.0], dtype=jnp.float64)
    result = trace_poincare_phi(
        _spec(),
        jnp.array(1.0),
        jnp.array(0.0),
        phis,
        _constant_cylindrical_field(0.0, -2.0, 0.5),
    )
    assert int(result.status) == 0
    np.testing.assert_allclose(np.asarray(result.punctures[-1]), [1.0, -0.25], atol=1e-8)


def test_differentiability_gate_matches_central_difference():
    """Gradient through Diffrax punctures is finite and matches an FD oracle."""
    phis = jnp.linspace(0.0, 0.7, 5)
    spec = _spec()

    def final_z(B_Z):
        result = trace_poincare_phi(
            spec,
            jnp.array(1.2),
            jnp.array(-0.1),
            phis,
            _constant_cylindrical_field(0.0, 1.5, B_Z),
        )
        return result.punctures[-1, 1]

    grad = float(jax.grad(final_z)(jnp.array(0.3, dtype=jnp.float64)))
    eps = 1e-5
    fd = (float(final_z(0.3 + eps)) - float(final_z(0.3 - eps))) / (2.0 * eps)
    assert np.isfinite(grad)
    np.testing.assert_allclose(grad, fd, rtol=1e-5, atol=1e-7)


def test_singular_bphi_fails_loudly_without_finite_wrong_punctures():
    """B_phi=0 is singular: output is non-finite and status is non-success."""
    phis = jnp.array([0.0, 0.1, 0.2], dtype=jnp.float64)
    result = trace_poincare_phi(
        _spec(max_steps=32),
        jnp.array(1.0),
        jnp.array(0.0),
        phis,
        _constant_cylindrical_field(1.0, 0.0, 0.0),
    )
    assert int(result.status) != 0
    assert bool(jnp.any(result.escaped) | jnp.any(~jnp.isfinite(result.punctures)))


def test_vmap_batch_equals_per_lane():
    """Batched phi tracing is the lane-wise single tracer under vmap."""
    phis = jnp.linspace(0.0, 1.25, 6)
    field = _constant_cylindrical_field(0.0, 2.0, 0.4)
    r0s = jnp.array([0.9, 1.1, 1.4], dtype=jnp.float64)
    z0s = jnp.array([0.0, -0.2, 0.3], dtype=jnp.float64)
    spec = _spec()
    batched = trace_poincare_phi_batched(spec, r0s, z0s, phis, field)
    for i in range(r0s.shape[0]):
        one = trace_poincare_phi(spec, r0s[i], z0s[i], phis, field)
        assert int(batched.status[i]) == int(one.status)
        np.testing.assert_allclose(
            np.asarray(batched.punctures[i]),
            np.asarray(one.punctures),
            rtol=0,
            atol=1e-8,
        )
        np.testing.assert_array_equal(
            np.asarray(batched.escaped[i]),
            np.asarray(one.escaped),
        )


def test_jit_static_shapes_single_and_batched():
    """Single and batched JIT calls keep static phi-grid output shapes."""
    phis = jnp.linspace(0.0, 1.0, 5)
    field = _constant_cylindrical_field(0.0, 1.0, 0.2)
    spec = _spec()

    single = jax.jit(
        lambda r0, z0: trace_poincare_phi(spec, r0, z0, phis, field).punctures
    )
    batched = jax.jit(
        lambda r0s, z0s: trace_poincare_phi_batched(spec, r0s, z0s, phis, field).punctures
    )

    assert single(jnp.array(1.0), jnp.array(0.0)).shape == (5, 2)
    assert batched(
        jnp.array([1.0, 1.2], dtype=jnp.float64),
        jnp.array([0.0, 0.1], dtype=jnp.float64),
    ).shape == (2, 5, 2)


def test_box_exit_marks_subsequent_planes_escaped():
    """A box event keeps earlier save points finite and marks the tail escaped."""
    phis = jnp.linspace(0.0, 2.0, 9)
    result = trace_poincare_phi(
        _spec(bounds_R=(0.0, 1.2)),
        jnp.array(1.0),
        jnp.array(0.0),
        phis,
        _constant_cylindrical_field(0.2, 1.0, 0.0),
    )
    assert int(result.status) == -1
    assert not bool(result.escaped[0])
    assert bool(result.escaped[-1])
    assert np.isfinite(float(result.punctures[0, 0]))


def test_public_adapter_returns_phi_grid_and_gallery_hits():
    """Adapter accepts the same JAX field style and returns objective-friendly grids."""
    phis = np.linspace(0.0, 2.0 * np.pi, 5)
    R, Z, escaped, status = compute_poincare_phi(
        ToroidalFieldJAX(1.3, 0.8),
        [1.1, 1.4],
        [0.0, 0.2],
        phis,
        rtol=1e-9,
        atol=1e-9,
    )
    assert R.shape == Z.shape == escaped.shape == (5, 2)
    assert status.shape == (2,)
    np.testing.assert_array_equal(status, np.array([0, 0], dtype=np.int32))
    np.testing.assert_allclose(R, np.array([[1.1, 1.4]] * 5), rtol=0, atol=1e-8)
    np.testing.assert_allclose(Z, np.array([[0.0, 0.2]] * 5), rtol=0, atol=1e-8)
    assert not escaped.any()

    hits = poincare_phi_to_phi_hits(phis, R, Z, escaped)
    assert len(hits) == 2
    assert hits[0].shape == (5, 5)
    np.testing.assert_allclose(hits[0][:, 0], phis)


def test_public_adapter_default_steps_accept_short_phi_span():
    """Derived default max_steps must stay positive for tiny valid spans."""
    R, Z, escaped, status = compute_poincare_phi(
        ToroidalFieldJAX(1.3, 0.8),
        [1.1],
        [0.0],
        np.array([0.0, 5.0e-4]),
        rtol=1e-9,
        atol=1e-9,
    )
    assert R.shape == Z.shape == escaped.shape == (2, 1)
    np.testing.assert_array_equal(status, np.array([0], dtype=np.int32))
    assert not escaped.any()


def test_adapter_import_does_not_require_diffrax():
    """Importing the adapter must not load the optional phi/diffrax tracer."""
    code = (
        "import sys;"
        f"sys.path.insert(0, {str(_REPO_ROOT)!r});"
        "from repo_bootstrap import bootstrap_local_simsopt;"
        f"bootstrap_local_simsopt({str(_REPO_ROOT / 'src')!r});"
        "import jax; jax.config.update('jax_enable_x64', True);"
        "import simsopt_jax_adapters.field.tracing;"
        "assert 'diffrax' not in sys.modules, 'adapter import pulled in diffrax';"
        "print('OK')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0 and "OK" in out.stdout, out.stderr


@pytest.mark.parametrize(
    "phis",
    [
        np.array([0.0]),
        np.array([0.0, 0.0]),
        np.array([0.0, 1.0, 0.5]),
        np.zeros((2, 2)),
    ],
)
def test_public_adapter_rejects_non_monotonic_phi_grids(phis):
    with pytest.raises(ValueError, match="phis"):
        compute_poincare_phi(ToroidalFieldJAX(1.3, 0.8), [1.0], [0.0], phis)


def test_core_rejects_concrete_non_monotonic_phi_grid():
    with pytest.raises(ValueError, match="phis"):
        trace_poincare_phi(
            _spec(),
            jnp.array(1.0),
            jnp.array(0.0),
            np.array([0.0, 1.0, 0.5]),
            _pure_toroidal_field,
        )
