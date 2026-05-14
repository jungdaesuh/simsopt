"""JAX-native LinkingNumber kernel and wrapper-gate tests."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import simsopt.geo.curveobjectives as curveobjectives_module
from simsopt.geo.curveobjectives import LinkingNumber
from simsopt.geo.curve import create_equally_spaced_curves
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.jax_core.curve_geometry import pair_linking_number_pure

import simsoptpp as sopp


def _two_orthogonally_linked_circles(numquadpoints: int = 64):
    """Build a linked-1 perpendicular pair (XY-plane circle + XZ-plane circle)."""
    qp = np.linspace(0.0, 1.0, numquadpoints, endpoint=False, dtype=np.float64)
    angle = 2.0 * math.pi * qp
    radius_a = 1.0
    radius_b = 0.5
    a_gamma = np.stack(
        [radius_a * np.cos(angle), radius_a * np.sin(angle), np.zeros_like(angle)],
        axis=1,
    )
    a_gammadash = np.stack(
        [
            -2.0 * math.pi * radius_a * np.sin(angle),
            2.0 * math.pi * radius_a * np.cos(angle),
            np.zeros_like(angle),
        ],
        axis=1,
    )
    b_gamma = np.stack(
        [
            radius_b * np.cos(angle) + radius_a,
            np.zeros_like(angle),
            radius_b * np.sin(angle),
        ],
        axis=1,
    )
    b_gammadash = np.stack(
        [
            -2.0 * math.pi * radius_b * np.sin(angle),
            np.zeros_like(angle),
            2.0 * math.pi * radius_b * np.cos(angle),
        ],
        axis=1,
    )
    dphi = qp[1] - qp[0]
    return a_gamma, a_gammadash, b_gamma, b_gammadash, dphi


def _two_far_apart_circles(numquadpoints: int = 64):
    """Build an unlinked pair (two coaxial XY circles)."""
    qp = np.linspace(0.0, 1.0, numquadpoints, endpoint=False, dtype=np.float64)
    angle = 2.0 * math.pi * qp
    a_gamma = np.stack([np.cos(angle), np.sin(angle), np.zeros_like(angle)], axis=1)
    a_gammadash = np.stack(
        [
            -2.0 * math.pi * np.sin(angle),
            2.0 * math.pi * np.cos(angle),
            np.zeros_like(angle),
        ],
        axis=1,
    )
    b_gamma = a_gamma + np.array([5.0, 0.0, 0.0], dtype=np.float64)
    b_gammadash = a_gammadash.copy()
    dphi = qp[1] - qp[0]
    return a_gamma, a_gammadash, b_gamma, b_gammadash, dphi


def _two_skew_circles_near_coplanar(numquadpoints: int = 96):
    """Two circles tilted slightly off coplanar but still linked = 1."""
    qp = np.linspace(0.0, 1.0, numquadpoints, endpoint=False, dtype=np.float64)
    angle = 2.0 * math.pi * qp
    a_gamma = np.stack([np.cos(angle), np.sin(angle), np.zeros_like(angle)], axis=1)
    a_gammadash = np.stack(
        [
            -2.0 * math.pi * np.sin(angle),
            2.0 * math.pi * np.cos(angle),
            np.zeros_like(angle),
        ],
        axis=1,
    )
    # Second circle tilted by a small angle around the x-axis, passing through the first.
    tilt = 0.05  # near-coplanar
    cos_t, sin_t = math.cos(tilt), math.sin(tilt)
    b_local = np.stack(
        [0.5 * np.cos(angle) + 1.0, 0.5 * np.sin(angle), np.zeros_like(angle)], axis=1
    )
    b_local_dash = np.stack(
        [
            -2.0 * math.pi * 0.5 * np.sin(angle),
            2.0 * math.pi * 0.5 * np.cos(angle),
            np.zeros_like(angle),
        ],
        axis=1,
    )
    rot = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_t, -sin_t],
            [0.0, sin_t, cos_t],
        ],
        dtype=np.float64,
    )
    b_gamma = b_local @ rot.T
    b_gammadash = b_local_dash @ rot.T
    dphi = qp[1] - qp[0]
    return a_gamma, a_gammadash, b_gamma, b_gammadash, dphi


def _cpp_pair_linking_number(g1, g1d, g2, g2d, dphi):
    """Run sopp.compute_linking_number on the single pair and return the integer."""
    return int(
        sopp.compute_linking_number(
            [np.asarray(g2, dtype=np.float64), np.asarray(g1, dtype=np.float64)],
            [np.asarray(g2d, dtype=np.float64), np.asarray(g1d, dtype=np.float64)],
            np.array([dphi, dphi], dtype=np.float64),
            1.0,
        )
    )


def _stage_pair(g1, g1d, g2, g2d, dphi):
    """Place pair inputs on the device explicitly to satisfy strict transfer guards."""
    return (
        jax.device_put(np.asarray(g1, dtype=np.float64)),
        jax.device_put(np.asarray(g1d, dtype=np.float64)),
        jax.device_put(np.asarray(g2, dtype=np.float64)),
        jax.device_put(np.asarray(g2d, dtype=np.float64)),
        jax.device_put(np.float64(dphi)),
    )


def test_pair_linking_number_matches_cpp_for_linked_pair():
    g1, g1d, g2, g2d, dphi = _two_orthogonally_linked_circles()
    cpp_value = _cpp_pair_linking_number(g1, g1d, g2, g2d, dphi)
    g1_d, g1d_d, g2_d, g2d_d, dphi_d = _stage_pair(g1, g1d, g2, g2d, dphi)
    jax_value = int(pair_linking_number_pure(g1_d, g1d_d, g2_d, g2d_d, dphi_d, dphi_d))
    assert cpp_value == 1
    assert jax_value == cpp_value


def test_pair_linking_number_matches_cpp_for_unlinked_pair():
    g1, g1d, g2, g2d, dphi = _two_far_apart_circles()
    cpp_value = _cpp_pair_linking_number(g1, g1d, g2, g2d, dphi)
    g1_d, g1d_d, g2_d, g2d_d, dphi_d = _stage_pair(g1, g1d, g2, g2d, dphi)
    jax_value = int(pair_linking_number_pure(g1_d, g1d_d, g2_d, g2d_d, dphi_d, dphi_d))
    assert cpp_value == 0
    assert jax_value == cpp_value


def test_pair_linking_number_matches_cpp_for_near_degenerate_pair():
    g1, g1d, g2, g2d, dphi = _two_skew_circles_near_coplanar()
    cpp_value = _cpp_pair_linking_number(g1, g1d, g2, g2d, dphi)
    g1_d, g1d_d, g2_d, g2d_d, dphi_d = _stage_pair(g1, g1d, g2, g2d, dphi)
    jax_value = int(pair_linking_number_pure(g1_d, g1d_d, g2_d, g2d_d, dphi_d, dphi_d))
    assert jax_value == cpp_value


def _two_inverted_linked_circles(numquadpoints: int = 64):
    """Mirror of ``_two_orthogonally_linked_circles`` but with curve A
    traversed in the *reverse* direction. The Gauss linking integral is
    sign-anti-symmetric in curve orientation, so the CPU oracle returns
    ``|link| = 1`` (the C++ ``compute_linking_number`` applies
    ``round(abs(total) / (4π))``, dropping the sign), giving the JAX
    kernel an opposite-orientation parity case alongside the standard
    linked / unlinked / near-coplanar fixtures.
    """
    a_gamma, a_gammadash, b_gamma, b_gammadash, dphi = _two_orthogonally_linked_circles(
        numquadpoints
    )
    return a_gamma[::-1].copy(), -a_gammadash[::-1].copy(), b_gamma, b_gammadash, dphi


def test_pair_linking_number_matches_cpp_for_reversed_orientation():
    """Reversed-orientation linked pair: oracle and JAX agree on |link|."""
    g1, g1d, g2, g2d, dphi = _two_inverted_linked_circles()
    cpp_value = _cpp_pair_linking_number(g1, g1d, g2, g2d, dphi)
    g1_d, g1d_d, g2_d, g2d_d, dphi_d = _stage_pair(g1, g1d, g2, g2d, dphi)
    jax_value = int(pair_linking_number_pure(g1_d, g1d_d, g2_d, g2d_d, dphi_d, dphi_d))
    # The C++ algorithm reports the absolute value of the Gauss integral
    # rounded to an integer; reversing one curve's orientation flips the
    # signed integral but leaves the magnitude (= 1) unchanged.
    assert cpp_value == 1
    assert jax_value == cpp_value


def test_pair_linking_number_returns_integer_jax_array():
    g1, g1d, g2, g2d, dphi = _two_orthogonally_linked_circles()
    g1_d, g1d_d, g2_d, g2d_d, dphi_d = _stage_pair(g1, g1d, g2, g2d, dphi)
    result = pair_linking_number_pure(g1_d, g1d_d, g2_d, g2d_d, dphi_d, dphi_d)
    assert isinstance(result, jax.Array)
    assert jnp.issubdtype(result.dtype, jnp.integer)
    assert result.shape == ()


def test_pair_linking_number_runs_under_strict_transfer_guard():
    g1, g1d, g2, g2d, dphi = _two_orthogonally_linked_circles()
    g1_d, g1d_d, g2_d, g2d_d, dphi_d = _stage_pair(g1, g1d, g2, g2d, dphi)
    g1_d.block_until_ready()
    g1d_d.block_until_ready()
    g2_d.block_until_ready()
    g2d_d.block_until_ready()
    dphi_d.block_until_ready()
    with jax.transfer_guard("disallow"):
        value = pair_linking_number_pure(g1_d, g1d_d, g2_d, g2d_d, dphi_d, dphi_d)
        value.block_until_ready()


def test_linking_number_objective_matches_cpp_multi_curve(monkeypatch):
    """LinkingNumber.J on JAX backend matches C++ across multi-curve sets."""
    monkeypatch.setattr(curveobjectives_module, "is_jax_backend", lambda: True)

    cases = []
    # Case 1: stellsym base curves -> linking 0.
    curves1 = create_equally_spaced_curves(
        2, 1, stellsym=True, R0=1, R1=0.5, order=5, numquadpoints=120
    )
    cases.append((curves1, 0))

    # Case 2: explicit linked pair -> linking 1.
    curve_a = CurveXYZFourier(200, 3)
    coeffs = curve_a.dofs_matrix
    coeffs[1][0] = 1.0
    coeffs[1][1] = 0.5
    coeffs[2][2] = 0.5
    curve_a.set_dofs(np.concatenate(coeffs))

    curve_b = CurveXYZFourier(150, 3)
    coeffs = curve_b.dofs_matrix
    coeffs[1][0] = 0.5
    coeffs[1][1] = 0.5
    coeffs[0][0] = 0.1
    coeffs[0][1] = 0.5
    coeffs[0][2] = 0.5
    curve_b.set_dofs(np.concatenate(coeffs))
    cases.append(([curve_a, curve_b], 1))

    # Case 3: triple-curve set (link sum across 3 pairs).
    curves3 = list(curves1) + [curve_a]
    cases.append((curves3, None))  # let CPU define the expected value

    for curves, expected in cases:
        objective = LinkingNumber(curves)

        def reject_cpp(*_args, **_kwargs):
            raise AssertionError("CPU sopp path must not run when backend is JAX")

        monkeypatch.setattr(
            curveobjectives_module.sopp, "compute_linking_number", reject_cpp
        )
        jax_value = objective.J()

        # Restore CPU oracle path for parity comparison.
        monkeypatch.setattr(
            curveobjectives_module.sopp,
            "compute_linking_number",
            sopp.compute_linking_number,
        )
        monkeypatch.setattr(curveobjectives_module, "is_jax_backend", lambda: False)
        cpu_value = LinkingNumber(curves).J()
        monkeypatch.setattr(curveobjectives_module, "is_jax_backend", lambda: True)

        assert isinstance(jax_value, int)
        assert jax_value == cpu_value
        if expected is not None:
            assert jax_value == expected


def test_linking_number_objective_matches_cpp_with_downsample(monkeypatch):
    """Downsample path on JAX backend matches C++ across stride values."""
    curves = create_equally_spaced_curves(
        3, 1, stellsym=True, R0=1, R1=0.5, order=5, numquadpoints=120
    )
    for downsample in (1, 2, 5):
        monkeypatch.setattr(curveobjectives_module, "is_jax_backend", lambda: False)
        cpu_value = LinkingNumber(curves, downsample).J()
        monkeypatch.setattr(curveobjectives_module, "is_jax_backend", lambda: True)
        jax_value = LinkingNumber(curves, downsample).J()
        assert isinstance(jax_value, int)
        assert jax_value == cpu_value


def test_linking_number_objective_cpu_oracle_path_uses_sopp(monkeypatch):
    """When is_jax_backend()=False, the C++ path is invoked."""
    monkeypatch.setattr(curveobjectives_module, "is_jax_backend", lambda: False)
    calls = {"count": 0}
    original = sopp.compute_linking_number

    def counting_cpp(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        curveobjectives_module.sopp, "compute_linking_number", counting_cpp
    )
    curves = create_equally_spaced_curves(
        2, 1, stellsym=True, R0=1, R1=0.5, order=5, numquadpoints=120
    )
    objective = LinkingNumber(curves)
    objective.J()
    assert calls["count"] == 1


def test_linking_number_objective_raises_under_target_lane_bypass(monkeypatch):
    """The wrapper raises when the strict target-lane purity guard is active."""
    from simsopt.backend.runtime import strict_target_lane_purity

    monkeypatch.setattr(curveobjectives_module, "is_jax_backend", lambda: True)
    monkeypatch.setenv("SIMSOPT_TARGET_LANE_STRICT", "1")
    curves = create_equally_spaced_curves(
        2, 1, stellsym=True, R0=1, R1=0.5, order=5, numquadpoints=64
    )
    objective = LinkingNumber(curves)
    with strict_target_lane_purity():
        with pytest.raises(RuntimeError, match="target-lane bypass: LinkingNumber.J"):
            objective.J()
