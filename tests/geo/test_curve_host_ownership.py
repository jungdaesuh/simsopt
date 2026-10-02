"""Strict host-ownership regressions for native curve geometry."""

from __future__ import annotations

from jax_test_support import (
    fixture_jax_runtime_guard,  # noqa: F401
    parity_device,
)

from pathlib import Path
from typing import Type, Union

import forward_roundoff_bound as fb
import jax
import numpy as np
from boozer_first_stage_roundoff import INVERSE_SQRT_EXTRA_ROUNDINGS
from parity_native_cpu import run_lane_child

from simsopt.geo.curve import kappa_pure
from simsopt.geo.curveobjectives import CurveLength, curve_length_pure
from simsopt.geo.curvexyzfourier import CurveXYZFourier, JaxCurveXYZFourier


CurveType = Union[Type[CurveXYZFourier], Type[JaxCurveXYZFourier]]

_REPO_ROOT = Path(__file__).resolve().parents[2]

# A fresh strict jax-gpu parity lane (``JAX_PLATFORMS=cuda``, transfer guard
# ``disallow``): no host JAX device exists, so ``native_jax_device`` resolves
# to the GPU and both restored kernels run there through explicit transfers.
_CUDA_ONLY_CURVE_LENGTH_CHILD = """\
import sys

import jax
import numpy as np

from simsopt.geo.curveobjectives import CurveLength
from simsopt.geo.curvexyzfourier import CurveXYZFourier

assert jax.config.jax_platforms == "cuda", jax.config.jax_platforms
assert {device.platform for device in jax.devices()} == {"gpu"}, jax.devices()
curve = CurveXYZFourier(int(sys.argv[1]), order=int(sys.argv[2]))
curve.x = np.load(sys.argv[3])
objective = CurveLength(curve)
with jax.transfer_guard("disallow"):
    value = objective.J()
    derivative = objective.dJ()
assert type(value) is np.float64, type(value)
assert type(derivative) is np.ndarray, type(derivative)
np.save(sys.argv[4], np.concatenate(([value], derivative)))
"""


def _unit_circle(curve_type: CurveType) -> Union[CurveXYZFourier, JaxCurveXYZFourier]:
    curve = curve_type(32, order=2)
    curve.set("xc(1)", 1.0)
    curve.set("ys(1)", 1.0)
    return curve


def test_native_curve_length_stays_host_owned_under_strict_transfer_guard() -> None:
    curve = _unit_circle(CurveXYZFourier)
    objective = CurveLength(curve)
    incremental_arclength = curve.incremental_arclength()
    # Upstream's value: ``jnp.mean`` of the arclengths on the host CPU device.
    expected_value = jax.device_get(
        curve_length_pure(jax.device_put(incremental_arclength, jax.devices("cpu")[0]))
    )
    expected_derivative = curve.dincremental_arclength_by_dcoeff_vjp(
        np.full_like(incremental_arclength, 1.0 / incremental_arclength.size)
    )(curve)

    with jax.transfer_guard("disallow"):
        value = objective.J()
        derivative = objective.dJ()

    assert isinstance(value, np.floating)
    np.testing.assert_array_equal(value, expected_value)
    np.testing.assert_allclose(
        derivative, expected_derivative, rtol=1.0e-14, atol=1.0e-14
    )


def test_native_curvature_stays_host_owned_without_changing_jax_curve() -> None:
    native_curve = _unit_circle(CurveXYZFourier)
    jax_curve = _unit_circle(JaxCurveXYZFourier)
    expected_native = (
        np.linalg.norm(
            np.cross(native_curve.gammadash(), native_curve.gammadashdash()),
            axis=1,
        )
        / np.linalg.norm(native_curve.gammadash(), axis=1) ** 3
    )

    with jax.transfer_guard("disallow"):
        native_curvature = native_curve.kappa().copy()

    expected_jax = np.asarray(
        kappa_pure(jax_curve.gammadash(), jax_curve.gammadashdash())
    )
    np.testing.assert_allclose(
        native_curvature, expected_native, rtol=1.0e-15, atol=0.0
    )
    np.testing.assert_allclose(jax_curve.kappa(), expected_jax, rtol=1.0e-15, atol=0.0)


def _curve_length_rounding(curve: CurveXYZFourier) -> fb.Bounded:
    """``CurveLength`` of the curve dofs, traced for any implementation.

    ``mean(norm(gammadash))``: the norms, their mean and the norm's VJP are
    traced from the shared host ``gammadash``, whose bytes both lanes compute
    with the same native code, and the native ``dgammadash_by_dcoeff`` map
    joins them to the dofs (shared host basis, its sums charged by path count).
    The norm's ``1/sqrt`` partial carries the rsqrt charge of
    :mod:`boozer_first_stage_roundoff`, so on GPU the bound is an engineering
    envelope in that constant.
    """
    gammadash = curve.gammadash()
    lengths = fb.extra_roundings(
        fb.norm(fb.seed(gammadash)), INVERSE_SQRT_EXTRA_ROUNDINGS
    )
    length_of_gammadash = fb.stack([fb.mean(lengths, 0)], axis=0)
    jacobian = curve.dgammadash_by_dcoeff().reshape(gammadash.size, -1)
    gammadash_of_dofs = fb.element(
        gammadash.reshape(-1),
        0.0,
        jacobian,
        np.abs(jacobian),
        np.zeros_like(jacobian),
    )
    return fb.compose(length_of_gammadash, gammadash_of_dofs)


def test_native_curve_length_kernels_stay_host_owned_in_a_cuda_only_process(
    tmp_path: Path,
) -> None:
    """Both restored kernels in a process whose JAX initialized CUDA alone.

    ``CurveLength.J`` (upstream's ``jnp.mean``) and ``dJ`` (upstream's jitted
    arclength VJP) must run under the strict transfer guard and return host
    NumPy values (PLAN.md amendment 8, B). Against this process's host-CPU lane
    both stay within twice the one-implementation rounding bound.
    """
    parity_device("gpu")
    curve = _unit_circle(CurveXYZFourier)
    curve.set("zc(2)", 0.3)
    curve.set("xs(2)", 0.1)
    dofs_path = tmp_path / "dofs.npy"
    out_path = tmp_path / "curve_length.npy"
    np.save(dofs_path, curve.x)
    objective = CurveLength(curve)
    with jax.transfer_guard("disallow"):
        expected_value = objective.J()
        expected_derivative = objective.dJ()

    completed = run_lane_child(
        "jax-gpu",
        _CUDA_ONLY_CURVE_LENGTH_CHILD,
        str(curve.quadpoints.size),
        str(curve.order),
        str(dofs_path),
        str(out_path),
        repo_root=_REPO_ROOT,
    )

    assert completed.returncode == 0, completed.stderr
    child = np.load(out_path)
    value_bound, derivative_bound = fb.cross_implementation_bounds(
        _curve_length_rounding(curve)
    )
    assert abs(child[0] - expected_value) <= value_bound[0]
    assert np.all(np.abs(child[1:] - expected_derivative) <= derivative_bound[0]), (
        np.max(np.abs(child[1:] - expected_derivative) - derivative_bound[0])
    )
