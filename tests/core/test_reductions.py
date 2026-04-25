import math

import numpy as np
import pytest

import jax
import jax.numpy as jnp
from benchmarks.validation_ladder_contract import parity_ladder_tolerances

from simsopt.jax_core.reductions import (
    compensated_sum_flat,
    pairwise_sum_axis,
    pairwise_sum_flat,
    scalar_square_sum,
    validate_reduction_mode,
)

_CPU_GPU_REDUCTION_TOLS = parity_ladder_tolerances("reduction_cpu_gpu")


def _numpy_pairwise_sum_axis(array, *, axis: int):
    axis_index = axis if axis >= 0 else array.ndim + axis
    axis_size = array.shape[axis_index]
    if axis_size == 0:
        return np.sum(array, axis=axis_index, dtype=np.float64)

    reduced = np.moveaxis(np.asarray(array, dtype=np.float64), axis_index, 0)
    padded_size = 1 << (axis_size - 1).bit_length()
    padded = np.zeros((padded_size,) + reduced.shape[1:], dtype=np.float64)
    padded[:axis_size] = reduced
    while padded.shape[0] > 1:
        pair_shape = (padded.shape[0] // 2, 2) + padded.shape[1:]
        paired = padded.reshape(pair_shape)
        padded = paired[:, 0, ...] + paired[:, 1, ...]
    return np.squeeze(padded, axis=0)


def _assert_strict_oracle_improves_reference(*, strict_oracle, default, reference):
    np.testing.assert_allclose(strict_oracle, reference, rtol=0.0, atol=0.0)
    assert abs(strict_oracle - reference) < abs(default - reference)


def _first_device_for_platform(platform: str):
    for device in jax.devices():
        if device.platform == platform:
            return device
    return None


def test_validate_reduction_mode_rejects_unknown_value():
    with pytest.raises(ValueError, match="Unknown reduction_mode"):
        validate_reduction_mode("not-a-mode")


def test_pairwise_sum_axis_matches_numpy_fixed_tree():
    values = np.arange(3 * 5 * 7, dtype=np.float64).reshape(3, 5, 7) - 20.0

    actual = np.asarray(pairwise_sum_axis(jnp.asarray(values), axis=1))
    expected = _numpy_pairwise_sum_axis(values, axis=1)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_pairwise_sum_flat_matches_numpy_fixed_tree():
    values = np.linspace(-7.5, 11.0, num=257, dtype=np.float64)

    actual = float(pairwise_sum_flat(jnp.asarray(values)))
    expected = float(_numpy_pairwise_sum_axis(values[None, :], axis=1)[0])

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_compensated_sum_flat_recovers_small_positive_terms():
    values = np.ones(10001, dtype=np.float64)
    values[0] = 1.0e16

    compensated = float(compensated_sum_flat(jnp.asarray(values)))
    reference = math.fsum(float(value) for value in values)
    plain = float(jnp.sum(jnp.asarray(values)))

    np.testing.assert_allclose(compensated, reference, rtol=0.0, atol=0.0)
    assert abs(compensated - reference) < abs(plain - reference)


def test_scalar_square_sum_strict_oracle_beats_default_vdot_on_dynamic_range():
    residual = np.ones(10001, dtype=np.float64)
    residual[0] = 1.0e8

    strict_oracle = float(
        scalar_square_sum(
            jnp.asarray(residual),
            reduction_mode="strict_oracle",
            default="vdot",
        )
    )
    default = float(
        scalar_square_sum(
            jnp.asarray(residual),
            reduction_mode="default",
            default="vdot",
        )
    )
    reference = math.fsum(float(value * value) for value in residual)

    _assert_strict_oracle_improves_reference(
        strict_oracle=strict_oracle,
        default=default,
        reference=reference,
    )


def test_pairwise_and_compensated_reductions_match_cpu_gpu_on_cancellation_stress():
    cpu_device = _first_device_for_platform("cpu")
    gpu_device = _first_device_for_platform("gpu") or _first_device_for_platform("cuda")
    if cpu_device is None or gpu_device is None:
        pytest.xfail("CPU and CUDA/GPU devices are required for reduction_cpu_gpu")

    values = np.ones((3, 257), dtype=np.float64)
    values[0, 0] = 1.0e16
    values[0, 1] = -1.0e16
    values[1, 0] = -1.0e12
    values[1, 1] = 1.0e12
    values[2, ::2] *= -1.0

    def evaluate_on_device(device):
        with jax.default_device(device):
            array = jnp.asarray(values)
            return (
                np.asarray(pairwise_sum_axis(array, axis=1)),
                float(pairwise_sum_flat(array)),
                float(compensated_sum_flat(array.reshape((-1,)))),
            )

    cpu_axis, cpu_flat, cpu_compensated = evaluate_on_device(cpu_device)
    gpu_axis, gpu_flat, gpu_compensated = evaluate_on_device(gpu_device)

    rtol = float(_CPU_GPU_REDUCTION_TOLS["rtol"])
    atol = float(_CPU_GPU_REDUCTION_TOLS["atol"])
    np.testing.assert_allclose(gpu_axis, cpu_axis, rtol=rtol, atol=atol)
    np.testing.assert_allclose(gpu_flat, cpu_flat, rtol=rtol, atol=atol)
    np.testing.assert_allclose(gpu_compensated, cpu_compensated, rtol=rtol, atol=atol)
