"""Regression coverage for optimizer-vector classification in the target lane."""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from simsopt.geo.optimizer_jax import _is_flat_optimizer_vector


@pytest.mark.parametrize(
    "x0",
    [
        pytest.param(np.arange(3, dtype=np.float64), id="numpy_float64"),
        pytest.param(np.arange(3, dtype=np.int64), id="numpy_int64"),
        pytest.param(jnp.arange(3, dtype=jnp.float64), id="jax_float64"),
        pytest.param(jnp.arange(3, dtype=jnp.bfloat16), id="jax_bfloat16"),
        pytest.param([1.0, np.float64(2.0)], id="numeric_list"),
        pytest.param((np.int64(1), 2.0), id="numeric_tuple"),
    ],
)
def test_is_flat_optimizer_vector_accepts_numeric_flat_inputs(x0):
    assert _is_flat_optimizer_vector(x0) is True


@pytest.mark.parametrize(
    "x0",
    [
        pytest.param(np.array([object()], dtype=object), id="numpy_object"),
        pytest.param(np.array(["1.0", "2.0"]), id="numpy_text"),
        pytest.param(np.array([b"1.0", b"2.0"]), id="numpy_bytes"),
        pytest.param(
            np.array(["2026-05-13", "2026-05-14"], dtype="datetime64[D]"),
            id="numpy_datetime",
        ),
        pytest.param(
            np.array([1, 2], dtype="timedelta64[D]"),
            id="numpy_timedelta",
        ),
        pytest.param(jax.random.split(jax.random.key(0), 2), id="jax_extended_dtype"),
        pytest.param([np.str_("1.0"), np.str_("2.0")], id="numpy_text_scalar_list"),
        pytest.param(
            [np.datetime64("2026-05-13"), np.datetime64("2026-05-14")],
            id="numpy_datetime_scalar_list",
        ),
        pytest.param(
            [np.timedelta64(1, "D"), np.timedelta64(2, "D")],
            id="numpy_timedelta_scalar_list",
        ),
    ],
)
def test_is_flat_optimizer_vector_rejects_non_numeric_flat_inputs(x0):
    assert _is_flat_optimizer_vector(x0) is False


@pytest.mark.parametrize(
    "x0",
    [
        pytest.param(np.arange(4, dtype=np.float64).reshape(2, 2), id="numpy_matrix"),
        pytest.param(jnp.arange(4, dtype=jnp.float64).reshape(2, 2), id="jax_matrix"),
    ],
)
def test_is_flat_optimizer_vector_rejects_non_vector_numeric_arrays(x0):
    assert _is_flat_optimizer_vector(x0) is False
