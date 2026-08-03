"""CPU bitwise gates for batched L-BFGS history operations."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from simsopt_jax.geo.optimizers.private import _lbfgsb_scipy as lbfgsb
from simsopt_jax.geo.optimizers.private._common import (
    private_optimizer_runtime_is_supported,
)

pytestmark = [
    pytest.mark.private_optimizer_runtime,
    pytest.mark.skipif(
        not private_optimizer_runtime_is_supported(jax.__version__),
        reason="L-BFGS-B history kernels are validated on the pinned JAX runtime.",
    ),
    pytest.mark.skipif(
        jax.default_backend() != "cpu",
        reason="History-kernel bitwise batching is currently qualified on CPU only.",
    ),
]


def _assert_same_bits(actual, expected) -> None:
    actual_array = np.asarray(jax.device_get(actual))
    expected_array = np.asarray(jax.device_get(expected))
    if np.issubdtype(actual_array.dtype, np.floating):
        np.testing.assert_array_equal(
            actual_array.view(np.uint64),
            expected_array.view(np.uint64),
        )
    else:
        np.testing.assert_array_equal(actual_array, expected_array)


def _legacy_matupd(
    ws,
    wy,
    sy,
    ss,
    d,
    r,
    itail,
    iupdat,
    col,
    head,
    rr,
    dr,
    stp,
    dtd,
):
    """Retain the element-loop implementation as a test-only bitwise oracle."""
    dtype = jnp.asarray(ws).dtype
    ws = jnp.asarray(ws, dtype=dtype)
    wy = jnp.asarray(wy, dtype=dtype)
    sy = jnp.asarray(sy, dtype=dtype)
    ss = jnp.asarray(ss, dtype=dtype)
    d = jnp.asarray(d, dtype=dtype)
    r = jnp.asarray(r, dtype=dtype)
    itail = jnp.asarray(itail, dtype=jnp.int32)
    iupdat = jnp.asarray(iupdat, dtype=jnp.int32)
    col = jnp.asarray(col, dtype=jnp.int32)
    head = jnp.asarray(head, dtype=jnp.int32)
    rr = jnp.asarray(rr, dtype=dtype)
    dr = jnp.asarray(dr, dtype=dtype)
    stp = jnp.asarray(stp, dtype=dtype)
    dtd = jnp.asarray(dtd, dtype=dtype)
    m = int(ws.shape[0])
    next_col = jnp.where(iupdat <= m, iupdat, col)
    next_itail = jnp.where(iupdat <= m, (head + iupdat - 1) % m, (itail + 1) % m)
    next_head = jnp.where(iupdat <= m, head, (head + 1) % m)
    ws = ws.at[next_itail, :].set(d)
    wy = wy.at[next_itail, :].set(r)
    theta = rr / dr
    rollover = iupdat > m

    def shift_ss_body(j, current_ss):
        active_column = rollover & (j < next_col)

        def shift_offset(offset, shifted_ss):
            value = shifted_ss[offset + 1, j]
            return shifted_ss.at[offset, j - 1].set(
                jnp.where(active_column, value, shifted_ss[offset, j - 1])
            )

        return jax.lax.fori_loop(0, j, shift_offset, current_ss)

    ss = jax.lax.fori_loop(1, m, shift_ss_body, ss)

    def shift_sy_body(j, current_sy):
        def shift_offset(offset, shifted_sy):
            active = rollover & (j < next_col) & (offset < (next_col - j))
            value = shifted_sy[j + offset, j]
            return shifted_sy.at[j - 1 + offset, j - 1].set(
                jnp.where(active, value, shifted_sy[j - 1 + offset, j - 1])
            )

        return jax.lax.fori_loop(0, m - j, shift_offset, current_sy)

    sy = jax.lax.fori_loop(1, m, shift_sy_body, sy)
    row = next_col - 1

    def update_row(j, values):
        current_sy, current_ss = values
        active = j < (next_col - 1)
        pointer = (next_head + j) % m
        sy_value = lbfgsb._lbfgsb_ddot(d, wy[pointer])
        ss_value = lbfgsb._lbfgsb_ddot(ws[pointer], d)
        current_sy = current_sy.at[row, j].set(
            jnp.where(active, sy_value, current_sy[row, j])
        )
        current_ss = current_ss.at[j, row].set(
            jnp.where(active, ss_value, current_ss[j, row])
        )
        return current_sy, current_ss

    sy, ss = jax.lax.fori_loop(0, m - 1, update_row, (sy, ss))
    diagonal = jnp.where(stp == 1.0, dtd, stp * stp * dtd)
    ss = ss.at[row, row].set(diagonal)
    sy = sy.at[row, row].set(dr)
    return ws, wy, sy, ss, next_itail, next_col, next_head, theta


def _legacy_two_loop_direction(state: lbfgsb.LbfgsbState):
    """Retain repeated curvature reductions as a test-only bitwise oracle."""
    n, m = lbfgsb._lbfgsb_state_dimensions(state)
    lws, lwy, lsy, *_ = lbfgsb._lbfgsb_workspace_offsets(n, m)
    wa = state.workspace.wa
    ws = wa[lws:lwy].reshape((m, n))
    wy = wa[lwy:lsy].reshape((m, n))
    head = state.workspace.isave[26]
    col = state.workspace.isave[27]
    theta = state.workspace.dsave[0]
    dtype = state.x.dtype
    positions = jnp.arange(m, dtype=jnp.int32)
    history_indices = (head + positions) % m
    active_history = positions < col
    reverse_indices = history_indices[::-1]
    reverse_active = active_history[::-1]

    def right_product(direction, index_active):
        index, active = index_active
        s_i = ws[index]
        y_i = wy[index]
        s_dot_y = lbfgsb._lbfgsb_ddot(s_i, y_i)
        safe_s_dot_y = jnp.where(active & (s_dot_y != 0.0), s_dot_y, 1.0)
        rho = active.astype(dtype) / safe_s_dot_y
        alpha = rho * lbfgsb._lbfgsb_ddot(s_i, direction)
        return direction - alpha * y_i, alpha

    direction, reverse_alphas = jax.lax.scan(
        right_product,
        -state.g,
        (reverse_indices, reverse_active),
    )
    direction = direction / theta
    alphas = reverse_alphas[::-1]

    def left_product(direction, index_active_alpha):
        index, active, alpha = index_active_alpha
        s_i = ws[index]
        y_i = wy[index]
        s_dot_y = lbfgsb._lbfgsb_ddot(s_i, y_i)
        safe_s_dot_y = jnp.where(active & (s_dot_y != 0.0), s_dot_y, 1.0)
        rho = active.astype(dtype) / safe_s_dot_y
        beta = rho * lbfgsb._lbfgsb_ddot(y_i, direction)
        return direction + (alpha - beta) * s_i, None

    direction, _ = jax.lax.scan(
        left_product,
        direction,
        (history_indices, active_history, alphas),
    )
    return direction


_JITTED_MATUPD = jax.jit(lbfgsb.lbfgsb_matupd)
_JITTED_LEGACY_MATUPD = jax.jit(_legacy_matupd)
_JITTED_TWO_LOOP = jax.jit(lbfgsb.lbfgsb_two_loop_direction)
_JITTED_LEGACY_TWO_LOOP = jax.jit(_legacy_two_loop_direction)


@pytest.mark.parametrize("n", (1, 47, 257, 1000))
@pytest.mark.parametrize(
    "iupdat,col,head,itail",
    (
        (1, 0, 0, 0),
        (7, 6, 0, 5),
        (10, 9, 0, 8),
        (11, 10, 0, 9),
        (14, 10, 3, 2),
    ),
)
def test_matupd_masked_history_shift_matches_element_loops_bitwise(
    n: int,
    iupdat: int,
    col: int,
    head: int,
    itail: int,
) -> None:
    m = 10
    rng = np.random.default_rng(100_000 + 100 * n + iupdat)
    ws = rng.standard_normal((m, n), dtype=np.float64)
    wy = rng.standard_normal((m, n), dtype=np.float64)
    sy = rng.standard_normal((m, m), dtype=np.float64)
    ss = rng.standard_normal((m, m), dtype=np.float64)
    d = rng.standard_normal(n, dtype=np.float64)
    r = rng.standard_normal(n, dtype=np.float64)
    sy[0, -1] = -0.0
    ss[-1, 0] = -0.0
    d[0] = -0.0
    inputs = (
        ws,
        wy,
        sy,
        ss,
        d,
        r,
        itail,
        iupdat,
        col,
        head,
        np.float64(7.25),
        np.float64(3.5),
        np.float64(1.0 if iupdat % 2 else 0.375),
        np.float64(2.75),
    )

    actual = _JITTED_MATUPD(*inputs)
    expected = _JITTED_LEGACY_MATUPD(*inputs)

    for actual_item, expected_item in zip(actual, expected, strict=True):
        _assert_same_bits(actual_item, expected_item)


def _sequential_row_dots(rows, vector):
    values = jnp.zeros((rows.shape[0],), dtype=rows.dtype)

    def body(index, current_values):
        return current_values.at[index].set(lbfgsb._lbfgsb_ddot(vector, rows[index]))

    return jax.lax.fori_loop(0, rows.shape[0], body, values)


def test_matupd_matrix_vector_row_batch_is_not_bitwise_on_cpu() -> None:
    m = 10
    mismatch_count = 0
    for n in (2, 47, 257, 1000):
        rng = np.random.default_rng(900_000 + n)
        rows = rng.standard_normal((m, n), dtype=np.float64)
        vector = rng.standard_normal(n, dtype=np.float64)
        if n == 2:
            vector = np.asarray(
                [-1.6105335681572335, 0.7880123651088977],
                dtype=np.float64,
            )
            rows[0] = np.asarray(
                [1.6255292696770385, -0.7393987894384814],
                dtype=np.float64,
            )
        for head in (0, 3, 9):
            pointers = (head + np.arange(m - 1)) % m
            active_rows = rows[pointers]
            sequential = jax.jit(_sequential_row_dots)(active_rows, vector)
            matrix_vector = jax.jit(lambda matrix, x: matrix @ x)(
                active_rows,
                vector,
            )
            sequential_bits = np.asarray(sequential).view(np.uint64)
            matrix_vector_bits = np.asarray(matrix_vector).view(np.uint64)
            mismatch_count += int(
                np.count_nonzero(sequential_bits != matrix_vector_bits)
            )

    assert mismatch_count > 0


@pytest.mark.parametrize("n", (1, 47, 257, 1000))
@pytest.mark.parametrize("col,head", ((0, 0), (4, 0), (10, 0), (10, 3), (10, 9)))
def test_two_loop_batched_curvatures_match_repeated_reductions_bitwise(
    n: int,
    col: int,
    head: int,
) -> None:
    m = 10
    rng = np.random.default_rng(700_000 + 100 * n + 10 * col + head)
    ws = rng.standard_normal((m, n), dtype=np.float64)
    wy = ws * rng.uniform(0.5, 1.5, size=(m, 1))
    wy += 1.0e-4 * rng.standard_normal((m, n), dtype=np.float64)
    gradient = rng.standard_normal(n, dtype=np.float64)
    state = lbfgsb.lbfgsb_initial_state(
        np.zeros(n, dtype=np.float64),
        m=m,
    )
    lws, lwy, lsy, *_ = lbfgsb._lbfgsb_workspace_offsets(n, m)
    wa = state.workspace.wa
    wa = wa.at[lws:lwy].set(jnp.asarray(ws).reshape((-1,)))
    wa = wa.at[lwy:lsy].set(jnp.asarray(wy).reshape((-1,)))
    workspace = state.workspace._replace(
        wa=wa,
        isave=state.workspace.isave.at[26].set(head).at[27].set(col),
        dsave=state.workspace.dsave.at[0].set(np.float64(1.75)),
    )
    state = state._replace(g=jnp.asarray(gradient), workspace=workspace)

    actual = _JITTED_TWO_LOOP(state)
    expected = _JITTED_LEGACY_TWO_LOOP(state)

    _assert_same_bits(actual, expected)
