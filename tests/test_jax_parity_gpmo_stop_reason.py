"""Upstream's GPMO stop reasons on the counts the providers really return.

Authority: ``src/simsoptpp/permanent_magnet_optimization.cpp`` at
``GPMO_ArbVec_backtracking`` (branch binary; the block quoted below is identical
to upstream ``9e027eac38028d57aa23777be52a781aa860e347``):

* ``:783`` ``num_nonzeros(0) = num_nonzero`` is written before the loop, then
  ``print_GPMO`` writes objective row 0 and increments ``print_iter``;
* ``:942-959`` the verbose record calls ``print_GPMO`` and then writes
  ``num_nonzeros(print_iter - 1)``, so objective rows and written counts stay in
  lockstep, and tests the never-written slot ``num_nonzeros(print_iter)``;
* ``:964-985`` the magnet-limit / grid-full exit calls ``print_GPMO`` and writes
  **no** count, so on that exit the written counts are one shorter than the
  recorded objective history.

Measured on this tree, bounded MUSE native lane (K=100, nhistory=20, cap 20):
objective prefix 6 rows, ``num_nonzeros`` = ``[0, 1, 6, 11, 16, 0, 0, ...]`` --
five written entries and an unwritten zero inside the trimmed prefix.
"""

from __future__ import annotations

import numpy as np
import pytest
from examples.jax.parity.cases._fixed_work_status import (
    GPMO_ITERATION_COUNT_COMPLETED,
    GPMO_MAGNET_CAP_REACHED,
    GPMO_NONZERO_COUNT_STALLED,
    gpmo_stop_reason,
)
from examples.jax.parity.cases._permanent_magnet_arbvec import _observation
from examples.jax.parity.input_bundle import InputBundle

#: The bounded MUSE grid, so the two limits are the ones the lanes really carry.
GRID_SIZE = 756
MAGNET_CAP = 20


def _bundle() -> InputBundle:
    return InputBundle(
        schema_version=2,
        case_id="native-permanent-magnet-muse",
        scale="bounded",
        random_seed=0,
        configuration={"ndipoles": GRID_SIZE, "max_magnets": MAGNET_CAP},
        configuration_fingerprint="configuration",
        arrays={},
        input_fingerprint="input",
    )


def _degenerate_values() -> dict[str, np.ndarray]:
    """Finite moments, no magnet placed: the only shape the stall exit can take."""
    return {
        "final:moments": np.zeros((GRID_SIZE, 3), dtype=np.float64),
        "final:nonzero_mask": np.zeros(GRID_SIZE, dtype=bool),
    }


def test_the_magnet_limit_row_carries_no_recorded_count() -> None:
    """A capped run's trimmed counts end in an unwritten zero, not a record.

    The sequence below is upstream-reachable: the stall ``break`` does not fire
    at the record that writes the final ``0`` because the slot before it is
    ``3``, and the cap is then reached in the following iterations, which adds
    one objective row and no count. Reading the last two entries of the trimmed
    array as "the two tested slots" therefore names an exit the run never took.
    """

    written = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 3, 0], dtype=np.int64)
    trimmed = np.concatenate((written, np.zeros(1, dtype=np.int64)))

    assert (
        gpmo_stop_reason(
            nonzero_count=MAGNET_CAP,
            grid_size=GRID_SIZE,
            magnet_cap=MAGNET_CAP,
            recorded_nonzero_counts=trimmed,
        )
        == GPMO_MAGNET_CAP_REACHED
    )
    # The stall exit itself is unchanged where it really is reachable: the run
    # left the loop at the verbose record, so no unwritten slot was appended.
    assert (
        gpmo_stop_reason(
            nonzero_count=0,
            grid_size=GRID_SIZE,
            magnet_cap=MAGNET_CAP,
            recorded_nonzero_counts=np.zeros(11, dtype=np.int64),
        )
        == GPMO_NONZERO_COUNT_STALLED
    )


def test_a_full_grid_outranks_the_cap_as_it_does_in_the_c_plus_plus() -> None:
    """``num_nonzero >= N`` is reported first inside the one limit block."""
    assert (
        gpmo_stop_reason(
            nonzero_count=GRID_SIZE,
            grid_size=GRID_SIZE,
            magnet_cap=MAGNET_CAP,
            recorded_nonzero_counts=np.zeros(11, dtype=np.int64),
        )
        == "grid_full"
    )


def test_a_provider_without_the_stall_exit_is_never_labelled_with_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the native kernel has upstream's fourth exit.

    ``GPMO_ArbVec_backtracking_jax`` (``simsopt_jax/core/pm_optimization.py``)
    has no "nonzero count unchanged" branch at all: at the same configuration it
    runs to ``K``. The two kernels also record different numbers of rows -- at
    bounded MUSE the native lane records 6 and the JAX kernel 20 -- so a shared
    derivation would make upstream's exit unreachable on the lane that has it
    and reachable on the lane that does not.
    """

    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_cpu_parity")
    bundle = _bundle()
    stalled = np.zeros(11, dtype=np.int64)
    native = _observation(
        "native-cpu",
        bundle,
        {},
        _degenerate_values(),
        ("run_arbitrary_vector_backtracking_gpmo",),
        driver="simsoptpp_gpmo_arbvec_backtracking",
        platform="cpu",
        precision="fp64",
        recorded_nonzero_counts=stalled,
    )
    jax_lane = _observation(
        "jax-cpu",
        bundle,
        {},
        _degenerate_values(),
        ("run_arbitrary_vector_backtracking_gpmo",),
        driver="simsopt_jax_gpmo_arbvec_backtracking",
        platform="cpu",
        precision="fp64",
        recorded_nonzero_counts=stalled,
    )

    assert native.raw_status == GPMO_NONZERO_COUNT_STALLED
    assert jax_lane.raw_status == GPMO_ITERATION_COUNT_COMPLETED
    # Whatever the stop reason, a run that placed no magnet returned the moments
    # it was given: neither lane is a success.
    assert native.success is False
    assert jax_lane.success is False
    assert native.normalized_status == jax_lane.normalized_status == "failed"
