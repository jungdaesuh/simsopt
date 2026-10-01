"""The MUSE parity case must judge the snapshot its post stages consume.

Official ``examples/2_Intermediate/permanent_magnet_MUSE.py:184-185`` replaces
``pm_opt.m`` with ``m_history[:, :, argmin(R2_history)]`` before the printed
volume, the dipole squared flux on the plotting surface and the total magnet
volume. The case follows that rule, so ``final:selected_history_*`` and every
``final:post_*`` observable are computed from the SELECTED snapshot while the
lane label was derived from the GPMO endpoint alone -- a non-finite or
degenerate selected row left ``success=True`` and published NaN post metrics.
The mirror example was fixed in wave 2 (round-1 finding lens2-pm-muse-status-2);
this is its parity twin.

The lane is driven with a stand-in provider result rather than a real solve:
the defect is in the labelling of a snapshot, and a GPMO run cannot be made to
return a NaN row on demand.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

from types import SimpleNamespace

import numpy as np
import pytest
import simsopt_jax_adapters.examples.muse as muse
import examples.jax.parity.cases.native_permanent_magnet_muse as muse_case
from examples.jax.parity.arbiter import LaneObservation
from examples.jax.parity.cases._permanent_magnet_arbvec import ArbVecLaneResult

#: Two recorded rows with a rising objective, the official PM4Stell shape, so
#: ``argmin`` selects row 0 and the endpoint is row 1.
RISING_OBJECTIVE_HISTORY = np.asarray([0.16177, 0.83456], dtype=np.float64)
ENDPOINT_MOMENT = 11.0


def _moment_history(selected_row: tuple[float, float, float]) -> np.ndarray:
    """``(ndipoles, 3, rows)``: row 0 is the selection, row 1 the endpoint."""
    history = np.empty((1, 3, 2), dtype=np.float64)
    history[0, :, 0] = np.asarray(selected_row, dtype=np.float64)
    history[0, :, 1] = ENDPOINT_MOMENT
    return history


def _observation(lane: str) -> LaneObservation:
    return LaneObservation(
        lane=lane,
        backend_mode="native_cpu" if lane == "native-cpu" else "jax_cpu_parity",
        platform="cpu",
        precision="fp64",
        scale="bounded",
        input_fingerprint="input",
        configuration_fingerprint="configuration",
        effective_construction_fingerprint="construction",
        driver="arbvec",
        normalized_status="not_applicable",
        raw_status="magnet_cap_reached",
        success=True,
        nit=None,
        nfev=None,
        njev=None,
        completed_workflow_stages=muse_case.GPMO_WORKFLOW_STAGES,
        provenance=None,
        values={"final:moments": np.full((1, 3), ENDPOINT_MOMENT, dtype=np.float64)},
    )


def _bundle() -> SimpleNamespace:
    return SimpleNamespace(
        configuration={
            "nphi": 2,
            "ntheta": 2,
            "coordinate_flag": "cartesian",
            "ndipoles": 1,
            "max_magnets": 1,
        },
        scale="bounded",
    )


def _arrays() -> dict[str, np.ndarray]:
    return {
        "dipole_grid_xyz": np.zeros((1, 3), dtype=np.float64),
        "moment_maxima": np.ones(1, dtype=np.float64),
        "response_matrix": np.eye(3, dtype=np.float64),
        "target": np.zeros(3, dtype=np.float64),
    }


def _install(
    monkeypatch: pytest.MonkeyPatch,
    selected_row: tuple[float, float, float],
) -> None:
    moment_history = _moment_history(selected_row)
    monkeypatch.setattr(
        muse_case,
        "execute_arbvec_case",
        lambda lane, _bundle, _arrays, _stages: ArbVecLaneResult(
            observation=_observation(lane),
            objective_history=RISING_OBJECTIVE_HISTORY,
            moment_history=moment_history,
        ),
    )
    monkeypatch.setattr(
        muse_case,
        "build_muse_post_geometry",
        lambda *_args, **_kwargs: object(),
    )

    def diagnostics(
        _geometry: object,
        _policy: object,
        final_moments: np.ndarray,
        _dipole_grid_xyz: np.ndarray,
        _moment_maxima: np.ndarray,
        *,
        coordinate_flag: str,
    ) -> muse.MusePostDiagnostics:
        selected = np.asarray(final_moments, dtype=np.float64)
        return muse.MusePostDiagnostics(
            dipole_squared_flux=float(np.sum(selected**2)),
            total_magnet_volume=float(np.sum(np.abs(selected))),
            trace_statuses=(-1,),
            trace_final_times=(1.0,),
            trace_hit_counts=(1,),
            trace_final_steps=(0.1,),
        )

    monkeypatch.setattr(muse_case, "run_native_muse_post_workflow", diagnostics)
    monkeypatch.setattr(muse_case, "run_jax_muse_post_workflow", diagnostics)


def test_a_non_finite_selected_snapshot_fails_the_lane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The endpoint is finite; the row the post stages consumed is not."""
    _install(monkeypatch, (float("nan"), 0.0, 0.0))

    observation = muse_case.execute("native-cpu", _bundle(), _arrays())

    assert np.all(np.isfinite(observation.values["final:moments"]))
    assert observation.success is False
    assert observation.normalized_status == "failed"
    # The provider's own stop reason is kept: the run did reach the cap.
    assert observation.raw_status == "magnet_cap_reached"


def test_a_degenerate_selected_snapshot_fails_the_lane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A selected row with no magnet at all is the ``x_init`` it started from."""
    _install(monkeypatch, (0.0, 0.0, 0.0))

    observation = muse_case.execute("native-cpu", _bundle(), _arrays())

    assert observation.success is False
    assert observation.normalized_status == "failed"


def test_a_usable_selected_snapshot_keeps_the_provider_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The control: a finite, non-degenerate selection is unaffected."""
    _install(monkeypatch, (7.0, 7.0, 7.0))

    observation = muse_case.execute("native-cpu", _bundle(), _arrays())

    assert observation.success is True
    assert observation.normalized_status == "not_applicable"
    assert observation.raw_status == "magnet_cap_reached"
    # 3 * 7^2 -- the selected row, not the 11.0 endpoint.
    assert float(observation.values["final:post_dipole_squared_flux"]) == 147.0
