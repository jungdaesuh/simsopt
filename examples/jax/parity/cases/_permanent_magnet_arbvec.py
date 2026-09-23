"""Shared exact-lane execution for fixed-grid arbitrary-vector GPMO mirrors.

Every lane runs with the official keyword arguments. ``initialize_default_kwargs``
sets ``verbose=True`` (``simsopt/util/permanent_magnet_helper_functions.py:250``)
and every official permanent-magnet script uses it, so the lanes record the same
history upstream records and the reachable stop set is upstream's, including the
fourth exit at ``permanent_magnet_optimization.cpp:948-959``.

The two providers record on different iteration grids, which is a property of
their kernels and is stated here rather than papered over:

* native, ``GPMO_ArbVec_backtracking``: one unconditional row for the initial
  state (``:790-792``), then ``k in {0, P, 2P, ...}`` and ``k == K - 1`` under
  ``verbose`` with ``P = int(K / nhistory)`` (``:942``), then one unconditional
  row at the magnet-limit break (``:965-985``);
* JAX, ``gpmo_arbvec_backtracking_solve(record_every=P)``: rows
  ``k in {P - 1, 2P - 1, ...} + {K - 1}``
  (``simsopt_jax/core/pm_optimization.py:564-568``), with the post-``done``
  carry repeating the endpoint.

Both grids therefore always contain the run endpoint, and they coincide when
``P == 1``. For ``P > 1`` they are offset by one iteration, so a non-monotone
objective history could select different snapshots; the selection objective is
published so that such a divergence is a failing comparison, not a silent one.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

import numpy as np
from examples.jax.parity.arbiter import LaneObservation
from examples.jax.parity.cases._fixed_work_status import (
    fixed_work_label,
    gpmo_stop_reason,
)
from examples.jax.parity.input_bundle import (
    InputBundle,
    effective_construction_fingerprint,
)
from examples.jax.parity.runtime import ParityLane
from simsopt_jax_adapters.examples.gpmo_rules import (
    gpmo_backtracking_outputs_usable,
    gpmo_history_period,
    recorded_history_length,
)


@dataclass(frozen=True)
class ArbVecLaneResult:
    """One lane's observation plus the history its provider recorded.

    The history is not published: the two kernels record on different iteration
    grids, so the rows are not a cross-lane comparable. It is returned so that
    the case which mirrors upstream's snapshot selection
    (``permanent_magnet_MUSE.py:184-185``) can apply it, and so that a test can
    compare the native rows with the official capture's ``history:R2``.
    """

    observation: LaneObservation
    #: Upstream's ``R2`` per recorded row, trimmed to the filled prefix.
    objective_history: np.ndarray
    #: ``(ndipoles, 3, rows)`` physical moments, the C++ ``m_history`` layout.
    moment_history: np.ndarray


def configuration_int(bundle: InputBundle, name: str) -> int:
    """Read one integer from a parity input configuration."""
    value = bundle.configuration[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"configuration {name} must be an integer")
    return value


def configuration_float(bundle: InputBundle, name: str) -> float:
    """Read one floating-point value from a parity input configuration."""
    value = bundle.configuration[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"configuration {name} must be numeric")
    return float(value)


def frozen_grid_arrays(grid) -> dict[str, np.ndarray]:
    """Copy the complete fixed optimization state from one native PM grid."""
    moments = np.array(grid.m, dtype=np.float64, copy=True).reshape((grid.ndipoles, 3))
    proxy_source = grid.m_proxy if hasattr(grid, "m_proxy") else grid.m
    return {
        "response_matrix": np.array(grid.A_obj, dtype=np.float64, copy=True),
        "target": np.array(grid.b_obj, dtype=np.float64, copy=True),
        "normal_norms": np.array(
            np.linalg.norm(grid.plasma_boundary.normal(), axis=-1).reshape(-1),
            dtype=np.float64,
            copy=True,
        ),
        "atb": np.array(grid.ATb, dtype=np.float64, copy=True).reshape(
            (grid.ndipoles, 3)
        ),
        "ata_scale": np.array(grid.ATA_scale, dtype=np.float64, copy=True),
        "initial_moments": np.array(
            grid.m0,
            dtype=np.float64,
            copy=True,
        ).reshape((grid.ndipoles, 3)),
        "moments": moments,
        "proxy_moments": np.array(
            proxy_source,
            dtype=np.float64,
            copy=True,
        ).reshape((grid.ndipoles, 3)),
        "moment_maxima": np.array(
            grid.m_maxima,
            dtype=np.float64,
            copy=True,
        ).reshape((grid.ndipoles,)),
        "dipole_grid_xyz": np.array(
            grid.dipole_grid_xyz,
            dtype=np.float64,
            copy=True,
        ),
        "polarization_vectors": np.array(
            grid.pol_vectors,
            dtype=np.float64,
            copy=True,
        ),
    }


def _array_digest(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def effective_grid_fingerprint(
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> str:
    """Fingerprint the complete frozen grid without derived recomputation."""
    return effective_construction_fingerprint(
        bundle,
        {
            "arrays": {
                name: {
                    "dtype": str(value.dtype),
                    "shape": list(value.shape),
                    "sha256": _array_digest(value),
                }
                for name, value in sorted(arrays.items())
            },
            **bundle.configuration,
        },
    )


def _values(
    arrays: dict[str, np.ndarray],
    final_moments: np.ndarray,
    final_residual: np.ndarray,
) -> dict[str, np.ndarray]:
    response = arrays["response_matrix"]
    target = arrays["target"]
    initial_moments = arrays["initial_moments"]
    initial_residual = response @ initial_moments.reshape(-1) - target
    nonzero_mask = np.linalg.norm(final_moments, axis=1) != 0.0
    return {
        "construction:response_matrix": response,
        "construction:target": target,
        "construction:moment_maxima": arrays["moment_maxima"],
        "construction:dipole_grid_xyz": arrays["dipole_grid_xyz"],
        "construction:polarization_vectors": arrays["polarization_vectors"],
        "initial:moments": initial_moments,
        "initial:residual": initial_residual,
        "initial:objective_sum_squares": np.asarray(
            np.vdot(initial_residual, initial_residual),
            dtype=np.float64,
        ),
        "final:moments": final_moments,
        "final:residual": final_residual,
        "final:objective_sum_squares": np.asarray(
            np.vdot(final_residual, final_residual),
            dtype=np.float64,
        ),
        "final:nonzero_mask": nonzero_mask,
    }


def _observation(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
    values: dict[str, np.ndarray],
    workflow_stages: tuple[str, ...],
    *,
    driver: str,
    platform: str,
    precision: str,
    recorded_nonzero_counts: np.ndarray,
) -> LaneObservation:
    # Upstream stops GPMO by the stalled nonzero count, by the magnet cap, by a
    # full grid, or by exhausting K (permanent_magnet_optimization.cpp:942-985);
    # none of the four is convergence, and the configured K is not a measured
    # nit/nfev, so the counters are null. "final objective < initial objective"
    # is NOT part of success: upstream PM4Stell itself rises from 0.16177 to
    # 0.83456 (official capture history:R2), so the branch gate would fail the
    # official run. Both objectives stay published as a compared diagnostic.
    # A run that placed NO magnet is degenerate whatever the stop reason was:
    # it returned the moments it was given, so it is never a success.
    # Upstream's fourth exit is a branch of the native kernel only: the JAX
    # kernel (``simsopt_jax/core/pm_optimization.py``) has no "nonzero count
    # unchanged" test and runs to ``K`` where the native solver breaks out, so
    # each lane is labelled from the exits its own provider can take. The two
    # kernels also record on different grids (6 rows against 20 at bounded
    # MUSE), which is why a shared derivation would make upstream's exit
    # unreachable on the lane that has it and reachable on the lane that does
    # not.
    nonzero_count = int(np.count_nonzero(values["final:nonzero_mask"]))
    grid_size = configuration_int(bundle, "ndipoles")
    magnet_cap = configuration_int(bundle, "max_magnets")
    label = fixed_work_label(
        raw_status=gpmo_stop_reason(
            nonzero_count=nonzero_count,
            grid_size=grid_size,
            magnet_cap=magnet_cap,
            recorded_nonzero_counts=(
                recorded_nonzero_counts if lane == "native-cpu" else None
            ),
        ),
        outputs_usable=gpmo_backtracking_outputs_usable(
            moments=values["final:moments"],
            nonzero_count=nonzero_count,
            grid_size=grid_size,
            magnet_cap=magnet_cap,
        ),
    )
    return LaneObservation(
        lane=lane,
        backend_mode=(
            "native_cpu" if lane == "native-cpu" else os.environ["SIMSOPT_BACKEND_MODE"]
        ),
        platform=platform,
        precision=precision,
        scale=bundle.scale,
        input_fingerprint=bundle.input_fingerprint,
        configuration_fingerprint=bundle.configuration_fingerprint,
        effective_construction_fingerprint=effective_grid_fingerprint(
            bundle,
            arrays,
        ),
        driver=driver,
        normalized_status=label.normalized_status,
        raw_status=label.raw_status,
        success=label.success,
        nit=None,
        nfev=None,
        njev=None,
        completed_workflow_stages=workflow_stages,
        provenance=None,
        values=values,
    )


def _execute_native(
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
    workflow_stages: tuple[str, ...],
) -> ArbVecLaneResult:
    import simsoptpp

    maxima = arrays["moment_maxima"]
    maxima_vector = np.repeat(maxima, 3)
    scaled_response = arrays["response_matrix"] * maxima_vector[None, :]
    (
        objective_history,
        _,
        normalized_moment_history,
        recorded_nonzero_counts,
        normalized_moments,
    ) = simsoptpp.GPMO_ArbVec_backtracking(
        np.ascontiguousarray(scaled_response.T),
        np.ascontiguousarray(arrays["target"]),
        np.sqrt(configuration_float(bundle, "regularization_l2")) * maxima_vector,
        np.ascontiguousarray(arrays["normal_norms"]),
        np.ascontiguousarray(arrays["polarization_vectors"]),
        K=configuration_int(bundle, "iterations"),
        # Official kwargs: initialize_default_kwargs('GPMO') sets verbose=True.
        verbose=True,
        nhistory=configuration_int(bundle, "history_count"),
        backtracking=configuration_int(bundle, "backtracking"),
        dipole_grid_xyz=np.ascontiguousarray(arrays["dipole_grid_xyz"]),
        Nadjacent=configuration_int(bundle, "adjacent_count"),
        thresh_angle=configuration_float(bundle, "threshold_angle"),
        max_nMagnets=configuration_int(bundle, "max_magnets"),
        x_init=np.zeros(
            (configuration_int(bundle, "ndipoles"), 3),
            dtype=np.float64,
        ),
    )
    final_moments = np.asarray(normalized_moments, dtype=np.float64) * maxima[:, None]
    final_residual = (
        arrays["response_matrix"] @ final_moments.reshape(-1) - arrays["target"]
    )
    # ``GPMO`` rescales every recorded row by ``mmax`` before returning it
    # (solve/permanent_magnet_optimization.py:471-472); the lane calls the
    # extension directly, so it applies the same rescale.
    recorded = recorded_history_length(np.asarray(objective_history))
    moment_history = (
        np.asarray(normalized_moment_history, dtype=np.float64)[:, :, :recorded]
        * maxima[:, None, None]
    )
    return ArbVecLaneResult(
        observation=_observation(
            "native-cpu",
            bundle,
            arrays,
            _values(arrays, final_moments, final_residual),
            workflow_stages,
            driver="simsoptpp_gpmo_arbvec_backtracking",
            platform="cpu",
            precision="fp64",
            recorded_nonzero_counts=np.asarray(
                recorded_nonzero_counts, dtype=np.int64
            ).reshape(-1)[:recorded],
        ),
        objective_history=np.asarray(objective_history, dtype=np.float64).reshape(-1)[
            :recorded
        ],
        moment_history=np.ascontiguousarray(moment_history),
    )


def _execute_jax(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
    workflow_stages: tuple[str, ...],
) -> ArbVecLaneResult:
    from simsopt_jax.backend.runtime import get_runtime_jax_device
    from simsopt_jax.geo.permanent_magnet_grid import PermanentMagnetGridJAX
    from simsopt_jax.solve.permanent_magnet import GPMO_ArbVec_backtracking_jax

    import jax

    device = get_runtime_jax_device()

    def put(name: str):
        return jax.device_put(arrays[name], device)

    grid = PermanentMagnetGridJAX(
        A_obj=put("response_matrix"),
        b_obj=put("target"),
        ATb=put("atb"),
        ATA_scale=put("ata_scale"),
        m0=put("initial_moments"),
        m=put("moments"),
        m_proxy=put("proxy_moments"),
        m_maxima=put("moment_maxima"),
        dipole_grid_xyz=put("dipole_grid_xyz"),
        coordinate_flag=str(bundle.configuration["coordinate_flag"]),
        R0=configuration_float(bundle, "R0"),
        nfp=configuration_int(bundle, "nfp"),
        stellsym=bool(bundle.configuration["stellsym"]),
        nphi=configuration_int(bundle, "nphi"),
        ntheta=configuration_int(bundle, "ntheta"),
        ndipoles=configuration_int(bundle, "ndipoles"),
        pol_vectors=put("polarization_vectors"),
    )
    device_result = GPMO_ArbVec_backtracking_jax(
        grid,
        K=configuration_int(bundle, "iterations"),
        reg_l2=configuration_float(bundle, "regularization_l2"),
        Nadjacent=configuration_int(bundle, "adjacent_count"),
        backtracking=configuration_int(bundle, "backtracking"),
        thresh_angle=configuration_float(bundle, "threshold_angle"),
        max_nMagnets=configuration_int(bundle, "max_magnets"),
        # Mirror of upstream's print period int(K / nhistory): the JAX kernel
        # records every ``record_every``-th iteration plus the last one.
        record_every=gpmo_history_period(
            iterations=configuration_int(bundle, "iterations"),
            history_count=configuration_int(bundle, "history_count"),
        ),
    )
    host_result, construction_arrays = jax.device_get(
        (
            device_result,
            {
                "response_matrix": grid.A_obj,
                "target": grid.b_obj,
                "atb": grid.ATb,
                "ata_scale": grid.ATA_scale,
                "initial_moments": grid.m0,
                "moments": grid.m,
                "proxy_moments": grid.m_proxy,
                "moment_maxima": grid.m_maxima,
                "dipole_grid_xyz": grid.dipole_grid_xyz,
                "polarization_vectors": grid.pol_vectors,
            },
        )
    )
    host_arrays = {
        **arrays,
        **{
            name: np.asarray(value, dtype=arrays[name].dtype)
            for name, value in construction_arrays.items()
        },
    }
    values = _values(
        host_arrays,
        np.asarray(host_result.m, dtype=np.float64),
        np.asarray(host_result.residual, dtype=np.float64),
    )
    platform = "cpu" if device is None else device.platform
    # ``residual_history`` is ``sum(r * r)`` per recorded row
    # (core/pm_optimization.py:2047); upstream records ``R2 = 0.5 * sum(r * r)``
    # (``print_GPMO``), so the halving makes the two histories the same
    # quantity. ``m_history`` arrives as ``(rows, ndipoles, 3)`` and is
    # transposed into the C++ ``(ndipoles, 3, rows)`` layout.
    jax_objective_history = 0.5 * np.asarray(
        host_result.residual_history, dtype=np.float64
    ).reshape(-1)
    jax_recorded = recorded_history_length(jax_objective_history)
    return ArbVecLaneResult(
        observation=_observation(
            lane,
            bundle,
            host_arrays,
            values,
            workflow_stages,
            driver="simsopt_jax_gpmo_arbvec_backtracking",
            platform="gpu" if platform in {"cuda", "gpu"} else platform,
            precision="fp64" if bool(jax.config.read("jax_enable_x64")) else "fp32",
            # Trimmed to the rows this kernel recorded, exactly as the native
            # lane trims its own; the label derivation then sees each
            # provider's filled prefix rather than one lane's padding.
            recorded_nonzero_counts=np.asarray(
                host_result.num_nonzeros_history, dtype=np.int64
            ).reshape(-1)[:jax_recorded],
        ),
        objective_history=jax_objective_history,
        moment_history=np.ascontiguousarray(
            np.transpose(np.asarray(host_result.m_history, dtype=np.float64), (1, 2, 0))
        ),
    )


def execute_arbvec_case(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
    workflow_stages: tuple[str, ...],
) -> ArbVecLaneResult:
    """Run one frozen arbitrary-vector GPMO case in its requested lane."""
    if lane == "native-cpu":
        return _execute_native(bundle, arrays, workflow_stages)
    return _execute_jax(lane, bundle, arrays, workflow_stages)
