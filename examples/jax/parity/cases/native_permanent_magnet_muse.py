"""Exact matched workflow for ``2_Intermediate/permanent_magnet_MUSE.py``."""

from __future__ import annotations

import hashlib
import io
from collections.abc import Mapping
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Final

import numpy as np
from examples.jax.parity.arbiter import LaneObservation
from examples.jax.parity.cases._fixed_work_status import fixed_work_label
from examples.jax.parity.cases._permanent_magnet_arbvec import (
    ArbVecLaneResult,
    execute_arbvec_case,
    frozen_grid_arrays,
)
from examples.jax.parity.input_bundle import InputBundle, create_input_bundle
from examples.jax.parity.official_reference import (
    CANONICAL_VARIANT,
    JsonValue,
    load_official_reference,
)
from examples.jax.parity.runtime import ParityLane
from simsopt.geo import PermanentMagnetGrid
from simsopt.util import FocusData, discretize_polarizations, polarization_axes
from simsopt_jax.examples import ExecutionScale
from simsopt_jax_adapters.examples.gpmo_rules import (
    gpmo_backtracking_outputs_usable,
    select_minimum_objective_snapshot,
)
from simsopt_jax_adapters.examples.muse import (
    MUSE_POST_WORKFLOW_STAGES,
    MusePostDiagnostics,
    build_muse_post_geometry,
    muse_post_workflow_policy,
    run_jax_muse_post_workflow,
    run_native_muse_post_workflow,
)

TEST_DATA = Path(__file__).resolve().parents[4] / "tests" / "test_files"
SURFACE_INPUT = TEST_DATA / "input.muse"
FAMUS_INPUT = TEST_DATA / "zot80.focus"
CASE_ID = "native-permanent-magnet-muse"
#: Which captured run of the official script each scale mirrors: the canonical
#: record is upstream's shipped run, the ``ci`` record upstream's own
#: ``in_github_actions`` run of the SAME script at ``9e027eac3``.
OFFICIAL_VARIANT_BY_SCALE: Final[Mapping[str, str]] = MappingProxyType(
    {"native_default": CANONICAL_VARIANT, "bounded": "ci"}
)

GPMO_WORKFLOW_STAGES = (
    "construct_muse_boundary_and_tf_coil_field",
    "construct_downsampled_famus_grid_and_face_polarizations",
    "evaluate_initial_normal_field_residual",
    "run_arbitrary_vector_backtracking_gpmo",
    "evaluate_final_moments_and_normal_field_residual",
)
#: Every lane runs the official post-GPMO half, so the case's stage contract is
#: the GPMO stages followed by the two post stages. The arbiter requires one
#: shared stage tuple (``arbiter.py:238``), and a native lane that skipped the
#: post stages would leave the official ``f_B`` on the plotting surface, the
#: total volume and the coil-only trace unmirrored.
WORKFLOW_STAGES = (*GPMO_WORKFLOW_STAGES, *MUSE_POST_WORKFLOW_STAGES)


def _official_configuration(scale: ExecutionScale) -> Mapping[str, JsonValue]:
    """Upstream's own configuration of this script at ``scale``."""
    record = load_official_reference(
        CASE_ID, variant=OFFICIAL_VARIANT_BY_SCALE[scale]
    ).structure("configuration")
    if not isinstance(record, Mapping):
        raise TypeError("the official configuration record is not a JSON object")
    return record


#: Read once per process; the loader imports nothing but the standard library
#: and numpy.
OFFICIAL_CONFIGURATION_BY_SCALE: Final[Mapping[str, Mapping[str, JsonValue]]] = (
    MappingProxyType(
        {scale: _official_configuration(scale) for scale in OFFICIAL_VARIANT_BY_SCALE}
    )
)


def _official_int(configuration: Mapping[str, JsonValue], name: str) -> int:
    value = configuration[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"official configuration {name} must be an integer")
    return value


def _official_float(configuration: Mapping[str, JsonValue], name: str) -> float:
    value = configuration[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"official configuration {name} must be numeric")
    return float(value)


def _scale_configuration(scale: ExecutionScale) -> dict[str, object]:
    """Upstream's own GPMO configuration, from the tracked official record.

    Both scales are upstream's: ``native_default`` is the shipped run and
    ``bounded`` is upstream's ``in_github_actions`` configuration
    (``permanent_magnet_MUSE.py:44-56, 152-162``). Neither is pasted here --
    the numbers live in ONE tracked place, ``examples/jax/parity/official_reference``,
    and the names below are this case's spelling of upstream's own keywords.
    ``ndipoles`` is deliberately NOT read from the record: it is what the
    branch's own grid build produces from the FAMUS inventory, and the tests
    compare it with upstream's rather than adopt it.
    """
    official = OFFICIAL_CONFIGURATION_BY_SCALE[scale]
    return {
        "nphi": _official_int(official, "nphi"),
        "ntheta": _official_int(official, "ntheta"),
        "downsample": _official_int(official, "downsample"),
        "iterations": _official_int(official, "K"),
        "backtracking": _official_int(official, "backtracking"),
        "max_magnets": _official_int(official, "max_nMagnets"),
        "history_count": _official_int(official, "nhistory"),
        "adjacent_count": _official_int(official, "Nadjacent"),
        "threshold_angle": _official_float(official, "thresh_angle"),
        # Upstream passes ``reg_l2=0`` (its ``gpmo_keyword_names`` carries the
        # keyword; the record stores no value for it).
        "regularization_l2": 0.0,
        "radial_extent": _official_float(official, "dr"),
        "coordinate_flag": "cartesian",
        "surface_input_sha256": hashlib.sha256(SURFACE_INPUT.read_bytes()).hexdigest(),
        "famus_input_sha256": hashlib.sha256(FAMUS_INPUT.read_bytes()).hexdigest(),
    }


def _build_cpu_grid(configuration: dict[str, object]):
    nphi = configuration["nphi"]
    ntheta = configuration["ntheta"]
    downsample = configuration["downsample"]
    radial_extent = configuration["radial_extent"]
    assert isinstance(nphi, int)
    assert isinstance(ntheta, int)
    assert isinstance(downsample, int)
    assert isinstance(radial_extent, float)
    post_geometry = build_muse_post_geometry(
        SURFACE_INPUT,
        TEST_DATA,
        nphi=nphi,
        ntheta=ntheta,
    )
    surface = post_geometry.optimization_surface
    field = post_geometry.coil_field
    field.set_points(surface.gamma().reshape((-1, 3)))
    normal_field = np.sum(
        field.B().reshape((nphi, ntheta, 3)) * surface.unitnormal(),
        axis=2,
    )
    magnet_data = FocusData(FAMUS_INPUT, downsample=downsample)
    axes, polarization_types = polarization_axes(["face"])
    positive_count = len(polarization_types) // 2
    orientation = np.arctan2(magnet_data.oy, magnet_data.ox)
    discretize_polarizations(
        magnet_data,
        orientation,
        axes[:positive_count],
        polarization_types[:positive_count],
    )
    polarizations = np.stack(
        (magnet_data.pol_x, magnet_data.pol_y, magnet_data.pol_z),
        axis=-1,
    )
    with redirect_stdout(io.StringIO()):
        return PermanentMagnetGrid.geo_setup_from_famus(
            surface,
            normal_field,
            FAMUS_INPUT,
            pol_vectors=polarizations,
            downsample=downsample,
            dr=radial_extent,
        )


def create_input(root: Path, scale: ExecutionScale) -> InputBundle:
    """Freeze the MUSE fixed-state grid consumed by every solver lane."""
    configuration = _scale_configuration(scale)
    grid = _build_cpu_grid(configuration)
    return create_input_bundle(
        root,
        case_id="native-permanent-magnet-muse",
        random_seed=0,
        arrays=frozen_grid_arrays(grid),
        configuration={
            **configuration,
            "R0": float(grid.R0),
            "nfp": int(grid.plasma_boundary.nfp),
            "stellsym": bool(grid.plasma_boundary.stellsym),
            "ndipoles": int(grid.ndipoles),
        },
        scale=scale,
    )


def _post_diagnostics(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
    final_moments: np.ndarray,
) -> MusePostDiagnostics:
    """Run the official post-GPMO half in the lane's own implementation."""
    nphi = bundle.configuration["nphi"]
    ntheta = bundle.configuration["ntheta"]
    coordinate_flag = bundle.configuration["coordinate_flag"]
    assert isinstance(nphi, int)
    assert isinstance(ntheta, int)
    assert isinstance(coordinate_flag, str)
    workflow = (
        run_native_muse_post_workflow
        if lane == "native-cpu"
        else run_jax_muse_post_workflow
    )
    return workflow(
        build_muse_post_geometry(
            SURFACE_INPUT,
            TEST_DATA,
            nphi=nphi,
            ntheta=ntheta,
        ),
        muse_post_workflow_policy(bundle.scale),
        final_moments,
        arrays["dipole_grid_xyz"],
        arrays["moment_maxima"],
        coordinate_flag=coordinate_flag,
    )


def observe(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
    result: ArbVecLaneResult,
) -> LaneObservation:
    """Derive this case's published observation from ONE GPMO solve.

    Split out of :func:`execute` so that a caller which also needs the recorded
    history -- the one-thread child of ``tests/parity_gpmo_native_child.py``,
    which compares that history with the official record and the endpoint with
    the same record -- can run the solve once and derive both from it, instead
    of solving twice and relying on an unstated determinism assumption.

    Official ``permanent_magnet_MUSE.py:184-185`` sets ``pm_opt.m`` to the
    recorded snapshot with the smallest ``R2`` *before* the printed volume, the
    dipole squared flux on the plotting surface and the total magnet volume, so
    the post metrics are computed from that snapshot and not from the GPMO
    endpoint. ``GPMO_ArbVec_backtracking`` is non-monotone (official PM4Stell
    rises from 0.16177 to 0.83456), so the two differ in general; at both
    official MUSE scales the recorded ``R2`` decreases monotonically and the
    selection lands on the endpoint, which the two published selection
    observables state.
    """
    observation = result.observation
    selection = select_minimum_objective_snapshot(
        result.objective_history,
        result.moment_history,
    )
    # The selected snapshot -- not the GPMO endpoint -- is what every published
    # ``final:selected_history_*`` and ``final:post_*`` observable below is
    # computed from, so it carries the same usability requirement: a non-finite
    # or degenerate row would otherwise leave the lane ``success`` while
    # feeding NaN into the post metrics. The endpoint's own check is already in
    # ``observation`` (``_permanent_magnet_arbvec._observation``); this adds the
    # second array the case publishes from, through the same shared rule.
    # The magnet cap is the endpoint's bound and ``_observation`` already
    # applied it; every recorded row was produced under the same cap, because
    # the solver breaks at the iteration that reaches it. What the selected row
    # adds is finiteness and non-degeneracy, so the grid its own array declares
    # is the bound passed here.
    selected_dipoles = selection.moments.shape[0]
    selected_usable = gpmo_backtracking_outputs_usable(
        moments=selection.moments,
        nonzero_count=int(np.count_nonzero(np.linalg.norm(selection.moments, axis=1))),
        grid_size=selected_dipoles,
        magnet_cap=selected_dipoles,
    )
    label = fixed_work_label(
        raw_status=observation.raw_status,
        outputs_usable=bool(observation.success and selected_usable),
    )
    selected_residual = (
        arrays["response_matrix"] @ selection.moments.reshape(-1) - arrays["target"]
    )
    post_diagnostics = _post_diagnostics(
        lane,
        bundle,
        arrays,
        selection.moments,
    )
    # The trace outcome is a reported diagnostic: official MUSE asserts nothing
    # about it, so it never demotes the GPMO lane label (completion policy,
    # "branch-added gates stay diagnostics"). ``final:post_coil_only_trace_status``
    # carries the fact and is compared across lanes.
    return replace(
        observation,
        normalized_status=label.normalized_status,
        raw_status=label.raw_status,
        success=label.success,
        # Re-derived from the published keys and the new status: the arbiter
        # ties ``optimizer_outcome`` to the normalized status, and this case
        # adds observables to the ones the GPMO helper labelled.
        applicability={},
        completed_workflow_stages=WORKFLOW_STAGES,
        values={
            **observation.values,
            # The selection evidence, recomputed host-side in both lanes from
            # the selected moments so the comparison is of the same arithmetic.
            # The raw row index is not published: the two kernels record on
            # different iteration grids (see ``_permanent_magnet_arbvec``), so
            # it is not a cross-lane comparable; the example publishes it.
            "final:selected_history_objective_sum_squares": np.asarray(
                np.vdot(selected_residual, selected_residual),
                dtype=np.float64,
            ),
            "final:selected_history_equals_endpoint": np.asarray(
                np.array_equal(
                    selection.moments,
                    np.asarray(observation.values["final:moments"], dtype=np.float64),
                )
            ),
            "final:post_dipole_squared_flux": np.asarray(
                post_diagnostics.dipole_squared_flux,
                dtype=np.float64,
            ),
            "final:post_total_magnet_volume": np.asarray(
                post_diagnostics.total_magnet_volume,
                dtype=np.float64,
            ),
            "final:post_coil_only_trace_status": np.asarray(
                post_diagnostics.trace_statuses,
                dtype=np.int64,
            ),
            "final:post_coil_only_trace_final_times": np.asarray(
                post_diagnostics.trace_final_times,
                dtype=np.float64,
            ),
            "final:post_coil_only_trace_hit_counts": np.asarray(
                post_diagnostics.trace_hit_counts,
                dtype=np.int64,
            ),
        },
    )


def execute(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> LaneObservation:
    """Execute the exact MUSE permanent-magnet workflow in one solver lane."""
    return observe(
        lane,
        bundle,
        arrays,
        execute_arbvec_case(lane, bundle, arrays, GPMO_WORKFLOW_STAGES),
    )
