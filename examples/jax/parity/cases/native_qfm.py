"""Exact matched workflow for ``1_Simple/qfm.py``."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final, Mapping

import numpy as np
from examples.jax.parity.arbiter import LaneObservation
from examples.jax.parity.input_bundle import (
    InputBundle,
    create_input_bundle,
    effective_construction_fingerprint,
)
from examples.jax.parity.official_reference import (
    OfficialReference,
    load_official_reference,
)
from examples.jax.parity.runtime import ParityLane
from simsopt_contracts.optimization_endpoint import (
    StoppingReason,
    normalized_terminal_status,
    scipy_minimize_stopping_reason,
)
from simsopt.configs.zoo import get_data
from simsopt.field import BiotSavart
from simsopt.geo import Area, QfmSurface, SurfaceRZFourier, ToroidalFlux, Volume
from simsopt_jax.backend.dtypes import explicit_device_array
from simsopt_jax.backend.runtime import get_runtime_jax_device
from simsopt_jax.examples import ExecutionScale
from simsopt_jax.examples.qfm_host_scipy import (
    QFM_EXACT_METHOD,
    QFM_HOST_SCIPY_DRIVER,
    QFM_SURFACE_RESOLUTION,
    build_qfm_host_kernels,
    solve_qfm_host_scipy_sequence,
)
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX

import jax

WORKFLOW_STAGES = (
    "construct_fitted_ncsx_qfm_surface",
    "evaluate_initial_qfm_state",
    "capture_volume_target",
    "solve_volume_penalty",
    "solve_volume_exact_constraint",
    "capture_toroidal_flux_target",
    "solve_toroidal_flux_penalty",
    "solve_toroidal_flux_exact_constraint",
    "check_volume_after_toroidal_flux",
    "capture_area_target",
    "solve_area_penalty",
    "solve_area_exact_constraint",
    "check_volume_after_area",
    "publish_final_qfm_state",
)

_LABELS = ("volume", "toroidal_flux", "area")

#: The official shipped run of ``examples/1_Simple/qfm.py`` at upstream
#: ``9e027eac3``, from the one tracked home for official numbers.
OFFICIAL_REFERENCE: Final[OfficialReference] = load_official_reference("native-qfm")
#: Exact-stage label residuals the official run leaves behind, per label
#: (``<label>:exact:label_residual_abs``). Upstream states no feasibility gate;
#: these are what its own three SLSQP calls actually leave.
OFFICIAL_EXACT_LABEL_RESIDUALS: Final[Mapping[str, float]] = MappingProxyType(
    {
        label: float(OFFICIAL_REFERENCE.scalar(f"{label}:exact:label_residual_abs"))
        for label in _LABELS
    }
)
#: Gross-failure ceiling on an exact stage's ``|label - target|`` -- for the
#: ``native_default`` scale and ONLY there. Upstream's ``1_Simple/qfm.py`` has
#: no reduced CI configuration (no ``in_github_actions``; ``mpol = ntor = 5``
#: and 25 quadrature nodes are unconditional, and the official-reference
#: fixture has no ``ci`` variant for this case), so the branch's ``bounded``
#: scale has no official run and therefore no official residual: a number
#: derived from the shipped run cannot gate a problem the shipped run never
#: solved. Ten times the largest residual the official run leaves is not
#: upstream's feasibility gate -- upstream states none, and the branch's
#: demoted 1e-8 gate rejects the official run itself -- it is the line an
#: order of magnitude above where the official solve lands, so a lane that
#: reaches it has failed grossly.
NATIVE_DEFAULT_LABEL_RESIDUAL_GROSS_FAILURE_CEILING: Final[float] = 10.0 * max(
    OFFICIAL_EXACT_LABEL_RESIDUALS.values()
)
#: The two SciPy methods ``1_Simple/qfm.py`` alternates, once per stage. The
#: exact stage's method is the solver module's own constant; the penalty stage
#: runs SciPy L-BFGS-B (``QFM_PENALTY_DRIVER = Driver.SCIPY_LBFGSB``, dispatched
#: at ``src/simsopt_jax/solve/dispatch.py:362``). Which status vocabulary each
#: method speaks is the contract's
#: (``SCIPY_MINIMIZE_STATUS_CONVENTION_BY_METHOD``), not this module's.
QFM_PENALTY_METHOD: Final[str] = "L-BFGS-B"


@dataclass(frozen=True)
class QfmProviderCall:
    """One of the six SciPy calls of the official QFM sequence.

    The fields are exactly what a provider reports about itself, so one record
    is built the same way from a lane's own results and from the official
    capture's ``provider_calls``.
    """

    method: str
    provider_success: bool
    provider_status: int
    iterations: int
    max_iterations: int
    endpoint_finite: bool


def qfm_stage_stopping_reason(call: QfmProviderCall) -> StoppingReason:
    """Classify one QFM SciPy call into the contract's stopping vocabulary.

    Both emitters are owned by the contract: ``scipy-lbfgsb`` knows that
    L-BFGS-B's status 1 merges the iteration and the evaluation budget, and
    ``scipy-slsqp`` knows that SLSQP's iteration limit is mode 9 while its
    status 1 is the transient "function evaluation required". The routing from
    a ``scipy.optimize.minimize`` method to its emitter, and what a stopping
    reason does and does not read, are the contract's too
    (:func:`scipy_minimize_stopping_reason`) -- so the shipped mirror
    ``examples/jax/1_Simple/qfm.py``, which cannot import this module, reaches
    the SAME classification through the same owner.
    """
    return scipy_minimize_stopping_reason(
        method=call.method,
        provider_success=call.provider_success,
        provider_status=call.provider_status,
        iterations=call.iterations,
        max_iterations=call.max_iterations,
        endpoint_finite=call.endpoint_finite,
    )


def qfm_stopping_reasons(
    calls: Sequence[QfmProviderCall],
) -> tuple[StoppingReason, ...]:
    """Classify the six calls of one QFM sequence, in call order."""
    return tuple(qfm_stage_stopping_reason(call) for call in calls)


def official_qfm_stopping_reasons() -> tuple[StoppingReason, ...]:
    """The official run's own six stopping reasons, from the tracked fixture.

    Read as a classification and never as SciPy's message text: the spelling is
    a SciPy version detail (``pyproject.toml`` allows ``scipy>=1.13``, whose
    L-BFGS-B spells the same task with underscores), while the classification
    is the fact upstream established.
    """
    return qfm_stopping_reasons(
        tuple(
            QfmProviderCall(
                method=str(call.method),
                provider_success=bool(call.result["success"]),
                provider_status=int(call.result["status"]),
                iterations=int(call.result["nit"]),
                max_iterations=int(call.options["maxiter"]),
                endpoint_finite=bool(np.isfinite(float(call.result["fun"]))),
            )
            for call in OFFICIAL_REFERENCE.provider_calls
        )
    )


def _ragged_snapshot(values: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    flattened = [np.asarray(value, dtype=np.float64).reshape(-1) for value in values]
    offsets = np.cumsum([0, *(value.size for value in flattened)], dtype=np.int64)
    return np.concatenate(flattened), offsets


def create_input(root: Path, scale: ExecutionScale) -> InputBundle:
    """Snapshot NCSX at bounded or official-example surface resolution."""

    _, _, magnetic_axis, nfp, biotsavart = get_data("ncsx")
    resolution = QFM_SURFACE_RESOLUTION[scale]
    order = resolution.order
    quadrature_size = resolution.quadrature_size
    quadrature_phi = np.linspace(0.0, 1.0 / nfp, quadrature_size, endpoint=False)
    quadrature_theta = np.linspace(0.0, 1.0, quadrature_size, endpoint=False)
    surface = SurfaceRZFourier(
        mpol=order,
        ntor=order,
        stellsym=True,
        nfp=nfp,
        quadpoints_phi=quadrature_phi,
        quadpoints_theta=quadrature_theta,
    )
    surface.fit_to_curve(magnetic_axis, 0.2, flip_theta=True)
    coil_curve_dofs, coil_curve_dof_offsets = _ragged_snapshot(
        [np.asarray(coil.curve.local_full_x) for coil in biotsavart.coils]
    )
    coil_currents, coil_current_offsets = _ragged_snapshot(
        [np.asarray(coil.current.local_full_x) for coil in biotsavart.coils]
    )
    return create_input_bundle(
        root,
        case_id="native-qfm",
        random_seed=0,
        arrays={
            "initial_parameters": np.asarray(surface.x, dtype=np.float64),
            "quadrature_phi": quadrature_phi,
            "quadrature_theta": quadrature_theta,
            "coil_curve_dofs": coil_curve_dofs,
            "coil_curve_dof_offsets": coil_curve_dof_offsets,
            "coil_currents": coil_currents,
            "coil_current_offsets": coil_current_offsets,
        },
        configuration={
            "nfp": nfp,
            "mpol": order,
            "ntor": order,
            "stellsym": True,
            "constraint_weight": 1.0,
            "tolerance": 1.0e-12,
            "max_steps": 80 if scale == "bounded" else 1000,
        },
        scale=scale,
    )


def _configuration_float(bundle: InputBundle, name: str) -> float:
    value = bundle.configuration[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"configuration {name} must be numeric")
    return float(value)


def _configuration_int(bundle: InputBundle, name: str) -> int:
    value = bundle.configuration[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"configuration {name} must be an integer")
    return value


def _surface_from_state(bundle: InputBundle, arrays: dict[str, np.ndarray]):

    surface = SurfaceRZFourier(
        mpol=_configuration_int(bundle, "mpol"),
        ntor=_configuration_int(bundle, "ntor"),
        stellsym=bool(bundle.configuration["stellsym"]),
        nfp=_configuration_int(bundle, "nfp"),
        quadpoints_phi=arrays["quadrature_phi"],
        quadpoints_theta=arrays["quadrature_theta"],
    )
    surface.x = arrays["initial_parameters"]
    return surface


def _problem_components(bundle: InputBundle, arrays: dict[str, np.ndarray]):

    _, _, _, _, biotsavart = get_data("ncsx")
    surface = _surface_from_state(bundle, arrays)
    coil_curve_dofs, coil_curve_dof_offsets = _ragged_snapshot(
        [np.asarray(coil.curve.local_full_x) for coil in biotsavart.coils]
    )
    coil_currents, coil_current_offsets = _ragged_snapshot(
        [np.asarray(coil.current.local_full_x) for coil in biotsavart.coils]
    )
    snapshots = (
        ("coil_curve_dofs", coil_curve_dofs),
        ("coil_curve_dof_offsets", coil_curve_dof_offsets),
        ("coil_currents", coil_currents),
        ("coil_current_offsets", coil_current_offsets),
    )
    for name, actual in snapshots:
        if not np.array_equal(arrays[name], actual):
            raise ValueError(f"effective construction did not consume {name}")
    fingerprint = effective_construction_fingerprint(
        bundle,
        {
            "surface_dofs": np.asarray(surface.x).tolist(),
            "quadrature_phi": np.asarray(surface.quadpoints_phi).tolist(),
            "quadrature_theta": np.asarray(surface.quadpoints_theta).tolist(),
            "coil_curve_dofs": coil_curve_dofs.tolist(),
            "coil_curve_dof_offsets": coil_curve_dof_offsets.tolist(),
            "coil_currents": coil_currents.tolist(),
            "coil_current_offsets": coil_current_offsets.tolist(),
            **dict(bundle.configuration),
        },
    )
    return biotsavart, surface, fingerprint


def _native_state(qfm_surface, label, target: float) -> dict[str, np.ndarray]:
    parameters = np.array(qfm_surface.surface.x, dtype=np.float64, copy=True)
    qfm_value, qfm_gradient = qfm_surface.qfm_objective(parameters, derivatives=1)
    label_value = float(label.J())
    return {
        "parameters": parameters,
        "qfm_value": np.asarray(qfm_value, dtype=np.float64),
        "qfm_gradient": np.array(qfm_gradient, dtype=np.float64, copy=True),
        "label_value": np.asarray(label_value, dtype=np.float64),
        "label_gradient": np.array(
            label.dJ_by_dsurfacecoefficients(),
            dtype=np.float64,
            copy=True,
        ),
        "label_residual_abs": np.asarray(
            abs(label_value - target),
            dtype=np.float64,
        ),
    }


def qfm_scientific_predicate(
    values: dict[str, np.ndarray],
    scale: ExecutionScale,
) -> bool:
    """Whether the lane produced a finite, non-degenerate, feasible QFM solve.

    Three conditions hold at every scale, none of them invented here.

    * Every published quantity is finite, checked on the FULL arrays before
      anything selects from them.
    * The QFM value decreased, which upstream's own run does: ``initial``
      ``qfm_value`` down to ``area:exact:qfm_value``, four orders.
    * The run moved off its start. Without this a lane that never took a step
      satisfies the two conditions above.

    At ``native_default`` -- and ONLY there, because upstream's script has no
    reduced configuration and the branch's ``bounded`` scale therefore has no
    official run to be measured against -- each exact stage must also leave a
    label residual below
    :data:`NATIVE_DEFAULT_LABEL_RESIDUAL_GROSS_FAILURE_CEILING`. The branch's
    per-stage ``label_residual_abs <= 1e-8`` gate stays demoted to a published
    observable: the official run ends at 4.076e-07 and would fail it.
    """
    finite = all(bool(np.all(np.isfinite(value))) for value in values.values())
    decreased = float(values["area:exact:qfm_value"]) < float(
        values["initial:qfm_value"]
    )
    moved = bool(
        np.any(values["area:exact:parameters"] != values["initial:parameters"])
    )
    if scale != "native_default":
        return finite and decreased and moved
    feasible = all(
        float(values[f"{stage}:exact:label_residual_abs"])
        < NATIVE_DEFAULT_LABEL_RESIDUAL_GROSS_FAILURE_CEILING
        for stage in _LABELS
    )
    return finite and decreased and moved and feasible


def _prefixed(
    stage: str,
    phase: str,
    state: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    return {f"{stage}:{phase}:{name}": value for name, value in state.items()}


def _provider_call(
    method: str,
    *,
    success: bool,
    status: int,
    iterations: int,
    max_iterations: int,
    state: dict[str, np.ndarray],
    phase: str,
) -> QfmProviderCall:
    """One SciPy call, with the finiteness of the endpoint it actually left."""
    return QfmProviderCall(
        method=method,
        provider_success=success,
        provider_status=status,
        iterations=iterations,
        max_iterations=max_iterations,
        endpoint_finite=bool(
            np.all(np.isfinite(state[f"{phase}:parameters"]))
            and np.all(np.isfinite(state[f"{phase}:qfm_value"]))
        ),
    )


def _native(bundle: InputBundle, arrays: dict[str, np.ndarray]) -> LaneObservation:

    biotsavart, surface, fingerprint = _problem_components(bundle, arrays)
    toroidal_field = BiotSavart(biotsavart.coils)
    initial_volume = float(Volume(surface).J())
    max_steps = _configuration_int(bundle, "max_steps")
    values: dict[str, np.ndarray] = {}
    provider_calls: list[QfmProviderCall] = []
    total_nit = 0
    total_nfev = 0
    total_njev = 0
    raw_statuses: list[str] = []

    for stage_name in _LABELS:
        if stage_name == "volume":
            label = Volume(surface)
        elif stage_name == "toroidal_flux":
            label = ToroidalFlux(surface, toroidal_field)
        else:
            label = Area(surface)
        target = float(label.J())
        qfm_surface = QfmSurface(biotsavart, surface, label, target)
        initial = _native_state(qfm_surface, label, target)
        if stage_name == "volume":
            values.update(
                {
                    "initial:parameters": initial["parameters"],
                    "initial:qfm_value": initial["qfm_value"],
                    "initial:qfm_gradient": initial["qfm_gradient"],
                }
            )
        values[f"{stage_name}:target"] = np.asarray(target, dtype=np.float64)
        values.update(_prefixed(stage_name, "initial", initial))

        penalty = qfm_surface.minimize_qfm_penalty_constraints_LBFGS(
            tol=_configuration_float(bundle, "tolerance"),
            maxiter=_configuration_int(bundle, "max_steps"),
            constraint_weight=_configuration_float(bundle, "constraint_weight"),
        )
        values.update(
            _prefixed(
                stage_name,
                "penalty",
                _native_state(qfm_surface, label, target),
            )
        )
        exact = qfm_surface.minimize_qfm_exact_constraints_SLSQP(
            tol=_configuration_float(bundle, "tolerance"),
            maxiter=_configuration_int(bundle, "max_steps"),
        )
        exact_state = _native_state(qfm_surface, label, target)
        values.update(_prefixed(stage_name, "exact", exact_state))
        volume_residual = float(Volume(surface).J()) - initial_volume
        values[f"{stage_name}:volume_persistence_objective"] = np.asarray(
            0.5 * volume_residual * volume_residual,
            dtype=np.float64,
        )

        penalty_info = penalty["info"]
        exact_info = exact["info"]
        total_nit += int(penalty["iter"]) + int(exact["iter"])
        total_nfev += int(penalty_info.nfev) + int(exact_info.nfev)
        total_njev += int(penalty_info.njev) + int(exact_info.njev)
        raw_statuses.extend((str(penalty_info.message), str(exact_info.message)))
        # ``minimize_qfm_*`` hand back SciPy's own ``OptimizeResult`` under
        # ``info``, so this is the provider's real report, not a relabelling.
        provider_calls.extend(
            (
                _provider_call(
                    QFM_PENALTY_METHOD,
                    success=bool(penalty["success"]),
                    status=int(penalty_info.status),
                    iterations=int(penalty["iter"]),
                    max_iterations=max_steps,
                    state=values,
                    phase=f"{stage_name}:penalty",
                ),
                _provider_call(
                    QFM_EXACT_METHOD,
                    success=bool(exact["success"]),
                    status=int(exact_info.status),
                    iterations=int(exact["iter"]),
                    max_iterations=max_steps,
                    state=values,
                    phase=f"{stage_name}:exact",
                ),
            )
        )

    terminal = normalized_terminal_status(
        scientific_predicate=qfm_scientific_predicate(values, bundle.scale),
        stage_stopping_reasons=qfm_stopping_reasons(provider_calls),
    )

    return LaneObservation(
        lane="native-cpu",
        backend_mode="native_cpu",
        platform="cpu",
        precision="fp64",
        scale=bundle.scale,
        input_fingerprint=bundle.input_fingerprint,
        configuration_fingerprint=bundle.configuration_fingerprint,
        effective_construction_fingerprint=fingerprint,
        driver="simsopt_lbfgsb_then_slsqp_qfm_sequence",
        normalized_status=terminal.normalized_status,
        raw_status="; ".join(raw_statuses),
        success=terminal.success,
        nit=total_nit,
        nfev=total_nfev,
        njev=total_njev,
        completed_workflow_stages=WORKFLOW_STAGES,
        provenance=None,
        values=values,
    )


def _jax_state(state) -> dict[str, np.ndarray]:
    return {
        "parameters": np.asarray(state.parameters, dtype=np.float64),
        "qfm_value": np.asarray(state.qfm_value, dtype=np.float64),
        "qfm_gradient": np.asarray(state.qfm_gradient, dtype=np.float64),
        "label_value": np.asarray(state.label_value, dtype=np.float64),
        "label_gradient": np.asarray(state.label_gradient, dtype=np.float64),
        "label_residual_abs": np.asarray(
            state.label_residual_abs,
            dtype=np.float64,
        ),
    }


def _jax(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> LaneObservation:

    biotsavart, surface, fingerprint = _problem_components(bundle, arrays)
    device = get_runtime_jax_device()
    field = BiotSavartJAX(biotsavart.coils)
    coil_set_spec = field.coil_set_spec_from_dofs(
        explicit_device_array(field.x, dtype=np.float64, device=device)
    )
    kernels = build_qfm_host_kernels(
        initial_parameters=arrays["initial_parameters"],
        quadpoints_phi=arrays["quadrature_phi"],
        quadpoints_theta=arrays["quadrature_theta"],
        coil_set_spec=coil_set_spec,
        mpol=surface.mpol,
        ntor=surface.ntor,
        nfp=surface.nfp,
        stellsym=surface.stellsym,
    )
    max_steps = _configuration_int(bundle, "max_steps")
    result = solve_qfm_host_scipy_sequence(
        arrays["initial_parameters"],
        kernels=kernels,
        max_steps=max_steps,
        tolerance=_configuration_float(bundle, "tolerance"),
        constraint_weight=_configuration_float(bundle, "constraint_weight"),
    )
    values: dict[str, np.ndarray] = {
        "initial:parameters": np.asarray(
            result.volume.initial.parameters,
            dtype=np.float64,
        ),
        "initial:qfm_value": np.asarray(
            result.volume.initial.qfm_value,
            dtype=np.float64,
        ),
        "initial:qfm_gradient": np.asarray(
            result.volume.initial.qfm_gradient,
            dtype=np.float64,
        ),
    }
    provider_calls: list[QfmProviderCall] = []
    total_nit = 0
    total_nfev = 0
    total_njev = 0
    raw_statuses: list[str] = []
    for stage_name in _LABELS:
        stage = getattr(result, stage_name)
        values[f"{stage_name}:target"] = np.asarray(stage.target, dtype=np.float64)
        values.update(_prefixed(stage_name, "initial", _jax_state(stage.initial)))
        values.update(_prefixed(stage_name, "penalty", _jax_state(stage.penalty)))
        values.update(_prefixed(stage_name, "exact", _jax_state(stage.exact)))
        values[f"{stage_name}:volume_persistence_objective"] = np.asarray(
            stage.volume_persistence_objective,
            dtype=np.float64,
        )
        total_nit += int(stage.penalty_optimizer.nit) + int(stage.exact_optimizer.nit)
        total_nfev += int(stage.penalty_optimizer.nfev) + int(
            stage.exact_optimizer.nfev
        )
        total_njev += int(stage.penalty_optimizer.njev) + int(
            stage.exact_optimizer.njev
        )
        raw_statuses.extend(
            (
                f"{stage.penalty_optimizer.status}:{stage.penalty_optimizer.message}",
                f"{stage.exact_optimizer.status}:{stage.exact_optimizer.message}",
            )
        )
        provider_calls.extend(
            (
                _provider_call(
                    QFM_PENALTY_METHOD,
                    success=bool(stage.penalty_optimizer.success),
                    status=int(stage.penalty_optimizer.status),
                    iterations=int(stage.penalty_optimizer.nit),
                    max_iterations=max_steps,
                    state=values,
                    phase=f"{stage_name}:penalty",
                ),
                _provider_call(
                    QFM_EXACT_METHOD,
                    success=bool(stage.exact_optimizer.success),
                    status=int(stage.exact_optimizer.status),
                    iterations=int(stage.exact_optimizer.nit),
                    max_iterations=max_steps,
                    state=values,
                    phase=f"{stage_name}:exact",
                ),
            )
        )

    terminal = normalized_terminal_status(
        scientific_predicate=qfm_scientific_predicate(values, bundle.scale),
        stage_stopping_reasons=qfm_stopping_reasons(provider_calls),
    )
    platform = jax.devices()[0].platform
    return LaneObservation(
        lane=lane,
        backend_mode=os.environ["SIMSOPT_BACKEND_MODE"],
        platform="gpu" if platform in {"cuda", "gpu"} else platform,
        precision="fp64" if bool(jax.config.read("jax_enable_x64")) else "fp32",
        scale=bundle.scale,
        input_fingerprint=bundle.input_fingerprint,
        configuration_fingerprint=bundle.configuration_fingerprint,
        effective_construction_fingerprint=fingerprint,
        driver=QFM_HOST_SCIPY_DRIVER,
        normalized_status=terminal.normalized_status,
        raw_status=",".join(raw_statuses),
        success=terminal.success,
        nit=total_nit,
        nfev=total_nfev,
        njev=total_njev,
        completed_workflow_stages=WORKFLOW_STAGES,
        provenance=None,
        values=values,
    )


def execute(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> LaneObservation:
    """Execute the exact QFM sequence in the selected isolated lane."""
    if lane == "native-cpu":
        return _native(bundle, arrays)
    return _jax(lane, bundle, arrays)
