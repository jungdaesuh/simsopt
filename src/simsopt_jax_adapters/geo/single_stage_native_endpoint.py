"""Native NEQ-GNTR1 endpoint authority over caller-supplied historical bytes.

The module owns validated historical input, fixed native continuation, direct
716-state evaluation without an inner solve, and reduced 461-gradient telemetry.
Every operation restores the wrapped mutable SIMSOPT graph before returning.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Final, Protocol, cast

import jax.numpy as jnp
import numpy as np
from examples.jax.parity.cases.native_boozerqa import (
    _prepare_native_variant_runtime,
)
from examples.jax.parity.cases.native_single_stage_boozer_vacuum import SPEC
from examples.jax.parity.input_bundle import InputBundle
from simsopt.geo.surfaceobjectives import boozer_surface_residual
from simsopt_jax.objectives.single_stage_fullspace import (
    FROZEN_LAYOUT,
    FullSpaceObjectiveConfig,
    FullSpaceState,
)

from .single_stage_fullspace import SingleStageFullSpaceBootstrap

SCHEMA_VERSION: Final = "single-stage-native-endpoint-v1"
SSOT_SHA256: Final = "d082baa587b9db580ac3ef8c99a3123ed83564586b605200f7c2cfa6feb909a9"
EXACT_NEWTON_TOLERANCE: Final = 1.0e-13
EXACT_NEWTON_MAXIMUM_ITERATIONS: Final = 20
COARSE_SEGMENT_COUNT: Final = 256
REFINED_SEGMENT_COUNT: Final = 512
BRANCH_ROOT_TOLERANCE: Final = 1.0e-10
SCALED_BOOZER_TOLERANCE: Final = 1.0e-10
SEALED_OBSERVABLE_RTOL: Final = 1.0e-12
SEALED_OBSERVABLE_ATOL: Final = 1.0e-15
_COIL_DOF_COUNT: Final = 461
_SURFACE_DOF_COUNT: Final = 253
_BOOZER_EQUALITY_COUNT: Final = 254
_EQUALITY_COUNT: Final = 255
_PAYLOAD_DTYPE: Final = "<f8"


class NativeEndpointError(RuntimeError):
    """The native reference or endpoint evidence failed a frozen gate."""


@dataclass(frozen=True, slots=True)
class HistoricalNativeObservablePaths:
    objective: tuple[str, ...]
    iota: tuple[str, ...]
    volume: tuple[str, ...]
    non_qs: tuple[str, ...]
    boozer_residual_value: tuple[str, ...]
    boozer_residual_rms: tuple[str, ...]
    major_radius_penalty: tuple[str, ...]
    length_penalty: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class HistoricalNativeParameterMetadata:
    """Caller-owned extraction contract for one exact historical JSON payload."""

    source_sha256: str
    parameter_path: tuple[str, ...]
    parameter_little_endian_sha256: str
    parameter_dtype: str
    parameter_shape: tuple[int, ...]
    observable_paths: HistoricalNativeObservablePaths


@dataclass(frozen=True, slots=True)
class SealedNativeObservables:
    objective: float
    iota: float
    volume: float
    non_qs: float
    boozer_residual_value: float
    boozer_residual_rms: float
    major_radius_penalty: float
    length_penalty: float


@dataclass(frozen=True, slots=True)
class HistoricalNativeParameters:
    source_sha256: str
    parameter_little_endian_sha256: str
    parameters: np.ndarray
    sealed_observables: SealedNativeObservables


@dataclass(frozen=True, slots=True)
class NativeObjectiveTerms:
    non_qs: float
    residual: float
    iota: float
    major_radius: float
    length: float


@dataclass(frozen=True, slots=True)
class NativeFullSpaceObjectiveContract:
    """Exact full-space targets, weights, and static objective choices."""

    iota_target: float
    major_radius_target: float
    length_target: float
    volume_target: float
    non_qs_weight: float
    residual_weight: float
    iota_weight: float
    major_radius_weight: float
    length_weight: float
    non_qs_axis: int
    weight_inv_modB: bool


@dataclass(frozen=True, slots=True)
class NativeStateObservables:
    iota: float
    G: float
    volume: float
    major_radius: float
    total_length: float
    non_qs_ratio: float
    boozer_residual_value: float
    boozer_residual_rms: float
    fixed_first_base_current: float


@dataclass(frozen=True, slots=True)
class NativeExplicitStateEvaluation:
    """Genuine native values at one supplied 461+253+1+1 state."""

    state: np.ndarray
    state_little_endian_sha256: str
    objective_terms: NativeObjectiveTerms
    objective: float
    observables: NativeStateObservables
    masked_boozer_equalities: np.ndarray
    volume_equality: float
    raw_equalities: np.ndarray
    all_finite: bool


@dataclass(frozen=True, slots=True)
class NativeReducedEvaluation:
    """Native implicit objective and 461-gradient telemetry."""

    parameters: np.ndarray
    objective: float
    gradient: np.ndarray
    gradient_infinity_norm: float
    gradient_l2_norm: float
    inner_solver_success: bool
    solver_residual_l2: float
    solver_residual_infinity_norm: float
    all_finite: bool


@dataclass(frozen=True, slots=True)
class NativeContinuationStep:
    segment_count: int
    index: int
    predecessor_index: int | None
    coil_little_endian_sha256: str
    seed_root_little_endian_sha256: str
    root_little_endian_sha256: str
    newton_iterations: int
    raw_equalities: np.ndarray
    raw_equalities_little_endian_sha256: str
    residual_l2: float
    residual_infinity_norm: float
    scaled_boozer_infinity_norm: float


@dataclass(frozen=True, slots=True)
class NativeDyadicPathEvidence:
    segment_count: int
    roots: np.ndarray
    steps: tuple[NativeContinuationStep, ...]


@dataclass(frozen=True, slots=True)
class NativeReferenceEvidence:
    schema_version: str
    ssot_sha256: str
    historical_input: HistoricalNativeParameters
    state: np.ndarray
    endpoint: NativeExplicitStateEvaluation
    coarse_path: NativeDyadicPathEvidence
    refined_path: NativeDyadicPathEvidence
    common_knot_root_infinity_difference: float
    sealed_observables_match: bool
    fixed_first_base_current: float
    usable: bool


@dataclass(frozen=True, slots=True)
class NativeAcceptedIntervalEvidence:
    index: int
    supplied_state_little_endian_sha256: str
    direct_step: NativeContinuationStep
    midpoint_step: NativeContinuationStep
    refined_step: NativeContinuationStep
    direct_root: np.ndarray
    midpoint_root: np.ndarray
    refined_root: np.ndarray
    direct_refined_infinity_difference: float
    supplied_refined_infinity_difference: float


@dataclass(frozen=True, slots=True)
class NativeAcceptedPathEvidence:
    bootstrap_step: NativeContinuationStep
    intervals: tuple[NativeAcceptedIntervalEvidence, ...]
    first_failing_index: int | None
    failure_reason: str | None
    usable: bool


class _NativeCurrent(Protocol):
    def get_value(self) -> float: ...


class _NativeCoil(Protocol):
    current: _NativeCurrent


class _NativeBiotSavart(Protocol):
    coils: Sequence[_NativeCoil]


class _NativeSurface(Protocol):
    def get_dofs(self) -> np.ndarray: ...

    def set_dofs(self, dofs: np.ndarray) -> None: ...

    def major_radius(self) -> float: ...


class _NativeSolver(Protocol):
    surface: _NativeSurface
    biotsavart: _NativeBiotSavart
    res: Mapping[str, object]
    need_to_run_code: bool

    def solve_residual_equation_exactly_newton(
        self,
        *,
        tol: float,
        maxiter: int,
        iota: float,
        G: float,
        verbose: bool,
    ) -> Mapping[str, object]: ...


class _NativeObjective(Protocol):
    x: np.ndarray

    def J(self) -> float: ...

    def dJ(self) -> np.ndarray: ...


class _NativeDirectNonQs(_NativeObjective, Protocol):
    axis: int

    def fixed_surface_value_and_derivative(self) -> tuple[float, object]: ...


class _NativeDirectResidual(_NativeObjective, Protocol):
    def fixed_surface_value_derivative_and_y_partial(
        self,
        iota: float,
        G: float,
        *,
        weight_inv_modB: bool,
    ) -> tuple[float, object, np.ndarray]: ...


class _NativePenalty(Protocol):
    obj: _NativeObjective

    def J(self) -> float: ...


class _NativeBaselineAnchor(Protocol):
    iota: float
    G: float
    iota_target: float
    volume_target: float
    major_radius_target: float
    total_length_target: float
    inner_solver_success: bool


class _NativeCandidateResult(Protocol):
    objective: float
    gradient: np.ndarray
    inner_solver_success: bool
    solver_residual_l2: float
    solver_residual_inf: float


class _PreparedNativeRuntime(Protocol):
    solver: _NativeSolver
    objective: _NativeObjective
    non_qs: _NativeDirectNonQs
    residual: _NativeDirectResidual
    volume: _NativeObjective
    radius_penalty: _NativePenalty
    length_penalty: _NativePenalty
    initial_parameters: np.ndarray
    initial_solution_success: bool
    baseline_anchor: _NativeBaselineAnchor

    def evaluate_candidate(self, parameters: np.ndarray) -> _NativeCandidateResult: ...


@dataclass(frozen=True, slots=True)
class _NativeGraphSnapshot:
    parameters: np.ndarray
    surface_dofs: np.ndarray
    solver_result: Mapping[str, object]
    need_to_run_code: bool


@dataclass(frozen=True, slots=True)
class NativeSingleStageEndpointRuntime:
    """Thread-safe façade over one mutable authoritative native runtime."""

    _prepared: _PreparedNativeRuntime = field(repr=False, compare=False)
    bootstrap_state: np.ndarray
    bootstrap_root: np.ndarray
    exact_mask: np.ndarray
    exact_mask_indices: np.ndarray
    objective_contract: NativeFullSpaceObjectiveContract
    fixed_first_base_current: float
    _lock: threading.RLock = field(
        default_factory=threading.RLock,
        repr=False,
        compare=False,
    )

    def evaluate_state(self, state: np.ndarray) -> NativeExplicitStateEvaluation:
        """Evaluate a supplied full state without invoking an inner solve."""

        with self._lock, _restore_native_graph(self._prepared):
            return _evaluate_state(self, state)

    def evaluate_reduced(self, parameters: np.ndarray) -> NativeReducedEvaluation:
        """Evaluate the native implicit reduced objective and gradient."""

        checked = _require_fp64_vector(
            parameters,
            _COIL_DOF_COUNT,
            "native reduced parameters",
        )
        with self._lock, _restore_native_graph(self._prepared):
            result = self._prepared.evaluate_candidate(checked)
            gradient = _require_fp64_vector(
                result.gradient,
                _COIL_DOF_COUNT,
                "native reduced gradient",
            )
            scalars = np.asarray(
                (
                    result.objective,
                    result.solver_residual_l2,
                    result.solver_residual_inf,
                ),
                dtype=np.float64,
            )
            all_finite = bool(np.all(np.isfinite(scalars)))
            if not all_finite:
                raise NativeEndpointError("native reduced evaluation is nonfinite")
            return NativeReducedEvaluation(
                parameters=_readonly_array(checked),
                objective=float(result.objective),
                gradient=_readonly_array(gradient),
                gradient_infinity_norm=float(np.linalg.norm(gradient, ord=np.inf)),
                gradient_l2_norm=float(np.linalg.norm(gradient)),
                inner_solver_success=bool(result.inner_solver_success),
                solver_residual_l2=float(result.solver_residual_l2),
                solver_residual_infinity_norm=float(result.solver_residual_inf),
                all_finite=True,
            )

    def reconstruct_native_reference(
        self,
        historical: HistoricalNativeParameters,
    ) -> NativeReferenceEvidence:
        """Reconstruct and refine the frozen historical native endpoint branch."""

        with self._lock, _restore_native_graph(self._prepared):
            coarse = _replay_dyadic_path(
                self,
                historical.parameters,
                COARSE_SEGMENT_COUNT,
            )
            refined = _replay_dyadic_path(
                self,
                historical.parameters,
                REFINED_SEGMENT_COUNT,
            )
            common_difference = float(np.max(np.abs(coarse.roots - refined.roots[::2])))
            if common_difference > BRANCH_ROOT_TOLERANCE:
                raise NativeEndpointError(
                    "native 256/512 common-knot roots exceed the branch tolerance"
                )
            terminal_parameters = _path_parameters(
                self.bootstrap_state[:_COIL_DOF_COUNT],
                historical.parameters,
                COARSE_SEGMENT_COUNT,
                COARSE_SEGMENT_COUNT,
            )
            if not np.array_equal(terminal_parameters, historical.parameters):
                raise NativeEndpointError(
                    "native continuation terminal parameters differ from history"
                )
            terminal_root = coarse.roots[-1]
            state = _pack_state(terminal_parameters, terminal_root)
            endpoint = _evaluate_state(self, state)
            sealed_match = _sealed_observables_match(
                endpoint,
                historical.sealed_observables,
            )
            if not sealed_match:
                raise NativeEndpointError(
                    "reconstructed native endpoint differs from sealed observables"
                )
            return NativeReferenceEvidence(
                schema_version=SCHEMA_VERSION,
                ssot_sha256=SSOT_SHA256,
                historical_input=historical,
                state=_readonly_array(state),
                endpoint=endpoint,
                coarse_path=coarse,
                refined_path=refined,
                common_knot_root_infinity_difference=common_difference,
                sealed_observables_match=True,
                fixed_first_base_current=self.fixed_first_base_current,
                usable=True,
            )

    def audit_accepted_states(
        self,
        physical_states: np.ndarray,
    ) -> NativeAcceptedPathEvidence:
        """Audit direct and midpoint-refined native roots for accepted states."""

        states = _require_fp64_matrix(
            physical_states,
            FROZEN_LAYOUT.total_dof_count,
            "accepted physical states",
        )
        if states.shape[0] < 1 or states.shape[0] > COARSE_SEGMENT_COUNT + 1:
            raise ValueError("accepted physical states must contain 1..257 rows")
        if not np.array_equal(states[0], self.bootstrap_state):
            raise NativeEndpointError("accepted-state row 0 is not the bootstrap state")
        intervals: list[NativeAcceptedIntervalEvidence] = []
        predecessor_root = np.array(self.bootstrap_root, copy=True)
        predecessor_coil = np.array(
            self.bootstrap_state[:_COIL_DOF_COUNT],
            copy=True,
        )
        with self._lock, _restore_native_graph(self._prepared):
            self._prepared.objective.x = predecessor_coil
            initial_masked, initial_equalities, _volume = _raw_equalities(
                self,
                self.bootstrap_root,
            )
            initial_scaled_boozer_inf = float(
                np.linalg.norm(initial_masked, ord=np.inf)
                / np.sqrt(_BOOZER_EQUALITY_COUNT)
            )
            bootstrap_step = NativeContinuationStep(
                segment_count=states.shape[0] - 1,
                index=0,
                predecessor_index=None,
                coil_little_endian_sha256=_array_sha256(predecessor_coil),
                seed_root_little_endian_sha256=_array_sha256(predecessor_root),
                root_little_endian_sha256=_array_sha256(predecessor_root),
                newton_iterations=0,
                raw_equalities=_readonly_array(initial_equalities),
                raw_equalities_little_endian_sha256=_array_sha256(initial_equalities),
                residual_l2=float(np.linalg.norm(initial_equalities)),
                residual_infinity_norm=float(
                    np.linalg.norm(initial_equalities, ord=np.inf)
                ),
                scaled_boozer_infinity_norm=initial_scaled_boozer_inf,
            )
            first_current = float(
                self._prepared.solver.biotsavart.coils[0].current.get_value()
            )
            if first_current != self.fixed_first_base_current:
                return NativeAcceptedPathEvidence(
                    bootstrap_step=bootstrap_step,
                    intervals=(),
                    first_failing_index=0,
                    failure_reason="accepted bootstrap fixed current changed",
                    usable=False,
                )
            if initial_scaled_boozer_inf > SCALED_BOOZER_TOLERANCE:
                return NativeAcceptedPathEvidence(
                    bootstrap_step=bootstrap_step,
                    intervals=(),
                    first_failing_index=0,
                    failure_reason="accepted bootstrap scaled Boozer gate failed",
                    usable=False,
                )
            for index in range(1, states.shape[0]):
                supplied = states[index]
                candidate_coil = supplied[:_COIL_DOF_COUNT]
                supplied_root = supplied[_COIL_DOF_COUNT:]
                try:
                    direct, direct_step = _solve_from_seed(
                        self,
                        candidate_coil,
                        predecessor_root,
                        segment_count=states.shape[0] - 1,
                        index=index,
                        predecessor_index=index - 1,
                    )
                    midpoint_coil = predecessor_coil + np.float64(0.5) * (
                        candidate_coil - predecessor_coil
                    )
                    midpoint, midpoint_step = _solve_from_seed(
                        self,
                        midpoint_coil,
                        predecessor_root,
                        segment_count=2 * (states.shape[0] - 1),
                        index=2 * index - 1,
                        predecessor_index=2 * index - 2,
                    )
                    refined, refined_step = _solve_from_seed(
                        self,
                        candidate_coil,
                        midpoint,
                        segment_count=2 * (states.shape[0] - 1),
                        index=2 * index,
                        predecessor_index=2 * index - 1,
                    )
                except NativeEndpointError as error:
                    return NativeAcceptedPathEvidence(
                        bootstrap_step=bootstrap_step,
                        intervals=tuple(intervals),
                        first_failing_index=index,
                        failure_reason=str(error),
                        usable=False,
                    )
                direct_refined = float(np.max(np.abs(direct - refined)))
                supplied_refined = float(np.max(np.abs(supplied_root - refined)))
                interval = NativeAcceptedIntervalEvidence(
                    index=index,
                    supplied_state_little_endian_sha256=_array_sha256(supplied),
                    direct_step=direct_step,
                    midpoint_step=midpoint_step,
                    refined_step=refined_step,
                    direct_root=_readonly_array(direct),
                    midpoint_root=_readonly_array(midpoint),
                    refined_root=_readonly_array(refined),
                    direct_refined_infinity_difference=direct_refined,
                    supplied_refined_infinity_difference=supplied_refined,
                )
                intervals.append(interval)
                if (
                    direct_refined > BRANCH_ROOT_TOLERANCE
                    or supplied_refined > BRANCH_ROOT_TOLERANCE
                ):
                    return NativeAcceptedPathEvidence(
                        bootstrap_step=bootstrap_step,
                        intervals=tuple(intervals),
                        first_failing_index=index,
                        failure_reason=(
                            f"accepted native branch mismatch at row {index}"
                        ),
                        usable=False,
                    )
                predecessor_coil = np.array(candidate_coil, copy=True)
                predecessor_root = np.array(refined, copy=True)
        return NativeAcceptedPathEvidence(
            bootstrap_step=bootstrap_step,
            intervals=tuple(intervals),
            first_failing_index=None,
            failure_reason=None,
            usable=True,
        )


def _json_value_at(document: object, path: tuple[str, ...], name: str) -> object:
    value = document
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            raise ValueError(f"historical native source lacks {name}")
        value = value[key]
    return value


def _finite_json_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"historical native {name} must be a JSON number")
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"historical native {name} must be finite")
    return scalar


def load_historical_native_parameters(
    source_bytes: bytes,
    metadata: HistoricalNativeParameterMetadata,
) -> HistoricalNativeParameters:
    """Verify source bytes and extract the retained FP64 461-vector and scalars."""

    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if source_sha256 != metadata.source_sha256:
        raise NativeEndpointError("historical native source SHA-256 mismatch")
    if metadata.parameter_dtype != _PAYLOAD_DTYPE:
        raise TypeError("historical native parameter dtype must be <f8")
    if metadata.parameter_shape != (_COIL_DOF_COUNT,):
        raise ValueError("historical native parameter shape must be (461,)")
    document: object = json.loads(source_bytes)
    raw_parameters = _json_value_at(
        document,
        metadata.parameter_path,
        "parameters",
    )
    if not isinstance(raw_parameters, list) or len(raw_parameters) != _COIL_DOF_COUNT:
        raise ValueError("historical native parameters must contain 461 values")
    parameters = np.asarray(
        tuple(
            _finite_json_float(value, f"parameters[{index}]")
            for index, value in enumerate(raw_parameters)
        ),
        dtype=np.float64,
    )
    parameter_sha256 = _array_sha256(parameters)
    if parameter_sha256 != metadata.parameter_little_endian_sha256:
        raise NativeEndpointError("historical native parameter SHA-256 mismatch")
    paths = metadata.observable_paths
    sealed = SealedNativeObservables(
        objective=_finite_json_float(
            _json_value_at(document, paths.objective, "objective"),
            "objective",
        ),
        iota=_finite_json_float(
            _json_value_at(document, paths.iota, "iota"),
            "iota",
        ),
        volume=_finite_json_float(
            _json_value_at(document, paths.volume, "volume"),
            "volume",
        ),
        non_qs=_finite_json_float(
            _json_value_at(document, paths.non_qs, "non_qs"),
            "non_qs",
        ),
        boozer_residual_value=_finite_json_float(
            _json_value_at(
                document,
                paths.boozer_residual_value,
                "boozer_residual_value",
            ),
            "boozer_residual_value",
        ),
        boozer_residual_rms=_finite_json_float(
            _json_value_at(
                document,
                paths.boozer_residual_rms,
                "boozer_residual_rms",
            ),
            "boozer_residual_rms",
        ),
        major_radius_penalty=_finite_json_float(
            _json_value_at(
                document,
                paths.major_radius_penalty,
                "major_radius_penalty",
            ),
            "major_radius_penalty",
        ),
        length_penalty=_finite_json_float(
            _json_value_at(document, paths.length_penalty, "length_penalty"),
            "length_penalty",
        ),
    )
    return HistoricalNativeParameters(
        source_sha256=source_sha256,
        parameter_little_endian_sha256=parameter_sha256,
        parameters=_readonly_array(parameters),
        sealed_observables=sealed,
    )


def _config_fp64_scalar(value: object, name: str) -> float:
    scalar = np.asarray(value)
    if scalar.dtype != np.dtype(np.float64) or scalar.shape != ():
        raise TypeError(f"full-space {name} must be a float64 scalar")
    result = float(scalar)
    if not np.isfinite(result):
        raise ValueError(f"full-space {name} must be finite")
    return result


def _objective_contract(
    config: FullSpaceObjectiveConfig,
) -> NativeFullSpaceObjectiveContract:
    weights = (
        _config_fp64_scalar(config.non_qs_weight, "non_qs_weight"),
        _config_fp64_scalar(config.residual_weight, "residual_weight"),
        _config_fp64_scalar(config.iota_weight, "iota_weight"),
        _config_fp64_scalar(config.major_radius_weight, "major_radius_weight"),
        _config_fp64_scalar(config.length_weight, "length_weight"),
    )
    if any(weight < 0.0 for weight in weights):
        raise ValueError("full-space objective weights must be nonnegative")
    if isinstance(config.non_qs_axis, bool) or config.non_qs_axis not in (0, 1):
        raise ValueError("full-space non_qs_axis must be 0 or 1")
    if not isinstance(config.weight_inv_modB, bool):
        raise TypeError("full-space weight_inv_modB must be bool")
    return NativeFullSpaceObjectiveContract(
        iota_target=_config_fp64_scalar(config.iota_target, "iota_target"),
        major_radius_target=_config_fp64_scalar(
            config.major_radius_target,
            "major_radius_target",
        ),
        length_target=_config_fp64_scalar(config.length_target, "length_target"),
        volume_target=_config_fp64_scalar(config.volume_target, "volume_target"),
        non_qs_weight=weights[0],
        residual_weight=weights[1],
        iota_weight=weights[2],
        major_radius_weight=weights[3],
        length_weight=weights[4],
        non_qs_axis=config.non_qs_axis,
        weight_inv_modB=config.weight_inv_modB,
    )


def build_native_single_stage_endpoint_runtime(
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
    bootstrap: SingleStageFullSpaceBootstrap,
) -> NativeSingleStageEndpointRuntime:
    """Bind the canonical bootstrap to the authoritative native example runtime."""

    if bundle.case_id != SPEC.case_id or bundle.scale != "native_default":
        raise ValueError("native endpoint runtime requires the native-default NEQ case")
    bootstrap_state = _require_fp64_vector(
        np.asarray(bootstrap.z0),
        FROZEN_LAYOUT.total_dof_count,
        "canonical bootstrap state",
    )
    objective_contract = _objective_contract(bootstrap.problem.config)
    prepared = cast(
        _PreparedNativeRuntime,
        _prepare_native_variant_runtime(bundle, arrays, SPEC),
    )
    if (
        not prepared.initial_solution_success
        or not prepared.baseline_anchor.inner_solver_success
    ):
        raise NativeEndpointError("authoritative native bootstrap solve failed")
    initial_parameters = _require_fp64_vector(
        prepared.initial_parameters,
        _COIL_DOF_COUNT,
        "native bootstrap parameters",
    )
    if not np.array_equal(initial_parameters, bootstrap_state[:_COIL_DOF_COUNT]):
        raise NativeEndpointError("native and full-space bootstrap coils differ")
    native_root = _root_from_components(
        prepared.solver.surface.get_dofs(),
        prepared.baseline_anchor.iota,
        prepared.baseline_anchor.G,
    )
    bootstrap_root = bootstrap_state[_COIL_DOF_COUNT:]
    if float(np.max(np.abs(native_root - bootstrap_root))) > BRANCH_ROOT_TOLERANCE:
        raise NativeEndpointError("native and full-space bootstrap roots differ")
    native_major_radius = float(prepared.solver.surface.major_radius())
    if (
        prepared.baseline_anchor.iota_target != float(native_root[-2])
        or prepared.baseline_anchor.major_radius_target != native_major_radius
    ):
        raise NativeEndpointError("native objective targets differ from native state")
    with _restore_native_graph(prepared):
        prepared.solver.surface.set_dofs(bootstrap_root[:_SURFACE_DOF_COUNT])
        bootstrap_major_radius = float(prepared.solver.surface.major_radius())
    if (
        objective_contract.iota_target != float(bootstrap_root[-2])
        or objective_contract.major_radius_target != bootstrap_major_radius
        or objective_contract.length_target
        != prepared.baseline_anchor.total_length_target
        or objective_contract.volume_target != prepared.baseline_anchor.volume_target
    ):
        raise NativeEndpointError(
            "full-space objective targets differ from canonical state/invariants"
        )
    native_weights = np.asarray((1.0, SPEC.residual_weight, 1.0, 1.0, 1.0))
    fullspace_weights = np.asarray(
        (
            objective_contract.non_qs_weight,
            objective_contract.residual_weight,
            objective_contract.iota_weight,
            objective_contract.major_radius_weight,
            objective_contract.length_weight,
        )
    )
    if not np.array_equal(native_weights, fullspace_weights):
        raise NativeEndpointError("native and full-space objective weights differ")
    if prepared.non_qs.axis != objective_contract.non_qs_axis:
        raise NativeEndpointError("native and full-space non-QS axes differ")
    if objective_contract.weight_inv_modB:
        raise NativeEndpointError("native exact route requires weight_inv_modB=False")
    exact_mask = _validated_mask(prepared.solver.res.get("mask"))
    exact_mask_indices = np.asarray(bootstrap.problem.exact_mask_indices)
    if exact_mask_indices.dtype != np.dtype(np.int32):
        raise TypeError("full-space exact mask indices must use int32")
    if exact_mask_indices.shape != (_BOOZER_EQUALITY_COUNT,):
        raise ValueError("full-space exact mask indices must have shape (254,)")
    native_mask_indices = np.flatnonzero(exact_mask).astype(np.int32, copy=False)
    if not np.array_equal(native_mask_indices, exact_mask_indices):
        raise NativeEndpointError(
            "native exact mask/order differs from full-space exact_mask_indices"
        )
    first_current = float(prepared.solver.biotsavart.coils[0].current.get_value())
    expected_current = bootstrap.first_base_current.value
    if not np.isfinite(first_current) or first_current != expected_current:
        raise NativeEndpointError("native fixed first base current differs")
    return NativeSingleStageEndpointRuntime(
        _prepared=prepared,
        bootstrap_state=_readonly_array(bootstrap_state),
        bootstrap_root=_readonly_array(bootstrap_root),
        exact_mask=_readonly_array(exact_mask, dtype=np.bool_),
        exact_mask_indices=_readonly_array(exact_mask_indices, dtype=np.int32),
        objective_contract=objective_contract,
        fixed_first_base_current=first_current,
    )


def _readonly_array(
    values: np.ndarray,
    *,
    dtype: np.dtype[np.generic] | type[np.generic] = np.float64,
) -> np.ndarray:
    array = np.array(values, dtype=dtype, copy=True, order="C")
    array.setflags(write=False)
    return array


def _require_fp64_vector(values: np.ndarray, size: int, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype != np.dtype(np.float64):
        raise TypeError(f"{name} must use float64")
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return np.array(array, copy=True, order="C")


def _require_fp64_matrix(values: np.ndarray, width: int, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype != np.dtype(np.float64):
        raise TypeError(f"{name} must use float64")
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"{name} must have shape (n, {width})")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return np.array(array, copy=True, order="C")


def _array_sha256(values: np.ndarray) -> str:
    canonical = np.ascontiguousarray(values, dtype=np.dtype("<f8"))
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _root_from_components(
    surface_dofs: np.ndarray,
    iota: object,
    G: object,
) -> np.ndarray:
    surface = _require_fp64_vector(
        np.asarray(surface_dofs),
        _SURFACE_DOF_COUNT,
        "native surface state",
    )
    scalars = np.asarray((iota, G), dtype=np.float64)
    if not np.all(np.isfinite(scalars)):
        raise NativeEndpointError("native iota/G state is nonfinite")
    return np.concatenate((surface, scalars))


def _pack_state(coil_dofs: np.ndarray, root: np.ndarray) -> np.ndarray:
    checked_coils = _require_fp64_vector(
        coil_dofs,
        _COIL_DOF_COUNT,
        "native coil state",
    )
    checked_root = _require_fp64_vector(
        root,
        _SURFACE_DOF_COUNT + 2,
        "native inner root",
    )
    packed = FROZEN_LAYOUT.pack(
        FullSpaceState(
            coil_dofs=jnp.asarray(checked_coils, dtype=jnp.float64),
            surface_dofs=jnp.asarray(
                checked_root[:_SURFACE_DOF_COUNT],
                dtype=jnp.float64,
            ),
            iota=jnp.asarray(checked_root[-2], dtype=jnp.float64),
            G=jnp.asarray(checked_root[-1], dtype=jnp.float64),
        )
    )
    return _require_fp64_vector(
        np.asarray(packed),
        FROZEN_LAYOUT.total_dof_count,
        "packed native full-space state",
    )


def _snapshot_native_graph(prepared: _PreparedNativeRuntime) -> _NativeGraphSnapshot:
    return _NativeGraphSnapshot(
        parameters=np.array(prepared.objective.x, dtype=np.float64, copy=True),
        surface_dofs=np.array(
            prepared.solver.surface.get_dofs(),
            dtype=np.float64,
            copy=True,
        ),
        solver_result=dict(prepared.solver.res),
        need_to_run_code=bool(prepared.solver.need_to_run_code),
    )


def _restore_snapshot(
    prepared: _PreparedNativeRuntime,
    snapshot: _NativeGraphSnapshot,
) -> None:
    prepared.objective.x = snapshot.parameters
    prepared.solver.surface.set_dofs(snapshot.surface_dofs)
    prepared.solver.res = snapshot.solver_result
    prepared.solver.need_to_run_code = snapshot.need_to_run_code


@contextmanager
def _restore_native_graph(prepared: _PreparedNativeRuntime) -> Iterator[None]:
    snapshot = _snapshot_native_graph(prepared)
    try:
        yield
    finally:
        _restore_snapshot(prepared, snapshot)


def _validated_mask(value: object) -> np.ndarray:
    mask = np.asarray(value)
    if mask.dtype != np.dtype(np.bool_) or mask.ndim != 1:
        raise NativeEndpointError("native exact residual mask must be a bool vector")
    if int(np.count_nonzero(mask)) != _BOOZER_EQUALITY_COUNT:
        raise NativeEndpointError("native exact residual mask must select 254 values")
    return np.array(mask, dtype=np.bool_, copy=True)


def _raw_equalities(
    runtime: NativeSingleStageEndpointRuntime,
    root: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    prepared = runtime._prepared
    surface = prepared.solver.surface
    surface.set_dofs(root[:_SURFACE_DOF_COUNT])
    full_residual = np.asarray(
        boozer_surface_residual(
            surface,
            float(root[-2]),
            float(root[-1]),
            prepared.solver.biotsavart,
            derivatives=0,
            weight_inv_modB=runtime.objective_contract.weight_inv_modB,
        )[0]
    )
    if full_residual.dtype != np.dtype(np.float64):
        raise TypeError("native Boozer residual must use float64")
    if full_residual.shape != runtime.exact_mask.shape:
        raise NativeEndpointError("native Boozer residual/mask ordering differs")
    masked = full_residual[runtime.exact_mask_indices]
    volume_equality = float(
        prepared.volume.J() - runtime.objective_contract.volume_target
    )
    equalities = np.concatenate((masked, np.asarray((volume_equality,))))
    if equalities.shape != (_EQUALITY_COUNT,):
        raise NativeEndpointError("native equality ordering is not 254+1")
    if not np.all(np.isfinite(equalities)):
        raise NativeEndpointError("native equalities are nonfinite")
    return masked, equalities, volume_equality


def _evaluate_state(
    runtime: NativeSingleStageEndpointRuntime,
    state: np.ndarray,
) -> NativeExplicitStateEvaluation:
    checked = _require_fp64_vector(
        state,
        FROZEN_LAYOUT.total_dof_count,
        "native explicit state",
    )
    unpacked = FROZEN_LAYOUT.unpack(jnp.asarray(checked, dtype=jnp.float64))
    coils = np.asarray(unpacked.coil_dofs, dtype=np.float64)
    surface_dofs = np.asarray(unpacked.surface_dofs, dtype=np.float64)
    iota = float(np.asarray(unpacked.iota))
    G = float(np.asarray(unpacked.G))
    prepared = runtime._prepared
    contract = runtime.objective_contract
    prepared.objective.x = coils
    prepared.solver.surface.set_dofs(surface_dofs)
    non_qs = float(prepared.non_qs.fixed_surface_value_and_derivative()[0])
    residual = float(
        prepared.residual.fixed_surface_value_derivative_and_y_partial(
            iota,
            G,
            weight_inv_modB=contract.weight_inv_modB,
        )[0]
    )
    volume = float(prepared.volume.J())
    major_radius = float(prepared.solver.surface.major_radius())
    total_length = float(prepared.length_penalty.obj.J())
    iota_penalty = 0.5 * (iota - contract.iota_target) ** 2
    major_radius_penalty = 0.5 * (major_radius - contract.major_radius_target) ** 2
    length_penalty = (
        0.5
        * max(
            total_length - contract.length_target,
            0.0,
        )
        ** 2
    )
    terms = NativeObjectiveTerms(
        non_qs=non_qs,
        residual=residual,
        iota=iota_penalty,
        major_radius=major_radius_penalty,
        length=length_penalty,
    )
    objective = float(
        contract.non_qs_weight * terms.non_qs
        + contract.residual_weight * terms.residual
        + contract.iota_weight * terms.iota
        + contract.major_radius_weight * terms.major_radius
        + contract.length_weight * terms.length
    )
    root = np.concatenate((surface_dofs, np.asarray((iota, G))))
    masked, raw_equalities, volume_equality = _raw_equalities(runtime, root)
    first_current = float(prepared.solver.biotsavart.coils[0].current.get_value())
    if first_current != runtime.fixed_first_base_current:
        raise NativeEndpointError("native fixed first base current changed")
    observables = NativeStateObservables(
        iota=iota,
        G=G,
        volume=volume,
        major_radius=major_radius,
        total_length=total_length,
        non_qs_ratio=non_qs,
        boozer_residual_value=residual,
        boozer_residual_rms=float(np.sqrt(2.0 * residual)),
        fixed_first_base_current=first_current,
    )
    finite_values = np.asarray(
        (
            objective,
            terms.non_qs,
            terms.residual,
            terms.iota,
            terms.major_radius,
            terms.length,
            iota,
            G,
            volume,
            major_radius,
            total_length,
            observables.boozer_residual_rms,
        ),
        dtype=np.float64,
    )
    all_finite = bool(
        np.all(np.isfinite(checked))
        and np.all(np.isfinite(raw_equalities))
        and np.all(np.isfinite(finite_values))
    )
    if not all_finite:
        raise NativeEndpointError("native explicit-state evaluation is nonfinite")
    return NativeExplicitStateEvaluation(
        state=_readonly_array(checked),
        state_little_endian_sha256=_array_sha256(checked),
        objective_terms=terms,
        objective=float(objective),
        observables=observables,
        masked_boozer_equalities=_readonly_array(masked),
        volume_equality=volume_equality,
        raw_equalities=_readonly_array(raw_equalities),
        all_finite=True,
    )


def _path_parameters(
    initial: np.ndarray,
    final: np.ndarray,
    segment_count: int,
    index: int,
) -> np.ndarray:
    if index == 0:
        return np.array(initial, copy=True)
    if index == segment_count:
        return np.array(final, copy=True)
    fraction = np.float64(index) / np.float64(segment_count)
    return initial + fraction * (final - initial)


def _solve_from_seed(
    runtime: NativeSingleStageEndpointRuntime,
    parameters: np.ndarray,
    seed_root: np.ndarray,
    *,
    segment_count: int,
    index: int,
    predecessor_index: int,
) -> tuple[np.ndarray, NativeContinuationStep]:
    checked_parameters = _require_fp64_vector(
        parameters,
        _COIL_DOF_COUNT,
        "native continuation parameters",
    )
    checked_seed = _require_fp64_vector(
        seed_root,
        _SURFACE_DOF_COUNT + 2,
        "native continuation seed",
    )
    prepared = runtime._prepared
    prepared.objective.x = checked_parameters
    prepared.solver.surface.set_dofs(checked_seed[:_SURFACE_DOF_COUNT])
    prepared.solver.need_to_run_code = True
    result = prepared.solver.solve_residual_equation_exactly_newton(
        tol=EXACT_NEWTON_TOLERANCE,
        maxiter=EXACT_NEWTON_MAXIMUM_ITERATIONS,
        iota=float(checked_seed[-2]),
        G=float(checked_seed[-1]),
        verbose=False,
    )
    success = result.get("success")
    if not isinstance(success, (bool, np.bool_)):
        raise NativeEndpointError("native continuation success flag must be bool")
    if not bool(success):
        raise NativeEndpointError(
            f"native continuation solve failed at {segment_count}:{index}"
        )
    result_mask = _validated_mask(result.get("mask"))
    if not np.array_equal(result_mask, runtime.exact_mask):
        raise NativeEndpointError(
            f"native residual ordering changed at {segment_count}:{index}"
        )
    root = _root_from_components(
        prepared.solver.surface.get_dofs(),
        result.get("iota"),
        result.get("G"),
    )
    masked, equalities, _volume = _raw_equalities(runtime, root)
    residual_l2 = float(np.linalg.norm(equalities))
    residual_inf = float(np.linalg.norm(equalities, ord=np.inf))
    scaled_boozer_inf = float(
        np.linalg.norm(masked, ord=np.inf) / np.sqrt(_BOOZER_EQUALITY_COUNT)
    )
    if scaled_boozer_inf > SCALED_BOOZER_TOLERANCE:
        raise NativeEndpointError(
            f"native scaled Boozer gate failed at {segment_count}:{index}"
        )
    first_current = float(prepared.solver.biotsavart.coils[0].current.get_value())
    if first_current != runtime.fixed_first_base_current:
        raise NativeEndpointError("native fixed first base current changed")
    iterations = result.get("iter")
    if isinstance(iterations, bool) or not isinstance(iterations, (int, np.integer)):
        raise NativeEndpointError("native Newton iteration evidence is invalid")
    if int(iterations) < 0 or int(iterations) > EXACT_NEWTON_MAXIMUM_ITERATIONS:
        raise NativeEndpointError("native Newton iterations must be in [0, 20]")
    return root, NativeContinuationStep(
        segment_count=segment_count,
        index=index,
        predecessor_index=predecessor_index,
        coil_little_endian_sha256=_array_sha256(checked_parameters),
        seed_root_little_endian_sha256=_array_sha256(checked_seed),
        root_little_endian_sha256=_array_sha256(root),
        newton_iterations=int(iterations),
        raw_equalities=_readonly_array(equalities),
        raw_equalities_little_endian_sha256=_array_sha256(equalities),
        residual_l2=residual_l2,
        residual_infinity_norm=residual_inf,
        scaled_boozer_infinity_norm=scaled_boozer_inf,
    )


def _replay_dyadic_path(
    runtime: NativeSingleStageEndpointRuntime,
    final_parameters: np.ndarray,
    segment_count: int,
) -> NativeDyadicPathEvidence:
    initial_parameters = runtime.bootstrap_state[:_COIL_DOF_COUNT]
    roots = np.empty((segment_count + 1, _SURFACE_DOF_COUNT + 2), dtype=np.float64)
    roots[0] = runtime.bootstrap_root
    runtime._prepared.objective.x = initial_parameters
    initial_masked, initial_equalities, _volume = _raw_equalities(
        runtime,
        runtime.bootstrap_root,
    )
    initial_scaled_boozer_inf = float(
        np.linalg.norm(initial_masked, ord=np.inf) / np.sqrt(_BOOZER_EQUALITY_COUNT)
    )
    if initial_scaled_boozer_inf > SCALED_BOOZER_TOLERANCE:
        raise NativeEndpointError("native bootstrap scaled Boozer gate failed")
    initial_step = NativeContinuationStep(
        segment_count=segment_count,
        index=0,
        predecessor_index=None,
        coil_little_endian_sha256=_array_sha256(initial_parameters),
        seed_root_little_endian_sha256=_array_sha256(runtime.bootstrap_root),
        root_little_endian_sha256=_array_sha256(runtime.bootstrap_root),
        newton_iterations=0,
        raw_equalities=_readonly_array(initial_equalities),
        raw_equalities_little_endian_sha256=_array_sha256(initial_equalities),
        residual_l2=float(np.linalg.norm(initial_equalities)),
        residual_infinity_norm=float(np.linalg.norm(initial_equalities, ord=np.inf)),
        scaled_boozer_infinity_norm=initial_scaled_boozer_inf,
    )
    steps = [initial_step]
    predecessor = np.array(runtime.bootstrap_root, copy=True)
    for index in range(1, segment_count + 1):
        parameters = _path_parameters(
            initial_parameters,
            final_parameters,
            segment_count,
            index,
        )
        root, step = _solve_from_seed(
            runtime,
            parameters,
            predecessor,
            segment_count=segment_count,
            index=index,
            predecessor_index=index - 1,
        )
        roots[index] = root
        steps.append(step)
        predecessor = np.array(root, copy=True)
    return NativeDyadicPathEvidence(
        segment_count=segment_count,
        roots=_readonly_array(roots),
        steps=tuple(steps),
    )


def _sealed_observables_match(
    endpoint: NativeExplicitStateEvaluation,
    sealed: SealedNativeObservables,
) -> bool:
    observed = np.asarray(
        (
            endpoint.objective,
            endpoint.observables.iota,
            endpoint.observables.volume,
            endpoint.objective_terms.non_qs,
            endpoint.objective_terms.residual,
            endpoint.observables.boozer_residual_rms,
            endpoint.objective_terms.major_radius,
            endpoint.objective_terms.length,
        ),
        dtype=np.float64,
    )
    expected = np.asarray(
        (
            sealed.objective,
            sealed.iota,
            sealed.volume,
            sealed.non_qs,
            sealed.boozer_residual_value,
            sealed.boozer_residual_rms,
            sealed.major_radius_penalty,
            sealed.length_penalty,
        ),
        dtype=np.float64,
    )
    return bool(
        np.allclose(
            observed,
            expected,
            rtol=SEALED_OBSERVABLE_RTOL,
            atol=SEALED_OBSERVABLE_ATOL,
        )
    )


__all__ = (
    "BRANCH_ROOT_TOLERANCE",
    "COARSE_SEGMENT_COUNT",
    "EXACT_NEWTON_MAXIMUM_ITERATIONS",
    "EXACT_NEWTON_TOLERANCE",
    "REFINED_SEGMENT_COUNT",
    "SCHEMA_VERSION",
    "SSOT_SHA256",
    "HistoricalNativeObservablePaths",
    "HistoricalNativeParameterMetadata",
    "HistoricalNativeParameters",
    "NativeAcceptedIntervalEvidence",
    "NativeAcceptedPathEvidence",
    "NativeContinuationStep",
    "NativeDyadicPathEvidence",
    "NativeEndpointError",
    "NativeExplicitStateEvaluation",
    "NativeFullSpaceObjectiveContract",
    "NativeObjectiveTerms",
    "NativeReducedEvaluation",
    "NativeReferenceEvidence",
    "NativeSingleStageEndpointRuntime",
    "NativeStateObservables",
    "SealedNativeObservables",
    "build_native_single_stage_endpoint_runtime",
    "load_historical_native_parameters",
)
