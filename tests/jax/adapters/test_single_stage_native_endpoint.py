from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import simsopt_jax_adapters.geo.single_stage_native_endpoint as native_module
from examples.jax.parity.cases.native_single_stage_boozer_vacuum import SPEC
from examples.jax.parity.input_bundle import InputBundle
from simsopt_jax.objectives.single_stage_fullspace import (
    FROZEN_LAYOUT,
    FullSpaceObjectiveConfig,
    FullSpaceProblem,
    FullSpaceState,
)
from simsopt_jax_adapters.geo.single_stage_fullspace import (
    Float64Fingerprint,
    SingleStageFullSpaceBootstrap,
)
from simsopt_jax_adapters.geo.single_stage_native_endpoint import (
    BRANCH_ROOT_TOLERANCE,
    COARSE_SEGMENT_COUNT,
    EXACT_NEWTON_MAXIMUM_ITERATIONS,
    EXACT_NEWTON_TOLERANCE,
    REFINED_SEGMENT_COUNT,
    SSOT_SHA256,
    HistoricalNativeObservablePaths,
    HistoricalNativeParameterMetadata,
    NativeContinuationStep,
    NativeEndpointError,
    NativeSingleStageEndpointRuntime,
    build_native_single_stage_endpoint_runtime,
    load_historical_native_parameters,
)

_COIL_COUNT = 461
_SURFACE_COUNT = 253
_ROOT_COUNT = 255
_FULL_RESIDUAL_COUNT = 507
_FIRST_CURRENT = 7.0
_VOLUME_TARGET = 2.0
_IOTA_TARGET = -0.4
_G_INITIAL = 12.0


class _FakeCurrent:
    def __init__(self, value: float) -> None:
        self.value = value

    def get_value(self) -> float:
        return self.value


@dataclass
class _FakeCoil:
    current: _FakeCurrent


@dataclass
class _FakeBiotSavart:
    coils: tuple[_FakeCoil, ...]


class _FakeSurface:
    def __init__(self, dofs: np.ndarray) -> None:
        self._dofs = np.array(dofs, dtype=np.float64, copy=True)

    def get_dofs(self) -> np.ndarray:
        return np.array(self._dofs, copy=True)

    def set_dofs(self, dofs: np.ndarray) -> None:
        self._dofs = np.array(dofs, dtype=np.float64, copy=True)

    def major_radius(self) -> float:
        return 1.5


class _FakeObjective:
    def __init__(self, parameters: np.ndarray) -> None:
        self._x = np.array(parameters, dtype=np.float64, copy=True)
        self.solver: _FakeSolver | None = None

    @property
    def x(self) -> np.ndarray:
        return np.array(self._x, copy=True)

    @x.setter
    def x(self, parameters: np.ndarray) -> None:
        self._x = np.array(parameters, dtype=np.float64, copy=True)
        if self.solver is not None:
            self.solver.need_to_run_code = True

    def J(self) -> float:
        return float(1.0e-12 * np.vdot(self._x, self._x))

    def dJ(self) -> np.ndarray:
        return 2.0e-12 * self._x


class _FakeNonQs:
    def __init__(self, objective: _FakeObjective) -> None:
        self.objective = objective
        self.axis = 0

    def fixed_surface_value_and_derivative(self) -> tuple[float, object]:
        return self.objective.J(), object()


class _FakeResidual(_FakeObjective):
    def fixed_surface_value_derivative_and_y_partial(
        self,
        iota: float,
        G: float,
        *,
        weight_inv_modB: bool,
    ) -> tuple[float, object, np.ndarray]:
        assert weight_inv_modB is False
        assert np.isfinite(iota)
        assert np.isfinite(G)
        return 0.0, object(), np.zeros(2, dtype=np.float64)


class _FakeScalarObjective:
    def __init__(self, value: float) -> None:
        self.value = value
        self.x = np.empty(0, dtype=np.float64)

    def J(self) -> float:
        return self.value

    def dJ(self) -> np.ndarray:
        return np.empty(0, dtype=np.float64)


@dataclass
class _FakePenalty:
    obj: _FakeScalarObjective

    def J(self) -> float:
        return 0.0


@dataclass(frozen=True)
class _FakeAnchor:
    iota: float = _IOTA_TARGET
    G: float = _G_INITIAL
    iota_target: float = _IOTA_TARGET
    volume_target: float = _VOLUME_TARGET
    major_radius_target: float = 1.5
    total_length_target: float = 6.0
    inner_solver_success: bool = True


@dataclass(frozen=True)
class _FakeCandidate:
    objective: float
    gradient: np.ndarray
    inner_solver_success: bool
    solver_residual_l2: float
    solver_residual_inf: float


@dataclass(frozen=True)
class _SolveCall:
    parameters: np.ndarray
    seed_root: np.ndarray
    output_root: np.ndarray
    tolerance: float
    maximum_iterations: int


class _FakeSolver:
    def __init__(
        self,
        objective: _FakeObjective,
        initial_surface: np.ndarray,
        initial_parameters: np.ndarray,
    ) -> None:
        self.objective = objective
        objective.solver = self
        self.surface = _FakeSurface(initial_surface)
        self.biotsavart = _FakeBiotSavart(
            coils=(_FakeCoil(_FakeCurrent(_FIRST_CURRENT)),)
        )
        self.initial_parameters = np.array(initial_parameters, copy=True)
        self.mask = np.zeros(_FULL_RESIDUAL_COUNT, dtype=np.bool_)
        self.mask[:254] = True
        self.res: dict[str, object] = {
            "success": True,
            "residual": np.zeros(_FULL_RESIDUAL_COUNT, dtype=np.float64),
            "mask": np.array(self.mask, copy=True),
            "iter": 0,
            "iota": _IOTA_TARGET,
            "G": _G_INITIAL,
        }
        self.need_to_run_code = False
        self.calls: list[_SolveCall] = []
        self.fail_call: int | None = None
        self.diverge_call: int | None = None
        self.drift_current_call: int | None = None
        self.wrong_mask_call: int | None = None
        self.success_override: object | None = None
        self.iteration_override: object | None = None

    def expected_root(self, parameters: np.ndarray) -> np.ndarray:
        delta = float(parameters[0] - self.initial_parameters[0])
        surface = np.linspace(-1.0, 1.0, _SURFACE_COUNT, dtype=np.float64)
        surface = surface + 1.0e-4 * delta
        return np.concatenate(
            (
                surface,
                np.asarray(
                    (_IOTA_TARGET + 1.0e-3 * delta, _G_INITIAL + 1.0e-2 * delta)
                ),
            )
        )

    def solve_residual_equation_exactly_newton(
        self,
        *,
        tol: float,
        maxiter: int,
        iota: float,
        G: float,
        verbose: bool,
    ) -> dict[str, object]:
        assert verbose is False
        call_number = len(self.calls) + 1
        seed = np.concatenate((self.surface.get_dofs(), np.asarray((iota, G))))
        output = self.expected_root(self.objective.x)
        if self.diverge_call == call_number:
            output = output + 1.0e-3
        self.surface.set_dofs(output[:_SURFACE_COUNT])
        if self.drift_current_call == call_number:
            self.biotsavart.coils[0].current.value += 1.0
        mask = np.array(self.mask, copy=True)
        if self.wrong_mask_call == call_number:
            mask[253] = False
        success: object = self.fail_call != call_number
        if self.success_override is not None:
            success = self.success_override
        iterations: object = 1
        if self.iteration_override is not None:
            iterations = self.iteration_override
        self.res = {
            "success": success,
            "residual": np.zeros(_FULL_RESIDUAL_COUNT, dtype=np.float64),
            "mask": mask,
            "iter": iterations,
            "iota": output[-2],
            "G": output[-1],
            "weight_inv_modB": False,
        }
        self.need_to_run_code = False
        self.calls.append(
            _SolveCall(
                parameters=self.objective.x,
                seed_root=seed,
                output_root=np.array(output, copy=True),
                tolerance=tol,
                maximum_iterations=maxiter,
            )
        )
        return self.res


class _FakeVolume(_FakeScalarObjective):
    pass


class _FakePrepared:
    def __init__(self, initial_parameters: np.ndarray, surface: np.ndarray) -> None:
        self.objective = _FakeObjective(initial_parameters)
        self.solver = _FakeSolver(self.objective, surface, initial_parameters)
        self.non_qs = _FakeNonQs(self.objective)
        self.residual = _FakeResidual(initial_parameters)
        self.residual._x = self.objective._x
        self.residual.solver = self.solver
        self.volume = _FakeVolume(_VOLUME_TARGET)
        self.radius_penalty = _FakePenalty(_FakeScalarObjective(1.5))
        self.length_penalty = _FakePenalty(_FakeScalarObjective(6.0))
        self.initial_parameters = np.array(initial_parameters, copy=True)
        self.initial_solution_success = True
        self.initial_iota = _IOTA_TARGET
        self.initial_volume = _VOLUME_TARGET
        self.baseline_anchor = _FakeAnchor()

    def evaluate_candidate(self, parameters: np.ndarray) -> _FakeCandidate:
        self.objective.x = parameters
        gradient = self.objective.dJ()
        return _FakeCandidate(
            objective=self.objective.J(),
            gradient=gradient,
            inner_solver_success=True,
            solver_residual_l2=0.0,
            solver_residual_inf=0.0,
        )


def _initial_parameters() -> np.ndarray:
    return np.linspace(-2.0, 2.0, _COIL_COUNT, dtype=np.float64)


def _initial_root() -> np.ndarray:
    return np.concatenate(
        (
            np.linspace(-1.0, 1.0, _SURFACE_COUNT, dtype=np.float64),
            np.asarray((_IOTA_TARGET, _G_INITIAL), dtype=np.float64),
        )
    )


def _bootstrap() -> SingleStageFullSpaceBootstrap:
    parameters = _initial_parameters()
    root = _initial_root()
    z0 = FROZEN_LAYOUT.pack(
        FullSpaceState(
            coil_dofs=jnp.asarray(parameters),
            surface_dofs=jnp.asarray(root[:_SURFACE_COUNT]),
            iota=jnp.asarray(root[-2]),
            G=jnp.asarray(root[-1]),
        )
    )
    config = FullSpaceObjectiveConfig(
        iota_target=jnp.asarray(_IOTA_TARGET, dtype=jnp.float64),
        major_radius_target=jnp.asarray(1.5, dtype=jnp.float64),
        length_target=jnp.asarray(6.0, dtype=jnp.float64),
        volume_target=jnp.asarray(_VOLUME_TARGET, dtype=jnp.float64),
        non_qs_weight=jnp.asarray(1.0, dtype=jnp.float64),
        residual_weight=jnp.asarray(1.0, dtype=jnp.float64),
        iota_weight=jnp.asarray(1.0, dtype=jnp.float64),
        major_radius_weight=jnp.asarray(1.0, dtype=jnp.float64),
        length_weight=jnp.asarray(1.0, dtype=jnp.float64),
        non_qs_axis=0,
        weight_inv_modB=False,
        length_coil_indices=(0, 1, 2),
    )
    problem = cast(
        "FullSpaceProblem",
        SimpleNamespace(
            config=config,
            exact_mask_indices=jnp.arange(254, dtype=jnp.int32),
        ),
    )
    return SingleStageFullSpaceBootstrap(
        problem=problem,
        z0=z0,
        targets=(),
        initial_boozer_residual_norm=0.0,
        first_base_current=Float64Fingerprint(
            name="first_base_current",
            value=_FIRST_CURRENT,
            hexadecimal=float(_FIRST_CURRENT).hex(),
            little_endian_sha256="test",
        ),
    )


def _bundle() -> InputBundle:
    return InputBundle(
        schema_version=2,
        case_id=SPEC.case_id,
        scale="native_default",
        random_seed=1,
        configuration={},
        configuration_fingerprint="configuration",
        arrays={},
        input_fingerprint="input",
    )


def _replace_bootstrap_problem(
    bootstrap: SingleStageFullSpaceBootstrap,
    *,
    config: FullSpaceObjectiveConfig | None = None,
    exact_mask_indices: jax.Array | None = None,
) -> SingleStageFullSpaceBootstrap:
    problem = cast(
        "FullSpaceProblem",
        SimpleNamespace(
            config=bootstrap.problem.config if config is None else config,
            exact_mask_indices=(
                bootstrap.problem.exact_mask_indices
                if exact_mask_indices is None
                else exact_mask_indices
            ),
        ),
    )
    return replace(bootstrap, problem=problem)


@pytest.fixture
def runtime_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[NativeSingleStageEndpointRuntime, _FakePrepared]:
    prepared = _FakePrepared(_initial_parameters(), _initial_root()[:_SURFACE_COUNT])

    def fake_prepare(
        bundle: InputBundle,
        arrays: dict[str, np.ndarray],
        spec: object,
    ) -> _FakePrepared:
        assert bundle.case_id == SPEC.case_id
        assert arrays == {}
        assert spec is SPEC
        return prepared

    monkeypatch.setattr(native_module, "_prepare_native_variant_runtime", fake_prepare)
    monkeypatch.setattr(
        native_module,
        "boozer_surface_residual",
        lambda *_args, **_kwargs: (np.zeros(_FULL_RESIDUAL_COUNT, dtype=np.float64),),
    )
    runtime = build_native_single_stage_endpoint_runtime(
        _bundle(),
        {},
        _bootstrap(),
    )
    return runtime, prepared


def _array_sha256(values: np.ndarray) -> str:
    canonical = np.ascontiguousarray(values, dtype=np.dtype("<f8"))
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _assert_raw_step_evidence(step: NativeContinuationStep) -> None:
    assert step.raw_equalities.shape == (254 + 1,)
    assert step.raw_equalities.dtype == np.float64
    assert step.raw_equalities.flags.writeable is False
    assert np.all(np.isfinite(step.raw_equalities))
    assert step.raw_equalities_little_endian_sha256 == _array_sha256(
        step.raw_equalities
    )
    assert step.residual_l2 == np.linalg.norm(step.raw_equalities)
    assert step.residual_infinity_norm == np.linalg.norm(
        step.raw_equalities,
        ord=np.inf,
    )
    assert step.scaled_boozer_infinity_norm == (
        np.linalg.norm(step.raw_equalities[:254], ord=np.inf) / np.sqrt(254.0)
    )


def _coil_sensitive_boozer_residual(
    prepared: _FakePrepared,
):
    def evaluate(
        surface: _FakeSurface,
        iota: float,
        G: float,
        _biotsavart: object,
        *,
        derivatives: int,
        weight_inv_modB: bool,
    ) -> tuple[np.ndarray]:
        assert derivatives == 0
        assert weight_inv_modB is False
        expected = prepared.solver.expected_root(prepared.objective.x)
        observed = np.concatenate(
            (surface.get_dofs(), np.asarray((iota, G), dtype=np.float64))
        )
        residual = np.zeros(_FULL_RESIDUAL_COUNT, dtype=np.float64)
        residual[0] = np.linalg.norm(observed - expected, ord=np.inf)
        return (residual,)

    return evaluate


def _historical_payload(
    runtime: NativeSingleStageEndpointRuntime,
    prepared: _FakePrepared,
) -> tuple[bytes, HistoricalNativeParameterMetadata]:
    final_parameters = _initial_parameters() + 1.0
    final_root = prepared.solver.expected_root(final_parameters)
    endpoint = runtime.evaluate_state(
        np.concatenate((final_parameters, final_root)).astype(np.float64)
    )
    document = {
        "endpoint": {
            "parameters": final_parameters.tolist(),
            "objective": endpoint.objective,
            "iota": endpoint.observables.iota,
            "volume": endpoint.observables.volume,
            "non_qs": endpoint.objective_terms.non_qs,
            "boozer_residual_value": endpoint.objective_terms.residual,
            "boozer_residual_rms": endpoint.observables.boozer_residual_rms,
            "major_radius_penalty": endpoint.objective_terms.major_radius,
            "length_penalty": endpoint.objective_terms.length,
        }
    }
    source = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    paths = HistoricalNativeObservablePaths(
        objective=("endpoint", "objective"),
        iota=("endpoint", "iota"),
        volume=("endpoint", "volume"),
        non_qs=("endpoint", "non_qs"),
        boozer_residual_value=("endpoint", "boozer_residual_value"),
        boozer_residual_rms=("endpoint", "boozer_residual_rms"),
        major_radius_penalty=("endpoint", "major_radius_penalty"),
        length_penalty=("endpoint", "length_penalty"),
    )
    metadata = HistoricalNativeParameterMetadata(
        source_sha256=hashlib.sha256(source).hexdigest(),
        parameter_path=("endpoint", "parameters"),
        parameter_little_endian_sha256=_array_sha256(final_parameters),
        parameter_dtype="<f8",
        parameter_shape=(_COIL_COUNT,),
        observable_paths=paths,
    )
    return source, metadata


def test_historical_input_is_byte_bound_fp64_and_immutable(
    runtime_setup: tuple[NativeSingleStageEndpointRuntime, _FakePrepared],
) -> None:
    runtime, prepared = runtime_setup
    source, metadata = _historical_payload(runtime, prepared)

    historical = load_historical_native_parameters(source, metadata)

    assert historical.parameters.shape == (_COIL_COUNT,)
    assert historical.parameters.dtype == np.float64
    assert historical.parameters.flags.writeable is False
    assert historical.source_sha256 == hashlib.sha256(source).hexdigest()
    assert historical.parameter_little_endian_sha256 == _array_sha256(
        historical.parameters
    )


def test_historical_input_rejects_source_parameter_dtype_and_shape_drift(
    runtime_setup: tuple[NativeSingleStageEndpointRuntime, _FakePrepared],
) -> None:
    runtime, prepared = runtime_setup
    source, metadata = _historical_payload(runtime, prepared)

    with pytest.raises(NativeEndpointError, match="source SHA-256"):
        load_historical_native_parameters(source + b" ", metadata)
    with pytest.raises(NativeEndpointError, match="parameter SHA-256"):
        load_historical_native_parameters(
            source,
            replace(metadata, parameter_little_endian_sha256="0" * 64),
        )
    with pytest.raises(TypeError, match="dtype"):
        load_historical_native_parameters(
            source,
            replace(metadata, parameter_dtype="<f4"),
        )
    with pytest.raises(ValueError, match="shape"):
        load_historical_native_parameters(
            source,
            replace(metadata, parameter_shape=(460,)),
        )


def test_explicit_state_preserves_frozen_order_and_never_solves(
    runtime_setup: tuple[NativeSingleStageEndpointRuntime, _FakePrepared],
) -> None:
    runtime, prepared = runtime_setup
    state = np.concatenate((_initial_parameters() + 0.25, _initial_root()))
    original_parameters = prepared.objective.x
    original_surface = prepared.solver.surface.get_dofs()

    evaluation = runtime.evaluate_state(state)
    changed_state = np.concatenate((_initial_parameters() + 0.75, _initial_root()))
    changed_evaluation = runtime.evaluate_state(changed_state)

    assert evaluation.state.shape == (461 + 253 + 1 + 1,)
    assert evaluation.masked_boozer_equalities.shape == (254,)
    assert evaluation.raw_equalities.shape == (254 + 1,)
    assert evaluation.raw_equalities[-1] == evaluation.volume_equality
    assert evaluation.observables.fixed_first_base_current == _FIRST_CURRENT
    assert evaluation.all_finite
    assert changed_evaluation.objective != evaluation.objective
    assert changed_evaluation.objective == pytest.approx(
        1.0e-12 * np.vdot(changed_state[:_COIL_COUNT], changed_state[:_COIL_COUNT])
    )
    assert runtime.objective_contract.weight_inv_modB is False
    assert runtime.objective_contract.non_qs_axis == 0
    assert runtime.exact_mask_indices.flags.writeable is False
    assert prepared.solver.calls == []
    np.testing.assert_array_equal(prepared.objective.x, original_parameters)
    np.testing.assert_array_equal(prepared.solver.surface.get_dofs(), original_surface)

    reduced = runtime.evaluate_reduced(_initial_parameters() + 0.5)
    assert reduced.gradient.shape == (_COIL_COUNT,)
    assert reduced.gradient.dtype == np.float64
    assert reduced.inner_solver_success
    assert reduced.all_finite


def test_runtime_rejects_fullspace_contract_and_native_mask_order_drift(
    runtime_setup: tuple[NativeSingleStageEndpointRuntime, _FakePrepared],
) -> None:
    _runtime, prepared = runtime_setup
    bootstrap = _bootstrap()
    config = bootstrap.problem.config

    with pytest.raises(NativeEndpointError, match="objective weights"):
        build_native_single_stage_endpoint_runtime(
            _bundle(),
            {},
            _replace_bootstrap_problem(
                bootstrap,
                config=replace(
                    config,
                    residual_weight=jnp.asarray(2.0, dtype=jnp.float64),
                ),
            ),
        )
    with pytest.raises(NativeEndpointError, match="objective targets"):
        build_native_single_stage_endpoint_runtime(
            _bundle(),
            {},
            _replace_bootstrap_problem(
                bootstrap,
                config=replace(
                    config,
                    iota_target=jnp.asarray(_IOTA_TARGET + 1.0e-6, dtype=jnp.float64),
                ),
            ),
        )
    with pytest.raises(NativeEndpointError, match="non-QS axes"):
        build_native_single_stage_endpoint_runtime(
            _bundle(),
            {},
            _replace_bootstrap_problem(
                bootstrap,
                config=replace(config, non_qs_axis=1),
            ),
        )
    with pytest.raises(NativeEndpointError, match="weight_inv_modB=False"):
        build_native_single_stage_endpoint_runtime(
            _bundle(),
            {},
            _replace_bootstrap_problem(
                bootstrap,
                config=replace(config, weight_inv_modB=True),
            ),
        )

    changed_mask = np.array(prepared.solver.mask, copy=True)
    changed_mask[253] = False
    changed_mask[254] = True
    prepared.solver.res["mask"] = changed_mask
    with pytest.raises(NativeEndpointError, match="exact mask/order"):
        build_native_single_stage_endpoint_runtime(_bundle(), {}, bootstrap)


def test_runtime_accepts_independent_root_roundoff_with_exact_conventions(
    runtime_setup: tuple[NativeSingleStageEndpointRuntime, _FakePrepared],
) -> None:
    _runtime, prepared = runtime_setup
    native_surface = prepared.solver.surface.get_dofs()
    native_surface[0] = np.nextafter(native_surface[0], np.inf)
    prepared.solver.surface.set_dofs(native_surface)
    native_iota = float(np.nextafter(np.float64(_IOTA_TARGET), -np.inf))
    prepared.baseline_anchor = replace(
        prepared.baseline_anchor,
        iota=native_iota,
        iota_target=native_iota,
    )

    rebuilt = build_native_single_stage_endpoint_runtime(_bundle(), {}, _bootstrap())

    assert rebuilt.objective_contract.iota_target == _IOTA_TARGET
    assert rebuilt.bootstrap_root[-2] == _IOTA_TARGET


@pytest.mark.parametrize(
    "bad_state",
    (
        np.zeros(715, dtype=np.float64),
        np.zeros(716, dtype=np.float32),
        np.full(716, np.nan, dtype=np.float64),
    ),
)
def test_explicit_state_rejects_shape_fp64_and_nonfinite_drift(
    runtime_setup: tuple[NativeSingleStageEndpointRuntime, _FakePrepared],
    bad_state: np.ndarray,
) -> None:
    runtime, _prepared = runtime_setup

    with pytest.raises((TypeError, ValueError)):
        runtime.evaluate_state(bad_state)


def test_reference_runs_independent_256_and_512_predecessor_only_paths(
    runtime_setup: tuple[NativeSingleStageEndpointRuntime, _FakePrepared],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, prepared = runtime_setup
    source, metadata = _historical_payload(runtime, prepared)
    historical = load_historical_native_parameters(source, metadata)
    monkeypatch.setattr(
        native_module,
        "boozer_surface_residual",
        _coil_sensitive_boozer_residual(prepared),
    )

    reference = runtime.reconstruct_native_reference(historical)

    assert reference.coarse_path.roots.shape == (COARSE_SEGMENT_COUNT + 1, 255)
    assert reference.refined_path.roots.shape == (REFINED_SEGMENT_COUNT + 1, 255)
    assert reference.ssot_sha256 == (
        "d082baa587b9db580ac3ef8c99a3123ed83564586b605200f7c2cfa6feb909a9"
    )
    assert reference.ssot_sha256 == SSOT_SHA256
    assert reference.common_knot_root_infinity_difference <= BRANCH_ROOT_TOLERANCE
    assert reference.coarse_path.roots.flags.writeable is False
    assert reference.sealed_observables_match
    assert reference.usable
    for step in reference.coarse_path.steps + reference.refined_path.steps:
        _assert_raw_step_evidence(step)
    assert len(prepared.solver.calls) == COARSE_SEGMENT_COUNT + REFINED_SEGMENT_COUNT
    coarse_calls = prepared.solver.calls[:COARSE_SEGMENT_COUNT]
    refined_calls = prepared.solver.calls[COARSE_SEGMENT_COUNT:]
    for calls in (coarse_calls, refined_calls):
        np.testing.assert_array_equal(calls[0].seed_root, _initial_root())
        for previous, current in zip(calls, calls[1:]):
            np.testing.assert_array_equal(current.seed_root, previous.output_root)
        assert all(call.tolerance == EXACT_NEWTON_TOLERANCE for call in calls)
        assert all(
            call.maximum_iterations == EXACT_NEWTON_MAXIMUM_ITERATIONS for call in calls
        )
    np.testing.assert_array_equal(
        reference.state[:_COIL_COUNT],
        historical.parameters,
    )
    assert prepared.solver.biotsavart.coils[0].current.get_value() == _FIRST_CURRENT


def test_reference_rejects_bootstrap_residual_and_sealed_observable_drift(
    runtime_setup: tuple[NativeSingleStageEndpointRuntime, _FakePrepared],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, prepared = runtime_setup
    source, metadata = _historical_payload(runtime, prepared)
    historical = load_historical_native_parameters(source, metadata)
    bad_sealed = replace(
        historical,
        sealed_observables=replace(
            historical.sealed_observables,
            objective=historical.sealed_observables.objective + 1.0e-8,
        ),
    )

    with pytest.raises(NativeEndpointError, match="sealed observables"):
        runtime.reconstruct_native_reference(bad_sealed)

    monkeypatch.setattr(
        native_module,
        "boozer_surface_residual",
        lambda *_args, **_kwargs: (np.ones(_FULL_RESIDUAL_COUNT, dtype=np.float64),),
    )
    with pytest.raises(NativeEndpointError, match="bootstrap scaled Boozer"):
        runtime.reconstruct_native_reference(historical)


@pytest.mark.parametrize(
    ("failure_field", "failure_value", "match"),
    (
        ("fail_call", 1, "solve failed"),
        ("wrong_mask_call", 1, "mask must select 254"),
        ("drift_current_call", 1, "fixed first base current changed"),
        ("diverge_call", COARSE_SEGMENT_COUNT + 2, "common-knot roots"),
        ("success_override", 1, "success flag must be bool"),
        ("iteration_override", -1, "iterations must be in"),
        ("iteration_override", 21, "iterations must be in"),
    ),
)
def test_reference_fails_closed_on_solve_order_current_and_branch_drift(
    runtime_setup: tuple[NativeSingleStageEndpointRuntime, _FakePrepared],
    failure_field: str,
    failure_value: int,
    match: str,
) -> None:
    runtime, prepared = runtime_setup
    source, metadata = _historical_payload(runtime, prepared)
    historical = load_historical_native_parameters(source, metadata)
    setattr(prepared.solver, failure_field, failure_value)

    with pytest.raises(NativeEndpointError, match=match):
        runtime.reconstruct_native_reference(historical)


def test_accepted_interval_requires_direct_midpoint_and_supplied_root_agreement(
    runtime_setup: tuple[NativeSingleStageEndpointRuntime, _FakePrepared],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, prepared = runtime_setup
    middle_parameters = _initial_parameters() + 0.25
    final_parameters = _initial_parameters() + 0.5
    middle_root = prepared.solver.expected_root(middle_parameters)
    final_root = prepared.solver.expected_root(final_parameters)
    states = np.stack(
        (
            np.asarray(runtime.bootstrap_state),
            np.concatenate((middle_parameters, middle_root)),
            np.concatenate((final_parameters, final_root)),
        )
    )
    prepared.objective.x = final_parameters
    monkeypatch.setattr(
        native_module,
        "boozer_surface_residual",
        _coil_sensitive_boozer_residual(prepared),
    )

    evidence = runtime.audit_accepted_states(states)

    assert evidence.usable
    assert evidence.first_failing_index is None
    assert evidence.failure_reason is None
    assert evidence.bootstrap_step.index == 0
    assert len(evidence.intervals) == 2
    assert all(
        interval.direct_refined_infinity_difference == 0.0
        for interval in evidence.intervals
    )
    assert all(
        interval.supplied_refined_infinity_difference == 0.0
        for interval in evidence.intervals
    )
    assert len(prepared.solver.calls) == 6
    first_authoritative_refined_root = prepared.solver.calls[2].output_root
    np.testing.assert_array_equal(
        prepared.solver.calls[3].seed_root,
        first_authoritative_refined_root,
    )
    np.testing.assert_array_equal(
        prepared.solver.calls[4].seed_root,
        first_authoritative_refined_root,
    )
    assert all(
        interval.supplied_state_little_endian_sha256
        == _array_sha256(states[interval.index])
        for interval in evidence.intervals
    )
    assert evidence.intervals[1].direct_step.predecessor_index == 1
    assert evidence.intervals[1].midpoint_step.predecessor_index == 2
    assert evidence.intervals[1].refined_step.predecessor_index == 3
    for step in (evidence.bootstrap_step,) + tuple(
        step
        for interval in evidence.intervals
        for step in (
            interval.direct_step,
            interval.midpoint_step,
            interval.refined_step,
        )
    ):
        _assert_raw_step_evidence(step)
    for interval in evidence.intervals:
        assert interval.midpoint_root.shape == (_ROOT_COUNT,)
        assert interval.midpoint_root.dtype == np.float64
        assert interval.midpoint_root.flags.writeable is False
        assert np.all(np.isfinite(interval.midpoint_root))
        assert _array_sha256(interval.midpoint_root) == (
            interval.midpoint_step.root_little_endian_sha256
        )

    mismatched = np.array(states, copy=True)
    mismatched[2, -1] += 2.0 * BRANCH_ROOT_TOLERANCE
    mismatched_evidence = runtime.audit_accepted_states(mismatched)
    assert not mismatched_evidence.usable
    assert mismatched_evidence.first_failing_index == 2
    assert mismatched_evidence.failure_reason is not None
    assert "branch mismatch" in mismatched_evidence.failure_reason

    prepared.solver.fail_call = len(prepared.solver.calls) + 1
    solve_failure = runtime.audit_accepted_states(states)
    assert not solve_failure.usable
    assert solve_failure.first_failing_index == 1
    assert solve_failure.failure_reason is not None
    assert "solve failed" in solve_failure.failure_reason

    prepared.solver.fail_call = None
    monkeypatch.setattr(
        native_module,
        "boozer_surface_residual",
        lambda *_args, **_kwargs: (np.ones(_FULL_RESIDUAL_COUNT, dtype=np.float64),),
    )
    bootstrap_failure = runtime.audit_accepted_states(states)
    assert not bootstrap_failure.usable
    assert bootstrap_failure.first_failing_index == 0
    assert bootstrap_failure.failure_reason == (
        "accepted bootstrap scaled Boozer gate failed"
    )


def test_accepted_interval_rejects_direct_refined_native_root_mismatch(
    runtime_setup: tuple[NativeSingleStageEndpointRuntime, _FakePrepared],
) -> None:
    runtime, prepared = runtime_setup
    final_parameters = _initial_parameters() + 0.5
    final_root = prepared.solver.expected_root(final_parameters)
    states = np.stack(
        (
            np.asarray(runtime.bootstrap_state),
            np.concatenate((final_parameters, final_root)),
        )
    )
    prepared.solver.diverge_call = 1

    evidence = runtime.audit_accepted_states(states)

    assert not evidence.usable
    assert evidence.first_failing_index == 1
    assert evidence.failure_reason is not None
    assert "branch mismatch" in evidence.failure_reason
    assert len(evidence.intervals) == 1
    assert (
        evidence.intervals[0].direct_refined_infinity_difference > BRANCH_ROOT_TOLERANCE
    )
