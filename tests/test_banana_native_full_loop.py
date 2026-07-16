from __future__ import annotations

import ast
from hashlib import sha256
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Iterator

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
DRIVER_PATH = (
    REPO_ROOT
    / "examples"
    / "single_stage_optimization"
    / "banana_opt"
    / "native_banana_drivers.py"
)
CLI_PATH = (
    REPO_ROOT
    / "examples"
    / "single_stage_optimization"
    / "SINGLE_STAGE"
    / "single_stage_banana_native.py"
)
NATIVE_MODULE_NAME = (
    "examples.single_stage_optimization.banana_opt.native_banana_drivers"
)
BANANA_PACKAGE_NAME = "examples.single_stage_optimization.banana_opt"


class _StubOptimizable:
    def __init__(
        self,
        value: float = 0.0,
        gradient: np.ndarray | None = None,
        *,
        depends_on: list[_StubOptimizable] | None = None,
        label: str = "stub",
        **_kwargs: object,
    ) -> None:
        self.value = float(value)
        self.gradient = np.asarray(
            [0.0, 0.0] if gradient is None else gradient,
            dtype=np.float64,
        )
        self.parents = [] if depends_on is None else list(depends_on)
        self.label = label
        self.x = np.asarray([0.5, -1.5], dtype=np.float64)
        self.dof_names = ["fake:dof0", "fake:dof1"]

    def J(self) -> float:
        return self.value

    def dJ(self, *, partials: bool = False) -> np.ndarray:
        del partials
        return self.gradient.copy()

    def __mul__(self, factor: float) -> _StubExpression:
        return _StubExpression("scale", (self,), float(factor))

    def __rmul__(self, factor: float) -> _StubExpression:
        return self * factor

    def __add__(self, other: _StubOptimizable) -> _StubExpression:
        return _StubExpression("sum", (self, other), 1.0)


class _StubExpression(_StubOptimizable):
    def __init__(
        self,
        operation: str,
        operands: tuple[_StubOptimizable, ...],
        factor: float,
    ) -> None:
        super().__init__(depends_on=list(operands), label=operation)
        self.operation = operation
        self.operands = operands
        self.factor = factor

    def J(self) -> float:
        if self.operation == "scale":
            return self.factor * self.operands[0].J()
        return sum(operand.J() for operand in self.operands)

    def dJ(self, *, partials: bool = False) -> np.ndarray:
        del partials
        if self.operation == "scale":
            return self.factor * self.operands[0].dJ()
        return sum(
            (operand.dJ() for operand in self.operands),
            start=np.zeros(2, dtype=np.float64),
        )


class _StubSaveable:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def save(self, filename: str) -> None:
        Path(filename).write_bytes(self.payload)


class _StubSurface(_StubSaveable):
    def __init__(self) -> None:
        super().__init__(b"surface-artifact")
        self._dofs = np.asarray([9.0, 10.0], dtype=np.float64)

    def get_dofs(self) -> np.ndarray:
        return self._dofs.copy()

    def set_dofs(self, dofs: np.ndarray) -> None:
        self._dofs = np.asarray(dofs, dtype=np.float64).copy()

    def volume(self) -> float:
        return 0.875


class _StubBoozerSurface(_StubSaveable):
    def __init__(self) -> None:
        super().__init__(b"boozer-artifact")
        self.surface = _StubSurface()
        self.res: dict[str, float | bool] = {
            "success": True,
            "iota": 0.125,
            "G": -2.25,
        }
        self.need_to_run_code = False
        self.solve_warm_starts: list[tuple[float, float]] = []

    def run_code(self, iota: float, G: float) -> dict[str, float | bool]:
        self.solve_warm_starts.append((float(iota), float(G)))
        return self.res


def _identity_decorator(function):
    return function


def _module(name: str, **attributes: object) -> ModuleType:
    module = ModuleType(name)
    module.__dict__.update(attributes)
    return module


@pytest.fixture
def native_driver(monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    """Import the native driver against deterministic, non-scientific seams."""
    stub_class = type("StubScientificClass", (), {})
    simsopt = _module("simsopt")
    setattr(simsopt, "__path__", [])
    core = _module("simsopt._core")
    setattr(core, "__path__", [])
    derivative = _module("simsopt._core.derivative", derivative_dec=_identity_decorator)
    optimizable = _module(
        "simsopt._core.optimizable",
        Optimizable=_StubOptimizable,
        load=lambda _path: None,
    )
    field = _module("simsopt.field", BiotSavart=stub_class)
    geo = _module(
        "simsopt.geo",
        BoozerResidual=stub_class,
        BoozerSurface=stub_class,
        CurveCurveDistance=stub_class,
        CurveLength=stub_class,
        CurveSurfaceDistance=stub_class,
        Iotas=stub_class,
        LpCurveCurvature=stub_class,
        NonQuasiSymmetricRatio=stub_class,
        Surface=stub_class,
        SurfaceRZFourier=stub_class,
        SurfaceXYZFourier=stub_class,
        SurfaceXYZTensorFourier=stub_class,
        Volume=stub_class,
    )
    objectives = _module("simsopt.objectives", QuadraticPenalty=stub_class)

    previous_module = sys.modules.pop(NATIVE_MODULE_NAME, None)
    package = sys.modules.get(BANANA_PACKAGE_NAME)
    previous_attribute = (
        getattr(package, "native_banana_drivers", None) if package is not None else None
    )
    if package is not None and hasattr(package, "native_banana_drivers"):
        delattr(package, "native_banana_drivers")

    for name, module in {
        "simsopt": simsopt,
        "simsopt._core": core,
        "simsopt._core.derivative": derivative,
        "simsopt._core.optimizable": optimizable,
        "simsopt.field": field,
        "simsopt.geo": geo,
        "simsopt.objectives": objectives,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    from examples.single_stage_optimization.banana_opt import (
        native_banana_drivers,
    )

    try:
        yield native_banana_drivers
    finally:
        sys.modules.pop(NATIVE_MODULE_NAME, None)
        package = sys.modules.get(BANANA_PACKAGE_NAME)
        if package is not None and hasattr(package, "native_banana_drivers"):
            delattr(package, "native_banana_drivers")
        if previous_module is not None:
            sys.modules[NATIVE_MODULE_NAME] = previous_module
        if package is not None and previous_attribute is not None:
            setattr(package, "native_banana_drivers", previous_attribute)


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _array_sha256(values: np.ndarray) -> str:
    canonical = np.asarray(values, dtype="<f8").reshape(-1)
    return sha256(canonical.tobytes(order="C")).hexdigest()


def test_native_common_objective_executes_all_seven_weighted_terms(
    native_driver: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native = native_driver
    raw_values = {
        "non_quasisymmetric_ratio": 1.0,
        "boozer_residual": 2.0,
        "iota": 3.0,
        "length_max": 4.0,
        "coil_coil_distance": 5.0,
        "coil_surface_distance": 6.0,
        "curvature": 7.0,
    }

    def objective(label: str) -> _StubOptimizable:
        value = raw_values.get(label, -1.0)
        return _StubOptimizable(value, np.asarray([value, -value]), label=label)

    monkeypatch.setattr(
        native,
        "NonQuasiSymmetricRatio",
        lambda *_args: objective("non_quasisymmetric_ratio"),
    )
    monkeypatch.setattr(
        native,
        "BoozerResidual",
        lambda *_args: objective("boozer_residual"),
    )
    monkeypatch.setattr(native, "Iotas", lambda *_args: objective("iota_source"))
    monkeypatch.setattr(
        native,
        "CurveLength",
        lambda *_args: objective("length_source"),
    )

    def quadratic_penalty(source: _StubOptimizable, *_args: object) -> _StubOptimizable:
        label = "iota" if source.label == "iota_source" else "length_max"
        return objective(label)

    monkeypatch.setattr(native, "QuadraticPenalty", quadratic_penalty)
    monkeypatch.setattr(
        native,
        "CurveCurveDistance",
        lambda *_args: objective("coil_coil_distance"),
    )
    monkeypatch.setattr(
        native,
        "CurveSurfaceDistance",
        lambda *_args: objective("coil_surface_distance"),
    )
    monkeypatch.setattr(
        native,
        "LpCurveCurvature",
        lambda *_args: objective("curvature"),
    )

    weights = native.common_only_weights(
        nonqs=2.0,
        bres=3.0,
        iota=5.0,
        length=7.0,
        ccdist=11.0,
        csdist=13.0,
        curvature=17.0,
    )
    coils = [
        SimpleNamespace(curve=f"curve-{index}")
        for index in range(native.BANANA_IDX + native.N_BANANA)
    ]
    combined, terms, recorded_csd = native.build_native_single_stage_objective(
        boozer_surface=SimpleNamespace(surface=object()),
        biotsavart=SimpleNamespace(coils=coils),
        iota_target=0.125,
        weights=weights,
    )

    expected_names = (
        "non_quasisymmetric_ratio",
        "boozer_residual",
        "iota",
        "length_max",
        "coil_coil_distance",
        "coil_surface_distance",
        "curvature",
    )
    expected_weights = (2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0)
    expected_raw = tuple(raw_values[name] for name in expected_names)
    assert native.COMMON_OBJECTIVE_TERM_NAMES == expected_names
    assert tuple(term.name for term in terms) == expected_names
    assert tuple(term.weight for term in terms) == expected_weights
    assert tuple(term.objective.J() for term in terms) == expected_raw
    assert combined.J() == pytest.approx(
        sum(weight * raw for weight, raw in zip(expected_weights, expected_raw))
    )
    assert recorded_csd.last_value == raw_values["coil_surface_distance"]


def test_native_state_payload_serializes_values_hashes_and_ordered_terms(
    native_driver: ModuleType,
) -> None:
    native = native_driver
    names = native.COMMON_OBJECTIVE_TERM_NAMES
    raw_values = (1.25, 2.5, 3.75, 4.0, 5.25, 0.0, 7.5)
    weights = (2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0)
    terms = tuple(
        native.WeightedTerm(name, weight, _StubOptimizable(raw))
        for name, weight, raw in zip(names, weights, raw_values)
    )
    dofs = np.asarray([1.25, -2.5], dtype=np.float64)
    gradient = np.asarray([0.75, -1.5], dtype=np.float64)
    objective = sum(weight * raw for weight, raw in zip(weights, raw_values))
    state = native._AcceptedState(
        optimizer_dofs=dofs,
        surface_dofs=np.asarray([9.0, 10.0]),
        iota=0.125,
        G=-2.25,
        objective=objective,
        gradient=gradient,
    )

    payload = native._state_payload(state, terms=terms, volume=0.875)

    assert set(payload) == {
        "objective",
        "gradient_norm",
        "iota",
        "G",
        "volume",
        "dofs",
        "dof_count",
        "dofs_sha256",
        "gradient",
        "gradient_count",
        "gradient_sha256",
        "terms",
    }
    assert payload["objective"] == pytest.approx(objective)
    assert payload["gradient_norm"] == pytest.approx(np.linalg.norm(gradient))
    assert payload["iota"] == 0.125
    assert payload["G"] == -2.25
    assert payload["volume"] == 0.875
    assert payload["dofs"] == dofs.tolist()
    assert payload["dof_count"] == 2
    assert payload["dofs_sha256"] == _array_sha256(dofs)
    assert payload["gradient"] == gradient.tolist()
    assert payload["gradient_count"] == 2
    assert payload["gradient_sha256"] == _array_sha256(gradient)
    term_payload = payload["terms"]
    assert isinstance(term_payload, dict)
    assert tuple(term_payload) == names
    assert term_payload == {
        name: {"raw": raw, "weight": weight, "weighted": weight * raw}
        for name, weight, raw in zip(names, weights, raw_values)
    }


def test_native_run_emits_executable_full_loop_result_contract(
    native_driver: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    native = native_driver
    biotsavart_path = tmp_path / "biot_savart_seed.json"
    surface_path = tmp_path / "surface_seed.json"
    boozer_state_path = tmp_path / "boozer_state_seed.json"
    biotsavart_path.write_bytes(b"biotsavart-seed")
    surface_path.write_bytes(b"surface-seed")
    boozer_state_path.write_text('{"iota": 0.125, "G": -2.25}\n')

    weights = native.common_only_weights(
        nonqs=2.0,
        bres=3.0,
        iota=5.0,
        length=7.0,
        ccdist=11.0,
        csdist=13.0,
        curvature=17.0,
    )
    config = native.NativeSingleStageConfig(
        biotsavart_file=biotsavart_path,
        surface_path=surface_path,
        boozer_state_path=boozer_state_path,
        output_root=tmp_path / "run",
        overwrite=False,
        run_config_sha256="a" * 64,
        objective_profile="common-seven-term",
        vmec_s=1.0,
        surface_scale=None,
        mpol=10,
        ntor=10,
        nphi=65,
        ntheta=32,
        constraint_weight=1.0,
        iota_target=0.125,
        boozer_bfgs_tol=1.0e-10,
        boozer_bfgs_maxiter=50,
        boozer_newton_tol=1.0e-11,
        boozer_newton_maxiter=40,
        boozer_limited_memory=True,
        outer_maxiter=3,
        outer_maxcor=20,
        outer_maxls=10,
        outer_ftol=1.0e-12,
        outer_gtol=1.0e-12,
        weights=weights,
    )
    biotsavart = _StubSaveable(b"biotsavart-artifact")
    boozer_surface = _StubBoozerSurface()
    raw_values = (1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 7.0)
    weight_values = (2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0)
    terms = tuple(
        native.WeightedTerm(
            name,
            weight,
            _StubOptimizable(raw, np.asarray([raw, -raw]), label=name),
        )
        for name, weight, raw in zip(
            native.COMMON_OBJECTIVE_TERM_NAMES,
            weight_values,
            raw_values,
        )
    )
    recorded_csd = native._RecordedObjective(terms[5].objective)
    terms = (
        *terms[:5],
        native.WeightedTerm(terms[5].name, 13.0, recorded_csd),
        terms[6],
    )
    objective = native._weighted_sum(terms)

    monkeypatch.setattr(native, "_load_biotsavart", lambda _path: biotsavart)
    monkeypatch.setattr(native, "_load_surface", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        native,
        "_load_initial_boozer_state",
        lambda _path: native.BoozerSolveState(iota=0.125, G=-2.25),
    )
    monkeypatch.setattr(
        native,
        "build_native_boozer_surface",
        lambda **_kwargs: boozer_surface,
    )
    monkeypatch.setattr(
        native,
        "build_native_single_stage_objective",
        lambda **_kwargs: (objective, terms, recorded_csd),
    )

    def fake_minimize(function, initial_dofs, **kwargs: object) -> SimpleNamespace:
        assert kwargs["jac"] is True
        assert kwargs["method"] == "L-BFGS-B"
        value, gradient = function(initial_dofs)
        return SimpleNamespace(
            x=np.asarray(initial_dofs, dtype=np.float64),
            fun=value,
            jac=gradient,
            success=True,
            status=0,
            message="converged in deterministic fake",
            nit=0,
            nfev=1,
            njev=1,
        )

    monkeypatch.setattr(native, "minimize", fake_minimize)

    payload = native.run_native_single_stage(config)

    assert set(payload) == {
        "schema_version",
        "driver",
        "backend",
        "precision",
        "constraint_method",
        "mixed_precision",
        "comparison_schema_version",
        "objective_contract",
        "config",
        "inputs",
        "input_sha256",
        "run_config_sha256",
        "runtime",
        "optimizer",
        "timings",
        "initial_state",
        "final_state",
        "outputs",
    }
    assert payload["schema_version"] == 1
    assert payload["comparison_schema_version"] == 1
    assert payload["driver"] == "single_stage_banana_native"
    assert payload["backend"] == "native-simsopt-cpu"
    assert payload["precision"] == "float64"
    assert payload["mixed_precision"] is False
    contract = payload["objective_contract"]
    assert isinstance(contract, dict)
    assert contract["id"] == "banana-single-stage-common-v1"
    assert contract["ordered_terms"] == list(native.COMMON_OBJECTIVE_TERM_NAMES)
    assert contract["weights"] == dict(
        zip(native.COMMON_OBJECTIVE_TERM_NAMES, weight_values)
    )
    assert contract["inactive_term_requirements"] == {"coil_surface_distance": 0.0}
    assert contract["dof_names"] == ["fake:dof0", "fake:dof1"]
    assert contract["dof_count"] == 2
    expected_name_bytes = json.dumps(
        ["fake:dof0", "fake:dof1"],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert contract["dof_names_sha256"] == sha256(expected_name_bytes).hexdigest()
    expected_objective = sum(
        weight * raw for weight, raw in zip(weight_values, raw_values)
    )
    for state_name in ("initial_state", "final_state"):
        state = payload[state_name]
        assert isinstance(state, dict)
        assert state["objective"] == pytest.approx(expected_objective)
        assert state["iota"] == 0.125
        assert state["G"] == -2.25
        assert state["volume"] == 0.875
        assert state["dofs"] == [0.5, -1.5]
        assert state["dofs_sha256"] == _array_sha256(np.asarray([0.5, -1.5]))
        assert tuple(state["terms"]) == native.COMMON_OBJECTIVE_TERM_NAMES
        assert state["terms"]["coil_surface_distance"]["raw"] == 0.0
    assert payload["optimizer"]["success"] is True
    assert payload["optimizer"]["evaluations"] == 1
    assert payload["optimizer"]["rejected_evaluations"] == 0
    assert boozer_surface.solve_warm_starts == [
        (0.125, -2.25),
        (0.125, -2.25),
        (0.125, -2.25),
    ]
    assert payload["input_sha256"] == {
        "biotsavart": sha256(b"biotsavart-seed").hexdigest(),
        "surface": sha256(b"surface-seed").hexdigest(),
        "boozer_state": sha256(b'{"iota": 0.125, "G": -2.25}\n').hexdigest(),
    }
    results_path = config.output_root / "results.json"
    assert json.loads(results_path.read_text(encoding="utf-8")) == payload
    assert (config.output_root / "biot_savart_opt.json").read_bytes() == (
        b"biotsavart-artifact"
    )
    assert (config.output_root / "surf_opt.json").read_bytes() == b"surface-artifact"
    assert (config.output_root / "boozersurface_opt.json").read_bytes() == (
        b"boozer-artifact"
    )


def test_native_entrypoint_has_no_direct_jax_runtime_dependency() -> None:
    imported_modules: set[str] = set()
    for path in (DRIVER_PATH, CLI_PATH):
        for node in ast.walk(_tree(path)):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported_modules.add(node.module)
    assert not any(
        module == "jax"
        or module.startswith("jax.")
        or module.startswith("simsopt_jax_adapters")
        for module in imported_modules
    )
