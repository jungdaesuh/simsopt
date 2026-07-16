from __future__ import annotations

from collections.abc import Iterable
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest


jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
pytest.importorskip("simsoptpp")

from examples.single_stage_optimization.banana_opt import (  # noqa: E402
    jax_banana_drivers as banana_drivers,
)
from examples.single_stage_optimization.banana_opt.jax_banana_drivers import (  # noqa: E402
    GlobalRadiusCurvatureJAX,
    EllipseWidthJAX,
    PoloidalExtentJAX,
    banana_base_curve,
    boozer_solver_grid_shape,
    build_biotsavart,
    build_boozer_surface_copy,
    comparison_backend_label,
    minimize_single_stage_soft_penalty,
    read_banana_dofs,
)
from examples.single_stage_optimization.banana_opt.jax_banana_types import (  # noqa: E402
    BANANA_TARGETS,
    DEFAULT_BANANA_DOFS,
    DEFAULT_BOOZER_CONSTRAINT_WEIGHT,
    DEFAULT_PROXY_RZ,
    HBT_BANANA_WS,
    BoozerSolveState,
    SingleStageWeights,
    Stage2Weights,
)
from simsopt.geo import SurfaceXYZTensorFourier  # noqa: E402
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_ROOT = REPO_ROOT / "examples" / "single_stage_optimization"
STAGE2_SCRIPT = EXAMPLE_ROOT / "STAGE_2" / "banana_coil_solver.py"
SINGLE_STAGE_SCRIPT = EXAMPLE_ROOT / "SINGLE_STAGE" / "single_stage_banana_example.py"


def _run_driver(
    script: Path, args: list[str], *, timeout: int
) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "PYTHONNOUSERSITE": "1",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }
    result = subprocess.run(
        [sys.executable, str(script), *args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def _single_match(paths: Iterable[Path]) -> Path:
    matches = sorted(paths)
    assert len(matches) == 1, matches
    return matches[0]


def _assert_finite_positive_gradient(objective) -> None:
    value = float(objective.J())
    gradient = np.asarray(objective.dJ(), dtype=float)
    assert np.isfinite(value)
    assert value > 0.0
    assert gradient.shape == np.asarray(objective.x).shape
    assert np.all(np.isfinite(gradient))
    assert np.linalg.norm(gradient) > 0.0


def test_boozer_solver_grid_distinguishes_penalty_and_exact_modes():
    common = {
        "mpol": 2,
        "ntor": 3,
        "nphi": 19,
        "ntheta": 16,
        "stellsym": True,
    }

    assert DEFAULT_BOOZER_CONSTRAINT_WEIGHT == pytest.approx(1.0)
    assert boozer_solver_grid_shape(
        **common,
        constraint_weight=DEFAULT_BOOZER_CONSTRAINT_WEIGHT,
    ) == (19, 16)
    assert boozer_solver_grid_shape(
        **common,
        constraint_weight=None,
    ) == (7, 5)


@pytest.mark.parametrize(
    ("device_platform", "backend_label"),
    (("cpu", "jax-cpu"), ("gpu", "jax-cuda")),
)
def test_comparison_backend_label_matches_execution_platform(
    device_platform: str,
    backend_label: str,
) -> None:
    assert comparison_backend_label(device_platform) == backend_label


def test_comparison_backend_label_rejects_unsupported_platform() -> None:
    with pytest.raises(ValueError, match="only JAX CPU or CUDA"):
        comparison_backend_label("tpu")


def test_single_stage_minimizer_penalizes_rejected_boozer_candidates(monkeypatch):
    class FakeObjective:
        def __init__(self):
            self.x = np.asarray([1.0, -2.0])
            self.gradient_calls = 0

        def J(self) -> float:
            return float(np.dot(self.x, self.x))

        def dJ(self) -> np.ndarray:
            self.gradient_calls += 1
            if self.gradient_calls == 2:
                raise banana_drivers.BoozerAdjointLinearSolveError(
                    "expected rejected candidate"
                )
            return 2.0 * self.x

    class FakeSurface:
        def __init__(self):
            self.dofs = np.asarray([0.5, 0.25])

        def get_dofs(self) -> np.ndarray:
            return self.dofs.copy()

        def set_dofs(self, dofs: np.ndarray) -> None:
            self.dofs = np.asarray(dofs, dtype=float).copy()

    class FakeBoozerSurface:
        def __init__(self):
            self.calls = 0
            self.res = None
            self.surface = FakeSurface()

        def run_code(self, iota: float, G: float):
            self.calls += 1
            success = self.calls != 2
            self.res = {
                "success": success,
                "iota": iota + 0.01,
                "G": G,
            }
            return self.res

    objective = FakeObjective()
    boozersurface = FakeBoozerSurface()
    messages: list[str] = []

    def fake_minimize(fun, x0, **kwargs):
        del kwargs
        baseline_value, baseline_gradient = fun(x0)
        rejected_value, rejected_gradient = fun(x0 + 1.0)
        rejected_adjoint_value, rejected_adjoint_gradient = fun(x0 + 0.5)
        assert rejected_value > baseline_value
        assert rejected_adjoint_value > baseline_value
        np.testing.assert_array_equal(rejected_gradient, baseline_gradient)
        np.testing.assert_array_equal(rejected_adjoint_gradient, baseline_gradient)
        return SimpleNamespace(x=np.asarray(x0, dtype=float))

    monkeypatch.setattr(banana_drivers, "minimize", fake_minimize)
    _, tracker = minimize_single_stage_soft_penalty(
        objective=objective,
        boozersurface=boozersurface,
        initial_state=BoozerSolveState(iota=0.15, G=-2.0),
        maxiter=1,
        log=messages.append,
    )

    assert tracker.evaluations == 3
    assert tracker.rejected_evaluations == 2
    assert any("rejected_boozer_solve=true" in message for message in messages)
    assert any("rejected_boozer_adjoint=true" in message for message in messages)


def test_single_stage_minimizer_does_not_promote_unaccepted_trial_state(monkeypatch):
    class FakeObjective:
        def __init__(self):
            self.x = np.asarray([1.0, -2.0])

        def J(self) -> float:
            return float(np.dot(self.x, self.x))

        def dJ(self) -> np.ndarray:
            return 2.0 * self.x

    class FakeSurface:
        def __init__(self):
            self.dofs = np.asarray([0.5, 0.25])

        def get_dofs(self) -> np.ndarray:
            return self.dofs.copy()

        def set_dofs(self, dofs: np.ndarray) -> None:
            self.dofs = np.asarray(dofs, dtype=float).copy()

    class FakeBoozerSurface:
        def __init__(self):
            self.calls = 0
            self.inputs: list[tuple[float, float]] = []
            self.need_to_run_code = True
            self.res = None
            self.surface = FakeSurface()

        def run_code(self, iota: float, G: float):
            self.calls += 1
            self.inputs.append((iota, G))
            success = self.calls != 3
            self.surface.set_dofs(np.asarray([self.calls, -self.calls], dtype=float))
            self.res = {
                "success": success,
                "iota": iota + 0.1 * self.calls,
                "G": G,
            }
            return self.res

    objective = FakeObjective()
    boozersurface = FakeBoozerSurface()

    def fake_minimize(fun, x0, **kwargs):
        del kwargs
        baseline_value, baseline_gradient = fun(x0)
        trial_value, _ = fun(x0 + 0.25)
        rejected_value, rejected_gradient = fun(x0 + 0.5)
        assert trial_value != baseline_value
        assert rejected_value > baseline_value
        np.testing.assert_array_equal(rejected_gradient, baseline_gradient)
        return SimpleNamespace(x=np.asarray(x0, dtype=float))

    monkeypatch.setattr(banana_drivers, "minimize", fake_minimize)
    minimize_single_stage_soft_penalty(
        objective=objective,
        boozersurface=boozersurface,
        initial_state=BoozerSolveState(iota=0.15, G=-2.0),
        maxiter=1,
        log=lambda _message: None,
    )

    assert boozersurface.inputs == [
        (0.15, -2.0),
        (0.25, -2.0),
        (0.25, -2.0),
        (0.25, -2.0),
    ]


def test_single_stage_minimizer_promotes_callback_accepted_trial_state(monkeypatch):
    class FakeObjective:
        def __init__(self):
            self.x = np.asarray([1.0, -2.0])

        def J(self) -> float:
            return float(np.dot(self.x, self.x))

        def dJ(self) -> np.ndarray:
            return 2.0 * self.x

    class FakeSurface:
        def __init__(self):
            self.dofs = np.asarray([0.5, 0.25])

        def get_dofs(self) -> np.ndarray:
            return self.dofs.copy()

        def set_dofs(self, dofs: np.ndarray) -> None:
            self.dofs = np.asarray(dofs, dtype=float).copy()

    class FakeBoozerSurface:
        def __init__(self):
            self.calls = 0
            self.inputs: list[tuple[float, float]] = []
            self.starting_surface_dofs: list[np.ndarray] = []
            self.need_to_run_code = True
            self.res = None
            self.surface = FakeSurface()

        def run_code(self, iota: float, G: float):
            self.calls += 1
            self.inputs.append((iota, G))
            self.starting_surface_dofs.append(self.surface.get_dofs())
            self.surface.set_dofs(np.asarray([self.calls, -self.calls], dtype=float))
            self.res = {
                "success": True,
                "iota": iota + 0.1 * self.calls,
                "G": G,
            }
            return self.res

    objective = FakeObjective()
    boozersurface = FakeBoozerSurface()
    accepted_dofs = objective.x + 0.25
    later_trial_dofs = objective.x + 0.5

    def fake_minimize(fun, x0, *, callback, **kwargs):
        del kwargs
        fun(x0)
        fun(accepted_dofs)
        callback(accepted_dofs)
        fun(later_trial_dofs)
        return SimpleNamespace(x=accepted_dofs.copy())

    monkeypatch.setattr(banana_drivers, "minimize", fake_minimize)
    _, tracker = minimize_single_stage_soft_penalty(
        objective=objective,
        boozersurface=boozersurface,
        initial_state=BoozerSolveState(iota=0.15, G=-2.0),
        maxiter=1,
        log=lambda _message: None,
    )

    assert tracker.iterations == 1
    assert tracker.rejected_evaluations == 0
    np.testing.assert_allclose(
        boozersurface.inputs,
        [
            (0.15, -2.0),
            (0.25, -2.0),
            (0.45, -2.0),
            (0.45, -2.0),
        ],
    )
    np.testing.assert_allclose(
        boozersurface.starting_surface_dofs,
        [
            (0.5, 0.25),
            (1.0, -1.0),
            (2.0, -2.0),
            (2.0, -2.0),
        ],
    )
    np.testing.assert_array_equal(objective.x, accepted_dofs)


def test_banana_geometry_jax_objectives_have_finite_gradients():
    biotsavart = build_biotsavart(
        tf_current_ka=-80.0,
        tf_fix_current=True,
        banana_current_ka=16.0,
        banana_fix_current=True,
        banana_order=1,
        banana_dofs=dict(DEFAULT_BANANA_DOFS),
        proxy_current_ka=0.0,
        proxy_rz=DEFAULT_PROXY_RZ,
        vf_current_ka=0.0,
        vf_fix_current=True,
    )
    curve = banana_base_curve(BiotSavartJAX(biotsavart.coils))

    objectives = [
        PoloidalExtentJAX(curve, HBT_BANANA_WS.major_radius, theta_target=0.05),
        EllipseWidthJAX(curve),
        GlobalRadiusCurvatureJAX(
            curve,
            minimum_radius=2.0,
            exp_weight=0.25,
        ),
    ]
    for objective in objectives:
        _assert_finite_positive_gradient(objective)


def test_banana_driver_defaults_match_reference_contract():
    assert Stage2Weights().length == pytest.approx(5.0e-2)
    assert SingleStageWeights().ccdist == pytest.approx(1.0e6)
    assert SingleStageWeights().width == pytest.approx(1.0e1)
    assert Stage2Weights().global_curvature_radius == pytest.approx(1.0e3)
    assert SingleStageWeights().global_curvature_radius == pytest.approx(1.0e3)
    assert BANANA_TARGETS.width_max == pytest.approx(0.197)
    assert BANANA_TARGETS.width_min == pytest.approx(0.050)
    assert BANANA_TARGETS.global_curvature_radius_min == pytest.approx(0.010)
    assert BANANA_TARGETS.global_curvature_radius_exp_weight == pytest.approx(0.010)


def test_read_banana_dofs_accepts_reference_yaml(tmp_path):
    dofs_path = tmp_path / "banana_dofs.yaml"
    dofs_path.write_text(
        "\n".join(
            (
                "phic(0):    0.0600",
                "phic(1):    0.0300",
                "thetac(0):  0.5000",
                "thetas(1):  0.1000",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    assert read_banana_dofs(dofs_path) == pytest.approx(
        {
            "phic(0)": 0.06,
            "phic(1)": 0.03,
            "thetac(0)": 0.5,
            "thetas(1)": 0.1,
        }
    )


def test_read_banana_dofs_keeps_table_format_contract(tmp_path):
    dofs_path = tmp_path / "banana_dofs.txt"
    dofs_path.write_text(
        "\n".join(
            (
                "phic(0) 0.0600",
                "phic(1) 0.0300",
                "thetac(0) 0.5000",
                "thetas(1) 0.1000",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    assert read_banana_dofs(dofs_path) == pytest.approx(
        {
            "phic(0)": 0.06,
            "phic(1)": 0.03,
            "thetac(0)": 0.5,
            "thetas(1)": 0.1,
        }
    )


def test_build_boozer_surface_copy_regrids_tensor_surface_seed():
    source = SurfaceXYZTensorFourier(
        mpol=2,
        ntor=2,
        nfp=1,
        stellsym=False,
        quadpoints_phi=np.linspace(0.0, 1.0, 7, endpoint=False),
        quadpoints_theta=np.linspace(0.0, 1.0, 8, endpoint=False),
    )

    copied = build_boozer_surface_copy(
        source,
        mpol=2,
        ntor=2,
        nphi=8,
        ntheta=8,
    )

    assert isinstance(copied, SurfaceXYZTensorFourier)
    assert copied.mpol == 2
    assert copied.ntor == 2
    assert copied.quadpoints_phi.size == 8
    assert copied.quadpoints_theta.size == 8
    assert np.all(np.isfinite(copied.gamma()))


@pytest.mark.slow
def test_soft_penalty_cli_stage2_to_single_stage_smoke(tmp_path):
    stage2_root = tmp_path / "stage2"
    single_stage_seed_spec = tmp_path / "single_stage_seed_spec.json"

    _run_driver(
        STAGE2_SCRIPT,
        [
            "--backend",
            "jax",
            "--maxiter",
            "0",
            "--nphi",
            "8",
            "--ntheta",
            "7",
            "--order",
            "1",
            "--output-root",
            str(stage2_root),
            "--skip-postprocess",
        ],
        timeout=90,
    )

    stage2_dir = stage2_root
    stage2_results = stage2_dir / "results.json"
    stage2_payload = json.loads(stage2_results.read_text())
    assert (stage2_dir / "surf_opt.json").exists()
    assert (stage2_dir / "biot_savart_opt.json").exists()
    assert (stage2_dir / "boozersurface_opt.json").exists()
    assert stage2_payload["driver"] == "stage2_jax_soft_penalty"
    assert stage2_payload["optimizer"]["nfev"] > 0
    assert np.isfinite(stage2_payload["diagnostics"]["J"])
    assert "J_squared_flux" in stage2_payload["diagnostics"]
    assert "J_global_curvature_radius" in stage2_payload["diagnostics"]

    _run_driver(
        SINGLE_STAGE_SCRIPT,
        [
            "--backend",
            "jax",
            "--compile-jax-runtime-seed-spec",
            "--warm-start-run-dir",
            str(stage2_dir),
            "--jax-runtime-seed-spec",
            str(single_stage_seed_spec),
            "--mpol",
            "1",
            "--ntor",
            "1",
            "--nphi",
            "8",
            "--ntheta",
            "8",
        ],
        timeout=90,
    )
    assert single_stage_seed_spec.exists()
    seed_payload = json.loads(single_stage_seed_spec.read_text())
    assert seed_payload["driver"] == "single_stage_jax_runtime_seed_spec"
    assert seed_payload["biotsavart"] == str(stage2_dir / "biot_savart_opt.json")
    assert seed_payload["surface"] == str(stage2_dir / "surf_opt.json")
    assert seed_payload["boozersurface"] == str(stage2_dir / "boozersurface_opt.json")
