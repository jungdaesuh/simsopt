"""N6 serial JAX solve tests.

Lane key: derivative_heavy for explicit traceable Jacobian checks.
"""

import ast
import csv
import inspect
import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from benchmarks.validation_ladder_contract import parity_ladder_tolerances
from jax.sharding import Mesh
from monty.tempfile import ScratchDir
from simsopt._core.optimizable import Optimizable
from simsopt.objectives.constrained import ConstrainedProblem
from simsopt.objectives.functions import Identity
from simsopt.objectives.least_squares import LeastSquaresProblem
from simsopt.solve.serial import (
    least_squares_serial_solve,
    serial_solve,
)
from simsopt_jax.solve.contracts import OptimizerResult
from simsopt_jax.solve.driver import Driver
from simsopt_jax.solve.optimistix.contracts import (
    OptimistixLMOptions,
)
from simsopt_jax.solve.serial import (
    TraceableEqualityConstrainedProblem,
    TraceableLeastSquaresProblem,
    TraceableScalarProblem,
    constrained_serial_solve_jax,
    least_squares_serial_solve_jax,
    serial_solve_jax,
    traceable_least_squares_jacobian,
)
from simsopt_jax.solve.simsopt.contracts import (
    SimsoptBFGSOptions,
    SimsoptLBFGSBOptions,
    SimsoptLMGMRESOptions,
    SimsoptLMQROptions,
)

_DIRECT_KERNEL_TOLS = parity_ladder_tolerances("direct_kernel")
_DERIVATIVE_HEAVY_TOLS = parity_ladder_tolerances("derivative_heavy")
_WHOLE_SOLVE_TOLS = parity_ladder_tolerances("gpu-runtime")


def _read_simsopt_log_rows(path):
    with open(path, newline="") as log_file:
        lines = log_file.readlines()
    header_index = next(
        index
        for index, line in enumerate(lines)
        if line.startswith("function_evaluation,seconds")
    )
    return list(csv.DictReader(lines[header_index:]))


def _assert_log_records_bounded_initial_and_final_states(path):
    rows = _read_simsopt_log_rows(path)
    assert len(rows) == 2
    assert [int(row["function_evaluation"]) for row in rows] == [0, 1]
    return rows


def test_serial_jax_has_no_implicit_optional_optimizer_backend() -> None:
    """SIMSOPT-compatible serial APIs must not silently select optional solvers."""
    source_path = Path(inspect.getfile(least_squares_serial_solve_jax))
    syntax_tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_modules = {
        alias.name
        for node in ast.walk(syntax_tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or ""
        for node in ast.walk(syntax_tree)
        if isinstance(node, ast.ImportFrom)
    }

    assert "optimistix" not in imported_modules
    assert "optax" not in imported_modules


def test_serial_jax_numerical_solves_do_not_use_host_callbacks() -> None:
    """The shared CPU/GPU numerical path must not require a host callback."""
    source = Path(inspect.getfile(least_squares_serial_solve_jax)).read_text(
        encoding="utf-8"
    )

    assert "jax.debug.callback" not in source


def test_constrained_serial_solve_jax_fails_until_custom_contract_exists():
    """An optional backend must not masquerade as SIMSOPT constrained parity."""
    problem = TraceableEqualityConstrainedProblem(
        objective_fn=lambda x: jnp.sum(x * x),
        equality_constraint_fn=lambda x: jnp.asarray((jnp.sum(x) - 1.0,)),
        x=jnp.zeros(2, dtype=jnp.float64),
    )

    with np.testing.assert_raises_regex(
        NotImplementedError,
        "no SIMSOPT-owned constrained solver",
    ):
        constrained_serial_solve_jax(problem)


def _weighted_quadratic_residual(x):
    return jnp.array(
        (
            x[0] - 1.0,
            jnp.sqrt(2.0) * (x[1] - 2.0),
            jnp.sqrt(3.0) * (x[2] - 3.0),
        ),
        dtype=x.dtype,
    )


class _HostQuadratic:
    def __init__(self):
        self.target = np.array([1.5, -0.5])
        self.x = np.array([0.0, 0.0])
        self.lower_bounds = np.full(2, -np.inf)
        self.upper_bounds = np.full(2, np.inf)

    @property
    def dof_size(self):
        return int(self.x.size)

    @property
    def bounds(self):
        return self.lower_bounds, self.upper_bounds

    def __call__(self, x):
        self.x = np.asarray(x, dtype=float)
        delta = self.x - self.target
        return float(delta @ delta)


class _HostEqualityQuadratic(Optimizable):
    def __init__(self):
        self.target = np.array([2.0, 0.0])
        super().__init__(np.array([0.0, 0.0]))

    def objective_fn(self):
        delta = self.full_x - self.target
        return float(delta @ delta)

    def equality_fn(self):
        return float(np.sum(self.full_x) - 1.0)


def test_least_squares_serial_solve_jax_matches_host_quadratic_problem():
    """Traceable JAX lane reaches the same optimum as the host toy problem."""
    with ScratchDir("."):
        iden1 = Identity()
        iden2 = Identity()
        iden3 = Identity()
        host_prob = LeastSquaresProblem.from_tuples(
            [
                (iden1.f, 1, 1),
                (iden2.f, 2, 2),
                (iden3.f, 3, 3),
            ]
        )
        least_squares_serial_solve(host_prob)
        for name in os.listdir("."):
            if name.startswith("simsopt_"):
                os.remove(name)

        jax_prob = TraceableLeastSquaresProblem(
            residual_fn=_weighted_quadratic_residual,
            x=jnp.array([0.0, 0.0, 0.0], dtype=jnp.float64),
        )
        result = least_squares_serial_solve_jax(jax_prob, max_steps=64)

        expected_x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(
            np.asarray(host_prob.x),
            expected_x,
            rtol=float(_WHOLE_SOLVE_TOLS["whole_solve_value_rtol"]),
            atol=float(_WHOLE_SOLVE_TOLS["whole_solve_value_atol"]),
        )
        np.testing.assert_allclose(
            np.asarray(jax_prob.x),
            expected_x,
            rtol=float(_WHOLE_SOLVE_TOLS["whole_solve_value_rtol"]),
            atol=float(_WHOLE_SOLVE_TOLS["whole_solve_value_atol"]),
        )
        assert host_prob.objective() <= 1e-16
        assert float(jax_prob.objective()) <= 1e-16
        assert result.driver == Driver.SIMSOPT_LM_GMRES
        assert isinstance(result, OptimizerResult)
        assert isinstance(result.options_used, SimsoptLMGMRESOptions)
        np.testing.assert_allclose(result.x, expected_x)
        expected_residual = np.asarray(jax_prob.residuals())
        expected_gradient = np.diag(np.sqrt([1.0, 2.0, 3.0])) @ expected_residual
        np.testing.assert_allclose(result.residual, expected_residual)
        np.testing.assert_allclose(result.jac, expected_gradient)
        assert result.residual_jacobian is None
        assert result.hessian is None
        assert result.fun == pytest.approx(0.5 * float(jax_prob.objective()))
        assert result.success
        assert result.status in (0, 1, 2)
        assert 0 < result.nit <= 64
        assert result.nfev > 0
        assert result.njev > 0
        log_files = [name for name in os.listdir(".") if name.startswith("simsopt_")]
        assert len(log_files) == 1
        jax_log = log_files[0]
        with open(jax_log) as log_file:
            log_text = log_file.read()
        assert "Problem type:\nleast_squares\nnparams:\n3\n" in log_text
        assert (
            "function_evaluation,seconds,x(0),x(1),x(2),objective_function" in log_text
        )
        rows = _assert_log_records_bounded_initial_and_final_states(jax_log)
        objectives = [float(row["objective_function"]) for row in rows]
        assert objectives[0] > objectives[-1]
        assert min(objectives) <= 1e-16


def test_serial_solve_jax_matches_host_general_quadratic_problem():
    """Traceable scalar JAX lane reaches the host general-solve optimum."""
    with ScratchDir("."):
        host_prob = _HostQuadratic()
        serial_solve(host_prob)
        for name in os.listdir("."):
            if name.startswith("simsopt_"):
                os.remove(name)

        jax_prob = TraceableScalarProblem(
            objective_fn=lambda x: jnp.sum((x - jnp.array([1.5, -0.5])) ** 2),
            x=jnp.array([0.0, 0.0], dtype=jnp.float64),
        )
        result = serial_solve_jax(jax_prob, max_steps=64)

        expected_x = np.array([1.5, -0.5])
        np.testing.assert_allclose(
            np.asarray(host_prob.x),
            expected_x,
            rtol=float(_WHOLE_SOLVE_TOLS["whole_solve_value_rtol"]),
            atol=float(_WHOLE_SOLVE_TOLS["whole_solve_value_atol"]),
        )
        np.testing.assert_allclose(
            np.asarray(jax_prob.x),
            expected_x,
            rtol=float(_DIRECT_KERNEL_TOLS["rtol"]),
            atol=float(_DIRECT_KERNEL_TOLS["atol"]),
        )
        assert host_prob(host_prob.x) <= 1e-12
        assert float(jax_prob.objective()) <= 1e-16
        assert result.driver == Driver.SIMSOPT_BFGS
        assert isinstance(result, OptimizerResult)
        assert isinstance(result.options_used, SimsoptBFGSOptions)
        np.testing.assert_allclose(result.x, expected_x)
        np.testing.assert_allclose(result.jac, np.zeros(2), atol=1e-14)
        assert result.residual is None
        assert result.residual_jacobian is None
        assert result.hessian is None
        assert result.fun == pytest.approx(float(jax_prob.objective()))
        assert result.success
        assert result.status == 0
        assert 0 < result.nit <= 64
        assert result.nfev > 0
        assert result.njev > 0
        log_files = [name for name in os.listdir(".") if name.startswith("simsopt_")]
        assert len(log_files) == 1
        with open(log_files[0]) as log_file:
            log_text = log_file.read()
        assert "Problem type:\ngeneral\nnparams:\n2\n" in log_text
        assert "function_evaluation,seconds,x(0),x(1),objective_function" in log_text
        rows = _assert_log_records_bounded_initial_and_final_states(log_files[0])
        objectives = [float(row["objective_function"]) for row in rows]
        assert objectives[0] > objectives[-1]
        assert min(objectives) <= 1e-16


def test_serial_solve_jax_supports_simsopt_owned_limited_memory_driver():
    """Fast scalar lanes can opt into the SIMSOPT-owned O(n) history solver."""
    with ScratchDir("."):
        problem = TraceableScalarProblem(
            objective_fn=lambda x: jnp.sum((x - jnp.array([1.5, -0.5])) ** 2),
            x=jnp.array([0.0, 0.0], dtype=jnp.float64),
        )
        result = serial_solve_jax(
            problem,
            driver=Driver.SIMSOPT_LBFGSB,
            max_steps=64,
        )

        np.testing.assert_allclose(problem.x, np.array([1.5, -0.5]), atol=1.0e-10)
        assert result.driver == Driver.SIMSOPT_LBFGSB
        assert isinstance(result.options_used, SimsoptLBFGSBOptions)
        assert result.success is True


def test_serial_solve_jax_accepts_bounded_limited_memory_history() -> None:
    with ScratchDir("."):
        problem = TraceableScalarProblem(
            objective_fn=lambda x: jnp.sum((x - 1.0) ** 2),
            x=jnp.zeros(4, dtype=jnp.float64),
        )
        result = serial_solve_jax(
            problem,
            driver=Driver.SIMSOPT_LBFGSB,
            max_steps=32,
            maxcor=24,
        )

    assert isinstance(result.options_used, SimsoptLBFGSBOptions)
    assert result.options_used.maxcor == 24


def test_serial_solve_jax_forwards_relative_step_tolerance():
    problem = TraceableScalarProblem(
        objective_fn=lambda x: jnp.sum(x * x),
        x=jnp.asarray([1.0], dtype=jnp.float64),
    )

    with ScratchDir("."):
        result = serial_solve_jax(
            problem,
            rtol=0.0,
            atol=1.0e-8,
            max_steps=16,
        )

    assert isinstance(result.options_used, SimsoptBFGSOptions)
    assert result.options_used.xrtol == 0.0


@pytest.mark.parametrize(
    ("driver", "options_type"),
    [
        (Driver.SIMSOPT_LM_GMRES, SimsoptLMGMRESOptions),
        (Driver.SIMSOPT_LM_QR, SimsoptLMQROptions),
    ],
)
def test_least_squares_serial_solve_jax_honors_requested_gradient_tolerance(
    driver, options_type
) -> None:
    requested_tolerance = 1.0e-12
    problem = TraceableLeastSquaresProblem(
        residual_fn=lambda x: x,
        x=jnp.asarray([5.0e-11], dtype=jnp.float64),
    )

    with ScratchDir("."):
        result = least_squares_serial_solve_jax(
            problem,
            driver=driver,
            rtol=requested_tolerance,
            atol=requested_tolerance,
            max_steps=8,
        )

    assert isinstance(result.options_used, options_type)
    assert result.options_used.gtol == requested_tolerance
    assert result.nit > 0
    assert abs(float(problem.x[0])) < requested_tolerance


def test_deprecated_lm_keyword_preserves_explicit_optimistix_selection():
    least_squares_problem = TraceableLeastSquaresProblem(
        residual_fn=lambda x: x - 1.0,
        x=jnp.asarray([0.0], dtype=jnp.float64),
    )
    with ScratchDir("."), pytest.warns(DeprecationWarning, match="OPTIMISTIX_LM"):
        least_squares_result = least_squares_serial_solve_jax(
            least_squares_problem,
            optimizer="lm",
            max_steps=32,
        )
    assert least_squares_result.driver == Driver.OPTIMISTIX_LM
    assert isinstance(least_squares_result.options_used, OptimistixLMOptions)


@pytest.mark.parametrize(
    ("solve", "problem", "optimizer", "message"),
    [
        (
            least_squares_serial_solve_jax,
            TraceableLeastSquaresProblem(
                residual_fn=lambda x: x - 1.0,
                x=jnp.asarray([0.0], dtype=jnp.float64),
            ),
            "gauss_newton",
            "no typed backend-neutral driver",
        ),
        (
            serial_solve_jax,
            TraceableScalarProblem(
                objective_fn=lambda x: jnp.sum((x - 1.0) ** 2),
                x=jnp.asarray([0.0], dtype=jnp.float64),
            ),
            "bfgs",
            "no behavior-equivalent typed driver",
        ),
    ],
)
def test_legacy_optimizer_keywords_without_equivalent_typed_drivers_fail_closed(
    solve,
    problem,
    optimizer,
    message,
):
    initial_x = np.asarray(problem.x).copy()
    with pytest.raises(NotImplementedError, match=message):
        solve(problem, optimizer=optimizer)
    np.testing.assert_array_equal(np.asarray(problem.x), initial_x)


@pytest.mark.parametrize(
    ("solve", "problem"),
    [
        (
            least_squares_serial_solve_jax,
            TraceableLeastSquaresProblem(
                residual_fn=lambda x: x - 100.0,
                x=jnp.asarray([0.0], dtype=jnp.float64),
            ),
        ),
        (
            serial_solve_jax,
            TraceableScalarProblem(
                objective_fn=lambda x: jnp.sum((x - 100.0) ** 2),
                x=jnp.asarray([0.0], dtype=jnp.float64),
            ),
        ),
    ],
)
def test_failed_serial_solve_does_not_publish_state_or_log(solve, problem):
    initial_x = np.asarray(problem.x).copy()
    with ScratchDir("."):
        with pytest.raises(RuntimeError, match="failed with driver="):
            solve(problem, max_steps=1)
        assert not list(Path(".").glob("simsopt_*.dat"))
    np.testing.assert_array_equal(np.asarray(problem.x), initial_x)


def test_serial_solvers_preserve_nondefault_device_placement(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    source = """
import jax
import jax.numpy as jnp
import numpy as np
from simsopt_jax.solve.serial import TraceableScalarProblem, serial_solve_jax

device = jax.devices()[1]
initial = jax.device_put(np.asarray([2.0], dtype=np.float64), device)
problem = TraceableScalarProblem(lambda x: jnp.sum((x - 1.0) ** 2), initial)
serial_solve_jax(problem, max_steps=16)
assert problem.x.sharding == initial.sharding
"""
    environment = dict(os.environ)
    environment.update(
        {
            "JAX_PLATFORMS": "cpu",
            "JAX_ENABLE_X64": "1",
            "MPI4PY_RC_INITIALIZE": "false",
            "PYTHONPATH": str(repo_root / "src"),
            "XLA_FLAGS": "--xla_force_host_platform_device_count=2",
        }
    )

    completed = subprocess.run(
        [sys.executable, "-c", source],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_traceable_least_squares_jacfwd_matches_shard_map_jacobian():
    """Derivative-heavy lane: AD Jacobian equals explicit sharded finite difference."""
    matrix = jnp.array(
        (
            (1.0, 2.0, -0.5),
            (-3.0, 0.25, 4.0),
        ),
        dtype=jnp.float64,
    )
    prob = TraceableLeastSquaresProblem(
        residual_fn=lambda x: matrix @ x,
        x=jnp.array([0.5, -0.25, 1.5], dtype=jnp.float64),
    )
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("dof",))

    expected = traceable_least_squares_jacobian(prob, prob.x, method="jacfwd")
    actual = traceable_least_squares_jacobian(
        prob,
        prob.x,
        method="shard_map",
        abs_step=2.0**-30,
        mesh=mesh,
    )

    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(expected),
        rtol=float(_DERIVATIVE_HEAVY_TOLS["first_derivative_rtol"]),
        atol=float(_DERIVATIVE_HEAVY_TOLS["first_derivative_atol"]),
    )


def test_least_squares_serial_solve_jax_rejects_host_graph_problem():
    """The JAX lanes require explicit traceable state, not host graphs."""
    least_squares_host_prob = LeastSquaresProblem.from_tuples([(Identity().f, 1, 1)])
    scalar_host_prob = _HostQuadratic()
    host_model = _HostEqualityQuadratic()
    constrained_host_prob = ConstrainedProblem(
        host_model.objective_fn,
        tuples_nlc=[(host_model.equality_fn, 0.0, 0.0)],
    )

    with np.testing.assert_raises(TypeError):
        least_squares_serial_solve_jax(least_squares_host_prob)
    with np.testing.assert_raises(TypeError):
        serial_solve_jax(scalar_host_prob)
    with np.testing.assert_raises(TypeError):
        constrained_serial_solve_jax(constrained_host_prob)
