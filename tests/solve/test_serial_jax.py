"""N6 serial JAX solve tests.

Lane key: derivative_heavy for explicit traceable Jacobian checks.
"""

import csv
import os

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from monty.tempfile import ScratchDir

from benchmarks.validation_ladder_contract import parity_ladder_tolerances
from simsopt._core.optimizable import Optimizable
from simsopt.objectives.constrained import ConstrainedProblem
from simsopt.objectives.functions import Identity
from simsopt.objectives.least_squares import LeastSquaresProblem
from simsopt.solve.serial import (
    constrained_serial_solve,
    least_squares_serial_solve,
    serial_solve,
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


def _assert_log_records_multiple_ordered_evaluations(path):
    rows = _read_simsopt_log_rows(path)
    assert len(rows) >= 3
    assert [int(row["function_evaluation"]) for row in rows] == list(range(len(rows)))
    return rows


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
        least_squares_serial_solve_jax(jax_prob, max_steps=64)

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
        log_files = [name for name in os.listdir(".") if name.startswith("simsopt_")]
        assert len(log_files) == 1
        jax_log = log_files[0]
        with open(jax_log) as log_file:
            log_text = log_file.read()
        assert "Problem type:\nleast_squares\nnparams:\n3\n" in log_text
        assert (
            "function_evaluation,seconds,x(0),x(1),x(2),objective_function" in log_text
        )
        rows = _assert_log_records_multiple_ordered_evaluations(jax_log)
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
        serial_solve_jax(jax_prob, max_steps=64)

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
        log_files = [name for name in os.listdir(".") if name.startswith("simsopt_")]
        assert len(log_files) == 1
        with open(log_files[0]) as log_file:
            log_text = log_file.read()
        assert "Problem type:\ngeneral\nnparams:\n2\n" in log_text
        assert "function_evaluation,seconds,x(0),x(1),objective_function" in log_text
        rows = _assert_log_records_multiple_ordered_evaluations(log_files[0])
        objectives = [float(row["objective_function"]) for row in rows]
        assert objectives[0] > objectives[-1]
        assert min(objectives) <= 1e-16


def test_constrained_serial_solve_jax_matches_host_slsqp_equality_problem():
    """Traceable AL lane matches host SLSQP on an equality-constrained QP."""
    with ScratchDir("."):
        host_model = _HostEqualityQuadratic()
        host_prob = ConstrainedProblem(
            host_model.objective_fn,
            tuples_nlc=[(host_model.equality_fn, 0.0, 0.0)],
        )
        constrained_serial_solve(
            host_prob,
            grad=True,
            options={"ftol": 1e-10, "maxiter": 200},
        )
        for name in os.listdir("."):
            if name.startswith("simsopt_") or name.startswith("constraints_"):
                os.remove(name)

        jax_prob = TraceableEqualityConstrainedProblem(
            objective_fn=lambda x: jnp.sum((x - jnp.array([2.0, 0.0])) ** 2),
            equality_constraint_fn=lambda x: jnp.array([jnp.sum(x) - 1.0]),
            x=jnp.array([0.0, 0.0], dtype=jnp.float64),
        )
        constrained_serial_solve_jax(
            jax_prob,
            max_outer=4,
            inner_max_steps=128,
            rtol=1e-10,
            atol=1e-10,
        )

        expected_x = np.array([1.5, -0.5])
        np.testing.assert_allclose(
            np.asarray(host_prob.x),
            expected_x,
            rtol=float(_DERIVATIVE_HEAVY_TOLS["first_derivative_rtol"]),
            atol=float(_DERIVATIVE_HEAVY_TOLS["first_derivative_atol"]),
        )
        np.testing.assert_allclose(
            np.asarray(jax_prob.x),
            expected_x,
            rtol=float(_DIRECT_KERNEL_TOLS["rtol"]),
            atol=float(_DIRECT_KERNEL_TOLS["atol"]),
        )
        np.testing.assert_allclose(
            np.asarray(jax_prob.equality_constraints()),
            np.zeros(1),
            atol=float(_WHOLE_SOLVE_TOLS["whole_solve_value_atol"]),
        )
        assert float(jax_prob.objective()) <= 0.50000001
        log_files = [name for name in os.listdir(".") if name.startswith("simsopt_")]
        assert len(log_files) == 1
        with open(log_files[0]) as log_file:
            log_text = log_file.read()
        assert "Problem type:\nconstrained\nnparams:\n2\n" in log_text
        rows = _assert_log_records_multiple_ordered_evaluations(log_files[0])
        objectives = [float(row["objective_function"]) for row in rows]
        assert objectives[0] > objectives[-1]
        assert min(objectives) <= 0.50000001


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
