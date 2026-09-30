"""Analytic benchmarks: small problems whose KKT point (x*, lambda*) is known in
closed form, solved with default settings, unboxed and boxed. Each pins
``converged`` at x* and lambda* for one start situation the loop must handle:
a feasible start whose subproblem minimizer is infeasible, an infeasible
start, a start at the optimum, an active row beside a slack row, a bound
beside a row, and rows that change activity during the run.
"""

import unittest
from typing import NamedTuple, Optional, Sequence

import numpy as np

from simsopt_alm import ALMSettings, augmented_inequality_objective, minimize_alm

INNER_OPTIONS = {"maxiter": 1000}
# Unboxed (the default) and boxed inner solves.
SETTINGS_VARIANTS = {
    "default": ALMSettings(),
    "boxed": ALMSettings(trust_radius_init=0.25),
}
X_ATOL = 1.0e-5
MULTIPLIER_ATOL = 1.0e-4


def pulled_out(x, multipliers, penalty):
    """min (x - 3)^2 s.t. x - 1 <= 0: x* = 1, lambda* = 4."""
    return augmented_inequality_objective(
        (x[0] - 3.0) ** 2, [2.0 * (x[0] - 3.0)], [x[0] - 1.0], [[1.0]], multipliers, penalty
    )


def active_beside_slack(x, multipliers, penalty):
    """min (x0 - 3)^2 + (x1 - 1)^2 s.t. x0 - 1 <= 0, x1 - 5 <= 0:
    x* = (1, 1), lambda* = (4, 0)."""
    return augmented_inequality_objective(
        (x[0] - 3.0) ** 2 + (x[1] - 1.0) ** 2,
        [2.0 * (x[0] - 3.0), 2.0 * (x[1] - 1.0)],
        [x[0] - 1.0, x[1] - 5.0],
        [[1.0, 0.0], [0.0, 1.0]],
        multipliers,
        penalty,
    )


def row_beside_bound(x, multipliers, penalty):
    """min (x0 - 3)^2 + (x1 - 3)^2 s.t. x1 - 1 <= 0, with x0 <= 2 a base
    bound: x* = (2, 1), lambda* = 4 (the bound's multiplier, 2, is not the
    solver's)."""
    return augmented_inequality_objective(
        (x[0] - 3.0) ** 2 + (x[1] - 3.0) ** 2,
        [2.0 * (x[0] - 3.0), 2.0 * (x[1] - 3.0)],
        [x[1] - 1.0],
        [[0.0, 1.0]],
        multipliers,
        penalty,
    )


def released_row(x, multipliers, penalty):
    """min (x - 0.5)^2 s.t. x - 1 <= 0: x* = 0.5, lambda* = 0."""
    return augmented_inequality_objective(
        (x[0] - 0.5) ** 2, [2.0 * (x[0] - 0.5)], [x[0] - 1.0], [[1.0]], multipliers, penalty
    )


def reached_row(x, multipliers, penalty):
    """min (x - 2)^2 s.t. x - 1 <= 0: x* = 1, lambda* = 2."""
    return augmented_inequality_objective(
        (x[0] - 2.0) ** 2, [2.0 * (x[0] - 2.0)], [x[0] - 1.0], [[1.0]], multipliers, penalty
    )


def trading_rows(x, multipliers, penalty):
    """min (x0 - 2)^2 + (x1 - 2)^2 s.t. x0 + x1 - 2 <= 0, x0 - x1 - 1 <= 0:
    x* = (1, 1), lambda* = (2, 0). From (3, 0) both rows are violated; the
    second ends slack."""
    return augmented_inequality_objective(
        (x[0] - 2.0) ** 2 + (x[1] - 2.0) ** 2,
        [2.0 * (x[0] - 2.0), 2.0 * (x[1] - 2.0)],
        [x[0] + x[1] - 2.0, x[0] - x[1] - 1.0],
        [[1.0, 1.0], [1.0, -1.0]],
        multipliers,
        penalty,
    )


class Benchmark(NamedTuple):
    evaluate: object
    names: Sequence[str]
    x0: Sequence[float]
    x_star: Sequence[float]
    multipliers_star: Sequence[float]
    base_bounds: Optional[Sequence[tuple]] = None
    initial_multipliers: Optional[Sequence[float]] = None


class KKTBenchmarkTests(unittest.TestCase):
    def assert_kkt(self, benchmark: Benchmark):
        for variant, settings in SETTINGS_VARIANTS.items():
            with self.subTest(variant=variant):
                result = minimize_alm(
                    np.asarray(benchmark.x0, dtype=float),
                    list(benchmark.names),
                    benchmark.evaluate,
                    settings,
                    dict(INNER_OPTIONS),
                    base_bounds=benchmark.base_bounds,
                    initial_multipliers=(
                        None
                        if benchmark.initial_multipliers is None
                        else np.asarray(benchmark.initial_multipliers, dtype=float)
                    ),
                )
                self.assertEqual(result.termination_reason, "converged", result.message)
                self.assertTrue(result.success)
                self.assertLessEqual(result.max_violation, settings.feasibility_tol)
                np.testing.assert_allclose(result.x, benchmark.x_star, atol=X_ATOL)
                # converged certifies the shifted multipliers max(0, lambda + rho g).
                shifted = np.maximum(
                    0.0, result.multipliers + result.penalty * result.constraint_values
                )
                np.testing.assert_allclose(
                    shifted, benchmark.multipliers_star, atol=MULTIPLIER_ATOL
                )

    def test_feasible_start_whose_subproblem_minimizer_is_infeasible(self):
        # At lambda = 0, rho = 1 the subproblem minimizer 7/3 violates the row
        # by 4/3, more than the first feasibility slack rho^-0.1 = 1.
        self.assert_kkt(Benchmark(pulled_out, ["g"], [0.0], [1.0], [4.0]))

    def test_infeasible_start(self):
        self.assert_kkt(Benchmark(pulled_out, ["g"], [2.0], [1.0], [4.0]))

    def test_start_at_the_optimum(self):
        self.assert_kkt(Benchmark(pulled_out, ["g"], [1.0], [1.0], [4.0]))

    def test_start_at_the_kkt_pair(self):
        self.assert_kkt(
            Benchmark(pulled_out, ["g"], [1.0], [1.0], [4.0], initial_multipliers=[4.0])
        )

    def test_active_row_beside_a_slack_row(self):
        self.assert_kkt(
            Benchmark(active_beside_slack, ["active", "slack"], [0.0, 0.0], [1.0, 1.0], [4.0, 0.0])
        )

    def test_row_beside_a_bound(self):
        self.assert_kkt(
            Benchmark(
                row_beside_bound, ["row"], [0.0, 0.0], [2.0, 1.0], [4.0],
                base_bounds=[(-5.0, 2.0), (None, None)],
            )
        )

    def test_violated_row_that_ends_slack(self):
        self.assert_kkt(Benchmark(released_row, ["g"], [3.0], [0.5], [0.0]))

    def test_slack_row_that_ends_active(self):
        self.assert_kkt(Benchmark(reached_row, ["g"], [0.0], [1.0], [2.0]))

    def test_rows_that_trade_activity(self):
        self.assert_kkt(
            Benchmark(trading_rows, ["sum", "difference"], [3.0, 0.0], [1.0, 1.0], [2.0, 0.0])
        )


if __name__ == "__main__":
    unittest.main()
