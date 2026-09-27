#!/usr/bin/env python
"""Check the sign convention (``g <= 0`` is feasible) of every constraint row
of the problem in ``alm_problem.py`` at its sign probes, and the scale of
every row at ``problem.x0``.

    PYTHONPATH=<problem dir> python sign_check.py [--smoke]

Signs: ``problem.sign_probes()`` gives points where rows are known, from the
physics and independently of the row code, to be violated or satisfied. Each
listed row must have ``g > 0`` (violated) or ``g <= 0`` (satisfied) there; a
mismatch, a non-finite value or a probe naming an unknown row fails. A row
never probed on one side is reported as a coverage warning. A row in
``problem.shared_source_rows`` has probe expectations that read the same
source as the row (e.g. the Boozer template's iota from the solve itself):
its probes check the sign and the bound, not the value, and a note says so.

Scales, at ``problem.x0`` (warnings, not failures): the constraint gradient
norms must not spread over more than ``SPREAD_LIMIT`` (the penalty is shared
by all rows, so a row with a far larger gradient dominates the subproblem);
no row value may exceed ``ORDER_ONE_LIMIT`` in magnitude; |f(x0)| should lie
within [1 / ORDER_ONE_LIMIT, ORDER_ONE_LIMIT]; and a row with a zero gradient
cannot be moved. The last line printed is ``SIGN_CHECK {json}``; the exit
status is 0 when no sign fails.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from alm_problem import build_problem

RESULT_PREFIX = "SIGN_CHECK "
SPREAD_LIMIT = 1e3
ORDER_ONE_LIMIT = 1e3


def check_probe(probe, physics, constraint_names) -> dict:
    values = dict(zip(constraint_names, physics.constraint_values.tolist()))
    failures = []
    for name in probe.violated + probe.satisfied:
        if name not in values:
            failures.append(f"{name}: not a row of this problem")
        elif not np.isfinite(values[name]):
            failures.append(f"{name}: non-finite value {values[name]}")
    for name in probe.violated:
        if name in values and np.isfinite(values[name]) and not values[name] > 0.0:
            failures.append(f"{name}: expected violated (g > 0), got g = {values[name]:+.4e}")
    for name in probe.satisfied:
        if name in values and np.isfinite(values[name]) and not values[name] <= 0.0:
            failures.append(f"{name}: expected satisfied (g <= 0), got g = {values[name]:+.4e}")
    return {"label": probe.label, "passed": not failures, "failures": failures,
            "violated": list(probe.violated), "satisfied": list(probe.satisfied), "values": values}


def check_scales(physics, constraint_names) -> dict:
    values = np.asarray(physics.constraint_values, dtype=float)
    gradient_norms = np.array([np.linalg.norm(grad) for grad in physics.constraint_grads])
    nonzero_norms = gradient_norms[gradient_norms > 0.0]
    spread = float(nonzero_norms.max() / nonzero_norms.min()) if nonzero_norms.size else None
    objective = abs(float(physics.base_value))
    warnings = []
    if spread is not None and spread > SPREAD_LIMIT:
        warnings.append(f"constraint gradient norms spread over {spread:.2e} (> {SPREAD_LIMIT:.0e}); "
                        "rescale the rows (divide each by its bound or typical size)")
    for name, value, norm in zip(constraint_names, values, gradient_norms):
        if abs(value) > ORDER_ONE_LIMIT:
            warnings.append(f"{name}: |g(x0)| = {abs(value):.2e} is not O(1); divide the row by its scale")
        if norm == 0.0:
            warnings.append(f"{name}: zero gradient at x0; the solver cannot move this row")
    if not 1.0 / ORDER_ONE_LIMIT <= objective <= ORDER_ONE_LIMIT:
        warnings.append(f"|f(x0)| = {objective:.2e} is not O(1); divide f by a reference value "
                        "so the stationarity tolerance means something")
    return {
        "objective": float(physics.base_value),
        "objective_gradient_norm": float(np.linalg.norm(physics.base_grad)),
        "rows": {name: {"value": float(value), "gradient_norm": float(norm)}
                 for name, value, norm in zip(constraint_names, values, gradient_norms)},
        "gradient_norm_spread": spread,
        "warnings": warnings,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--smoke", action="store_true", help="build the problem at its smoke size")
    args = parser.parse_args(argv)

    problem = build_problem(smoke=args.smoke)
    names = tuple(problem.constraint_names)
    probes = [check_probe(probe, problem.physics(probe.x), names) for probe in problem.sign_probes()]
    probed_violated = {name for probe in probes for name in probe["violated"]}
    probed_satisfied = {name for probe in probes for name in probe["satisfied"]}
    coverage_warnings = (
        [f"{name}: never probed where it is violated" for name in names if name not in probed_violated]
        + [f"{name}: never probed where it is satisfied" for name in names if name not in probed_satisfied]
    )
    scales = check_scales(problem.physics(problem.x0), names)
    passed = all(probe["passed"] for probe in probes)
    shared_source_notes = [
        f"{name}: its probe expectations read the same source as the row, so they check its sign "
        "and bound, not its value" for name in problem.shared_source_rows
    ]

    for probe in probes:
        print(f"probe {probe['label']!r}: {'PASS' if probe['passed'] else 'FAIL'}")
        for failure in probe["failures"]:
            print(f"    {failure}")
    for warning in coverage_warnings + scales["warnings"]:
        print(f"warning: {warning}")
    for note in shared_source_notes:
        print(f"note: {note}")
    print(RESULT_PREFIX + json.dumps({
        "passed": passed,
        "problem": problem.name,
        "probes": probes,
        "coverage_warnings": coverage_warnings,
        "shared_source_rows": list(problem.shared_source_rows),
        "scales": scales,
    }))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
