"""``simsopt_alm.hybrid``: the signal-mismatch arms, reachable only with
the hybrid quartet.

The hybrid module owns what a hard-feasible step does while the surrogate and
hard constraint signals disagree (repair, subproblem limit, penalty raise). Only ``DefaultContinuationPolicy`` calls it, and only on a signal
mismatch, which needs an evaluator that returns the hybrid quartet. So an
evaluator without the quartet never reaches a hybrid rule: with every hybrid
function made to fail, the non-quartet goldens still replay.
"""

import ast
import inspect
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from typing import List, Set
from unittest.mock import patch

from simsopt_alm import hybrid

GOLDEN_DIR = Path(__file__).resolve().parent / "alm_golden"
if str(GOLDEN_DIR) not in sys.path:
    sys.path.insert(0, str(GOLDEN_DIR))
import alm_golden_scenarios as golden  # noqa: E402

PACKAGE_ROOT = Path(hybrid.__file__).resolve().parent

# Goldens whose evaluators return no hybrid quartet.
NON_HYBRID_GOLDENS = (
    "toy_convex",
    "penalty_ramp_to_cap",
    "resume_penalty_ramp_mid_run",
    "plateau_restore_best_feasible",
    "resume_plateau_best_feasible",
    "trust_radius_retries",
    "frozen_warm_start_restore",
    "multiplier_cap_process_budget",
    "inner_iteration_budget",
    "dual_update_penalty_cap",
    "preinner_converged",
    "zero_inner_budget",
)


class HybridRuleReached(AssertionError):
    pass


def _hybrid_functions() -> List[str]:
    return sorted(
        name
        for name, value in vars(hybrid).items()
        if inspect.isfunction(value) and value.__module__ == hybrid.__name__
    )


def _all_hybrid_rules_fail(stack: ExitStack) -> None:
    for name in _hybrid_functions():

        def reached(*args, _name=name, **kwargs):
            raise HybridRuleReached(f"hybrid.{_name} ran without the hybrid quartet")

        stack.enter_context(patch.object(hybrid, name, reached))


class AlmHybridInertnessTests(unittest.TestCase):
    def test_non_hybrid_goldens_never_reach_a_hybrid_rule(self):
        for name in NON_HYBRID_GOLDENS:
            with self.subTest(scenario=name):
                with ExitStack() as stack:
                    _all_hybrid_rules_fail(stack)
                    trajectory = golden.SCENARIOS_BY_NAME[name].run()
                self.assertIsNone(golden.golden_difference(name, trajectory))

    def test_a_mismatch_golden_does_reach_the_hybrid_rules(self):
        # Positive control: the patch above does intercept the rules in use.
        with ExitStack() as stack:
            _all_hybrid_rules_fail(stack)
            with self.assertRaises(HybridRuleReached):
                golden.SCENARIOS_BY_NAME["hybrid_mismatch_penalty_increase"].run()


def _imported_modules(path: Path) -> Set[str]:
    """Package-relative and absolute module names ``path`` imports."""
    names: Set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = "." * node.level + (node.module or "")
            separator = "" if base.endswith(".") else "."
            names.add(base)
            names.update(base + separator + alias.name for alias in node.names)
    return names


def _imports_hybrid(path: Path) -> bool:
    return any(
        name.split(".")[-1] == "hybrid" for name in _imported_modules(path)
    )


class AlmHybridImportBoundaryTests(unittest.TestCase):
    def test_only_the_default_policy_module_imports_hybrid(self):
        importers = sorted(
            path.name
            for path in PACKAGE_ROOT.glob("*.py")
            if path.name != "hybrid.py" and _imports_hybrid(path)
        )
        self.assertEqual(importers, ["policy.py"])

    def test_hybrid_imports_neither_the_loop_nor_a_plugin(self):
        imported = _imported_modules(PACKAGE_ROOT / "hybrid.py")
        for forbidden in (".control", ".policy", ".history", ".checkpoint"):
            with self.subTest(module=forbidden):
                self.assertNotIn(forbidden, imported)


if __name__ == "__main__":
    unittest.main()
