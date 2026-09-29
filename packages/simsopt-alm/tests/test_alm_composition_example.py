"""``examples/alm_composition_example.py`` runs and shows what it says.

The example composes the library solver with the history recorder, checkpoint
and resume, and a stateful evaluator's snapshot/restore. Its ``main()`` returns
what it demonstrated; this test runs it and checks those claims: the resumed
run ends bitwise where the uninterrupted one did, the solver restored the
evaluator's own state before resuming, and the history covers the resumed part.
"""

import ast
import contextlib
import io
import sys
import unittest
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

import alm_composition_example  # noqa: E402


class AlmCompositionExampleTests(unittest.TestCase):
    def test_the_example_resumes_bitwise_with_the_evaluator_state_restored(self):
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            summary = alm_composition_example.main()

        self.assertTrue(summary["uninterrupted_success"], summary)
        self.assertGreaterEqual(summary["checkpoints"], 3)
        self.assertGreater(summary["resumed_from_outer"], 0)
        # The state matters: a cold evaluator would solve on another branch.
        self.assertTrue(summary["cold_start_finds_another_branch"], summary)
        self.assertTrue(summary["resumed_matches_uninterrupted_bitwise"], summary)
        self.assertTrue(summary["evaluator_state_restored_before_resume"], summary)
        self.assertGreater(summary["resumed_history_entries"], 0)
        self.assertIn("resumed run matches the uninterrupted run bit for bit", printed.getvalue())

    def test_the_example_imports_only_the_solver(self):
        tree = ast.parse(
            (EXAMPLES_DIR / "alm_composition_example.py").read_text(encoding="utf-8")
        )
        simsopt_modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module.startswith("simsopt")
        } | {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
            if alias.name.startswith("simsopt")
        }
        self.assertEqual(
            sorted(simsopt_modules),
            ["simsopt_alm", "simsopt_alm.checkpoint", "simsopt_alm.history"],
        )


if __name__ == "__main__":
    unittest.main()
