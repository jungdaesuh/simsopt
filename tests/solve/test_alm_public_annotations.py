"""Every public annotation of ``simsopt.solve.alm``, its opt-in plugins and the
library examples resolves with ``typing.get_type_hints``.

The library examples are the ALM example code ``tests/solve`` imports. A caller
that introspects the public API (a type checker, a ``TypedDict`` consumer,
documentation tooling) needs every name in those annotations to exist at
runtime, on every supported Python.
"""

import inspect
import sys
import typing
import unittest
from pathlib import Path

import simsopt.solve.alm as alm
from simsopt.solve.alm import checkpoint as alm_checkpoint
from simsopt.solve.alm import continuation as alm_continuation
from simsopt.solve.alm import history as alm_history
from simsopt.solve.alm import policy as alm_policy

LIBRARY_EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples" / "2_Intermediate"
if str(LIBRARY_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(LIBRARY_EXAMPLES_DIR))

import alm_composition_example  # noqa: E402
import alm_typed_evaluator_example  # noqa: E402
import boozerQA_alm  # noqa: E402  (module scope builds no problem)

EXAMPLE_MODULES = (
    alm_composition_example,
    alm_typed_evaluator_example,
    boozerQA_alm,
)


def _public_annotated_objects():
    """``(label, object)`` for each public class, its public methods and each
    public function defined by the package, its plugin modules or the examples."""
    modules = (alm, alm_checkpoint, alm_continuation, alm_history, alm_policy,
               *EXAMPLE_MODULES)
    owners = {module.__name__ for module in EXAMPLE_MODULES}
    seen = set()
    for module in modules:
        for name in sorted(vars(module)):
            value = getattr(module, name)
            if name.startswith("_") or id(value) in seen:
                continue
            if not (inspect.isclass(value) or inspect.isfunction(value)):
                continue
            owner = getattr(value, "__module__", "")
            if not (owner.startswith("simsopt.solve.alm") or owner in owners):
                continue
            seen.add(id(value))
            yield f"{value.__module__}.{name}", value
            if inspect.isclass(value):
                for method_name, method in vars(value).items():
                    if inspect.isfunction(method) and not method_name.startswith("_"):
                        yield f"{value.__module__}.{name}.{method_name}", method


class AlmPublicAnnotationTests(unittest.TestCase):
    def test_every_public_annotation_resolves(self):
        labels = []
        for label, value in _public_annotated_objects():
            with self.subTest(label=label):
                typing.get_type_hints(value)
            labels.append(label)
        self.assertIn("simsopt.solve.alm.control.minimize_alm", labels)
        self.assertIn("simsopt.solve.alm.checkpoint.ALMTransitionSnapshot", labels)
        self.assertIn("simsopt.solve.alm.history.ALMHistoryRecorder", labels)
        self.assertIn("boozerQA_alm.BoozerQAProblem", labels)


if __name__ == "__main__":
    unittest.main()
