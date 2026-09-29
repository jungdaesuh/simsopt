"""The toy example in the ``simsopt_alm`` package docstring runs as shown.

``_package_docstring_example`` is that example's code, written out here (the
docstring is never executed); one test pins the docstring to this code, the
other runs it and checks the output the docstring's comment promises.
"""

import contextlib
import inspect
import io
import sys
import textwrap
import unittest

import numpy as np
from simsopt_alm import (ALMPhysics, ALMSettings, cached_alm_evaluator,
                         minimize_alm)

import simsopt_alm as alm

EXAMPLE_IMPORTS = """\
import numpy as np
from simsopt_alm import (ALMPhysics, ALMSettings, cached_alm_evaluator,
                         minimize_alm)
"""


def _package_docstring_example():
    def physics(x):
        return ALMPhysics(
            base_value=float(x @ x),
            base_grad=2.0 * x,
            constraint_values=np.array([1.0 - x[0]]),
            constraint_grads=(np.array([-1.0, 0.0]),),
        )

    result = minimize_alm(np.array([3.0, 2.0]), ["x0_at_least_one"],
                          cached_alm_evaluator(physics), ALMSettings(),
                          {"maxiter": 200})
    print(result.termination_reason, result.x.round(4) + 0.0)  # converged [1. 0.]
    return result


def _docstring_example_block() -> str:
    """The indented literal block after the docstring's ``(solution x = (1, 0))::``."""
    lines = alm.__doc__.splitlines()
    start = next(i for i, line in enumerate(lines) if line.endswith("(1, 0))::")) + 2
    stop = next(
        i for i in range(start, len(lines)) if lines[i] and not lines[i].startswith(" ")
    )
    return textwrap.dedent("\n".join(lines[start:stop]).rstrip() + "\n")


class AlmPackageDocstringExampleTests(unittest.TestCase):
    def test_docstring_shows_exactly_this_code(self):
        body_lines = inspect.getsource(_package_docstring_example).splitlines(True)[1:]
        text = textwrap.dedent("".join(body_lines))
        suffix = "return result\n"
        body = text[:-len(suffix)] if text.endswith(suffix) else text

        self.assertIn(
            EXAMPLE_IMPORTS,
            inspect.getsource(sys.modules[__name__]),
            "EXAMPLE_IMPORTS must be this module's own import lines",
        )
        self.assertEqual(
            _docstring_example_block(),
            EXAMPLE_IMPORTS + "\n" + body,
            "the package docstring's toy example drifted from the tested code",
        )

    def test_toy_example_converges_to_the_documented_answer(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = _package_docstring_example()

        self.assertEqual(output.getvalue(), "converged [1. 0.]\n")
        self.assertTrue(result.success, result.message)
        np.testing.assert_allclose(result.x, [1.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(result.multipliers, [2.0], rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
