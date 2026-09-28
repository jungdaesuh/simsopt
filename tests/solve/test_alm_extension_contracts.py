"""The contracts an outside evaluator, cache or policy author codes against.

``ALMEvaluation`` is the evaluator's dict as a ``TypedDict`` (required keys
by inheritance, the rest optional; Python 3.9 has no ``Required``), and
``ALMEvaluator`` / ``CachedALMEvaluator`` the callables; ``minimize_alm`` and
``cached_alm_evaluator`` are annotated with them. The all-or-none hybrid
quartet stays a runtime check: four optional keys cannot say "all or none".
The generated API docs cover the plugin and policy modules.
"""

import ast
import collections.abc
import contextlib
import io
import re
import sys
import typing
import unittest
from pathlib import Path

import numpy as np

import simsopt.solve.alm as alm
from simsopt.solve.alm import boundary, checkpoint, continuation, control, core
from simsopt.solve.alm import evaluation, events, history, hybrid, inner, policy
from simsopt.solve.alm import problem, taylor

DOCS_ALM_RST = Path(__file__).resolve().parents[2] / "docs" / "source" / "simsopt.solve.alm.rst"
EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples" / "2_Intermediate"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

import alm_typed_evaluator_example as typed_example  # noqa: E402

ALM_MODULES = (alm, boundary, checkpoint, continuation, control, core, evaluation,
               events, history, hybrid, inner, policy, problem, taylor)

REQUIRED_KEYS = {
    "total",
    "grad",
    "constraint_values",
    "feasibility_values",
    "dual_update_values",
    "constraint_grads",
}
HYBRID_QUARTET = {
    "hard_signed_constraint_values",
    "hard_violation_values",
    "surrogate_signed_constraint_values",
    "hard_dual_update_values",
}


def _halfspace_evaluation(x, multipliers, penalty):
    return alm.augmented_inequality_objective(
        float(x @ x), 2.0 * x, np.array([1.0 - x[0]]), [np.array([-1.0, 0.0])],
        multipliers, penalty,
    )


# ``typing.TypedDict`` records ``__required_keys__`` / ``__optional_keys__``
# from Python 3.9; on 3.8 only the declared keys (the type hints) are visible.
requires_typed_dict_key_split = unittest.skipIf(
    sys.version_info < (3, 9),
    "typing.TypedDict records required and optional keys from Python 3.9",
)


class AlmEvaluationSchemaTests(unittest.TestCase):
    @requires_typed_dict_key_split
    def test_the_schema_separates_required_from_optional_keys(self):
        schema = alm.ALMEvaluation
        self.assertEqual(set(schema.__required_keys__), REQUIRED_KEYS)
        self.assertLessEqual(
            HYBRID_QUARTET
            | {"base_value", "base_total", "physics_total", "base_grad", "metric_grad",
               "stationarity_norm", "constraint_scales", "constraint_activity_tolerances",
               "nonfinite_evaluation", "search_step_success"},
            set(schema.__optional_keys__),
        )

    def test_augmented_inequality_objective_returns_every_required_key(self):
        evaluation = _halfspace_evaluation(np.array([3.0, 2.0]), np.zeros(1), 1.0)
        self.assertLessEqual(REQUIRED_KEYS, set(evaluation))

    def test_a_partial_hybrid_quartet_is_still_rejected_at_runtime(self):
        def three_of_four(x, multipliers, penalty):
            evaluation = _halfspace_evaluation(x, multipliers, penalty)
            evaluation.update(
                hard_signed_constraint_values=evaluation["constraint_values"],
                hard_violation_values=evaluation["feasibility_values"],
                surrogate_signed_constraint_values=evaluation["constraint_values"],
            )
            return evaluation

        with self.assertRaisesRegex(KeyError, "hard_dual_update_values"):
            alm.minimize_alm(
                np.array([3.0, 2.0]), ["x0_at_least_one"], three_of_four,
                alm.ALMSettings(), {"maxiter": 50},
            )


def _solve(evaluate_problem, *, maxiter=50):
    return alm.minimize_alm(
        np.array([3.0, 2.0]), ["x0_at_least_one"], evaluate_problem,
        alm.ALMSettings(max_outer_iterations=3), {"maxiter": maxiter},
    )


def _edited(**edits):
    """The half-space evaluator with ``edits`` applied to every evaluation
    (a value of ``...`` deletes the key)."""

    def evaluate(x, multipliers, penalty):
        evaluation = _halfspace_evaluation(x, multipliers, penalty)
        for key, value in edits.items():
            if value is ...:
                del evaluation[key]
            else:
                evaluation[key] = value
        return evaluation

    return evaluate


def _trial_steps_flagged(flag):
    """The half-space evaluator whose every trial point away from the start
    reports ``search_step_success=flag`` (its evaluation is otherwise exact)."""
    start = np.array([3.0, 2.0])

    def evaluate(x, multipliers, penalty):
        evaluation = _halfspace_evaluation(x, multipliers, penalty)
        if not np.array_equal(x, start):
            evaluation["search_step_success"] = flag
        return evaluation

    return evaluate


class AlmEvaluationBoundaryTests(unittest.TestCase):
    """The evaluator's dict is checked where it enters the solver, at the
    outer iterate and at every inner trial point."""

    def test_a_missing_or_null_required_key_is_named(self):
        for key in sorted(REQUIRED_KEYS):
            for value in (..., None):
                with self.subTest(key=key, value=value):
                    with self.assertRaisesRegex(KeyError, key):
                        _solve(_edited(**{key: value}))

    def test_constraint_gradients_need_one_row_per_constraint(self):
        for rows in ([], [np.array([-1.0, 0.0])] * 2):
            with self.subTest(rows=len(rows)):
                with self.assertRaisesRegex(ValueError, "constraint_grads"):
                    _solve(_edited(constraint_grads=rows))

    def test_constraint_gradient_rows_are_shaped_like_x(self):
        for row in (np.array([-1.0]), np.array([-1.0, 0.0, 0.0]), np.array([[-1.0, 0.0]])):
            with self.subTest(shape=row.shape):
                with self.assertRaisesRegex(ValueError, "constraint_grads"):
                    _solve(_edited(constraint_grads=[row]))

    def test_a_trial_point_evaluation_is_checked_too(self):
        def missing_rows_away_from_start(x, multipliers, penalty):
            evaluation = _halfspace_evaluation(x, multipliers, penalty)
            if not np.array_equal(x, [3.0, 2.0]):
                del evaluation["constraint_grads"]
            return evaluation

        with self.assertRaisesRegex(KeyError, "constraint_grads"):
            _solve(missing_rows_away_from_start)

    def test_a_numpy_false_search_flag_rejects_the_trial_step(self):
        rejected = _solve(_trial_steps_flagged(False))
        np.testing.assert_array_equal(rejected.x, [3.0, 2.0])
        numpy_rejected = _solve(_trial_steps_flagged(np.bool_(False)))
        np.testing.assert_array_equal(numpy_rejected.x, rejected.x)
        self.assertEqual(numpy_rejected.termination_reason, rejected.termination_reason)

    def test_a_numpy_true_search_flag_accepts_the_trial_step(self):
        accepted = _solve(_trial_steps_flagged(True))
        numpy_accepted = _solve(_trial_steps_flagged(np.bool_(True)))
        self.assertFalse(np.array_equal(accepted.x, [3.0, 2.0]))
        np.testing.assert_array_equal(numpy_accepted.x, accepted.x)

    def test_a_search_flag_that_is_not_a_bool_is_rejected(self):
        for flag in (0, 1, None, "False", 0.0):
            with self.subTest(flag=flag):
                with self.assertRaisesRegex(ValueError, "search_step_success"):
                    _solve(_trial_steps_flagged(flag))


def _declared_evaluation_keys():
    return set(typing.get_type_hints(alm.ALMEvaluation))


# A dict the solver reads as an evaluation: ``evaluation``, ``candidate_eval``,
# ``measured.evaluation``, ``state.final_eval``, ...
_EVALUATION_NAME = re.compile(r"(?:\w+_)?eval(?:uation)?")


def _is_evaluation(node: ast.expr) -> bool:
    if isinstance(node, ast.Name):
        name = node.id
    elif isinstance(node, ast.Attribute):
        name = node.attr
    else:
        return False
    return _EVALUATION_NAME.fullmatch(name) is not None


def _string_members(node, module, function):
    """The strings of a field tuple: a literal, a local of ``function`` or a
    module constant."""
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        members = set()
        for element in node.elts:
            if isinstance(element, ast.Constant) and isinstance(element.value, str):
                members.add(element.value)
            elif isinstance(element, ast.Starred):
                members |= _string_members(element.value, module, function)
        return members
    if isinstance(node, ast.Call) and node.args:  # frozenset((...))
        return _string_members(node.args[0], module, function)
    if isinstance(node, ast.Name):
        for assignment in ast.walk(function) if function is not None else ():
            if isinstance(assignment, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == node.id
                for target in assignment.targets
            ):
                return _string_members(assignment.value, module, function)
        value = getattr(module, node.id, None)
        if isinstance(value, (tuple, frozenset)) and all(isinstance(v, str) for v in value):
            return set(value)
    return set()


def _key_reads(tree):
    """``(dict_node, key_node)`` for each dict read inside ``tree``."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load):
            key = node.slice
            if sys.version_info < (3, 9) and isinstance(key, ast.Index):
                key = key.value  # Python 3.8 wraps a subscript key in ast.Index
            yield node.value, key
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
              and node.func.attr in ("get", "pop") and node.args):
            yield node.func.value, node.args[0]
        elif (isinstance(node, ast.Compare) and len(node.ops) == 1
              and isinstance(node.ops[0], (ast.In, ast.NotIn))):
            yield node.comparators[0], node.left
        elif isinstance(node, ast.Call):
            # helper(evaluation, "key", ...)
            yield from zip(node.args, node.args[1:])


def _field_loops(tree):
    """``(loop_variable, iterable, body_nodes)`` for each ``for`` statement and
    comprehension, the body being where the variable is in scope."""
    for node in ast.walk(tree):
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name):
            yield node.target.id, node.iter, node.body
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            body = [node.key, node.value] if isinstance(node, ast.DictComp) else [node.elt]
            for generator in node.generators:
                if isinstance(generator.target, ast.Name):
                    yield generator.target.id, generator.iter, [*body, *generator.ifs]


def _solver_read_evaluation_keys():
    """Every evaluation key the ALM package reads, by name or through a field
    tuple it loops over: ``{key: [module:line, ...]}``."""
    reads = {}

    def record(keys, path, node):
        for name in keys:
            reads.setdefault(name, []).append(f"{path.name}:{node.lineno}")

    for module in ALM_MODULES:
        path = Path(module.__file__)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for container, key in _key_reads(tree):
            if (_is_evaluation(container) and isinstance(key, ast.Constant)
                    and isinstance(key.value, str)):
                record({key.value}, path, key)
        functions = [node for node in ast.walk(tree)
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        for function in [None, *functions]:
            loops = _field_loops(function if function is not None else tree)
            for variable, iterable, body in loops:
                members = _string_members(iterable, module, function)
                for statement in body:
                    for container, key in _key_reads(statement):
                        if (_is_evaluation(container) and isinstance(key, ast.Name)
                                and key.id == variable):
                            record(members, path, key)
    return reads


class AlmEvaluationSchemaCompletenessTests(unittest.TestCase):
    """``ALMEvaluation`` declares every field the solver owns: what the
    canonical builders emit, what the loop reads, copies or strips. Keys an
    application adds beyond these stay undeclared diagnostics."""

    def test_the_package_module_list_is_complete(self):
        on_disk = {path.stem for path in Path(alm.__file__).parent.glob("*.py")}
        scanned = {Path(module.__file__).stem for module in ALM_MODULES}
        self.assertEqual(scanned, on_disk)

    def test_every_canonical_builder_key_is_declared(self):
        builders = {
            "augmented_inequality_objective": _halfspace_evaluation(
                np.array([3.0, 2.0]), np.array([0.5]), 2.0),
            "ALMPhysics.evaluation": alm.ALMPhysics(
                base_value=13.0, base_grad=np.array([6.0, 4.0]),
                constraint_values=np.array([-2.0]),
                constraint_grads=(np.array([-1.0, 0.0]),),
            ).evaluation(np.array([0.5]), 2.0),
        }
        for builder, output in builders.items():
            with self.subTest(builder=builder):
                self.assertEqual(set(output) - _declared_evaluation_keys(), set())

    def test_every_field_the_solver_copies_or_strips_is_declared(self):
        for label, fields in (
            ("evaluation._OWNED_EVALUATION_ARRAY_FIELDS", evaluation._OWNED_EVALUATION_ARRAY_FIELDS),
            ("inner._INNER_CACHE_OWNED_KEYS", inner._INNER_CACHE_OWNED_KEYS),
            ("checkpoint._TRANSITION_OMITTED_JACOBIAN_KEYS",
             checkpoint._TRANSITION_OMITTED_JACOBIAN_KEYS),
        ):
            with self.subTest(fields=label):
                self.assertEqual(set(fields) - _declared_evaluation_keys(), set())

    def test_every_key_the_solver_reads_is_declared(self):
        reads = _solver_read_evaluation_keys()
        # The scan must see the reads round 3 named, or it proves nothing.
        self.assertLessEqual(
            {"metric_stationarity_norm", "max_violation", "positive_shift_values",
             "augmented_term_by_constraint", "raw_hard_violation_values", "total"},
            set(reads),
        )
        undeclared = {key: sites for key, sites in reads.items()
                      if key not in _declared_evaluation_keys()}
        self.assertEqual(undeclared, {}, "the solver reads these evaluation keys; "
                         "declare them in ALMEvaluation")

    def test_each_declared_field_has_its_runtime_type(self):
        # The builder's floats are floats and its arrays ndarrays, as declared.
        output = _halfspace_evaluation(np.array([3.0, 2.0]), np.array([0.5]), 2.0)
        hints = typing.get_type_hints(alm.ALMEvaluation)
        for key, value in output.items():
            with self.subTest(key=key):
                declared = hints[key]
                if declared is float:
                    self.assertIsInstance(value, float)
                elif typing.get_origin(declared) is typing.Union:
                    self.assertIsInstance(value, np.ndarray)
                    self.assertIn(np.ndarray, typing.get_args(declared))
                elif declared is np.ndarray:
                    self.assertIsInstance(value, np.ndarray)
                else:
                    self.assertEqual(typing.get_origin(declared), typing.get_origin(
                        typing.Sequence[np.ndarray]))
                    self.assertTrue(all(isinstance(v, np.ndarray) for v in value))


    def test_list_valued_constraint_scales_match_the_declared_type(self):
        # Callers may pass constraint_scales as a list through
        # ALMPhysics extras, and the solver accepts it via np.asarray; the
        # schema must admit that representation, not only ndarray.
        physics = alm.ALMPhysics(
            base_value=0.0,
            base_grad=np.zeros(1),
            constraint_values=np.array([1.0]),
            constraint_grads=(np.array([1.0]),),
            extras={"constraint_scales": [2.0]},
        )
        value = physics.evaluation(np.zeros(1), 1.0)["constraint_scales"]
        declared = typing.get_type_hints(alm.ALMEvaluation)["constraint_scales"]

        self.assertIsInstance(value, list)
        self.assertIn(np.ndarray, typing.get_args(declared))
        self.assertTrue(
            any(typing.get_origin(arg) is collections.abc.Sequence
                for arg in typing.get_args(declared)),
            f"constraint_scales is declared {declared}, but a list is a supported value",
        )


class AlmTypedEvaluatorExampleTests(unittest.TestCase):
    """``examples/2_Intermediate/alm_typed_evaluator_example.py``: an evaluator
    typed against the schema, with optional keys and an application key
    declared by subclassing it."""

    def test_the_example_solves_and_keeps_its_own_diagnostic(self):
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            result = typed_example.main()
        self.assertEqual(result.termination_reason, "converged")
        np.testing.assert_allclose(result.x, [1.0, 0.0], atol=1e-5)
        self.assertIn("distance_to_boundary", result.evaluation)
        self.assertIn("distance to the boundary", printed.getvalue())

    def test_the_subclass_extends_the_schema(self):
        subclass = typed_example.HalfspaceEvaluation
        self.assertLess(_declared_evaluation_keys(), set(typing.get_type_hints(subclass)))
        self.assertIs(typing.get_type_hints(typed_example.evaluate_halfspace)["return"], subclass)

    @requires_typed_dict_key_split
    def test_the_subclass_keeps_the_required_keys(self):
        subclass = typed_example.HalfspaceEvaluation
        self.assertEqual(set(subclass.__required_keys__), REQUIRED_KEYS)
        self.assertEqual(set(subclass.__required_keys__) | set(subclass.__optional_keys__),
                         set(typing.get_type_hints(subclass)))

    def test_the_evaluator_returns_exactly_the_declared_keys_with_their_types(self):
        # What a type checker verifies statically, checked at run time.
        output = typed_example.evaluate_halfspace(np.array([3.0, 2.0]), np.array([0.5]), 2.0)
        hints = typing.get_type_hints(typed_example.HalfspaceEvaluation)
        self.assertLessEqual(set(output), set(hints))
        self.assertIn("metric_stationarity_norm", output)
        for key, value in output.items():
            with self.subTest(key=key):
                if hints[key] is float:
                    self.assertIsInstance(value, float)
                elif hints[key] is np.ndarray:
                    self.assertIsInstance(value, np.ndarray)


class AlmEvaluatorProtocolTests(unittest.TestCase):
    def test_the_entry_points_are_annotated_with_the_protocols(self):
        self.assertIs(
            typing.get_type_hints(alm.minimize_alm)["evaluate_problem"], alm.ALMEvaluator
        )
        self.assertIs(
            typing.get_type_hints(alm.cached_alm_evaluator)["return"], alm.CachedALMEvaluator
        )
        self.assertIs(
            typing.get_type_hints(alm.ALMEvaluator.__call__)["return"], alm.ALMEvaluation
        )
        self.assertIn("cache_clear", vars(alm.CachedALMEvaluator))
        self.assertIn(alm.ALMEvaluator, alm.CachedALMEvaluator.__mro__)

    def test_the_cached_evaluator_has_what_its_protocol_promises(self):
        def physics(x):
            return alm.ALMPhysics(
                base_value=float(x @ x), base_grad=2.0 * x,
                constraint_values=np.array([1.0 - x[0]]),
                constraint_grads=(np.array([-1.0, 0.0]),),
            )

        evaluator = alm.cached_alm_evaluator(physics)
        evaluation = evaluator(np.array([3.0, 2.0]), np.zeros(1), 1.0)
        self.assertLessEqual(REQUIRED_KEYS, set(evaluation))
        evaluator.cache_clear()

    def test_the_contracts_are_public(self):
        for name in ("ALMEvaluation", "ALMEvaluator", "CachedALMEvaluator"):
            with self.subTest(name=name):
                self.assertIn(name, alm.__all__)


class AlmApiDocumentationTests(unittest.TestCase):
    def test_the_generated_docs_cover_the_extension_modules(self):
        text = DOCS_ALM_RST.read_text(encoding="utf-8")
        for module in (
            "simsopt.solve.alm",
            "simsopt.solve.alm.policy",
            "simsopt.solve.alm.continuation",
            "simsopt.solve.alm.events",
            "simsopt.solve.alm.boundary",
            "simsopt.solve.alm.evaluation",
            "simsopt.solve.alm.problem",
            "simsopt.solve.alm.history",
            "simsopt.solve.alm.checkpoint",
        ):
            with self.subTest(module=module):
                self.assertIn(f".. automodule:: {module}\n", text)

    def test_the_solve_page_links_the_alm_page(self):
        text = (DOCS_ALM_RST.parent / "simsopt.solve.rst").read_text(encoding="utf-8")
        self.assertIn(".. toctree::", text)
        self.assertIn("\n   simsopt.solve.alm\n", text)


if __name__ == "__main__":
    unittest.main()
