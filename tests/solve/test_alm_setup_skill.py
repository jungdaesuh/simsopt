"""The simsopt-alm-setup skill (``.claude/skills/simsopt-alm-setup``) against
the package it documents.

Drift guards: every API name the skill's references, templates and scripts
use resolves in ``simsopt.solve.alm`` or the module the skill names; the
termination reasons of ``references/termination.md`` are exactly those the
package source can return (and ``success`` exactly its converge decisions);
``references/settings.md`` lists every ``ALMSettings`` field with its default;
the Python floor and the branch base match upstream; and
``docs/alm_setup_guide.md`` is what ``build_guide.py`` generates.

Runs: the scripts and templates run in fresh interpreters, in scratch
directories (a template's ``finish`` writes output), with this tree's ``src``
first on ``PYTHONPATH``. A template runs as the skill generates it: copied to
``alm_problem.py`` next to ``run_alm.py``.
"""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from dataclasses import MISSING, fields
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np

import simsopt.geo.signed_constraints as signed_constraints
import simsopt.solve.alm as alm
from simsopt.solve.alm import ALMResult, ALMSettings, checkpoint, control, history, policy

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
PACKAGE_DIR = SRC_DIR / "simsopt" / "solve" / "alm"
SKILL_DIR = REPO_ROOT / ".claude" / "skills" / "simsopt-alm-setup"
SCRIPTS_DIR = SKILL_DIR / "scripts"
TEMPLATES_DIR = SKILL_DIR / "templates"
REFERENCES_DIR = SKILL_DIR / "references"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import build_guide  # noqa: E402  (the skill's scripts; neither imports alm_problem)
import check_env  # noqa: E402

# Every module the skill may name, keyed by its dotted path.
MODULES = {
    "simsopt.solve.alm": alm,
    "simsopt.solve.alm.checkpoint": checkpoint,
    "simsopt.solve.alm.control": control,
    "simsopt.solve.alm.history": history,
    "simsopt.solve.alm.policy": policy,
    "simsopt.geo.signed_constraints": signed_constraints,
}
TERMINATION_KEYWORDS = {"termination_reason", "restored_termination_reason", "max_outer_termination"}
# Loop carriers whose string defaults are termination reasons.
TERMINATION_CARRIERS = {"max_outer_termination", "exhausted_termination"}
# Calls whose ``action=`` becomes the reason when the inner budget runs out.
BUDGET_ACTION_CALLS = {"_exhausted_termination", "ALMRaisePenalty"}


# A problem module whose claimed gradients are exact or wrong by MODE_VALUE:
# f = sum(x^3) and one row sum(sin x) (scaled by 1e-7 for "small"), or
# constants for "constant". The must-fail modes are the Crucible repros.
GRADIENT_PROBE_PROBLEM = """\
from types import SimpleNamespace

import numpy as np

from simsopt.solve.alm import ALMPhysics

MODE = MODE_VALUE


def physics(x):
    x = np.asarray(x, dtype=float)
    f, grad_f = float(np.sum(x ** 3)), 3.0 * x ** 2
    row, grad_row = float(np.sum(np.sin(x))), np.cos(x)
    if MODE == "small":
        row, grad_row = 1e-7 * row, 1e-7 * grad_row
    if MODE == "constant":
        f, grad_f, row, grad_row = 1.5, np.zeros_like(x), 0.25, np.zeros_like(x)
    claimed = {
        "exact": (grad_f, grad_row),
        "constant": (grad_f, grad_row),
        "zero": (np.zeros_like(x), np.zeros_like(x)),
        "factor": (1.5 * grad_f, 2.0 * grad_row),
        "small": (grad_f, 2.0 * grad_row),
    }[MODE]
    return ALMPhysics(base_value=f, base_grad=claimed[0], constraint_values=np.array([row]),
                      constraint_grads=(claimed[1],))


def build_problem(smoke=False):
    return SimpleNamespace(name="probe", x0=np.array([0.3, 0.7, 1.1]), physics=physics,
                           constraint_names=("row",), taylor_epsilons=None)
"""


# f alone, for the round-off cases of Crucible round 2: a huge value with a
# small derivative (claimed 0 or right), and a derivative A near the Taylor
# test's absolute 1e-10 floor (claimed 0 or right).
ROUNDOFF_PROBE_PROBLEM = """\
from types import SimpleNamespace

import numpy as np

from simsopt.solve.alm import ALMPhysics

CASE, N, A = CASE_VALUE, N_VALUE, A_VALUE


def physics(x):
    x = np.asarray(x, dtype=float)
    value, grad = {
        "big_value_zero_grad": (1e8 + 1e-3 * x.sum(), np.zeros_like(x)),
        "big_value_right_grad": (1e8 + 1e-3 * x.sum(), 1e-3 * np.ones_like(x)),
        "straddle_zero_grad": (A * x.sum() + 2e-5 * (x ** 3).sum(), np.zeros_like(x)),
        "straddle_right_grad": (A * x.sum() + 2e-5 * (x ** 3).sum(), A + 6e-5 * x ** 2),
    }[CASE]
    return ALMPhysics(base_value=float(value), base_grad=grad, constraint_values=np.zeros(0),
                      constraint_grads=())


def build_problem(smoke=False):
    return SimpleNamespace(name=CASE, x0=np.zeros(N), physics=physics, constraint_names=(),
                           taylor_epsilons=None)
"""


# Crucible round 3: one row per (case, parameter, relative gradient error
# DELTA). "cubic": q = sum(x) + K sum(x^3) at x0 = 0, where truncation
# K e^2 dominates the differences; "offset": q = F + sum(x), where round-off
# of F dominates. The claimed gradient is (1 + DELTA) times the true one.
SWEEP_CUBIC_K = (1e2, 1e6, 1e10)
SWEEP_OFFSET_F = (1.0, 1e4, 1e8)
SWEEP_DELTAS = (0.05, 0.03, -0.008, 3e-3, 1e-3)
SWEEP_PROBE_PROBLEM = """\
from types import SimpleNamespace

import numpy as np

from simsopt.solve.alm import ALMPhysics

ROWS = ROWS_VALUE


def row(x, case, parameter, delta):
    true_grad = 1.0 + (3.0 * parameter * x ** 2 if case == "cubic" else 0.0 * x)
    value = x.sum() + parameter * (x ** 3).sum() if case == "cubic" else parameter + x.sum()
    return float(value), (1.0 + delta) * true_grad


def physics(x):
    x = np.asarray(x, dtype=float)
    rows = [row(x, *spec) for spec in ROWS]
    return ALMPhysics(base_value=float(x @ x), base_grad=2.0 * x,
                      constraint_values=np.array([value for value, _grad in rows]),
                      constraint_grads=tuple(grad for _value, grad in rows))


def build_problem(smoke=False):
    return SimpleNamespace(name="sweep", x0=np.zeros(2), physics=physics,
                           constraint_names=tuple(f"{case}_{parameter:g}_{delta:g}" for case, parameter, delta in ROWS),
                           taylor_epsilons=None)
"""


def skill_markdown() -> Dict[str, str]:
    paths = [SKILL_DIR / "SKILL.md", *sorted(REFERENCES_DIR.glob("*.md"))]
    return {path.relative_to(SKILL_DIR).as_posix(): path.read_text() for path in paths}


def skill_python() -> Dict[str, str]:
    paths = sorted(TEMPLATES_DIR.glob("*.py")) + sorted(SCRIPTS_DIR.glob("*.py"))
    return {path.relative_to(SKILL_DIR).as_posix(): path.read_text() for path in paths}


def string_value(node) -> Optional[str]:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def call_name(node: ast.Call) -> Optional[str]:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def package_termination_reasons() -> Set[str]:
    """Every string the package source can return as ``termination_reason``."""
    reasons = set()
    for path in sorted(PACKAGE_DIR.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.keyword) and node.arg in TERMINATION_KEYWORDS:
                reasons.add(string_value(node.value))
            elif (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
                  and node.target.id in TERMINATION_CARRIERS and node.value is not None):
                reasons.add(string_value(node.value))
            elif isinstance(node, ast.Call) and call_name(node) in BUDGET_ACTION_CALLS:
                reasons.update(string_value(keyword.value) for keyword in node.keywords
                               if keyword.arg == "action")
    return reasons - {None}


def package_success_reasons() -> Set[str]:
    """The ``termination_reason`` of every ``ALMConverge`` the package builds."""
    reasons = set()
    for path in sorted(PACKAGE_DIR.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and call_name(node) == "ALMConverge":
                reasons.update(string_value(keyword.value) for keyword in node.keywords
                               if keyword.arg == "termination_reason")
    return reasons - {None}


def documented_termination_rows() -> Dict[str, str]:
    text = (REFERENCES_DIR / "termination.md").read_text()
    rows = re.findall(r"^\| `([a-z_]+)` \| (yes|no) \|", text, re.MULTILINE)
    return dict(rows)


def public_names(module) -> Set[str]:
    """``__all__``, else the public classes and functions defined in the module."""
    if hasattr(module, "__all__"):
        return set(module.__all__)
    return {name for name, value in vars(module).items()
            if not name.startswith("_") and getattr(value, "__module__", None) == module.__name__}


def api_table() -> Dict[str, str]:
    """``name -> module`` from the table of api.md's "Names" section."""
    text = (REFERENCES_DIR / "api.md").read_text()
    section = re.search(r"^## Names\n(.*?)^## ", text, re.MULTILINE | re.DOTALL).group(1)
    return dict(re.findall(r"^\| `(\w+)` \| `([\w.]+)` \|", section, re.MULTILINE))


def python_fences(text: str) -> List[str]:
    return re.findall(r"^```python\n(.*?)^```", text, re.MULTILINE | re.DOTALL)


def documented_imports(source: str) -> List[tuple]:
    """``(module, name)`` for every ``from <documented module> import name``."""
    return [(node.module, alias.name) for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.ImportFrom) and node.module in MODULES
            for alias in node.names]


def child_env(problem_dir: Optional[Path] = None) -> dict:
    python_path = ([] if problem_dir is None else [str(problem_dir)]) + [str(SRC_DIR)]
    if os.environ.get("PYTHONPATH"):
        python_path.append(os.environ["PYTHONPATH"])
    return {**os.environ, "PYTHONPATH": os.pathsep.join(python_path)}


def run_python(arguments: List[str], cwd: Path, problem_dir: Optional[Path] = None,
               timeout: int = 900) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *arguments], cwd=cwd, env=child_env(problem_dir),
                          capture_output=True, text=True, timeout=timeout, check=False)


def result_line(completed: subprocess.CompletedProcess, prefix: str) -> dict:
    lines = [line for line in completed.stdout.splitlines() if line.startswith(prefix)]
    if len(lines) != 1:
        raise AssertionError(f"expected one {prefix!r} line (exit {completed.returncode})\n"
                             f"stdout:\n{completed.stdout[-4000:]}\nstderr:\n{completed.stderr[-4000:]}")
    return json.loads(lines[0][len(prefix):])


def generate_problem(directory: Path, template: str, source: Optional[str] = None) -> None:
    """Write ``alm_problem.py`` (the template, or ``source``) and ``run_alm.py``."""
    text = (TEMPLATES_DIR / f"{template}.py").read_text() if source is None else source
    (directory / "alm_problem.py").write_text(text)
    shutil.copy(TEMPLATES_DIR / "run_alm.py", directory / "run_alm.py")


def replaced_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise AssertionError(f"the template no longer contains {old!r} exactly once; "
                             "update this test's edit")
    return text.replace(old, new)


def diff_sides(diff: str) -> tuple:
    """The old and new file of a unified diff that carries full context."""
    body = [line for line in diff.splitlines()
            if not line.startswith(("--- ", "+++ ", "@@"))]
    old = "\n".join(line[1:] for line in body if line[:1] in (" ", "-")) + "\n"
    new = "\n".join(line[1:] for line in body if line[:1] in (" ", "+")) + "\n"
    return old, new


def module_assignments(source: str) -> Set[str]:
    return {target.id for node in ast.parse(source).body if isinstance(node, (ast.Assign, ast.AnnAssign))
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
            if isinstance(target, ast.Name)}


class SkillDocumentsThePackageTests(unittest.TestCase):
    def test_frontmatter_names_the_skill(self):
        frontmatter = re.match(r"\A---\n(.*?)\n---\n", (SKILL_DIR / "SKILL.md").read_text(), re.DOTALL)
        self.assertIsNotNone(frontmatter, "SKILL.md has no frontmatter")
        self.assertIn(f"name: {SKILL_DIR.name}\n", frontmatter.group(1) + "\n")
        self.assertRegex(frontmatter.group(1), r"(?m)^description: \".+\"$")

    def test_every_api_table_name_resolves_in_its_module(self):
        table = api_table()
        self.assertGreater(len(table), 20, "the api.md table was not parsed")
        for name, module_path in table.items():
            with self.subTest(name=name):
                self.assertIn(module_path, MODULES, f"{module_path} is not a module the skill may name")
                self.assertTrue(hasattr(MODULES[module_path], name), f"{module_path}.{name} does not exist")

    def test_backticked_api_names_are_in_the_api_table(self):
        table = api_table()
        package_names = {name for module in MODULES.values() for name in public_names(module)}
        for source, text in skill_markdown().items():
            spans = re.findall(r"`([^`\n]+)`", text)
            for identifier in {re.match(r"[A-Za-z_]\w*", span).group(0) for span in spans
                               if re.match(r"[A-Za-z_]\w*", span)}:
                with self.subTest(source=source, name=identifier):
                    if re.fullmatch(r"ALM[A-Z]\w*", identifier):
                        self.assertIn(identifier, package_names, f"{identifier} is not in the package")
                    if identifier in package_names:
                        self.assertIn(identifier, table, f"{identifier} is used but not in api.md's table")
            for dotted in re.findall(r"`(simsopt(?:\.\w+)+)`", text):
                with self.subTest(source=source, name=dotted):
                    module_path, _, attribute = dotted.rpartition(".")
                    self.assertTrue(dotted in MODULES or (module_path in MODULES
                                                          and hasattr(MODULES[module_path], attribute)),
                                    f"{dotted} does not resolve")
            for attribute in re.findall(r"`result\.(\w+)", text):
                with self.subTest(source=source, name=attribute):
                    self.assertIn(attribute, {field.name for field in fields(ALMResult)})

    def test_skill_python_imports_resolve(self):
        sources = dict(skill_python())
        for source, text in skill_markdown().items():
            for index, fence in enumerate(python_fences(text)):
                sources[f"{source} (python block {index})"] = fence
        checked = 0
        for source, text in sources.items():
            for module_path, name in documented_imports(text):
                checked += 1
                with self.subTest(source=source, name=f"{module_path}.{name}"):
                    self.assertTrue(hasattr(MODULES[module_path], name), f"{module_path}.{name} does not exist")
        self.assertGreater(checked, 20, "no documented imports were found")

    def test_settings_reference_lists_every_field_with_its_default(self):
        text = (REFERENCES_DIR / "settings.md").read_text()
        documented = dict(re.findall(r"^\| `(\w+)` \| ([^|]+?) \|", text, re.MULTILINE))
        package_fields = {field.name: field.default for field in fields(ALMSettings)}
        self.assertEqual(set(documented), set(package_fields),
                         "settings.md's table and ALMSettings list different fields")
        for name, default in package_fields.items():
            with self.subTest(field=name):
                self.assertIsNot(default, MISSING)
                if default is None or isinstance(default, bool):
                    self.assertEqual(documented[name], repr(default))
                else:
                    self.assertEqual(float(documented[name]), float(default))

    def test_termination_reference_matches_the_package_reasons(self):
        documented = set(documented_termination_rows())
        package = package_termination_reasons()
        self.assertEqual(sorted(package - documented), [],
                         "termination reasons in the package but not in termination.md")
        self.assertEqual(sorted(documented - package), [],
                         "termination reasons in termination.md that the package cannot return")

    def test_termination_success_column_matches_converge_decisions(self):
        rows = documented_termination_rows()
        self.assertEqual({reason for reason, success in rows.items() if success == "yes"},
                         package_success_reasons())

    def test_python_floor_matches_pyproject(self):
        requires = re.search(r'^requires-python = ">=([\d.]+)"', (REPO_ROOT / "pyproject.toml").read_text(),
                             re.MULTILINE).group(1)
        self.assertEqual(check_env.MIN_PYTHON, tuple(int(part) for part in requires.split(".")))

    def test_install_reference_names_the_branch_base(self):
        self.assertIn(f"`{check_env.ALM_UPSTREAM_BASE[:9]}`", (REFERENCES_DIR / "install.md").read_text())

    def test_guide_is_generated_from_the_skill(self):
        self.assertTrue(build_guide.GUIDE_PATH.exists(), "docs/alm_setup_guide.md is missing")
        self.assertTrue(build_guide.GUIDE_PATH.read_text() == build_guide.render_guide(),
                        "docs/alm_setup_guide.md is stale: run "
                        ".claude/skills/simsopt-alm-setup/scripts/build_guide.py")

    def test_guide_drops_the_skill_only_blocks(self):
        blocks = re.findall(r"<!-- skill-only -->\n(.*?)<!-- /skill-only -->",
                            (SKILL_DIR / "SKILL.md").read_text(), re.DOTALL)
        self.assertTrue(blocks, "SKILL.md has no skill-only block")
        guide = build_guide.render_guide()
        for block in blocks:
            with self.subTest(block=block[:40]):
                self.assertNotIn(block.strip(), guide)


class CheckEnvRouteTests(unittest.TestCase):
    """``check_env.choose_route``: the first route whose condition holds."""

    @staticmethod
    def report(*, blockers=(), alm_importable=False, checkout=None) -> dict:
        return {"blockers": list(blockers), "alm": {"importable": alm_importable}, "checkout": checkout}

    @staticmethod
    def checkout(*, has_alm_sources=False, is_git=True, upstream_remotes_with_alm=()) -> dict:
        git = {"is_git": is_git}
        if is_git:
            git["upstream_remotes_with_alm"] = list(upstream_remotes_with_alm)
        return {"has_alm_sources": has_alm_sources, "git": git}

    def test_routes(self):
        cases = {
            "blocked": self.report(blockers=["Python too old"], alm_importable=True),
            "ready": self.report(alm_importable=True, checkout=self.checkout()),
            "reinstall": self.report(checkout=self.checkout(has_alm_sources=True)),
            "upstream": self.report(checkout=self.checkout(upstream_remotes_with_alm=["upstream"])),
            "merge-fork": self.report(checkout=self.checkout()),
            "copy": self.report(checkout=None),
        }
        for route, report in cases.items():
            with self.subTest(route=route):
                self.assertEqual(check_env.choose_route(report), route)
        self.assertEqual(check_env.choose_route(self.report(checkout=self.checkout(is_git=False))), "copy")

    def test_remote_urls_with_spaces_parse(self):
        remotes = check_env.parse_remotes(
            "origin\thttps://github.com/o/simsopt (fetch)\n"
            "origin\thttps://github.com/o/simsopt (push)\n"
            "local\t/data/my repos/simsopt (fetch)\n"
            "local\t/data/my repos/simsopt (push)\n"
            # git 2.53 annotates a partial clone's filter.
            "upstream\thttps://github.com/hiddenSymmetries/simsopt.git (fetch) [blob:none]\n"
            "upstream\thttps://github.com/hiddenSymmetries/simsopt.git (push)\n"
            # A remote with no URL, and one with a push URL only (git 2.53).
            "nourl\t\n"
            "pushonly\t\n"
            "pushonly\thttps://example.com/push.git (push)\n")
        self.assertEqual(remotes, {"origin": "https://github.com/o/simsopt", "local": "/data/my repos/simsopt",
                                   "upstream": "https://github.com/hiddenSymmetries/simsopt.git",
                                   "nourl": None, "pushonly": None})

    def test_an_import_failure_names_the_missing_module_or_parent(self):
        """Absent ALM package, absent parent package, and a broken import differ."""
        module = "simsopt.solve.alm"
        cases = {
            "ModuleNotFoundError: No module named 'simsopt.solve.alm'": "simsopt.solve.alm",
            "ModuleNotFoundError: No module named 'simsopt.solve'": "simsopt.solve",
            "ModuleNotFoundError: No module named 'simsopt'": "simsopt",
            "ModuleNotFoundError: No module named 'scipy'": None,
            "ModuleNotFoundError: No module named 'simsopt.solve.almanac'": None,
            "ImportError: cannot import name 'nnls' from 'scipy.optimize'": None,
            "SyntaxError: invalid syntax": None,
        }
        for error, missing in cases.items():
            with self.subTest(error=error):
                self.assertEqual(check_env.missing_import(error, module), missing)

    def test_an_absent_parent_package_is_not_reported_as_a_broken_alm(self):
        blocker = check_env.alm_import_blocker(
            {"importable": False, "error": "ModuleNotFoundError: No module named 'simsopt.solve'"})
        self.assertIn("simsopt.solve", blocker)
        self.assertNotIn("fails to import", blocker)
        self.assertIn("fails to import", check_env.alm_import_blocker(
            {"importable": False, "error": "ImportError: cannot import name 'nnls' from 'scipy.optimize'"}))
        self.assertIsNone(check_env.alm_import_blocker(
            {"importable": False, "error": "ModuleNotFoundError: No module named 'simsopt.solve.alm'"}))
        self.assertIsNone(check_env.alm_import_blocker({"importable": True, "error": None}))

    def test_merging_routes_need_a_clean_tree(self):
        dirty = {"git": {"dirty": True}}
        clean = {"git": {"dirty": False}}
        for route in ("upstream", "merge-fork"):
            with self.subTest(route=route):
                self.assertEqual(len(check_env.route_blockers(route, dirty)), 1)
                self.assertEqual(check_env.route_blockers(route, clean), [])
        self.assertEqual(check_env.route_blockers("copy", None), [])

    def test_repository_urls_compare_by_host_owner_and_name(self):
        for url in ("git@github.com:hiddenSymmetries/simsopt.git",
                    "https://github.com/hiddenSymmetries/simsopt",
                    "https://user@GitHub.com/hiddenSymmetries/simsopt.git/"):
            with self.subTest(url=url):
                self.assertEqual(check_env.normalized_repository(url), check_env.UPSTREAM_REPOSITORY)


class SkillScriptsRunTests(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.scratch = Path(scratch.name)

    def test_check_env_reports_this_tree_ready(self):
        completed = run_python([str(SCRIPTS_DIR / "check_env.py"), "--python", sys.executable,
                                "--checkout", str(REPO_ROOT)], cwd=self.scratch)
        report = result_line(completed, check_env.RESULT_PREFIX)
        self.assertEqual(report["route"], "ready", report)
        self.assertEqual(completed.returncode, 0)
        self.assertTrue(report["python"]["meets_floor"])
        self.assertTrue(report["alm"]["importable"])
        self.assertTrue(report["signed_constraints"]["importable"])
        self.assertTrue(report["checkout"]["has_alm_sources"])

    def test_smoke_toy_passes(self):
        completed = run_python([str(SCRIPTS_DIR / "smoke_toy.py")], cwd=self.scratch)
        summary = result_line(completed, "SMOKE_TOY ")
        self.assertTrue(summary["passed"], summary)
        self.assertEqual(completed.returncode, 0)
        self.assertEqual([problem["termination_reason"] for problem in summary["problems"]],
                         ["converged", "converged"])

    def test_gradient_check_passes_on_the_generic_template(self):
        generate_problem(self.scratch, "generic")
        completed = run_python([str(SCRIPTS_DIR / "gradient_check.py")], cwd=self.scratch,
                               problem_dir=self.scratch)
        summary = result_line(completed, "GRADIENT_CHECK ")
        self.assertTrue(summary["passed"], summary)
        self.assertEqual(completed.returncode, 0)
        self.assertEqual([quantity["quantity"] for quantity in summary["quantities"]],
                         ["f", "sum_at_most_max", "x0_at_least_min"])
        self.assertEqual({quantity["verdict"] for quantity in summary["quantities"]}, {"passed"})
        # A quadratic f and linear rows are differenced exactly.
        self.assertTrue(all(quantity["relative_error"] < 1e-10 for quantity in summary["quantities"]))

    def test_gradient_check_fails_on_a_wrong_gradient(self):
        source = replaced_once((TEMPLATES_DIR / "generic.py").read_text(),
                               '"coordinate_sum": (float(x[0] + x[1]), np.array([1.0, 1.0])),',
                               '"coordinate_sum": (float(x[0] + x[1]), np.array([1.0, 0.9])),')
        generate_problem(self.scratch, "generic", source)
        completed = run_python([str(SCRIPTS_DIR / "gradient_check.py")], cwd=self.scratch,
                               problem_dir=self.scratch)
        summary = result_line(completed, "GRADIENT_CHECK ")
        self.assertEqual(completed.returncode, 1)
        self.assertEqual({quantity["quantity"]: quantity["verdict"] for quantity in summary["quantities"]},
                         {"f": "passed", "sum_at_most_max": "failed", "x0_at_least_min": "passed"})

    def run_gradient_probe(self, mode: str, *arguments: str) -> subprocess.CompletedProcess:
        (self.scratch / "alm_problem.py").write_text(GRADIENT_PROBE_PROBLEM.replace("MODE_VALUE", repr(mode)))
        return run_python([str(SCRIPTS_DIR / "gradient_check.py"), *arguments], cwd=self.scratch,
                          problem_dir=self.scratch)

    @staticmethod
    def verdicts(completed: subprocess.CompletedProcess) -> Dict[str, str]:
        summary = result_line(completed, "GRADIENT_CHECK ")
        return {quantity["quantity"]: quantity["verdict"] for quantity in summary["quantities"]}

    def test_gradient_check_passes_exact_gradients(self):
        completed = self.run_gradient_probe("exact")
        self.assertEqual(self.verdicts(completed), {"f": "passed", "row": "passed"})
        self.assertEqual(completed.returncode, 0)

    def test_gradient_check_fails_zeroed_gradients(self):
        """A forgotten gradient on a quantity that does change is wrong, not vacuous."""
        completed = self.run_gradient_probe("zero")
        self.assertEqual(self.verdicts(completed), {"f": "failed", "row": "failed"})
        self.assertEqual(completed.returncode, 1)

    def test_gradient_check_does_not_pass_a_constant_quantity(self):
        completed = self.run_gradient_probe("constant")
        self.assertEqual(self.verdicts(completed), {"f": "not_tested", "row": "not_tested"})
        self.assertEqual(completed.returncode, 1)
        self.assertIn("NOT TESTED", completed.stdout)

    def test_gradient_check_fails_wrong_gradients(self):
        completed = self.run_gradient_probe("factor")
        self.assertEqual(self.verdicts(completed), {"f": "failed", "row": "failed"})
        self.assertEqual(completed.returncode, 1)

    def test_gradient_check_rejects_fewer_than_three_steps(self):
        """One step passed a gradient off by x1.5 and x2; the error model needs
        residuals, so fewer than three steps is an argument error."""
        for epsilons in ("1e-3", "1e-3,5e-4"):
            with self.subTest(epsilons=epsilons):
                completed = self.run_gradient_probe("factor", "--epsilons", epsilons)
                self.assertEqual(completed.returncode, 2, completed.stdout)
                self.assertIn("at least 3 steps", completed.stderr)
                self.assertNotIn("GRADIENT_CHECK", completed.stdout)

    def test_gradient_check_fails_a_wrong_gradient_on_a_tiny_row(self):
        """A row of size 1e-7 whose gradient is twice the truth fails; the
        tolerance is relative to the finite difference, not absolute."""
        completed = self.run_gradient_probe("small")
        self.assertEqual(self.verdicts(completed), {"f": "passed", "row": "failed"})
        self.assertEqual(completed.returncode, 1)

    def run_roundoff_probe(self, case: str, *arguments: str, n: int = 3,
                           a: float = 0.0) -> subprocess.CompletedProcess:
        source = (ROUNDOFF_PROBE_PROBLEM.replace("CASE_VALUE", repr(case)).replace("N_VALUE", repr(n))
                  .replace("A_VALUE", repr(a)))
        (self.scratch / "alm_problem.py").write_text(source)
        return run_python([str(SCRIPTS_DIR / "gradient_check.py"), *arguments], cwd=self.scratch,
                          problem_dir=self.scratch)

    def test_gradient_check_never_passes_by_the_floor_of_a_small_step(self):
        """Value 1e8, derivative 1e-3, claimed 0: the fit weighs each step by its
        round-off, so the small steps' large floors cannot hide the miss."""
        for arguments in ((), ("--epsilons", "1,0.5,0.25")):
            with self.subTest(arguments=arguments):
                completed = self.run_roundoff_probe("big_value_zero_grad", *arguments)
                self.assertEqual(self.verdicts(completed), {"f": "failed"})
                self.assertEqual(completed.returncode, 1)

    def test_gradient_check_passes_a_right_gradient_on_a_big_value_at_informative_steps(self):
        completed = self.run_roundoff_probe("big_value_right_grad", "--epsilons", "1,0.5,0.25")
        self.assertEqual(self.verdicts(completed), {"f": "passed"})
        self.assertEqual(completed.returncode, 0)

    def test_gradient_check_does_not_trust_a_ratio_below_the_absolute_floor(self):
        """True derivative 3e-11, claimed 0: the library's ratio test cannot see
        it (its floor is 1e-10); the relative error can."""
        completed = self.run_roundoff_probe("straddle_zero_grad", n=1, a=0.3e-10)
        self.assertEqual(self.verdicts(completed), {"f": "failed"})
        self.assertEqual(completed.returncode, 1)
        completed = self.run_roundoff_probe("straddle_right_grad", n=1, a=0.3e-10)
        self.assertEqual(self.verdicts(completed), {"f": "passed"})
        self.assertEqual(completed.returncode, 0)

    def test_gradient_check_rejects_zero_directions(self):
        completed = self.run_gradient_probe("exact", "--directions", "0")
        self.assertEqual(completed.returncode, 2, completed.stderr[-2000:])
        self.assertIn("--directions", completed.stderr)
        self.assertNotIn("Traceback", completed.stderr)

    def test_gradient_check_passes_the_stage2_rows_at_small_steps(self):
        """The curvature rows at 1e-5 to 2.5e-6, where round-off can stop the
        errors from falling, still pass (the case the accuracy verdict is for)."""
        generate_problem(self.scratch, "stage2")
        completed = run_python([str(SCRIPTS_DIR / "gradient_check.py"), "--smoke"], cwd=self.scratch,
                               problem_dir=self.scratch)
        verdicts = self.verdicts(completed)
        self.assertEqual(completed.returncode, 0, verdicts)
        self.assertEqual(set(verdicts.values()), {"passed"})
        self.assertEqual(len(verdicts), 11)

    def run_sweep(self, rows: list, *arguments: str) -> Dict[str, str]:
        (self.scratch / "alm_problem.py").write_text(SWEEP_PROBE_PROBLEM.replace("ROWS_VALUE", repr(rows)))
        completed = run_python([str(SCRIPTS_DIR / "gradient_check.py"), *arguments], cwd=self.scratch,
                               problem_dir=self.scratch)
        verdicts = self.verdicts(completed)
        verdicts.pop("f")
        return verdicts

    def test_gradient_check_fails_wrong_gradients_dominated_by_truncation(self):
        """F-A: q = sum(x) + K sum(x^3), truncation K e^2 up to 1e10 x the step
        squared; every wrong gradient fails and every right one passes, at the
        library's steps and at the templates' smaller ones."""
        rows = [("cubic", k, delta) for k in SWEEP_CUBIC_K for delta in (0.0,) + SWEEP_DELTAS]
        for arguments in ((), ("--epsilons", "1e-5,5e-6,2.5e-6")):
            with self.subTest(arguments=arguments):
                verdicts = self.run_sweep(rows, *arguments)
                for name, verdict in verdicts.items():
                    right = name.endswith("_0")
                    self.assertEqual(verdict, "passed" if right else "failed", name)

    def test_gradient_check_fails_wrong_gradients_on_offset_values(self):
        """F-B: q = F + sum(x), round-off of F up to 1e8: every gradient 0.1% to 5%
        wrong fails where the steps can measure it and is never passed; right
        ones pass there."""
        rows = [("offset", f, delta) for f in SWEEP_OFFSET_F for delta in (0.0,) + SWEEP_DELTAS]
        verdicts = self.run_sweep(rows)
        for name, verdict in verdicts.items():
            self.assertEqual(verdict, "passed" if name.endswith("_0") else "failed", name)
        # At the templates' tiny steps round-off of 1e8 is about 1e-3 of the change:
        # a wrong gradient fails or is undecided, never passed; a right one is not failed.
        verdicts = self.run_sweep(rows, "--epsilons", "1e-5,5e-6,2.5e-6")
        for name, verdict in verdicts.items():
            if name.startswith("offset_1e+08"):
                allowed = ("passed", "not_tested") if name.endswith("_0") else ("failed", "not_tested")
                self.assertIn(verdict, allowed, name)
            else:
                self.assertEqual(verdict, "passed" if name.endswith("_0") else "failed", name)

    def test_sign_check_passes_on_the_generic_template(self):
        generate_problem(self.scratch, "generic")
        completed = run_python([str(SCRIPTS_DIR / "sign_check.py")], cwd=self.scratch,
                               problem_dir=self.scratch)
        summary = result_line(completed, "SIGN_CHECK ")
        self.assertTrue(summary["passed"], summary)
        self.assertEqual(completed.returncode, 0)
        self.assertEqual(summary["coverage_warnings"], [])
        self.assertEqual(summary["scales"]["warnings"], [])
        self.assertEqual(summary["shared_source_rows"], [])

    def test_sign_check_fails_on_a_flipped_row(self):
        source = replaced_once((TEMPLATES_DIR / "generic.py").read_text(),
                               'sense=LOWER_BOUND, bound=0.25', 'sense=UPPER_BOUND, bound=0.25')
        generate_problem(self.scratch, "generic", source)
        completed = run_python([str(SCRIPTS_DIR / "sign_check.py")], cwd=self.scratch,
                               problem_dir=self.scratch)
        summary = result_line(completed, "SIGN_CHECK ")
        self.assertEqual(completed.returncode, 1)
        failures = [failure for probe in summary["probes"] for failure in probe["failures"]]
        self.assertEqual(len(failures), 2, failures)
        self.assertTrue(all(failure.startswith("x0_at_least_min:") for failure in failures), failures)


class TemplateSmokeTests(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.scratch = Path(scratch.name)

    def run_template(self, template: str, *arguments: str) -> dict:
        completed = run_python(["run_alm.py", *arguments], cwd=self.scratch)
        summary = result_line(completed, "ALM_RESULT ")
        self.assertEqual(completed.returncode, 0, completed.stderr[-4000:])
        self.assertIn(summary["termination_reason"], package_termination_reasons())
        self.assertEqual(summary["problem"], template)
        return summary

    def test_generic_template_converges_to_the_known_solution(self):
        generate_problem(self.scratch, "generic")
        summary = self.run_template("generic", "--smoke")
        self.assertEqual(summary["termination_reason"], "converged")
        np.testing.assert_allclose(summary["finish"]["x"], [1.5, 0.5], atol=1e-5)
        # The sum row is divided by its bound 2, so its multiplier is 2 x 1.
        np.testing.assert_allclose(list(summary["multipliers"].values()), [2.0, 0.0], atol=1e-3)

    def test_stage2_template_smoke_run_ends_with_a_reason(self):
        generate_problem(self.scratch, "stage2")
        summary = self.run_template("stage2", "--smoke")
        self.assertEqual(len(summary["constraint_values"]), 10)
        self.assertTrue((self.scratch / "output_alm" / "biot_savart_opt_alm.json").exists())
        # f is the squared flux plus the length regularizer; both are reported by name.
        self.assertLessEqual({"objective", "squared_flux"}, set(summary["finish"]))
        self.assertLess(summary["finish"]["squared_flux"], summary["finish"]["objective"])

    def test_boozer_template_smoke_run_ends_with_a_reason(self):
        generate_problem(self.scratch, "boozer_single_stage")
        summary = self.run_template("boozer_single_stage", "--smoke")
        self.assertEqual(len(summary["constraint_values"]), 12)
        self.assertTrue(summary["finish"]["surface_solved"])

    def test_boozer_sign_check_names_the_rows_it_cannot_check_independently(self):
        """The iota and major-radius probes read the same source as the rows."""
        generate_problem(self.scratch, "boozer_single_stage")
        completed = run_python([str(SCRIPTS_DIR / "sign_check.py"), "--smoke"], cwd=self.scratch,
                               problem_dir=self.scratch)
        summary = result_line(completed, "SIGN_CHECK ")
        self.assertEqual(completed.returncode, 0, summary["probes"])
        self.assertEqual(summary["shared_source_rows"],
                         ["iota_min", "iota_max", "major_radius_min", "major_radius_max"])
        self.assertIn("same source", completed.stdout)

    def build_fails(self, template: str, old: str, new: str) -> str:
        """Build the edited template; it must raise; return the error line."""
        generate_problem(self.scratch, template, replaced_once((TEMPLATES_DIR / f"{template}.py").read_text(),
                                                               old, new))
        completed = run_python(["-c", "from alm_problem import build_problem; build_problem(smoke=True)"],
                               cwd=self.scratch, problem_dir=self.scratch)
        self.assertNotEqual(completed.returncode, 0, "the build accepted an invalid scale")
        return completed.stderr.strip().splitlines()[-1]

    def test_templates_reject_invalid_scales(self):
        for scale in ("0.0", "-1.0", "float('nan')"):
            with self.subTest(template="generic", scale=scale):
                error = self.build_fails("generic", "sense=LOWER_BOUND, bound=0.25, scale=1.0)",
                                         f"sense=LOWER_BOUND, bound=0.25, scale={scale})")
                self.assertRegex(error, r"^ValueError: .*x0_at_least_min.*scale")
        with self.subTest(template="stage2"):
            error = self.build_fails("stage2", "CC_MIN_DISTANCE = 0.1 ", "CC_MIN_DISTANCE = 0.0 ")
            self.assertRegex(error, r"^ValueError: CC_MIN_DISTANCE")
        with self.subTest(template="boozer_single_stage"):
            error = self.build_fails("boozer_single_stage", "IOTA_TARGET: Optional[float] = None",
                                     "IOTA_TARGET: Optional[float] = 0.0")
            self.assertRegex(error, r"^ValueError: .*IOTA_SCALE")

    def test_stage2_length_scopes_count_the_right_coils(self):
        """The all-coils row is 2 x nfp = 4 times the base-coil sum (16 circles of
        length pi at x0); the per-coil rows bound each base coil; the sign probes
        (circles of minor radius 0.9 and 0.15 m) check both sides of each row
        against lengths measured over the physical curves."""
        cases = {
            "SUM_OF_ALL_COILS": (60.0, {"length_of_all_coils": (16 * np.pi - 60.0) / 60.0}),
            "PER_BASE_COIL": (4.0, {f"length_of_base_coil_{i}": (np.pi - 4.0) / 4.0 for i in range(4)}),
        }
        template = (TEMPLATES_DIR / "stage2.py").read_text()
        for scope, (bound, expected) in cases.items():
            with self.subTest(scope=scope):
                source = replaced_once(template, "MAX_LENGTH = None\n", f"MAX_LENGTH = {bound}\n")
                source = replaced_once(source, "LENGTH_SCOPE = SUM_OF_BASE_COILS\n", f"LENGTH_SCOPE = {scope}\n")
                generate_problem(self.scratch, "stage2", source)
                completed = run_python([str(SCRIPTS_DIR / "sign_check.py"), "--smoke"], cwd=self.scratch,
                                       problem_dir=self.scratch)
                summary = result_line(completed, "SIGN_CHECK ")
                self.assertEqual(completed.returncode, 0, summary["probes"])
                rows = summary["scales"]["rows"]
                for name, value in expected.items():
                    self.assertAlmostEqual(rows[name]["value"], value, places=6)
                self.assertEqual([warning for warning in summary["coverage_warnings"] if "length" in warning], [])

    def test_existing_script_example_runs(self):
        """references/existing-script.md: its diff turns the penalty script into
        one that calls the generated files; the adapted script runs (CI size),
        and the constants its table names exist in the Stage-2 template."""
        text = (REFERENCES_DIR / "existing-script.md").read_text()
        diffs = re.findall(r"^```diff\n(.*?)^```", text, re.MULTILINE | re.DOTALL)
        self.assertEqual(len(diffs), 1)
        before, after = diff_sides(diffs[0])
        penalty_names = module_assignments(before)
        self.assertLessEqual({"CC_WEIGHT", "CS_WEIGHT", "CURVATURE_WEIGHT", "MSC_WEIGHT", "JF"}, penalty_names)
        # Quoted as upstream writes it.
        self.assertIn("LENGTH_WEIGHT = Weight(1e-6)\n", before)
        template_names = module_assignments((TEMPLATES_DIR / "stage2.py").read_text())
        table = [line for line in text.splitlines() if line.startswith("| `")]
        named = {name for line in table for span in re.findall(r"`([^`]+)`", line)
                 for name in re.findall(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*\b", span)}
        self.assertEqual(sorted(named - penalty_names - template_names), [],
                         "existing-script.md names constants the Stage-2 template does not define")

        (self.scratch / "alm_stage2").mkdir()
        generate_problem(self.scratch / "alm_stage2", "stage2")
        (self.scratch / "my_stage2.py").write_text(after)
        completed = subprocess.run([sys.executable, "my_stage2.py"], cwd=self.scratch,
                                   env={**child_env(), "CI": "true"}, capture_output=True, text=True,
                                   timeout=900, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr[-4000:])
        # The script prints the termination reason first on its last line.
        self.assertIn(completed.stdout.strip().splitlines()[-1].split()[0], package_termination_reasons())
        self.assertTrue((self.scratch / "output" / "biot_savart_opt.json").exists())

    def test_runner_history_checkpoints_and_resume(self):
        generate_problem(self.scratch, "generic")
        full = self.run_template("generic", "--smoke", "--history", "history.json",
                                 "--checkpoints", "checkpoints")
        entries = json.loads((self.scratch / "history.json").read_text())
        self.assertGreaterEqual(len(entries), full["outer_iterations"])
        self.assertEqual(entries[-1]["action"], "converged")
        self.assertTrue((self.scratch / "checkpoints" / "final.pkl").exists())
        resumed = self.run_template("generic", "--smoke", "--resume", "checkpoints/outer_003.pkl")
        for key in ("termination_reason", "outer_iterations", "inner_iterations", "multipliers", "finish"):
            with self.subTest(key=key):
                self.assertEqual(resumed[key], full[key])


if __name__ == "__main__":
    unittest.main()
