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
                           constraint_names=("row",))
"""


# Crucible round 6: noisy quantities, like an iterative inner solve. Each has
# the value (1 + sum x + sum x^2) (1 + NOISE u), u uniform in [-1, 1] drawn
# from Python's hash of the point (so PYTHONHASHSEED picks the noise), and
# claims (1 + DELTA) times the true gradient. The objective is the round-6
# repro itself (``hash(x.tobytes())``, x0 = (0, 0)); each row salts its hash
# with its index. Rows are (NOISE, DELTA).
NOISY_PROBE_PROBLEM = """\
from types import SimpleNamespace

import numpy as np

from simsopt.solve.alm import ALMPhysics

OBJECTIVE, ROWS = OBJECTIVE_VALUE, ROWS_VALUE


def noisy(x, salt, noise, delta):
    key = x.tobytes() if salt is None else (x.tobytes(), salt)
    draw = np.random.RandomState(abs(hash(key)) % (2 ** 32)).uniform(-1.0, 1.0)
    return (1.0 + x.sum() + (x ** 2).sum()) * (1.0 + noise * draw), (1.0 + delta) * (1.0 + 2.0 * x)


def physics(x):
    x = np.asarray(x, dtype=float)
    value, grad = noisy(x, None, *OBJECTIVE)
    rows = [noisy(x, index, *row) for index, row in enumerate(ROWS)]
    return ALMPhysics(base_value=float(value), base_grad=grad,
                      constraint_values=np.array([float(value) for value, _grad in rows]),
                      constraint_grads=tuple(grad for _value, grad in rows))


def build_problem(smoke=False):
    return SimpleNamespace(name="noisy", x0=np.zeros(2), physics=physics,
                           constraint_names=tuple(f"noise_{noise:g}_delta_{delta:g}" for noise, delta in ROWS))
"""
# Round-6 seeds of the noise.
NOISY_HASH_SEEDS = tuple(range(8))


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
    return SimpleNamespace(name=CASE, x0=np.zeros(N), physics=physics, constraint_names=())
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
                           constraint_names=tuple(f"{case}_{parameter:g}_{delta:g}" for case, parameter, delta in ROWS))
"""


# Crucible rounds 4 and 5: rows whose verdicts must be attributed correctly.
# Each row is (case, parameter, relative error of the claimed gradient,
# absolute error added to it); x0 = (0, 0) and f = x0 + x1 (an O(1) gradient,
# which sets the absolute floor of a derivative).
SHAPE_PROBE_PROBLEM = """\
from types import SimpleNamespace

import numpy as np

from simsopt.solve.alm import ALMPhysics

ROWS = ROWS_VALUE


def row(x, case, parameter, relative, absolute):
    if case == "const":          # identically parameter
        value, grad = parameter + 0.0 * x.sum(), np.zeros_like(x)
    elif case == "zero":         # identically 0
        value, grad = 0.0 * x.sum(), np.zeros_like(x)
    elif case == "quadratic":    # parameter + sum(x^2): zero derivative at 0
        value, grad = parameter + (x ** 2).sum(), 2.0 * x
    elif case == "hinge":        # inactive clipped row max(0, x0 - parameter)
        value, grad = max(0.0, x[0] - parameter), np.zeros_like(x)
    elif case == "abs":          # |x0 - parameter| + x1: a kink parameter away
        value, grad = abs(x[0] - parameter) + x[1], np.array([np.sign(x[0] - parameter) or 1.0, 1.0])
    elif case == "max_tie":      # max(x0, x1) at the tie, subgradient e0
        value, grad = max(x[0], x[1]), np.array([1.0, 0.0])
    elif case == "sin":          # sum(sin(parameter x)) / parameter
        value, grad = np.sin(parameter * x).sum() / parameter, np.cos(parameter * x)
    elif case == "offset_sin":   # 1e4 + sum(sin(parameter x)) / parameter: steep on a large offset
        value, grad = 1e4 + np.sin(parameter * x).sum() / parameter, np.cos(parameter * x)
    elif case == "tanh":         # sum(x) + tanh(parameter (x0 - 0.3 / parameter)): a steep smooth step
        value = x.sum() + np.tanh(parameter * x[0] - 0.3)
        grad = np.array([1.0 + parameter / np.cosh(parameter * x[0] - 0.3) ** 2, 1.0])
    elif case == "exp":          # parameter + sum(exp(x))
        value, grad = parameter + np.exp(x).sum(), np.exp(x)
    elif case == "scaled":       # parameter (1 + sum(x)): a huge value with a derivative of its size
        value, grad = parameter * (1.0 + x.sum()), parameter * np.ones_like(x)
    elif case == "overflow":     # 1e300 exp(50 sum(x)): infinite at the largest steps
        value, grad = parameter * np.exp(50.0 * x.sum()), 50.0 * parameter * np.exp(50.0 * x.sum()) * np.ones_like(x)
    elif case == "cancel":       # (parameter + sum x) - parameter: value 0, round-off of parameter
        value, grad = (parameter + x.sum()) - parameter, np.ones_like(x)
    else:                        # "offset": parameter + sum(x)
        value, grad = parameter + x.sum(), np.ones_like(x)
    return float(value), (1.0 + relative) * grad + absolute


def physics(x):
    x = np.asarray(x, dtype=float)
    rows = [row(x, *spec) for spec in ROWS]
    return ALMPhysics(base_value=float(x.sum()), base_grad=np.ones_like(x),
                      constraint_values=np.array([value for value, _grad in rows]),
                      constraint_grads=tuple(grad for _value, grad in rows))


def build_problem(smoke=False):
    return SimpleNamespace(name="shape", x0=np.zeros(2), physics=physics,
                           constraint_names=tuple("_".join(f"{item:g}" if not isinstance(item, str) else item
                                                           for item in spec) for spec in ROWS))
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


def child_env(problem_dir: Optional[Path] = None, overrides: Optional[Dict[str, str]] = None) -> dict:
    python_path = ([] if problem_dir is None else [str(problem_dir)]) + [str(SRC_DIR)]
    if os.environ.get("PYTHONPATH"):
        python_path.append(os.environ["PYTHONPATH"])
    return {**os.environ, "PYTHONPATH": os.pathsep.join(python_path), **(overrides or {})}


def run_python(arguments: List[str], cwd: Path, problem_dir: Optional[Path] = None, timeout: int = 900,
               overrides: Optional[Dict[str, str]] = None) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *arguments], cwd=cwd, env=child_env(problem_dir, overrides),
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

    def test_gradient_check_on_exact_zeroed_constant_and_wrong_gradients(self):
        """Exact gradients and a constant with zero gradients pass; zeroed or
        scaled gradients on changing quantities fail (Crucible round 1)."""
        cases = {"exact": {"f": "passed", "row": "passed"}, "constant": {"f": "passed", "row": "passed"},
                 "zero": {"f": "failed", "row": "failed"}, "factor": {"f": "failed", "row": "failed"},
                 "small": {"f": "passed", "row": "failed"}}
        for mode, expected in cases.items():
            with self.subTest(mode=mode):
                completed = self.run_gradient_probe(mode)
                self.assertEqual(self.verdicts(completed), expected)
                self.assertEqual(completed.returncode, 0 if set(expected.values()) == {"passed"} else 1)

    def run_roundoff_probe(self, case: str, *arguments: str, n: int = 3,
                           a: float = 0.0) -> subprocess.CompletedProcess:
        source = (ROUNDOFF_PROBE_PROBLEM.replace("CASE_VALUE", repr(case)).replace("N_VALUE", repr(n))
                  .replace("A_VALUE", repr(a)))
        (self.scratch / "alm_problem.py").write_text(source)
        return run_python([str(SCRIPTS_DIR / "gradient_check.py"), *arguments], cwd=self.scratch,
                          problem_dir=self.scratch)

    def test_gradient_check_on_a_big_value_with_a_small_derivative(self):
        """Value 1e8, derivative 1e-3: a zero claim fails (the small steps, where
        1e8 hides the change, cannot pass it); a right claim is never failed,
        and passes with steps large enough to resolve it to 1e-6."""
        completed = self.run_roundoff_probe("big_value_zero_grad")
        self.assertEqual(self.verdicts(completed), {"f": "failed"})
        self.assertNotEqual(self.verdicts(self.run_roundoff_probe("big_value_right_grad"))["f"], "failed")
        completed = self.run_roundoff_probe("big_value_right_grad", "--max-step", "1000")
        self.assertEqual(self.verdicts(completed), {"f": "passed"})

    def test_gradient_check_sees_a_derivative_below_the_libraries_floor(self):
        """True derivative 3e-11, claimed 0: fails; claimed right: passes."""
        self.assertEqual(self.verdicts(self.run_roundoff_probe("straddle_zero_grad", n=1, a=0.3e-10)),
                         {"f": "failed"})
        self.assertEqual(self.verdicts(self.run_roundoff_probe("straddle_right_grad", n=1, a=0.3e-10)),
                         {"f": "passed"})

    def test_gradient_check_rejects_invalid_arguments(self):
        (self.scratch / "alm_problem.py").write_text(GRADIENT_PROBE_PROBLEM.replace("MODE_VALUE", repr("exact")))
        for arguments in (("--directions", "0"), ("--max-step", "0"), ("--max-step", "-1"), ("--max-step", "inf"),
                          ("--min-step", "x"), ("--max-step", "1e-6", "--min-step", "1e-3"),
                          ("--steps-per-decade", "0"), ("--seed", "-1"), ("--seed", str(2 ** 32))):
            with self.subTest(arguments=arguments):
                completed = run_python([str(SCRIPTS_DIR / "gradient_check.py"), *arguments],
                                       cwd=self.scratch, problem_dir=self.scratch)
                self.assertEqual(completed.returncode, 2, completed.stderr[-2000:])
                self.assertNotIn("Traceback", completed.stderr)

    def test_gradient_check_passes_the_stage2_rows(self):
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
        """q = sum(x) + K sum(x^3), K up to 1e10: every gradient 0.1% to 5% wrong
        fails and every right one passes (Crucible round 3)."""
        rows = [("cubic", k, delta) for k in SWEEP_CUBIC_K for delta in (0.0,) + SWEEP_DELTAS]
        for name, verdict in self.run_sweep(rows).items():
            self.assertEqual(verdict, "passed" if name.endswith("_0") else "failed", name)

    def test_gradient_check_fails_wrong_gradients_on_offset_values(self):
        """q = F + sum(x), F up to 1e8: every gradient 0.1% to 5% wrong fails and
        every right one passes (Crucible round 3)."""
        rows = [("offset", f, delta) for f in SWEEP_OFFSET_F for delta in (0.0,) + SWEEP_DELTAS]
        for name, verdict in self.run_sweep(rows).items():
            self.assertEqual(verdict, "passed" if name.endswith("_0") else "failed", name)

    def run_shape(self, rows: list, *arguments: str) -> Dict[str, dict]:
        (self.scratch / "alm_problem.py").write_text(SHAPE_PROBE_PROBLEM.replace("ROWS_VALUE", repr(rows)))
        completed = run_python([str(SCRIPTS_DIR / "gradient_check.py"), *arguments], cwd=self.scratch,
                               problem_dir=self.scratch)
        self.last_completed = completed
        summary = result_line(completed, "GRADIENT_CHECK ")
        return {quantity["quantity"]: quantity for quantity in summary["quantities"] if quantity["quantity"] != "f"}

    def assert_verdicts(self, quantities: Dict[str, dict], expected: Dict[str, str]) -> None:
        self.assertEqual({name: quantity["verdict"] for name, quantity in quantities.items()}, expected,
                         {name: quantity["note"] for name, quantity in quantities.items()})

    def test_gradient_check_passes_flat_and_zero_rows(self):
        """A constant, a zero-derivative and an inactive clipped row pass a zero
        claim through a near-zero window; a clearly nonzero claim fails; an
        exactly zero row passes a round-off-size claim (R5-7: its floor comes
        from the objective's gradient, not 0) and fails a real one; a constant
        of 1e4 passes a claim at its own round-off (R6-F5) and fails 1e-6."""
        rows = [("const", 1.0, 0.0, 0.0), ("quadratic", 3.0, 0.0, 0.0), ("hinge", 1e-4, 0.0, 0.0),
                ("const", 1.0, 0.0, 1e-6), ("quadratic", 3.0, 0.0, 1e-6),
                ("zero", 0.0, 0.0, 1e-17), ("zero", 0.0, 0.0, 1e-6),
                ("const", 1e4, 0.0, 1e-12), ("const", 1e4, 0.0, 1e-6)]
        self.assert_verdicts(self.run_shape(rows), {
            "const_1_0_0": "passed", "quadratic_3_0_0": "passed", "hinge_0.0001_0_0": "passed",
            "const_1_0_1e-06": "failed", "quadratic_3_0_1e-06": "failed",
            "zero_0_0_1e-17": "passed", "zero_0_0_1e-06": "failed",
            "const_10000_0_1e-12": "passed", "const_10000_0_1e-06": "failed"})

    def test_gradient_check_passes_steep_smooth_rows(self):
        """R5-1: tanh(1e5 x0 - 0.3) and sin(K x)/K up to K = 1e5 with correct
        gradients pass (the sweep reaches steps below their feature size); 0.1%
        wrong ones fail."""
        rows = [("tanh", k, 0.0, 0.0) for k in (1e3, 1e5)] + [("sin", k, 0.0, 0.0) for k in (1e3, 1e4, 1e5)]
        rows += [(case, k, 1e-3, 0.0) for case, k, _relative, _absolute in rows]
        quantities = self.run_shape(rows)
        for name, quantity in quantities.items():
            with self.subTest(row=name):
                self.assertEqual(quantity["verdict"], "failed" if "_0.001_" in name else "passed", quantity["note"])

    def test_gradient_check_verdict_does_not_depend_on_an_added_constant(self):
        """R5-2: c + sum(exp(x)) for c = 0, 1, 100: the same verdicts, right
        gradients passing and 0.1% wrong ones failing."""
        rows = [("exp", c, relative, 0.0) for c in (0.0, 1.0, 100.0) for relative in (0.0, 1e-3)]
        quantities = self.run_shape(rows)
        for name, quantity in quantities.items():
            with self.subTest(row=name):
                self.assertEqual(quantity["verdict"], "failed" if "_0.001_" in name else "passed", quantity["note"])

    def test_gradient_check_never_fails_a_right_gradient_behind_a_cancellation(self):
        """R5-3: (F + sum x) - F has value 0 but the round-off of F; right
        gradients are never failed (identical values at small steps are the
        precision of F, not a derivative) and pass for F = 1e8; a 1% wrong one
        fails."""
        rows = [("cancel", f, 0.0, 0.0) for f in (1e8, 1e11, 1e12, 1e13)] + [("cancel", 1e8, 0.01, 0.0)]
        quantities = self.run_shape(rows)
        self.assertEqual(quantities["cancel_1e+08_0_0"]["verdict"], "passed")
        self.assertEqual(quantities["cancel_1e+08_0.01_0"]["verdict"], "failed")
        for f in ("1e+11", "1e+12", "1e+13"):
            with self.subTest(offset=f):
                self.assertNotEqual(quantities[f"cancel_{f}_0_0"]["verdict"], "failed")

    def test_gradient_check_not_tested_note_is_one_fixed_message(self):
        """R5-5: an undecidable row (steep on a large offset) is NOT TESTED with
        what was seen and one fixed piece of advice, never a step to retry."""
        quantities = self.run_shape([("offset_sin", 1e3, 0.0, 0.0)])
        quantity = quantities["offset_sin_1000_0_0"]
        self.assertEqual(quantity["verdict"], "not_tested")
        self.assertIn("check at a nearby point or inspect the quantity", quantity["note"])
        for step_advice in ("--max-step", "--min-step", "--epsilons", "larger", "smaller"):
            self.assertNotIn(step_advice, quantity["note"])
        self.assertEqual(self.last_completed.returncode, 1)

    def test_gradient_check_json_is_finite_for_huge_values(self):
        """R5-6: values of 1e220 and ones that overflow to inf at the largest
        steps keep the JSON finite (null, not NaN), and are still judged."""
        rows = [("scaled", 1e220, 0.0, 0.0), ("scaled", 1e220, 1.0, 0.0), ("overflow", 1e300, 0.0, 0.0)]
        quantities = self.run_shape(rows)
        line = [line for line in self.last_completed.stdout.splitlines() if line.startswith("GRADIENT_CHECK ")][0]
        self.assertNotIn("NaN", line)
        self.assertNotIn("Infinity", line)
        self.assertIn("null", line)
        self.assert_verdicts(quantities, {"scaled_1e+220_0_0": "passed", "scaled_1e+220_1_0": "failed",
                                          "overflow_1e+300_0_0": "passed"})

    def run_noisy(self, objective: tuple, rows: list, hash_seed: int) -> Dict[str, dict]:
        source = (NOISY_PROBE_PROBLEM.replace("OBJECTIVE_VALUE", repr(objective))
                  .replace("ROWS_VALUE", repr(rows)))
        (self.scratch / "alm_problem.py").write_text(source)
        completed = run_python([str(SCRIPTS_DIR / "gradient_check.py")], cwd=self.scratch,
                               problem_dir=self.scratch, overrides={"PYTHONHASHSEED": str(hash_seed)})
        summary = result_line(completed, "GRADIENT_CHECK ")
        return {quantity["quantity"]: quantity for quantity in summary["quantities"]}

    def test_gradient_check_fails_wrong_gradients_on_noisy_quantities(self):
        """R6-1: relative noise 1e-8 to 1e-5 (an iterative inner solve) with a
        gradient 5%, 50% or 500% wrong, or zero, fails at every round-6 seed:
        the large steps converge away from the claim, and the noise at small
        steps can pass nothing. The round-6 repro (objective, noise 1e-6, 500%
        wrong, PYTHONHASHSEED=1) passed before. At noise 1e-4 to 1e-3 three
        steps seldom agree to 1e-4, so these may be NOT TESTED: never passed."""
        failing = [(noise, delta) for noise in (1e-8, 1e-7, 1e-6, 1e-5) for delta in (0.05, 0.5, 5.0, -1.0)]
        unresolved = [(noise, delta) for noise in (1e-4, 1e-3) for delta in (0.05, 0.5, 5.0)]
        for hash_seed in NOISY_HASH_SEEDS:
            quantities = self.run_noisy((1e-6, 5.0), failing + unresolved, hash_seed)
            for name, quantity in quantities.items():
                with self.subTest(hash_seed=hash_seed, quantity=name):
                    if name == "f" or any(name == f"noise_{noise:g}_delta_{delta:g}" for noise, delta in failing):
                        self.assertEqual(quantity["verdict"], "failed", quantity["note"])
                    else:
                        self.assertNotEqual(quantity["verdict"], "passed", quantity["note"])

    def test_gradient_check_never_fails_a_right_gradient_on_noisy_quantities(self):
        """R6-2: the same noisy quantities with the right gradient are never
        failed (NOT TESTED is allowed where the noise hides the 1e-6 agreement)
        and pass at noise 1e-8."""
        rows = [(noise, 0.0) for noise in (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3)]
        for hash_seed in NOISY_HASH_SEEDS:
            quantities = self.run_noisy((1e-6, 0.0), rows, hash_seed)
            for name, quantity in quantities.items():
                with self.subTest(hash_seed=hash_seed, quantity=name):
                    self.assertNotEqual(quantity["verdict"], "failed", quantity["note"])
            self.assertEqual(quantities["noise_1e-08_delta_0"]["verdict"], "passed",
                             quantities["noise_1e-08_delta_0"]["note"])

    def test_gradient_check_kinks(self):
        """A kink 3e-6 away passes (the sweep reaches steps below it); a kink
        exactly at x0 (max at a tie) is not passed: the central differences
        converge to the average of the one-sided slopes."""
        quantities = self.run_shape([("abs", 3e-6, 0.0, 0.0), ("max_tie", 0.0, 0.0, 0.0)])
        self.assertEqual(quantities["abs_3e-06_0_0"]["verdict"], "passed")
        self.assertNotEqual(quantities["max_tie_0_0_0"]["verdict"], "passed")

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

    def test_boozer_template_gradients_pass_at_seed_3(self):
        """R5-4: the Boozer template's gradient check exited 1 at seed 3."""
        generate_problem(self.scratch, "boozer_single_stage")
        completed = run_python([str(SCRIPTS_DIR / "gradient_check.py"), "--smoke", "--seed", "3"],
                               cwd=self.scratch, problem_dir=self.scratch)
        summary = result_line(completed, "GRADIENT_CHECK ")
        self.assertEqual(completed.returncode, 0,
                         {quantity["quantity"]: quantity["note"] for quantity in summary["quantities"]
                          if not quantity["passed"]})

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
