"""The simsopt-alm-setup skill (``.claude/skills/simsopt-alm-setup``) against
the package it documents.

Drift guards: every API name the skill's references, templates and scripts
use resolves in ``simsopt_alm`` or the module the skill names; the
termination reasons of ``references/termination.md`` are exactly those the
package source can return (and ``success`` exactly its converge decisions);
``references/settings.md`` lists every ``ALMSettings`` field with its default;
the Python floor matches the package's ``pyproject.toml`` and the install
commands its location in the repository; and ``docs/alm_setup_guide.md`` is
what ``build_guide.py`` generates.

Runs: the scripts and templates run in fresh interpreters of this test's
Python, which imports the installed ``simsopt_alm`` and simsopt, in scratch
directories (a template's ``finish`` writes output). A template runs as the
skill generates it: copied to ``alm_problem.py`` next to ``run_alm.py``.
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
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

import simsopt_alm as alm
import simsopt_alm.signed_constraints as signed_constraints
from simsopt_alm import ALMResult, ALMSettings, checkpoint, control, history, policy

# The package directory (pyproject.toml, src/, tests/) and the repository
# around it, which holds the skill and the generated guide.
DISTRIBUTION_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DISTRIBUTION_ROOT.parents[1]
PACKAGE_DIR = Path(alm.__file__).resolve().parent
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
    "simsopt_alm": alm,
    "simsopt_alm.checkpoint": checkpoint,
    "simsopt_alm.control": control,
    "simsopt_alm.history": history,
    "simsopt_alm.policy": policy,
    "simsopt_alm.signed_constraints": signed_constraints,
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

from simsopt_alm import ALMPhysics

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


# Crucible rounds 6 and 7: noisy quantities, like an iterative inner solve.
# Each has the value (1 + sum x + sum x^2) (1 + NOISE u), u drawn from
# Python's hash of the point (so PYTHONHASHSEED picks the noise): uniform in
# [-1, 1], or (round 7) "tri" in {-1, 0, 1} or "cauchy" (Student t, 1 dof);
# it claims (1 + DELTA) times the true gradient. The objective is the
# round-6 repro itself (``hash(x.tobytes())``, x0 = (0, 0)); each row salts
# its hash with its index. Rows are (NOISE, DELTA) or (NOISE, DELTA, KIND).
NOISY_PROBE_PROBLEM = """\
from types import SimpleNamespace

import numpy as np

from simsopt_alm import ALMPhysics

OBJECTIVE, ROWS, NAMES = OBJECTIVE_VALUE, ROWS_VALUE, NAMES_VALUE


def noisy(x, salt, noise, delta, kind="uniform"):
    key = x.tobytes() if salt is None else (x.tobytes(), salt)
    random = np.random.RandomState(abs(hash(key)) % (2 ** 32))
    draw = {"uniform": random.uniform(-1.0, 1.0), "tri": float(random.randint(-1, 2)),
            "cauchy": random.standard_t(1)}[kind]
    return (1.0 + x.sum() + (x ** 2).sum()) * (1.0 + noise * draw), (1.0 + delta) * (1.0 + 2.0 * x)


def physics(x):
    x = np.asarray(x, dtype=float)
    value, grad = noisy(x, None, *OBJECTIVE)
    rows = [noisy(x, index, *row) for index, row in enumerate(ROWS)]
    return ALMPhysics(base_value=float(value), base_grad=grad,
                      constraint_values=np.array([float(value) for value, _grad in rows]),
                      constraint_grads=tuple(grad for _value, grad in rows))


def build_problem(smoke=False):
    return SimpleNamespace(name="noisy", x0=np.zeros(2), physics=physics, constraint_names=NAMES)
"""
# Crucible round 7: quantities whose finite differences depend on the step
# size, at x0 = X0_VALUE (every dof), with f = sum(x). Rows are
# (case, parameter, claim): "fp32term" adds parameter sum(x^2) evaluated in
# float32; "quant" adds sum(x^2) rounded to parameter; "stale" adds y from an
# inner solve warm-started at y(x0) and stopped at tolerance parameter;
# "hinge_active" is max(0, x0 - parameter), active just past its kink;
# "bigval" is parameter + 1e-3 sum(x). Round 8: "fp32row" is
# float32(1 + parameter sum(sin x)) (a quantity evaluated in float32);
# "fp32value" is float32(parameter + sum(x)); "ripple" and "ripple_fine" add
# A sum(sin(parameter x)) with A * parameter = 3e-4 and 1e-5; "tanh_step" adds
# A sum(tanh(parameter (x - x0))) with A * parameter = 1e-5. The claim is
# "full" (right), "partial" (the flat, rippling or stepping term's derivative
# left out), "zero" or "double".
STEP_PROBE_PROBLEM = """\
from types import SimpleNamespace

import numpy as np

from simsopt_alm import ALMPhysics

X0, ROWS, NAMES = X0_VALUE, ROWS_VALUE, NAMES_VALUE


def row(x, case, parameter, claim):
    if case == "fp32term":
        term = np.float32(parameter) * np.sum(np.asarray(x, np.float32) ** 2, dtype=np.float32)
        value, full, partial = x.sum() + float(term), 1.0 + 2.0 * parameter * x, np.ones_like(x)
    elif case == "quant":
        value = x.sum() + parameter * np.round((x ** 2).sum() / parameter)
        full, partial = 1.0 + 2.0 * x, np.ones_like(x)
    elif case == "stale":
        start = np.full_like(x, X0)
        target, inner = x.sum() + 0.1 * (x ** 2).sum(), start.sum() + 0.1 * (start ** 2).sum()
        while abs(target - inner) >= parameter:
            inner += 0.5 * (target - inner)
        value, full, partial = x.sum() + inner, 2.0 + 0.2 * x, np.ones_like(x)
    elif case == "hinge_active":
        value, full, partial = max(0.0, x[0] - parameter), np.eye(len(x))[0], np.zeros_like(x)
    elif case == "fp32row":
        value = float(np.float32(1.0 + parameter * np.sin(x).sum()))
        full, partial = parameter * np.cos(x), np.zeros_like(x)
    elif case == "fp32value":
        value, full, partial = float(np.float32(parameter + x.sum())), np.ones_like(x), np.zeros_like(x)
    elif case in ("ripple", "ripple_fine"):
        amplitude = (3e-4 if case == "ripple" else 1e-5) / parameter
        value = x.sum() + amplitude * np.sin(parameter * x).sum()
        full, partial = 1.0 + amplitude * parameter * np.cos(parameter * x), np.ones_like(x)
    elif case == "tanh_step":
        amplitude, start = 1e-5 / parameter, np.full_like(x, X0)
        value = np.sin(x).sum() + amplitude * np.tanh(parameter * (x - start)).sum()
        full = np.cos(x) + amplitude * parameter / np.cosh(parameter * (x - start)) ** 2
        partial = np.cos(x)
    else:  # "bigval"
        value, full, partial = parameter + 1e-3 * x.sum(), 1e-3 * np.ones_like(x), np.zeros_like(x)
    return float(value), {"full": full, "partial": partial, "zero": np.zeros_like(x), "double": 2.0 * full}[claim]


def physics(x):
    x = np.asarray(x, dtype=float)
    rows = [row(x, *spec) for spec in ROWS]
    return ALMPhysics(base_value=float(x.sum()), base_grad=np.ones_like(x),
                      constraint_values=np.array([value for value, _grad in rows]),
                      constraint_grads=tuple(grad for _value, grad in rows))


def build_problem(smoke=False):
    return SimpleNamespace(name="step", x0=np.full(2, X0), physics=physics, constraint_names=NAMES)
"""
# Crucible round 7 (C4): the round's own noisy objective, (1 + sum x + sum x^2
# + sum sin x) (1 + LEVEL u) with u from sha256 of the point and TRIAL
# ("tri" in {-1, 0, 1}, "cauchy" Student t with 1 dof), at x0 drawn from
# TRIAL, with the right gradient; checked with --seed TRIAL + 1.
NOISE_TRIAL_PROBLEM = """\
import hashlib
from types import SimpleNamespace

import numpy as np

from simsopt_alm import ALMPhysics

KIND, LEVEL, TRIAL = KIND_VALUE, LEVEL_VALUE, TRIAL_VALUE


def physics(x):
    x = np.asarray(x, dtype=float)
    digest = hashlib.sha256(x.tobytes() + bytes([TRIAL])).digest()[:4]
    random = np.random.RandomState(int.from_bytes(digest, "little"))
    draw = random.standard_t(1) if KIND == "cauchy" else float(random.randint(-1, 2))
    value = 1.0 + x.sum() + (x ** 2).sum() + np.sin(x).sum()
    return ALMPhysics(base_value=float(value * (1.0 + LEVEL * draw)), base_grad=1.0 + 2.0 * x + np.cos(x),
                      constraint_values=np.zeros(0), constraint_grads=())


def build_problem(smoke=False):
    return SimpleNamespace(name="noise_trial", x0=np.random.RandomState(TRIAL).uniform(-1.0, 1.0, 3),
                           physics=physics, constraint_names=())
"""
# Round-7 trials whose right claims the round-6 checker failed.
NOISE_TRIALS = (("tri", 1e-3, 13), ("cauchy", 1e-4, 33))
# Round-6 seeds of the noise.
NOISY_HASH_SEEDS = tuple(range(8))


def probe_row_names(specs: Iterable[tuple]) -> Tuple[str, ...]:
    """The constraint name of each probe row spec (``NAMES_VALUE`` of the
    probe problems): its items joined by "_", numbers in ``:g`` form."""
    return tuple("_".join(item if isinstance(item, str) else f"{item:g}" for item in spec) for spec in specs)


def noisy_row_names(rows: Iterable[tuple]) -> Tuple[str, ...]:
    """``NOISY_PROBE_PROBLEM``'s row names: noise_NOISE_delta_DELTA[_KIND]."""
    return probe_row_names(("noise",) + tuple(row[:1]) + ("delta",) + tuple(row[1:]) for row in rows)


# f alone, for the round-off cases of Crucible round 2: a huge value with a
# small derivative (claimed 0 or right), and a derivative A near the Taylor
# test's absolute 1e-10 floor (claimed 0 or right).
ROUNDOFF_PROBE_PROBLEM = """\
from types import SimpleNamespace

import numpy as np

from simsopt_alm import ALMPhysics

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

from simsopt_alm import ALMPhysics

ROWS, NAMES = ROWS_VALUE, NAMES_VALUE


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
    return SimpleNamespace(name="sweep", x0=np.zeros(2), physics=physics, constraint_names=NAMES)
"""


# Crucible rounds 4 and 5: rows whose verdicts must be attributed correctly.
# Each row is (case, parameter, relative error of the claimed gradient,
# absolute error added to it); x0 = (0, 0) and f = x0 + x1 (an O(1) gradient,
# which sets the absolute floor of a derivative).
SHAPE_PROBE_PROBLEM = """\
from types import SimpleNamespace

import numpy as np

from simsopt_alm import ALMPhysics

ROWS, NAMES = ROWS_VALUE, NAMES_VALUE


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
    return SimpleNamespace(name="shape", x0=np.zeros(2), physics=physics, constraint_names=NAMES)
"""


# A problem module with rows NAMES_VALUE and values VALUES_VALUE + x[0] at
# x0 = 0 (exact gradients); its one probe expects the first name violated.
CONTRACT_PROBE_PROBLEM = """\
from types import SimpleNamespace

import numpy as np

from simsopt_alm import ALMPhysics

NAMES = NAMES_VALUE
VALUES = VALUES_VALUE


def physics(x):
    x = np.asarray(x, dtype=float)
    return ALMPhysics(base_value=1.0 + x[0] + float(x @ x), base_grad=np.array([1.0, 0.0]) + 2.0 * x,
                      constraint_values=np.array(VALUES) + x[0],
                      constraint_grads=tuple(np.array([1.0, 0.0]) for _ in VALUES))


def build_problem(smoke=False):
    probe = SimpleNamespace(label="x0", x=np.zeros(2), violated=NAMES[:1], satisfied=())
    return SimpleNamespace(name="contract", x0=np.zeros(2), physics=physics, constraint_names=NAMES,
                           shared_source_rows=(), sign_probes=lambda: (probe,))
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
    python_path = [] if problem_dir is None else [str(problem_dir)]
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
    # An empty line is a blank context line (as git apply and GNU patch read
    # it), so the diff carries no trailing whitespace.
    old = "\n".join(line[1:] for line in body if line[:1] in ("", " ", "-")) + "\n"
    new = "\n".join(line[1:] for line in body if line[:1] in ("", " ", "+")) + "\n"
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
            for dotted in re.findall(r"`(simsopt(?:_alm(?:\.\w+)*|(?:\.\w+)+))`", text):
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
        requires = re.search(r'^requires-python = ">=([\d.]+)"',
                             (DISTRIBUTION_ROOT / "pyproject.toml").read_text(), re.MULTILINE).group(1)
        self.assertEqual(check_env.MIN_PYTHON, tuple(int(part) for part in requires.split(".")))

    def test_install_commands_name_this_package(self):
        self.assertEqual(REPO_ROOT / check_env.PACKAGE_SUBDIRECTORY, DISTRIBUTION_ROOT)
        name = re.search(r'^name = "([\w-]+)"', (DISTRIBUTION_ROOT / "pyproject.toml").read_text(),
                         re.MULTILINE).group(1)
        self.assertEqual(check_env.DISTRIBUTION, name)
        text = (REFERENCES_DIR / "install.md").read_text()
        self.assertIn(f'install "{check_env.install_url("<fork-url>")}"', text)
        self.assertIn(f"git clone -b {check_env.ALM_BRANCH} <fork-url> <clone>", text)
        self.assertIn(f"install -e <clone>/{check_env.PACKAGE_SUBDIRECTORY.as_posix()}", text)

    def test_guide_is_generated_from_the_skill(self):
        self.assertTrue(build_guide.GUIDE_PATH.exists(), "docs/alm_setup_guide.md is missing")
        self.assertTrue(build_guide.GUIDE_PATH.read_text() == build_guide.render_guide(),
                        "docs/alm_setup_guide.md is stale: run "
                        ".claude/skills/simsopt-alm-setup/scripts/build_guide.py")

    def test_skill_files_and_guide_pass_the_whitespace_check(self):
        """What ``git diff --check`` rejects, over every skill file and the guide."""
        self.assertEqual(build_guide.whitespace_errors(), [])

    def test_whitespace_check_finds_what_git_diff_check_rejects(self):
        with tempfile.TemporaryDirectory() as scratch:
            directory = Path(scratch)
            cases = {
                "clean.md": ("a\n\tb\n", []),
                "trailing.md": ("a \nb\t\n", [1, 2]),
                "space_before_tab.py": ("x = 1\n \ty = 2\n", [2]),
                "blank_at_end.md": ("a\n\n\n", [2]),
                "conflict.md": ("<<<<<<< ours\n=======\n", [1, 2]),
            }
            for name, (text, _lines) in cases.items():
                (directory / name).write_text(text)
            errors = build_guide.whitespace_errors(sorted(directory.iterdir()))
            for name, (_text, lines) in cases.items():
                with self.subTest(file=name):
                    self.assertEqual([error.split(":")[1] for error in errors
                                      if Path(error.split(":")[0]).name == name],
                                     [str(line) for line in lines])

    def test_hard_rows_are_described_as_sampled_extrema(self):
        """R16-05: the hard values are extrema over the sampled points; no
        skill file, template or the guide calls them exact feasibility or
        claims an engineering tolerance, and the kernel reference asks for a
        resolution check."""
        texts = {**skill_markdown(), **skill_python(),
                 "guide": (REPO_ROOT / "docs" / "alm_setup_guide.md").read_text()}
        for name, text in texts.items():
            with self.subTest(file=name):
                self.assertNotRegex(text, r"(?i)exactly feasible|engineering tolerance")
        api = skill_markdown()["references/api.md"]
        self.assertIn("sampled quadrature points", api)
        self.assertIn("higher resolution", api)

    def test_scaling_advice_divides_by_a_positive_scale(self):
        """R17-02: dividing f by f(x0) maximizes a negative f and divides by 0
        at a zero start; every piece of advice names a positive scale."""
        texts = {**skill_markdown(), "guide": (REPO_ROOT / "docs" / "alm_setup_guide.md").read_text()}
        for name, text in texts.items():
            with self.subTest(file=name):
                self.assertNotRegex(text, r"(?i)divid\w* (it|f|the objective) by its initial value")
        termination = skill_markdown()["references/termination.md"]
        self.assertIn("divide it by a positive scale: |f(x0)|, or a chosen positive reference "
                      "scale when f(x0) = 0", termination)

    def test_docs_state_the_review_round_fixes(self):
        """The cross-lab review of the package: converged needs both hybrid
        channels within feasibility_tol; maxiter is an exact whole-call
        budget; the history's constraint_values is the signed g; the penalty
        is one scalar; an outer evaluation flagged nonfinite_evaluation
        raises; an evaluator may reuse its arrays."""
        markdown = skill_markdown()
        texts = {**markdown, "guide": (REPO_ROOT / "docs" / "alm_setup_guide.md").read_text()}
        for name, text in texts.items():
            with self.subTest(file=name):
                self.assertNotIn("that row's entry when the penalty is per row", text)
                self.assertNotIn("share what was left at its start", text)
                self.assertNotIn("only a custom continuation policy ends an outer iteration this way", text)
                self.assertNotIn("`result.nit <= maxiter`", text)
        termination = markdown["references/termination.md"]
        self.assertIn("the positive part of the g that L uses", termination)
        self.assertIn("No hybrid signal mismatch judged at `feasibility_tol`", termination)
        self.assertIn("ρ = `result.penalty` (one scalar penalty shared by every row)", termination)
        self.assertIn("never exceeded: `result.nit - total_inner_iterations <=\nmaxiter`", termination)
        self.assertIn("raising `max_outer_iterations` does not help", termination)
        self.assertIn("so the call never runs more", markdown["references/settings.md"])
        self.assertIn("resumed checkpoint's `total_inner_iterations`", markdown["references/pitfalls.md"])
        api = markdown["references/api.md"]
        self.assertIn("`violation_values` the per-row violation", api)
        self.assertIn("no `.copy()` is needed", api)
        self.assertIn("a hybrid problem can converge only where, at every active row", api)
        self.assertIn("A non-finite value, or `nonfinite_evaluation=True`, at a trial", api)
        self.assertIn("`success` needs both channels within\n`feasibility_tol`", api)
        self.assertIn("`converged` needs\n   the smooth and the hard rows within `feasibility_tol`", markdown["SKILL.md"])

    def test_equations_name_the_quantities_in_use(self):
        """R16-06: the complementarity gap is f minus the ordinary Lagrangian
        at the shifted multipliers (core._complementarity_gap), and the Boozer
        ratio divides by the integral of B_QA squared (simsopt's
        NonQuasiSymmetricRatio)."""
        settings = skill_markdown()["references/settings.md"]
        self.assertNotIn("the gap is f - L", settings)
        self.assertIn("f - ℓ(x, λ⁺), where ℓ(x, λ⁺) = f + Σ λ⁺_i g_i is the ordinary Lagrangian", settings)
        boozer = skill_python()["templates/boozer_single_stage.py"]
        self.assertIn(r"J = (\int_S B_nonQA^2 dS) / (\int_S B_QA^2 dS)", boozer)

    def test_the_alm_workflow_pins_every_action_to_a_commit(self):
        """R16-07: a tag can move; each ``uses:`` of the package's workflow
        names a full commit SHA, with its version tag in a comment."""
        workflow = (REPO_ROOT / ".github" / "workflows" / "alm.yml").read_text()
        uses = re.findall(r"^\s*-?\s*uses:\s*(.+)$", workflow, flags=re.MULTILINE)
        self.assertTrue(uses, "the workflow names no action")
        for reference in uses:
            with self.subTest(uses=reference):
                self.assertRegex(reference, r"^[\w.-]+/[\w.-]+@[0-9a-f]{40} # v\d+(\.\d+)*\b")

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

    CLONE = Path("/clones/simsopt")

    @classmethod
    def report(cls, *, blockers=(), alm_file=None, checkout=False) -> dict:
        return {"blockers": list(blockers),
                "alm": {"importable": alm_file is not None, "file": alm_file},
                "checkout": {"path": str(cls.CLONE),
                             "package": str(cls.CLONE / check_env.PACKAGE_SUBDIRECTORY)} if checkout else None}

    def test_routes(self):
        clone_file = str(self.CLONE / check_env.PACKAGE_SUBDIRECTORY / "src" / "simsopt_alm" / "__init__.py")
        other_file = "/site-packages/simsopt_alm/__init__.py"
        cases = {
            "blocked": self.report(blockers=["Python too old"], alm_file=other_file),
            "ready": self.report(alm_file=other_file),
            "ready with the clone's package": self.report(alm_file=clone_file, checkout=True),
            "editable": self.report(checkout=True),
            "editable over another install": self.report(alm_file=other_file, checkout=True),
            "install": self.report(),
        }
        for case, report in cases.items():
            with self.subTest(case=case):
                self.assertEqual(check_env.choose_route(report), case.split(" ")[0])

    def test_template_readiness_needs_the_signed_constraints_for_the_coil_templates(self):
        def modules(alm_importable, signed_importable):
            return {"alm": {"importable": alm_importable}, "signed_constraints": {"importable": signed_importable}}

        self.assertEqual(check_env.template_readiness(modules(True, False)),
                         {"generic": True, "stage2": False, "boozer_single_stage": False})
        self.assertEqual(set(check_env.template_readiness(modules(True, True)).values()), {True})
        self.assertEqual(set(check_env.template_readiness(modules(False, False)).values()), {False})

    def test_template_readiness_names_each_templates_checked_imports(self):
        """``check_env.TEMPLATE_MODULES`` lists, for every problem template,
        exactly the checked modules (the solver, the signed constraints) it imports."""
        problem_templates = {path.stem for path in TEMPLATES_DIR.glob("*.py")} - {"run_alm"}
        self.assertEqual(set(check_env.TEMPLATE_MODULES), problem_templates)
        for name in sorted(problem_templates):
            tree = ast.parse((TEMPLATES_DIR / f"{name}.py").read_text())
            imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)} | {
                alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
            with self.subTest(template=name):
                self.assertEqual(
                    {module for module in check_env.ALM_MODULES if module in imported},
                    set(check_env.TEMPLATE_MODULES[name]))

    def test_an_import_failure_names_the_missing_module_or_parent(self):
        """Absent package, absent parent package, and a broken import differ."""
        module = check_env.SIGNED_CONSTRAINTS_MODULE
        cases = {
            "ModuleNotFoundError: No module named 'simsopt_alm.signed_constraints'": module,
            "ModuleNotFoundError: No module named 'simsopt_alm'": "simsopt_alm",
            "ModuleNotFoundError: No module named 'simsopt'": None,
            "ModuleNotFoundError: No module named 'scipy'": None,
            "ModuleNotFoundError: No module named 'simsopt_alm.signed_constraintsx'": None,
            "ImportError: cannot import name 'Derivative' from 'simsopt._core.derivative'": None,
            "SyntaxError: invalid syntax": None,
        }
        for error, missing in cases.items():
            with self.subTest(error=error):
                self.assertEqual(check_env.missing_import(error, module), missing)

    def test_an_absent_package_is_not_a_blocker_and_a_broken_one_is(self):
        for module in check_env.ALM_MODULES:
            with self.subTest(module=module):
                self.assertIsNone(check_env.import_blocker(
                    {"importable": False, "error": "ModuleNotFoundError: No module named 'simsopt_alm'"}, module))
                self.assertIsNone(check_env.import_blocker({"importable": True, "error": None}, module))
                self.assertIn(f"{module} exists but fails to import", check_env.import_blocker(
                    {"importable": False, "error": "ImportError: cannot import name 'nnls' from 'scipy.optimize'"},
                    module))

    def test_the_default_install_is_the_published_alm_library_package(self):
        self.assertEqual(check_env.FORK_URL, "https://github.com/jungdaesuh/simsopt.git")
        self.assertEqual(check_env.install_url(check_env.FORK_URL),
                         "git+https://github.com/jungdaesuh/simsopt.git@alm-library#subdirectory=packages/simsopt-alm")
        for path in sorted(SKILL_DIR.rglob("*")) + [REPO_ROOT / "docs" / "alm_setup_guide.md"]:
            if path.is_file() and path.suffix in (".md", ".py"):
                with self.subTest(path=path.name):
                    self.assertNotIn("<owner>", path.read_text(encoding="utf-8"))


class SkillScriptsRunTests(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.scratch = Path(scratch.name)

    def check_env(self, python: str, *arguments: str) -> tuple:
        completed = run_python([str(SCRIPTS_DIR / "check_env.py"), "--python", python, *arguments],
                               cwd=self.scratch)
        return completed, result_line(completed, check_env.RESULT_PREFIX)

    def test_check_env_reports_this_interpreter_ready(self):
        completed, report = self.check_env(sys.executable)
        self.assertEqual(report["route"], "ready", report)
        self.assertEqual(completed.returncode, 0)
        self.assertEqual(report["blockers"], [])
        self.assertTrue(report["python"]["meets_floor"])
        self.assertEqual(report["templates"], {"generic": True, "stage2": True, "boozer_single_stage": True})
        # Provenance: the files the imports resolve to and the installed distribution.
        alm_file = os.path.realpath(alm.__file__)
        self.assertEqual(report["alm"]["file"], alm_file)
        self.assertEqual(report["signed_constraints"]["file"], os.path.realpath(signed_constraints.__file__))
        self.assertEqual(report["distribution"]["version"],
                         re.search(r'^version = "([^"]+)"', (DISTRIBUTION_ROOT / "pyproject.toml").read_text(),
                                   re.MULTILINE).group(1))
        self.assertIn(f"import simsopt_alm: {alm_file}", completed.stdout.splitlines())

    def make_repository(self, name: str, branch: str, sources: tuple) -> Path:
        """A git repository at ``scratch/name`` on ``branch`` with ``sources``
        committed (empty files), as a clone of the branch would hold them."""
        repository = self.scratch / name
        for source in sources:
            (repository / source).parent.mkdir(parents=True, exist_ok=True)
            (repository / source).write_text("")
        for command in (["init", "-q", "-b", branch], ["add", "-A"],
                        ["-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false",
                         "commit", "-q", "--no-verify", "-m", "clone"]):
            subprocess.run(["git", "-C", str(repository), *command], check=True, capture_output=True)
        return repository

    def test_check_env_accepts_only_a_clone_of_the_branch(self):
        """``--checkout``: a valid clone routes to ``editable`` (this
        interpreter imports another simsopt_alm); a new or empty directory is
        a place to clone into; anything else is a blocker with the clone
        command, never ``editable``."""
        clone = self.make_repository("clone", check_env.ALM_BRANCH, check_env.CLONE_SOURCES)
        completed, report = self.check_env(sys.executable, "--checkout", str(clone))
        self.assertEqual(report["route"], "editable", report)
        self.assertEqual(report["blockers"], [])
        self.assertTrue(report["checkout"]["is_alm_clone"])
        head = subprocess.run(["git", "-C", str(clone), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
        self.assertIn(f"checkout: {clone.resolve()} at {head} (branch {check_env.ALM_BRANCH})",
                      completed.stdout.splitlines())

        (self.scratch / "empty").mkdir()
        for target in (self.scratch / "new", self.scratch / "empty"):
            with self.subTest(clone_target=target.name):
                _completed, report = self.check_env(sys.executable, "--checkout", str(target))
                self.assertEqual(report["route"], "editable", report)
                self.assertTrue(report["checkout"]["clone_target"])

        fake = self.scratch / "fake"
        (fake / check_env.PACKAGE_SUBDIRECTORY).mkdir(parents=True)
        (fake / check_env.PACKAGE_SUBDIRECTORY / "pyproject.toml").write_text("")
        cases = {
            "only pyproject.toml, no git": (fake, "is not a git repository"),
            "wrong branch": (self.make_repository("other-branch", "main", check_env.CLONE_SOURCES),
                             "on branch 'main', not 'alm-library'"),
            "not the top level": (clone / "packages", "not the top level of its git repository"),
            "no package sources": (self.make_repository("no-sources", check_env.ALM_BRANCH, (Path("README"),)),
                                   "packages/simsopt-alm/src/simsopt_alm/__init__.py is missing"),
        }
        for case, (directory, problem) in cases.items():
            with self.subTest(case=case):
                completed, report = self.check_env(sys.executable, "--checkout", str(directory))
                self.assertEqual(report["route"], "blocked", report)
                self.assertEqual(completed.returncode, 1)
                self.assertFalse(report["checkout"]["is_alm_clone"])
                self.assertEqual(len(report["blockers"]), 1, report)
                self.assertIn("is not a clone of the alm-library branch", report["blockers"][0])
                self.assertIn(problem, report["blockers"][0])
                self.assertIn(f"git clone -b alm-library {check_env.FORK_URL} <new directory>",
                              report["blockers"][0])

    def isolated_python(self) -> Path:
        """An interpreter without site-packages or PYTHONPATH: no simsopt and
        no package, as in a fresh environment."""
        wrapper = self.scratch / "fresh-python"
        wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" -I -S "$@"\n')
        wrapper.chmod(0o755)
        return wrapper

    def test_check_env_in_a_fresh_interpreter_offers_the_install_routes(self):
        fresh = str(self.isolated_python())
        completed, report = self.check_env(fresh)
        self.assertEqual(report["route"], "install", report)
        self.assertEqual(report["blockers"], [])
        self.assertEqual(completed.returncode, 1)
        self.assertEqual(report["install_url"], check_env.install_url(check_env.FORK_URL))
        self.assertFalse(report["simsopt"]["importable"])
        self.assertEqual(set(report["templates"].values()), {False})
        self.assertTrue(any("only the generic template" in note for note in report["notes"]), report["notes"])

        new_clone = self.scratch / "new-clone"
        _completed, report = self.check_env(fresh, "--checkout", str(new_clone))
        self.assertEqual(report["route"], "editable", report)
        self.assertFalse(report["checkout"]["exists"])

    def run_contract_probe(self, script: str, names: tuple, values: tuple) -> subprocess.CompletedProcess:
        source = (CONTRACT_PROBE_PROBLEM.replace("NAMES_VALUE", repr(names))
                  .replace("VALUES_VALUE", repr(values)))
        (self.scratch / "alm_problem.py").write_text(source)
        return run_python([str(SCRIPTS_DIR / script)], cwd=self.scratch, problem_dir=self.scratch)

    def test_checks_reject_duplicate_constraint_names(self):
        """Two rows named alike: a name-keyed check would see only one of them
        (the last), so a wrong row could pass. Rows (-1, +1), probe expects violated."""
        for script, prefix in (("sign_check.py", "SIGN_CHECK "), ("gradient_check.py", "GRADIENT_CHECK ")):
            with self.subTest(script=script):
                completed = self.run_contract_probe(script, ("dup", "dup"), (-1.0, 1.0))
                self.assertNotEqual(completed.returncode, 0, completed.stdout[-2000:])
                self.assertNotIn(prefix, completed.stdout)
                self.assertIn("constraint_names must be unique", completed.stderr)

    def test_checks_reject_a_row_count_that_differs_from_the_names(self):
        for script, prefix in (("sign_check.py", "SIGN_CHECK "), ("gradient_check.py", "GRADIENT_CHECK ")):
            with self.subTest(script=script):
                completed = self.run_contract_probe(script, ("only",), (-1.0, 1.0))
                self.assertNotEqual(completed.returncode, 0, completed.stdout[-2000:])
                self.assertNotIn(prefix, completed.stdout)
                self.assertIn("2 constraint values and 2 constraint gradients for 1 constraint_names",
                              completed.stderr)

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
        """Exact gradients pass; zeroed or scaled gradients on changing
        quantities fail (Crucible round 1). A problem whose objective and row
        are constant has no derivative scale to resolve zero against: NOT
        TESTED (round 7)."""
        cases = {"exact": {"f": "passed", "row": "passed"}, "constant": {"f": "not_tested", "row": "not_tested"},
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

    def assert_reported_rows(self, reported: Iterable[str], expected: Iterable[str]) -> None:
        """The checker judged exactly the requested rows, in order: a verdict
        loop over fewer (or no) rows would check nothing."""
        self.assertEqual(list(reported), list(expected), "the checker did not report the requested rows")

    def run_sweep(self, rows: list, *arguments: str) -> Dict[str, str]:
        (self.scratch / "alm_problem.py").write_text(SWEEP_PROBE_PROBLEM.replace("ROWS_VALUE", repr(rows))
                                                     .replace("NAMES_VALUE", repr(probe_row_names(rows))))
        completed = run_python([str(SCRIPTS_DIR / "gradient_check.py"), *arguments], cwd=self.scratch,
                               problem_dir=self.scratch)
        verdicts = self.verdicts(completed)
        verdicts.pop("f")
        return verdicts

    def test_gradient_check_fails_wrong_gradients_dominated_by_truncation(self):
        """q = sum(x) + K sum(x^3), K up to 1e10: every gradient 0.1% to 5% wrong
        fails and every right one passes (Crucible round 3)."""
        rows = [("cubic", k, delta) for k in SWEEP_CUBIC_K for delta in (0.0,) + SWEEP_DELTAS]
        verdicts = self.run_sweep(rows)
        self.assert_reported_rows(verdicts, probe_row_names(rows))
        for name, verdict in verdicts.items():
            self.assertEqual(verdict, "passed" if name.endswith("_0") else "failed", name)

    def test_gradient_check_fails_wrong_gradients_on_offset_values(self):
        """q = F + sum(x), F up to 1e8: every gradient 0.1% to 5% wrong fails and
        every right one passes (Crucible round 3)."""
        rows = [("offset", f, delta) for f in SWEEP_OFFSET_F for delta in (0.0,) + SWEEP_DELTAS]
        verdicts = self.run_sweep(rows)
        self.assert_reported_rows(verdicts, probe_row_names(rows))
        for name, verdict in verdicts.items():
            self.assertEqual(verdict, "passed" if name.endswith("_0") else "failed", name)

    def run_shape(self, rows: list, *arguments: str) -> Dict[str, dict]:
        (self.scratch / "alm_problem.py").write_text(SHAPE_PROBE_PROBLEM.replace("ROWS_VALUE", repr(rows))
                                                     .replace("NAMES_VALUE", repr(probe_row_names(rows))))
        completed = run_python([str(SCRIPTS_DIR / "gradient_check.py"), *arguments], cwd=self.scratch,
                               problem_dir=self.scratch)
        self.last_completed = completed
        summary = result_line(completed, "GRADIENT_CHECK ")
        return {quantity["quantity"]: quantity for quantity in summary["quantities"] if quantity["quantity"] != "f"}

    def assert_verdicts(self, quantities: Dict[str, dict], expected: Dict[str, str]) -> None:
        self.assertEqual({name: quantity["verdict"] for name, quantity in quantities.items()}, expected,
                         {name: quantity["note"] for name, quantity in quantities.items()})

    def test_gradient_check_passes_flat_and_zero_rows(self):
        """A constant, a zero-derivative and a clipped row inactive over the whole
        sweep pass a zero claim through a zero window; a clearly nonzero claim
        fails. A clipped row whose bound the sweep reaches (1 or 1e-4 away) is
        NOT TESTED (round 8): its larger steps resolve the active slope, as a
        float32 row's resolve the slope its small steps cannot. A constant
        of 1e4 passes a claim inside the round-off band of its values (R6-F5).
        An exactly zero row has a band of 0, so a round-off-size claim of 1e-17
        cannot pass (round 7: a nonzero claim never passes through a floor);
        the round-off floor of the objective's gradient only keeps it from
        failing: NOT TESTED."""
        rows = [("const", 1.0, 0.0, 0.0), ("quadratic", 3.0, 0.0, 0.0), ("hinge", 100.0, 0.0, 0.0),
                ("hinge", 1.0, 0.0, 0.0), ("hinge", 1e-4, 0.0, 0.0),
                ("const", 1.0, 0.0, 1e-6), ("quadratic", 3.0, 0.0, 1e-6),
                ("zero", 0.0, 0.0, 1e-17), ("zero", 0.0, 0.0, 1e-6),
                ("const", 1e4, 0.0, 1e-12), ("const", 1e4, 0.0, 1e-6)]
        self.assert_verdicts(self.run_shape(rows), {
            "const_1_0_0": "passed", "quadratic_3_0_0": "passed", "hinge_100_0_0": "passed",
            "hinge_1_0_0": "not_tested", "hinge_0.0001_0_0": "not_tested",
            "const_1_0_1e-06": "failed", "quadratic_3_0_1e-06": "failed",
            "zero_0_0_1e-17": "not_tested", "zero_0_0_1e-06": "failed",
            "const_10000_0_1e-12": "passed", "const_10000_0_1e-06": "failed"})

    def test_gradient_check_passes_steep_smooth_rows(self):
        """R5-1: tanh(1e5 x0 - 0.3) and sin(K x)/K up to K = 1e5 with correct
        gradients pass (the sweep reaches steps below their feature size); 0.1%
        wrong ones fail."""
        rows = [("tanh", k, 0.0, 0.0) for k in (1e3, 1e5)] + [("sin", k, 0.0, 0.0) for k in (1e3, 1e4, 1e5)]
        rows += [(case, k, 1e-3, 0.0) for case, k, _relative, _absolute in rows]
        quantities = self.run_shape(rows)
        self.assert_reported_rows(quantities, probe_row_names(rows))
        for name, quantity in quantities.items():
            with self.subTest(row=name):
                self.assertEqual(quantity["verdict"], "failed" if "_0.001_" in name else "passed", quantity["note"])

    def test_gradient_check_verdict_does_not_depend_on_an_added_constant(self):
        """R5-2: c + sum(exp(x)) for c = 0, 1, 100: the same verdicts, right
        gradients passing and 0.1% wrong ones failing."""
        rows = [("exp", c, relative, 0.0) for c in (0.0, 1.0, 100.0) for relative in (0.0, 1e-3)]
        quantities = self.run_shape(rows)
        self.assert_reported_rows(quantities, probe_row_names(rows))
        for name, quantity in quantities.items():
            with self.subTest(row=name):
                self.assertEqual(quantity["verdict"], "failed" if "_0.001_" in name else "passed", quantity["note"])

    def test_gradient_check_does_not_decide_behind_a_cancellation(self):
        """(F + sum x) - F has value 0 and the round-off of F: identical values
        below some step (the computed function is flat there), the slope
        above it. The step ranges disagree, so the right claim, a 1% wrong
        one and a claim of 0 are all NOT TESTED (round 8: a zero range cannot
        judge while other steps resolve a change)."""
        rows = [("cancel", 1e8, 0.0, 0.0), ("cancel", 1e8, 0.01, 0.0), ("cancel", 1e11, 0.0, 0.0),
                ("cancel", 1e13, -1.0, 0.0), ("cancel", 1e13, 0.0, 0.0)]
        self.assert_verdicts(self.run_shape(rows), {
            "cancel_1e+08_0_0": "not_tested", "cancel_1e+08_0.01_0": "not_tested", "cancel_1e+11_0_0": "not_tested",
            "cancel_1e+13_-1_0": "not_tested", "cancel_1e+13_0_0": "not_tested"})

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
                  .replace("ROWS_VALUE", repr(rows)).replace("NAMES_VALUE", repr(noisy_row_names(rows))))
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
            self.assert_reported_rows(quantities, ("f",) + noisy_row_names(failing + unresolved))
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
            self.assert_reported_rows(quantities, ("f",) + noisy_row_names(rows))
            for name, quantity in quantities.items():
                with self.subTest(hash_seed=hash_seed, quantity=name):
                    self.assertNotEqual(quantity["verdict"], "failed", quantity["note"])
            self.assertEqual(quantities["noise_1e-08_delta_0"]["verdict"], "passed",
                             quantities["noise_1e-08_delta_0"]["note"])

    def test_gradient_check_kinks(self):
        """A kink 3e-6 away: steps above it converge to the average slope and
        steps below it to the derivative, so NOT TESTED (round 7). A kink
        exactly at x0 (max at a tie) is not passed: the central differences
        converge to the average of the one-sided slopes."""
        quantities = self.run_shape([("abs", 3e-6, 0.0, 0.0), ("max_tie", 0.0, 0.0, 0.0)])
        self.assertEqual(quantities["abs_3e-06_0_0"]["verdict"], "not_tested")
        self.assertNotEqual(quantities["max_tie_0_0_0"]["verdict"], "passed")

    def run_step(self, x0: float, rows: list) -> Dict[str, dict]:
        source = (STEP_PROBE_PROBLEM.replace("X0_VALUE", repr(x0)).replace("ROWS_VALUE", repr(rows))
                  .replace("NAMES_VALUE", repr(probe_row_names(rows))))
        (self.scratch / "alm_problem.py").write_text(source)
        completed = run_python([str(SCRIPTS_DIR / "gradient_check.py")], cwd=self.scratch,
                               problem_dir=self.scratch)
        summary = result_line(completed, "GRADIENT_CHECK ")
        return {quantity["quantity"]: quantity for quantity in summary["quantities"] if quantity["quantity"] != "f"}

    def test_gradient_check_does_not_decide_step_dependent_differences(self):
        """R7-1 (C1): a float32 term, a quantized term and a warm-started inner
        solve go flat at small steps, so large steps converge to the full
        derivative and small ones to the rest: the right claim is NOT TESTED
        (not FAIL) and one missing the flat term's derivative never passes."""
        rows = [(case, parameter, claim) for case, parameter in (("fp32term", 1.0), ("fp32term", 0.01),
                                                                 ("quant", 1e-7), ("stale", 1e-8))
                for claim in ("full", "partial")]
        quantities = self.run_step(0.5, rows)
        self.assert_reported_rows(quantities, probe_row_names(rows))
        for name, quantity in quantities.items():
            with self.subTest(row=name):
                if name.endswith("_full"):
                    self.assertEqual(quantity["verdict"], "not_tested", quantity["note"])
                    self.assertIn("depend on the step size", quantity["note"])
                else:
                    self.assertNotEqual(quantity["verdict"], "passed", quantity["note"])

    def test_gradient_check_does_not_fail_a_kink_just_past_x0(self):
        """R7-2 (C2): max(0, x0 - 1e-4) at x0 = 1e-4 + 1e-9: steps above the kink
        converge to half the slope, the two below it to the slope; the right
        claim is NOT TESTED, a zero or doubled one never passes."""
        quantities = self.run_step(0.000100001, [("hinge_active", 1e-4, claim) for claim in ("full", "zero", "double")])
        self.assertEqual(quantities["hinge_active_0.0001_full"]["verdict"], "not_tested",
                         quantities["hinge_active_0.0001_full"]["note"])
        for claim in ("zero", "double"):
            self.assertNotEqual(quantities[f"hinge_active_0.0001_{claim}"]["verdict"], "passed")

    def test_gradient_check_on_big_values_with_small_derivatives(self):
        """R7-4 (C5): F + 1e-3 sum(x) for F = 1e10 to 1e12 hides the change at
        small steps; claims of 0 and twice the truth never pass, and the right
        one never fails."""
        rows = [("bigval", value, claim) for value in (1e10, 1e11, 1e12) for claim in ("full", "zero", "double")]
        quantities = self.run_step(0.5, rows)
        self.assert_reported_rows(quantities, probe_row_names(rows))
        for name, quantity in quantities.items():
            with self.subTest(row=name):
                if name.endswith("_full"):
                    self.assertNotEqual(quantity["verdict"], "failed", quantity["note"])
                else:
                    self.assertNotEqual(quantity["verdict"], "passed", quantity["note"])

    def test_gradient_check_on_float32_quantities(self):
        """R8-1: float32(1 + S sum(sin x)) is flat at small steps; its round-off
        band is float32's, so a claim of 0 or twice the truth never passes and
        the right one is NOT TESTED, not failed. The same for float32(V +
        sum(x)) with V = 1e4."""
        rows = [("fp32row", value, claim) for value in (1e-1, 1e-2, 1e-3) for claim in ("full", "zero", "double")]
        rows += [("fp32value", 1e4, claim) for claim in ("full", "zero")]
        quantities = self.run_step(0.3, rows)
        self.assert_reported_rows(quantities, probe_row_names(rows))
        for name, quantity in quantities.items():
            with self.subTest(row=name):
                if name.endswith("_full"):
                    self.assertEqual(quantity["verdict"], "not_tested", quantity["note"])
                else:
                    self.assertNotEqual(quantity["verdict"], "passed", quantity["note"])

    def test_gradient_check_needs_every_converged_range_at_the_claim(self):
        """R8-2: at x0 = 0 a ripple or a sharp step of relative size 1e-5 to 3e-4
        moves the small-step differences off the smooth slope; a claim leaving that term
        out agrees with the large steps but never passes, and the right claim
        never fails."""
        rows = [(case, frequency, claim) for case, frequencies in (("ripple", (1e3, 3e4, 1e6)),
                                                                   ("ripple_fine", (1e3, 3e4)),
                                                                   ("tanh_step", (1e5,)))
                for frequency in frequencies for claim in ("full", "partial")]
        quantities = self.run_step(0.0, rows)
        self.assert_reported_rows(quantities, probe_row_names(rows))
        for name, quantity in quantities.items():
            with self.subTest(row=name):
                if name.endswith("_full"):
                    self.assertNotEqual(quantity["verdict"], "failed", quantity["note"])
                else:
                    self.assertNotEqual(quantity["verdict"], "passed", quantity["note"])

    def test_gradient_check_never_fails_right_gradients_under_biased_noise(self):
        """R7-3 (C4): noise that takes a few values ("tri", {-1, 0, 1}) or has
        heavy tails ("cauchy") can bias three steps alike; a right claim is
        never failed (the scatter of the smaller steps bounds the bias), and a
        wrong one never passes. The round-7 trials the round-6 checker failed
        come first."""
        for kind, level, trial in NOISE_TRIALS:
            source = (NOISE_TRIAL_PROBLEM.replace("KIND_VALUE", repr(kind)).replace("LEVEL_VALUE", repr(level))
                      .replace("TRIAL_VALUE", repr(trial)))
            (self.scratch / "alm_problem.py").write_text(source)
            completed = run_python([str(SCRIPTS_DIR / "gradient_check.py"), "--seed", str(trial + 1)],
                                   cwd=self.scratch, problem_dir=self.scratch)
            with self.subTest(kind=kind, level=level, trial=trial):
                self.assertNotEqual(self.verdicts(completed)["f"], "failed", completed.stdout[-2000:])
        rows = [(1e-3, 0.0, "tri"), (1e-4, 0.0, "cauchy"), (1e-4, 0.0, "uniform"),
                (1e-3, 0.5, "tri"), (1e-4, 0.5, "cauchy")]
        for hash_seed in NOISY_HASH_SEEDS:
            quantities = self.run_noisy((1e-8, 0.0), rows, hash_seed)
            self.assert_reported_rows(quantities, ("f",) + noisy_row_names(rows))
            for name, quantity in quantities.items():
                with self.subTest(hash_seed=hash_seed, quantity=name):
                    if "_delta_0_" in name:
                        self.assertNotEqual(quantity["verdict"], "failed", quantity["note"])
                    elif name != "f":
                        self.assertNotEqual(quantity["verdict"], "passed", quantity["note"])

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

    def test_checks_refuse_a_point_the_physics_flags_unusable(self):
        # minimize_alm raises at an outer evaluation flagged
        # nonfinite_evaluation=True; the checks refuse such a start or probe.
        source = replaced_once((TEMPLATES_DIR / "generic.py").read_text(),
                               "            constraint_grads=tuple(grad for _value, grad in rows),\n",
                               "            constraint_grads=tuple(grad for _value, grad in rows),\n"
                               "            extras={\"nonfinite_evaluation\": bool(np.array_equal(x, [3.0, 2.0]))},\n")
        generate_problem(self.scratch, "generic", source)
        gradient = run_python([str(SCRIPTS_DIR / "gradient_check.py")], cwd=self.scratch,
                              problem_dir=self.scratch)
        self.assertEqual(gradient.returncode, 1)
        self.assertEqual(set(self.verdicts(gradient).values()), {"nonfinite"})
        sign = run_python([str(SCRIPTS_DIR / "sign_check.py")], cwd=self.scratch, problem_dir=self.scratch)
        summary = result_line(sign, "SIGN_CHECK ")
        self.assertEqual(sign.returncode, 1)
        self.assertEqual([probe["failures"][:1] for probe in summary["probes"]],
                         [["the physics flags this point unusable (nonfinite_evaluation=True)"], []])

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

    # The objective line of each coil template, which these tests replace by
    # FACTOR x the base-coil length sum (both templates define coil_lengths
    # first), and a probe printing the built problem's objective scale, its
    # scaled f at x0 and the cosine between the scaled gradient and the
    # gradient of the objective written into the template.
    OBJECTIVE_LINES = {
        "stage2": "unscaled = self.squared_flux + LENGTH_WEIGHT * sum(self.coil_lengths)",
        "boozer_single_stage": "unscaled = NonQuasiSymmetricRatio(self.boozer_surface, "
                               "BiotSavart(biot_savart.coils))",
    }
    OBJECTIVE_PROBE = (
        "import json, numpy as np\n"
        "from alm_problem import build_problem\n"
        "problem = build_problem(smoke=True)\n"
        "requested = {factor} * sum(problem.coil_lengths)\n"
        "scaled, wanted = problem.objective.dJ(), requested.dJ()\n"
        "norms = np.linalg.norm(scaled) * np.linalg.norm(wanted)\n"
        "print('OBJECTIVE ' + json.dumps({{'scale': problem.objective_scale, "
        "'scaled_value': float(problem.objective.J()), 'requested_value': float(requested.J()), "
        "'cosine': float(np.dot(scaled, wanted) / norms) if norms else None}}))\n"
    )

    def build_with_objective(self, template: str, factor: str) -> dict:
        generate_problem(self.scratch, template, replaced_once(
            (TEMPLATES_DIR / f"{template}.py").read_text(), self.OBJECTIVE_LINES[template],
            f"unscaled = {factor} * sum(self.coil_lengths)"))
        completed = run_python(["-c", self.OBJECTIVE_PROBE.format(factor=factor)],
                               cwd=self.scratch, problem_dir=self.scratch)
        self.assertEqual(completed.returncode, 0, completed.stderr[-4000:])
        return result_line(completed, "OBJECTIVE ")

    def test_templates_minimize_a_negative_objective(self):
        """R16-03: f / f(x0) turned a negative f into its negation, so the
        solver maximized it; the templates divide by |f(x0)|."""
        for template in self.OBJECTIVE_LINES:
            with self.subTest(template=template):
                built = self.build_with_objective(template, "-1.0")
                self.assertLess(built["requested_value"], 0.0)
                self.assertAlmostEqual(built["scale"], -built["requested_value"], places=12)
                self.assertAlmostEqual(built["scaled_value"], -1.0, places=12)
                self.assertAlmostEqual(built["cosine"], 1.0, places=12,
                                       msg="the scaled gradient does not point along the requested one")

    def test_templates_build_with_a_zero_initial_objective(self):
        """R16-03: a zero f(x0) divided by zero; the templates divide by
        ZERO_OBJECTIVE_SCALE (1, in f's units) instead."""
        for template in self.OBJECTIVE_LINES:
            with self.subTest(template=template):
                built = self.build_with_objective(template, "0.0")
                self.assertEqual(built["requested_value"], 0.0)
                self.assertEqual(built["scale"], 1.0)
                self.assertEqual(built["scaled_value"], 0.0)

    def test_templates_reject_an_invalid_zero_objective_scale(self):
        for template in self.OBJECTIVE_LINES:
            for scale in ("0.0", "-1.0", "float('nan')"):
                with self.subTest(template=template, scale=scale):
                    error = self.build_fails(template, "ZERO_OBJECTIVE_SCALE = 1.0",
                                             f"ZERO_OBJECTIVE_SCALE = {scale}")
                    self.assertRegex(error, r"^ValueError: ZERO_OBJECTIVE_SCALE")

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
        # The history's constraint_values is the result's signed g; the
        # clipped per-row violations are violation_values.
        self.assertEqual(entries[-1]["constraint_values"], list(full["constraint_values"].values()))
        self.assertEqual(entries[-1]["violation_values"],
                         [max(value, 0.0) for value in full["constraint_values"].values()])
        self.assertTrue((self.scratch / "checkpoints" / "final.pkl").exists())
        resumed = self.run_template("generic", "--smoke", "--resume", "checkpoints/outer_003.pkl")
        # Every field of the runner's summary (success, objective,
        # feasibility, stationarity, restore, last iterate, finish, ...), not
        # a chosen few.
        self.assertEqual(list(resumed), list(full))
        for key in full:
            with self.subTest(key=key):
                self.assertEqual(resumed[key], full[key])


if __name__ == "__main__":
    unittest.main()
