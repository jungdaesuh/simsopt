#!/usr/bin/env python
"""Report whether a Python environment can run the ALM solver package
``simsopt_alm`` (the ``simsopt-alm`` distribution) and which install route
applies.

    python check_env.py [--python INTERPRETER] [--checkout DIR] [--fork-url URL]

It runs on Python 3.7 or newer and inspects the interpreter the optimization
runs with (default: the one running this script; any Python that runs
``python -c``) in child processes, so it never imports the package itself:
the Python version against the package's floor, whether ``simsopt_alm`` and
``simsopt_alm.signed_constraints`` import and from which file, how the
``simsopt-alm`` distribution is installed (editable or not, from its PEP 610
metadata), and which ``simsopt`` the interpreter imports. The package
installs beside any simsopt (a fork, an editable checkout, a PyPI release)
and changes nothing in it. It reads local git refs only and changes nothing.

``--checkout DIR`` asks for an editable install from a clone of the
``alm-library`` branch at ``DIR`` (cloned there first when ``DIR`` does not
exist or is empty), for users who want to read or change the solver's
sources; without it the plain install from ``--fork-url`` applies. An
existing ``DIR`` counts as a clone only when it is the top level of a git
repository, is on the ``alm-library`` branch and has the package's sources
(``CLONE_SOURCES``); anything else is a blocker.

``templates`` in the report says which problem templates can run: the
generic one needs the solver, the Stage-2 and Boozer ones also the signed
constraints, which import simsopt (``TEMPLATE_MODULES``).

``route`` is the first that applies: ``blocked`` (Python below the floor; an
ALM module that exists but fails to import: the report quotes the error; or
``--checkout`` names a non-empty directory that is not an ``alm-library``
clone), ``ready`` (``simsopt_alm`` imports, and with ``--checkout`` from that
clone), ``editable`` (``--checkout`` given: clone if needed, then install the
clone's package editable), ``install`` (install the package from
``--fork-url``). ``blockers`` lists what must be fixed first. When simsopt
does not import, ``ready`` covers the generic template only, and a note says
so.

It prints the route, the blockers and notes, the file each module imports
from, the clone's HEAD (with ``--checkout``), and, last,
``CHECK_ENV {json}``. The exit status is 0 when the route is ``ready``.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import unquote

# The package's Python floor: requires-python of packages/simsopt-alm.
MIN_PYTHON = (3, 8)
ALM_BRANCH = "alm-library"
# The repository that publishes the alm-library branch (--fork-url overrides it).
FORK_URL = "https://github.com/jungdaesuh/simsopt.git"
DISTRIBUTION = "simsopt-alm"
# The package's directory in a clone of the alm-library branch.
PACKAGE_SUBDIRECTORY = Path("packages") / "simsopt-alm"
# What a clone of the alm-library branch must hold to install the package.
CLONE_SOURCES = (
    PACKAGE_SUBDIRECTORY / "pyproject.toml",
    PACKAGE_SUBDIRECTORY / "src" / "simsopt_alm" / "__init__.py",
)
ALM_MODULE = "simsopt_alm"
SIGNED_CONSTRAINTS_MODULE = "simsopt_alm.signed_constraints"
ALM_MODULES = (ALM_MODULE, SIGNED_CONSTRAINTS_MODULE)
# The ALM modules each problem template imports.
TEMPLATE_MODULES = {
    "generic": (ALM_MODULE,),
    "stage2": ALM_MODULES,
    "boozer_single_stage": ALM_MODULES,
}
RESULT_PREFIX = "CHECK_ENV "
# The report key of each ALM module.
REPORT_KEYS = {ALM_MODULE: "alm", SIGNED_CONSTRAINTS_MODULE: "signed_constraints"}

VERSION_PROBE = "import json, sys; print(json.dumps(list(sys.version_info[:3])))"
SIMSOPT_VERSION_PROBE = (
    "import json, simsopt; print(json.dumps(getattr(simsopt, '__version__', None)))"
)
METADATA_PROBE = (
    "import json, importlib.metadata as m; d = m.distribution(" + repr(DISTRIBUTION) + "); "
    "print(json.dumps({'version': d.version, 'direct_url': d.read_text('direct_url.json')}))"
)


def install_url(fork_url: str) -> str:
    """The pip requirement that installs the package from ``fork_url``."""
    return f"git+{fork_url}@{ALM_BRANCH}#subdirectory={PACKAGE_SUBDIRECTORY.as_posix()}"


def run(command: List[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)


def last_line(text: str) -> str:
    lines = [line for line in text.strip().splitlines() if line.strip()]
    return lines[-1] if lines else ""


def probe(python: str, code: str) -> subprocess.CompletedProcess:
    return run([python, "-c", code])


def imports(python: str, module: str) -> dict:
    """Whether ``module`` imports, the file it resolves to, or the error."""
    completed = probe(python, f"import json, os, {module}; "
                              f"print(json.dumps(os.path.realpath({module}.__file__)))")
    if completed.returncode != 0:
        return {"importable": False, "file": None, "error": last_line(completed.stderr)}
    return {"importable": True, "file": json.loads(last_line(completed.stdout)), "error": None}


def missing_import(error: str, module: str) -> Optional[str]:
    """The module an import's last error line reports as absent, when that is
    ``module`` itself or one of its parent packages; None for any other
    failure (``module`` exists but fails, or something it imports is absent)."""
    match = re.fullmatch(r"ModuleNotFoundError: No module named '([\w.]+)'", error)
    if match and (module == match.group(1) or module.startswith(match.group(1) + ".")):
        return match.group(1)
    return None


def import_blocker(result: dict, module: str) -> Optional[str]:
    """What stops every route when ALM ``module`` does not import for a reason
    other than the package being absent; None when it imports or is absent."""
    if result["importable"] or missing_import(result["error"], module) is not None:
        return None
    return f"{module} exists but fails to import ({result['error']}); fix that error first"


def template_readiness(report: dict) -> Dict[str, bool]:
    """Whether each problem template's ALM modules all import."""
    return {name: all(report[key]["importable"] for key in (REPORT_KEYS[module] for module in modules))
            for name, modules in TEMPLATE_MODULES.items()}


def git(checkout: Path, *arguments: str) -> subprocess.CompletedProcess:
    return run(["git", "-C", str(checkout), *arguments])


def git_state(checkout: Path) -> dict:
    toplevel = git(checkout, "rev-parse", "--show-toplevel")
    if toplevel.returncode != 0:
        return {"is_git": False}
    return {
        "is_git": True,
        "toplevel": str(Path(toplevel.stdout.strip()).resolve()),
        "branch": git(checkout, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip(),
        "head": git(checkout, "rev-parse", "HEAD").stdout.strip() or None,
    }


def clone_problems(path: Path, git_report: dict) -> List[str]:
    """Why the existing directory ``path`` is not a clone of the alm-library
    branch; empty when it is one."""
    if not git_report["is_git"]:
        return ["it is not a git repository"]
    problems = []
    if Path(git_report["toplevel"]) != path:
        problems.append(f"it is not the top level of its git repository ({git_report['toplevel']})")
    if git_report["branch"] != ALM_BRANCH:
        problems.append(f"it is on branch {git_report['branch']!r}, not {ALM_BRANCH!r}")
    problems.extend(f"{source.as_posix()} is missing" for source in CLONE_SOURCES
                    if not (path / source).is_file())
    return problems


def checkout_state(path: Path) -> dict:
    """The ``--checkout`` directory (resolved): whether it is a place to clone
    into (absent or empty) or a valid clone, why not, and its git state."""
    exists = path.exists()
    clone_target = not exists or (path.is_dir() and not any(path.iterdir()))
    git_report = {"is_git": False} if clone_target else git_state(path)
    problems = [] if clone_target else clone_problems(path, git_report)
    return {
        "path": str(path),
        "package": str(path / PACKAGE_SUBDIRECTORY),
        "exists": exists,
        "clone_target": clone_target,
        "is_alm_clone": not clone_target and not problems,
        "problems": problems,
        "git": git_report,
    }


def imports_from_checkout(report: dict) -> bool:
    """Whether ``simsopt_alm`` imports from the ``--checkout`` clone's package."""
    package_src = Path(report["checkout"]["package"]) / "src"
    return package_src in Path(report["alm"]["file"]).parents


def choose_route(report: dict) -> str:
    if report["blockers"]:
        return "blocked"
    checkout = report["checkout"]
    if report["alm"]["importable"] and (checkout is None or imports_from_checkout(report)):
        return "ready"
    if checkout is not None:
        return "editable"
    return "install"


def distribution_install(direct_url_text: Optional[str]) -> str:
    """``editable`` or ``non-editable`` from PEP 610 ``direct_url.json``;
    ``unknown`` without one (an install from an index records none)."""
    if direct_url_text is None:
        return "unknown"
    return "editable" if json.loads(direct_url_text).get("dir_info", {}).get("editable") else "non-editable"


def build_report(python: str, checkout_argument: Optional[Path], fork_url: str) -> dict:
    version = json.loads(probe(python, VERSION_PROBE).stdout)
    simsopt_import = imports(python, "simsopt")
    simsopt_version = None
    if simsopt_import["importable"]:
        simsopt_version = json.loads(last_line(probe(python, SIMSOPT_VERSION_PROBE).stdout))
    module_imports = {module: imports(python, module) for module in ALM_MODULES}
    distribution = {"version": None, "install": "unknown", "url": None}
    metadata = probe(python, METADATA_PROBE)
    if metadata.returncode == 0:
        found = json.loads(last_line(metadata.stdout))
        distribution = {
            "version": found["version"],
            "install": distribution_install(found["direct_url"]),
            "url": None if found["direct_url"] is None else unquote(json.loads(found["direct_url"])["url"]),
        }
    checkout = None if checkout_argument is None else checkout_state(checkout_argument.resolve())

    blockers = []
    if tuple(version) < MIN_PYTHON:
        blockers.append(f"Python {'.'.join(map(str, version))} is below the package's floor "
                        f"{'.'.join(map(str, MIN_PYTHON))}; use a newer interpreter")
    blockers.append(import_blocker(module_imports[ALM_MODULE], ALM_MODULE))
    # Without simsopt the signed constraints cannot import; that limits the
    # templates (a note below), it does not block the solver.
    if simsopt_import["importable"]:
        blockers.append(import_blocker(module_imports[SIGNED_CONSTRAINTS_MODULE], SIGNED_CONSTRAINTS_MODULE))
    if checkout is not None and checkout["problems"]:
        blockers.append(f"{checkout['path']} is not a clone of the {ALM_BRANCH} branch "
                        f"({'; '.join(checkout['problems'])}): clone one with "
                        f"`git clone -b {ALM_BRANCH} {fork_url} <new directory>` and pass "
                        "--checkout <new directory> (or name a new or empty directory to clone into)")
    report = {
        "python": {"executable": python, "version": ".".join(map(str, version)),
                   "floor": ".".join(map(str, MIN_PYTHON)), "meets_floor": tuple(version) >= MIN_PYTHON},
        "simsopt": {**simsopt_import, "version": simsopt_version},
        **{REPORT_KEYS[module]: result for module, result in module_imports.items()},
        "distribution": distribution,
        "checkout": checkout,
        "fork_url": fork_url,
        "install_url": install_url(fork_url),
        "blockers": [blocker for blocker in blockers if blocker is not None],
        "notes": [],
    }
    report["templates"] = template_readiness(report)
    report["route"] = choose_route(report)

    if not simsopt_import["importable"]:
        report["notes"].append(
            f"simsopt does not import ({simsopt_import['error']}): only the generic template can run; "
            "the Stage-2 and Boozer templates need simsopt in this interpreter (your own, or "
            f"pip install \"{DISTRIBUTION}[simsopt] @ {install_url(fork_url)}\" for PyPI's)")
    if report["route"] == "editable" and module_imports[ALM_MODULE]["importable"]:
        report["notes"].append(f"the interpreter imports simsopt_alm from {module_imports[ALM_MODULE]['file']}, "
                               f"not from the clone {checkout['path']}: the editable install replaces it")
    if report["route"] == "ready" and distribution["install"] == "non-editable":
        report["notes"].append(f"{DISTRIBUTION} is installed non-editable: reinstall to pick up a newer "
                               f"{ALM_BRANCH} (pip install --force-reinstall --no-deps ...)")
    return report


def provenance_lines(report: dict) -> List[str]:
    """Where each module imports from, how the package is installed, and the
    clone's current commit."""
    lines = []
    for module, key in (("simsopt", "simsopt"), *((module, REPORT_KEYS[module]) for module in ALM_MODULES)):
        result = report[key]
        where = result["file"] if result["importable"] else f"does not import ({result['error']})"
        if module == "simsopt" and result["importable"]:
            where += f" (version {result['version']})"
        lines.append(f"import {module}: {where}")
    distribution = report["distribution"]
    if distribution["version"] is None:
        lines.append(f"{DISTRIBUTION}: not installed")
    else:
        source = "" if distribution["url"] is None else f" from {distribution['url']}"
        lines.append(f"{DISTRIBUTION} {distribution['version']}: {distribution['install']}{source}")
    checkout = report["checkout"]
    if checkout is not None:
        if checkout["git"]["is_git"]:
            lines.append(f"checkout: {checkout['path']} at {checkout['git']['head']} "
                         f"(branch {checkout['git']['branch']})")
        else:
            lines.append(f"checkout: {checkout['path']} ({'to be cloned' if checkout['clone_target'] else 'not a clone'})")
    return lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--python", default=sys.executable,
                        help="the interpreter the optimization runs with (default: this one)")
    parser.add_argument("--checkout", type=Path,
                        help=f"a clone of the {ALM_BRANCH} branch to install editable "
                             "(cloned there first when it does not exist)")
    parser.add_argument("--fork-url", default=FORK_URL,
                        help=f"another repository that publishes the {ALM_BRANCH} branch "
                             f"(default: {FORK_URL})")
    args = parser.parse_args(argv)
    report = build_report(args.python, args.checkout, args.fork_url)
    print(f"route: {report['route']}")
    for blocker in report["blockers"]:
        print(f"BLOCKER: {blocker}")
    for note in report["notes"]:
        print(f"note: {note}")
    for line in provenance_lines(report):
        print(line)
    print(RESULT_PREFIX + json.dumps(report))
    return 0 if report["route"] == "ready" else 1


if __name__ == "__main__":
    sys.exit(main())
