#!/usr/bin/env python
"""Report whether a Python environment can run simsopt's ALM solver
(``simsopt.solve.alm``) and which install route applies.

    python check_env.py [--python INTERPRETER] [--checkout DIR] [--fork-url URL]

It runs on Python 3.7 or newer and inspects the interpreter the optimization
runs with (default: the one running this script; any Python that runs
``python -c``) in child processes, so it never imports simsopt itself: the
Python version against the ALM floor, whether
``simsopt``, ``simsopt.solve.alm`` and ``simsopt.geo.signed_constraints``
import and from which file, how simsopt is installed (editable or not, from
its PEP 610 metadata), and, for the simsopt source checkout (``--checkout``,
else the editable install's directory, else the ``src`` layout around the
imported package), its git state and HEAD commit. It reads local git refs
only and changes nothing.

The ALM sources are the solver package and the signed-constraint module.
``templates`` in the report says which problem templates can run: the
generic one needs the solver, the Stage-2 and Boozer ones also the signed
constraints (``TEMPLATE_MODULES``).

``route`` is the first that applies: ``blocked`` (Python below the floor; no
importable simsopt and no simsopt source checkout to install; a simsopt
without the ``simsopt.solve`` package; or an ALM module that exists but fails
to import: the report quotes the error), ``install`` (no importable simsopt:
install the source checkout, then rerun), ``ready`` (both ALM modules import,
so every template can run), ``reinstall`` (the checkout has the ALM sources
but the interpreter finds no such modules), ``upstream`` (a hiddenSymmetries
remote's master has them), ``merge-fork`` (a git checkout: merge the
``alm-library`` branch), ``copy`` (no git checkout). ``blockers`` lists what
must be fixed before the route can run (for ``upstream`` and ``merge-fork``
also uncommitted changes).

It prints the route, the blockers and notes, the file each module imports
from (the simsopt version shown is the one recorded when simsopt was
installed; an editable install keeps it after a merge), the checkout's HEAD,
and, last, ``CHECK_ENV {json}``. The exit status is 0 when the route is
``ready``.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import unquote, urlparse

# The ALM package's Python floor: upstream simsopt's requires-python.
MIN_PYTHON = (3, 8)
# The upstream commit the alm-library branch is based on.
ALM_UPSTREAM_BASE = "9e027eac38028d57aa23777be52a781aa860e347"
ALM_BRANCH = "alm-library"
# The repository that publishes the alm-library branch (--fork-url overrides it).
FORK_URL = "https://github.com/jungdaesuh/simsopt.git"
UPSTREAM_REPOSITORY = "github.com/hiddensymmetries/simsopt"
UPSTREAM_URL = "https://github.com/hiddenSymmetries/simsopt"
ALM_MODULE = "simsopt.solve.alm"
SIGNED_CONSTRAINTS_MODULE = "simsopt.geo.signed_constraints"
ALM_MODULES = (ALM_MODULE, SIGNED_CONSTRAINTS_MODULE)
ALM_INIT = Path("src") / "simsopt" / "solve" / "alm" / "__init__.py"
# What a checkout needs to provide both ALM modules.
ALM_SOURCES = (ALM_INIT, Path("src") / "simsopt" / "geo" / "signed_constraints.py")
# The ALM modules each problem template imports.
TEMPLATE_MODULES = {
    "generic": (ALM_MODULE,),
    "stage2": ALM_MODULES,
    "boozer_single_stage": ALM_MODULES,
}
SIMSOPT_PACKAGE_INIT = Path("src") / "simsopt" / "__init__.py"
RESULT_PREFIX = "CHECK_ENV "
# The report key of each ALM module.
REPORT_KEYS = {ALM_MODULE: "alm", SIGNED_CONSTRAINTS_MODULE: "signed_constraints"}

VERSION_PROBE = "import json, sys; print(json.dumps(list(sys.version_info[:3])))"
SIMSOPT_PROBE = (
    "import json, os, numpy, scipy, simsopt; print(json.dumps({"
    "'file': os.path.realpath(simsopt.__file__), 'version': getattr(simsopt, '__version__', None), "
    "'numpy': numpy.__version__, 'scipy': scipy.__version__}))"
)
METADATA_PROBE = (
    "import json, importlib.metadata as m; "
    "print(json.dumps(m.distribution('simsopt').read_text('direct_url.json')))"
)


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
    other than being absent; None when it imports or is simply absent."""
    if result["importable"]:
        return None
    missing = missing_import(result["error"], module)
    if missing == module:
        return None
    if missing is not None:
        return (f"the imported simsopt has no {missing} ({result['error']}): it is incomplete or "
                "older than the ALM package supports; reinstall or upgrade simsopt first")
    return f"{module} exists but fails to import ({result['error']}); fix that error first"


def simsopt_import_blocker(simsopt_import: dict, checkout: Optional[dict]) -> Optional[str]:
    """What stops every route when simsopt itself does not import: nothing when
    a simsopt source checkout can be installed (the ``install`` route)."""
    if simsopt_import["importable"] or (checkout is not None and checkout["is_simsopt_source"]):
        return None
    if checkout is None:
        where = "no simsopt source checkout was given"
    else:
        where = (f"{checkout['path']} is not a simsopt source checkout (no pyproject.toml and "
                 f"{SIMSOPT_PACKAGE_INIT.as_posix()})")
    return (f"simsopt does not import ({simsopt_import['error']}) and {where}: rerun with --checkout "
            f"<simsopt source checkout> (clone {UPSTREAM_URL} if there is none) for "
            "the install route, or install simsopt first")


def template_readiness(report: dict) -> Dict[str, bool]:
    """Whether each problem template's ALM modules all import."""
    return {name: all(report[key]["importable"] for key in (REPORT_KEYS[module] for module in modules))
            for name, modules in TEMPLATE_MODULES.items()}


# ``git remote -v`` prints ``name<TAB>url (fetch|push)``, possibly followed by
# annotations such as a partial clone's `` [blob:none]``; a URL may contain
# spaces. A remote without a URL prints ``name<TAB>`` alone.
URL_AND_KIND = re.compile(r"^(.*) \((fetch|push)\)(?: .*)?$")


def parse_remotes(remote_listing: str) -> Dict[str, Optional[str]]:
    """``name -> fetch URL`` from ``git remote -v``, with None for a remote that
    has no fetch URL (none at all, a push URL only, or a line this parser does
    not recognize)."""
    remotes: Dict[str, Optional[str]] = {}
    for line in remote_listing.splitlines():
        name, _tab, url_and_kind = line.partition("\t")
        remotes.setdefault(name, None)
        match = URL_AND_KIND.match(url_and_kind)
        if match is not None and match.group(2) == "fetch":
            remotes[name] = match.group(1)
    return remotes


def route_blockers(route: str, checkout: Optional[dict]) -> List[str]:
    """What stops ``route`` itself: the routes that merge need a clean tree."""
    if route in ("upstream", "merge-fork") and checkout["git"]["dirty"]:
        return ["the checkout has uncommitted changes to tracked files; commit or stash them "
                "before merging"]
    return []


def normalized_repository(url: str) -> str:
    """``host/owner/repo`` in lower case, from an https or scp-style git URL."""
    url = url.strip().lower()
    scp_style = re.match(r"^[\w.-]+@([\w.-]+):(.+)$", url)
    if scp_style:
        url = f"https://{scp_style.group(1)}/{scp_style.group(2)}"
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    if path.endswith(".git"):
        path = path[: -len(".git")]
    return f"{parsed.netloc.split('@')[-1]}{path}"


def git(checkout: Path, *arguments: str) -> subprocess.CompletedProcess:
    return run(["git", "-C", str(checkout), *arguments])


def git_state(checkout: Path, fork_url: str) -> dict:
    toplevel = git(checkout, "rev-parse", "--show-toplevel")
    if toplevel.returncode != 0:
        return {"is_git": False}
    root = Path(toplevel.stdout.strip())
    remotes = parse_remotes(git(root, "remote", "-v").stdout)
    fork_repository = normalized_repository(fork_url)
    base_check = git(root, "merge-base", "--is-ancestor", ALM_UPSTREAM_BASE, "HEAD").returncode
    upstream_with_alm = [
        name for name, url in remotes.items()
        if url is not None and normalized_repository(url) == UPSTREAM_REPOSITORY
        and all(git(root, "cat-file", "-e", f"{name}/master:{source.as_posix()}").returncode == 0
                for source in ALM_SOURCES)
    ]
    return {
        "is_git": True,
        "root": str(root),
        "branch": git(root, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip(),
        "head": git(root, "rev-parse", "HEAD").stdout.strip() or None,
        "dirty": bool(git(root, "status", "--porcelain", "--untracked-files=no").stdout.strip()),
        "remotes": remotes,
        "fork_remote": next((name for name, url in remotes.items()
                             if url is not None and normalized_repository(url) == fork_repository), None),
        "alm_branch_refs": git(root, "branch", "-r", "--list", f"*/{ALM_BRANCH}").stdout.split(),
        "upstream_remotes_with_alm": upstream_with_alm,
        # True/False, or None when the base commit is not in this clone yet.
        "contains_alm_upstream_base": {0: True, 1: False}.get(base_check),
    }


def checkout_from(simsopt_file: Optional[str], direct_url: Optional[dict]) -> Optional[Path]:
    """The simsopt source checkout behind an install, when there is one."""
    if direct_url is not None and direct_url.get("dir_info", {}).get("editable"):
        return Path(unquote(urlparse(direct_url["url"]).path))
    if simsopt_file is not None:
        package = Path(simsopt_file).parent
        if package.parent.name == "src" and (package.parent.parent / "pyproject.toml").exists():
            return package.parent.parent
    return None


def choose_route(report: dict) -> str:
    if report["blockers"]:
        return "blocked"
    if not report["simsopt"]["importable"]:
        return "install"
    if all(template_readiness(report).values()):
        return "ready"
    checkout = report["checkout"]
    if checkout is not None and checkout["has_alm_sources"]:
        return "reinstall"
    if checkout is not None and checkout["git"].get("upstream_remotes_with_alm"):
        return "upstream"
    if checkout is not None and checkout["git"]["is_git"]:
        return "merge-fork"
    return "copy"


def build_report(python: str, checkout_argument: Optional[Path], fork_url: str) -> dict:
    version = json.loads(probe(python, VERSION_PROBE).stdout)
    simsopt_import = imports(python, "simsopt")
    simsopt_info = None
    direct_url = None
    if simsopt_import["importable"]:
        simsopt_info = json.loads(last_line(probe(python, SIMSOPT_PROBE).stdout))
        metadata = probe(python, METADATA_PROBE)
        if metadata.returncode == 0:
            direct_url_text = json.loads(last_line(metadata.stdout))
            direct_url = None if direct_url_text is None else json.loads(direct_url_text)
    if direct_url is None:
        install = "unknown"
    else:
        install = "editable" if direct_url.get("dir_info", {}).get("editable") else "non-editable"

    checkout_path = checkout_argument or checkout_from(
        None if simsopt_info is None else simsopt_info["file"], direct_url)
    checkout = None
    if checkout_path is not None:
        checkout_path = checkout_path.resolve()
        checkout = {
            "path": str(checkout_path),
            "is_simsopt_source": ((checkout_path / "pyproject.toml").exists()
                                  and (checkout_path / SIMSOPT_PACKAGE_INIT).exists()),
            "has_alm_sources": all((checkout_path / source).exists() for source in ALM_SOURCES),
            "git": git_state(checkout_path, fork_url),
        }

    blockers = []
    if tuple(version) < MIN_PYTHON:
        blockers.append(f"Python {'.'.join(map(str, version))} is below the ALM floor "
                        f"{'.'.join(map(str, MIN_PYTHON))}; use a newer interpreter")
    simsopt_blocker = simsopt_import_blocker(simsopt_import, checkout)
    if simsopt_blocker is not None:
        blockers.append(simsopt_blocker)
    if simsopt_import["importable"]:
        module_imports = {module: imports(python, module) for module in ALM_MODULES}
        blockers.extend(blocker for blocker in (import_blocker(result, module)
                                                for module, result in module_imports.items())
                        if blocker is not None)
    else:
        module_imports = {module: {"importable": False, "file": None, "error": "simsopt does not import"}
                          for module in ALM_MODULES}
    report = {
        "python": {"executable": python, "version": ".".join(map(str, version)),
                   "floor": ".".join(map(str, MIN_PYTHON)), "meets_floor": tuple(version) >= MIN_PYTHON},
        "simsopt": {**simsopt_import, "install": install, "info": simsopt_info},
        **{REPORT_KEYS[module]: result for module, result in module_imports.items()},
        "checkout": checkout,
        "fork_url": fork_url,
        "blockers": blockers,
        "notes": [],
    }
    report["templates"] = template_readiness(report)
    report["route"] = choose_route(report)

    route = report["route"]
    if route == "install" and not checkout["has_alm_sources"]:
        report["notes"].append("the checkout lacks the ALM sources: after the install, rerun "
                               "check_env.py for the route that adds them")
    if route not in ("blocked", "install") and not all(report["templates"].values()):
        report["notes"].append(
            "templates that can run now: "
            + (", ".join(name for name, ready in report["templates"].items() if ready) or "none")
            + f"; the route adds what the others import ({', '.join(ALM_MODULES)})")
    if (checkout is not None and simsopt_info is not None
            and checkout_path not in Path(simsopt_info["file"]).parents):
        report["notes"].append(f"the interpreter imports simsopt from {simsopt_info['file']}, "
                               f"not from the checkout {checkout_path}")
    blockers.extend(route_blockers(route, checkout))
    if route == "merge-fork" and checkout["git"]["contains_alm_upstream_base"] is False:
        report["notes"].append(f"HEAD does not contain the branch's upstream base "
                               f"{ALM_UPSTREAM_BASE[:9]}: merging also brings in the upstream "
                               "commits up to it; the copy route avoids that")
    if route in ("reinstall", "upstream", "merge-fork") and install == "non-editable":
        report["notes"].append("simsopt is installed non-editable: reinstall from the checkout "
                               "after its sources change")
    if route == "copy" and install == "non-editable":
        report["notes"].append("copied files in site-packages are lost when simsopt is "
                               "reinstalled or upgraded")
    return report


def provenance_lines(report: dict) -> List[str]:
    """Where each module imports from, and the checkout's current commit: the
    simsopt version string is recorded when simsopt is installed, so after a
    merge into an editable install it names the commit that was installed."""
    lines = []
    info = report["simsopt"]["info"]
    for module, key in (("simsopt", "simsopt"), *((module, REPORT_KEYS[module]) for module in ALM_MODULES)):
        result = report[key]
        where = result["file"] if result["importable"] else f"does not import ({result['error']})"
        if module == "simsopt" and info is not None:
            where += f" (version {info['version']}, recorded when simsopt was installed)"
        lines.append(f"import {module}: {where}")
    checkout = report["checkout"]
    if checkout is None:
        lines.append("checkout: none found")
    elif checkout["git"]["is_git"]:
        lines.append(f"checkout: {checkout['path']} at {checkout['git']['head']} "
                     f"(branch {checkout['git']['branch']})")
    else:
        lines.append(f"checkout: {checkout['path']} (not a git checkout)")
    return lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--python", default=sys.executable,
                        help="the interpreter the optimization runs with (default: this one)")
    parser.add_argument("--checkout", type=Path, help="the simsopt source checkout, if known")
    parser.add_argument("--fork-url", default=FORK_URL,
                        help="another repository that publishes the alm-library branch "
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
