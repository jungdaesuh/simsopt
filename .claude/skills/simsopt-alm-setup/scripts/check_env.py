#!/usr/bin/env python
"""Report whether a Python environment can run simsopt's ALM solver
(``simsopt.solve.alm``) and which install route applies.

    python check_env.py [--python INTERPRETER] [--checkout DIR] [--fork-url URL]

It runs on Python 3.7 or newer and inspects the interpreter the optimization
runs with (default: the one running this script; any Python that runs
``python -c``) in child processes, so it never imports simsopt itself: the
Python version against the ALM floor, whether
``simsopt``, ``simsopt.solve.alm`` and ``simsopt.geo.signed_constraints``
import, how simsopt is installed (editable or not, from its PEP 610 metadata),
and, for the simsopt source checkout (``--checkout``, else the editable
install's directory, else the ``src`` layout around the imported package), its
git state. It reads local git refs only and changes nothing.

``route`` is the first that applies: ``blocked`` (Python below the floor, no
importable simsopt, a simsopt without the ``simsopt.solve`` package, or a
``simsopt.solve.alm`` that exists but fails to import: the report quotes the
error), ``ready`` (the solver imports),
``reinstall`` (the checkout has the ALM sources but the interpreter finds no
such module), ``upstream`` (a hiddenSymmetries remote's master has them),
``merge-fork`` (a git checkout: merge the ``alm-library`` branch), ``copy``
(no git checkout). ``blockers`` lists what must be fixed before the route can
run (for ``upstream`` and ``merge-fork`` also uncommitted changes). The last
line printed is ``CHECK_ENV {json}``; the exit status is 0 when the route is
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
FORK_URL_PLACEHOLDER = "https://github.com/<owner>/simsopt"
UPSTREAM_REPOSITORY = "github.com/hiddensymmetries/simsopt"
ALM_MODULE = "simsopt.solve.alm"
ALM_INIT = Path("src") / "simsopt" / "solve" / "alm" / "__init__.py"
RESULT_PREFIX = "CHECK_ENV "

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
    completed = probe(python, f"import {module}")
    return {"importable": completed.returncode == 0,
            "error": None if completed.returncode == 0 else last_line(completed.stderr)}


def missing_import(error: str, module: str) -> Optional[str]:
    """The module an import's last error line reports as absent, when that is
    ``module`` itself or one of its parent packages; None for any other
    failure (``module`` exists but fails, or something it imports is absent)."""
    match = re.fullmatch(r"ModuleNotFoundError: No module named '([\w.]+)'", error)
    if match and (module == match.group(1) or module.startswith(match.group(1) + ".")):
        return match.group(1)
    return None


def alm_import_blocker(alm_import: dict) -> Optional[str]:
    """What stops every route when ``simsopt.solve.alm`` does not import for a
    reason other than being absent; None when it imports or is simply absent."""
    if alm_import["importable"]:
        return None
    missing = missing_import(alm_import["error"], ALM_MODULE)
    if missing == ALM_MODULE:
        return None
    if missing is not None:
        return (f"the imported simsopt has no {missing} ({alm_import['error']}): it is incomplete or "
                "older than the ALM package supports; reinstall or upgrade simsopt first")
    return f"{ALM_MODULE} exists but fails to import ({alm_import['error']}); fix that error first"


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
        and git(root, "cat-file", "-e", f"{name}/master:{ALM_INIT.as_posix()}").returncode == 0
    ]
    return {
        "is_git": True,
        "root": str(root),
        "branch": git(root, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip(),
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
    if report["alm"]["importable"]:
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
            "has_alm_sources": (checkout_path / ALM_INIT).exists(),
            "git": git_state(checkout_path, fork_url),
        }

    blockers = []
    not_imported = {"importable": False, "error": "simsopt does not import"}
    if tuple(version) < MIN_PYTHON:
        blockers.append(f"Python {'.'.join(map(str, version))} is below the ALM floor "
                        f"{'.'.join(map(str, MIN_PYTHON))}; use a newer interpreter")
    if not simsopt_import["importable"]:
        blockers.append(f"simsopt does not import ({simsopt_import['error']}); install simsopt first")
    if simsopt_import["importable"]:
        alm_import = imports(python, ALM_MODULE)
        alm_blocker = alm_import_blocker(alm_import)
        if alm_blocker is not None:
            blockers.append(alm_blocker)
    else:
        alm_import = not_imported
    report = {
        "python": {"executable": python, "version": ".".join(map(str, version)),
                   "floor": ".".join(map(str, MIN_PYTHON)), "meets_floor": tuple(version) >= MIN_PYTHON},
        "simsopt": {**simsopt_import, "install": install, "info": simsopt_info},
        "alm": alm_import,
        "signed_constraints": (imports(python, "simsopt.geo.signed_constraints")
                               if simsopt_import["importable"] else not_imported),
        "checkout": checkout,
        "fork_url": fork_url,
        "blockers": blockers,
        "notes": [],
    }
    report["route"] = choose_route(report)

    route = report["route"]
    if (checkout is not None and simsopt_info is not None
            and checkout_path not in Path(simsopt_info["file"]).parents):
        report["notes"].append(f"the interpreter imports simsopt from {simsopt_info['file']}, "
                               f"not from the checkout {checkout_path}")
    blockers.extend(route_blockers(route, checkout))
    if route == "merge-fork" and checkout["git"]["contains_alm_upstream_base"] is False:
        report["notes"].append(f"HEAD does not contain the branch's upstream base "
                               f"{ALM_UPSTREAM_BASE[:9]}: merging also brings in the upstream "
                               "commits up to it; the copy route avoids that")
    if route == "merge-fork" and fork_url == FORK_URL_PLACEHOLDER:
        report["notes"].append("the fork URL is the placeholder; pass --fork-url with the "
                               "repository that publishes the alm-library branch")
    if route in ("reinstall", "upstream", "merge-fork") and install == "non-editable":
        report["notes"].append("simsopt is installed non-editable: reinstall from the checkout "
                               "after its sources change")
    if route == "copy" and install == "non-editable":
        report["notes"].append("copied files in site-packages are lost when simsopt is "
                               "reinstalled or upgraded")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--python", default=sys.executable,
                        help="the interpreter the optimization runs with (default: this one)")
    parser.add_argument("--checkout", type=Path, help="the simsopt source checkout, if known")
    parser.add_argument("--fork-url", default=FORK_URL_PLACEHOLDER,
                        help="the repository that publishes the alm-library branch")
    args = parser.parse_args(argv)
    report = build_report(args.python, args.checkout, args.fork_url)
    print(f"route: {report['route']}")
    for blocker in report["blockers"]:
        print(f"BLOCKER: {blocker}")
    for note in report["notes"]:
        print(f"note: {note}")
    print(RESULT_PREFIX + json.dumps(report))
    return 0 if report["route"] == "ready" else 1


if __name__ == "__main__":
    sys.exit(main())
