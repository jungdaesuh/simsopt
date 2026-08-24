"""Native-test coverage manifest: deterministic inventory, validation, report.

This is the first executable slice of the coverage contract formerly defined
by ``docs/jax_native_unit_test_coverage_implementation_plan.md`` (Draft,
2026-07-29; commit 6fec6e4ca; removed from this branch by the 2026-08-24 docs
curation). That plan originated the artifact names, the allowed disposition
vocabulary, the required parity dimensions, and the fail-closed rules this
module implements. It was executed for the domains seeded by the mirror-wave
plan (formerly ``docs/jax_native_test_mirror_wave_implementation_plan.md``,
commit 2221b542a; also removed by the 2026-08-24 docs curation), item 6 of
its Implementation Plan.

Scope of this slice: rows are FILE-level for the native test surface plus
CAPABILITY records for the seeded domains. The 2026-07-29 plan's full
per-function ledger over every native test definition is explicitly deferred to
its own later phases and is not implemented here.

Slice policies declared here, pending contract amendment
----------------------------------------------------------
The plan above did not enumerate every operational rule this module needs to
run unattended. The following are this slice's own additions, each closing a
specific gap the plan left open, each open to revision when the plan is
amended:

* ``UNCLASSIFIED`` -- the planning state for a pinned native file this slice
  has not yet mapped to a capability. Needed because the plan requires every
  pinned file to have a row but does not name a state for "not reviewed yet".
* ``LOCAL_TOLERANCE_OWNER`` -- the sentinel a capability's ``tolerance_owner``
  takes when its bound tests draw a numeric tolerance from anywhere other
  than ``src/simsopt_jax/parity_tolerances.py``. Needed because the plan
  names the central tolerance module but does not define what to record for
  a test that predates it or bypasses it with a local literal.
* ``GENERIC_REASONS`` -- the fixed set of stand-in strings ("unsupported",
  "n/a", "tbd", ...) a ``reason`` may not equal. Needed to make the plan's
  "no generic 'unsupported' labels" requirement machine-checkable instead of
  a style guideline nothing enforces.
* ``MINIMUM_REASON_CHARACTERS`` -- the shortest a non-generic ``reason`` may
  be. Needed for the same purpose: the plan asks for a real rationale but
  states no checkable floor, and a single non-generic word would otherwise
  pass ``GENERIC_REASONS`` without being a rationale.

Native test surface
-------------------
The native surface is the upstream ``hiddenSymmetries/simsopt`` test tree that
this fork actually contains, not a name-shaped guess. A path glob cannot
express it: the fork adds 56 non-``jax``-named ``test_*.py`` files inside the
native domain directories (``tests/geo/test_nested_ls_reduced.py`` and
friends), so ``tests/{configs,core,field,geo,mhd,objectives,solve,util}`` minus
``*jax*`` over-includes them. The surface is therefore enumerated from git:

    every path under ``tests/`` in the pinned upstream authority commit whose
    basename matches ``test_*.py``

The pinned commit is ``baseline.upstream_authority_commit`` in the manifest. It
is the merge base of this fork's HEAD with upstream ``master``, so every
enumerated file is present in the working tree by construction. Files that
upstream added *after* that merge base (for example ``tests/util/test_logger.py``
at ``4ad6fd99...``) are outside this baseline and are reported as an upstream
drift follow-up rather than silently classified.

Source-tree hash recipe
-----------------------
``baseline.source_tree_hash`` is ``sha256`` over, for each enumerated path in
ascending byte order, the utf-8 encoding of::

    f"{path}\n{sha256_hexdigest(working_tree_file_bytes)}\n"

Editing any enumerated native test file changes the hash and fails ``--check``
until a maintainer re-mints it with ``--write``, which is the drift gate the
2026-07-29 plan requires.

Usage
-----
Read-only by default. ``tests/jax/test_native_unit_coverage_manifest.py``
runs it this way (``test_check_mode_exits_zero_without_touching_the_artifacts``).
No CI job invokes this script today (verified); wiring it into a
PR-blocking CI job is a filed follow-up in the manifest's own
``contract.follow_ups``, outside this slice's file ownership::

    python scripts/jax_native_unit_coverage.py

Regenerate the manifest hash and the generated document after an intentional
change::

    python scripts/jax_native_unit_coverage.py --write

``--write`` never invents rows: an unrowed native file fails in both modes, so
an upstream addition must be classified by a human before the artifacts move.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "tests/fixtures/jax_native_unit_coverage_manifest.json"
DOC_PATH = REPO_ROOT / "docs/jax_native_unit_test_coverage.md"

# Both plan documents were removed from this branch by the 2026-08-24 docs
# curation; these hold historical citations, not live doc-relative paths, so
# render_document renders them as plain text rather than a markdown link.
# No validator existence check touches either constant.
COVERAGE_CONTRACT_HISTORICAL_REFERENCE = (
    "formerly docs/jax_native_unit_test_coverage_implementation_plan.md, "
    "commit 6fec6e4ca, removed by the 2026-08-24 docs curation"
)
MIRROR_WAVE_HISTORICAL_REFERENCE = (
    "formerly docs/jax_native_test_mirror_wave_implementation_plan.md, "
    "commit 2221b542a, removed by the 2026-08-24 docs curation"
)

# Verbatim from the 2026-07-29 plan, section "Allowed dispositions".
ALLOWED_DISPOSITIONS = (
    "jax_equivalent",
    "jax_partial",
    "jax_missing",
    "hybrid_boundary",
    "shared_python",
    "native_only",
)
# The plan's explicit planning state for a native file this slice has not
# classified. It is listed in the report and is never counted as covered.
UNCLASSIFIED = "unclassified"

# Verbatim from the 2026-07-29 plan, section "Required parity dimensions".
ALLOWED_OBSERVABLES = (
    "value_or_residual",
    "gradient_jacobian_vjp_hessian",
    "shape_dtype_pytree",
    "symmetry_periodicity_orientation",
    "batching_broadcasting_edge_cases",
    "deterministic_seeded_behavior",
    "mutation_cache_state_tokens",
    "exception_status_behavior",
    "device_placement_and_transfer",
    "jit_vmap_autodiff_compatibility",
)

REQUIRED_LANES = ("native_cpu", "jax_cpu", "jax_gpu")
LANE_STATES = ("passing", "pending", "not_applicable")

CENTRAL_TOLERANCE_OWNER = "src/simsopt_jax/parity_tolerances.py"
# A capability whose parity test declares its own tolerances instead of the
# centralized owner. Recorded honestly; centralization is a filed follow-up.
LOCAL_TOLERANCE_OWNER = "local_test_tolerances"
NO_TOLERANCE_OWNER = "not_applicable"

# The plan fails "generic 'unsupported' labels" closed alongside empty reasons.
GENERIC_REASONS = frozenset(
    {"unsupported", "n/a", "na", "tbd", "todo", "none", "not supported", "-"}
)
MINIMUM_REASON_CHARACTERS = 40

_EVIDENCE_RE = re.compile(r"^(?P<path>[^:]+)(?::(?P<start>\d+)(?:-(?P<end>\d+))?)?$")
# A collectible pytest node id: either a bare function ('path::function', the
# pytest-style JAX mirrors) or a unittest method ('path::Class::function',
# every native test in this repo). Exactly one optional class segment is
# accepted; 'path::A::B::C' and deeper are rejected by construction, since
# only one '(identifier)::' prefix is optional before the final identifier.
_TEST_ID_RE = re.compile(
    r"^(?P<path>[^:]+)::"
    r"(?:(?P<class_name>[A-Za-z_][A-Za-z0-9_]*)::)?"
    r"(?P<function>[A-Za-z_][A-Za-z0-9_]*)$"
)


class ManifestError(RuntimeError):
    """Raised when the manifest or the repository cannot be inspected at all."""


def _git(repo_root: Path, *arguments: str) -> str:
    """Return stdout of a git command, failing loudly with git's own stderr."""
    completed = subprocess.run(
        ("git", *arguments),
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ManifestError(
            f"git {' '.join(arguments)} failed with exit {completed.returncode} "
            f"in {repo_root}: {completed.stderr.strip()}"
        )
    return completed.stdout


def native_test_surface(repo_root: Path, upstream_commit: str) -> tuple[str, ...]:
    """Return the sorted native test files pinned at ``upstream_commit``.

    A path qualifies when it lives under ``tests/`` in that commit and its
    basename matches ``test_*.py``. Directory ``__init__.py`` files, shared
    helper modules, and the fork's own additions are all outside the surface.
    """
    listing = _git(
        repo_root, "ls-tree", "-r", "--name-only", upstream_commit, "--", "tests/"
    )
    paths = [
        line
        for line in listing.splitlines()
        if line.startswith("tests/")
        and line.endswith(".py")
        and Path(line).name.startswith("test_")
    ]
    return tuple(sorted(paths))


def source_tree_hash(repo_root: Path, paths: tuple[str, ...]) -> str:
    """Return the ``sha256:`` digest of the enumerated files' working-tree bytes."""
    digest = hashlib.sha256()
    for path in sorted(paths):
        file_digest = hashlib.sha256((repo_root / path).read_bytes()).hexdigest()
        digest.update(f"{path}\n{file_digest}\n".encode())
    return f"sha256:{digest.hexdigest()}"


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def dump_manifest(manifest: dict[str, Any], manifest_path: Path) -> None:
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _reason_failures(check: str, label: str, reason: object) -> list[str]:
    if not isinstance(reason, str) or not reason.strip():
        return [f"{check}: {label} has an empty reason"]
    if reason.strip().lower() in GENERIC_REASONS:
        return [f"{check}: {label} has the generic reason {reason.strip()!r}"]
    if len(reason.strip()) < MINIMUM_REASON_CHARACTERS:
        return [
            f"{check}: {label} reason is shorter than "
            f"{MINIMUM_REASON_CHARACTERS} characters: {reason.strip()!r}"
        ]
    return []


def _evidence_failures(
    check: str,
    repo_root: Path,
    label: str,
    evidence: object,
    surface: tuple[str, ...],
) -> list[str]:
    """Validate one capability/row's evidence list against the working tree.

    Each entry is either a bare string ``'path[:start[-end]]'`` (allowed only
    when ``path`` is on the pinned native surface, which is itself protected
    by ``baseline.source_tree_hash``) or an object ``{"cite": "...", "anchor":
    "..."}`` for any other path, where ``anchor`` is a distinctive substring
    required to appear verbatim in the cited file -- the drift protection the
    hash gives pinned files, applied to everything else evidence can cite.
    """
    if not isinstance(evidence, list) or not evidence:
        return [f"{check}: {label} has no evidence"]
    surface_set = set(surface)
    failures: list[str] = []
    for item in evidence:
        anchor: object = None
        if isinstance(item, str):
            cite = item
        elif isinstance(item, dict):
            cite = item.get("cite")
            anchor = item.get("anchor")
            if not isinstance(cite, str):
                failures.append(
                    f"{check}: {label} evidence entry {item!r} has no 'cite' string"
                )
                continue
        else:
            failures.append(f"{check}: {label} has a blank evidence entry")
            continue
        if not cite.strip():
            failures.append(f"{check}: {label} has a blank evidence entry")
            continue
        match = _EVIDENCE_RE.match(cite.strip())
        if match is None:
            failures.append(
                f"{check}: {label} evidence {cite!r} is not 'path' or "
                "'path:line' or 'path:start-end'"
            )
            continue
        path = match.group("path")
        target = repo_root / path
        if not target.is_file():
            failures.append(f"{check}: {label} evidence path does not exist: {path}")
            continue
        file_text = target.read_text(encoding="utf-8")
        if path not in surface_set:
            if not isinstance(anchor, str) or not anchor.strip():
                failures.append(
                    f"{check}: {label} evidence {cite!r} cites a path outside the "
                    "pinned native surface and has no anchor"
                )
            elif anchor not in file_text:
                failures.append(
                    f"{check}: {label} evidence {cite!r} anchor {anchor!r} was not "
                    f"found in {path}"
                )
        end = match.group("end") or match.group("start")
        if end is None:
            continue
        line_count = len(file_text.splitlines())
        if int(end) > line_count:
            failures.append(
                f"{check}: {label} evidence {cite!r} points past the end of "
                f"{path} ({line_count} lines)"
            )
    return failures


def _defines_test_function(
    tree: ast.Module, function: str, class_name: str | None
) -> bool:
    """Return whether ``tree`` structurally defines ``function`` at the right scope.

    ``class_name is None`` requires a module-level function/async-function
    definition named ``function``. Otherwise it requires a class named
    ``class_name`` at module level whose body defines ``function``. Walking
    the AST (not searching source text) means a ``def function(`` that only
    appears inside a docstring, a comment, or a different class cannot
    satisfy this check.
    """
    function_node_types = (ast.FunctionDef, ast.AsyncFunctionDef)
    if class_name is None:
        return any(
            isinstance(node, function_node_types) and node.name == function
            for node in tree.body
        )
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return any(
                isinstance(item, function_node_types) and item.name == function
                for item in node.body
            )
    return False


def _test_id_failures(
    check: str, repo_root: Path, label: str, field: str, test_ids: object
) -> list[str]:
    if not isinstance(test_ids, list):
        return [f"{check}: {label} {field} must be a list"]
    failures: list[str] = []
    for test_id in test_ids:
        match = _TEST_ID_RE.match(test_id) if isinstance(test_id, str) else None
        if match is None:
            failures.append(
                f"{check}: {label} {field} entry {test_id!r} is not "
                "'path::function' or 'path::Class::function'"
            )
            continue
        path = match.group("path")
        target = repo_root / path
        if not target.is_file():
            failures.append(f"{check}: {label} {field} file does not exist: {path}")
            continue
        class_name = match.group("class_name")
        function = match.group("function")
        try:
            tree = ast.parse(target.read_text(encoding="utf-8"), filename=str(target))
        except SyntaxError as error:
            failures.append(
                f"{check}: {label} {field} file {path} is not valid Python: {error}"
            )
            continue
        if not _defines_test_function(tree, function, class_name):
            location = f"class {class_name}" if class_name else "module level"
            failures.append(
                f"{check}: {label} {field} names {function} which is not defined "
                f"at {location} in {path}"
            )
    return failures


def _validate_header(manifest: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for key in ("baseline", "capabilities", "contract", "native_test_files"):
        if key not in manifest:
            failures.append(f"manifest_schema: missing top-level key {key!r}")
    baseline = manifest.get("baseline", {})
    for key in (
        "baseline_commit",
        "source_tree_hash",
        "source_tree_hash_recipe",
        "upstream_authority_commit",
    ):
        value = baseline.get(key)
        if not isinstance(value, str) or not value.strip():
            failures.append(f"manifest_schema: baseline.{key} is missing or empty")
    return failures


def _validate_file_rows(
    repo_root: Path, manifest: dict[str, Any], surface: tuple[str, ...]
) -> list[str]:
    failures: list[str] = []
    rows = manifest.get("native_test_files", [])
    seen: dict[str, int] = {}
    for row in rows:
        path = row.get("path")
        if not isinstance(path, str) or not path:
            failures.append("manifest_schema: a native_test_files row has no path")
            continue
        seen[path] = seen.get(path, 0) + 1

    for path, count in sorted(seen.items()):
        if count > 1:
            failures.append(f"duplicate_row: {path} appears {count} times")

    surface_set = set(surface)
    for path in surface:
        if path not in seen:
            failures.append(
                f"native_surface_complete: {path} is on the pinned native test "
                "surface but has no manifest row"
            )
    for path in sorted(seen):
        if path not in surface_set:
            failures.append(
                f"unknown_row: {path} is not on the pinned native test surface"
            )
        elif not (repo_root / path).is_file():
            failures.append(f"stale_path: manifest row {path} does not exist on disk")

    capability_ids = {
        capability.get("id")
        for capability in manifest.get("capabilities", [])
        if isinstance(capability.get("id"), str)
    }
    for row in rows:
        path = row.get("path")
        if not isinstance(path, str) or not path:
            continue
        disposition = row.get("disposition")
        if disposition == UNCLASSIFIED:
            continue
        label = f"row {path}"
        if disposition not in ALLOWED_DISPOSITIONS:
            failures.append(
                f"disposition_vocabulary: {label} has disposition {disposition!r}, "
                f"not one of {ALLOWED_DISPOSITIONS + (UNCLASSIFIED,)}"
            )
        failures.extend(_reason_failures("empty_reason", label, row.get("reason")))
        failures.extend(
            _evidence_failures(
                "stale_path", repo_root, label, row.get("evidence"), surface
            )
        )
        row_capabilities = row.get("capabilities")
        if not isinstance(row_capabilities, list) or not row_capabilities:
            failures.append(
                f"capability_mapping: {label} is classified {disposition!r} but "
                "names no capability"
            )
            continue
        for capability_id in row_capabilities:
            if capability_id not in capability_ids:
                failures.append(
                    f"orphan_reference: {label} names unknown capability "
                    f"{capability_id!r}"
                )
    return failures


def _validate_capabilities(
    repo_root: Path, manifest: dict[str, Any], surface: tuple[str, ...]
) -> list[str]:
    failures: list[str] = []
    rows = {
        row.get("path")
        for row in manifest.get("native_test_files", [])
        if isinstance(row.get("path"), str)
    }
    referenced: set[str] = set()
    for row in manifest.get("native_test_files", []):
        for capability_id in row.get("capabilities", []) or []:
            referenced.add(capability_id)

    for capability in manifest.get("capabilities", []):
        capability_id = capability.get("id")
        if not isinstance(capability_id, str) or not capability_id:
            failures.append("manifest_schema: a capability record has no id")
            continue
        label = f"capability {capability_id}"
        if capability_id not in referenced:
            failures.append(
                f"orphan_reference: {label} is referenced by no native test file row"
            )
        disposition = capability.get("disposition")
        if disposition not in ALLOWED_DISPOSITIONS:
            failures.append(
                f"disposition_vocabulary: {label} has disposition {disposition!r}, "
                f"not one of {ALLOWED_DISPOSITIONS}"
            )
        domain = capability.get("domain")
        if not isinstance(domain, str) or not domain.strip():
            failures.append(f"manifest_schema: {label} has no domain")
        title = capability.get("title")
        if not isinstance(title, str) or not title.strip():
            # render_document dereferences capability['title'] unconditionally
            # (the "Dated decisions" table and every per-domain capability
            # bullet); a manifest that passed validation without this check
            # could pass --write's read-only validate() call and then KeyError
            # inside render_document itself.
            failures.append(f"manifest_schema: {label} has no title")
        failures.extend(
            _reason_failures("empty_reason", label, capability.get("reason"))
        )
        failures.extend(
            _evidence_failures(
                "stale_path", repo_root, label, capability.get("evidence"), surface
            )
        )

        # The contract asks each record to select the *applicable* parity
        # dimensions. A native_only capability is outside the JAX product
        # boundary, so its applicable set is legitimately empty; every other
        # disposition owes at least one observable.
        observables = capability.get("observables")
        if not isinstance(observables, list):
            failures.append(f"manifest_schema: {label} observables must be a list")
        elif not observables and disposition != "native_only":
            failures.append(f"manifest_schema: {label} declares no observables")
        else:
            for observable in observables:
                if observable not in ALLOWED_OBSERVABLES:
                    failures.append(
                        f"manifest_schema: {label} declares unknown observable "
                        f"{observable!r}"
                    )

        lanes = capability.get("required_lanes")
        if not isinstance(lanes, dict):
            failures.append(f"manifest_schema: {label} declares no required_lanes")
        else:
            for lane in REQUIRED_LANES:
                state = lanes.get(lane)
                if state not in LANE_STATES:
                    failures.append(
                        f"manifest_schema: {label} lane {lane} is {state!r}, "
                        f"not one of {LANE_STATES}"
                    )
            if disposition == "jax_missing":
                # "jax_missing" means the capability itself has no JAX side.
                # A "passing" jax_cpu/jax_gpu lane on such a row cannot be
                # evidence for the missing capability -- at best it is a
                # substitute test for something else, which belongs in the
                # reason/blocker text, not in a lane state that reads as
                # coverage of the thing this row says does not exist.
                for lane in ("jax_cpu", "jax_gpu"):
                    if lanes.get(lane) == "passing":
                        failures.append(
                            f"jax_missing_lane_contradiction: {label} is "
                            f"jax_missing but declares {lane}=passing; a "
                            "passing lane is a substitute test, not evidence "
                            "of the missing capability, so it must be "
                            "not_applicable"
                        )

        tolerance_owner = capability.get("tolerance_owner")
        if tolerance_owner not in (
            CENTRAL_TOLERANCE_OWNER,
            LOCAL_TOLERANCE_OWNER,
            NO_TOLERANCE_OWNER,
        ):
            failures.append(
                f"tolerance_owner: {label} names {tolerance_owner!r}; the contract "
                f"centralizes tolerances in {CENTRAL_TOLERANCE_OWNER}"
            )
        elif tolerance_owner == CENTRAL_TOLERANCE_OWNER:
            # Necessary condition for a central-owner claim: every bound test
            # file must actually import the owner module. The converse
            # (a file that imports it but also mixes in local literals) is not
            # mechanically decidable here, so a conservative
            # ``local_test_tolerances`` label is always permitted.
            for test_id in capability.get("test_ids", []):
                bound_path = str(test_id).split("::", 1)[0]
                bound_file = repo_root / bound_path
                if not bound_file.exists():
                    continue  # reported by the orphan_test_id existence check
                if "simsopt_jax.parity_tolerances" not in bound_file.read_text():
                    failures.append(
                        f"tolerance_owner_unsupported: {label} claims "
                        f"{CENTRAL_TOLERANCE_OWNER} but {bound_path} never "
                        "imports simsopt_jax.parity_tolerances"
                    )

        jax_api = capability.get("jax_api")
        if not isinstance(jax_api, list):
            failures.append(f"manifest_schema: {label} jax_api must be a list")
        else:
            for entry in jax_api:
                if not isinstance(entry, str) or not (repo_root / entry).is_file():
                    failures.append(f"stale_path: {label} jax_api path {entry!r}")

        native_files = capability.get("native_test_files")
        if not isinstance(native_files, list) or not native_files:
            failures.append(
                f"orphan_reference: {label} references no native test definition"
            )
        else:
            for path in native_files:
                if path not in rows:
                    failures.append(
                        f"orphan_reference: {label} names native file {path!r} "
                        "which has no manifest row"
                    )

        failures.extend(
            _test_id_failures(
                "orphan_test_id",
                repo_root,
                label,
                "test_ids",
                capability.get("test_ids"),
            )
        )
        failures.extend(
            _test_id_failures(
                "orphan_test_id",
                repo_root,
                label,
                "native_test_ids",
                capability.get("native_test_ids"),
            )
        )

        if disposition in ("native_only", "hybrid_boundary", "shared_python"):
            decision = capability.get("decision")
            if not isinstance(decision, dict):
                failures.append(
                    f"unreviewed_decision: {label} is {disposition!r} without a "
                    "decision record"
                )
            else:
                for key in ("decided_by", "decided_on", "proposed_on"):
                    value = decision.get(key)
                    if not isinstance(value, str) or not value.strip():
                        failures.append(
                            f"unreviewed_decision: {label} decision.{key} is empty"
                        )
    return failures


def validate(
    repo_root: Path,
    manifest: dict[str, Any],
    *,
    check_hash: bool = True,
    check_doc: bool = True,
    doc_path: Path = DOC_PATH,
) -> list[str]:
    """Return every fail-closed violation found in ``manifest``, in check order."""
    failures = _validate_header(manifest)
    if failures:
        return failures
    upstream_commit = manifest["baseline"]["upstream_authority_commit"]
    surface = native_test_surface(repo_root, upstream_commit)
    failures.extend(_validate_file_rows(repo_root, manifest, surface))
    failures.extend(_validate_capabilities(repo_root, manifest, surface))
    if check_hash:
        # A pinned surface file missing from the working tree (deleted,
        # renamed outside the manifest) must fail closed through this same
        # list, not crash source_tree_hash's unconditional .read_bytes().
        missing = [path for path in surface if not (repo_root / path).is_file()]
        if missing:
            for path in missing:
                failures.append(
                    f"stale_path: {path} is on the pinned native test surface "
                    "but is missing from the working tree; source_tree_hash "
                    "cannot be computed"
                )
        else:
            expected = source_tree_hash(repo_root, surface)
            recorded = manifest["baseline"]["source_tree_hash"]
            if recorded != expected:
                failures.append(
                    "source_tree_drift: native test files changed since the "
                    f"manifest was minted (recorded {recorded}, computed "
                    f"{expected}); re-mint with --write after classifying the "
                    "change"
                )
    if check_doc and not failures:
        # render_document dereferences several capability fields
        # unconditionally (title, evidence, required_lanes, ...); only
        # attempt it once every other check above has already passed, so a
        # manifest with an earlier schema violation is reported through that
        # violation, never through an uncaught KeyError here.
        rendered = render_document(manifest, surface)
        try:
            recorded_doc = doc_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            failures.append(f"generated_doc_drift: {doc_path} does not exist")
        else:
            if recorded_doc != rendered:
                failures.append(
                    "generated_doc_drift: the generated document does not "
                    f"match render_document(manifest); re-mint with --write "
                    f"({doc_path})"
                )
    return failures


def _counts(values: list[str]) -> list[tuple[str, int]]:
    tally: dict[str, int] = {}
    for value in values:
        tally[value] = tally.get(value, 0) + 1
    return sorted(tally.items())


def _evidence_display(item: object) -> str:
    """Render one evidence entry (a bare cite, or a ``{cite, anchor}`` object)."""
    if isinstance(item, dict):
        cite = item.get("cite", "")
        anchor = item.get("anchor")
        if anchor:
            return f"`{cite}` (anchor: `{anchor}`)"
        return f"`{cite}`"
    return f"`{item}`"


def render_document(manifest: dict[str, Any], surface: tuple[str, ...]) -> str:
    """Return the generated coverage index; never hand-edit its counts."""
    baseline = manifest["baseline"]
    rows = sorted(manifest["native_test_files"], key=lambda row: row["path"])
    capabilities = sorted(manifest["capabilities"], key=lambda item: item["id"])
    classified = [row for row in rows if row["disposition"] != UNCLASSIFIED]
    unclassified = [row for row in rows if row["disposition"] == UNCLASSIFIED]

    lines: list[str] = []
    lines.append("# JAX Coverage of Native SIMSOPT Unit-Test Capabilities")
    lines.append("")
    lines.append(
        "**Generated file — do not hand-edit.** Regenerate with "
        "`python scripts/jax_native_unit_coverage.py --write`."
    )
    lines.append("")
    lines.append(
        f"Coverage contract: {COVERAGE_CONTRACT_HISTORICAL_REFERENCE}. "
        "That plan (Draft, 2026-07-29) owns the schema, the disposition "
        "vocabulary, and the fail-closed rules; this document reports them."
    )
    lines.append("")
    lines.append(
        f"Executing wave: {MIRROR_WAVE_HISTORICAL_REFERENCE}, "
        "Implementation Plan item 6."
    )
    lines.append("")
    lines.append("## Baseline")
    lines.append("")
    lines.append(f"- Fork baseline commit: `{baseline['baseline_commit']}`")
    lines.append(
        f"- Upstream authority commit: `{baseline['upstream_authority_commit']}` "
        f"({baseline['upstream_authority_description']})"
    )
    lines.append(f"- Native test surface: {len(surface)} files")
    lines.append(f"- Source-tree hash: `{baseline['source_tree_hash']}`")
    lines.append(f"- Hash recipe: {baseline['source_tree_hash_recipe']}")
    lines.append("")
    lines.append("## How to read this report")
    lines.append("")
    lines.append(
        "- Rows are **file-level**. The 2026-07-29 plan's full per-function "
        "ledger over every native test definition is deferred to its own later "
        "phases and is not claimed here."
    )
    lines.append(
        "- `unclassified` is a valid planning state: the file is listed, and it "
        "is **never** counted as covered."
    )
    lines.append(
        "- `jax_partial` and `jax_missing` are valid planning states that fail "
        "final completion. They are **not** converted into a percent-covered "
        "claim, per the contract."
    )
    lines.append(
        "- A classified file row's `reason` states which capabilities of that "
        "file this slice enumerated; anything it does not name is still open."
    )
    equivalent = [
        capability
        for capability in capabilities
        if capability["disposition"] == "jax_equivalent"
    ]
    if equivalent:
        lines.append(
            f"- {len(equivalent)} capability record(s) reach `jax_equivalent`: "
            "native CPU, JAX CPU and strict JAX GPU evidence for every declared "
            "observable."
        )
    else:
        lines.append(
            "- **No capability record reaches `jax_equivalent`.** The contract "
            "requires native CPU, JAX CPU *and* strict JAX GPU evidence, and no "
            "GPU lane was executed in this wave, so every otherwise-complete "
            "capability is recorded as `jax_partial` with `jax_gpu=pending`."
        )
    lines.append("")
    lines.append("## Native test files by disposition")
    lines.append("")
    lines.append("| Disposition | Files |")
    lines.append("| --- | --- |")
    for disposition, count in _counts([row["disposition"] for row in rows]):
        lines.append(f"| `{disposition}` | {count} |")
    lines.append(f"| **total** | **{len(rows)}** |")
    lines.append("")
    scoped_rollup = [row for row in classified if row.get("partial_scope")]
    if scoped_rollup:
        lines.append(
            f"*Scoped rollup: {len(scoped_rollup)} of the {len(classified)} "
            "classified rows above enumerate only PART of their file's native "
            "test functions (see each row's `reason` for exactly which "
            f"capabilities it covers), so {len(classified)}/{len(rows)} "
            "classified/total must not be read as "
            f'"{len(classified)} files fully covered":*'
        )
        for row in scoped_rollup:
            lines.append(f"- `{row['path']}`")
        lines.append("")
    lines.append("## Capability records by domain and disposition")
    lines.append("")
    lines.append("| Domain | Disposition | Capabilities |")
    lines.append("| --- | --- | --- |")
    pairs = [
        f"{capability['domain']}\t{capability['disposition']}"
        for capability in capabilities
    ]
    for pair, count in _counts(pairs):
        domain, disposition = pair.split("\t")
        lines.append(f"| {domain} | `{disposition}` | {count} |")
    lines.append(f"| **total** | | **{len(capabilities)}** |")
    lines.append("")
    lines.append("## Dated decisions")
    lines.append("")
    lines.append("| Capability | Disposition | Proposed | Proposed by | Frozen | By |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for capability in capabilities:
        decision = capability.get("decision")
        if not decision:
            continue
        lines.append(
            f"| {capability['id']} — {capability['title']} | "
            f"`{capability['disposition']}` | {decision['proposed_on']} | "
            f"{decision.get('proposed_by', '')} | "
            f"{decision['decided_on']} | {decision['decided_by']} |"
        )
    lines.append("")
    lines.append("## Capability records")
    lines.append("")
    for domain in sorted({capability["domain"] for capability in capabilities}):
        lines.append(f"### {domain}")
        lines.append("")
        for capability in capabilities:
            if capability["domain"] != domain:
                continue
            lanes = capability["required_lanes"]
            lane_text = ", ".join(f"{lane}={lanes[lane]}" for lane in REQUIRED_LANES)
            lines.append(f"- **{capability['id']} — {capability['title']}**")
            lines.append(f"  - disposition: `{capability['disposition']}`")
            lines.append(f"  - reason: {capability['reason']}")
            lines.append(f"  - lanes: {lane_text}")
            lines.append(
                "  - evidence: "
                + ", ".join(_evidence_display(item) for item in capability["evidence"])
            )
            if capability["test_ids"]:
                lines.append(
                    "  - JAX-side tests: "
                    + ", ".join(f"`{item}`" for item in capability["test_ids"])
                )
            if capability.get("blocker"):
                lines.append(f"  - blocker: {capability['blocker']}")
            lines.append(f"  - tolerance owner: `{capability['tolerance_owner']}`")
        lines.append("")
    lines.append("## Classified native test files")
    lines.append("")
    lines.append("| Native test file | Domain | Disposition | Capabilities |")
    lines.append("| --- | --- | --- | --- |")
    for row in classified:
        lines.append(
            f"| `{row['path']}` | {row['domain']} | `{row['disposition']}` | "
            f"{', '.join(row['capabilities'])} |"
        )
    lines.append("")
    lines.append(f"## Unclassified native test files ({len(unclassified)})")
    lines.append("")
    lines.append(
        "These files are on the pinned native surface and have no capability "
        "mapping yet. They are listed here so the gap is visible, and they are "
        "not counted as covered."
    )
    lines.append("")
    for row in unclassified:
        lines.append(f"- `{row['path']}`")
    lines.append("")
    lines.append("## Follow-ups")
    lines.append("")
    for follow_up in manifest["contract"]["follow_ups"]:
        lines.append(f"- {follow_up}")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--write",
        action="store_true",
        help="re-mint the source-tree hash and regenerate the coverage document",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST_PATH,
        help="manifest to validate (default: the repository manifest)",
    )
    parser.add_argument(
        "--doc",
        type=Path,
        default=DOC_PATH,
        help="generated coverage document path",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="repository root the manifest paths are relative to",
    )
    arguments = parser.parse_args(argv)

    repo_root = arguments.repo_root.resolve()
    manifest = load_manifest(arguments.manifest)
    failures = validate(
        repo_root,
        manifest,
        check_hash=not arguments.write,
        check_doc=not arguments.write,
        doc_path=arguments.doc,
    )
    if failures:
        for failure in failures:
            print(failure, file=sys.stderr)
        print(
            f"jax_native_unit_coverage: {len(failures)} violation(s)", file=sys.stderr
        )
        return 1

    surface = native_test_surface(
        repo_root, manifest["baseline"]["upstream_authority_commit"]
    )
    if arguments.write:
        manifest["baseline"]["source_tree_hash"] = source_tree_hash(repo_root, surface)
        dump_manifest(manifest, arguments.manifest)
        arguments.doc.write_text(render_document(manifest, surface), encoding="utf-8")
        print(
            f"jax_native_unit_coverage: wrote {arguments.manifest} and {arguments.doc}"
        )
        return 0

    print(
        f"jax_native_unit_coverage: {len(surface)} native test files, "
        f"{len(manifest['capabilities'])} capability records, no violations"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
