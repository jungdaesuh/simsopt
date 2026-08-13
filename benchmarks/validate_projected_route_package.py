"""Validate the projected-route certificate package from its own bytes alone.

The campaign's production validator --
``run_single_stage_projected_route_gpu_root.validate_root_artifact`` -- judges
the sealed root against the LIVE repository: it re-derives the execution-source
manifest from the working tree, imports the campaign's frozen constants, and
re-reads the native endpoint state through an absolute path into a DIFFERENT
campaign directory.  That is the right instrument for the producer, and it is
the wrong one for a reader: it answers "is this receipt consistent with this
checkout of this repository on this box", and it stops answering anything at
all the moment a single ``.py`` file lands under ``benchmarks/``, ``examples/``
or ``src/`` and moves the manifest the receipt names.

This module is the reader's instrument, and its contract is the complement:

* It opens NOTHING outside the package directory it is given.  No repository
  file, no git object, no other campaign directory, no network.  Every read
  goes through one confined reader that resolves under the package root and
  refuses symlinks, so the confinement is a property of the code rather than a
  promise, and the test suite audits the syscalls to prove it.
* It imports NOTHING from this repository, and no third-party package.  Pure
  standard library: the certificate has to stay checkable after this repository
  moves on, and an import of ``benchmarks.*`` would reintroduce exactly the
  coupling the module exists to remove.
* The native endpoint reference -- the reference side of the quality gate --
  comes from the sealed supplement copy, digest-pinned, not from the absolute
  path the receipt records.

The package is three sealed directories:

``final/``
    the root artifact as published (607 manifest members plus the manifest).
``provenance-supplement/``
    the three files the receipt names by digest but does not carry, plus the
    supplement manifest and the cross-version revalidation record.
``composite-manifest/``
    this module's binding: one canonical document naming every member of the
    other two by digest, the digest of the supplement manifest, the digest of
    ``root-evidence.json``, and one composite digest over the whole member
    table -- the single value a plan, a commit message or a reviewer can pin.

Validation is a REBUILD, not a read-back.  ``build_composite_document`` derives
the whole binding from the package's bytes; validation rebuilds it and compares
the result to the stored document byte for byte.  A tampered member, a missing
member, an extra file, a hand-edited composite digest and a non-canonical
encoding are therefore all one failure mode with one cause, and none of them
can pass by agreeing with a field that was read rather than recomputed.

Two duplications are deliberate and both are pinned by tests against the
repository's own definitions, because the alternative is an import that would
break the standalone contract: the canonical JSON encoding
(``canonical_json_bytes``) and the pinned-term gate table
(``PINNED_ENDPOINT_QUALITY_GATES``).

Usage::

    python benchmarks/validate_projected_route_package.py <package-root>
    python benchmarks/validate_projected_route_package.py <package-root> \\
        --expect-composite-sha256 <hex>
    python benchmarks/validate_projected_route_package.py <package-root> \\
        --emit-composite-manifest
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import sys
from ast import literal_eval
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Final, TypeAlias

JsonValue: TypeAlias = (
    None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
)

COMPOSITE_SCHEMA_VERSION: Final = (
    "single-stage-projected-route-gpu-package-composite-v1"
)
RESULT_SCHEMA_VERSION: Final = (
    "single-stage-projected-route-gpu-package-validation-v1"
)
PACKAGE_VERDICT: Final = "PACKAGE_VALIDATED"

FINAL_DIRECTORY: Final = "final"
SUPPLEMENT_DIRECTORY: Final = "provenance-supplement"
COMPOSITE_DIRECTORY: Final = "composite-manifest"
CANONICAL_SUBDIRECTORY: Final = "canonical"
PACKAGE_DIRECTORIES: Final = (
    FINAL_DIRECTORY,
    SUPPLEMENT_DIRECTORY,
    COMPOSITE_DIRECTORY,
)

ARTIFACT_MANIFEST_FILENAME: Final = "artifact-manifest.json"
ROOT_EVIDENCE_FILENAME: Final = "root-evidence.json"
SUPPLEMENT_MANIFEST_FILENAME: Final = "supplement-manifest.json"
COMPOSITE_MANIFEST_FILENAME: Final = "composite-manifest.json"
SOURCE_SNAPSHOT_DIRECTORY: Final = "source-snapshot"
SOURCE_MANIFEST_FILENAME: Final = "source-manifest.json"

ARTIFACT_MANIFEST_SCHEMA_VERSION: Final = (
    "single-stage-projected-route-gpu-root-manifest-v1"
)
ROOT_EVIDENCE_SCHEMA_VERSION: Final = "single-stage-projected-route-gpu-root-v1"
SUPPLEMENT_SCHEMA_VERSION: Final = (
    "single-stage-projected-route-gpu-root-provenance-supplement-v1"
)
REVALIDATION_SCHEMA_VERSION: Final = (
    "single-stage-projected-route-gpu-root-revalidation-record-v1"
)
AUTHORITY_SCHEMA_VERSION: Final = (
    "single-stage-neq-gntr3-execution-source-authority-v1"
)
VERDICT_CLAIM_DISCHARGED: Final = "CLAIM_DISCHARGED"

# The gate table of plan section 1.1, restated rather than imported so this
# module stays standalone.  ``test_validate_projected_route_package`` asserts it
# is identical to ``rehearse_single_stage_projected_route_cpu``'s, which is the
# only place the two spellings are allowed to differ from each other: here.
PINNED_ENDPOINT_QUALITY_GATES: Final = {
    "constraint.boozer|inf": ("absolute", 1.0e-10),
    "constraint.volume": ("absolute", 1.0e-10),
    "observable.iota": ("relative", 1.0e-4),
    "observable.major_radius": ("relative", 1.0e-4),
    "observable.non_qs_ratio": ("not_worse", 1.0e-4),
    "observable.total_length": ("not_worse", 1.0e-4),
    "observable.volume": ("relative", 1.0e-6),
    "raw.non_qs": ("not_worse", 1.0e-4),
    "raw.residual": ("absolute", 1.0e-10),
    "weighted_total": ("not_worse", 1.0e-6),
}
INFORMATIONAL_ENDPOINT_OBSERVABLES: Final = ("observable.G", "state.G")

# ``exact_numeric_tree_sha256`` hashes ``repr(tree_definition)`` before the
# leaf, and the published terminal state is one bare array, so its whole tree
# prefix is this constant.  Re-deriving the digest is what binds the published
# endpoint COORDINATES to the receipt that reports them.
BARE_ARRAY_TREE_DEFINITION_REPR: Final = "PyTreeDef(*)"
FLOAT64_LITTLE_ENDIAN_DESCRIPTION: Final = "<f8"

SEALED_FILE_MODE: Final = 0o444
SEALED_DIRECTORY_MODE: Final = 0o555

_SHA256_LENGTH: Final = 64
_HEX_DIGITS: Final = frozenset("0123456789abcdef")

COMPOSITE_SHA256_DEFINITION: Final = (
    "sha256 over the canonical JSON encoding of the member table as a list of "
    "[package_relative_path, sha256] pairs sorted by path, covering every "
    "member of final/ and provenance-supplement/ and nothing else"
)
COMPOSITE_PURPOSE: Final = (
    "Bind final/, the provenance supplement and their digests into one "
    "composite package identity, so a single sha256 pins the whole certificate "
    "and the package can be validated without the repository that produced it."
)
EXECUTED_SOURCE_CLOSURE_STATEMENT: Final = (
    "The closure this package carries is the EXECUTED-SOURCE closure: every "
    "module the certified chain actually bound at run time, byte for byte. The "
    "repository-wide execution-source authority is a larger freeze -- it "
    "enumerates every file admitted to the certified surface, executed or not "
    "-- and it is CITED by digest (its bytes are carried in the supplement) "
    "rather than embedded file by file. Members of that authority which no "
    "module bound are absent from this package by design: they did not run, so "
    "their bytes cannot affect the published numbers, and embedding them would "
    "enlarge the certificate without narrowing what it proves. This validator "
    "enforces the narrower closure exactly: every bound module resolves to "
    "package bytes at the digest the authority states, and no authority member "
    "that is absent from the package may appear among the bound modules."
)


class PackageValidationError(RuntimeError):
    """The package was refused; nothing about it is asserted beyond the cause."""


def canonical_json_bytes(value: JsonValue) -> bytes:
    """Serialize the evidence protocol's sole UTF-8 JSON representation.

    Byte-identical to ``single_stage_fullspace_snapshot.canonical_json_bytes``;
    duplicated here, and pinned by test, to keep this module import-free.
    """

    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def load_json_bytes(payload: bytes, *, where: str) -> JsonValue:
    """Parse UTF-8 JSON, refusing duplicate keys; canonicality is not required.

    Two of the sealed supplement documents were written pretty-printed, so a
    reader that demanded canonical bytes everywhere could not read the package
    at all.  Canonicality is therefore MEASURED per document and reported, and
    required only where this module itself is the producer.
    """

    def reject_duplicates(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
        result: dict[str, JsonValue] = {}
        for key, value in pairs:
            if key in result:
                raise PackageValidationError(f"duplicate key {key!r} in {where}")
            result[key] = value
        return result

    try:
        return json.loads(payload.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PackageValidationError(f"{where} is not UTF-8 JSON") from error


def sha256_hex(payload: bytes) -> str:
    """Digest raw bytes the one way every member of this package is named by."""

    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: object, where: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in _HEX_DIGITS for character in value)
    ):
        raise PackageValidationError(f"{where} is not a lowercase sha256 digest")
    return value


def _mapping(value: object, where: str) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise PackageValidationError(f"{where} is not a JSON document")
    return value


def _sequence(value: object, where: str) -> list[JsonValue]:
    if not isinstance(value, list):
        raise PackageValidationError(f"{where} is not a JSON list")
    return value


def _exact_keys(document: Mapping[str, JsonValue], keys: frozenset[str], where: str) -> None:
    if frozenset(document) != keys:
        raise PackageValidationError(
            f"{where} keys differ: missing {sorted(keys - frozenset(document))}, "
            f"unexpected {sorted(frozenset(document) - keys)}"
        )


def _float(value: object, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PackageValidationError(f"{where} is not a number")
    number = float(value)
    if not math.isfinite(number):
        raise PackageValidationError(f"{where} is not finite")
    return number


class PackageReader:
    """Every byte this module reads, and the only place it can read from.

    The confinement is structural: the package root is resolved once, each
    member path is normalized and re-checked against it, and every member is
    ``lstat``-ed so a symlink -- the one filesystem object that could point out
    of the package after the path check passed -- is refused rather than
    followed.
    """

    def __init__(self, package_root: Path) -> None:
        self._root = package_root.resolve(strict=True)
        if not self._root.is_dir():
            raise PackageValidationError(f"package root {self._root} is not a directory")

    @property
    def root(self) -> Path:
        """The one directory this reader is allowed to open files under."""

        return self._root

    def _resolve(self, relative: str) -> Path:
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or not pure.parts:
            raise PackageValidationError(f"package path {relative!r} is not relative")
        path = self._root.joinpath(*pure.parts)
        if os.path.normpath(path) != str(path) or not str(path).startswith(
            f"{self._root}{os.sep}"
        ):
            raise PackageValidationError(f"package path {relative!r} escapes the package")
        return path

    def read(self, relative: str) -> bytes:
        """Read one regular, non-symlink package member by package path."""

        path = self._resolve(relative)
        try:
            metadata = path.lstat()
        except OSError as error:
            raise PackageValidationError(f"package member {relative!r} is unreadable") from error
        if not stat.S_ISREG(metadata.st_mode):
            raise PackageValidationError(
                f"package member {relative!r} is not a regular file"
            )
        return path.read_bytes()

    def mode(self, relative: str) -> int:
        """The permission bits of one package member or directory."""

        return stat.S_IMODE(self._resolve(relative).lstat().st_mode)

    def exists(self, relative: str) -> bool:
        """Whether one package path exists, without following a symlink."""

        return self._resolve(relative).exists()

    def walk_files(self, relative_root: str) -> Iterator[str]:
        """Every regular file under one package subtree, as package paths.

        A non-regular entry is refused rather than skipped: a symlink or a
        device node inside a sealed tree is an anomaly the reader must not be
        able to hide by ignoring it.
        """

        root = self._resolve(relative_root)
        for directory, directory_names, file_names in os.walk(root, followlinks=False):
            directory_names.sort()
            for name in sorted(file_names):
                path = Path(directory) / name
                if not stat.S_ISREG(path.lstat().st_mode):
                    raise PackageValidationError(
                        f"package tree {relative_root!r} carries a non-regular "
                        f"entry: {path.relative_to(self._root).as_posix()}"
                    )
                yield path.relative_to(self._root).as_posix()

    def walk_directories(self, relative_root: str) -> Iterator[str]:
        """Every directory under one package subtree, as package paths."""

        root = self._resolve(relative_root)
        for directory, directory_names, _ in os.walk(root, followlinks=False):
            directory_names.sort()
            for name in directory_names:
                path = Path(directory) / name
                yield path.relative_to(self._root).as_posix()


def read_float64_vector(payload: bytes, *, where: str) -> tuple[int, bytes]:
    """Decode a 1-D little-endian float64 ``.npy`` without importing NumPy.

    Returns the element count and the C-contiguous value bytes, which are
    exactly what the campaign's content digest is taken over -- so the array's
    identity is checkable from the package with the standard library alone.
    """

    if payload[:6] != b"\x93NUMPY":
        raise PackageValidationError(f"{where} is not a .npy file")
    major = payload[6]
    if major == 1:
        header_start, header_length = 10, int.from_bytes(payload[8:10], "little")
    elif major == 2:
        header_start, header_length = 12, int.from_bytes(payload[8:12], "little")
    else:
        raise PackageValidationError(f"{where} uses .npy format version {major}")
    header_end = header_start + header_length
    try:
        header = literal_eval(payload[header_start:header_end].decode("latin1").strip())
    except (SyntaxError, ValueError, UnicodeDecodeError) as error:
        raise PackageValidationError(f"{where} has an unreadable .npy header") from error
    if not isinstance(header, dict):
        raise PackageValidationError(f"{where} has no .npy header dictionary")
    shape = header.get("shape")
    if (
        header.get("descr") != FLOAT64_LITTLE_ENDIAN_DESCRIPTION
        or header.get("fortran_order") is not False
        or not isinstance(shape, tuple)
        or len(shape) != 1
    ):
        raise PackageValidationError(
            f"{where} is not a 1-D little-endian float64 array: {header!r}"
        )
    values = payload[header_end:]
    if len(values) != int(shape[0]) * 8:
        raise PackageValidationError(f"{where} .npy payload length disagrees with its shape")
    return int(shape[0]), values


def bare_array_tree_sha256(length: int, values: bytes) -> str:
    """Re-derive ``exact_numeric_tree_sha256`` for one bare float64 vector."""

    digest = hashlib.sha256()
    digest.update(BARE_ARRAY_TREE_DEFINITION_REPR.encode("utf-8"))
    digest.update(b"array\0")
    digest.update(FLOAT64_LITTLE_ENDIAN_DESCRIPTION.encode("utf-8"))
    digest.update(repr((length,)).encode("utf-8"))
    digest.update(values)
    return digest.hexdigest()


def pinned_term_verdict(name: str, terminal: float, native: float) -> dict[str, JsonValue]:
    """Judge one pinned term exactly as ``_pinned_term_verdict`` does."""

    comparison, band = PINNED_ENDPOINT_QUALITY_GATES[name]
    difference = terminal - native
    if comparison == "absolute":
        measured = abs(difference)
    elif comparison == "relative":
        measured = abs(difference) / abs(native)
    else:
        measured = difference / abs(native)
    return {
        "band": band,
        "comparison": comparison,
        "measured": measured,
        "passed": bool(measured <= band),
    }


def _member_table(members: Mapping[str, JsonValue]) -> list[JsonValue]:
    return [
        [path, _mapping(members[path], f"member {path}")["sha256"]]
        for path in sorted(members)
    ]


def composite_sha256(members: Mapping[str, JsonValue]) -> str:
    """The one digest that names this whole package; see the definition string."""

    return sha256_hex(canonical_json_bytes(_member_table(members)))


def _final_members(reader: PackageReader) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
    """Re-derive final/'s membership from its own manifest and its own tree."""

    manifest_bytes = reader.read(f"{FINAL_DIRECTORY}/{ARTIFACT_MANIFEST_FILENAME}")
    manifest = _mapping(
        load_json_bytes(manifest_bytes, where=ARTIFACT_MANIFEST_FILENAME),
        ARTIFACT_MANIFEST_FILENAME,
    )
    _exact_keys(
        manifest,
        frozenset({"directories", "files", "schema_version"}),
        ARTIFACT_MANIFEST_FILENAME,
    )
    if manifest["schema_version"] != ARTIFACT_MANIFEST_SCHEMA_VERSION:
        raise PackageValidationError("artifact manifest states another schema version")
    members: dict[str, JsonValue] = {
        f"{FINAL_DIRECTORY}/{ARTIFACT_MANIFEST_FILENAME}": {
            "sha256": sha256_hex(manifest_bytes),
            "size_bytes": len(manifest_bytes),
        }
    }
    for entry in _sequence(manifest["files"], "artifact manifest files"):
        record = _mapping(entry, "artifact manifest file entry")
        _exact_keys(
            record,
            frozenset({"mode", "relative_path", "sha256", "size_bytes"}),
            "artifact manifest file entry",
        )
        relative = str(record["relative_path"])
        member = f"{FINAL_DIRECTORY}/{relative}"
        if member in members:
            raise PackageValidationError(f"artifact manifest lists {relative!r} twice")
        payload = reader.read(member)
        digest = sha256_hex(payload)
        if digest != _require_sha256(record["sha256"], f"manifest entry {relative}"):
            raise PackageValidationError(
                f"final/{relative} hashes to {digest}, not the manifest's "
                f"{record['sha256']}"
            )
        if len(payload) != int(record["size_bytes"]):
            raise PackageValidationError(f"final/{relative} is not the manifest's size")
        members[member] = {"sha256": digest, "size_bytes": len(payload)}
    on_disk = frozenset(reader.walk_files(FINAL_DIRECTORY))
    if on_disk != frozenset(members):
        raise PackageValidationError(
            "final/ tree is not its manifest's membership: unclaimed "
            f"{sorted(on_disk - frozenset(members))}, missing "
            f"{sorted(frozenset(members) - on_disk)}"
        )
    declared_directories = {
        f"{FINAL_DIRECTORY}/{_mapping(entry, 'artifact manifest directory')['relative_path']}"
        for entry in _sequence(manifest["directories"], "artifact manifest directories")
    }
    if declared_directories != frozenset(reader.walk_directories(FINAL_DIRECTORY)):
        raise PackageValidationError("final/ directory tree is not its manifest's")
    return members, manifest


def _supplement_members(
    reader: PackageReader,
) -> tuple[dict[str, JsonValue], dict[str, JsonValue], dict[str, dict[str, JsonValue]]]:
    """Re-derive the supplement's membership from the supplement manifest."""

    manifest_member = f"{SUPPLEMENT_DIRECTORY}/{SUPPLEMENT_MANIFEST_FILENAME}"
    manifest_bytes = reader.read(manifest_member)
    manifest = _mapping(
        load_json_bytes(manifest_bytes, where=SUPPLEMENT_MANIFEST_FILENAME),
        SUPPLEMENT_MANIFEST_FILENAME,
    )
    _exact_keys(
        manifest,
        frozenset(
            {
                "added_post_seal",
                "added_utc",
                "direction",
                "entries",
                "entry_count",
                "purpose",
                "schema_version",
                "supplements",
                "verification",
            }
        ),
        SUPPLEMENT_MANIFEST_FILENAME,
    )
    if manifest["schema_version"] != SUPPLEMENT_SCHEMA_VERSION:
        raise PackageValidationError("supplement manifest states another schema version")
    entries = _sequence(manifest["entries"], "supplement entries")
    if len(entries) != int(manifest["entry_count"]):
        raise PackageValidationError("supplement entry_count is not its entry list length")
    members: dict[str, JsonValue] = {
        manifest_member: {
            "sha256": sha256_hex(manifest_bytes),
            "size_bytes": len(manifest_bytes),
        }
    }
    by_role: dict[str, dict[str, JsonValue]] = {}
    for entry in entries:
        record = _mapping(entry, "supplement entry")
        required = frozenset(
            {"file_sha256", "name", "role", "size_bytes", "source_absolute_path", "why_outside_final"}
        )
        if not required <= frozenset(record):
            raise PackageValidationError(
                f"supplement entry keys differ: missing {sorted(required - frozenset(record))}"
            )
        name = str(record["name"])
        member = f"{SUPPLEMENT_DIRECTORY}/{name}"
        payload = reader.read(member)
        digest = sha256_hex(payload)
        if digest != _require_sha256(record["file_sha256"], f"supplement entry {name}"):
            raise PackageValidationError(
                f"{member} hashes to {digest}, not the supplement's {record['file_sha256']}"
            )
        if len(payload) != int(record["size_bytes"]):
            raise PackageValidationError(f"{member} is not the supplement's size")
        members[member] = {"sha256": digest, "size_bytes": len(payload)}
        role = str(record["role"])
        if role in by_role:
            raise PackageValidationError(f"supplement carries two {role!r} entries")
        by_role[role] = record
    on_disk = frozenset(reader.walk_files(SUPPLEMENT_DIRECTORY))
    if on_disk != frozenset(members):
        raise PackageValidationError(
            "provenance-supplement/ tree is not its manifest's membership: unclaimed "
            f"{sorted(on_disk - frozenset(members))}, missing "
            f"{sorted(frozenset(members) - on_disk)}"
        )
    if list(reader.walk_directories(SUPPLEMENT_DIRECTORY)):
        raise PackageValidationError("provenance-supplement/ carries a subdirectory")
    return members, manifest, by_role


def _quality_gate(ledger: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    """Recompute the pinned ten from the ledger's own two sides."""

    terminal = _mapping(ledger["terminal"], "endpoint ledger terminal side")
    native = _mapping(ledger["native"], "endpoint ledger native side")
    declared = _mapping(ledger["pinned_term_gate"], "pinned term gate")
    declared_terms = _mapping(declared["terms"], "pinned term gate terms")
    if frozenset(
        str(term) for term in _sequence(ledger["pinned_quality_terms"], "pinned terms")
    ) != frozenset(PINNED_ENDPOINT_QUALITY_GATES):
        raise PackageValidationError(
            "the ledger's pinned term set is not the gate table of section 1.1"
        )
    if frozenset(declared_terms) != frozenset(PINNED_ENDPOINT_QUALITY_GATES):
        raise PackageValidationError("the published gate judges another term set")
    if frozenset(
        str(term)
        for term in _sequence(ledger["informational_observables"], "informational terms")
    ) != frozenset(INFORMATIONAL_ENDPOINT_OBSERVABLES):
        raise PackageValidationError("the ledger names other informational observables")
    if frozenset(INFORMATIONAL_ENDPOINT_OBSERVABLES) & frozenset(
        PINNED_ENDPOINT_QUALITY_GATES
    ):
        raise PackageValidationError("an informational observable is being gated")
    if not bool(ledger["gated_at_this_budget"]):
        raise PackageValidationError("the discharging attempt's physics gate did not run")
    terms: dict[str, JsonValue] = {}
    for name in sorted(PINNED_ENDPOINT_QUALITY_GATES):
        verdict = pinned_term_verdict(
            name,
            _float(terminal[name], f"terminal {name}"),
            _float(native[name], f"native {name}"),
        )
        published = _mapping(declared_terms[name], f"published gate term {name}")
        if (
            published["comparison"] != verdict["comparison"]
            or _float(published["band"], f"published band {name}") != verdict["band"]
            or _float(published["measured"], f"published measure {name}") != verdict["measured"]
            or bool(published["passed"]) is not verdict["passed"]
        ):
            raise PackageValidationError(
                f"pinned term {name} re-derives to {verdict!r}, not the published "
                f"{published!r}"
            )
        if not verdict["passed"]:
            raise PackageValidationError(f"pinned term {name} does not pass its band")
        terms[name] = verdict
    failed = _sequence(declared["failed_terms"], "failed terms")
    if failed or not bool(declared["passed"]):
        raise PackageValidationError(f"the published gate did not pass: {failed}")
    return {"passed": True, "pinned_term_count": len(terms), "terms": terms}


def _executed_source_closure(
    reader: PackageReader,
    *,
    attempt_evidence: Mapping[str, JsonValue],
    final_members: Mapping[str, JsonValue],
    authority_name: str,
) -> dict[str, JsonValue]:
    """Bind every module that EXECUTED to bytes this package carries.

    This is the closure the certificate makes: the authority manifest is the
    repository-wide freeze and is cited by digest, while the modules that
    actually ran are resolved, one by one, to bytes inside the package.
    """

    execution_sources = _mapping(attempt_evidence["execution_sources"], "execution_sources")
    manifest_evidence = _mapping(execution_sources["manifest"], "execution-source manifest")
    authority_member = f"{SUPPLEMENT_DIRECTORY}/{authority_name}"
    authority_bytes = reader.read(authority_member)
    if sha256_hex(authority_bytes) != _require_sha256(
        manifest_evidence["manifest_sha256"], "cited authority digest"
    ):
        raise PackageValidationError(
            "the supplement's execution-source authority is not the one the receipt cites"
        )
    authority = _mapping(
        load_json_bytes(authority_bytes, where=authority_name), authority_name
    )
    _exact_keys(
        authority,
        frozenset({"entries", "entries_sha256", "schema_version"}),
        authority_name,
    )
    if authority["schema_version"] != AUTHORITY_SCHEMA_VERSION:
        raise PackageValidationError("the execution-source authority states another schema")
    entries = _mapping(authority["entries"], "authority entries")
    if sha256_hex(canonical_json_bytes(entries)) != _require_sha256(
        authority["entries_sha256"], "authority entries digest"
    ):
        raise PackageValidationError("the authority's own entries digest disagrees")
    if str(manifest_evidence["entries_sha256"]) != str(authority["entries_sha256"]):
        raise PackageValidationError("the receipt cites another authority entries digest")
    if len(entries) != int(manifest_evidence["entry_count"]):
        raise PackageValidationError("the receipt cites another authority entry count")

    snapshot_prefix = f"{FINAL_DIRECTORY}/{SOURCE_SNAPSHOT_DIRECTORY}/"
    supplement_sources: dict[str, str] = {}
    for member, record in _supplement_source_paths(reader).items():
        supplement_sources[record] = member
    in_snapshot = 0
    in_supplement = 0
    bound_paths: set[str] = set()
    for entry in _sequence(execution_sources["bound_modules"], "bound modules"):
        module = _mapping(entry, "bound module")
        _exact_keys(
            module,
            frozenset({"module", "relative_path", "sha256", "size_bytes"}),
            "bound module",
        )
        relative = str(module["relative_path"])
        digest = _require_sha256(module["sha256"], f"bound module {relative}")
        authority_entry = entries.get(relative)
        if authority_entry is None or _mapping(
            authority_entry, f"authority entry {relative}"
        )["sha256"] != digest:
            raise PackageValidationError(
                f"bound module {relative!r} is not the authority's byte identity"
            )
        snapshot_member = f"{snapshot_prefix}{relative}"
        if snapshot_member in final_members:
            carried = _mapping(final_members[snapshot_member], snapshot_member)["sha256"]
            in_snapshot += 1
        elif relative in supplement_sources:
            carried = sha256_hex(reader.read(supplement_sources[relative]))
            in_supplement += 1
        else:
            raise PackageValidationError(
                f"executed module {relative!r} has no bytes in this package"
            )
        if carried != digest:
            raise PackageValidationError(
                f"executed module {relative!r} is carried at {carried}, not {digest}"
            )
        bound_paths.add(relative)
    if not bound_paths:
        raise PackageValidationError("the attempt binds no module at all")
    if _sequence(
        execution_sources["unmanifested_repository_modules"], "unmanifested modules"
    ):
        raise PackageValidationError(
            "the attempt bound repository modules outside the manifest's roots"
        )
    package_paths = {
        member[len(snapshot_prefix) :]
        for member in final_members
        if member.startswith(snapshot_prefix)
    } | frozenset(supplement_sources)
    absent = sorted(frozenset(entries) - package_paths)
    executed_but_absent = sorted(frozenset(absent) & bound_paths)
    if executed_but_absent:
        raise PackageValidationError(
            f"executed modules are missing from the package: {executed_but_absent}"
        )
    return {
        "authority_cited_not_embedded_members": len(absent),
        "authority_entries_sha256": str(authority["entries_sha256"]),
        "authority_entry_count": len(entries),
        "authority_manifest_sha256": sha256_hex(authority_bytes),
        "authority_member": authority_member,
        "authority_members_absent_from_package": absent,
        "bound_module_count": len(bound_paths),
        "bound_modules_in_final_snapshot": in_snapshot,
        "bound_modules_in_supplement": in_supplement,
        "closure": "executed-source",
        "statement": EXECUTED_SOURCE_CLOSURE_STATEMENT,
    }


def _supplement_source_paths(reader: PackageReader) -> dict[str, str]:
    """Map each supplement member carrying module bytes to its repository path."""

    manifest = _mapping(
        load_json_bytes(
            reader.read(f"{SUPPLEMENT_DIRECTORY}/{SUPPLEMENT_MANIFEST_FILENAME}"),
            where=SUPPLEMENT_MANIFEST_FILENAME,
        ),
        SUPPLEMENT_MANIFEST_FILENAME,
    )
    sources: dict[str, str] = {}
    for entry in _sequence(manifest["entries"], "supplement entries"):
        record = _mapping(entry, "supplement entry")
        if record["role"] != "executed_module_bytes":
            continue
        absolute = str(record["source_absolute_path"])
        marker = "/src/"
        if marker not in absolute:
            raise PackageValidationError(
                f"executed module bytes {record['name']!r} name no repository path"
            )
        relative = f"src/{absolute.rsplit(marker, 1)[1]}"
        sources[f"{SUPPLEMENT_DIRECTORY}/{record['name']}"] = relative
    return sources


def _native_endpoint_reference(
    reader: PackageReader,
    *,
    evidence: Mapping[str, JsonValue],
    ledger: Mapping[str, JsonValue],
    entry: Mapping[str, JsonValue],
) -> dict[str, JsonValue]:
    """Check the gate's reference side against the copy this package carries."""

    member = f"{SUPPLEMENT_DIRECTORY}/{entry['name']}"
    payload = reader.read(member)
    file_digest = sha256_hex(payload)
    length, values = read_float64_vector(payload, where=member)
    content_digest = sha256_hex(values)
    preflight = _mapping(
        _mapping(evidence["supervisor"], "supervisor")["preflight"], "preflight"
    )
    pinned = {
        "ledger file": str(ledger["native_state_sha256"]),
        "ledger content": str(ledger["native_state_content_sha256"]),
        "preflight file": str(preflight["native_endpoint_state_sha256"]),
        "preflight content": str(preflight["native_endpoint_state_content_sha256"]),
        "supplement file": str(entry["file_sha256"]),
        "supplement content": str(entry["content_sha256"]),
    }
    for where, digest in pinned.items():
        expected = content_digest if where.endswith("content") else file_digest
        if _require_sha256(digest, where) != expected:
            raise PackageValidationError(
                f"the native endpoint state the package carries disagrees with the "
                f"{where} digest: {digest} != {expected}"
            )
    if str(ledger["native_state_relative_path"]) != f"{content_digest}.npy":
        raise PackageValidationError(
            "the receipt names a native state file that is not its content digest"
        )
    if str(preflight["native_endpoint_state_path"]).rsplit("/", 1)[-1] != (
        f"{content_digest}.npy"
    ):
        raise PackageValidationError(
            "the preflight names a native state path that is not its content digest"
        )
    return {
        "content_sha256": content_digest,
        "element_count": length,
        "file_sha256": file_digest,
        "member": member,
        "source_absolute_path": str(preflight["native_endpoint_state_path"]),
    }


def _timing(evidence: Mapping[str, JsonValue], attempt: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    """Recompute every published boundary's ratio against the frozen bar."""

    bar = _float(_mapping(evidence["claim"], "claim")["wall_seconds_bar"], "bar")
    timing = _mapping(
        _mapping(attempt["evidence"], "attempt evidence")["timing_seconds"], "attempt timing"
    )
    cold = _mapping(evidence["cold_lane"], "cold lane")
    cold_timing = _mapping(
        _mapping(cold["evidence"], "cold-lane evidence")["timing_seconds"], "cold timing"
    )
    boundaries = {
        "certified_engine_compile_plus_solve": _float(timing["engine_wall"], "engine wall"),
        "cold_lane_engine_compile_plus_solve": _float(
            cold_timing["engine_wall"], "cold engine wall"
        ),
        "cold_lane_internal_attempt_wall": _float(
            cold_timing["attempt_wall"], "cold attempt wall"
        ),
        "cold_lane_supervised_wall": _float(
            cold["supervised_seconds"], "cold supervised wall"
        ),
        "warm_internal_attempt_wall": _float(timing["attempt_wall"], "attempt wall"),
        "warm_supervised_wall": _float(attempt["supervised_seconds"], "supervised wall"),
    }
    if str(evidence["timing_boundary"]) != "engine_compile_plus_solve":
        raise PackageValidationError("the receipt states another certified timing boundary")
    for name, seconds in boundaries.items():
        if seconds <= 0.0 or seconds >= bar:
            raise PackageValidationError(
                f"boundary {name} ({seconds} s) does not beat the {bar} s bar"
            )
    return {
        "bar_seconds": bar,
        "boundary_seconds": boundaries,
        "ratios": {name: bar / seconds for name, seconds in boundaries.items()},
        "strictest_boundary": "cold_lane_supervised_wall",
    }


def _canonical_corrections(reader: PackageReader) -> list[JsonValue]:
    """State, per supplement JSON, whether its sealed bytes are canonical.

    The sealed originals are never rewritten.  Where the producer wrote a
    pretty-printed document, this package carries the canonical re-serialization
    beside the binding, and the correction records both digests so a reader can
    see exactly what was normalized and what was not.
    """

    corrections: list[JsonValue] = []
    for name in sorted(
        member.rsplit("/", 1)[-1]
        for member in reader.walk_files(SUPPLEMENT_DIRECTORY)
        if member.endswith(".json")
    ):
        member = f"{SUPPLEMENT_DIRECTORY}/{name}"
        payload = reader.read(member)
        document = load_json_bytes(payload, where=name)
        canonical = canonical_json_bytes(document)
        is_canonical = canonical == payload
        record: dict[str, JsonValue] = {
            "canonical_sha256": sha256_hex(canonical),
            "member": member,
            "original_is_canonical": is_canonical,
            "original_sha256": sha256_hex(payload),
        }
        if not is_canonical:
            record["canonical_relative_path"] = (
                f"{COMPOSITE_DIRECTORY}/{CANONICAL_SUBDIRECTORY}/{name}"
            )
        corrections.append(record)
    return corrections


def _validate_revalidation_record(reader: PackageReader, *, member: str) -> dict[str, JsonValue]:
    """Schema-check the cross-version revalidation record the supplement carries."""

    record = _mapping(load_json_bytes(reader.read(member), where=member), member)
    required = frozenset(
        {
            "added_post_seal",
            "added_utc",
            "attempted_and_refused",
            "conclusion",
            "lanes",
            "method",
            "result_digest_meaning",
            "schema_version",
            "target",
            "why",
        }
    )
    _exact_keys(record, required, member)
    if record["schema_version"] != REVALIDATION_SCHEMA_VERSION:
        raise PackageValidationError(f"{member} states another schema version")
    digests: set[str] = set()
    versions: list[str] = []
    for lane in _sequence(record["lanes"], f"{member} lanes"):
        entry = _mapping(lane, f"{member} lane")
        lane_keys = frozenset(
            {
                "different_version_from_producer",
                "environment",
                "ephemeral",
                "jax_version",
                "jaxlib_version",
                "lane",
                "outcome",
                "python_version",
                "ran_utc",
                "result_sha256",
                "verdict_recomputed",
            }
        )
        _exact_keys(entry, lane_keys, f"{member} lane")
        if entry["outcome"] != "PASS" or entry["verdict_recomputed"] != VERDICT_CLAIM_DISCHARGED:
            raise PackageValidationError(f"{member} carries a lane that did not pass")
        digests.add(_require_sha256(entry["result_sha256"], f"{member} lane result"))
        versions.append(str(entry["jaxlib_version"]))
    if len(digests) != 1:
        raise PackageValidationError(
            f"{member} lanes do not agree on one result document: {sorted(digests)}"
        )
    return {
        "jaxlib_versions": sorted(versions),
        "lane_count": len(versions),
        "member": member,
        "result_sha256": digests.pop(),
    }


def build_composite_document(reader: PackageReader) -> dict[str, JsonValue]:
    """Derive the whole composite binding from the package's bytes.

    Every number in the returned document is recomputed here; nothing is copied
    from a field that states it.  Validation is this function run again and
    compared, which is why it must stay free of clocks, environment and paths.
    """

    final_members, _ = _final_members(reader)
    supplement_members, supplement_manifest, by_role = _supplement_members(reader)
    members: dict[str, JsonValue] = {**final_members, **supplement_members}

    root_member = f"{FINAL_DIRECTORY}/{ROOT_EVIDENCE_FILENAME}"
    root_bytes = reader.read(root_member)
    root_digest = sha256_hex(root_bytes)
    evidence = _mapping(load_json_bytes(root_bytes, where=ROOT_EVIDENCE_FILENAME), "root evidence")
    if evidence["schema_version"] != ROOT_EVIDENCE_SCHEMA_VERSION:
        raise PackageValidationError("root evidence states another schema version")
    if evidence["verdict"] != VERDICT_CLAIM_DISCHARGED:
        raise PackageValidationError(f"root evidence publishes {evidence['verdict']!r}")
    supplements = _mapping(supplement_manifest["supplements"], "supplement target")
    if str(supplements["root_evidence_sha256"]) != root_digest:
        raise PackageValidationError(
            "the supplement supplements another root evidence document"
        )
    if str(supplements["artifact_relative_path"]) != FINAL_DIRECTORY:
        raise PackageValidationError("the supplement names another artifact directory")

    attempts = _sequence(evidence["attempts"], "attempts")
    discharging = [
        _mapping(attempt, "attempt")
        for attempt in attempts
        if _mapping(attempt, "attempt")["outcome"] == "LATCHED"
    ]
    if len(discharging) != 1:
        raise PackageValidationError(
            f"the receipt publishes {len(discharging)} latched attempts, not one"
        )
    attempt = discharging[0]
    attempt_evidence = _mapping(attempt["evidence"], "attempt evidence")
    ledger = _mapping(attempt_evidence["endpoint_ledger"], "endpoint ledger")

    terminal_member = f"{FINAL_DIRECTORY}/{attempt['artifact_relative_path']}/terminal-coordinates.npy"
    length, values = read_float64_vector(reader.read(terminal_member), where=terminal_member)
    agreement = _mapping(attempt_evidence["endpoint_agreement"], "endpoint agreement")
    terminal_digest = bare_array_tree_sha256(length, values)
    if terminal_digest != _require_sha256(
        agreement["terminal_state_sha256"], "published terminal state digest"
    ):
        raise PackageValidationError(
            f"the published endpoint coordinates hash to {terminal_digest}, not the "
            f"receipt's {agreement['terminal_state_sha256']}"
        )

    claim = _mapping(evidence["claim"], "claim")
    solve = _mapping(attempt_evidence["solve"], "solve")
    target = _float(claim["target_objective"], "target objective")
    tolerance = _float(claim["feasibility_tolerance"], "feasibility tolerance")
    terminal_objective = _float(solve["terminal_objective"], "terminal objective")
    maximum_feasibility = _float(solve["maximum_feasibility_inf"], "whole-run feasibility")
    if terminal_objective > target:
        raise PackageValidationError("the latched attempt did not reach the target objective")
    if maximum_feasibility > tolerance:
        raise PackageValidationError("the whole-run feasibility bound is not satisfied")

    return {
        "canonical_corrections": _canonical_corrections(reader),
        "composite_sha256": composite_sha256(members),
        "composite_sha256_definition": COMPOSITE_SHA256_DEFINITION,
        "executed_source_closure": _executed_source_closure(
            reader,
            attempt_evidence=attempt_evidence,
            final_members=final_members,
            authority_name=str(by_role["execution_authority_manifest"]["name"]),
        ),
        "final": {
            "artifact_manifest_sha256": str(
                _mapping(
                    members[f"{FINAL_DIRECTORY}/{ARTIFACT_MANIFEST_FILENAME}"], "manifest member"
                )["sha256"]
            ),
            "file_count": len(final_members),
            "manifest_member_count": len(final_members) - 1,
            "root_evidence_sha256": root_digest,
        },
        "latch": {
            "maximum_feasibility_inf": maximum_feasibility,
            "feasibility_tolerance": tolerance,
            "target_objective": target,
            "terminal_objective": terminal_objective,
            "terminal_state_sha256": terminal_digest,
        },
        "member_count": len(members),
        "members": members,
        "native_endpoint_reference": _native_endpoint_reference(
            reader,
            evidence=evidence,
            ledger=ledger,
            entry=by_role["native_endpoint_reference_state"],
        ),
        "purpose": COMPOSITE_PURPOSE,
        "quality_gate": _quality_gate(ledger),
        "revalidation": _validate_revalidation_record(
            reader,
            member=f"{SUPPLEMENT_DIRECTORY}/{by_role['revalidation_record']['name']}",
        ),
        "schema_version": COMPOSITE_SCHEMA_VERSION,
        "supplement": {
            "entry_count": int(supplement_manifest["entry_count"]),
            "manifest_sha256": str(
                _mapping(
                    members[f"{SUPPLEMENT_DIRECTORY}/{SUPPLEMENT_MANIFEST_FILENAME}"],
                    "supplement manifest member",
                )["sha256"]
            ),
        },
        "timing": _timing(evidence, attempt),
        "verdict": str(evidence["verdict"]),
    }


def composite_directory_payloads(document: Mapping[str, JsonValue]) -> dict[str, bytes]:
    """Every byte the composite directory must carry for one derived binding."""

    payloads = {
        f"{COMPOSITE_DIRECTORY}/{COMPOSITE_MANIFEST_FILENAME}": canonical_json_bytes(
            dict(document)
        )
    }
    return payloads


def emit_composite_manifest(
    package_root: Path, canonical_sources: Mapping[str, bytes] | None = None
) -> dict[str, JsonValue]:
    """Write the composite directory beside the sealed trees it binds."""

    reader = PackageReader(package_root)
    document = build_composite_document(reader)
    composite_root = reader.root / COMPOSITE_DIRECTORY
    canonical_root = composite_root / CANONICAL_SUBDIRECTORY
    composite_root.mkdir(exist_ok=True)
    corrections = [
        _mapping(record, "canonical correction")
        for record in _sequence(document["canonical_corrections"], "canonical corrections")
    ]
    if any("canonical_relative_path" in record for record in corrections):
        canonical_root.mkdir(exist_ok=True)
    for record in corrections:
        relative = record.get("canonical_relative_path")
        if relative is None:
            continue
        member = str(record["member"])
        payload = canonical_json_bytes(load_json_bytes(reader.read(member), where=member))
        (reader.root / str(relative)).write_bytes(payload)
    for relative, payload in composite_directory_payloads(document).items():
        (reader.root / relative).write_bytes(payload)
    return document


def _validate_composite_directory(
    reader: PackageReader, document: Mapping[str, JsonValue]
) -> None:
    """The composite directory carries the binding and its corrections, exactly."""

    expected: dict[str, bytes] = dict(composite_directory_payloads(document))
    for record in _sequence(document["canonical_corrections"], "canonical corrections"):
        correction = _mapping(record, "canonical correction")
        relative = correction.get("canonical_relative_path")
        if relative is None:
            continue
        member = str(correction["member"])
        payload = canonical_json_bytes(load_json_bytes(reader.read(member), where=member))
        if sha256_hex(payload) != str(correction["canonical_sha256"]):
            raise PackageValidationError(
                f"the canonical re-serialization of {member} is not its recorded digest"
            )
        expected[str(relative)] = payload
    on_disk = frozenset(reader.walk_files(COMPOSITE_DIRECTORY))
    if on_disk != frozenset(expected):
        raise PackageValidationError(
            f"{COMPOSITE_DIRECTORY}/ tree differs: unclaimed "
            f"{sorted(on_disk - frozenset(expected))}, missing "
            f"{sorted(frozenset(expected) - on_disk)}"
        )
    for relative, payload in expected.items():
        carried = reader.read(relative)
        if carried != payload:
            raise PackageValidationError(f"{relative} is not the bytes it must carry")


def _validate_sealed_modes(reader: PackageReader) -> None:
    """Every package file is 0444 and every package directory is 0555."""

    for directory in PACKAGE_DIRECTORIES:
        for relative in (directory, *reader.walk_directories(directory)):
            mode = reader.mode(relative)
            if mode != SEALED_DIRECTORY_MODE:
                raise PackageValidationError(
                    f"package directory {relative} is {mode:04o}, not "
                    f"{SEALED_DIRECTORY_MODE:04o}"
                )
        for relative in reader.walk_files(directory):
            mode = reader.mode(relative)
            if mode != SEALED_FILE_MODE:
                raise PackageValidationError(
                    f"package member {relative} is {mode:04o}, not {SEALED_FILE_MODE:04o}"
                )


def validate_package(
    package_root: Path,
    *,
    expect_composite_sha256: str | None = None,
    require_sealed_modes: bool = True,
) -> dict[str, JsonValue]:
    """Re-derive the whole package binding and judge the stored one against it."""

    reader = PackageReader(package_root)
    for directory in PACKAGE_DIRECTORIES:
        if not reader.exists(directory):
            raise PackageValidationError(f"the package carries no {directory}/ directory")
    if require_sealed_modes:
        _validate_sealed_modes(reader)
    document = build_composite_document(reader)
    stored_bytes = reader.read(f"{COMPOSITE_DIRECTORY}/{COMPOSITE_MANIFEST_FILENAME}")
    rebuilt_bytes = canonical_json_bytes(document)
    if stored_bytes != rebuilt_bytes:
        stored = load_json_bytes(stored_bytes, where=COMPOSITE_MANIFEST_FILENAME)
        if canonical_json_bytes(stored) != stored_bytes:
            raise PackageValidationError(
                f"{COMPOSITE_MANIFEST_FILENAME} is not canonically encoded"
            )
        differing = sorted(
            key
            for key in frozenset(_mapping(stored, "stored binding")) | frozenset(document)
            if _mapping(stored, "stored binding").get(key) != document.get(key)
        )
        raise PackageValidationError(
            f"the stored binding is not the one this package derives; differing "
            f"blocks: {differing}"
        )
    derived = str(document["composite_sha256"])
    if derived != composite_sha256(_mapping(document["members"], "members")):
        raise PackageValidationError("the composite digest is not its own member table's")
    if expect_composite_sha256 is not None and derived != _require_sha256(
        expect_composite_sha256, "expected composite digest"
    ):
        raise PackageValidationError(
            f"package composite digest is {derived}, not the expected "
            f"{expect_composite_sha256}"
        )
    _validate_composite_directory(reader, document)
    closure = _mapping(document["executed_source_closure"], "closure")
    return {
        "composite_sha256": derived,
        "executed_module_count": closure["bound_module_count"],
        "authority_entry_count": closure["authority_entry_count"],
        "authority_cited_not_embedded_members": closure["authority_cited_not_embedded_members"],
        "member_count": document["member_count"],
        "package_root": str(reader.root),
        "quality_gate_terms": _mapping(document["quality_gate"], "gate")["pinned_term_count"],
        "root_evidence_sha256": _mapping(document["final"], "final")["root_evidence_sha256"],
        "schema_version": RESULT_SCHEMA_VERSION,
        "sealed_modes_checked": require_sealed_modes,
        "verdict": PACKAGE_VERDICT,
        "receipt_verdict": document["verdict"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Validate one package, or emit the composite binding for a new one."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package_root", type=Path)
    parser.add_argument("--expect-composite-sha256", default=None)
    parser.add_argument("--emit-composite-manifest", action="store_true")
    parser.add_argument("--no-require-sealed-modes", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        if arguments.emit_composite_manifest:
            document = emit_composite_manifest(arguments.package_root)
            sys.stdout.buffer.write(
                canonical_json_bytes(
                    {
                        "composite_sha256": document["composite_sha256"],
                        "emitted": f"{COMPOSITE_DIRECTORY}/{COMPOSITE_MANIFEST_FILENAME}",
                        "member_count": document["member_count"],
                    }
                )
            )
            return 0
        result = validate_package(
            arguments.package_root,
            expect_composite_sha256=arguments.expect_composite_sha256,
            require_sealed_modes=not arguments.no_require_sealed_modes,
        )
    except PackageValidationError as failure:
        sys.stdout.buffer.write(
            canonical_json_bytes({"refusal": str(failure), "verdict": "PACKAGE_REFUSED"})
        )
        return 1
    sys.stdout.buffer.write(canonical_json_bytes(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
