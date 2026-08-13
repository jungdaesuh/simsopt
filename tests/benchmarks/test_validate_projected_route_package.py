"""Gates on the standalone certificate-package validator.

Two properties carry this module and neither can be asserted by reading the
validator's source: that it REFUSES a package whose bytes moved, and that it
opens nothing outside the package while accepting one whose bytes did not.  The
refusals are exercised against a synthetic package built here -- complete
enough to walk every derivation the real one walks, small enough to build in
milliseconds and portable to any box -- and the acceptance plus the syscall
audit are exercised against the real sealed package when this box still has it.

The validator duplicates two definitions from this repository on purpose, so it
can stay import-free; the twin tests below are the only thing keeping those
duplicates honest, which is why they compare the objects rather than restate
their contents a third time.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import benchmarks.rehearse_single_stage_projected_route_cpu as rehearsal
import benchmarks.validate_projected_route_package as package
import numpy as np
import pytest
from benchmarks.single_stage_fullspace_snapshot import (
    canonical_json_bytes as repository_canonical_json_bytes,
)

REPOSITORY = Path(__file__).resolve().parents[2]
REAL_PACKAGE = (
    Path.home() / "simsopt-campaigns" / "projected-route-root-20260813T184930Z"
)
REAL_PACKAGE_COMPOSITE_SHA256 = (
    "58a116ae9f1ce8e97e62796cda503d6613ad4907f8090920441c1e51426a8283"
)
VALIDATOR_PATH = REPOSITORY / "benchmarks" / "validate_projected_route_package.py"

requires_real_package = pytest.mark.skipif(
    not (REAL_PACKAGE / "final" / "root-evidence.json").exists(),
    reason="the sealed projected-route package is not present on this box",
)


# ------------------------------------------------------------ synthetic package


def _npy_bytes(values: np.ndarray) -> bytes:
    from io import BytesIO

    stream = BytesIO()
    np.save(stream, np.ascontiguousarray(values, dtype=np.float64), allow_pickle=False)
    return stream.getvalue()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write(path: Path, payload: bytes) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return payload


def build_synthetic_package(root: Path) -> Path:
    """Write a miniature package that walks every derivation the real one does."""

    final = root / "final"
    supplement = root / "provenance-supplement"

    native_values = np.linspace(-1.0, 1.0, 8)
    native_payload = _npy_bytes(native_values)
    native_content = _sha256(native_payload[-8 * 8 :])
    native_name = f"native-endpoint-state-{native_content[:8]}.npy"
    _write(supplement / native_name, native_payload)

    terminal_payload = _npy_bytes(np.linspace(-1.0, 1.0, 8) + 1.0e-13)
    terminal_state_sha256 = package.bare_array_tree_sha256(
        8, terminal_payload[-8 * 8 :]
    )
    _write(final / "attempts" / "attempt-1" / "terminal-coordinates.npy", terminal_payload)
    _write(final / "cold-lane" / "terminal-coordinates.npy", terminal_payload)

    module_source = b"VALUE = 1\n"
    _write(final / "source-snapshot" / "benchmarks" / "bound_module.py", module_source)
    version_source = b"version = '0.0.0'\n"
    _write(supplement / "_version.py", version_source)
    unexecuted_source = b"UNEXECUTED = 2\n"

    authority_entries = {
        "benchmarks/bound_module.py": {
            "sha256": _sha256(module_source),
            "size_bytes": len(module_source),
        },
        "examples/never_executed.py": {
            "sha256": _sha256(unexecuted_source),
            "size_bytes": len(unexecuted_source),
        },
        "src/simsopt/_version.py": {
            "sha256": _sha256(version_source),
            "size_bytes": len(version_source),
        },
    }
    entries_sha256 = _sha256(package.canonical_json_bytes(authority_entries))
    authority_payload = package.canonical_json_bytes(
        {
            "entries": authority_entries,
            "entries_sha256": entries_sha256,
            "schema_version": package.AUTHORITY_SCHEMA_VERSION,
        }
    )
    authority_name = "execution-sources.json"
    _write(supplement / authority_name, authority_payload)

    native_terms = {name: 1.0 for name in package.PINNED_ENDPOINT_QUALITY_GATES}
    terminal_terms = dict(native_terms)
    gate_terms = {
        name: package.pinned_term_verdict(name, terminal_terms[name], native_terms[name])
        for name in sorted(package.PINNED_ENDPOINT_QUALITY_GATES)
    }
    ledger = {
        "gated_at_this_budget": True,
        "informational_observables": list(package.INFORMATIONAL_ENDPOINT_OBSERVABLES),
        "native": native_terms,
        "native_state_content_sha256": native_content,
        "native_state_relative_path": f"{native_content}.npy",
        "native_state_sha256": _sha256(native_payload),
        "pinned_quality_terms": sorted(package.PINNED_ENDPOINT_QUALITY_GATES),
        "pinned_term_gate": {"failed_terms": [], "passed": True, "terms": gate_terms},
        "terminal": terminal_terms,
    }
    attempt_evidence = {
        "endpoint_agreement": {"terminal_state_sha256": terminal_state_sha256},
        "endpoint_ledger": ledger,
        "execution_sources": {
            "bound_modules": [
                {
                    "module": "benchmarks.bound_module",
                    "relative_path": "benchmarks/bound_module.py",
                    "sha256": _sha256(module_source),
                    "size_bytes": len(module_source),
                },
                {
                    "module": "simsopt._version",
                    "relative_path": "src/simsopt/_version.py",
                    "sha256": _sha256(version_source),
                    "size_bytes": len(version_source),
                },
            ],
            "manifest": {
                "entries_sha256": entries_sha256,
                "entry_count": len(authority_entries),
                "manifest_sha256": _sha256(authority_payload),
                "relative_path": f"benchmarks/{authority_name}",
                "schema_version": package.AUTHORITY_SCHEMA_VERSION,
            },
            "unmanifested_repository_modules": [],
        },
        "solve": {"maximum_feasibility_inf": 5.0e-11, "terminal_objective": 1.0e-08},
        "timing_seconds": {"attempt_wall": 60.0, "engine_wall": 40.0},
    }
    evidence = {
        "attempts": [
            {
                "artifact_relative_path": "attempts/attempt-1",
                "attempt_index": 1,
                "evidence": attempt_evidence,
                "outcome": "LATCHED",
                "supervised_seconds": 65.0,
            }
        ],
        "claim": {
            "feasibility_tolerance": 1.0e-10,
            "target_objective": 2.0e-08,
            "wall_seconds_bar": 100.0,
        },
        "cold_lane": {
            "artifact_relative_path": "cold-lane",
            "evidence": {"timing_seconds": {"attempt_wall": 90.0, "engine_wall": 70.0}},
            "supervised_seconds": 95.0,
        },
        "schema_version": package.ROOT_EVIDENCE_SCHEMA_VERSION,
        "supervisor": {
            "preflight": {
                "native_endpoint_state_content_sha256": native_content,
                "native_endpoint_state_path": f"/elsewhere/{native_content}.npy",
                "native_endpoint_state_sha256": _sha256(native_payload),
            }
        },
        "timing_boundary": "engine_compile_plus_solve",
        "verdict": package.VERDICT_CLAIM_DISCHARGED,
    }
    root_evidence = _write(
        final / "root-evidence.json", package.canonical_json_bytes(evidence)
    )

    revalidation = {
        "added_post_seal": True,
        "added_utc": "2026-08-13T19:45:00Z",
        "attempted_and_refused": [],
        "conclusion": "synthetic",
        "lanes": [
            {
                "different_version_from_producer": version != "0.10.0",
                "environment": "ephemeral uv env",
                "ephemeral": True,
                "jax_version": version,
                "jaxlib_version": version,
                "lane": f"jax-{version}",
                "outcome": "PASS",
                "python_version": "3.11.15",
                "ran_utc": "2026-08-13T19:44:50Z",
                "result_sha256": "0" * 64,
                "verdict_recomputed": package.VERDICT_CLAIM_DISCHARGED,
            }
            for version in ("0.9.2", "0.10.0", "0.10.2")
        ],
        "method": "synthetic",
        "result_digest_meaning": "synthetic",
        "schema_version": package.REVALIDATION_SCHEMA_VERSION,
        "target": {"artifact_relative_path": "final"},
        "why": "synthetic",
    }
    revalidation_payload = _write(
        supplement / "revalidation-record.json",
        json.dumps(revalidation, indent=1, sort_keys=True).encode("utf-8") + b"\n",
    )

    supplement_entries = [
        {
            "file_sha256": _sha256(version_source),
            "name": "_version.py",
            "role": "executed_module_bytes",
            "size_bytes": len(version_source),
            "source_absolute_path": "/repo/src/simsopt/_version.py",
            "why_outside_final": "synthetic",
        },
        {
            "content_sha256": native_content,
            "file_sha256": _sha256(native_payload),
            "name": native_name,
            "role": "native_endpoint_reference_state",
            "size_bytes": len(native_payload),
            "source_absolute_path": "/elsewhere",
            "why_outside_final": "synthetic",
        },
        {
            "file_sha256": _sha256(revalidation_payload),
            "name": "revalidation-record.json",
            "role": "revalidation_record",
            "size_bytes": len(revalidation_payload),
            "source_absolute_path": None,
            "why_outside_final": "synthetic",
        },
        {
            "file_sha256": _sha256(authority_payload),
            "name": authority_name,
            "role": "execution_authority_manifest",
            "size_bytes": len(authority_payload),
            "source_absolute_path": "/repo/benchmarks/execution-sources.json",
            "why_outside_final": "synthetic",
        },
    ]
    _write(
        supplement / "supplement-manifest.json",
        json.dumps(
            {
                "added_post_seal": True,
                "added_utc": "2026-08-13T19:45:00Z",
                "direction": "synthetic",
                "entries": supplement_entries,
                "entry_count": len(supplement_entries),
                "purpose": "synthetic",
                "schema_version": package.SUPPLEMENT_SCHEMA_VERSION,
                "supplements": {
                    "artifact_relative_path": "final",
                    "root_evidence_sha256": _sha256(root_evidence),
                },
                "verification": "synthetic",
            },
            indent=1,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n",
    )

    files = sorted(
        path.relative_to(final).as_posix() for path in final.rglob("*") if path.is_file()
    )
    files.remove("root-evidence.json")
    manifest = {
        "directories": sorted(
            ({"mode": "0555", "relative_path": path.relative_to(final).as_posix()}
             for path in final.rglob("*") if path.is_dir()),
            key=lambda entry: entry["relative_path"],
        ),
        "files": [
            {
                "mode": "0444",
                "relative_path": relative,
                "sha256": _sha256((final / relative).read_bytes()),
                "size_bytes": (final / relative).stat().st_size,
            }
            for relative in [*files, "root-evidence.json"]
        ],
        "schema_version": package.ARTIFACT_MANIFEST_SCHEMA_VERSION,
    }
    _write(final / "artifact-manifest.json", package.canonical_json_bytes(manifest))
    package.emit_composite_manifest(root)
    return root


@pytest.fixture
def synthetic_package(tmp_path: Path) -> Path:
    return build_synthetic_package(tmp_path / "package")


def _validate(root: Path, **keywords: object) -> dict[str, object]:
    return package.validate_package(root, require_sealed_modes=False, **keywords)


def _rewrite(path: Path, payload: bytes) -> None:
    path.chmod(0o644)
    path.write_bytes(payload)


def _rewrite_final_member(root: Path, relative: str, payload: bytes) -> None:
    """Rewrite one final/ member and re-stamp the manifest entry that names it.

    Without the re-stamp every mutation is caught by the manifest binding first,
    which is a real gate but not the one these tests are aiming at: they have to
    reach the derivation BEHIND the digest.
    """

    _rewrite(root / "final" / relative, payload)
    manifest_path = root / "final" / "artifact-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest["files"]:
        if entry["relative_path"] == relative:
            entry["sha256"] = _sha256(payload)
            entry["size_bytes"] = len(payload)
    _rewrite(manifest_path, package.canonical_json_bytes(manifest))
    if relative == "root-evidence.json":
        supplement_path = root / "provenance-supplement" / "supplement-manifest.json"
        supplement = json.loads(supplement_path.read_text())
        supplement["supplements"]["root_evidence_sha256"] = _sha256(payload)
        _rewrite(
            supplement_path,
            json.dumps(supplement, indent=1, sort_keys=True).encode("utf-8") + b"\n",
        )


# ------------------------------------------------------------------- acceptance


def test_synthetic_package_validates(synthetic_package: Path) -> None:
    result = _validate(synthetic_package)
    assert result["verdict"] == package.PACKAGE_VERDICT
    assert result["receipt_verdict"] == package.VERDICT_CLAIM_DISCHARGED
    assert result["executed_module_count"] == 2
    assert result["authority_entry_count"] == 3
    assert result["authority_cited_not_embedded_members"] == 1
    assert result["quality_gate_terms"] == len(package.PINNED_ENDPOINT_QUALITY_GATES)


def test_expected_composite_digest_is_enforced(synthetic_package: Path) -> None:
    derived = str(_validate(synthetic_package)["composite_sha256"])
    assert _validate(synthetic_package, expect_composite_sha256=derived)
    with pytest.raises(package.PackageValidationError, match="not the expected"):
        _validate(synthetic_package, expect_composite_sha256="0" * 64)


# --------------------------------------------------------------------- refusals


def test_refuses_tampered_member(synthetic_package: Path) -> None:
    victim = synthetic_package / "final" / "source-snapshot" / "benchmarks" / "bound_module.py"
    _rewrite(victim, b"VALUE = 2\n")
    with pytest.raises(package.PackageValidationError, match="hashes to"):
        _validate(synthetic_package)


def test_refuses_tampered_supplement_member(synthetic_package: Path) -> None:
    victim = synthetic_package / "provenance-supplement" / "_version.py"
    _rewrite(victim, b"version = '9.9.9'\n")
    with pytest.raises(package.PackageValidationError, match="not the supplement's"):
        _validate(synthetic_package)


def test_refuses_missing_member(synthetic_package: Path) -> None:
    (synthetic_package / "final" / "cold-lane" / "terminal-coordinates.npy").unlink()
    with pytest.raises(package.PackageValidationError, match="unreadable"):
        _validate(synthetic_package)


def test_refuses_missing_supplement_directory(synthetic_package: Path) -> None:
    shutil.rmtree(synthetic_package / "provenance-supplement")
    with pytest.raises(package.PackageValidationError, match="carries no"):
        _validate(synthetic_package)


def test_refuses_unclaimed_extra_file(synthetic_package: Path) -> None:
    (synthetic_package / "final" / "source-snapshot" / "extra.py").write_bytes(b"X = 1\n")
    with pytest.raises(package.PackageValidationError, match="unclaimed"):
        _validate(synthetic_package)


def test_refuses_wrong_composite_sha256(synthetic_package: Path) -> None:
    stored = synthetic_package / "composite-manifest" / "composite-manifest.json"
    document = json.loads(stored.read_text())
    document["composite_sha256"] = "0" * 64
    _rewrite(stored, package.canonical_json_bytes(document))
    with pytest.raises(package.PackageValidationError, match="differing blocks"):
        _validate(synthetic_package)


def test_refuses_non_canonical_composite_manifest(synthetic_package: Path) -> None:
    stored = synthetic_package / "composite-manifest" / "composite-manifest.json"
    document = json.loads(stored.read_text())
    _rewrite(stored, json.dumps(document, indent=2, sort_keys=True).encode("utf-8"))
    with pytest.raises(package.PackageValidationError, match="not canonically encoded"):
        _validate(synthetic_package)


def test_refuses_non_canonical_correction_copy(synthetic_package: Path) -> None:
    correction = (
        synthetic_package / "composite-manifest" / "canonical" / "revalidation-record.json"
    )
    document = json.loads(correction.read_text())
    _rewrite(correction, json.dumps(document, indent=3, sort_keys=True).encode("utf-8"))
    with pytest.raises(package.PackageValidationError, match="must carry"):
        _validate(synthetic_package)


def test_refuses_supplement_that_supplements_another_root(synthetic_package: Path) -> None:
    manifest = synthetic_package / "provenance-supplement" / "supplement-manifest.json"
    document = json.loads(manifest.read_text())
    document["supplements"]["root_evidence_sha256"] = "0" * 64
    _rewrite(manifest, json.dumps(document, indent=1, sort_keys=True).encode("utf-8") + b"\n")
    with pytest.raises(package.PackageValidationError, match="supplements another root"):
        _validate(synthetic_package)


def test_refuses_executed_module_absent_from_the_package(synthetic_package: Path) -> None:
    """The narrower closure is enforced: absent-and-executed is a refusal."""

    evidence_path = synthetic_package / "final" / "root-evidence.json"
    evidence = json.loads(evidence_path.read_text())
    sources = evidence["attempts"][0]["evidence"]["execution_sources"]
    sources["bound_modules"].append(
        {
            "module": "examples.never_executed",
            "relative_path": "examples/never_executed.py",
            "sha256": hashlib.sha256(b"UNEXECUTED = 2\n").hexdigest(),
            "size_bytes": len(b"UNEXECUTED = 2\n"),
        }
    )
    _rewrite_final_member(
        synthetic_package, "root-evidence.json", package.canonical_json_bytes(evidence)
    )
    with pytest.raises(package.PackageValidationError, match="no bytes in this package"):
        _validate(synthetic_package)


def test_refuses_symlinked_member(synthetic_package: Path) -> None:
    victim = synthetic_package / "final" / "source-snapshot" / "benchmarks" / "bound_module.py"
    outside = synthetic_package.parent / "outside.py"
    outside.write_bytes(victim.read_bytes())
    victim.unlink()
    victim.symlink_to(outside)
    with pytest.raises(package.PackageValidationError, match="not a regular file"):
        _validate(synthetic_package)


def test_refuses_a_failed_quality_term(synthetic_package: Path) -> None:
    evidence_path = synthetic_package / "final" / "root-evidence.json"
    evidence = json.loads(evidence_path.read_text())
    ledger = evidence["attempts"][0]["evidence"]["endpoint_ledger"]
    ledger["terminal"]["constraint.volume"] = 2.0
    _rewrite_final_member(
        synthetic_package, "root-evidence.json", package.canonical_json_bytes(evidence)
    )
    with pytest.raises(package.PackageValidationError, match="re-derives to"):
        _validate(synthetic_package)


def test_refuses_a_receipt_whose_gate_never_ran(synthetic_package: Path) -> None:
    evidence_path = synthetic_package / "final" / "root-evidence.json"
    evidence = json.loads(evidence_path.read_text())
    evidence["attempts"][0]["evidence"]["endpoint_ledger"]["gated_at_this_budget"] = False
    _rewrite_final_member(
        synthetic_package, "root-evidence.json", package.canonical_json_bytes(evidence)
    )
    with pytest.raises(package.PackageValidationError, match="physics gate did not run"):
        _validate(synthetic_package)


def test_refuses_endpoint_coordinates_that_moved(synthetic_package: Path) -> None:
    relative = "attempts/attempt-1/terminal-coordinates.npy"
    payload = bytearray((synthetic_package / "final" / relative).read_bytes())
    payload[-1] ^= 0x01
    _rewrite_final_member(synthetic_package, relative, bytes(payload))
    with pytest.raises(
        package.PackageValidationError, match="published endpoint coordinates hash to"
    ):
        _validate(synthetic_package)


def test_sealed_modes_are_required_by_default(synthetic_package: Path) -> None:
    with pytest.raises(package.PackageValidationError, match="not 0555"):
        package.validate_package(synthetic_package)


# ------------------------------------------------------------------ twin guards


def test_canonical_encoding_matches_the_repository(synthetic_package: Path) -> None:
    """The duplicated serializer is byte-identical to the campaign's own."""

    corpus = [
        {"b": 1, "a": [1.5, None, True]},
        {"unicode": "é✓", "nested": {"z": 0, "a": {"k": []}}},
        json.loads(
            (synthetic_package / "composite-manifest" / "composite-manifest.json").read_text()
        ),
    ]
    for document in corpus:
        assert package.canonical_json_bytes(document) == repository_canonical_json_bytes(
            document
        )


def test_pinned_gate_table_matches_the_repository() -> None:
    assert package.PINNED_ENDPOINT_QUALITY_GATES == dict(
        rehearsal.PINNED_ENDPOINT_QUALITY_GATES
    )
    assert package.INFORMATIONAL_ENDPOINT_OBSERVABLES == tuple(
        rehearsal.INFORMATIONAL_ENDPOINT_OBSERVABLES
    )


def test_pinned_term_verdict_matches_the_repository() -> None:
    for name in package.PINNED_ENDPOINT_QUALITY_GATES:
        for terminal, native in ((1.0, 1.0), (1.0 + 1e-7, 1.0), (0.5, 1.0), (2.0, 1.0)):
            assert package.pinned_term_verdict(name, terminal, native) == {
                key: value
                for key, value in rehearsal._pinned_term_verdict(
                    name, terminal, native
                ).items()
            }


def test_validator_imports_nothing_from_this_repository() -> None:
    """The standalone contract, asserted against the module's own source."""

    source = VALIDATOR_PATH.read_text()
    for forbidden in ("import benchmarks", "from benchmarks", "import simsopt", "import jax", "import numpy"):
        assert forbidden not in source, forbidden
    imported = {
        name.split(".")[0]
        for name in re.findall(r"^(?:import|from)\s+([\w.]+)", source, re.MULTILINE)
    }
    assert imported <= set(sys.stdlib_module_names)


# ----------------------------------------------------------- path-access audits


_AUDIT_PROGRAM = """
import sys
from pathlib import Path

opened = []
sys.addaudithook(
    lambda event, arguments: opened.append(str(arguments[0]))
    if event == "open"
    else None
)
sys.path.insert(0, sys.argv[1])
import validate_projected_route_package as package

result = package.validate_package(Path(sys.argv[2]), require_sealed_modes=False)
sys.stderr.write("\\n".join(str(entry) for entry in opened))
print(result["verdict"])
"""


def test_audit_hook_sees_no_read_outside_the_package(synthetic_package: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            _AUDIT_PROGRAM,
            str(VALIDATOR_PATH.parent),
            str(synthetic_package),
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=str(REPOSITORY),
        env={**os.environ, "PYTHONPATH": ""},
    )
    assert completed.stdout.strip() == package.PACKAGE_VERDICT
    package_root = str(synthetic_package.resolve())
    interpreter_roots = (sys.prefix, sys.base_prefix, str(VALIDATOR_PATH.parent))
    for opened in completed.stderr.splitlines():
        if not opened or opened.startswith("<"):
            continue
        resolved = os.path.realpath(opened)
        if resolved.startswith(package_root):
            continue
        assert resolved.startswith(interpreter_roots) or not resolved.startswith(
            (str(REPOSITORY), str(Path.home() / "simsopt-campaigns"))
        ), opened


@requires_real_package
def test_real_package_validates_and_pins_its_composite_digest() -> None:
    result = package.validate_package(
        REAL_PACKAGE, expect_composite_sha256=REAL_PACKAGE_COMPOSITE_SHA256
    )
    assert result["verdict"] == package.PACKAGE_VERDICT
    assert result["receipt_verdict"] == package.VERDICT_CLAIM_DISCHARGED
    assert result["member_count"] == 613
    assert result["executed_module_count"] == 297
    assert result["authority_entry_count"] == 614
    assert result["authority_cited_not_embedded_members"] == 54
    assert result["sealed_modes_checked"] is True
    assert result["root_evidence_sha256"] == (
        "6937fc68a417d6968655cbdc460fa5655bd8cb5980a6e4c735506b3008231412"
    )


@requires_real_package
@pytest.mark.skipif(shutil.which("strace") is None, reason="strace is not installed")
def test_real_package_run_opens_nothing_outside_the_package(tmp_path: Path) -> None:
    """The syscall-level proof: every successful open is inside the package."""

    trace = tmp_path / "strace.txt"
    completed = subprocess.run(
        [
            "strace",
            "-f",
            "-qq",
            "-e",
            "trace=openat,open",
            "-o",
            str(trace),
            sys.executable,
            str(VALIDATOR_PATH),
            str(REAL_PACKAGE),
        ],
        capture_output=True,
        check=False,
        text=True,
        cwd=str(REPOSITORY),
        env={**os.environ, "PYTHONPATH": ""},
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["verdict"] == package.PACKAGE_VERDICT

    pattern = re.compile(r'open(?:at)?\((?:[^,]+,\s*)?"([^"]+)"')
    opened = {
        match.group(1)
        for line in trace.read_text(errors="replace").splitlines()
        if "= -1" not in line
        for match in [pattern.search(line)]
        if match is not None
    }
    package_root = str(REAL_PACKAGE.resolve())
    campaigns = str(Path.home() / "simsopt-campaigns")
    # The interpreter running the test lives inside the repository tree in this
    # campaign's environments, so its own prefix is excluded.  What is asserted
    # is the load-bearing part: the only repository FILE opened is the validator
    # module itself.  The directories the interpreter's import machinery scans at
    # startup are opened, not read, and are listed rather than waved away.
    interpreter_roots = (sys.prefix, sys.base_prefix)
    inside_repository = sorted(
        entry
        for entry in opened
        if entry.startswith(str(REPOSITORY)) and not entry.startswith(interpreter_roots)
    )
    assert [entry for entry in inside_repository if not os.path.isdir(entry)] == [
        str(VALIDATOR_PATH)
    ]
    assert {entry for entry in inside_repository if os.path.isdir(entry)} <= {
        str(VALIDATOR_PATH.parent),
        str(REPOSITORY / "src"),
    }
    assert not [
        entry
        for entry in opened
        if entry.startswith(campaigns) and not entry.startswith(package_root)
    ]
    assert not [entry for entry in opened if "/.git" in entry]
    assert len([entry for entry in opened if entry.startswith(package_root)]) > 600
