"""Authority gate for an imported in-tree native extension."""

from __future__ import annotations

import dataclasses
import hashlib
import os
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import pytest
from examples.jax.parity import provenance


def _source(path: str, payload: bytes, *, tracked: bool) -> provenance.ExecutedSource:
    return provenance.ExecutedSource(
        path=path,
        sha256=hashlib.sha256(payload).hexdigest(),
        git_blob_id="a" * 40 if tracked else None,
    )


def _collect_with_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    executed_sources: tuple[provenance.ExecutedSource, ...],
    loaded_binary: Path,
    build_commit: str,
) -> provenance.LaneProvenance:
    commit = "b" * 40
    monkeypatch.setattr(
        provenance,
        "collect_repository_state",
        lambda _root: provenance.RepositoryState(
            commit, False, hashlib.sha256(b"").hexdigest(), ()
        ),
    )
    monkeypatch.setattr(
        provenance,
        "collect_executed_sources",
        lambda _root: executed_sources,
    )
    monkeypatch.setattr(
        provenance,
        "collect_explicit_sources",
        lambda _root, _paths: (
            _source("examples/jax/run_parity.py", b"runner", tracked=True),
        ),
    )
    monkeypatch.setattr(
        provenance,
        "_device_metadata",
        lambda: ((), None, "unavailable", None, {}),
    )
    monkeypatch.setitem(
        sys.modules,
        "simsoptpp",
        SimpleNamespace(__file__=str(loaded_binary), __version__="test"),
    )
    monkeypatch.setitem(
        sys.modules,
        "simsopt._version",
        SimpleNamespace(commit_id=f"g{commit[:9]}"),
    )
    monkeypatch.setenv("SIMSOPT_PARITY_SIMSOPTPP_BUILD_COMMIT", build_commit)
    return provenance.collect_lane_provenance(
        tmp_path,
        measurement_synchronization="native synchronous execution",
    )


def test_matching_operator_build_commit_does_not_establish_binary_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commit = "b" * 40
    relative = "build/cp311/simsoptpp.cpython-311-x86_64-linux-gnu.so"
    binary = tmp_path / relative
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"current compiled extension")
    receipt = _collect_with_sources(
        tmp_path,
        monkeypatch,
        executed_sources=(
            _source(relative, binary.read_bytes(), tracked=False),
            _source("src/simsopt/_version.py", b"generated version", tracked=False),
        ),
        loaded_binary=binary,
        build_commit=commit,
    )

    assert receipt.simsoptpp_checkout_compatible is False
    assert receipt.simsoptpp_sha256 == hashlib.sha256(binary.read_bytes()).hexdigest()
    assert any(
        source.path == relative and source.git_blob_id is None
        for source in receipt.executed_sources
    )
    assert receipt.authoritative is False


@pytest.mark.parametrize(
    "mutation", ("unrelated_py", "other_so", "wrong_hash", "old_build")
)
def test_unverified_untracked_source_or_old_build_remains_non_authoritative(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    relative = "build/cp311/simsoptpp.cpython-311-x86_64-linux-gnu.so"
    binary = tmp_path / relative
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"current compiled extension")
    extension_source = _source(
        relative,
        b"other extension bytes" if mutation == "wrong_hash" else binary.read_bytes(),
        tracked=False,
    )
    sources = (
        extension_source,
        _source("src/simsopt/_version.py", b"generated version", tracked=False),
    )
    if mutation == "unrelated_py":
        sources += (_source("scratch/untracked.py", b"foreign code", tracked=False),)
    elif mutation == "other_so":
        sources += (
            _source("build/cp311/other.so", b"foreign extension", tracked=False),
        )
    receipt = _collect_with_sources(
        tmp_path,
        monkeypatch,
        executed_sources=sources,
        loaded_binary=binary,
        build_commit="a" * 40 if mutation == "old_build" else "b" * 40,
    )

    assert receipt.authoritative is False


def _clean_build_fixture(tmp_path: Path) -> tuple[Path, Path]:
    def git(*arguments: str) -> None:
        subprocess.run(
            ("git", *arguments), cwd=tmp_path, check=True, capture_output=True
        )

    git("init")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "Test")
    (tmp_path / "src/simsoptpp").mkdir(parents=True)
    (tmp_path / "src/simsoptpp/kernel.cpp").write_text("native source")
    (tmp_path / ".gitignore").write_text("build/\n")
    git("add", ".")
    git("commit", "-m", "source")
    binary = tmp_path / "build/simsoptpp.so"
    binary.parent.mkdir()
    binary.write_bytes(b"compiled native source")
    receipt = provenance.write_native_build_receipt(
        tmp_path,
        binary,
        build_command=("cmake", "--build", "build"),
        toolchain={"compiler": "test compiler 1"},
    )
    return binary, receipt


def test_native_build_receipt_binds_clean_source_and_binary(tmp_path: Path) -> None:
    binary, receipt = _clean_build_fixture(tmp_path)
    binding = provenance.verify_native_build_receipt(tmp_path, binary)
    assert binding is not None
    assert binding[1] == hashlib.sha256(receipt.read_bytes()).hexdigest()


@pytest.mark.parametrize("outside_checkout", (False, True))
def test_snapshot_native_binding_uses_contained_source_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outside_checkout: bool,
) -> None:
    binary, _ = _clean_build_fixture(tmp_path)
    binding = provenance.verify_native_build_receipt(tmp_path, binary)
    assert binding is not None
    relative = binary.relative_to(tmp_path).as_posix()
    source = _source(relative, binary.read_bytes(), tracked=False)
    if outside_checkout:
        external = tmp_path.parent / f"{tmp_path.name}-external.so"
        external.write_bytes(binary.read_bytes())
        binary = external
    identity = provenance.SnapshotLaneIdentity(
        profile_id="native_cpu",
        lane="native-cpu",
        backend_mode="native_cpu",
        driver="test-native",
        execution_platform="cpu",
        runtime_identity_sha256="a" * 64,
        source_sha256="b" * 64,
        gpu_uuid="",
        snapshot_root=tmp_path,
        repository_commit=binding[0],
        repository_dirty=False,
        tracked_diff_sha256=hashlib.sha256(b"").hexdigest(),
        untracked_files=(),
        manifest_entries={relative: source.sha256},
        native_extension_path=binary,
        native_extension_sha256=source.sha256,
        interpreter_path=Path(sys.executable).resolve(),
        python_version=sys.version.split()[0],
        jax_version="test-jax",
        jaxlib_version=version("jaxlib"),
        bound_environment={},
        static_environment=provenance.normalize_snapshot_lane_environment(os.environ),
    )
    monkeypatch.setitem(
        sys.modules,
        "simsoptpp",
        SimpleNamespace(__file__=str(binary), __version__="test"),
    )
    monkeypatch.delitem(sys.modules, "simsopt._version", raising=False)
    monkeypatch.setattr(
        provenance,
        "_device_metadata",
        lambda: ((), None, "unavailable", "test-jax", {}),
    )
    monkeypatch.setattr(provenance, "_snapshot_executed_sources", lambda _id: (source,))
    if outside_checkout:
        with pytest.raises(
            ValueError, match="snapshot native extension is outside its snapshot root"
        ):
            provenance.collect_snapshot_lane_provenance(
                identity, measurement_synchronization="native synchronous execution"
            )
        return
    receipt = provenance.collect_snapshot_lane_provenance(
        identity, measurement_synchronization="native synchronous execution"
    )
    assert receipt.authoritative
    assert receipt.generated_source_bindings == {
        relative: f"local-build-receipt-sha256:{binding[1]}"
    }
    # This isolates the native binding; copied snapshot source manifests have
    # their own verification and do not thereby become Git-checkout authority.
    provenance.validate_authoritative_provenance(tmp_path, receipt)


@pytest.mark.parametrize("mutation", ("binary", "source", "omitted_input"))
def test_native_build_receipt_rejects_mismatched_binary_or_inputs(
    tmp_path: Path,
    mutation: str,
) -> None:
    binary, receipt = _clean_build_fixture(tmp_path)
    if mutation == "binary":
        binary.write_bytes(b"binary from a different build")
    elif mutation == "source":
        (tmp_path / "src/simsoptpp/kernel.cpp").write_text("different source")
    else:
        document = provenance._json_object(receipt, "test receipt")
        document["build_inputs"] = {}
        receipt.write_bytes(provenance._canonical_json_bytes(document))
    with pytest.raises(ValueError, match="(binary bytes|build inputs|input bindings)"):
        provenance.verify_native_build_receipt(tmp_path, binary)


def test_receipt_only_descendant_commit_keeps_identical_build_inputs(
    tmp_path: Path,
) -> None:
    binary, _ = _clean_build_fixture(tmp_path)
    before = provenance.verify_native_build_receipt(tmp_path, binary)
    (tmp_path / "README.md").write_text("Build report")
    subprocess.run(("git", "add", "README.md"), cwd=tmp_path, check=True)
    subprocess.run(
        ("git", "commit", "-m", "report"), cwd=tmp_path, check=True, capture_output=True
    )
    assert provenance.verify_native_build_receipt(tmp_path, binary) == before


def test_legacy_env_only_native_authority_receipt_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binary = tmp_path / "simsoptpp.so"
    binary.write_bytes(b"unverified binary")
    receipt = _collect_with_sources(
        tmp_path,
        monkeypatch,
        executed_sources=(_source("simsoptpp.so", binary.read_bytes(), tracked=False),),
        loaded_binary=binary,
        build_commit="b" * 40,
    )
    payload = provenance.lane_provenance_payload(receipt)
    payload["authoritative"] = True
    payload.pop("generated_source_bindings")
    with pytest.raises(ValueError, match="lacks a local build binding"):
        provenance.lane_provenance_from_payload(payload)


@pytest.mark.parametrize(
    "mutation",
    (
        "none",
        "binary_hash",
        "build_commit",
        "compatibility",
        "omitted_binary",
        "wrong_binding_key",
        "unknown_generated",
        "dirty",
        "untracked",
        "tracked_diff",
        "false_generated_version",
    ),
)
def test_authority_validates_exact_binary_and_generated_source_bindings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    binary, _receipt_path = _clean_build_fixture(tmp_path)
    binding = provenance.verify_native_build_receipt(tmp_path, binary)
    assert binding is not None
    relative = binary.relative_to(tmp_path).as_posix()
    receipt = _collect_with_sources(
        tmp_path,
        monkeypatch,
        executed_sources=(_source(relative, binary.read_bytes(), tracked=False),),
        loaded_binary=binary,
        build_commit="b" * 40,
    )
    receipt = dataclasses.replace(
        receipt,
        repository_commit=binding[0],
        tracked_diff_sha256=hashlib.sha256(b"").hexdigest(),
        authoritative=True,
        generated_source_bindings={
            relative: f"local-build-receipt-sha256:{binding[1]}"
        },
    )
    if mutation == "none":
        provenance.validate_authoritative_provenance(tmp_path, receipt)
        return
    if mutation == "binary_hash":
        receipt = dataclasses.replace(receipt, simsoptpp_sha256="0" * 64)
    elif mutation == "build_commit":
        receipt = dataclasses.replace(receipt, simsoptpp_build_commit="f" * 40)
    elif mutation == "compatibility":
        receipt = dataclasses.replace(receipt, simsoptpp_checkout_compatible=False)
    elif mutation == "omitted_binary":
        receipt = dataclasses.replace(receipt, executed_sources=())
    elif mutation == "wrong_binding_key":
        receipt = dataclasses.replace(
            receipt,
            generated_source_bindings={
                "scratch.py": f"local-build-receipt-sha256:{binding[1]}"
            },
        )
    elif mutation == "unknown_generated":
        receipt = dataclasses.replace(
            receipt,
            executed_sources=receipt.executed_sources
            + (_source("scratch.py", b"foreign code", tracked=False),),
        )
    elif mutation == "dirty":
        receipt = dataclasses.replace(receipt, repository_dirty=True)
    elif mutation == "untracked":
        receipt = dataclasses.replace(receipt, untracked_files=("scratch.py",))
    elif mutation == "tracked_diff":
        receipt = dataclasses.replace(receipt, tracked_diff_sha256="0" * 64)
    else:
        version = tmp_path / "src/simsopt/_version.py"
        version.parent.mkdir(parents=True)
        version.write_text("commit_id = 'gwrongcommit'\n")
        receipt = dataclasses.replace(
            receipt,
            simsopt_version_commit="g" + binding[0][:9],
            executed_sources=receipt.executed_sources
            + (
                _source("src/simsopt/_version.py", version.read_bytes(), tracked=False),
            ),
            generated_source_bindings={
                **receipt.generated_source_bindings,
                "src/simsopt/_version.py": "setuptools-scm checkout commit",
            },
        )
    with pytest.raises(ValueError, match="(authoritative|generated version)"):
        provenance.validate_authoritative_provenance(tmp_path, receipt)


@pytest.mark.parametrize(
    "addition",
    (
        "",
        "import sys\nsys.modules['numpy'].exp = lambda value: 0.0\n",
        "innocent = 42\n",
        "version = __import__('sys').version\n",
    ),
)
def test_generated_version_module_rejects_executable_and_unknown_additions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    addition: str,
) -> None:
    binary, _ = _clean_build_fixture(tmp_path)
    binding = provenance.verify_native_build_receipt(tmp_path, binary)
    assert binding is not None
    commit = binding[0]
    version = tmp_path / "src/simsopt/_version.py"
    version.parent.mkdir(parents=True)
    content = (
        "from __future__ import annotations\n"
        "__all__ = ['__version__', '__version_tuple__', 'version', 'version_tuple', '__commit_id__', 'commit_id']\n"
        "version: str\n__version__: str\n"
        "version_tuple: tuple[int | str, ...]\n__version_tuple__: tuple[int | str, ...]\n"
        "commit_id: str | None\n__commit_id__: str | None\n"
        "__version__ = version = '1.0.0'\n"
        "__version_tuple__ = version_tuple = (1, 0, 0)\n"
        f"__commit_id__ = commit_id = 'g{commit[:9]}'\n"
    )
    version.write_text(content + addition)
    relative = binary.relative_to(tmp_path).as_posix()
    monkeypatch.setattr(
        provenance,
        "collect_repository_state",
        lambda _root: provenance.RepositoryState(
            commit, False, hashlib.sha256(b"").hexdigest(), ()
        ),
    )
    monkeypatch.setattr(
        provenance,
        "collect_executed_sources",
        lambda _root: (
            _source(relative, binary.read_bytes(), tracked=False),
            _source("src/simsopt/_version.py", version.read_bytes(), tracked=False),
        ),
    )
    monkeypatch.setattr(
        provenance, "collect_explicit_sources", lambda _root, _paths: ()
    )
    monkeypatch.setattr(
        provenance, "_device_metadata", lambda: ((), None, "unavailable", None, {})
    )
    monkeypatch.setitem(sys.modules, "simsopt", SimpleNamespace(__version__="1.0.0"))
    monkeypatch.setitem(
        sys.modules, "simsopt._version", SimpleNamespace(commit_id=f"g{commit[:9]}")
    )
    monkeypatch.setitem(
        sys.modules,
        "simsoptpp",
        SimpleNamespace(__file__=str(binary), __version__="dev"),
    )
    if addition:
        with pytest.raises(ValueError, match="generated version module"):
            provenance.collect_lane_provenance(
                tmp_path, measurement_synchronization="native synchronous execution"
            )
    else:
        receipt = provenance.collect_lane_provenance(
            tmp_path, measurement_synchronization="native synchronous execution"
        )
        assert receipt.authoritative
        provenance.validate_authoritative_provenance(tmp_path, receipt)
