"""Lane provenance records executed-source and native-extension identity."""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import hashlib
import sys
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
    replacement_bytes: bytes | None = None,
) -> provenance.LaneProvenance:
    """Load ``loaded_binary``, establish its identity, then (optionally) replace it
    with ``replacement_bytes`` as a lane's execution might, and collect the receipt."""
    monkeypatch.setattr(
        provenance,
        "collect_repository_state",
        lambda _root: provenance.RepositoryState(
            "b" * 40, False, hashlib.sha256(b"").hexdigest(), ()
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
    loaded_extension = provenance.loaded_extension_identity()
    if replacement_bytes is not None:
        loaded_binary.write_bytes(replacement_bytes)
    return provenance.collect_lane_provenance(
        tmp_path,
        measurement_synchronization="native synchronous execution",
        loaded_extension=loaded_extension,
    )


def _untracked_extension_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[provenance.LaneProvenance, Path, str]:
    relative = "build/cp311/simsoptpp.cpython-311-x86_64-linux-gnu.so"
    binary = tmp_path / relative
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"current compiled extension")
    receipt = _collect_with_sources(
        tmp_path,
        monkeypatch,
        executed_sources=(_source(relative, binary.read_bytes(), tracked=False),),
        loaded_binary=binary,
    )
    return receipt, binary, relative


def test_receipt_binds_the_loaded_extension_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt, binary, relative = _untracked_extension_receipt(tmp_path, monkeypatch)

    assert receipt.simsoptpp_path == str(binary.resolve())
    assert receipt.simsoptpp_sha256 == hashlib.sha256(binary.read_bytes()).hexdigest()
    assert any(
        source.path == relative and source.git_blob_id is None
        for source in receipt.executed_sources
    )
    assert (
        provenance.lane_provenance_from_payload(
            provenance.lane_provenance_payload(receipt)
        )
        == receipt
    )


@pytest.mark.parametrize(
    "retired_field", ("authoritative", "generated_source_bindings")
)
def test_payload_rejects_fields_outside_the_receipt_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    retired_field: str,
) -> None:
    receipt, _binary, _relative = _untracked_extension_receipt(tmp_path, monkeypatch)
    payload = provenance.lane_provenance_payload(receipt)
    payload[retired_field] = True

    with pytest.raises(ValueError, match="invalid fields"):
        provenance.lane_provenance_from_payload(payload)


@pytest.mark.parametrize("cleared_field", ("simsoptpp_path", "simsoptpp_sha256"))
def test_payload_rejects_an_extension_path_or_digest_recorded_alone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cleared_field: str,
) -> None:
    receipt, _binary, _relative = _untracked_extension_receipt(tmp_path, monkeypatch)
    payload = provenance.lane_provenance_payload(receipt)
    payload[cleared_field] = None

    with pytest.raises(ValueError, match="must be recorded together"):
        provenance.lane_provenance_from_payload(payload)


def test_receipt_refuses_an_extension_replaced_during_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lane that loaded binary A cannot record binary B written over A's path."""
    binary = tmp_path / "simsoptpp.so"
    binary.write_bytes(b"binary the lane loaded")

    with pytest.raises(ValueError, match="changed between lane start and receipt"):
        _collect_with_sources(
            tmp_path,
            monkeypatch,
            executed_sources=(),
            loaded_binary=binary,
            replacement_bytes=b"binary written during execution",
        )


def test_receipt_refuses_an_extension_loaded_after_identity_was_established(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An extension imported only after the lane started has no pre-execution identity."""
    binary = tmp_path / "simsoptpp.so"
    binary.write_bytes(b"late extension")
    monkeypatch.setattr(provenance, "collect_repository_state", lambda _root: None)
    monkeypatch.setitem(
        sys.modules,
        "simsoptpp",
        SimpleNamespace(__file__=str(binary), __version__="test"),
    )

    with pytest.raises(ValueError, match="changed between lane start and receipt"):
        provenance.collect_lane_provenance(
            tmp_path,
            measurement_synchronization="native synchronous execution",
            loaded_extension=None,
        )
