"""Lane provenance records executed-source and native-extension identity."""

from __future__ import annotations

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
) -> provenance.LaneProvenance:
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
    return provenance.collect_lane_provenance(
        tmp_path,
        measurement_synchronization="native synchronous execution",
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
