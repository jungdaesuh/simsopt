"""Tests for typed JAX runtime and host boundaries."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import jax
import numpy as np
import pytest
from simsopt_jax import numerical_policy
from simsopt_jax.numerical_policy import (
    MIXED_DENSE_IR_CERTIFICATE_KEY_WORD_MAX,
    MIXED_DENSE_IR_CERTIFICATE_PROBABILITY_MODEL,
    MIXED_DENSE_IR_CERTIFICATE_PROBE_SAMPLING_MODEL,
    MIXED_DENSE_IR_CERTIFICATE_PRNG_IMPL,
    CertificateProbeAuthority,
    CertificateProbeEvidence,
    CertificateProbeKeyData,
    fresh_certificate_probe_key_data,
    resolve_certificate_probe_authority,
)
from simsopt_jax.runtime.host_boundary import runtime_certificate_probe_key


@pytest.mark.parametrize(
    "key_data",
    (
        CertificateProbeKeyData(0, 0),
        CertificateProbeKeyData(17, 29),
        CertificateProbeKeyData(
            MIXED_DENSE_IR_CERTIFICATE_KEY_WORD_MAX,
            MIXED_DENSE_IR_CERTIFICATE_KEY_WORD_MAX,
        ),
    ),
)
def test_certificate_probe_key_survives_strict_transfer_guard(
    key_data: CertificateProbeKeyData,
):
    with jax.transfer_guard("disallow"):
        key = runtime_certificate_probe_key(key_data)

    assert str(jax.random.key_impl(key)) == MIXED_DENSE_IR_CERTIFICATE_PRNG_IMPL
    np.testing.assert_array_equal(
        np.asarray(jax.device_get(jax.random.key_data(key))),
        np.asarray(key_data.words, dtype=np.uint32),
    )


def test_fresh_certificate_probe_key_data_mints_both_words(monkeypatch):
    words = iter((11, 23))

    def next_word(bit_count: int) -> int:
        assert bit_count == 32
        return next(words)

    monkeypatch.setattr(numerical_policy.secrets, "randbits", next_word)

    assert fresh_certificate_probe_key_data() == CertificateProbeKeyData(11, 23)


def test_certificate_probe_authority_round_trips_both_uint32_words():
    authority = CertificateProbeAuthority(
        source="supplied_replay",
        key_data=CertificateProbeKeyData(
            MIXED_DENSE_IR_CERTIFICATE_KEY_WORD_MAX,
            MIXED_DENSE_IR_CERTIFICATE_KEY_WORD_MAX - 1,
        ),
    )

    payload = authority.as_json()

    assert payload["prng_impl"] == MIXED_DENSE_IR_CERTIFICATE_PRNG_IMPL
    assert payload["sampling_model"] == MIXED_DENSE_IR_CERTIFICATE_PROBE_SAMPLING_MODEL
    assert payload["probability_model"] == MIXED_DENSE_IR_CERTIFICATE_PROBABILITY_MODEL
    assert CertificateProbeAuthority.from_json(payload) == authority


def test_certificate_probe_authority_distinguishes_fresh_from_replay(monkeypatch):
    monkeypatch.setattr(
        numerical_policy,
        "fresh_certificate_probe_key_data",
        lambda: CertificateProbeKeyData(11, 23),
    )
    replay_key_data = CertificateProbeKeyData(29, 31)

    fresh = resolve_certificate_probe_authority(None)
    replay = resolve_certificate_probe_authority(replay_key_data)

    assert fresh == CertificateProbeAuthority(
        source="fresh_runtime",
        key_data=CertificateProbeKeyData(11, 23),
    )
    assert replay == CertificateProbeAuthority(
        source="supplied_replay",
        key_data=replay_key_data,
    )


def test_certificate_probe_evidence_round_trip_binds_trust_to_fresh_challenge():
    key_data = CertificateProbeKeyData(11, 23)
    evidence = CertificateProbeEvidence(
        authority=CertificateProbeAuthority(
            source="fresh_runtime",
            key_data=key_data,
        ),
        observed_key_data=key_data,
        active=True,
        proposal_trusted=True,
        fp64_rebuild_count=0,
        fallback_attempted=False,
        fallback_success=False,
    )

    restored = CertificateProbeEvidence.from_json(
        evidence.as_json(),
        require_claim_eligible=True,
    )

    assert restored == evidence


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"observed_key_data": CertificateProbeKeyData(29, 31)}, "observed key"),
        ({"proposal_trusted": False}, "rebuild decisions"),
        (
            {
                "proposal_trusted": False,
                "fp64_rebuild_count": 1,
                "fallback_attempted": False,
            },
            "fallback evidence",
        ),
        (
            {
                "proposal_trusted": False,
                "fp64_rebuild_count": 1,
                "fallback_attempted": True,
                "fallback_success": False,
            },
            "did not certify",
        ),
        ({"fallback_success": True}, "success without an attempt"),
    ),
)
def test_certificate_probe_evidence_rejects_inconsistent_decisions(overrides, message):
    key_data = CertificateProbeKeyData(11, 23)
    fields = {
        "authority": CertificateProbeAuthority(
            source="fresh_runtime",
            key_data=key_data,
        ),
        "observed_key_data": key_data,
        "active": True,
        "proposal_trusted": True,
        "fp64_rebuild_count": 0,
        "fallback_attempted": False,
        "fallback_success": False,
        **overrides,
    }
    evidence = CertificateProbeEvidence(**fields)

    with pytest.raises(ValueError, match=message):
        evidence.require_valid_for_mixed()


def test_replay_certificate_probe_evidence_cannot_authorize_fresh_claim():
    key_data = CertificateProbeKeyData(11, 23)
    evidence = CertificateProbeEvidence(
        authority=CertificateProbeAuthority(
            source="supplied_replay",
            key_data=key_data,
        ),
        observed_key_data=key_data,
        active=True,
        proposal_trusted=True,
        fp64_rebuild_count=0,
        fallback_attempted=False,
        fallback_success=False,
    )

    with pytest.raises(ValueError, match="cannot authorize"):
        CertificateProbeEvidence.from_json(
            evidence.as_json(),
            require_claim_eligible=True,
        )


def test_certificate_probe_authority_preserves_both_words_with_x64_disabled():
    repo_root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repo_root / "src")
    environment["JAX_ENABLE_X64"] = "False"
    environment["JAX_PLATFORMS"] = "cpu"
    completed = subprocess.run(
        (
            sys.executable,
            str(repo_root / "tests" / "subprocess" / "jax_runtime_cases.py"),
            "certificate-probe-key-x64-disabled",
        ),
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    "serialized_key_data",
    (
        17,
        [17],
        [17, 23, 29],
        [True, 23],
        [17, -1],
        [17, MIXED_DENSE_IR_CERTIFICATE_KEY_WORD_MAX + 1],
    ),
)
def test_certificate_probe_key_data_rejects_noncanonical_json(
    serialized_key_data: object,
):
    with pytest.raises(ValueError, match="Certificate probe key"):
        CertificateProbeKeyData.from_json(serialized_key_data)
