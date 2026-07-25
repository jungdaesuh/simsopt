"""Tests for typed JAX runtime and host boundaries."""

from __future__ import annotations

import jax
import numpy as np
import pytest
from simsopt_jax import numerical_policy
from simsopt_jax.numerical_policy import (
    MIXED_DENSE_IR_CERTIFICATE_KEY_WORD_MAX,
    MIXED_DENSE_IR_CERTIFICATE_PRNG_IMPL,
    CertificateProbeKeyData,
    fresh_certificate_probe_key_data,
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
