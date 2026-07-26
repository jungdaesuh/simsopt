"""Exact producer/consumer identities for dense linear-solve factor handoffs.

Terminology is identity / integrity receipt, never authorization or access-key.
Factors cross a consumer boundary only when sealed into an atomic handoff and
re-validated against the consumer's live state.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Callable, cast

import numpy as np
from simsopt_jax.backend import get_backend_policy, get_chunk_tuning
from simsopt_jax.runtime.exact_numeric_identity import exact_numeric_tree_sha256

# Private construction token: only :func:`build_exact_factor_handoff` may mint
# a valid :class:`ExactFactorHandoff`. Callers cannot affiliate raw factors with
# an integrity receipt after the fact by constructing the dataclass directly.
_EXACT_FACTOR_HANDOFF_TOKEN = object()


@dataclass(frozen=True)
class ExactHandoffProducerSeal:
    """Immutable canonical producer identity plus its integrity digest."""

    canonical_payload_json: str
    same_state_key_sha256: str

    def as_dict(self) -> dict[str, object]:
        """Return a detached JSON-compatible evidence mapping."""
        payload = json.loads(self.canonical_payload_json)
        payload["same_state_key_sha256"] = self.same_state_key_sha256
        return payload


@dataclass(frozen=True)
class ExactFactorHandoff:
    """Atomic factors plus producer integrity receipt.

    Construct only via :func:`build_exact_factor_handoff`. Consumers obtain
    factors exclusively through :func:`require_exact_factor_handoff`, which
    re-seals from live state and fails closed on substitution or mismatch.
    """

    _factors: object = field(repr=False)
    _integrity_receipt: ExactHandoffProducerSeal
    _construction_token: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._construction_token is not _EXACT_FACTOR_HANDOFF_TOKEN:
            raise RuntimeError(
                "ExactFactorHandoff must be built via build_exact_factor_handoff; "
                "raw factors cannot be affiliated with an integrity receipt "
                "after the fact."
            )
        if not isinstance(self._integrity_receipt, ExactHandoffProducerSeal):
            raise TypeError(
                "ExactFactorHandoff integrity receipt must be ExactHandoffProducerSeal."
            )

    @property
    def integrity_receipt(self) -> ExactHandoffProducerSeal:
        """Return the sealed producer identity bound to these factors."""
        return self._integrity_receipt


def canonical_json_sha256(value: Mapping[str, object]) -> str:
    """Hash one finite JSON mapping with a canonical byte representation."""
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def seal_exact_handoff_identity(
    identity_fields: Mapping[str, object],
) -> ExactHandoffProducerSeal:
    """Seal exact producer fields without permitting digest replacement."""
    if "same_state_key_sha256" in identity_fields:
        raise ValueError(
            "same_state_key_sha256 is protected and cannot be supplied or replaced."
        )
    canonical_payload_json = json.dumps(
        identity_fields,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return ExactHandoffProducerSeal(
        canonical_payload_json=canonical_payload_json,
        same_state_key_sha256=hashlib.sha256(
            canonical_payload_json.encode("utf-8")
        ).hexdigest(),
    )


def _validate_producer_seal(identity: ExactHandoffProducerSeal) -> None:
    payload = json.loads(identity.canonical_payload_json)
    canonical_payload_json = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    if (
        not isinstance(payload, dict)
        or canonical_payload_json != identity.canonical_payload_json
        or "same_state_key_sha256" in payload
    ):
        raise RuntimeError("Factor-handoff producer seal is not canonical.")
    expected = hashlib.sha256(
        identity.canonical_payload_json.encode("utf-8")
    ).hexdigest()
    if not hmac.compare_digest(expected, identity.same_state_key_sha256):
        raise RuntimeError("Factor-handoff producer seal failed its integrity check.")


def runtime_policy_identity() -> dict[str, object]:
    """Return the dtype, reduction, chunking, and relevant environment policy."""
    policy = get_backend_policy()
    chunk_tuning = get_chunk_tuning()
    return {
        "backend_policy": asdict(policy),
        "chunk_tuning": asdict(chunk_tuning),
        "reduction_environment": {
            name: os.environ.get(name)
            for name in (
                "XLA_FLAGS",
                "SIMSOPT_DENSE_OPERATOR_CHUNK_BATCH_SIZE",
                "SIMSOPT_HVP_OBJECTIVE_REMAT",
                "SIMSOPT_HVP_OBJECTIVE_REMAT_POLICY",
            )
        },
    }


def _certificate_coil_set_spec_from_dofs(
    state: Mapping[str, object],
) -> Callable[[object], object]:
    """Resolve the certificate coil-set mapper from production or test state.

    Production compiled-bundle state stores ``coil_set_spec_from_dofs`` and may
    omit a separate certificate mapper. Prefer an explicit certificate mapper
    when present; otherwise fall back to the production coil-set mapper.
    """
    certificate_spec_from_dofs = state.get("certificate_coil_set_spec_from_dofs")
    if certificate_spec_from_dofs is None:
        certificate_spec_from_dofs = state.get("coil_set_spec_from_dofs")
    if certificate_spec_from_dofs is None:
        raise KeyError(
            "Factor handoff identity requires certificate_coil_set_spec_from_dofs "
            "or coil_set_spec_from_dofs on state."
        )
    return cast(Callable[[object], object], certificate_spec_from_dofs)


def build_exact_handoff_identity(
    state: Mapping[str, object],
    *,
    coil_dofs: object,
    solved_x: object,
    factors: object,
    producer_graph_sha256: str,
) -> ExactHandoffProducerSeal:
    """Bind the live state and configuration consumed by one exact handoff."""
    objective_kwargs = cast(Mapping[str, object], state["objective_kwargs"])
    certificate_spec = _certificate_coil_set_spec_from_dofs(state)(coil_dofs)
    grid_modes = {
        name: objective_kwargs.get(name)
        for name in (
            "mpol",
            "ntor",
            "nfp",
            "stellsym",
            "label_mpol",
            "label_ntor",
            "label_nfp",
            "label_stellsym",
            "surface_kind",
            "label_surface_kind",
        )
    }
    grid_modes.update(
        {
            "quadpoints_phi_sha256": exact_numeric_tree_sha256(
                objective_kwargs["quadpoints_phi"]
            ),
            "quadpoints_theta_sha256": exact_numeric_tree_sha256(
                objective_kwargs["quadpoints_theta"]
            ),
        }
    )
    runtime_policy = runtime_policy_identity()
    identity = {
        "schema": "simsopt.exact_factor_handoff_identity.v1",
        "coil_state_sha256": exact_numeric_tree_sha256(coil_dofs),
        "solved_state_sha256": exact_numeric_tree_sha256(solved_x),
        "factor_tree_sha256": exact_numeric_tree_sha256(factors),
        "objective_and_weights_sha256": exact_numeric_tree_sha256(objective_kwargs),
        "stabilization": {
            "value": float(cast(float, state["linear_solve_stab"])),
            "sha256": exact_numeric_tree_sha256(
                np.asarray(cast(float, state["linear_solve_stab"]), dtype=np.float64)
            ),
        },
        "grid_modes": grid_modes,
        "coil_current_configuration_sha256": exact_numeric_tree_sha256(
            certificate_spec
        ),
        "dtype_reduction_policy": runtime_policy,
        "dtype_reduction_policy_sha256": canonical_json_sha256(runtime_policy),
        "producer_graph_sha256": producer_graph_sha256,
        "linearization_kind": cast(str, state["linearization_kind"]),
        "linear_solve_tolerance": float(cast(float, state["linear_solve_tol"])),
    }
    return seal_exact_handoff_identity(identity)


def build_exact_factor_handoff(
    state: Mapping[str, object],
    *,
    coil_dofs: object,
    solved_x: object,
    factors: object,
    producer_graph_sha256: str,
) -> ExactFactorHandoff:
    """Seal live factors with a producer integrity receipt in one atomic handoff.

    The integrity receipt is computed from the factors and live producer state at
    construction time; callers cannot attach a receipt to unaffiliated factors.
    """
    if factors is None:
        raise ValueError("Cannot build ExactFactorHandoff without factors.")
    integrity_receipt = build_exact_handoff_identity(
        state,
        coil_dofs=coil_dofs,
        solved_x=solved_x,
        factors=factors,
        producer_graph_sha256=producer_graph_sha256,
    )
    return ExactFactorHandoff(
        _factors=factors,
        _integrity_receipt=integrity_receipt,
        _construction_token=_EXACT_FACTOR_HANDOFF_TOKEN,
    )


def require_exact_factor_handoff(
    handoff: object,
    *,
    state: Mapping[str, object],
    coil_dofs: object,
    solved_x: object,
    producer_graph_sha256: str,
) -> object:
    """Release factors only after the consumer re-seals matching identity.

    Rebuilds consumer identity from the handoff's factors plus the consumer's
    live target state (coil dofs / solved decision vector / runtime policy /
    producer graph). Calls :func:`validate_same_state_identity` and returns the
    factors only on exact match. Fails closed on:

    * raw / unaffiliated factors (not an :class:`ExactFactorHandoff`)
    * forged or non-canonical integrity receipts
    * factor substitution under a stolen receipt
    * mismatched consumer state or producer graph
    """
    if not isinstance(handoff, ExactFactorHandoff):
        raise TypeError(
            "Factor consumer requires ExactFactorHandoff integrity receipt; "
            "unaffiliated raw factors are rejected."
        )
    if handoff._construction_token is not _EXACT_FACTOR_HANDOFF_TOKEN:
        raise RuntimeError(
            "ExactFactorHandoff construction token is invalid; factors are not released."
        )
    consumer_identity = build_exact_handoff_identity(
        state,
        coil_dofs=coil_dofs,
        solved_x=solved_x,
        factors=handoff._factors,
        producer_graph_sha256=producer_graph_sha256,
    )
    validate_same_state_identity(handoff.integrity_receipt, consumer_identity)
    return handoff._factors


def validate_same_state_identity(
    producer_identity: ExactHandoffProducerSeal,
    consumer_identity: ExactHandoffProducerSeal,
) -> None:
    """Fail unless a consumer independently rebuilds the producer identity."""
    _validate_producer_seal(producer_identity)
    _validate_producer_seal(consumer_identity)
    if producer_identity != consumer_identity:
        producer_fields = producer_identity.as_dict()
        consumer_fields = consumer_identity.as_dict()
        mismatches = sorted(
            key
            for key in set(producer_fields) | set(consumer_fields)
            if producer_fields.get(key) != consumer_fields.get(key)
        )
        raise RuntimeError(
            "Factor handoff does not bind the producer's exact state: "
            + ", ".join(mismatches)
        )
