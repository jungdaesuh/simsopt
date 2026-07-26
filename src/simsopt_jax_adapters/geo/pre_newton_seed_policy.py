"""SSOT for mixed pre-Newton seed gate, routing, and canonical fallback.

Owns the decision tree used by
``BoozerSurfaceJAX._run_traceable_mixed_pipeline``:

1. Seed gate: Newton-candidate merit **and** proposal pre-Newton success.
2. Route: accepted proposal → bounded mixed Newton; else full canonical FP64.
3. After bounded: keep certificate only on success; otherwise one canonical fallback.

Finite/nonfinite merit is **not** re-thresholded here. ``_newton_candidate_status``
already requires a finite proposal; this module only combines that outcome with
pre-Newton success and resolves telemetry / selected seed source.

Unconditional finite-endpoint note
----------------------------------
Non-mixed / non-speculative paths still feed the pre-Newton endpoint into Newton
polish even when pre-Newton did not converge (success=False). This module does
not change that policy. Mixed seed selection remains gated (success + merit).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal, TypeVar

import numpy as np
import numpy.typing as npt

SeedSource = Literal["proposal", "original"]
AcceptedStage = Literal["proposal_bounded", "canonical_fallback", "pending_bounded"]

# Bool-like values that support ``&`` (host bool or JAX boolean array).
BoolLike = TypeVar("BoolLike")
# Result payload of a routed mixed-pipeline branch (traced or host).
BranchResult = TypeVar("BranchResult")
BranchFn = Callable[[None], BranchResult]


@dataclass(frozen=True, slots=True)
class MixedPipelinePublishFlags:
    """Telemetry flags published with a mixed-pipeline stage result."""

    canonical_fallback_used: bool
    mixed_seed_accepted: bool
    mixed_bounded_certificate_accepted: bool


@dataclass(frozen=True, slots=True)
class MixedPreNewtonSeedPolicyResult:
    """Resolved mixed pre-Newton seed selection and pipeline telemetry.

    ``selected_seed_x`` is a host ``numpy`` copy of the vector that should seed
    the *next* pipeline stage (bounded mixed Newton or canonical pre-Newton),
    not the final polished endpoint.
    """

    seed_gate_accepted: bool
    seed_source: SeedSource
    selected_seed_x: npt.NDArray[np.floating]
    attempt_bounded_mixed: bool
    canonical_fallback_used: bool
    mixed_seed_accepted: bool
    mixed_bounded_certificate_accepted: bool
    accepted_stage: AcceptedStage


def combine_mixed_seed_gate(
    *,
    proposal_success: BoolLike,
    seed_candidate_accepted: BoolLike,
) -> BoolLike:
    """Combine proposal pre-Newton success with Newton-candidate merit.

    Works for concrete ``bool`` and JAX boolean arrays (``&``).
    """
    return seed_candidate_accepted & proposal_success


def mixed_bounded_result_flags(
    *,
    bounded_certificate_success: BoolLike,
) -> Mapping[str, bool | BoolLike]:
    """Telemetry flags when publishing a bounded-mixed attempt result.

    Matches historical inline construction: while building the bounded branch
    result, ``canonical_fallback_used`` is False and ``mixed_seed_accepted`` is
    True; the certificate bit mirrors bounded success. The result is only
    returned when bounded success is True (via the routed ``lax.cond``).
    """
    return {
        "canonical_fallback_used": False,
        "mixed_seed_accepted": True,
        "mixed_bounded_certificate_accepted": bounded_certificate_success,
    }


def mixed_canonical_fallback_flags(
    *,
    seed_gate_accepted: BoolLike,
) -> Mapping[str, bool | BoolLike]:
    """Telemetry flags when publishing the full canonical FP64 fallback path."""
    return {
        "canonical_fallback_used": True,
        "mixed_seed_accepted": seed_gate_accepted,
        "mixed_bounded_certificate_accepted": False,
    }


def route_after_seed_gate(
    seed_gate_accepted: BoolLike,
    *,
    when_accepted: BranchFn[BranchResult],
    when_rejected: BranchFn[BranchResult],
) -> BranchResult:
    """Route the mixed pipeline after the seed gate (SSOT decision tree step 2).

    Production passes JAX branch callables; host tests may pass pure functions.
    """
    import jax

    return jax.lax.cond(
        seed_gate_accepted,
        when_accepted,
        when_rejected,
        operand=None,
    )


def route_after_bounded_attempt(
    bounded_certificate_success: BoolLike,
    *,
    keep_bounded: BranchFn[BranchResult],
    run_canonical: BranchFn[BranchResult],
) -> BranchResult:
    """Keep the bounded certificate result or fall back to full FP64 (step 3)."""
    import jax

    return jax.lax.cond(
        bounded_certificate_success,
        keep_bounded,
        run_canonical,
        operand=None,
    )


def resolve_mixed_pre_newton_seed_policy(
    *,
    proposal_success: bool,
    seed_candidate_accepted: bool,
    proposal_x: npt.ArrayLike,
    original_x: npt.ArrayLike,
    bounded_certificate_success: bool | None = None,
) -> MixedPreNewtonSeedPolicyResult:
    """Resolve mixed pre-Newton seed selection (host-side pure SSOT).

    This is the complete host encoding of the same tree that production routes
    via :func:`route_after_seed_gate` and :func:`route_after_bounded_attempt`.
    Flag bits are produced only through
    :func:`mixed_bounded_result_flags` / :func:`mixed_canonical_fallback_flags`.

    Parameters
    ----------
    proposal_success
        Pre-Newton optimizer success bit for the speculative (compute-dtype)
        proposal stage. Covers maxiter / line-search failure / non-convergence.
    seed_candidate_accepted
        Outcome of ``_newton_candidate_status`` on the proposal seed under the
        live certificate objective (finite + stationarity/Armijo merit).
    proposal_x, original_x
        Decision vectors for the proposal endpoint and the immutable original
        seed snapshot.
    bounded_certificate_success
        ``None`` before the bounded stage runs; otherwise the bounded mixed
        Newton success bit used to decide keep-versus-canonical-fallback.

    Policy (no silent behavior change relative to prior inline tree)
    ---------------------------------------------------------------
    - ``seed_gate_accepted = seed_candidate_accepted and proposal_success``.
    - Gate reject → original seed, canonical fallback immediately.
    - Gate accept + ``bounded_certificate_success is None`` → proposal seed,
      attempt bounded mixed Newton (pending final certificate).
    - Gate accept + bounded success → keep proposal path, no canonical fallback.
    - Gate accept + bounded failure → original seed, one canonical fallback;
      ``mixed_seed_accepted`` remains True (seed was accepted; certificate was not).
    """
    seed_gate_accepted = bool(
        combine_mixed_seed_gate(
            proposal_success=bool(proposal_success),
            seed_candidate_accepted=bool(seed_candidate_accepted),
        )
    )
    proposal = np.asarray(proposal_x, dtype=np.float64).copy()
    original = np.asarray(original_x, dtype=np.float64).copy()

    if not seed_gate_accepted:
        return MixedPreNewtonSeedPolicyResult(
            seed_gate_accepted=False,
            seed_source="original",
            selected_seed_x=original,
            attempt_bounded_mixed=False,
            **_host_flags(mixed_canonical_fallback_flags(seed_gate_accepted=False)),
            accepted_stage="canonical_fallback",
        )

    if bounded_certificate_success is None:
        return MixedPreNewtonSeedPolicyResult(
            seed_gate_accepted=True,
            seed_source="proposal",
            selected_seed_x=proposal,
            attempt_bounded_mixed=True,
            canonical_fallback_used=False,
            mixed_seed_accepted=True,
            mixed_bounded_certificate_accepted=False,
            accepted_stage="pending_bounded",
        )

    if bool(bounded_certificate_success):
        flags = _host_flags(
            mixed_bounded_result_flags(bounded_certificate_success=True)
        )
        return MixedPreNewtonSeedPolicyResult(
            seed_gate_accepted=True,
            seed_source="proposal",
            selected_seed_x=proposal,
            attempt_bounded_mixed=False,
            accepted_stage="proposal_bounded",
            **flags,
        )

    return MixedPreNewtonSeedPolicyResult(
        seed_gate_accepted=True,
        seed_source="original",
        selected_seed_x=original,
        attempt_bounded_mixed=False,
        **_host_flags(mixed_canonical_fallback_flags(seed_gate_accepted=True)),
        accepted_stage="canonical_fallback",
    )


def _host_flags(flags: Mapping[str, bool | BoolLike]) -> dict[str, bool]:
    """Materialize flag mappings to concrete host bools for the dataclass."""
    return {
        "canonical_fallback_used": bool(flags["canonical_fallback_used"]),
        "mixed_seed_accepted": bool(flags["mixed_seed_accepted"]),
        "mixed_bounded_certificate_accepted": bool(
            flags["mixed_bounded_certificate_accepted"]
        ),
    }
