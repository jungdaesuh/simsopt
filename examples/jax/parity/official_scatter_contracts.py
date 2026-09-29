"""Contracts derived from upstream's own end-state scatter at the parity harness's scales.

The derivation reads one tracked upstream scatter record (``official_reference/9e027eac3/scatter/``) and nothing
else; no branch value, native or JAX, enters it. It uses only the pre-registered draws ``k = 0..8`` (the nine), fixed
before any sample was drawn: ``upstream_end_states`` -- upstream's end states of the nine, for a workflow whose end
STATE (which solution it lands on) is not determined by its input. Every draw whose upstream stage success flags are
all true enters the set (a failed solve never does), the arbiter groups the set into branches under the case's own
route comparators, each represented by its lowest-``k`` draw, and admits a lane whose end state matches one
representative -- or, beside a stage-wise contract, records the match informationally.

It is an engineering acceptance against upstream's own scatter, never an equivalence proof; the arbiter can therefore
only return ``quality-band`` under it. It may not be declared without a tracked same-state test proving the lanes
compute upstream's function (and gradient, where the workflow uses one) at upstream's recorded states; the
``same_state_proof`` argument names it and is carried into the derivation.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from examples.jax.parity.contracts import (
    UpstreamEndState,
    UpstreamEndStates,
)
from examples.jax.parity.official_reference import (
    OfficialUpstreamScatter,
    UpstreamScatterRun,
)
from simsopt_jax.examples import ExecutionScale

#: The pre-registered draws every contract is derived from (k = 0 unperturbed, k = 1..8 one ulp off the start).
PRE_REGISTERED_DRAWS: Final[tuple[int, ...]] = tuple(range(9))

#: A record's scale spelling -> the execution scale it names; a record at any other scale declares no contract.
_EXECUTION_SCALES: Final[Mapping[str, ExecutionScale]] = MappingProxyType(
    {"bounded": "bounded", "native_default": "native_default"}
)


def pre_registered_runs(
    scatter: OfficialUpstreamScatter,
) -> tuple[UpstreamScatterRun, ...]:
    """The record's nine pre-registered runs, in k order; any other run is never part of a contract."""
    runs = tuple(run for run in scatter.runs if run.k in PRE_REGISTERED_DRAWS)
    if tuple(run.k for run in runs) != PRE_REGISTERED_DRAWS:
        raise ValueError(
            f"{scatter.case_id} ({scatter.scale}): the record lacks pre-registered draws "
            f"{sorted(set(PRE_REGISTERED_DRAWS) - {run.k for run in runs})}"
        )
    return runs


def _provenance(scatter: OfficialUpstreamScatter) -> str:
    runner = (
        "the official script verbatim"
        if scatter.runner.kind == "verbatim"
        else "the official script with only its scale lines changed (record: runner.body_diff)"
    )
    return (
        f"upstream {scatter.upstream_commit[:9]} {scatter.official_script}, {runner}, one thread, "
        f"one-ulp start protocol, pre-registered draws k = 0..8"
    )


def upstream_end_states(
    scatter: OfficialUpstreamScatter,
    observables: tuple[str, ...],
    *,
    same_state_proof: str,
    disclosure: str = "",
) -> UpstreamEndStates:
    """The nine's successful end states of ``scatter``, keyed by lane observable, in ascending ``k``.

    A draw enters only if every one of upstream's own stage success flags the record carries is true; a record
    that carries none cannot declare an end-state set, because a failed solve could not then be told apart.
    """
    if scatter.scale not in _EXECUTION_SCALES:
        raise ValueError(
            f"{scatter.case_id}: record scale {scatter.scale!r} is not an execution scale"
        )
    if not scatter.success_keys:
        raise ValueError(
            f"{scatter.case_id} ({scatter.scale}): an end-state set needs upstream's own stage success "
            "flags, and the record carries none"
        )
    runs = pre_registered_runs(scatter)
    states = tuple(
        UpstreamEndState(
            k=run.k,
            values={observable: run.value(observable) for observable in observables},
        )
        for run in runs
        if run.workflow_success
    )
    failed = [run.k for run in runs if not run.workflow_success]
    return UpstreamEndStates(
        case_id=scatter.case_id,
        scale=_EXECUTION_SCALES[scatter.scale],
        observables=observables,
        states=states,
        derivation=(
            f"upstream end states at {scatter.scale}: {_provenance(scatter)}; the {len(states)} draws whose "
            f"upstream stage success flags ({', '.join(scatter.success_keys)}) are all true (failed: "
            f"{failed or 'none'}); the draws are grouped into branches under the case's own route comparator and "
            "tolerance for every judged key, each branch represented by its lowest-k draw, and a lane passes when "
            "its end state matches one representative on every judged key; "
            f"same-state proof: {same_state_proof}"
            + (f"; {disclosure}" if disclosure else "")
        ),
    )
